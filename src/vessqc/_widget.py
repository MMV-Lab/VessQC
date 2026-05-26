"""
Module for the definition of the class VessQcWidget

Imports
-------
napari, napari.utils.colormaps, numpy, pathlib.Path, qtpy.QtCore.QSize, 
qtpy.QtCore.QT, qtpy.QtWidgets, scipy.ndimage, SimpleITK, tifffile.imread, 
tifffile.imwrite

Exports
-------
VessQcWidget
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

import copy
from joblib import Parallel, delayed
import json
import numpy as np
import napari
from napari.utils.colormaps import CyclicLabelColormap
from pathlib import Path
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QWidget,
    QSizePolicy,
)
from scipy import ndimage
import SimpleITK as sitk
import tempfile
from tifffile import imread, imwrite
import time
from typing import TYPE_CHECKING

from ._data_manager import DataManager, DatasetTriplet
from ._loading_dialog import LoadingDialog
from ._segmentation_worker import SegmentationWorker

if TYPE_CHECKING:
    import napari


def _label_value_sparse(uncertainty, uncert, tolerance, structure, value_idx,
    num_unique_uncert, cancel_event=None):
    # Worker side
    # (03.07.2025)
    
    # Check for cancellation before expensive operations
    if cancel_event and cancel_event.is_set():
        return None

    mask = np.abs(uncertainty - uncert) < tolerance
    if not np.any(mask):
        return None
    
    # Check again before labeling (expensive operation)
    if cancel_event and cancel_event.is_set():
        return None

    labeled, num = ndimage.label(mask, structure)   # Segmentation
    if num == 0:
        return None

    # Calculate global unique labels directly
    # local labels: 1, 2, 3, ...
    # global labels: (local - 1) * num_unique_uncert + (value_idx + 1)
    labeled_global = (labeled - 1) * num_unique_uncert + (value_idx + 1)
    labeled_global[labeled == 0] = 0

    indices = np.where(mask)
    result = dict(
        indices       = indices,
        global_labels = labeled_global[indices],
        uncert        = uncert,
        num           = num
    )
    return result

def jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, tuple):
        return [jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonify(x) for x in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class VessQcWidget(QWidget):
    """
    Main widget of a Napari plugin for checking the calculation of blood vessels

    Attributes
    ----------
    viewer : class napari.viewer
        Napari viewer
    start_multiple_viewer : bool
        Call the multiple viewer and the cross widget?
    save_uncertainty : bool
        Save the file 'Uncertainty.tif'?
    areas : dict
        Contains information about the various areas
    parent : str
        Directory of data files
    suffix : str
        Extension of the data file (e.g '.tif')
    is_tifffile : bool
        Is the file extension '.tif' or '.tiff'?
    image : numpy.ndarray
        3D array with image data
    segPred : numpy.ndarray
        3D array with the vessel data
    uncertainty : numpy.ndarray
        3D array with uncertainties
    popup_window : QWidget
        Pop up window with uncertainty values

    Methods
    -------
    __init__(viewer: "napari.viewer.Viewer")
        Class constructor
    load_image()
        Read the image file and save it in an image layer
    read_segPred()
        Read the segPred and uncertanty data and save it in a label and an
        image layer
    find_segments(uncertainty: np.ndarray)
        Define areas that correspond to values of equal uncertainty
    show_popup_window()
        Define a pop-up window for the uncertainty list
    new_entry(segment: dict, grid_layout: QGridLayout, i: int):
        New entry for 'Area n' in the grid layout
    show_area()
        Show the data for a specific uncertanty in a new label layer
    done()
        Transfer data from the area to the segPred and uncertainty layer
        and close the layer for the area
    re_enable()
        Re-enable the data of a specific segment in the pop-up window
    compare_and_transfer(name: str)
        Compare old and new data of an area and transfer the changes to the
        segPred and uncertainty data
    save_intermediate_data()
        Save the segPred and uncertainty data to files on hard drive
    load_intermediate_data()
        Read the segPred and uncertainty data from files on hard drive
    save_final_result()
        Close all open area layers, close the pop-up window, save the
        segPred and if applicable also the uncertainty data to files on
        hard drive
    cbx_save_uncertainty(state: Qt.Checked)
        Toggle the bool variable save_uncertainty
    show_info()
        Show information about the current layer
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Class constructor

        Parameter
        ---------
        viewer : widget
            napari.viewer
        """

        # (03.05.2024)
        super().__init__()
        self.viewer = viewer
        self.segments = []
        self.save_uncertainty = False
        
        # Initialize data manager
        self.data_manager = DataManager()
        self.current_triplet = None
        self.loading_dialog = None  # Track loading dialog
        self.popup_window = None  # Track popup window
        self.current_zoomed_segment = None  # Track currently zoomed segment
        self._small_segments_label = None  # Track the label for small segments collection
        
        # Initialize segmentation worker
        self.segmentation_worker = SegmentationWorker(callback=self._on_segmentation_complete)
        self.segmentation_worker.start()
        
        # Cache for original labels (before threshold filtering)
        self.original_labels = None
        self.original_segments = None
        
        # Persistent image layer settings
        self.image_contrast_limits_relative = None  # Store as (min%, max%) relative to data range
        self.image_gamma = 1.0
        
        # Timer for debounced threshold updates
        self.threshold_timer = QTimer()
        self.threshold_timer.setSingleShot(True)
        self.threshold_timer.timeout.connect(self._apply_threshold_debounced)
        
        # Connect to viewer window close event
        if hasattr(viewer.window, '_qt_window'):
            viewer.window._qt_window.destroyed.connect(self._on_viewer_destroyed)

        # Define the layout of the main widget
        self.setLayout(QVBoxLayout())

        # Define some labels and buttons
        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)
        self.layout().addWidget(label1)

        # Loading button
        btnLoadDataset = QPushButton('Load Dataset')
        btnLoadDataset.clicked.connect(self.load_dataset_dialog)
        self.layout().addWidget(btnLoadDataset)

        btnShowUncert = QPushButton('Show uncertainty data')
        btnShowUncert.clicked.connect(self.show_uncertainty)
        self.layout().addWidget(btnShowUncert)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)
        self.layout().addWidget(label2)

        label3 = QLabel('Curation')
        label3.setFont(font)
        self.layout().addWidget(label3)
        
        # Threshold adjustment
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel('Min segment size:'))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(200)  # Minimum value is 200
        self.threshold_spinbox.setMaximum(1000)
        self.threshold_spinbox.setValue(200)
        self.threshold_spinbox.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spinbox)
        self.layout().addLayout(threshold_layout)
        
        # Top 5 segments quick access panel
        self.top5_groupbox = QGroupBox('Top 5 Segments (by uncertainty)')
        top5_layout = QVBoxLayout()
        self.top5_groupbox.setLayout(top5_layout)
        self.top5_groupbox.setVisible(False)  # Hidden until dataset loaded
        self.layout().addWidget(self.top5_groupbox)

        btnPopupWindow = QPushButton('Show list of segments')
        btnPopupWindow.clicked.connect(self.show_popup_window)
        self.layout().addWidget(btnPopupWindow)
        
        # Neighboring segments controls
        self.cbx_show_neighbors = QCheckBox('Show neighboring segments')
        self.cbx_show_neighbors.stateChanged.connect(self._toggle_neighbors_visibility)
        self.layout().addWidget(self.cbx_show_neighbors)
        
        btnIdentifySegment = QPushButton('Identify segment (click on voxel)')
        btnIdentifySegment.setCheckable(True)  # Make it a toggle button
        btnIdentifySegment.toggled.connect(self._toggle_identify_mode)
        self.layout().addWidget(btnIdentifySegment)
        self._identify_mode = False  # Track identify mode state

        btnSaveIntermediate = QPushButton('Save intermediate data')
        btnSaveIntermediate.clicked.connect(self.save_intermediate_data)
        self.layout().addWidget(btnSaveIntermediate)

        label4 = QLabel('_______________')
        label4.setAlignment(Qt.AlignHCenter)
        self.layout().addWidget(label4)

        btnSaveResult = QPushButton('Save final result')
        btnSaveResult.clicked.connect(self.save_final_result)
        self.layout().addWidget(btnSaveResult)

        cbxSaveUncertainty = QCheckBox('Save uncertainty')
        cbxSaveUncertainty.stateChanged.connect(self.checkbox_save_uncertainty)
        self.layout().addWidget(cbxSaveUncertainty)
    
    def closeEvent(self, event):
        """Handle widget close event"""
        print("DEBUG: Widget closeEvent triggered")
        
        # Stop segmentation worker (daemon thread will terminate with process)
        if hasattr(self, 'segmentation_worker'):
            print("DEBUG: Stopping segmentation worker...")
            self.segmentation_worker.stop()
            print("DEBUG: Segmentation worker stop signal sent (daemon thread will terminate with process)")
        
        # Close loading dialog if open
        if self.loading_dialog:
            print("DEBUG: Closing loading dialog")
            self.loading_dialog.close()
            self.loading_dialog = None
        
        # Close popup window if open
        if self.popup_window:
            print("DEBUG: Closing popup window")
            self.popup_window.close()
            self.popup_window = None
        
        event.accept()
    
    def _on_viewer_destroyed(self):
        """Handle viewer window destruction"""
        print("DEBUG: Viewer window destroyed, cleaning up...")
        
        # Stop segmentation worker (daemon thread will terminate with process)
        if hasattr(self, 'segmentation_worker'):
            self.segmentation_worker.stop()
            print("DEBUG: Segmentation worker stop signal sent")
        
        # Close all child windows
        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
        
        if self.popup_window:
            self.popup_window.close()
            self.popup_window = None

    def load_image(self):
        """
        DEPRECATED: Use load_dataset_dialog() instead
        Read the image file and save it in an image layer
        """
        # (23.05.2024)
        QMessageBox.information(self, 'Deprecated', 
                              'This method is deprecated.\nPlease use "Load Dataset" button instead.')
        return

        # Legacy code below (kept for tests)
        #

        # Find and load the image file
        filter1 = "TIFF files (*.tif *.tiff);;NIfTI files (*.nii *.nii.gz);;\
            All files (*.*)"
        filename, _ = QFileDialog.getOpenFileName(self, 'Load image file', '',
            filter1)

        if filename == '':                      # Cancel has been pressed
            QMessageBox.information(self, 'Cancel button',
                'The cancel button has been pressed.')
            return

        filename = Path(filename)
        self.parent = filename.parent           # The data directory
        self.stem1  = filename.stem             # Name of the input file
        suffix      = filename.suffix.lower()   # File extension
        # Truncate the extension .nii
        if suffix == '.gz' and self.stem1[-4:] == '.nii':
            self.stem1 = self.stem1[:-4]

        # Load the image file
        print('Load', filename)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.image = imread(filename)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(filename)
                self.image = sitk.GetArrayFromImage(sitk_image)
            else:
                QMessageBox.information(self, 'Unknown file type',
                    'Unknown file type: %s%s!' % (self.stem1, suffix))
                return
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        self.viewer.add_image(self.image, name=self.stem1)   # Show the image
        self.segments.clear()

    def read_segPred(self):
        """
        DEPRECATED: Use load_dataset_dialog() instead
        Read the segPred and uncertanty data and save it in a label and an
        image layer
        """
        # (23.05.2024, revised on 05.02.2025)
        QMessageBox.information(self, 'Deprecated', 
                              'This method is deprecated.\nPlease use "Load Dataset" button instead.')
        return
        
        # Legacy code below (kept for tests)
        #
        # Search for the segPred file
        self.stem2 = self.stem1[:-3] + '_segPred'   # Replace _IM by _segPred
        filename = self.parent.joinpath(self.stem2)

        if filename.with_suffix('.tif').is_file():
            filename = filename.with_suffix('.tif')
            suffix = '.tif'
        elif filename.with_suffix('.tiff').is_file():
            filename = filename.with_suffix('.tiff')
            suffix = '.tiff'
        elif filename.with_suffix('.nii').is_file():
            filename = filename.with_suffix('.nii')
            suffix = '.nii'
        elif filename.with_suffix('.nii.gz').is_file():
            filename = filename.with_suffix('.nii.gz')
            suffix = '.gz'
        else:
            QMessageBox.information(self, 'File not found',
                'No segPred file %s found!' % (filename))
            return

        # Read the segPred file
        print('Load', filename)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.segPred = imread(filename)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(filename)
                self.segPred = sitk.GetArrayFromImage(sitk_image)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Save the segPred data in a label layer
        self.viewer.add_labels(self.segPred, name=self.stem2)

        # Search for the uncertainty file
        self.stem3 = self.stem1[:-3] + '_uncertainty'
        filename = self.parent.joinpath(self.stem3)

        if filename.with_suffix('.tif').is_file():
            filename = filename.with_suffix('.tif')
            suffix = '.tif'
        elif filename.with_suffix('.tiff').is_file():
            filename = filename.with_suffix('.tiff')
            suffix = '.tiff'
        elif filename.with_suffix('.nii').is_file():
            filename = filename.with_suffix('.nii')
            suffix = '.nii'
        elif filename.with_suffix('.nii.gz').is_file():
            filename = filename.with_suffix('.nii.gz')
            suffix = '.gz'
        else:
            QMessageBox.information(self, 'File not found',
                'No uncertainty file %s found!' % (filename))
            return

        # Read the uncertainty file
        print('Load', filename)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.uncertainty = imread(filename)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(filename)
                self.uncertainty = sitk.GetArrayFromImage(sitk_image)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Show the last created label layer
        QApplication.processEvents()

        if self.segments == []:
            self.find_segments(self.uncertainty)

    def show_uncertainty(self, uncertainty: np.ndarray):
        """ Show an image layer with the uncertainty data """

        # (12.08.2025)
        if hasattr(self, 'uncertainty'):
            self.viewer.add_image(self.uncertainty, name='uncertainty',
                colormap='inferno')
        else:
            QMessageBox.information(self, 'Note', 'Uncertainty is not defined')

    def find_segments(self, uncertainty: np.ndarray):
        """ Define segments that correspond to values of equal uncertainty """

        # (09.08.2024, revised on 03.07.2025, updated 07.10.2025)
        t0 = time.time()                # UNIX timestamp
        print('The segmentation will take some time.')
        print('Processing uncertainty values...')
        
        # Process events to keep UI responsive
        QApplication.processEvents()

        # Find unique uncertainty values only where segPred > 0 (vessel regions)
        # This prevents segmenting the entire volume including background
        vessel_mask = self.segPred > 0
        uncertainty_vessels = uncertainty[vessel_mask]
        
        # Round to 3 decimals to handle continuous data (matches tolerance 1e-3)
        unique_uncertainties = np.unique(np.round(uncertainty_vessels, decimals=3))
        # Filter zeros (in case tiny values rounded to 0.000)
        unique_uncertainties = unique_uncertainties[unique_uncertainties > 0]
        num_unique_uncert = len(unique_uncertainties)
        print(f'Found {num_unique_uncert} unique uncertainty values in vessel regions (after rounding to 3 decimals)')
        tolerance = 1e-3
        structure = np.ones((3, 3, 3), dtype=int)   # Connectivity

        # Process events before heavy computation
        QApplication.processEvents()
        
        print('Running parallel segmentation (this may take a while)...')
        # Note: Parallel processing blocks, but is much faster than sequential
        # Using threading backend for faster cleanup when switching datasets
        results = Parallel(n_jobs=-1, backend='threading', verbose=5)(
            delayed(_label_value_sparse)(
                uncertainty, uncert, tolerance, structure, idx,
                num_unique_uncert
            )
            for idx, uncert in enumerate(unique_uncertainties)
        )
        
        print('Parallel processing complete, assembling results...')
        QApplication.processEvents()

        self.labels = np.zeros_like(uncertainty, dtype=int)
        uncert_values = {0: 0.0}    # Dictionary of all uncertanty values

        for i, result in enumerate(results):
            if result is None:
                continue
            indices = result['indices']
            labels  = result['global_labels']
            uncert  = result['uncert']
            num     = result['num']

            self.labels[indices] = labels
     
            # Form a dictionary with the uncertainty values that correspond to
            # the respective labels
            keys     = list(np.unique(labels))
            values   = [uncert] * num
            u_values = dict(zip(keys, values))
            uncert_values = {**uncert_values, **u_values}

            # Process events periodically during result assembly
            if i % 10 == 0:
                QApplication.processEvents()

        print(f'Segmentation done in {time.time() - t0:.1f}s')
        QApplication.processEvents()

        print('Filtering small segments...')
        QApplication.processEvents()
        
        # Cache labels BEFORE any Noise grouping for threshold filtering
        self.original_labels = self.labels.copy()
        
        # Determine all labels that appear less than 200 times
        min_size = 200
        counts = np.bincount(self.labels.ravel())
        small_labels = np.where(counts < min_size)[0]
        small_labels = small_labels[small_labels != 0]
        print(f'Found {len(small_labels)} small segments (< {min_size} pixels)')

        # Replaces all labels that occur less than 200 times with the value
        # max(labels) + 1
        max_label = np.max(self.labels) + 1
        self._small_segments_label = max_label  # Store for later reference
        mask = np.isin(self.labels, small_labels)
        self.labels[mask] = max_label
        
        QApplication.processEvents()

        # Create a structure for storing the data
        print('Creating segment metadata...')
        unique_labels = np.unique(self.labels)
        unique_labels = unique_labels[unique_labels != 0]
        counts = np.bincount(self.labels.ravel())
        uncert_values[max_label] = 0.9999

        # Calculate actual max uncertainty for each label from original data
        print('Calculating actual max uncertainties for segments...')
        for label_val in unique_labels:
            if label_val != max_label:  # Skip the Noise segment
                mask = self.labels == label_val
                segment_uncert = uncertainty[mask]
                segment_uncert = segment_uncert[segment_uncert > 0]
                if len(segment_uncert) > 0:
                    # Use actual max from original data, not rounded value
                    uncert_values[label_val] = float(np.max(segment_uncert))

        self.segments = list()
        for label in unique_labels:
            segment = dict(
                name        = '',
                label       = label,
                uncertainty = uncert_values[label],
                counts      = counts[label],
                coords      = None,     # coordinates of cropped image
                done        = False,
            )
            self.segments.append(segment)

        # Sort by 'uncertainty' ascending
        self.segments.sort(key=lambda x: x['uncertainty'])
        print(f'Created {len(self.segments)} segments')
        
        QApplication.processEvents()

        # Determine the names of the segments
        for i, segment in enumerate(self.segments, start=1):
            # Check if this is the "Noise" collection (highest label)
            if segment['label'] == max_label:
                segment['name'] = "Noise"
            else:
                # Use actual label ID for segment name
                segment['name'] = f"Segment_{segment['label']}"
        
        # Debug: Print first few segment names to verify
        print(f"DEBUG: First 5 segment names after assignment:")
        for seg in self.segments[:5]:
            print(f"  {seg['name']} (label: {seg['label']})")
        
        # Cache original_segments (excluding Noise) for threshold filtering
        self.original_segments = [s for s in self.segments if not self._is_small_segment(s)]
        
        # Remember the Noise label from original segmentation
        self._original_noise_label = max_label
        
        # Process events to keep UI responsive
        QApplication.processEvents()

        # Display the segments in an label layer
        print('Adding segmentation layer to viewer...')
        self.viewer.add_labels(self.labels, name='Segmentation')
        print('Segmentation complete!')
        
        # Update top 5 panel after segmentation
        self._update_top5_panel()

    def _is_small_segment(self, segment):
        """Check if a segment is the small segments collection"""
        small_label = getattr(self, '_small_segments_label', None)
        if small_label is not None and segment.get('label') == small_label:
            return True
        return False
    
    def show_popup_window(self):
        """ Define a pop-up window for the uncertainty list """

        # (24.05.2024)
        self.popup_window = QWidget()
        self.popup_window.setWindowTitle('Napari (segment list)')
        self.popup_window.setMinimumSize(QSize(350, 300))
        vbox_layout = QVBoxLayout()
        self.popup_window.setLayout(vbox_layout)

        # define a scroll area inside the pop-up window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        vbox_layout.addWidget(scroll_area)

        # Define a group box inside the scroll area
        group_box = QGroupBox('List of segments:')
        scroll_area.setWidget(group_box)
        grid_layout = QGridLayout()
        group_box.setLayout(grid_layout)

        # add widgets to the group box
        grid_layout.addWidget(QLabel('Segment'), 0, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), 0, 1)
        grid_layout.addWidget(QLabel('Counts'), 0, 2)
        grid_layout.addWidget(QLabel('done'), 0, 3)

        # Separate small segments from regular segments
        regular_segments = [s for s in self.segments if not self._is_small_segment(s) and not s['done']]
        small_segments = [s for s in self.segments if self._is_small_segment(s) and not s['done']]
        
        # Reverse regular segments (highest uncertainty first)
        regular_segments_reversed = list(reversed(regular_segments))
        
        # Track the highlighted button to scroll to it later
        highlighted_button = None
        
        # Display regular segments (highest uncertainty first)
        idx = 1
        for segment in regular_segments_reversed:
            button = self.new_entry(segment, grid_layout, idx)
            if button and self.current_zoomed_segment and segment.get('label') == self.current_zoomed_segment.get('label'):
                highlighted_button = button
            idx += 1
        
        # Display small segments at the end
        for segment in small_segments:
            button = self.new_entry(segment, grid_layout, idx)
            if button and self.current_zoomed_segment and segment.get('label') == self.current_zoomed_segment.get('label'):
                highlighted_button = button
            idx += 1

        # show a horizontal line
        idx += 1
        line = QWidget()
        line.setFixedHeight(3)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet('background-color: mediumblue')
        grid_layout.addWidget(line, idx, 0, 1, -1)

        # The treated areas are shown in the lower part of the group box
        treated_segments = [s for s in self.segments if s['done']]
        
        if len(treated_segments) > 0:
            idx += 1
            grid_layout.addWidget(QLabel('Segment'), idx, 0)
            grid_layout.addWidget(QLabel('Uncertainty'), idx, 1)
            grid_layout.addWidget(QLabel('Counts'), idx, 2)
            grid_layout.addWidget(QLabel('Re-enable'), idx, 3)

            idx += 1
            for segment in treated_segments:
                self.new_entry(segment, grid_layout, idx)
                idx += 1

        # Show the pop-up window
        self.popup_window.show()
        
        # Scroll to the highlighted segment if there is one
        if highlighted_button:
            # Use QTimer to ensure the layout is fully updated before scrolling
            QTimer.singleShot(0, lambda: scroll_area.ensureWidgetVisible(highlighted_button))
    
    def _refresh_popup_window(self):
        """Refresh the popup window content"""
        if not self.popup_window or not self.popup_window.isVisible():
            return
        
        # Update content without closing/reopening
        self._update_popup_window_content()
    
    def _update_popup_window_content(self):
        """Update the popup window content without recreating the window"""
        if not self.popup_window:
            return
        
        # Get the existing layout
        vbox_layout = self.popup_window.layout()
        
        # Clear all existing widgets
        while vbox_layout.count():
            item = vbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Recreate the content (same as show_popup_window)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        vbox_layout.addWidget(scroll_area)
        
        group_box = QGroupBox('List of segments:')
        scroll_area.setWidget(group_box)
        grid_layout = QGridLayout()
        group_box.setLayout(grid_layout)
        
        # Add headers
        grid_layout.addWidget(QLabel('Segment'), 0, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), 0, 1)
        grid_layout.addWidget(QLabel('Counts'), 0, 2)
        grid_layout.addWidget(QLabel('done'), 0, 3)
        
        # Separate small segments from regular segments
        regular_segments = [s for s in self.segments if not self._is_small_segment(s) and not s['done']]
        small_segments = [s for s in self.segments if self._is_small_segment(s) and not s['done']]
        
        # Reverse regular segments (highest uncertainty first)
        regular_segments_reversed = list(reversed(regular_segments))
        
        # Track the highlighted button to scroll to it later
        highlighted_button = None
        
        # Display regular segments (highest uncertainty first)
        idx = 1
        for segment in regular_segments_reversed:
            button = self.new_entry(segment, grid_layout, idx)
            if button and self.current_zoomed_segment and segment.get('label') == self.current_zoomed_segment.get('label'):
                highlighted_button = button
            idx += 1
        
        # Display small segments at the end
        for segment in small_segments:
            button = self.new_entry(segment, grid_layout, idx)
            if button and self.current_zoomed_segment and segment.get('label') == self.current_zoomed_segment.get('label'):
                highlighted_button = button
            idx += 1
        
        # Show horizontal line
        idx += 1
        line = QWidget()
        line.setFixedHeight(3)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet('background-color: mediumblue')
        grid_layout.addWidget(line, idx, 0, 1, -1)
        
        # The treated areas are shown in the lower part of the group box
        treated_segments = [s for s in self.segments if s['done']]
        
        if len(treated_segments) > 0:
            idx += 1
            grid_layout.addWidget(QLabel('Segment'), idx, 0)
            grid_layout.addWidget(QLabel('Uncertainty'), idx, 1)
            grid_layout.addWidget(QLabel('Counts'), idx, 2)
            grid_layout.addWidget(QLabel('restore'), idx, 3)
            
            idx += 1
            for segment in treated_segments:
                self.new_entry(segment, grid_layout, idx)
                idx += 1
        
        # Scroll to the highlighted segment if there is one
        if highlighted_button:
            # Use QTimer to ensure the layout is fully updated before scrolling
            QTimer.singleShot(0, lambda: scroll_area.ensureWidgetVisible(highlighted_button))
        
    def new_entry(self, segment: dict, grid_layout: QGridLayout, idx: int):
        """
        New entry for 'Area n' in the grid layout

        Parameters
        ----------
        segment : dict
            'name', 'uncertainty', 'counts', 'com', and 'done'
            for a specific area
        grid_layout : QGridLayout
            Layout for a QGroupBox
        idx : int
            Index in the grid_layout
            
        Returns
        -------
        QPushButton
            The segment button widget (useful for scrolling to it)
        """

        # (13.08.2024, updated 07.10.2025)
        # Define some buttons and labels
        button1 = QPushButton(segment['name'])
        button1.clicked.connect(lambda: self.zoom_in(segment, 0.75))

        if segment['done']:
            # disable button1 for treated areas
            button1.setEnabled(False)
        
        # Highlight if this is the currently zoomed segment
        if (self.current_zoomed_segment is not None and 
            segment.get('label') == self.current_zoomed_segment.get('label')):
            button1.setStyleSheet('background-color: lightblue; font-weight: bold;')
        
        grid_layout.addWidget(button1, idx, 0)

        uncertainty = '%.3f' % (segment['uncertainty'])
        label1 = QLabel(uncertainty)
        grid_layout.addWidget(label1, idx, 1)

        counts = '%d' % (segment['counts'])
        label2 = QLabel(counts)
        grid_layout.addWidget(label2, idx, 2)

        if segment['done']:
            button3 = QPushButton('re-enable')
            button3.clicked.connect(lambda: self.re_enable(segment))
        else:
            button3 = QPushButton('done')
            button3.clicked.connect(lambda: self.done(segment))
        grid_layout.addWidget(button3, idx, 3)
        
        return button1

    def zoom_in(self, segment: dict, margin_factor: float):
        """
        Show a segment and its immediate surroundings in a 3D view.
        """

        # (25.06.2025, updated 07.10.2025)
        # Track currently zoomed segment
        self.current_zoomed_segment = segment
        
        # Update segment count before zooming
        label = segment['label']
        count = int(np.sum(self.labels == label))  # Convert to Python int
        segment['counts'] = count
        print(f"DEBUG: Updated count for {segment['name']}: {count}")
        
        # Save current image layer settings
        self._save_image_layer_settings()
        
        self.viewer.layers.clear()          # Delete all layers in Napari

        # Determine the segment to be displayed
        label = segment['label']            # target label
        mask  = (self.labels == label)      # Segment mask

        # Calculate bounding box
        coords = np.argwhere(mask)
        minz, miny, minx = coords.min(axis=0)
        maxz, maxy, maxx = coords.max(axis=0)

        # Enlarge box
        sz, sy, sx = maxz - minz + 1, maxy - miny + 1, maxx - minx + 1
        size = max(sx, sy, sz)
        
        # Adaptive margin: more context for smaller segments
        # Minimum margin of 30, scales up to proportional margin for larger segments
        base_margin = int(size * margin_factor / 2)
        min_margin = 30
        margin = max(min_margin, base_margin)

        # Limitation to the image
        shape = self.image.shape
        startz = max(minz - margin, 0)
        starty = max(miny - margin, 0)
        startx = max(minx - margin, 0)
        endz   = min(maxz + margin + 1, shape[0])
        endy   = min(maxy + margin + 1, shape[1])
        endx   = min(maxx + margin + 1, shape[2])

        # Save the coordinates of the cropped image
        segment['coords'] = [[startz, starty, startx], [endz, endy, endx]]

        # Cropping
        cropped_image = self.image[startz:endz, starty:endy, startx:endx]
        cropped_segPred = self.segPred[startz:endz, starty:endy,
            startx:endx]
        cropped_labels = self.labels[startz:endz, starty:endy, startx:endx]

        # Remap segPred to label 1 for consistent coloring (blue)
        cropped_segPred_display = np.where(cropped_segPred > 0, 1, 0).astype(np.uint8)
        
        # Remap segment to label 1 for consistent coloring (red)
        masked_labels = np.where(cropped_labels == label, 1, 0).astype(np.uint8)
        
        # Create neighboring segments layer (other segments in this region)
        neighboring_segments = np.where(
            (cropped_labels > 0) & (cropped_labels != label),
            cropped_labels,
            0
        ).astype(np.int32)

        # Display data in Napari
        name1 = 'Cropped ' + self.stem1
        name2 = 'Cropped ' + self.stem2
        name3 = segment['name']
        image_layer = self.viewer.add_image(cropped_image, name=name1)
        
        # Apply saved image layer settings
        self._apply_image_layer_settings(image_layer)
        
        # Connect event listener to save settings when user changes them
        image_layer.events.contrast_limits.connect(self._on_image_settings_changed)
        image_layer.events.gamma.connect(self._on_image_settings_changed)
        
        # Add segPred layer with blue color for vessels
        segpred_layer = self.viewer.add_labels(cropped_segPred_display, name=name2)
        # Use CyclicLabelColormap - provide same color multiple times to ensure consistency
        blue_colormap = CyclicLabelColormap(colors=['blue', 'blue'])
        segpred_layer.colormap = blue_colormap
        
        # Add segment layer with red color for the current segment  
        segment_layer = self.viewer.add_labels(masked_labels, name=name3)
        # Use CyclicLabelColormap - provide same color multiple times to ensure consistency
        red_colormap = CyclicLabelColormap(colors=['red', 'red'])
        segment_layer.colormap = red_colormap
        
        # Add neighboring segments layer (hidden by default)
        neighbors_layer = self.viewer.add_labels(neighboring_segments, name='Neighboring Segments')
        neighbors_layer.visible = False  # Hidden by default for clean view
        # Sync checkbox state if it exists
        if hasattr(self, 'cbx_show_neighbors'):
            neighbors_layer.visible = self.cbx_show_neighbors.isChecked()

        # Set the appropriate level and focus
        com = ndimage.center_of_mass(masked_labels)     # center of mass
        com = tuple(int(round(c)) for c in com)
        self.viewer.dims.current_step = com
        self.viewer.camera.center = com

        # Set selected label to 1 (since we remapped)
        segment_layer.selected_label = 1
        
        # Focus on the segment layer (the one being edited)
        self.viewer.layers.selection.active = segment_layer
        
        # Update popup window to show highlight (if open)
        if self.popup_window and self.popup_window.isVisible():
            self._update_popup_window_content()

    def done(self, segment: dict):
        """
        Transfer data from the segment to the labels, segPred and uncertainty
        layer and close the layers for the cropped images
        """

        # (18.07.2024, updated 07.10.2025)
        self.compare_and_transfer(segment)  # transfer of data
        segment['done'] = True              # mark this area as treated
        
        # Clear current zoomed segment since we're going back to overview
        self.current_zoomed_segment = None
        
        # Update segment count
        label = segment['label']
        count = int(np.sum(self.labels == label))  # Convert to Python int
        segment['counts'] = count
        print(f"DEBUG: Updated count for {segment['name']}: {count}")

        # Save current image layer settings
        self._save_image_layer_settings()
        
        # Close cropped images and show image, segPred und labels
        self.viewer.layers.clear()
        image_layer = self.viewer.add_image(self.image, name=self.stem1)
        
        # Apply saved image layer settings
        self._apply_image_layer_settings(image_layer)
        
        # Connect event listener to save settings when user changes them
        image_layer.events.contrast_limits.connect(self._on_image_settings_changed)
        image_layer.events.gamma.connect(self._on_image_settings_changed)
        
        self.viewer.add_labels(self.segPred, name=self.stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')

        # Update popup window if it was open
        if self.popup_window and self.popup_window.isVisible():
            self._update_popup_window_content()
        
        # Update top 5 panel after marking segment as done
        self._update_top5_panel()

    def re_enable(self, segment: dict):
        """ Re-enable the data of a specific area in the pop-up window """

        # (19.07.2024)
        segment['done'] = False
        
        # Update popup window content instead of recreating
        if self.popup_window and self.popup_window.isVisible():
            self._update_popup_window_content()
        else:
            self.show_popup_window()
        
        # Update top 5 panel after restoring segment
        self._update_top5_panel()

    def compare_and_transfer(self, segment: dict):
        """
        Compare old and new data and transfer the changes to the segPred,
        uncertainty and labels data

        Parameters
        ----------
        segment : dict
            Data of the segment
        """

        # (09.08.2024)
        name        = segment['name']
        label       = segment['label']
        uncertainty = segment['uncertainty']
        coords      = segment['coords']

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):

            # Data of the segment
            layer = self.viewer.layers[name]
            segment_data = layer.data

            # Original coordinates of the segment
            start_z, start_y, start_x = coords[0]
            end_z,   end_y,   end_x   = coords[1]

            # Remap segment_data back to original label (it was remapped to 1 for display)
            # segment_data contains 0 and 1, we need to convert 1 → original label
            segment_data_remapped = np.where(segment_data > 0, label, 0)

            # Create an empty image and insert the remapped segment data
            new_data = np.zeros_like(self.labels, dtype=int)
            new_data[start_z:end_z, start_y:end_y, start_x:end_x] = segment_data_remapped

            # compare new and old data
            old_data = np.where(self.labels == label, label, 0)
            delta = new_data - old_data

            add_data = np.where(delta > 0)       # new data points
            del_data = np.where(delta < 0)       # deleted data points

            # transfer the changes to the labels layer
            self.labels[add_data] = label
            self.labels[del_data] = 0

            # transfer the changes to the segPred layer
            self.segPred[add_data] = 1
            self.segPred[del_data] = 0

            # transfer the changes to the uncertainty layer
            self.uncertainty[add_data] = uncertainty
            self.uncertainty[del_data] = 0.0

    def save_intermediate_data(self):
        """
        Save the segPred, uncertainty and labels data to files on hard drive.
        """

        # (26.07.2024, updated 07.10.2025)
        # Always save to data directory with _temp suffix
        if not hasattr(self, 'parent') or not self.parent:
            QMessageBox.warning(self, 'No Data Directory', 
                              'Please load a dataset first using "Load Dataset"')
            return
        
        if not hasattr(self, 'segPred') or not hasattr(self, 'uncertainty'):
            QMessageBox.warning(self, 'No Data', 
                              'No segmentation data to save')
            return

        # 1st: save the segPred data with _temp suffix
        stem = self.stem2 + '_temp'
        filename = self.parent.joinpath(stem).with_suffix('.tif')
        print(f'Saving intermediate segPred: {filename}')
        try:
            imwrite(filename, self.segPred)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # 2nd: save the uncertainty data with _temp suffix
        stem = self.stem3 + '_temp'
        filename = self.parent.joinpath(stem).with_suffix('.tif')
        print(f'Saving intermediate uncertainty: {filename}')
        try:
            imwrite(filename, self.uncertainty)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # 3rd: Save the labels to segmentation directory
        if self.data_manager.segmentation_dir and self.current_triplet:
            labels_file = self.data_manager.segmentation_dir / f"{self.current_triplet.base_name}_labels.tif"
            print(f'Saving labels: {labels_file}')
            try:
                imwrite(labels_file, self.labels.astype(np.int32))
            except BaseException as error:
                print(f'Warning: Could not save labels: {error}')
            
            # 4th: Save the segments metadata (clean version without numpy arrays)
            segments_file = self.data_manager.segmentation_dir / f"{self.current_triplet.base_name}_segments.json"
            print(f'Saving segments metadata: {segments_file}')
            try:
                # Clean segments data - remove any numpy arrays or large data
                clean_segments = []
                for seg in self.segments:
                    # Clean coords - convert numpy arrays/scalars to lists of Python ints
                    coords_value = seg.get('coords')
                    if coords_value is not None:
                        try:
                            # Handle nested structures and numpy types
                            coords_value = [[int(c) for c in coord] for coord in coords_value]
                        except (TypeError, ValueError):
                            # If conversion fails, set to None
                            coords_value = None
                            print(f"Warning: Could not convert coords for segment {seg.get('label')}, setting to None")
                    
                    # Determine default name based on label
                    uncertainty = float(seg.get('uncertainty', 0.0))
                    label_value = int(seg.get('label', 0))
                    
                    # Check if this is the Noise segment (by label, not uncertainty)
                    if self._small_segments_label and label_value == self._small_segments_label:
                        default_name = 'Noise'
                    else:
                        default_name = f"Segment_{label_value}"
                    
                    clean_seg = {
                        'label': label_value,
                        'uncertainty': uncertainty,
                        'counts': int(seg.get('counts', 0)),  # Ensure int conversion
                        'coords': coords_value,  # Now clean
                        'done': bool(seg.get('done', False))
                    }
                    
                    # Only store custom_name if it differs from the default
                    if seg.get('name') and seg['name'] != default_name:
                        clean_seg['custom_name'] = str(seg['name'])
                    
                    clean_segments.append(clean_seg)
                
                with segments_file.open('w', encoding='utf-8') as f:
                    json.dump(clean_segments, f, indent=2, cls=NumpyEncoder)
                    print(f'DEBUG: Saved {len(clean_segments)} segments to JSON')
            except BaseException as error:
                print(f'Warning: Could not save segments: {error}')
        
        print(f'✓ Saved intermediate data with _temp suffix')
        
        # Refresh loading dialog if open to show updated priorities
        if self.loading_dialog and self.loading_dialog.isVisible():
            print(f'DEBUG: Refreshing loading dialog after saving intermediate data')
            try:
                # Mark dataset as having temp files and update file references
                if self.current_triplet:
                    temp_segpred = self.parent / f"{self.stem2}_temp.tif"
                    temp_uncertainty = self.parent / f"{self.stem3}_temp.tif"
                    
                    self.current_triplet.has_temp = True
                    self.current_triplet.temp_segpred = temp_segpred
                    self.current_triplet.temp_uncertainty = temp_uncertainty
                    
                    print(f'DEBUG: Marked dataset as having temp files:')
                    print(f'DEBUG:   temp_segpred: {temp_segpred.exists()} - {temp_segpred.name}')
                    print(f'DEBUG:   temp_uncertainty: {temp_uncertainty.exists()} - {temp_uncertainty.name}')
                
                # Recalculate priorities to reflect done segments
                self.data_manager.calculate_priorities(
                    callback=lambda: self.loading_dialog._on_priorities_calculated(),
                    threaded=False
                )
            except Exception as e:
                print(f'DEBUG: Error refreshing dialog: {e}')
                import traceback
                traceback.print_exc()
        
        QMessageBox.information(self, 'Success', 
                              'Intermediate data saved.\nThese _temp files will be loaded automatically next time.')
    def load_intermediate_data(self):
        """
        DEPRECATED: Intermediate data is now loaded automatically via _temp files
        """
        QMessageBox.information(self, 'Deprecated', 
                              'This method is deprecated.\n\n' +
                              'Intermediate data (_temp files) are now loaded automatically.\n' +
                              'Use "Load Dataset" to load a dataset with its temp files.')
        return

    def save_final_result(self):
        """
        Save final results to 'done' subdirectory and remove files from data directory
        """
        # (13.08.2024, updated 07.10.2025)
        
        if not hasattr(self, 'parent') or not self.parent:
            QMessageBox.warning(self, 'No Data Directory', 
                              'Please load a dataset first')
            return
        
        if not hasattr(self, 'current_triplet') or not self.current_triplet:
            QMessageBox.warning(self, 'No Dataset', 
                              'Please load a dataset using "Load Dataset" button')
            return
        
        # 1st: Close any open segment layers and transfer changes
        lst = [layer for layer in self.viewer.layers
            if (layer.name.startswith('Segment_') or layer.name == 'Noise') and
            isinstance(layer, napari.layers.Labels)]

        for layer in lst:
            name = layer.name
            print(f'Closing layer: {name}')
            segment = next((s for s in self.segments if s['name'] == name), None)
            if segment is not None:
                self.compare_and_transfer(segment)
                segment['done'] = True

        # 2nd: Create done subdirectory
        done_dir = self.parent / 'done'
        done_dir.mkdir(exist_ok=True)
        print(f'DEBUG: Created done directory: {done_dir}')
        
        triplet = self.current_triplet
        
        # 3rd: Save final results to done directory
        try:
            # Save raw image
            raw_dest = done_dir / triplet.raw_file.name
            print(f'Moving {triplet.raw_file.name} to done/')
            triplet.raw_file.rename(raw_dest)
            
            # Save segPred (curated)
            segpred_dest = done_dir / triplet.segpred_file.name
            print(f'Saving curated segPred to done/{segpred_dest.name}')
            imwrite(segpred_dest, self.segPred)
            
            # Save uncertainty if requested
            if self.save_uncertainty:
                uncertainty_dest = done_dir / triplet.uncertainty_file.name
                print(f'Saving curated uncertainty to done/{uncertainty_dest.name}')
                imwrite(uncertainty_dest, self.uncertainty)
            
        except Exception as error:
            QMessageBox.warning(self, 'Error Saving', str(error))
            return
        
        # 4th: Remove original files from data directory
        try:
            print(f'DEBUG: Removing original files from data directory...')
            
            # Remove original segPred and uncertainty
            if triplet.segpred_file.exists():
                triplet.segpred_file.unlink()
                print(f'  Removed {triplet.segpred_file.name}')
            
            if triplet.uncertainty_file.exists():
                triplet.uncertainty_file.unlink()
                print(f'  Removed {triplet.uncertainty_file.name}')
            
            # Remove temp files if they exist
            if triplet.temp_segpred and triplet.temp_segpred.exists():
                triplet.temp_segpred.unlink()
                print(f'  Removed {triplet.temp_segpred.name}')
            
            if triplet.temp_uncertainty and triplet.temp_uncertainty.exists():
                triplet.temp_uncertainty.unlink()
                print(f'  Removed {triplet.temp_uncertainty.name}')
            
            print(f'✓ Dataset finalized and moved to done/')
            
        except Exception as error:
            print(f'Warning: Could not remove all files: {error}')
        
        # 5th: Clean up cache files
        try:
            print(f'DEBUG: Cleaning up cache files for {triplet.base_name}...')
            
            # Remove segmentation cache files
            if triplet.segmentation_dir:
                labels_file = triplet.segmentation_dir / f"{triplet.base_name}_labels.tif"
                segments_file = triplet.segmentation_dir / f"{triplet.base_name}_segments.json"
                
                if labels_file.exists():
                    labels_file.unlink()
                    print(f'  Removed {labels_file.name}')
                
                if segments_file.exists():
                    segments_file.unlink()
                    print(f'  Removed {segments_file.name}')
            
            # Remove from data manager's datasets list
            if triplet in self.data_manager.datasets:
                self.data_manager.datasets.remove(triplet)
                print(f'  Removed from datasets list')
            
            # Update cache file (remove this dataset's entries)
            if self.data_manager.cache_file and self.data_manager.cache_file.exists():
                try:
                    with self.data_manager.cache_file.open('r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Remove from priorities
                    if triplet.base_name in cache_data.get('priorities', {}):
                        del cache_data['priorities'][triplet.base_name]
                    
                    # Remove from datasets
                    cache_data['datasets'] = [
                        d for d in cache_data.get('datasets', []) 
                        if d.get('base_name') != triplet.base_name
                    ]
                    
                    # Save updated cache
                    with self.data_manager.cache_file.open('w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    print(f'  Updated cache file')
                except Exception as e:
                    print(f'  Warning: Could not update cache: {e}')
            
            print(f'✓ Cache cleaned up')
            
        except Exception as error:
            print(f'Warning: Could not clean up cache: {error}')
        
        # 6th: Clear current dataset
        self.current_triplet = None
        
        # 7th: Refresh loading dialog if open
        if self.loading_dialog and self.loading_dialog.isVisible():
            print(f'DEBUG: Refreshing loading dialog after finalization')
            try:
                self.loading_dialog._refresh_datasets()
            except Exception as e:
                print(f'DEBUG: Error refreshing dialog: {e}')
        
        # 8th: Show results
        self.viewer.layers.clear()
        self.viewer.add_image(self.image, name=self.stem1)
        self.viewer.add_labels(self.segPred, name=self.stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')
        
        QMessageBox.information(self, 'Success', 
                              f'Final results saved to done/ subdirectory.\n' +
                              f'Original files and cache removed from data directory.')

    def checkbox_save_uncertainty(self, state: Qt.Checked):
        """ Toggle the bool variable save_uncertainty """

        if state == Qt.Checked:
            self.save_uncertainty = True
        else:
            self.save_uncertainty = False

    def show_info(self):
        """ Show information about the current layer """

        # (25.07.2024)
        layer = self.viewer.layers.selection.active
        print('layer:', layer.name)

        if isinstance(layer, napari.layers.Image):
            image = layer.data

            print('type:',  type(image))
            print('dtype:', image.dtype)
            print('size:',  image.size)
            print('ndim:',  image.ndim)
            print('shape:', image.shape)
    
    def load_dataset_dialog(self):
        """Open dialog to select and load a dataset"""
        # (07.10.2025)
        # Close existing dialog if open
        if self.loading_dialog:
            self.loading_dialog.close()
        
        self.loading_dialog = LoadingDialog(self.data_manager, self)
        if self.loading_dialog.exec_():
            triplet = self.loading_dialog.get_selected_triplet()
            if triplet:
                self.load_dataset_from_triplet(triplet)
        
        self.loading_dialog = None
    
    def load_dataset_from_triplet(self, triplet: DatasetTriplet):
        """Load a dataset from a DatasetTriplet"""
        # (07.10.2025)
        try:
            # Close segment list window when loading new dataset (contents will be outdated)
            if self.popup_window and self.popup_window.isVisible():
                self.popup_window.close()
                self.popup_window = None
            
            # Clear current zoomed segment
            self.current_zoomed_segment = None
            
            self.current_triplet = triplet
            raw_file, segpred_file, uncertainty_file = self.data_manager.get_files_for_loading(triplet)
            
            # Load raw image
            self.parent = raw_file.parent
            self.stem1 = raw_file.stem
            # Handle .nii.gz
            if self.stem1.endswith('.nii'):
                self.stem1 = self.stem1[:-4]
            # Handle .ome.tif(f)
            if self.stem1.endswith('.ome'):
                self.stem1 = self.stem1[:-4]
            
            suffix = raw_file.suffix.lower()
            if suffix == '.gz' and self.stem1.endswith('.nii'):
                self.stem1 = self.stem1[:-4]
            
            if suffix in ['.tif', '.tiff', '.ome.tif', '.ome.tiff']:
                self.image = imread(raw_file)
            elif suffix in ['.nii', '.gz']:
                sitk_image = sitk.ReadImage(str(raw_file))
                self.image = sitk.GetArrayFromImage(sitk_image)
            
            image_layer = self.viewer.add_image(self.image, name=self.stem1)
            
            # Apply saved settings if they exist
            if self.image_contrast_limits_relative is not None:
                self._apply_image_layer_settings(image_layer)
            
            # Connect event listener to save settings when user changes them
            image_layer.events.contrast_limits.connect(self._on_image_settings_changed)
            image_layer.events.gamma.connect(self._on_image_settings_changed)
            
            # Load segPred
            self.stem2 = segpred_file.stem
            if self.stem2.endswith('.nii'):
                self.stem2 = self.stem2[:-4]
            # Remove _temp suffix if present to get base stem
            if self.stem2.endswith('_temp'):
                self.stem2 = self.stem2[:-5]
            
            suffix = segpred_file.suffix.lower()
            if suffix in ['.tif', '.tiff', '.ome.tif', '.ome.tiff']:
                self.segPred = imread(segpred_file)
            elif suffix in ['.nii', '.gz']:
                sitk_image = sitk.ReadImage(str(segpred_file))
                self.segPred = sitk.GetArrayFromImage(sitk_image)
            
            self.viewer.add_labels(self.segPred, name=self.stem2)
            
            # Load uncertainty
            self.stem3 = uncertainty_file.stem
            if self.stem3.endswith('.nii'):
                self.stem3 = self.stem3[:-4]
            # Remove _temp suffix if present to get base stem
            if self.stem3.endswith('_temp'):
                self.stem3 = self.stem3[:-5]
            
            suffix = uncertainty_file.suffix.lower()
            if suffix in ['.tif', '.tiff', '.ome.tif', '.ome.tiff']:
                self.uncertainty = imread(uncertainty_file)
            elif suffix in ['.nii', '.gz']:
                sitk_image = sitk.ReadImage(str(uncertainty_file))
                self.uncertainty = sitk.GetArrayFromImage(sitk_image)
            
            # Process segments - use precomputed if available
            QApplication.processEvents()
            
            if triplet.has_segmentation and triplet.segmentation_dir:
                print(f"DEBUG: Loading precomputed segmentation for {triplet.base_name}")
                # Load precomputed labels
                labels_file = triplet.segmentation_dir / f"{triplet.base_name}_labels.tif"
                if labels_file.exists():
                    self.labels = imread(labels_file)
                    print(f"DEBUG:   Loaded labels from {labels_file.name}")
                    
                    # Load precomputed segments
                    segments_file = triplet.segmentation_dir / f"{triplet.base_name}_segments.json"
                    if segments_file.exists():
                        with segments_file.open('r', encoding='utf-8') as f:
                            self.segments = json.load(f)
                        print(f"DEBUG:   Loaded {len(self.segments)} segments from {segments_file.name}")
                        
                        # Synthesize names from labels (name field no longer stored in JSON)
                        # Find max label to identify Noise segment
                        max_label = max(seg.get('label', 0) for seg in self.segments)
                        self._small_segments_label = max_label  # Store for later reference
                        
                        for seg in self.segments:
                            # Check if there's a custom name
                            if 'custom_name' in seg:
                                seg['name'] = seg['custom_name']
                            else:
                                # Synthesize default name
                                # Check if this is the Noise segment (by label, not uncertainty)
                                if seg.get('label') == max_label:
                                    seg['name'] = 'Noise'
                                else:
                                    seg['name'] = f"Segment_{seg['label']}"
                        
                        # Cache original for threshold filtering
                        # Keep Noise label in labels so voxels aren't lost
                        self.original_labels = self.labels.copy()
                        
                        # Cache segments excluding Noise (will be recreated on threshold change)
                        self.original_segments = [s for s in self.segments if s.get('label') != max_label]
                        
                        # Remember the original Noise label for filtering
                        self._original_noise_label = max_label
                        
                        # Display the segmentation layer
                        self.viewer.add_labels(self.labels, name='Segmentation')
                        print(f"✓ Loaded dataset: {triplet.base_name} (using precomputed segmentation)")
                    else:
                        # Fallback to calculation
                        print(f"DEBUG:   Segments file not found, calculating...")
                        self.find_segments(self.uncertainty)
                        print(f"✓ Loaded dataset: {triplet.base_name}")
                else:
                    # Fallback to calculation
                    print(f"DEBUG:   Labels file not found, calculating...")
                    self.find_segments(self.uncertainty)
                    print(f"✓ Loaded dataset: {triplet.base_name}")
            else:
                # No precomputed segmentation, calculate
                print(f"DEBUG: No precomputed segmentation, calculating for {triplet.base_name}")
                self.find_segments(self.uncertainty)
                print(f"✓ Loaded dataset: {triplet.base_name}")
            
            # Update top 5 panel after loading
            self._update_top5_panel()
            
        except Exception as error:
            QMessageBox.warning(self, 'Error loading dataset', str(error))
    
    def _on_threshold_changed(self, value: int):
        """Handle threshold spinbox change with debouncing"""
        # Stop and restart timer (debounce)
        self.threshold_timer.stop()
        self.threshold_timer.start(500)  # Wait 500ms after last change
    
    def _apply_threshold_debounced(self):
        """Apply threshold after debounce delay"""
        min_size = self.threshold_spinbox.value()
        print(f"DEBUG: Applying threshold: {min_size}")
        self.apply_threshold_filter(min_size)
    
    def apply_threshold_filter(self, min_size: int = None):
        """Apply threshold filter to segments based on size"""
        # (07.10.2025)
        if min_size is None:
            min_size = self.threshold_spinbox.value()
        
        if not hasattr(self, 'labels') or self.labels is None:
            return
        
        # Cache original labels if not already cached
        if self.original_labels is None:
            self.original_labels = self.labels.copy()
            self.original_segments = copy.deepcopy(self.segments)
        
        # Restore from original
        self.labels = self.original_labels.copy()
        self.segments = copy.deepcopy(self.original_segments)
        
        # Find old Noise label and treat it as background for regrouping
        old_noise_label = getattr(self, '_original_noise_label', None)
        if old_noise_label is not None:
            # Temporarily set old Noise voxels to 0 so they can be regrouped
            old_noise_mask = self.labels == old_noise_label
            self.labels[old_noise_mask] = 0
        
        # Apply new threshold to find small segments
        counts = np.bincount(self.labels.ravel())
        small_labels = np.where(counts < min_size)[0]
        small_labels = small_labels[small_labels != 0]
        print(f'DEBUG: Found {len(small_labels)} small segments with threshold {min_size}')
        
        # If there were old Noise voxels, they should all go into new Noise
        # (they're currently 0, so not counted as small_labels)
        # Add them to the new Noise group
        
        # Replace small labels with new Noise label
        max_label = np.max(self.labels) + 1
        # Group both new small segments AND old Noise voxels
        mask = np.isin(self.labels, small_labels)
        if old_noise_label is not None:
            mask = mask | old_noise_mask  # Include old Noise voxels
        self.labels[mask] = max_label
        
        # Update segments list (exclude old Noise)
        unique_labels = np.unique(self.labels)
        unique_labels = unique_labels[unique_labels != 0]
        counts = np.bincount(self.labels.ravel())
        
        # Filter segments and update counts
        filtered_segments = []
        for segment in self.segments:
            label = segment['label']
            if label in unique_labels:
                # Bounds check for counts array access
                if label < len(counts):
                    segment['counts'] = int(counts[label])  # Convert to Python int
                    filtered_segments.append(segment)
        
        # Add the "Noise" group if it exists
        if max_label in unique_labels and max_label < len(counts):
            segment = dict(
                name='Noise',
                label=max_label,
                uncertainty=0.9999,
                counts=int(counts[max_label]),
                coords=None,
                done=False,
            )
            filtered_segments.append(segment)
            self._small_segments_label = max_label  # Update reference
        
        self.segments = filtered_segments
        self.segments.sort(key=lambda x: x['uncertainty'])
        
        # Assign names (skip if already named)
        segment_num = 1
        for segment in self.segments:
            if segment['name'] == '':
                segment['name'] = f'Segment_{segment["label"]}'  # Use label ID, not sequential
                segment_num += 1
            elif segment['name'] == 'Noise':
                # Keep the Noise name
                pass
        
        # Update the Segmentation layer if it exists
        if 'Segmentation' in self.viewer.layers:
            self.viewer.layers['Segmentation'].data = self.labels
        
        print(f'Threshold updated to {min_size}. Found {len(self.segments)} segments.')
        
        # Refresh popup window if it's open
        if self.popup_window and self.popup_window.isVisible():
            print(f'DEBUG: Refreshing popup window after threshold change')
            self._refresh_popup_window()
        
        # Update top 5 panel after threshold change
        self._update_top5_panel()
    
    def _save_image_layer_settings(self):
        """Save current image layer settings (contrast, gamma) for later restoration"""
        # (07.10.2025)
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                # Get current contrast limits and data range
                clim_min, clim_max = layer.contrast_limits
                data_min, data_max = layer.data.min(), layer.data.max()
                data_range = data_max - data_min
                
                # Calculate relative positions (0.0 to 1.0)
                if data_range > 0:
                    min_relative = (clim_min - data_min) / data_range
                    max_relative = (clim_max - data_min) / data_range
                    self.image_contrast_limits_relative = (min_relative, max_relative)
                else:
                    self.image_contrast_limits_relative = (0.0, 1.0)
                
                self.image_gamma = layer.gamma
                break
    
    def _apply_image_layer_settings(self, image_layer):
        """Apply saved image layer settings to a new image layer"""
        # (07.10.2025)
        if self.image_contrast_limits_relative is not None:
            # Calculate absolute contrast limits from relative positions
            data_min, data_max = image_layer.data.min(), image_layer.data.max()
            data_range = data_max - data_min
            
            min_relative, max_relative = self.image_contrast_limits_relative
            clim_min = data_min + (min_relative * data_range)
            clim_max = data_min + (max_relative * data_range)
            
            image_layer.contrast_limits = (clim_min, clim_max)
        
        if self.image_gamma is not None:
            image_layer.gamma = self.image_gamma
    
    def _on_image_settings_changed(self, event):
        """Called when user changes image layer settings (contrast or gamma)"""
        # (07.10.2025)
        # Automatically save the new settings
        self._save_image_layer_settings()
    
    def _update_top5_panel(self):
        """Update the top 5 segments quick access panel"""
        # (29.10.2025)
        if not hasattr(self, 'top5_groupbox'):
            return
        
        # Clear existing widgets immediately
        layout = self.top5_groupbox.layout()
        # Remove all items from layout
        items_to_remove = []
        while layout.count():
            items_to_remove.append(layout.takeAt(0))
        
        # Delete widgets and nested layouts
        for item in items_to_remove:
            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout():
                # Clear nested layout
                sublayout = item.layout()
                while sublayout.count():
                    subitem = sublayout.takeAt(0)
                    if subitem.widget():
                        subwidget = subitem.widget()
                        subwidget.setParent(None)
                        subwidget.deleteLater()
                sublayout.deleteLater()
        
        # Process events to ensure widgets are removed before adding new ones
        QApplication.processEvents()
        
        # Get top 5 undone segments (excluding Noise)
        undone_segments = [s for s in self.segments if not s['done'] and not self._is_small_segment(s)]
        # Sort by uncertainty descending (highest first)
        undone_segments_sorted = sorted(undone_segments, key=lambda x: x['uncertainty'], reverse=True)
        top5 = undone_segments_sorted[:5]
        
        if len(top5) == 0:
            self.top5_groupbox.setVisible(False)
            return
        
        self.top5_groupbox.setVisible(True)
        
        # Add each segment with buttons
        for segment in top5:
            row_layout = QHBoxLayout()
            
            # Segment info label
            info_label = QLabel(f"{segment['name']} ({segment['uncertainty']:.3f})")
            info_label.setMinimumWidth(150)
            row_layout.addWidget(info_label)
            
            # Zoom button
            btn_zoom = QPushButton('View')
            btn_zoom.setMaximumWidth(50)
            btn_zoom.clicked.connect(lambda checked, s=segment: self.zoom_in(s, 0.75))
            row_layout.addWidget(btn_zoom)
            
            # Done button
            btn_done = QPushButton('Done')
            btn_done.setMaximumWidth(50)
            btn_done.clicked.connect(lambda checked, s=segment: self.done(s))
            row_layout.addWidget(btn_done)
            
            layout.addLayout(row_layout)
    
    def _toggle_neighbors_visibility(self, state):
        """Toggle visibility of neighboring segments layer and update focus"""
        # (29.10.2025)
        if 'Neighboring Segments' not in self.viewer.layers:
            return
            
        neighbors_layer = self.viewer.layers['Neighboring Segments']
        neighbors_layer.visible = (state == Qt.Checked)
        
        # Update layer focus
        if state == Qt.Checked:
            # Focus on neighboring segments when shown
            self.viewer.layers.selection.active = neighbors_layer
        else:
            # Focus back on current segment when hidden
            if self.current_zoomed_segment and self.current_zoomed_segment['name'] in self.viewer.layers:
                self.viewer.layers.selection.active = self.viewer.layers[self.current_zoomed_segment['name']]
    
    def _toggle_identify_mode(self, checked):
        """Toggle identify mode on/off"""
        # (29.10.2025)
        self._identify_mode = checked
        
        if checked:
            # Enable identify mode - connect mouse click event
            if 'Neighboring Segments' in self.viewer.layers:
                neighbors_layer = self.viewer.layers['Neighboring Segments']
                neighbors_layer.mouse_double_click_callbacks.append(self._on_neighbor_clicked)
                neighbors_layer.visible = True  # Ensure visible in identify mode
                self.viewer.layers.selection.active = neighbors_layer
                
                # Sync checkbox
                if hasattr(self, 'cbx_show_neighbors'):
                    self.cbx_show_neighbors.blockSignals(True)
                    self.cbx_show_neighbors.setChecked(True)
                    self.cbx_show_neighbors.blockSignals(False)
            else:
                QMessageBox.information(self, 'Not in Segment View',
                    'Please zoom into a segment first to use identify mode.')
                # Uncheck the button
                sender = self.sender()
                if sender:
                    sender.setChecked(False)
        else:
            # Disable identify mode - disconnect event
            if 'Neighboring Segments' in self.viewer.layers:
                neighbors_layer = self.viewer.layers['Neighboring Segments']
                if self._on_neighbor_clicked in neighbors_layer.mouse_double_click_callbacks:
                    neighbors_layer.mouse_double_click_callbacks.remove(self._on_neighbor_clicked)
                # Don't auto-hide the layer - let user control it via checkbox
    
    def _on_neighbor_clicked(self, layer, event):
        """Handle click on neighboring segments layer"""
        # (29.10.2025)
        # Get clicked position
        coords = layer.world_to_data(event.position)
        z, y, x = [int(round(c)) for c in coords]
        
        # Check bounds
        if not (0 <= z < layer.data.shape[0] and
                0 <= y < layer.data.shape[1] and
                0 <= x < layer.data.shape[2]):
            return
        
        # Get segment label at clicked position
        segment_label = int(layer.data[z, y, x])
        
        if segment_label == 0:
            return  # Clicked on background
        
        # Find segment
        segment = next((s for s in self.segments if s['label'] == segment_label), None)
        if segment is None:
            return
        
        # Show dialog with jump option
        msg = QMessageBox(self)
        msg.setWindowTitle('Segment Identified')
        msg.setText(f"Segment: {segment['name']}\n"
                   f"Label: {segment_label}\n"
                   f"Uncertainty: {segment['uncertainty']:.4f}\n"
                   f"Voxel count: {segment['counts']}\n"
                   f"Status: {'Done' if segment['done'] else 'Not done'}")
        
        if not segment['done']:
            jump_button = msg.addButton('Jump to Segment', QMessageBox.AcceptRole)
            msg.addButton('Cancel', QMessageBox.RejectRole)
            msg.exec_()
            
            if msg.clickedButton() == jump_button:
                # Hide neighboring segments when jumping
                if hasattr(self, 'cbx_show_neighbors'):
                    self.cbx_show_neighbors.blockSignals(True)
                    self.cbx_show_neighbors.setChecked(False)
                    self.cbx_show_neighbors.blockSignals(False)
                
                self.zoom_in(segment, 0.75)
        else:
            msg.addButton('OK', QMessageBox.AcceptRole)
            msg.exec_()
    
    def _on_segmentation_complete(self, dataset_name: str, success: bool, error_message: str):
        """
        Callback when segmentation calculation completes
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset that was processed
        success : bool
            Whether the calculation succeeded
        error_message : str
            Error message if failed
        """
        try:
            print(f"\nDEBUG: Segmentation callback triggered for {dataset_name}")
            print(f"DEBUG:   Success: {success}")
            
            if hasattr(self, 'segmentation_worker'):
                print(f"DEBUG:   Queue size: {self.segmentation_worker.get_queue_size()}")
            
            if success:
                print(f"✓ Segmentation complete for {dataset_name}")
                
                # Mark dataset as having segmentation
                for triplet in self.data_manager.datasets:
                    if triplet.base_name == dataset_name:
                        triplet.has_segmentation = True
                        triplet.segmentation_dir = self.data_manager.segmentation_dir
                        print(f"DEBUG:   Marked {dataset_name} as having segmentation")
                        print(f"DEBUG:   Triplet now: has_segmentation={triplet.has_segmentation}")
                        break
                
                # Trigger priority recalculation (live update)
                if self.loading_dialog and self.loading_dialog.isVisible():
                    print(f"DEBUG:   Refreshing loading dialog (live update)")
                    try:
                        # Recalculate priorities directly without re-detecting
                        print(f"DEBUG:   Calling calculate_priorities...")
                        self.data_manager.calculate_priorities(
                            callback=lambda: self.loading_dialog._on_priorities_calculated(),
                            threaded=False  # Synchronous for immediate update
                        )
                    except Exception as e:
                        print(f"DEBUG:   Error refreshing dialog: {e}")
                else:
                    print(f"DEBUG:   Loading dialog not visible, skipping refresh")
            else:
                print(f"✗ Segmentation failed for {dataset_name}: {error_message}")
                try:
                    QMessageBox.warning(self, 'Segmentation Error', 
                                      f"Failed to calculate segmentation for {dataset_name}:\n{error_message}")
                except:
                    pass  # Ignore if UI is shutting down
        except Exception as e:
            print(f"DEBUG: Error in segmentation callback: {e}")
    
    def _queue_unsegmented_datasets(self):
        """Queue all datasets without segmentation for background processing"""
        if not self.data_manager.datasets:
            print("DEBUG: Cannot queue datasets - no datasets found")
            return
        
        # Ensure segmentation directory exists
        if not self.data_manager.segmentation_dir:
            if self.data_manager.data_directory:
                self.data_manager.segmentation_dir = (
                    self.data_manager.data_directory / self.data_manager.SEGMENTATION_DIR
                )
                self.data_manager.segmentation_dir.mkdir(exist_ok=True)
                print(f"DEBUG: Created segmentation directory: {self.data_manager.segmentation_dir}")
            else:
                print("DEBUG: Cannot queue datasets - no data directory set")
                return
        
        # Get initial queue size
        initial_queue_size = self.segmentation_worker.get_queue_size()
        print(f"\nDEBUG: Initial queue size: {initial_queue_size}")
        print(f"DEBUG: Checking {len(self.data_manager.datasets)} datasets for segmentation status")
        
        queued_count = 0
        already_have_seg = 0
        already_queued = 0
        
        # Get list of datasets already in queue (if we can track them)
        queued_datasets = getattr(self.segmentation_worker, '_queued_datasets', set())
        
        for triplet in self.data_manager.datasets:
            if not triplet.has_segmentation:
                # Check if already queued
                if triplet.base_name in queued_datasets:
                    print(f"  🔄 {triplet.base_name}: Already in queue, skipping")
                    already_queued += 1
                    continue
                
                print(f"  ⏳ {triplet.base_name}: No segmentation, adding to queue")
                # Add to worker queue
                uncertainty_file = (triplet.temp_uncertainty if triplet.temp_uncertainty 
                                  else triplet.uncertainty_file)
                segpred_file = (triplet.temp_segpred if triplet.temp_segpred 
                              else triplet.segpred_file)
                
                self.segmentation_worker.add_dataset(
                    dataset_name=triplet.base_name,
                    uncertainty_file=uncertainty_file,
                    segpred_file=segpred_file,
                    output_dir=self.data_manager.segmentation_dir
                )
                queued_datasets.add(triplet.base_name)
                queued_count += 1
            else:
                print(f"  ✓ {triplet.base_name}: Already has segmentation")
                already_have_seg += 1
        
        # Store queued datasets for tracking
        self.segmentation_worker._queued_datasets = queued_datasets
        
        final_queue_size = self.segmentation_worker.get_queue_size()
        
        print(f"\nDEBUG: Segmentation status summary:")
        print(f"  - Already calculated: {already_have_seg}")
        print(f"  - Already in queue: {already_queued}")
        print(f"  - Newly queued: {queued_count}")
        print(f"  - Initial queue size: {initial_queue_size}")
        print(f"  - Final queue size: {final_queue_size}")
        print(f"  - Expected: {initial_queue_size + queued_count}\n")
