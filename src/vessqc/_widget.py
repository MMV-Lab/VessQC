"""
_widget.py
==========

This module contains a Napari plugin that can be used to check and correct
3D views of blood vessels.

Functions
---------
_label_value_sparse
    Segments contiguous voxels with similar uncertainty and assigns them
    unique global labels.
_segment_uncertainties
    Segmend 3D voxels by unique uncertainty values
_merge_labels
    Merge results of segmentation into a single label volume
_merge_small_segments
    Merge labels that occur less than 'min_size' times into a new label.
_create_segments
    Create segment metadata from labels and uncertainties.
_compute_bbox
    Determine a bounding box
_expand_bbox
    Enlarge the bounding box
_crop_volumes
    Cropping the data
_display_cropped
    Plotting in Napari
_focus_viewer
    Focus the camera
_jsonify
    Converts Python data types into a form that can be saved as a JSON file

Classes
-------
ExampleQWidget
    Class for displaying and correcting a 3D image of blood vessels.
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

import copy
from dataclasses import asdict, dataclass
from joblib import Parallel, delayed
import json
from .multiple_viewer_widget import MultipleViewerWidget, CrossWidget
import numpy as np
import napari
from pathlib import Path
from qtpy.QtCore import QSize, Qt
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
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from scipy import ndimage
import SimpleITK as sitk
import tempfile
from tifffile import imread, imwrite
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import napari

def _label_value_sparse(uncertainty, uncert, idx, num_uncert):
    """
    Segments contiguous voxels with similar uncertainty and assigns them
    unique global labels.

    Parameters
    ----------
    uncertainty : np.ndarray
        3D array with information on the uncertainty of the calculated
        data points
    uncert : float
        Single uncertainty value
    idx : int
        Index of the unique uncertainty value
    num_uncert : int
        Number of unique uncertainty values

    Returns
    -------
    dict
        Dictionary with the keys:
        - indices : tuple of np.ndarray
            Indices of voxels belonging to the segment
        - global_labels : np.ndarray
            Global label values at the given indices
        - uncert : float
            Uncertainty value of the segment
        - num_features : int
            Number of found features

    None
        If no voxels match the criteria.
    """

    # (03.07.2025)
    tolerance = 1e-2
    structure = np.ones((3, 3, 3), dtype=int)       # Connectivity array
    mask = np.abs(uncertainty - uncert) < tolerance
    if not np.any(mask):
        return None

    labels, num_features = ndimage.label(mask, structure)    # Segmentation
    if num_features == 0:
        return None

    # Calculate global unique labels
    # labels = 1, 2, 3, ... num_features
    # e.g. global_labels = 3, 23, 43, ... for num_uncert = 20, idx = 3
    global_labels = idx + (labels - 1) * num_uncert
    global_labels[labels == 0] = 0

    indices = np.where(mask)
    result = dict(
        indices = indices,
        global_labels = global_labels[indices],
        uncert = uncert,
        num_features = num_features
    )
    return result

def _segment_uncertainties(uncertainty: np.ndarray):
    """
    Segmend 3D voxels by unique uncertainty values

    Parameters
    ----------
    uncertainty : np.ndarray
        3D array of uncertainty values.

    Returns
    -------
    result : list of dict
        Each dict contains 'incices', 'global_labels', 'uncert', 'num_features'
    """

    # (12.02.2026)
    unique_uncertainties = np.unique(uncertainty)
    unique_uncertainties = unique_uncertainties[unique_uncertainties > 0]
    num_uncert = len(unique_uncertainties)

    results = Parallel(n_jobs=-1)(
        delayed(_label_value_sparse)(uncertainty, uncert, idx, num_uncert)
        for idx, uncert in enumerate(unique_uncertainties, start=1)
    )

    return [r for r in results if r is not None]

def _merge_labels(results: list, shape: tuple):
    """
    Merge results of segmentation into a single label volume

    Parameters
    ----------
    results : list of dict
        Output of segment_uncertainties
    shape : tuple
        Shape of the original image

    Returns
    -------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping labels -> uncertainty
    """

    # (17.02.2026)
    labels = np.zeros(shape, dtype=np.int32)
    uncert_values = {0: 0.0}

    # Reconstruct der labels array with global labels
    for result in results:
        indices = result['indices']
        global_labels = result['global_labels']
        labels[indices] = global_labels

        unique_labels = np.unique(global_labels)
        unique_labels = unique_labels[unique_labels != 0]

        for lbl in unique_labels:
            uncert_values[lbl] = result['uncert']

    return labels, uncert_values

def _merge_small_segments(labels: np.ndarray, uncert_values: dict, min_size: int):
    """
    Merge labels that occur less than 'min_size' times into a new label.

    Parameters
    ----------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping label -> uncertainty
    min_size : int
        Minimum voxel count to keep a segment

    Returns
    -------
    labels : np.ndarray
        Updated label volume
    uncert_values : dict
        Updated uncertainty dictionary
    """

    # (06.03.2026)
    # Find all labels that appear less than min_size times
    counts = np.bincount(labels.ravel())
    small_labels = np.where(counts < min_size)[0]
    small_labels = small_labels[small_labels != 0]

    # Replaces all labels that occur less than min_size with max_label
    if len(small_labels) > 0:
        max_label = np.max(labels) + 1
        mask = np.isin(labels, small_labels)

        labels[mask] = max_label
        uncert_values[max_label] = 0.9999

    return labels, uncert_values

def _create_segments(labels: np.ndarray, uncert_values: dict):
    """
    Create segment metadata from labels and uncertainties.

    Parameters
    ----------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping labels -> uncertainty

    Returns
    -------
    segments : List
        List of Segments
    """

    # (18.02.2026, revised 29.04.2026)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    counts = np.bincount(labels.ravel())

    segments = [
        Segment(
            name = f"Segment_{i}",
            label = int(label),
            uncertainty = float(uncert_values.get(label, 0.0)),
            count = int(counts[label]),
        )
        for i, label in enumerate(unique_labels, start=1)
    ]

    # Sort by uncertainty ascending
    segments.sort(key=lambda x: x.uncertainty)
    return segments

def _compute_bbox(mask: np.ndarray):
    """Determine a bounding box"""

    # (06.03.2026)
    coords = np.argwhere(mask)

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    return [[min_z, min_y, min_x], [max_z, max_y, max_x]]

def _expand_bbox(b_box: list, shape: tuple, margin_factor: float):
    """
    Enlarge the bounding box

    Parameters
    ----------
    b_box : list
        Bounding box
    shape : tuple
        Shape of the image
    margin_factor : float
        Factor for enlarging the b_box

    Returns
    -------
    b_box : list
        Updated bounding box
    """

    # (06.03.2026)
    (min_z, min_y, min_x), (max_z, max_y, max_x) = b_box

    size_z = max_z - min_z + 1
    size_y = max_y - min_y + 1
    size_x = max_x - min_x + 1

    size = max(size_x, size_y, size_z)
    margin = int(size * margin_factor / 2)

    start_z = max(min_z - margin, 0)
    start_y = max(min_y - margin, 0)
    start_x = max(min_x - margin, 0)

    end_z   = min(max_z + margin + 1, shape[0])
    end_y   = min(max_y + margin + 1, shape[1])
    end_x   = min(max_x + margin + 1, shape[2])

    return [[start_z, start_y, start_x], [end_z, end_y, end_x]]

def _crop_volumes(b_box: list, image: np.ndarray, segPred: np.ndarray,
    labels: np.ndarray, label: np.int32):
    """
    Cropping the data

    Parameters
    ----------
    b_box : list
        Bounding box
    image : np.ndarray
        3D array with image data
    segPred : np.ndarray
        3D array with the predicted segmentation data
    labels : np,ndarray
        3D array with segmentation labels
    label : np.int32
        Singel label

    Returns
    -------
    dict
        Dictionary with the keys:
        - image : np.ndarray
        - segPred : np.ndarray
        - labels : np.ndarray
    """

    # (06.03.2026)
    (min_z, min_y, min_x), (max_z, max_y, max_x) = b_box

    cropped_image   = image[  min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_segPred = segPred[min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_labels  = labels[ min_z:max_z, min_y:max_y, min_x:max_x]

    # Keep only inside the box
    masked_labels = np.where(cropped_labels == label, label, 0)

    return {
        "image": cropped_image,
        "segPred": cropped_segPred,
        "labels": masked_labels
    }

def _display_cropped(viewer: napari.viewer.Viewer, stem1: str, stem2: str,
    segment_name: str, cropped: dict):
    """
    Plotting in Napari

    Parameters
    ----------
    viewer : napari.viewer.Viewer
    stem1 : str
        Name of the input data file
    stem2 : str
        Name of the '_segPred' data file
    segment_name : str
        Name of the segment
    croped : dict
        Dictionary with the keys:
        - image : np.ndarray
        - segPred : np.ndarray
        - labels : np.ndarray

    Returns
    -------
    layer : napari.layers.Layer
    """

    # (06.03.2026)
    name1 = 'Cropped ' + stem1
    name2 = 'Cropped ' + stem2

    viewer.add_image( cropped["image"],   name=name1)
    viewer.add_labels(cropped["segPred"], name=name2)

    layer = viewer.add_labels(cropped["labels"], name=segment_name)

    return layer

def _focus_viewer(viewer: napari.viewer.Viewer, labels: np.ndarray, label: int,
    layer: napari.layers.Layer):
    """
    Focus the camera

    Parameters
    ----------
    viewer : napari.viewer.Viewer
    labels : np.ndarray
        3D array with segmentation labels
    label : int
        selected label
    layer : napari.layers.Layer
    """

    # (06.03.2026)
    center_of_mass = ndimage.center_of_mass(labels)
    center_of_mass = tuple(int(round(c)) for c in center_of_mass)

    viewer.dims.current_step = center_of_mass
    viewer.camera.center = center_of_mass

    # Change to the matching color
    layer.selected_label = label

def _save_npy(array: np.ndarray, filename: Path):
    """
    Save the array in .npy format
    
    Parameters
    ----------
    array : np.ndarray
        Data array
    filename : Path
        Name of the .npy file
    """

    # (13.03.2026)
    print('Save file', filename)
    with filename.open("wb") as f:
        np.save(f, array)

def _load_npy(filename: Path):
    """
    Load an data array from a .npy file

    Parameters
    ----------
    filename : Path
        Name of the .npy file

    Returns
    -------
    np.ndarray
    """

    # (13.03.2026)
    print('Read file', filename)
    with filename.open("rb") as f:
        return np.load(f)

def _build_filename(stem: str, suffix: str):
    """
    Generate a filename

    Parameters
    ----------
    stem : str
        Name of the temporary file
    suffix : str
        File name extension

    Returns
    -------
        Name of the temporary file
    """

    # (24.04.2026)
    temp = Path(tempfile.gettempdir())
    return temp.joinpath(stem).with_suffix(suffix)


@dataclass
class Segment:
    """
    Metadata for a segmented vessel region.

    Attributes
    ----------
    name : str
        Display name of the segment.
    label : int
        Unique label value in the label volume.
    uncertainty : float
        Uncertainty assigned to the segment.
    count : int
        Number of voxels in the segment.
    coords : list | None
        Bounding box coordinates of the cropped region.
    done : bool
        True if the segment has already been processed.
    """

    # (29.04.2026)
    name: str
    label: str
    uncertainty: float
    count: int
    coords: Optional[list] = None
    done: bool = False


class ExampleQWidget(QWidget):
    """
    Class for displaying and correcting a 3D image of blood vessels

    Parameters
    ----------
    napari_viewer : napari.viewer.Viewer

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        Napari viewer
    segments : List
        List of Segments
    save_uncertainty : bool
        Save the file 'Uncertainty.tif'?
    dock_widget : MultipleViewerWidget
        MultipleViewerWidget from multiple_viewer_widget.py
    cross : QCheckBox
        Widget for displaying a crosshair
    parent : str
        Directory of the data files
    stem1 : str
        Name of the input data file
    stem2 : str
        Name of the '_segPred' data file
    stem3 : str
        Name of the '_uncertainty' data file
    image : np.ndarray
        3D array with image data
    segPred : np.ndarray
        3D array with the predicted segmentation data
    uncertainty : np.ndarray
        3D array with uncertainty data
    labels : np.ndarray
        3D array with segmentation labels
    popup_window : QWidget
        Pop up window with uncertainty values
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        """
        Class constructor

        Parameter
        ---------
        napari_viewer : napari.viewer.Viewer
        """

        # (03.05.2024)
        super().__init__()
        self.viewer = napari_viewer
        self.segments = []
        self.save_uncertainty = False

        # Define the layout of the main widget
        layout = QVBoxLayout(self)

        # Define some labels and buttons
        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)
        layout.addWidget(label1)

        btnLoadImage = QPushButton('Load image')
        btnLoadImage.clicked.connect(self.load_image)
        layout.addWidget(btnLoadImage)

        btnSegPred = QPushButton('Read segPred file')
        btnSegPred.clicked.connect(self.read_segPred)
        layout.addWidget(btnSegPred)

        btnShowUncert = QPushButton('Show uncertainty data')
        btnShowUncert.clicked.connect(self.show_uncertainty)
        layout.addWidget(btnShowUncert)

        # Test output
        btnInfo = QPushButton('Info')
        btnInfo.clicked.connect(self.show_info)
        layout.addWidget(btnInfo)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)
        layout.addWidget(label2)

        label3 = QLabel('Curation')
        label3.setFont(font)
        layout.addWidget(label3)

        btnPopupWindow = QPushButton('Show list of segments')
        btnPopupWindow.clicked.connect(self.show_popup_window)
        layout.addWidget(btnPopupWindow)

        btnSaveIntermediate = QPushButton('Save intermediate data')
        btnSaveIntermediate.clicked.connect(self.save_intermediate_data)
        layout.addWidget(btnSaveIntermediate)

        btnLoadIntermediate = QPushButton('Load intermediate data')
        btnLoadIntermediate.clicked.connect(self.load_intermediate_data)
        layout.addWidget(btnLoadIntermediate)

        label4 = QLabel('_______________')
        label4.setAlignment(Qt.AlignHCenter)
        layout.addWidget(label4)

        btnSaveResult = QPushButton('Save final result')
        btnSaveResult.clicked.connect(self.save_final_result)
        layout.addWidget(btnSaveResult)

        cbxSaveUncertainty = QCheckBox('Save uncertainty')
        cbxSaveUncertainty.stateChanged.connect(self.checkbox_save_uncertainty)
        layout.addWidget(cbxSaveUncertainty)

        # Insert the Napari “Multiple Viewer Widget”
        self.dock_widget = MultipleViewerWidget(self.viewer, parent=self)
        self.viewer.window.add_dock_widget(self.dock_widget, name='Sample')

        # Add the cross widget (on the left in the viewer area)
        self.cross = CrossWidget(self.viewer, parent=self)
        self.viewer.window.add_dock_widget(self.cross, name='Cross', area='left')

        self.setLayout(layout)

    def load_image(self):
        """Read the image file and store it in an image layer"""

        # (23.05.2024);
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
        self.parent = filename.parent           # The image directory
        self.stem1  = filename.stem             # Name of the image file
        suffix      = filename.suffix.lower()   # File extension
        # Truncate the extension .nii
        if suffix == '.gz' and self.stem1[-4:] == '.nii':
            self.stem1 = self.stem1[:-4]

        # Load the image file
        print('Load file', filename)
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
        Read the _segPred and _uncertainty data and save it in a label and an
        image layer.
        """

        # (23.05.2024, revised on 05.02.2025)
        # Search for the segPred file
        base = self.stem1.removesuffix('_IM')
        self.stem2 = base + '_segPred'
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
        print('Load file', filename)
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
        self.stem3 = base + '_uncertainty'
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
        print('Load file', filename)
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

    def show_uncertainty(self):
        """Show an image layer with the uncertainty data"""

        # (12.08.2025)
        if hasattr(self, 'uncertainty'):
            self.viewer.add_image(self.uncertainty, name='uncertainty',
                colormap='inferno')
        else:
            QMessageBox.information(self, 'Note', 'Uncertainty is not defined')

    def find_segments(self, uncertainty: np.ndarray):
        """
        Define segments that correspond to values of equal uncertainty

        Parameters
        ----------
        uncertainty : np.ndarray
            3D array with uncertainty data
        """

        # (09.08.2024, revised on 03.07.2025; 18.02.2026)
        t0 = time.time()                # UNIX timestamp
        print('The segmentation will take some time.')

        # 1st: Segmentation
        results = _segment_uncertainties(uncertainty)

        # 2nd: Merge labels
        self.labels, uncert_values = _merge_labels(results, uncertainty.shape)

        # 3rd: filter small segments
        self.labels, uncert_values = _merge_small_segments(self.labels,
            uncert_values, min_size=10)

        #4th: Create segment dictionaries
        self.segments = _create_segments(self.labels, uncert_values)

        print('Done in', time.time() - t0, 's')

        # 4th: Display the segments in an label layer
        self.viewer.add_labels(self.labels, name='Segmentation')

    def show_popup_window(self):
        """Define a pop-up window for the uncertainty list"""

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
        grid_layout.addWidget(QLabel('Count'), 0, 2)
        grid_layout.addWidget(QLabel('done'), 0, 3)

        # Define buttons and select values for some labels
        for idx, segment in enumerate(self.segments, start=1):
            # Show only the untreated areas
            if segment.done:
                continue
            else:
                self.new_entry(segment, grid_layout, idx)

        # show a horizontal line
        idx += 1
        line = QWidget()
        line.setFixedHeight(3)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet('background-color: mediumblue')
        grid_layout.addWidget(line, idx, 0, 1, -1)

        # The treated areas are shown in the lower part of the group box
        idx += 1
        grid_layout.addWidget(QLabel('Segment'), idx, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), idx, 1)
        grid_layout.addWidget(QLabel('Count'), idx, 2)
        grid_layout.addWidget(QLabel('Re-enable'), idx, 3)

        for idx, segment in enumerate(self.segments, start=idx+1):
            # show only the treated areas
            if segment.done:
                self.new_entry(segment, grid_layout, idx)
            else:
                continue

        # Show the pop-up window
        self.popup_window.show()
        
    def new_entry(self, segment: Segment, grid_layout: QGridLayout, idx: int):
        """
        New entry for 'Area n' in the grid layout

        Parameters
        ----------
        segment : Segment
            Metadata of the selected segment
        grid_layout : QGridLayout
            Layout for a QGroupBox
        idx : int
            Index in the grid_layout
        """

        # (13.08.2024)
        # Define some buttons and labels
        button1 = QPushButton(segment.name)
        button1.clicked.connect(lambda: self.zoom_in(segment, 0.75))

        if segment.done:
            # disable button1 for treated areas
            button1.setEnabled(False)
        grid_layout.addWidget(button1, idx, 0)

        uncertainty = '%.3f' % (segment.uncertainty)
        label1 = QLabel(uncertainty)
        grid_layout.addWidget(label1, idx, 1)

        count = '%d' % (segment.count)
        label2 = QLabel(count)
        grid_layout.addWidget(label2, idx, 2)

        if segment.done:
            button3 = QPushButton('re-enable')
            button3.clicked.connect(lambda: self.re_enable(segment))
        else:
            button3 = QPushButton('done')
            button3.clicked.connect(lambda: self.done(segment))
        grid_layout.addWidget(button3, idx, 3)

    def zoom_in(self, segment: Segment, margin_factor: float):
        """
        Show a segment and its surroundings in a 3D view

        Parameters
        ----------
        segment : Segment
            Metadata of the selected segment
        margin_factor : float
            Factor for enlarging the b_box
        """

        # (25.06.2025, revised 06.03.2026)
        self.viewer.layers.clear()      # Delete all layers in Napari

        # Determine the segment to be displayed
        label = segment.label           # target label
        mask  = self.labels == label    # Segment mask

        # 1st: Calculate bounding box
        b_box = _compute_bbox(mask)
        b_box = _expand_bbox(b_box, self.image.shape, margin_factor)

        # Save the coordinates of the cropped image
        segment.coords = b_box

        cropped = _crop_volumes(b_box, self.image, self.segPred, self.labels,
            label)

        layer = _display_cropped(self.viewer, self.stem1, self.stem2,
            segment.name, cropped)

        _focus_viewer(self.viewer, cropped["labels"], label, layer)

    def done(self, segment: Segment):
        """
        Transfer data from the segment to the labels, segPred and uncertainty
        layer and close the layers for the cropped images.

        Parameters
        ----------
        segment : Segment
            Metadata of the selected segment
        """

        # (18.07.2024)
        self.compare_and_transfer(segment)  # transfer of data
        segment.done = True                 # mark this area as treated

        # Close cropped images and show image, segPred und labels
        self.viewer.layers.clear()
        self.viewer.add_image(self.image, name=self.stem1)
        self.viewer.add_labels(self.segPred, name=self.stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')

        # open a new pop-up window
        self.show_popup_window()

    def re_enable(self, segment: Segment):
        """
        Re-enable the data of a specific area in the pop-up window.

        Parameters
        ----------
        segment : Segment
            Metadata of the selected segment
        """

        # (19.07.2024)
        segment.done = False
        self.show_popup_window()

    def compare_and_transfer(self, segment: Segment):
        """
        Compare old and new data and transfer the changes to the segPred,
        uncertainty and labels data.

        Parameters
        ----------
        segment : Segment
            Metadata of the selected segment
        """

        # (09.08.2024)
        name = segment.name
        label = segment.label
        uncertainty = segment.uncertainty
        coords = segment.coords

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):

            # Data of the segment
            layer = self.viewer.layers[name]
            segment_data = layer.data

            # Original coordinates of the segment
            start_z, start_y, start_x = coords[0]
            end_z,   end_y,   end_x   = coords[1]

            # Create an empty image and insert the segment data
            new_data = np.zeros_like(self.labels, dtype=int)
            new_data[start_z:end_z, start_y:end_y, start_x:end_x] = segment_data

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
        Save the segPred, uncertainty and labels data to files on hard drive
        """

        # (26.07.2024, revised 24.04.2026)
        base = self.stem1.removesuffix('_IM')

        try:
            stem = base + '_segPred'
            _save_npy(self.segPred, _build_filename(stem, '.npy'))

            stem = base + '_uncertainty'
            _save_npy(self.uncertainty, _build_filename(stem, '.npy'))

            stem = base + '_labels'
            _save_npy(self.labels, _build_filename(stem, '.npy'))

            stem = base + '_segments'
            filename = _build_filename(stem, '.json')

            print('Save file', filename)
            with filename.open('w', encoding='utf-8') as f:
                json.dump([asdict(seg) for seg in self.segments], f, indent=2)

        except OSError as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))

    def load_intermediate_data(self):
        """Read the segPred and uncertainty data from files on hard drive"""

        # (30.07.2024, revised 28.04.2026)
        base = self.stem1.removesuffix('_IM')

        try:
            stem2 = base + '_segPred'
            self.segPred = _load_npy(_build_filename(stem2, '.npy'))

            stem = base + '_uncertainty'
            self.uncertainty = _load_npy(_build_filename(stem, '.npy'))

            stem = base + '_labels'
            self.labels = _load_npy(_build_filename(stem, '.npy'))

            stem = base + '_segments'
            filename = _build_filename(stem, '.json')

            print('Read file', filename)
            with filename.open('r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct Segment objects by unpacking the dictionary
            # entries as keyword arguments into the Segment constructor.
            self.segments = [Segment(**seg) for seg in data]

        except OSError as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Close cropped images and show image, segPred und labels
        self.viewer.layers.clear()
        self.viewer.add_image(self.image, name=self.stem1)
        self.viewer.add_labels(self.segPred, name=stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')

    def save_final_result(self):
        """
        Close all open segment layers, save the segPred and if applicable also
        the uncertainty data to files on hard drive.
        """

        # (13.08.2024)
        # 1st: close the open segment layer
        segment_layers = [layer for layer in self.viewer.layers
            if layer.name.startswith('Segment_') and
            isinstance(layer, napari.layers.Labels)]

        for layer in segment_layers:
            name = layer.name
            print('Close', name)

            # The following expression contains a generator:
            segment = next((s for s in self.segments if s['name'] == name), None)
            if segment is not None:
                self.compare_and_transfer(segment)
                segment.done = True

        # 2nd: build a filename for the segPredNew data
        base = self.stem1.removesuffix('_IM')
        stem = base + '_segPred_New'
        filename = self.parent.joinpath(stem).with_suffix('.tif')
        default_filename = str(filename)
        filename, _ = QFileDialog.getSaveFileName(self, 'Save _segPred_New file',
             default_filename, 'TIFF files (*.tif *.tiff)')
        if filename == '':                      # Cancel button has been pressed
            QMessageBox.information(self, 'Cancel button',
                'The cancel button has been pressed.')
            return

        # 3rd: Save the segPredNew data
        print('Save file', filename)
        try:
            imwrite(filename, self.segPred)
        except OSError as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # 4th: Save the uncertaintyNew data
        if self.save_uncertainty:
            filename = filename[:-16] + '_uncertainty_New.tif'
            print('Save file', filename)
            try:
                imwrite(filename, self.uncertainty)
            except OSError as error:
                QMessageBox.warning(self, 'I/O Error:', str(error))

        # Close cropped images and show image, segPred und labels
        self.viewer.layers.clear()
        self.viewer.add_image(self.image, name=self.stem1)
        self.viewer.add_labels(self.segPred, name=self.stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')

    def checkbox_save_uncertainty(self, state: int):
        """
        Toggle the bool variable save_uncertainty

        Parameters
        ----------
        state : int
            Qt.Checked or Qt.Unchecked
        """

        if state == Qt.Checked:
            self.save_uncertainty = True
        else:
            self.save_uncertainty = False

    def show_info(self):
        """Show information about the current layer"""

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
            print('---')
            print('min:', np.min(image))
            print('median:', np.median(image))
            print('max:', np.max(image))
            print('mean: %.3f' % (np.mean(image)))
            print('std: %.3f' %  (np.std(image)))

        elif isinstance(layer, napari.layers.Labels):
            data = layer.data
            values, counts = np.unique(data, return_counts=True)

            print('type:', type(data))
            print('dtype:', data.dtype)
            print('size:',  data.size)
            print('ndim:',  data.ndim)
            print('shape:', data.shape)
            print('values:', values)
            print('counts:', counts)
        else:
            print('This is not an image or label layer!')
        print()
