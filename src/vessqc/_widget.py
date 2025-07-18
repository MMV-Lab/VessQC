"""
Module for the definition of the class ExampleQWidget

Imports
-------
napari, numpy, pathlib.Path, qtpy.QtCore.QSize, qtpy.QtCore.QT, qtpy.QtWidgets,
scipy.ndimage, SimpleITK, tifffile.imread, tifffile.imwrite

Exports
-------
ExampleQWidget
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

import numpy as np
import napari
import SimpleITK as sitk
import time
from joblib import Parallel, delayed
from pathlib import Path
from qtpy.QtCore import (
    QSize,
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (
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
from threading import Thread
from tifffile import imread, imwrite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

def _label_value_sparse(uncertainty, value, tolerance, structure, value_idx,
    num_unique_values):
    # Worker side
    # (03.07.2025)

    mask = np.abs(uncertainty - value) < tolerance
    if not np.any(mask):
        return None

    labeled, num = ndimage.label(mask, structure)   # Segmentation
    if num == 0:
        return None

    # Calculate global unique labels directly
    # local labels: 1, 2, 3, ...
    # → global: (local - 1) * num_unique_values + (value_idx + 1)
    labeled_global = (labeled - 1) * num_unique_values + (value_idx + 1)
    labeled_global[labeled == 0] = 0

    indices = np.where(mask)
    res = dict(
        indices =       indices,
        global_labels = labeled_global[indices],
        value =         value,
        num =           num
    )
    return res

class ExampleQWidget(QWidget):
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
    seg_pred : numpy.ndarray
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
    read_seg_pred()
        Read the segPred and uncertanty data and save it in a label and an
        image layer
    build_areas(uncertainty: np.ndarray)
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
    restore()
        Restore the data of a specific area in the pop-up window
    compare_and_transfer(name: str)
        Compare old and new data of an area and transfer the changes to the
        segPred and uncertainty data
    btn_save()
        Save the segPred and uncertainty data to files on drive
    reload()
        Read the segPred and uncertainty data from files on drive
    final_segmentation()
        Close all open area layers, close the pop-up window, save the
        segPred and if applicable also the uncertainty data to files on
        drive
    cbx_save_uncertainty(state: Qt.Checked)
        Toggle the bool variable save_uncertainty
    btn_info()
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
        self.save_uncertainty = False

        # Define the layout of the main widget
        self.setLayout(QVBoxLayout())

        # Define some labels and buttons
        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)
        self.layout().addWidget(label1)

        btnLoad = QPushButton('Load image')
        btnLoad.clicked.connect(self.load_image)
        self.layout().addWidget(btnLoad)

        btnSegmentation = QPushButton('Read segmentation')
        btnSegmentation.clicked.connect(self.read_seg_pred)
        self.layout().addWidget(btnSegmentation)

        # Test output
        btnInfo = QPushButton('Info')
        btnInfo.clicked.connect(self.btn_info)
        self.layout().addWidget(btnInfo)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)
        self.layout().addWidget(label2)

        label3 = QLabel('Curation')
        label3.setFont(font)
        self.layout().addWidget(label3)

        btnUncertainty = QPushButton('Load segment list')
        btnUncertainty.clicked.connect(self.show_popup_window)
        self.layout().addWidget(btnUncertainty)

        btnSave = QPushButton('Save intermediate curation')
        btnSave.clicked.connect(self.btn_save)
        self.layout().addWidget(btnSave)

        btnReload = QPushButton('Load saved curation')
        btnReload.clicked.connect(self.reload)
        self.layout().addWidget(btnReload)

        label4 = QLabel('_______________')
        label4.setAlignment(Qt.AlignHCenter)
        self.layout().addWidget(label4)

        btnFinalSegmentation = QPushButton('Save results')
        btnFinalSegmentation.clicked.connect(self.final_segmentation)
        self.layout().addWidget(btnFinalSegmentation)

        cbxSaveUncertainty = QCheckBox('Save uncertainty')
        cbxSaveUncertainty.stateChanged.connect(self.checkbox_save_uncertainty)
        self.layout().addWidget(cbxSaveUncertainty)

    def load_image(self):
        """
        Read the image file and save it in an image layer
        """

        # (23.05.2024);
        self.areas = []

        # Find and load the image file
        filter1 = "TIFF files (*.tif *.tiff);;NIfTI files (*.nii *.nii.gz);;\
            All files (*.*)"
        filename, _ = QFileDialog.getOpenFileName(self, 'Image file', '',
            filter1)

        if filename == '':                      # Cancel has been pressed
            QMessageBox.information(self, 'Cancel', 'Cancel has been pressed.')
            return
        else:
            path = Path(filename)
            self.parent = path.parent           # The data directory
            self.stem1  = path.stem             # Name of the input file
            suffix      = path.suffix.lower()   # File extension
            # Truncate the extension .nii
            if suffix == '.gz' and self.stem1[-4:] == '.nii':
                self.stem1 = self.stem1[:-4]

        # Load the image file
        print('Load', path)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.image = imread(path)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(path)
                self.image = sitk.GetArrayFromImage(sitk_image)
            else:
                QMessageBox.information(self, 'Unknown file type',
                    'Unknown file type: %s%s!' % (self.stem1, suffix))
                return
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        self.viewer.add_image(self.image, name=self.stem1)   # Show the image

    def read_seg_pred(self):
        """
        Read the segPred and uncertanty data and save it in a label and an
        image layer
        """

        # (23.05.2024, revised on 05.02.2025)
        # Search for the segPred file
        self.stem2 = self.stem1[:-3] + '_segPred'   # Replace _IM by _segPred
        path = self.parent / self.stem2

        if path.with_suffix('.tif').is_file():
            path = path.with_suffix('.tif')
            suffix = '.tif'
        elif path.with_suffix('.tiff').is_file():
            path = path.with_suffix('.tiff')
            suffix = '.tiff'
        elif path.with_suffix('.nii').is_file():
            path = path.with_suffix('.nii')
            suffix = '.nii'
        elif path.with_suffix('.nii.gz').is_file():
            path = path.with_suffix('.nii.gz')
            suffix = '.nii.gz'
        else:
            QMessageBox.information(self, 'File not found',
                'No segPred file %s found!' % (path))
            return

        # Read the segPred file
        print('Load', path)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.seg_pred = imread(path)
            elif suffix == '.nii' or suffix == '.nii.gz':
                sitk_image = sitk.ReadImage(path)
                self.seg_pred = sitk.GetArrayFromImage(sitk_image)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Save the segPred data in a label layer
        self.viewer.add_labels(self.seg_pred, name=self.stem2)

        # Search for the uncertainty file
        stem3 = self.stem1[:-3] + '_uncertainty'
        path = self.parent / stem3

        if path.with_suffix('.tif').is_file():
            path = path.with_suffix('.tif')
            suffix = '.tif'
        elif path.with_suffix('.tiff').is_file():
            path = path.with_suffix('.tiff')
            suffix = '.tiff'
        elif path.with_suffix('.nii').is_file():
            path = path.with_suffix('.nii')
            suffix = '.nii'
        elif path.with_suffix('.nii.gz').is_file():
            path = path.with_suffix('.nii.gz')
            suffix = '.gz'
        else:
            QMessageBox.information(self, 'File not found',
                'No uncertainty file %s found!' % (path))
            return

        # Read the uncertainty file
        print('Load', path)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.uncertainty = imread(path)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(path)
                self.uncertainty = sitk.GetArrayFromImage(sitk_image)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        if self.areas == []:
            self.build_areas(self.uncertainty)      # define areas

    """
    def remap_labels_and_dict(self, label_map, uncert_values):
        # Remap self.labels and the uncertainty-value dictionary to new
        # sequential labels
        # (08.07.2025)

        # Remap self.labels array in-place
        # The use of the auxiliary array new_labels is necessary because the
        # value ranges between new and old labels can overlap.
        new_labels = np.zeros_like(self.labels)
        for old, new in label_map.items():
            new_labels[self.labels == old] = new
        self.labels = new_labels

        # Remap uncertainty value dictionary
        remapped_dict = {
            label_map[old]: value
            for old, value in uncert_values.items()
            if old in label_map
        }
        return remapped_dict
    """

    def build_areas(self, uncertainty: np.ndarray):
        """ Define segments that correspond to values of equal uncertainty """

        # (09.08.2024, revised on 03.07.2025)
        t0 = time.time()                # UNIX timestamp
        print('The segmentation will take some time.')

        unique_values = np.unique(uncertainty)
        unique_values = unique_values[unique_values > 0]
        num_unique_values = len(unique_values)
        tolerance = 1e-2
        structure = np.ones((3, 3, 3), dtype=int)   # Connectivity

        results = Parallel(n_jobs=-1)(
            delayed(_label_value_sparse)(
                uncertainty, value, tolerance, structure, idx, num_unique_values
            )
            for idx, value in enumerate(unique_values)
        )

        self.labels = np.zeros_like(uncertainty, dtype=int)
        uncert_values = {0: 0.0}    # Dictionary of all uncertanty values

        for i, res in enumerate(results):
            if res is None:
                continue
            indices = res['indices']
            labels  = res['global_labels']
            value   = res['value']
            num     = res['num']

            self.labels[indices] = labels
     
            # Form a dictionary with the uncertainty values that correspond to
            # the respective labels
            keys   = list(np.unique(labels))
            values = [value] * num
            uval   = dict(zip(keys, values))
            uncert_values = {**uncert_values, **uval}

        print('Done in', time.time() - t0, 's')

        """
        # Remap of labels to 1...N
        old_labels = np.unique(self.labels)
        old_labels = old_labels[old_labels > 0]
        label_map = {old: new for new, old in enumerate(old_labels, start=1)}

        uncert_values = self.remap_labels_and_dict(label_map, uncert_values)

        t3 = time.time()
        print('Remapping in', t3 - t2, 's')
        """

        # Determine all labels that appear less than 10 times
        min_size = 10
        counts = np.bincount(self.labels.ravel())
        small_labels = np.where(counts < min_size)[0]
        small_labels = small_labels[small_labels != 0]

        # Replaces all labels that occur less than 10 times with the value
        # max(labels) + 1
        max_label = np.max(self.labels) + 1
        mask = np.isin(self.labels, small_labels)
        self.labels[mask] = max_label

        # Create a structure for storing the data
        all_labels = np.unique(self.labels)
        all_labels = all_labels[all_labels != 0]
        counts = np.bincount(self.labels.ravel())
        uncert_values[max_label] = 0.9999

        self.areas = list()
        for label in all_labels:
            segment = dict(
                name        = '',
                label       = label,
                uncertainty = uncert_values[label],
                counts      = counts[label],
                coords      = None,      # coordinates
                com         = None,      # center of mass
                site        = None,
                done        = False,
            )
            self.areas.append(segment)

        # Sort by 'uncertainty' ascending
        self.areas.sort(key=lambda x: x['uncertainty'])

        # Number the names
        for i, segment in enumerate(self.areas, start=1):
            segment['name'] = f"Segment {i}"

        # Display the segments in an label layer
        self.viewer.add_labels(self.labels, name='Segmentation')

    def show_popup_window(self):
        """ Define a pop-up window for the uncertainty list """

        # (24.05.2024)
        self.popup_window = QWidget()
        self.popup_window.setWindowTitle('napari')
        self.popup_window.setMinimumSize(QSize(350, 300))
        vbox_layout = QVBoxLayout()
        self.popup_window.setLayout(vbox_layout)

        # define a scroll area inside the pop-up window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        vbox_layout.addWidget(scroll_area)

        # Define a group box inside the scroll area
        group_box = QGroupBox('Uncertainty list')
        grid_layout = QGridLayout()
        group_box.setLayout(grid_layout)
        scroll_area.setWidget(group_box)

        # add widgets to the group box
        grid_layout.addWidget(QLabel('Segment'), 0, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), 0, 1)
        grid_layout.addWidget(QLabel('Counts'), 0, 2)
        # grid_layout.addWidget(QLabel('Zoom in'), 0, 3)
        grid_layout.addWidget(QLabel('done'), 0, 3)

        # Define buttons and select values for some labels
        i = 1
        for segment in self.areas:
            # Show only the untreated areas
            if segment['done']:
                continue
            else:
                self.new_entry(segment, grid_layout, i)
                i += 1

        # show a horizontal line
        line = QWidget()
        line.setFixedHeight(3)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet('background-color: mediumblue')
        grid_layout.addWidget(line, i, 0, 1, -1)
        i += 1

        # The treated areas are shown in the lower part of the group box
        grid_layout.addWidget(QLabel('Segment'), i, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), i, 1)
        grid_layout.addWidget(QLabel('Counts'), i, 2)
        # grid_layout.addWidget(QLabel('Zoom in'), i, 3)
        grid_layout.addWidget(QLabel('restore'), i, 3)
        i += 1

        for segment in self.areas:
            # show only the treated areas
            if segment['done']:
                self.new_entry(segment, grid_layout, i)
                i += 1
            else:
                continue

        # Show the pop-up window
        self.popup_window.show()
        
    def new_entry(self, segment: dict, grid_layout: QGridLayout, i: int):
        """
        New entry for 'Area n' in the grid layout

        Parameters
        ----------
        segment : dict
            'name', 'uncertainty', 'counts', 'com', 'site' and 'done'
            for a specific area
        grid_layout : QGridLayout
            Layout for a QGroupBox
        i : int
            Index in the grid_layout
        """

        # (13.08.2024)
        # Define some buttons and labels
        button1 = QPushButton(segment['name'])
        button1.clicked.connect(lambda: self.zoom_in(segment, 0.75))

        if segment['done']:
            # disable button1 for treated areas
            button1.setEnabled(False)
        grid_layout.addWidget(button1, i, 0)

        uncertainty = '%.4f' % (segment['uncertainty'])
        label1 = QLabel(uncertainty)
        grid_layout.addWidget(label1, i, 1)

        counts = '%d' % (segment['counts'])
        label2 = QLabel(counts)
        grid_layout.addWidget(label2, i, 2)

        """
        button2 = QPushButton('zoom in')
        button2.clicked.connect(lambda: self.zoom_in(segment, 0.75))

        if segment['done']:
            # disable button2 for treated areas
            button2.setEnabled(False)
        grid_layout.addWidget(button2, i, 3)
        """

        if segment['done']:
            button3 = QPushButton('restore')
            button3.clicked.connect(lambda: self.restore(segment))
        else:
            button3 = QPushButton('done')
            button3.clicked.connect(lambda: self.done(segment))
        grid_layout.addWidget(button3, i, 3)

    """
    def show_area(self, segment: dict):
        # Show the data for a specific segment in a new label layer

        # (29.05.2024)
        name  = segment['name']         # segment name: "Segment nn"
        label = segment['label']        # segment label

        # Check whether the layer 'Segment_nn' already exists
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):
            layer = self.viewer.layers[name]
            data = layer.data

            # Place the affected label layer at the top of the stack
            source_index = self.viewer.layers.index(layer)
            target_index = len(self.viewer.layers)
            self.viewer.layers.move(source_index, target_index)
            layer.visible = True
        else:
            # Show the data for a specific segment;
            site = np.where(self.labels == label)
            segment['site'] = site      # save the site for later use
            data = np.zeros_like(self.labels)
            data[site] = label + 1      # build a new label layer
            layer = self.viewer.add_labels(data, name=name)

        # Find the center of the data points
        if segment['com'] == None:
            com = ndimage.center_of_mass(data)
            com =  tuple(int(round(c)) for c in com)
            segment['com'] = com
            print('center of mass:', segment['com'])

        # Set the appropriate level and focus
        self.viewer.dims.current_step = com
        self.viewer.camera.center = com

        # Change to the matching color
        layer.selected_label = label + 1
    """

    def zoom_in(self, segment: dict, margin_factor: float):
        """
        Show a segment and its immediate surroundings in a 3D view.
        """

        # (25.06.2025)
        self.viewer.layers.clear()          # Delete all layers in Napari

        # Determine the segment to be displayed
        label = segment['label']            # target label
        mask  = (self.labels == label)      # Segment mask
        segment['site'] = np.where(mask)    # save the site for later use

        # Calculate bounding box
        coords = np.argwhere(mask)
        minz, miny, minx = coords.min(axis=0)
        maxz, maxy, maxx = coords.max(axis=0)

        # Enlage box
        sz, sy, sx = maxz - minz + 1, maxy - miny + 1, maxx - minx + 1
        mz = int(sz * margin_factor / 2)
        my = int(sy * margin_factor / 2)
        mx = int(sx * margin_factor / 2)

        # Limitation to the image
        shape = self.image.shape
        startz = max(minz - mz, 0)
        starty = max(miny - my, 0)
        startx = max(minx - mx, 0)
        endz   = min(maxz + mz + 1, shape[0])
        endy   = min(maxy + my + 1, shape[1])
        endx   = min(maxx + mx + 1, shape[2])

        # Save the coordinates of the image section
        segment['coords'] = [(startz, starty, startx), (endz, endy, endx)]

        # Cropping
        cropped_image = self.image[startz:endz, starty:endy, startx:endx]
        cropped_seg_pred = self.seg_pred[startz:endz, starty:endy,
            startx:endx]
        cropped_labels = self.labels[startz:endz, starty:endy, startx:endx]

        # Keep only inside the box
        masked_labels = np.where(cropped_labels == label, label, 0)

        # Display data in Napari
        name1 = 'Cropped ' + self.stem1
        name2 = 'Cropped ' + self.stem2
        name3 = segment['name']
        self.viewer.add_image(cropped_image, name=name1)
        self.viewer.add_labels(cropped_seg_pred, name=name2)
        layer = self.viewer.add_labels(masked_labels, name=name3)

        # Set the appropriate level and focus
        com = ndimage.center_of_mass(masked_labels)     # center of mass
        segment['com'] = com
        self.viewer.dims.current_step = com
        self.viewer.camera.center = com

        # Change to the matching color
        layer.selected_label = label

    def done(self, segment: dict):
        """
        Transfer data from the area to the labels, segPred and uncertainty
        layer and close the layer for the area
        """

        # (18.07.2024)
        self.compare_and_transfer(segment)  # transfer of data
        segment['done'] = True              # mark this area as treated

        # Show image, segPred und segments
        self.viewer.layers.clear()
        self.viewer.add_image(self.image, name=self.stem1)
        self.viewer.add_labels(self.seg_pred, name=self.stem2)
        self.viewer.add_labels(self.labels, name='Segmentation')

        # open a new pop-up window
        self.show_popup_window()

    def restore(self, segment: dict):
        """ Restore the data of a specific area in the pop-up window """

        # (19.07.2024)
        segment['done'] = False
        self.show_popup_window()

    def compare_and_transfer(self, segment: dict):
        """
        Compare old and new data and transfer the changes to the segPred
        and uncertainty data

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
        site        = segment['site']

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):

            # Data of the segment
            layer = self.viewer.layers[name]
            segment_data = layer.data

            # Original coordinates of the segment
            startz, starty, startx = coords[0]
            endz,   endy,   endx   = coords[1]

            # Create an empty image and insert the segment data
            new_data = np.zeros_like(self.labels, dtype=int)
            new_data[startz:endz, starty:endy, startx:endx] = segment_data

            # compare new and old data
            old_data = np.zeros_like(self.labels, dtype=int)
            old_data[site] = label
            delta = new_data - old_data

            add_data = np.where(delta > 0)       # new data points
            del_data = np.where(delta < 0)       # deleted data points

            # transfer the changes to the labels layer
            self.labels[add_data] = label
            self.labels[del_data] = 0

            # transfer the changes to the segPred layer
            self.seg_pred[add_data] = 1
            self.seg_pred[del_data] = 0

            # transfer the changes to the uncertainty layer
            self.uncertainty[add_data] = uncertainty
            self.uncertainty[del_data] = 0.0

    def btn_save(self):
        """ Save the segPred and uncertainty data to files on drive """

        # (26.07.2024)
        # 1st: save the segPred data
        filename = self.parent / '_segPred.npy'
        print('Save', filename)
        try:
            file = open(filename, 'wb')
            np.save(file, self.seg_pred)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
        finally:
            if 'file' in locals() and file:
                file.close()

        #2nd: save the uncertainty data
        filename = self.parent / '_Uncertainty.npy'
        print('Save', filename)
        try:
            file = open(filename, 'wb')
            np.save(file, self.uncertainty)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
        finally:
            if 'file' in locals() and file:
                file.close()

    def reload(self):
        """ Read the segPred and uncertainty data from files on drive """

        # (30.07.2024)
        # 1st: read the segPred data
        filename = self.parent / '_segPred.npy'
        print('Read', filename)
        try:
            file = open(filename, 'rb')
            self.seg_pred = np.load(file)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return
        finally:
            if 'file' in locals() and file:
                file.close()

        # If the 'segPred' layer already exists'
        if any(layer.name == 'segPred' and
            isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):
            layer = self.viewer.layers['segPred']
            layer.data = self.seg_pred
        else:
            self.viewer.add_labels(self.seg_pred, name='segPred')

        # 2st: read the uncertainty data
        filename = self.parent / '_Uncertainty.npy'
        print('Read', filename)
        try:
            file = open(filename, 'rb')
            self.uncertainty = np.load(file)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return
        finally:
            if 'file' in locals() and file:
                file.close()

        if self.areas == []:
            self.build_areas(self.uncertainty)      # define areas

    def final_segmentation(self):
        """
        Close all open area layers, close the pop-up window, save the
        segPred and if applicable also the uncertainty data to files on
        drive
        """

        # (13.08.2024)
        # 1st: close all open area layers
        lst = [layer for layer in self.viewer.layers
            if layer.name.startswith('Segment_') and
            isinstance(layer, napari.layers.Labels)]

        for layer in lst:
            name = layer.name
            print('Close', name)
            self.compare_and_transfer(name)
            self.viewer.layers.remove(layer)    # delete the layer 'Segment_n'

        if hasattr(self, 'popup_window'):       # close the pop-up window
            self.popup_window.close()

        # Build a filename for the segPred data
        filename = self.stem1[:-3] + '_segNew.tif'
        default_filename = str(self.parent / filename)
        filename, _ = QFileDialog.getSaveFileName(self, 'Segmentation file', \
            default_filename, 'TIFF files (*.tif *.tiff)')
        if filename == '':                      # Cancel has been pressed
            QMessageBox.information(self, 'Cancel', 'Cancel has been pressed.')
            return

        # Save the segPred data
        print('Save', filename)
        try:
            imwrite(filename, self.seg_pred)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Save the uncertainty data
        if self.save_uncertainty:
            filename = filename[:-11] + '_uncNew.tif'
            print('Save', filename)
            try:
                imwrite(filename, self.uncertainty)
            except BaseException as error:
                QMessageBox.warning(self, 'I/O Error:', str(error))

    def checkbox_save_uncertainty(self, state: Qt.Checked):
        """ Toggle the bool variable save_uncertainty """

        if state == Qt.Checked:
            self.save_uncertainty = True
        else:
            self.save_uncertainty = False

    def btn_info(self):
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
