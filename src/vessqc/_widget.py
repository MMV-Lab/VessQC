"""
Module for the definition of the class VessQC

Imports
-------
napari, numpy, pathlib.Path, qtpy.QtCore.QSize, qtpy.QtCore.QT, qtpy.QtWidgets,
scipy.ndimage, SimpleITK, tifffile.imread, tifffile.imwrite

Exports
-------
VessQC
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
import napari
import SimpleITK as sitk
import traceback
from tifffile import imread, imwrite
from pathlib import Path
from qtpy.QtCore import QSize, Qt
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

if TYPE_CHECKING:
    import napari


class VessQC(QWidget):
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
    segmentation : numpy.ndarray
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
    read_segmentation()
        Read the segmentation and uncertanty data and save it in a label and an
        image layer
    build_areas(image: np.ndarray)
        Define areas that correspond to values of equal uncertainty
    show_popup_window()
        Define a pop-up window for the uncertainty list
    new_entry(segment: dict, grid_layout: QGridLayout, i: int):
        New entry for 'Area n' in the grid layout
    show_area()
        Show the data for a specific uncertanty in a new label layer
    done()
        Transfer data from the area to the segmentation and uncertainty layer
        and close the layer for the area
    restore()
        Restore the data of a specific area in the pop-up window
    compare_and_transfer(name: str)
        Compare old and new data of an area and transfer the changes to the
        segmentation and uncertainty data
    btn_save()
        Save the segmentation and uncertainty data to files on drive
    reload()
        Read the segmentation and uncertainty data from files on drive
    final_segmentation()
        Close all open area layers, close the pop-up window, save the
        segmentation and if applicable also the uncertainty data to files on
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
        btnSegmentation.clicked.connect(self.read_segmentation)
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

        btnUncertainty = QPushButton('Load uncertainty list')
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

        btnFinalSegmentation = QPushButton('Generate final segmentation')
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
            self.parent = path.parent          # The data directory
            self.stem1 = path.stem             # Name of the input file
            suffix = path.suffix.lower()       # File extension
            # Truncate the .nii extension
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

    def read_segmentation(self):
        """
        Read the segmentation and uncertanty data and save it in a label and an
        image layer
        """

        # (23.05.2024, revised on 05.02.2025)
        # Search for the segmentation file
        stem2 = self.stem1[:-3] + '_segPred'        # Replace '_IM' by '_segPred'
        path = self.parent / stem2

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
                'No segmentation file %s found!' % (path))
            return

        # Read the segmentation file
        print('Load', path)
        try:
            if suffix == '.tif' or suffix == '.tiff':
                self.segmentation = imread(path)
            elif suffix == '.nii' or suffix == '.gz':
                sitk_image = sitk.ReadImage(path)
                self.segmentation = sitk.GetArrayFromImage(sitk_image)
        except BaseException as error:
            traceback.print_exc()               # For debug purposes
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Save the segmentation data in a label layer
        self.viewer.add_labels(self.segmentation, name='Segmentation')

        # Search for the uncertainty file
        stem2 = self.stem1[:-3] + '_uncertainty'
        path = self.parent / stem2

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
            traceback.print_exc()               # For debug purposes
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        print('Sorry, but the segmentation will take some time.')
        if self.areas == []:
            self.build_areas(self.uncertainty)      # define areas

    def build_areas(self, uncert: np.ndarray):
        """ Define segments that correspond to values of equal uncertainty """

        # (09.08.2024, revised on 20.05.2025)
        unique_values = np.unique(uncert)
        unique_values = unique_values[unique_values > 0]    # Values != 0

        tolerance = 1e-5            # Tolerance, if necessary
        uncert_values = [0.0]       # List of all uncertanty values
        self.labels = np.zeros_like(uncert, dtype=int)  # result array
        current_label = 0           # offset for labels
        structure = np.ones((3, 3, 3), dtype=int)       # Connectivity

        # For each value: Create mask and mark the segments with ndimage.label
        for value in unique_values:
            mask = np.abs(uncert - value) < tolerance
            mask.astype(int)
            labeled, num = ndimage.label(mask, structure)   # Segmentation

            # Move label values so that they are unique
            mask2 = labeled > 0
            labeled[mask2] += current_label
            self.labels[mask2] = labeled[mask2]
            current_label += num

            # Save the uncertainty value for each label
            uncert_values.extend([value] * num)

        # Summarize segments with less than 10 voxels
        min_size = 10
        counts = np.bincount(self.labels.ravel())

        # Ignore background (label == 0)
        small_labels = np.where(counts < min_size)[0]
        small_labels = small_labels[small_labels != 0]

        # Mask for all small segments
        mask_small = np.isin(self.labels, small_labels)
        new_max = np.max(self.labels) + 1
        self.labels[mask_small] = new_max

        # Create a structure for storing the data
        all_labels = np.unique(self.labels)
        all_labels = all_labels[all_labels != 0]
        counts = np.bincount(self.labels.ravel())
        uncert_values.append(np.median(unique_values))

        print(len(all_labels), 'segments')

        self.areas = []
        for label in all_labels:
            segment = {
                'name': 'Segment %d' % (label),
                'label': label,
                'uncertainty': uncert_values[label],
                'counts': counts[label],
                'com': None,                # center of mass
                'site': None,
                'done': False
            }
            self.areas.append(segment)

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
        grid_layout.addWidget(QLabel('Area'), 0, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), 0, 1)
        grid_layout.addWidget(QLabel('Counts'), 0, 2)
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
        grid_layout.addWidget(QLabel('Area'), i, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), i, 1)
        grid_layout.addWidget(QLabel('Counts'), i, 2)
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
        name = segment['name']
        button1 = QPushButton(name)
        button1.clicked.connect(self.show_area)

        if segment['done']:
            # disable button1 for treated areas
            button1.setEnabled(False)
        grid_layout.addWidget(button1, i, 0)

        uncertainty = '%.5f' % (segment['uncertainty'])
        label1 = QLabel(uncertainty)
        grid_layout.addWidget(label1, i, 1)

        counts = '%d' % (segment['counts'])
        label2 = QLabel(counts)
        grid_layout.addWidget(label2, i, 2)

        if segment['done']:
            button2 = QPushButton('restore', objectName=name)
            button2.clicked.connect(self.restore)
        else:
            button2 = QPushButton('done', objectName=name)
            button2.clicked.connect(self.done)
        grid_layout.addWidget(button2, i, 3)

    def show_area(self):
        """ Show the data for a specific segment in a new label layer """

        # (29.05.2024)
        name = self.sender().text()     # text of the button: Segment n
        hit = [d for d in self.areas if d.get('name') == name]  # d == dict.
        segment = hit[0]
        label = segment['label']        # segment label
        com = segment['com']            # center of mass

        # Check whether the layer 'Segment n' already exists
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):

            # Place the affected label layer at the top of the stack
            layer = self.viewer.layers[name]
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
            if com == None:
                com = ndimage.center_of_mass(data)
                com = tuple(int(round(c)) for c in com)
                segment['com'] = com
                print('center of mass:', com)

        # Set the appropriate level and focus
        self.viewer.dims.current_step = com
        self.viewer.camera.center = com

        # Change to the matching color
        layer.selected_label = label + 1

    def done(self):
        """
        Transfer data from the area to the segmentation and uncertainty layer and
        close the layer for the area
        """

        # (18.07.2024)
        name = self.sender().objectName()       # name of the object: 'Area n'
        self.compare_and_transfer(name)         # transfer of data
        layer = self.viewer.layers[name]
        self.viewer.layers.remove(layer)        # delete the layer 'Area n'
        self.show_popup_window()                # open a new pop-up window

    def restore(self):
        """ Restore the data of a specific area in the pop-up window """

        # (19.07.2024)
        name = self.sender().objectName()
        hit = [d for d in self.areas if d.get('name') == name]
        segment = hit[0]                # selected area
        segment['done'] = False
        self.show_popup_window()

    def compare_and_transfer(self, name: str):
        """
        Compare old and new data and transfer the changes to the segmentation
        and uncertainty data

        Parameters
        ----------
        name : str
            Name of the area (e.g. 'area 5')
        """

        # (09.08.2024)
        hit = [d for d in self.areas if d.get('name') == name]
        segment = hit[0]
        label = segment['label']                # segment label
        uncertainty = segment['uncertainty']

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):

            # search for the changed data points
            new_data = self.viewer.layers[name].data

            # compare new and old data
            site = segment['site']              # recall the old values
            old_data = np.zeros_like(new_data, dtype=int)
            old_data[site] = label + 1
            delta = new_data - old_data

            ind_add = np.where(delta > 0)       # new data points
            ind_del = np.where(delta < 0)       # deleted data points

            # transfer the changes to the segmentation layer
            self.segmentation[ind_add] = 1
            self.segmentation[ind_del] = 0
            self.viewer.layers['Segmentation'].data = self.segmentation

            # transfer the changes to the uncertainty layer
            self.uncertainty[ind_add] = uncertainty
            self.uncertainty[ind_del] = 0.0

            segment['done'] = True              # mark this area as treated

    def btn_save(self):
        """ Save the segmentation and uncertainty data to files on drive """

        # (26.07.2024)
        # 1st: save the segmentation data
        filename = self.parent / '_Segmentation.npy'
        print('Save', filename)

        try:
            file = open(filename, 'wb')
            np.save(file, self.segmentation)
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

        # 3rd: save the labels
        filename = self.parent / '_Labels.npy'
        print('Save', filename)

        try:
            file = open(filename, 'wb')
            np.save(file, self.labels)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
        finally:
            if 'file' in locals() and file:
                file.close()

    def reload(self):
        """ Read the segmentation and uncertainty data from files on drive """

        # (30.07.2024)
        # 1st: read the segmentation data
        filename = self.parent / '_Segmentation.npy'
        print('Read', filename)
        
        try:
            file = open(filename, 'rb')
            self.segmentation = np.load(file)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return
        finally:
            if 'file' in locals() and file:
                file.close()

        # If the 'Segmentation' layer already exists'
        if any(layer.name.startswith('Segmentation') and
            isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):
            self.viewer.layers['Segmentation'].data = self.segmentation
        else:
            self.viewer.add_labels(self.segmentation, name='Segmentation')

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

        # 3rd: read the labels
        filename = self.parent / '_Labels.npy'
        print('Read', filename)

        try:
            file = open(filename, 'rb')
            self.labels = np.load(file)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return
        finally:
            if 'file' in locals() and file:
                file.close()

        if self.areas == []:
            self.build_areas()          # define areas

    def final_segmentation(self):
        """
        Close all open area layers, close the pop-up window, save the
        segmentation and if applicable also the uncertainty data to files on
        drive
        """

        # (13.08.2024)
        # 1st: close all open area layers
        lst = [layer for layer in self.viewer.layers
            if layer.name.startswith('Segment ') and
            isinstance(layer, napari.layers.Labels)]

        for layer in lst:
            name = layer.name
            print('Close areas', name)
            self.compare_and_transfer(name)
            self.viewer.layers.remove(layer)    # delete the layer 'Segment n'

        if hasattr(self, 'popup_window'):       # close the pop-up window
            self.popup_window.close()

        # Build a filename for the segmentation data
        filename = self.stem1[:-3] + '_segNew.tif'
        default_filename = str(self.parent / filename)
        filename, _ = QFileDialog.getSaveFileName(self, 'Segmentation file', \
            default_filename, 'TIFF files (*.tif *.tiff)')
        if filename == '':                      # Cancel has been pressed
            QMessageBox.information(self, 'Cancel', 'Cancel has been pressed.')
            return

        # Save the segmentation data
        filename = Path(filename)
        print('Save', filename)
        try:
            imwrite(filename, self.segmentation)
        except BaseException as error:
            QMessageBox.warning(self, 'I/O Error:', str(error))
            return

        # Save the uncertainty data
        if self.save_uncertainty:
            filename2 = self.stem1[:-3] + '_uncNew.tif'
            filename2 = filename.parent / filename2
            print('Save', filename2)
            try:
                imwrite(filename2, self.uncertainty)
            except BaseException as error:
                QMessageBox.warning(self, 'I/O Error:', str(error))

    def checkbox_save_uncertainty(self, state: Qt.Checked):
        """ Toggle the bool variable save_uncertainty """

        if state == Qt.Checked:
            self.save_uncertainty = True
        else:
            self.save_uncertainty = False

    def btn_info(self):     # pragma: no cover
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
            print('shape:', data.shape)
            print('values:', values)
            print('counts:', counts)
        else:
            print('This is not an image or label layer!')
        print()
