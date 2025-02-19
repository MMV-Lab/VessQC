"""
Module for the definition of the class VessQC

Imports
-------
napari, numpy, pathlib.Path, qtpy.QtCore.QSize, qtpy.QtCore.QT, qtpy.QtWidgets,
scipy.ndimage, SimpleITK, tifffile.imread, tifffile.imwrite, time

Exports
-------
VessQC
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

from typing import TYPE_CHECKING

import numpy as np
import napari
import SimpleITK as sitk
import time
from tifffile import imread, imwrite
from scipy import ndimage
from pathlib import Path
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
# from vessqc._mv_widget import CrossWidget, MultipleViewerWidget

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
    build_areas()
        Define areas that correspond to values of equal uncertainty
    show_popup_window()
        Define a pop-up window for the uncertainty list
    new_entry(area_i: dict, grid_layout: QGridLayout, i: int):
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
        # self.start_multiple_viewer = True
        self.save_uncertainty = False

        # Define some labels and buttons
        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)

        btnLoad = QPushButton('Load image')
        btnLoad.clicked.connect(self.load_image)

        btnSegmentation = QPushButton('Read segmentation')
        btnSegmentation.clicked.connect(self.read_segmentation)

        # Test output
        btnInfo = QPushButton('Info')
        btnInfo.clicked.connect(self.btn_info)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)

        label3 = QLabel('Curation')
        label3.setFont(font)

        btnUncertainty = QPushButton('Load uncertainty list')
        btnUncertainty.clicked.connect(self.show_popup_window)

        btnSave = QPushButton('Save intermediate curation')
        btnSave.clicked.connect(self.btn_save)

        btnReload = QPushButton('Load saved curation')
        btnReload.clicked.connect(self.reload)

        label4 = QLabel('_______________')
        label4.setAlignment(Qt.AlignHCenter)

        btnFinalSegmentation = QPushButton('Generate final segmentation')
        btnFinalSegmentation.clicked.connect(self.final_segmentation)

        cbxSaveUncertainty = QCheckBox('Save uncertainty')
        cbxSaveUncertainty.stateChanged.connect(self.checkbox_save_uncertainty)

        # Define the layout of the main widget
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(btnLoad)
        self.layout().addWidget(btnSegmentation)
        self.layout().addWidget(btnInfo)
        self.layout().addWidget(label2)
        self.layout().addWidget(label3)
        self.layout().addWidget(btnUncertainty)
        self.layout().addWidget(btnSave)
        self.layout().addWidget(btnReload)
        self.layout().addWidget(label4)
        self.layout().addWidget(btnFinalSegmentation)
        self.layout().addWidget(cbxSaveUncertainty)

    def load_image(self):
        """
        Read the image file and save it in an image layer
        """

        # (23.05.2024);
        self.areas = [None]

        # Find and load the image file
        filter1 = "TIFF files (*.tif *.tiff);;NIfTI files (*.nii *.nii.gz);;\
            All files (*.*)"
        filename, _ = \
            QFileDialog.getOpenFileName(self, 'Image file', '', filter1)
        if filename == '':                      # Cancel has been pressed
            print('The "Cancel" button has been pressed.')
            return

        path = Path(filename)
        self.parent = path.parent              # The data directory
        self.stem1 = path.stem                 # Name of the file
        suffix = path.suffix.lower()           # File extension

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
                print('Unknown file type: %s%s!' % (self.stem1, suffix))
                return
        except BaseException as error:
            print('Error:', error)
            return

        self.viewer.add_image(self.image, name=self.stem1)   # Show the image

    def read_segmentation(self):
        """
        Read the segmentation and uncertanty data and save it in a label and an
        image layer
        """

        # (23.05.2024, revised on 05.02.2025)
        # Search for the segmentation file
        stem2 = self.stem1[:-3] + '_segPred'
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
            print('No segmentation file %s found!' % (path))
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
            print('Error:', error)
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
            print('No uncertainty file %s found!' % (path))
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
            print('Error:', error)
            return

        # Save the uncertanity data in an image layer
        self.viewer.add_image(self.uncertainty, name='Uncertainty', \
            blending='additive', visible=False)

        if self.areas == [None]:
            self.build_areas()              # define areas

    def build_areas(self):
        """ Define areas that correspond to values of equal uncertainty """

        # (09.08.2024)
        uncertainties, counts = np.unique(self.uncertainty, return_counts=True)
        n = len(uncertainties)
        self.areas = [None]                     # List of dictionaries

        for i in range(1, n):
            area_i = {'name': 'Area %d' % (i), 'uncertainty': uncertainties[i],
                'counts': counts[i], 'centroid': None, 'where': None,
                'done': False}
            self.areas.append(area_i)

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
        i = 0
        grid_layout.addWidget(QLabel('Area'), i, 0)
        grid_layout.addWidget(QLabel('Uncertainty'), i, 1)
        grid_layout.addWidget(QLabel('Counts'), i, 2)
        grid_layout.addWidget(QLabel('done'), i, 3)
        i += 1

        # Define buttons and select values for some labels
        for area_i in self.areas[1:]:
            if area_i['done']: continue
            else:                       # show only the untreated areas
                self.new_entry(area_i, grid_layout, i)
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

        for area_i in self.areas[1:]:
            if area_i['done']:          # show only the treated areas
                self.new_entry(area_i, grid_layout, i)
                i += 1
            else: continue

        # Show the pop-up window
        self.popup_window.show()
        
    def new_entry(self, area_i: dict, grid_layout: QGridLayout, i: int):
        """
        New entry for 'Area n' in the grid layout

        Parameters
        ----------
        area_i : dict
            name, uncertainty, counts, centroid, where and done for a specific
            area
        grid_layout : QGridLayout
            Layout for a QGroupBox
        i : int
            Index in the grid_layout
        """

        # (13.08.2024)
        name = area_i['name']
        uncertainty1 = '%.5f' % (area_i['uncertainty'])
        counts = '%d' % (area_i['counts'])
        done = area_i['done']

        # Define some buttons and labels
        button1 = QPushButton(name)
        button1.clicked.connect(self.show_area)
        label1 = QLabel(uncertainty1)
        label2 = QLabel(counts)

        if done:
            button1.setEnabled(False)       # enable button for treated areas
            button2 = QPushButton('restore', objectName=name)
            button2.clicked.connect(self.restore)
        else:
            button2 = QPushButton('done', objectName=name)
            button2.clicked.connect(self.done)

        # Arange the buttons and labels in the grid
        grid_layout.addWidget(button1, i, 0)
        grid_layout.addWidget(label1, i, 1)
        grid_layout.addWidget(label2, i, 2)
        grid_layout.addWidget(button2, i, 3)

    def show_area(self):
        """ Show the data for a specific uncertanty in a new label layer """

        # (29.05.2024)
        name = self.sender().text()         # text of the button: "Area n"
        index = int(name[5:])               # index = number of the "Area n"
        area_i = self.areas[index]          # selected area
        uncertainty1 = area_i['uncertainty']# uncertainty value of the area
        centroid = area_i['centroid']       # center of the data points

        # Check whether the layer 'name' already exists
        if any(layer.name == name and
            isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):
            # Place the affected label layer at the top of the stack
            layer = self.viewer.layers[name]
            source_index = self.viewer.layers.index(layer)
            target_index = len(self.viewer.layers)
            self.viewer.layers.move(source_index, target_index)
            layer.visible = True
            
        else:
            # Show the data for a specific uncertanty;
            where1 = np.where(self.uncertainty == uncertainty1)
            area_i['where'] = where1        # save the result for later use
            data = np.zeros(self.uncertainty.shape, dtype=np.int_)
            data[where1] = index + 1        # build a new label layer
            layer = self.viewer.add_labels(data, name=name)

            # Find the center of the data points
            if centroid == None:
                centroid = ndimage.center_of_mass(data)
                centroid = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
                area_i['centroid'] = centroid
                print('Centroid:', centroid)

        # Set the appropriate level and focus
        self.viewer.dims.current_step = centroid
        self.viewer.camera.center = centroid

        # Change to the matching color
        layer.selected_label = index + 1

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
        index = int(name[5:])
        self.areas[index]['done'] = False
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
        index = int(name[5:])                   # n = number of the area
        area_i = self.areas[index]              # selected area

        # If a label layer with this name exists:
        if any(layer.name == name and
            isinstance(layer, napari.layers.Labels)
            for layer in self.viewer.layers):
            # search for the changed data points
            new_data = self.viewer.layers[name].data

            # compare new and old data
            where1 = area_i['where']            # recall the old values
            old_data = np.zeros(new_data.shape, dtype=np.int_)
            old_data[where1] = index + 1
            delta = new_data - old_data

            ind_new = np.where(delta > 0)       # new data points
            ind_del = np.where(delta < 0)       # deleted data points

            # transfer the changes to the segmentation layer
            self.segmentation[ind_new] = 1
            self.segmentation[ind_del] = 0
            self.viewer.layers['Segmentation'].data = self.segmentation

            # transfer the changes to the uncertainty layer
            uncertainty = area_i['uncertainty']
            self.uncertainty[ind_new] = uncertainty
            self.uncertainty[ind_del] = 0.0
            self.viewer.layers['Uncertainty'].data = self.uncertainty

            area_i['done'] = True               # mark this area as treated

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
            print('Error:', error)
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
            print('Error:', error)
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
            print('Error:', error)
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
            print('Error:', error)
            return
        finally:
            if 'file' in locals() and file:
                file.close()

        # If the 'Uncertainty' layer already exists'
        if any(layer.name.startswith('Uncertainty') and
            isinstance(layer, napari.layers.Image)
            for layer in self.viewer.layers):
            self.viewer.layers['Uncertainty'].data = self.uncertainty
        else:
            self.viewer.add_image(self.uncertainty, name='Uncertainty', \
                blending='additive', visible=False)

        if self.areas == [None]:
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
            if layer.name.startswith('Area') and
            isinstance(layer, napari.layers.Labels)]

        for layer in lst:
            name = layer.name
            print('Close areas', name)
            self.compare_and_transfer(name)
            self.viewer.layers.remove(layer)    # delete the layer 'Area n'

        if hasattr(self, 'popup_window'):       # close the pop-up window
            self.popup_window.close()

        # Build a filename for the segmentation data
        filename = self.stem1[:-3] + '_segNew.tif'
        default_filename = str(self.parent / filename)
        filename, _ = QFileDialog.getSaveFileName(self, 'Segmentation file', \
            default_filename, 'TIFF files (*.tif *.tiff)')
        if filename == '':                      # Cancel has been pressed
            print('The "Cancel" button has been pressed.')
            return

        # Save the segmentation data
        filename = Path(filename)
        print('Save', filename)
        try:
            imwrite(filename, self.segmentation)
        except BaseException as error:
            print('Error:', error)
            return

        # Save the uncertainty data
        if self.save_uncertainty:
            filename2 = self.stem1[:-3] + '_uncNew.tif'
            filename2 = filename.parent / filename2
            print('Save', filename2)
            try:
                imwrite(filename2, self.uncertainty)
            except BaseException as error:
                print('Error:', error)

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

    """
    def on_close(self):
        # (29.05.2024)
        print("Good by!")
        if hasattr(self, 'popup_window'):
            self.popup_window.close()
    """
