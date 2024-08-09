"""
tbd.
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (03.05.2024)

from typing import TYPE_CHECKING

import numpy as np
import napari
import SimpleITK as sitk
import time
import warnings
from scipy import ndimage
from pathlib import Path
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
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
from vessqc._mv_widget import CrossWidget, MultipleViewerWidget

if TYPE_CHECKING:
    import napari


class VessQC(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    # (03.05.2024)
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.start_multiple_viewer = True

        # Define some labels and buttons
        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)

        btnLoad = QPushButton('Load file')
        btnLoad.clicked.connect(self.btn_load)

        btnSegmentation = QPushButton('Segmentation')
        btnSegmentation.clicked.connect(self.btn_segmentation)

        # Test output
        btnInfo = QPushButton('Info')
        btnInfo.clicked.connect(self.btn_info)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)

        label3 = QLabel('Curation')
        label3.setFont(font)

        btnUncertainty = QPushButton('Load uncertainty list')
        btnUncertainty.clicked.connect(self.btn_uncertainty)

        btnSave = QPushButton('Save intermediate curation')
        btnSave.clicked.connect(self.btn_save)

        btnReload = QPushButton('Load saved curation')
        btnReload.clicked.connect(self.btn_reload)

        label4 = QLabel('_______________')
        label4.setAlignment(Qt.AlignHCenter)

        btnFinalSeg = QPushButton('Generate final segmentation')
        btnFinalSeg.clicked.connect(self.btn_final_seg)

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
        self.layout().addWidget(btnFinalSeg)

        # Close the uncertanty_list when Napari is closed
        def wrapper(self, func, event):
            self.on_close()
            return func(event)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func = self.viewer.window._qt_window.closeEvent
            self.viewer.window._qt_window.closeEvent = \
                lambda event: wrapper(self, func, event)

    def btn_load(self):
        # (23.05.2024);
        if self.start_multiple_viewer:             # run this part only once!
            # Call the multiple viewer and the cross widget
            dock_widget = MultipleViewerWidget(self.viewer)
            cross_widget = CrossWidget(self.viewer)

            self.viewer.window.add_dock_widget(dock_widget, name="Sample")
            self.viewer.window.add_dock_widget(cross_widget, name="Cross", \
                area="left")
            self.start_multiple_viewer = False

        # Find and load the data file
        try:
            filter1 = "NIfTI files (*.nii *.nii.gz);;TIFF files (*.tif *.tiff);;\
                All files (*.*)"
            filename = QFileDialog.getOpenFileName(self, 'Input file', '', \
                filter1)

            if filename == ('', ''):                # Cancel has been pressed
                print('The "Cancel" button has been pressed.')
                return
            else:
                image_path = Path(filename[0])
                image_data = sitk.ReadImage(image_path)
                self.image = sitk.GetArrayFromImage(image_data)
                name1 = image_path.stem             # Name of the file
                self.parent = image_path.parent     # The data directory
                self.suffix = image_path.suffix.lower()

                # Show the image in Napari
                self.viewer.add_image(self.image, name=name1)
        except BaseException as error:
            print('Error:', error)

    def btn_segmentation(self):
        # (23.05.2024)
        # The file type depends on the extension
        if self.suffix == '.nii':
            name1 = 'Prediction.nii'
            name2 = 'Uncertainty.nii'
        elif self.suffix == '.gz':
            name1 = 'Prediction.nii.gz'
            name2 = 'Uncertainty.nii.gz'
        elif self.suffix == '.tif':
            name1 = 'Prediction.tif'
            name2 = 'Uncertainty.tif'
        elif self.suffix == 'tiff':
            name1 = 'Prediction.tiff'
            name2 = 'Uncertainty.tiff'
        else:
            print('Unknown file type')
            return

        # Load the prediction file
        try:
            prediction_path = self.parent / name1
            prediction_data = sitk.ReadImage(prediction_path)
            self.prediction = sitk.GetArrayFromImage(prediction_data)

            # Show prediction in a label layer
            self.viewer.add_labels(self.prediction, name='Prediction')
        except BaseException as error:
            print('Error:', error)

        # Load the uncertainty file and calculate data and counts
        try:
            uncertainty_path = self.parent / name2
            uncertainty_data = sitk.ReadImage(uncertainty_path)
            self.uncertainty = sitk.GetArrayFromImage(uncertainty_data)

            # Save uncertainty layer, but hide it
            self.viewer.add_image(self.uncertainty, name='Uncertainty', \
                blending='additive', visible=False)
        except BaseException as error:
            print('Error', error)

        self.build_areas()              # define areas

    def btn_uncertainty(self):
        # Define a pop-up window for the uncertainty list
        # (24.05.2024)
        self.popup_window = QWidget()
        self.popup_window.setWindowTitle('napari')
        self.popup_window.setMinimumSize(QSize(350, 300))
        popup_window_layout = QVBoxLayout()
        self.popup_window.setLayout(popup_window_layout)

        # define a scroll area inside the pop-up window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        popup_window_layout.addWidget(scroll_area)

        # Define a group box inside the scroll area
        group_box = QGroupBox('Uncertainty List')
        group_box_layout = QGridLayout()
        group_box.setLayout(group_box_layout)
        scroll_area.setWidget(group_box)

        # add widgets to the group box
        i = 0
        group_box_layout.addWidget(QLabel('Area'), i, 0)
        group_box_layout.addWidget(QLabel('Uncertainty'), i, 1)
        group_box_layout.addWidget(QLabel('Counts'), i, 2)
        group_box_layout.addWidget(QLabel('done'), i, 3)
        i += 1

        # Define buttons and select values for some labels
        for area_i in self.areas[1:]:
            if area_i['done']:      # show only the untreated areas
                continue

            name = area_i['name']
            button1 = QPushButton(name)
            button1.clicked.connect(self.btn_show_area)
            unc_value = '%.5f' % (area_i['unc_value'])
            label1 = QLabel(unc_value)
            counts = '%d' % (area_i['counts'])
            label2 = QLabel(counts)
            button2 = QPushButton('done', objectName=name)
            button2.clicked.connect(self.btn_done)

            group_box_layout.addWidget(button1, i, 0)
            group_box_layout.addWidget(label1, i, 1)
            group_box_layout.addWidget(label2, i, 2)
            group_box_layout.addWidget(button2, i, 3)
            i += 1

        # show a horizontal line
        line = QWidget()
        line.setFixedHeight(3)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet('background-color: mediumblue')
        group_box_layout.addWidget(line, i, 0, 1, -1)
        i += 1

        # The treated areas are shown in the lower part of the group box
        group_box_layout.addWidget(QLabel('Area'), i, 0)
        group_box_layout.addWidget(QLabel('Uncertainty'), i, 1)
        group_box_layout.addWidget(QLabel('Counts'), i, 2)
        group_box_layout.addWidget(QLabel('restore'), i, 3)
        i += 1

        for area_i in self.areas[1:]:
            if not area_i['done']:      # show only the treated areas
                continue

            name = area_i['name']
            button1 = QPushButton(name)
            button1.clicked.connect(self.btn_show_area)
            unc_value = '%.5f' % (area_i['unc_value'])
            label1 = QLabel(unc_value)
            counts = '%d' % (area_i['counts'])
            label2 = QLabel(counts)
            button2 = QPushButton('restore', objectName=name)
            button2.clicked.connect(self.btn_restore)

            group_box_layout.addWidget(button1, i, 0)
            group_box_layout.addWidget(label1, i, 1)
            group_box_layout.addWidget(label2, i, 2)
            group_box_layout.addWidget(button2, i, 3)
            i += 1

        # Show the pop-up window
        self.popup_window.show()
        
    def btn_show_area(self):
        # (29.05.2024)
        name1 = self.sender().text()        # text of the button: "Area n"
        index = int(name1[5:])              # n = number of the area
        area_i = self.areas[index]          # selected area
        unc_value = area_i['unc_value']     # uncertainty value of the area

        # Show the data for a specific uncertanty;
        where1 = np.where(self.uncertainty == unc_value)
        area_i['where'] = where1            # save np.where() for later use
        data = np.zeros(self.uncertainty.shape, dtype=np.int8)
        data[where1] = index + 1            # build a new label layer
        layer = self.viewer.add_labels(data, name=name1)

        # Find the center of the data points
        centroid = ndimage.center_of_mass(data)
        centroid = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
        print('Centroid:', centroid)

        # Set the appropriate level and focus
        self.viewer.dims.current_step = centroid
        self.viewer.camera.center = centroid

        # Change to the matching color
        layer.selected_label = index + 1

    def btn_done(self):
        # (18.07.2024)
        name = self.sender().objectName()       # name of the object: 'Area n'
        index = int(name[5:])                   # n = number of the area
        self.areas[index]['done'] = True        # mark this area as treated
        self.compare_and_transfer(name)         # transfer of data
        layer1 = self.viewer.layers[name]
        self.viewer.layers.remove(layer1)       # delete the layer 'Area n'
        self.btn_uncertainty()                  # open a new pop-up window

    def btn_restore(self):
        # (19.07.2024)
        name = self.sender().objectName()
        index = int(name[5:])
        self.areas[index]['done'] = False
        self.btn_uncertainty()

    def btn_save(self):
        # (26.07.2024)
        print('Save the intermediate data to disk')

        # 1st save the prediction data
        layer = self.viewer.layers['Prediction']
        data = layer.data

        try:
            filename = self.parent / '_Prediction.npy'
            file = open(filename, 'wb')
            np.save(file, data)
        except BaseException as error:
            print('Error:', error)
        finally:
            if 'file' in locals() and file:
                file.close()

        #2nd, save the uncertainty data
        layer = self.viewer.layers['Uncertainty']
        data = layer.data

        try:
            filename = self.parent / '_Uncertainty.npy'
            file = open(filename, 'wb')
            np.save(file, data)
        except BaseException as error:
            print('Error:', error)
        finally:
            if 'file' in locals() and file:
                file.close()

    def btn_reload(self):
        # (30.07.2024)
        print('Read the intermediate data from disk')
        
        # 1st, read the prediction data
        try:
            filename = self.parent / '_Prediction.npy'
            file = open(filename, 'rb')
            self.prediction = np.load(file)
        except BaseException as error:
            print('Error:', error)
        finally:
            if 'file' in locals() and file:
                file.close()

        # If the 'Prediction' layer already exists'
        if any(layer.name.startswith('Prediction') and isinstance(layer, \
            napari.layers.Labels) for layer in self.viewer.layers):
            layer = self.viewer.layers['Prediction']
            layer.data = self.prediction
        else:
            self.viewer.add_labels(self.prediction, name='Prediction')

        # 2st, read the uncertainty data
        try:
            filename = self.parent / '_Uncertainty.npy'
            file = open(filename, 'rb')
            self.uncertainty = np.load(file)
        except BaseException as error:
            print('Error:', error)
        finally:
            if 'file' in locals() and file:
                file.close()

        # If the 'Uncertainty' layer already exists'
        if any(layer.name.startswith('Uncertainty') and isinstance(layer, \
            napari.layers.Image) for layer in self.viewer.layers):
            layer = self.viewer.layers['Uncertainty']
            layer.data = self.uncertainty
        else:
            self.viewer.add_image(self.uncertainty, name='Uncertainty', \
                blending='additive', visible=False)

        self.build_areas()              # define areas

    def btn_final_seg(self):
        # (02.08.2024)
        print('Generate final segmentation')

    def btn_info(self):
        # (25.07.2024)
        # layer = self.viewer.layers['Area 5']
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
            print('max:', np.max(image))
            print('median:', np.median(image))
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

    def build_areas(self):
        # (09.08.2024) Define areas that correspond to values of equal 
        # uncertainty
        unc_values, counts = np.unique(self.uncertainty, return_counts=True)
        n = len(unc_values)
        self.areas = [None]                     # List of dictionaries

        for i in range(1, n):
            area_i = {'name': 'Area %d' % (i), 'unc_value': unc_values[i],
                'counts': counts[i], 'where': None, 'done': False}
            self.areas.append(area_i)

    def compare_and_transfer(self, name):
        # (09.08.2024) Compare old and new data and transfer the changes to
        # the prediction and uncertainty data
        index = int(name[5:])                   # n = number of the area
        area_i = self.areas[index]              # selected area

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels) \
            for layer in self.viewer.layers):
            # search for the changed data points
            layer1 = self.viewer.layers[name]
            new_data = layer1.data

            # compare new and old data
            where1 = area_i['where']            # recall the old values
            old_data = np.zeros(self.uncertainty.shape, dtype=np.int8)
            old_data[where1] = index + 1
            delta = new_data - old_data

            ind_new = np.where(delta > 0)       # new data points
            ind_del = np.where(delta < 0)       # deleted data points

            # transfer the changes to the prediction layer
            self.prediction[ind_new] = 1
            self.prediction[ind_del] = 0
            layer2 = self.viewer.layers['Prediction']
            layer2.data = self.prediction

            # transfer the changes to the uncertainty layer
            unc_value = area_i['unc_value']
            self.uncertainty[ind_new] = unc_value
            self.uncertainty[ind_del] = 0.0
            layer3 = self.viewer.layers['Uncertainty']
            layer3.data = self.uncertainty

    def on_close(self):
        # (29.05.2024)
        print("Good by!")
        if hasattr(self, 'popup_window'):
            self.popup_window.close()
