"""
Reference:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
"""
from typing import TYPE_CHECKING

import numpy as np
import napari
import SimpleITK as sitk
import tempfile
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
        self.parent = tempfile.gettempdir()     # Directory for temporari files
        self.areas = [None]                     # List of dictionaries

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

        btnSave = QPushButton('Save the areas to disk')
        btnSave.clicked.connect(self.btn_save)

        btnReload = QPushButton('Reload the areas')
        btnReload.clicked.connect(self.btn_reload)

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
        # Find the data file
        # (23.05.2024)
        filename = QFileDialog.getOpenFileName(self, 'Input file', filter='*.nii')
        image_file = Path(filename[0])
        self.parent = image_file.parent        # The data directory

        # Load the data file
        image_data = sitk.ReadImage(image_file)
        self.image = sitk.GetArrayFromImage(image_data)

        # Load the uncertainty file and calculate data and counts
        uncertainty_file = self.parent / 'Uncertainty.nii'
        uncertainty_data = sitk.ReadImage(uncertainty_file)
        self.uncertainty = sitk.GetArrayFromImage(uncertainty_data)
        unc_values, counts = np.unique(self.uncertainty, \
            return_counts=True)

        n = len(unc_values)
        for i in range(1, n):
            area_i = {'name': 'Area %d' % (i), 'unc_value': unc_values[i],
                'counts': counts[i], 'centroid': (0, 0, 0), 'done': False}
            self.areas.append(area_i)

        # Load the prediction file
        prediction_file = self.parent / 'Prediction.nii'
        prediction_data = sitk.ReadImage(prediction_file)
        self.prediction = sitk.GetArrayFromImage(prediction_data)

        # Call the Multiple Viewer and the Cross widget
        dock_widget = MultipleViewerWidget(self.viewer)
        cross = CrossWidget(self.viewer)

        self.viewer.window.add_dock_widget(dock_widget, name="Sample")
        self.viewer.window.add_dock_widget(cross, name="Cross", area="left")

        # Show the image in Napari
        self.viewer.add_image(self.image, name='Input_Vol')
        # self.viewer.dims.ndisplay = 3     # 3D view

        self.layer_list = [None for i in range(n+1)]    # List for the label layers.

    def btn_segmentation(self):
        # Show prediction in Napari
        # (23.05.2024)
        self.viewer.add_labels(self.prediction, name='Prediction')

        # Save uncertainty layer, but don't show it
        uncertainty_layer = self.viewer.add_image(self.uncertainty, \
            name='Uncertainty', blending='additive')
        uncertainty_layer.visible = False

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

        # Show the data of a specific uncertanty
        if self.layer_list[index] == None:
            # find all data points
            indices = np.where(self.uncertainty == unc_value)
            data = np.zeros(self.uncertainty.shape, dtype=np.int8)
            data[indices] = index + 1          # build a new label layer
            layer = self.viewer.add_labels(data, name=name1)

            # Find the center of the data points
            centroid = ndimage.center_of_mass(data)
            centroid = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
            area_i['centroid'] = centroid

            self.layer_list[index] = layer  # safe the label layer in a list
        else:
            # Change to the desired layer
            centroid = area_i['centroid']
            layer = self.layer_list[index]
            self.viewer.layers.selection.active = layer
            
        # Set the appropriate level and focus
        self.viewer.dims.current_step = centroid
        self.viewer.camera.center = centroid

        # Change to the matching color
        layer.selected_label = index + 1

        print('Centroid:', centroid)

    def btn_done(self):
        # (18.07.2024)
        name = self.sender().objectName()       # name of the object
        index = int(name[5:])                   # number of the area
        area_i = self.areas[index]              # selected area
        area_i['done'] = True                   # mark this area as treated
        self.btn_uncertainty()                  # open a new pop-up window

        # If a label layer with this name exists:
        if any(layer.name == name and isinstance(layer, napari.layers.Labels) \
            for layer in self.viewer.layers):
            # search for the changed data points
            layer = self.viewer.layers[name]
            new_data = layer.data
            unc_value = area_i['unc_value']
            indices = np.where(self.uncertainty == unc_value)
            old_data = np.zeros(self.uncertainty.shape, dtype=np.int8)
            old_data[indices] = index + 1
            delta = new_data - old_data

            values, counts = np.unique(delta, return_counts=True)
            print('values', values)
            print('counts', counts)

            ind_plus = np.where(delta > 1)          # new data points
            ind_minus = np.where(delta < -1)        # deleted data points

            # transfer the changes to the prediction layer
            self.prediction[ind_plus] = 1
            self.prediction[ind_minus] = 0
            layer = self.viewer.layers['Prediction']
            layer.data = self.prediction

    def btn_restore(self):
        # (19.07.2024)
        name = self.sender().objectName()
        index = int(name[5:])
        area_i = self.areas[index]
        area_i['done'] = False
        self.btn_uncertainty()

    def btn_save(self):
        # (26.07.2024)
        print('Save the data to disk')

        # 1st save the prediction data
        layer = self.viewer.layers['Prediction']
        data = layer.data

        try:
            filename = self.parent / '_Prediction.npz'
            file = open(filename, 'wb')
            np.save(file, data)
        except BaseException as error:
            print('Error', error)
        finally:
            if 'file' in locals() and file:
                file.close()

        #2nd, save the uncertainty data
        layer = self.viewer.layers['Uncertainty']
        data = layer.data

        try:
            filename = self.parent / '_Uncertainty.npz'
            file = open(filename, 'wb')
            np.save(file, data)
        except BaseException as error:
            print('Error', error)
        finally:
            if 'file' in locals() and file:
                file.close()

    def btn_reload(self):
        # (24.07.2024)
        print('Not yet implemented!')

    def btn_info(self):
        # layer = self.viewer.layers['Area 5']
        layer = self.viewer.layers.selection.active
        print('layer', layer.name)

        if isinstance(layer, napari.layers.Image):
            image = layer.data

            print('type',  type(image))
            print('dtype', image.dtype)
            print('size',  image.size)
            print('ndim',  image.ndim)
            print('shape', image.shape)
            print('---')

            print('min', np.min(image))
            print('max', np.max(image))
            print('median', np.median(image))
            print('mean %.3f' % (np.mean(image)))
            print('std %.3f' %  (np.std(image)))
        elif isinstance(layer, napari.layers.Labels):
            data = layer.data
            values, counts = np.unique(data, return_counts=True)

            print('type', type(data))
            print('dtype', data.dtype)
            print('shape', data.shape)
            print('values', values)
            print('counts', counts)
        else:
            print('This is not an image or label layer!')
        print()

    def on_close(self):
        # (29.05.2024)
        print("Good by!")
        if hasattr(self, 'popup_window'):
            self.popup_window.close()
