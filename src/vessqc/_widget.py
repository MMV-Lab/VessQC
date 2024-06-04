"""
This module contains four napari widgets declared in
different ways:

- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/
"""
from typing import TYPE_CHECKING

import SimpleITK as sitk
import numpy as np
import warnings
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
# from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari


class VessQC(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        label1 = QLabel('Vessel quality check')
        font = label1.font()
        font.setPointSize(12)
        label1.setFont(font)

        btnLoad = QPushButton('Load file')
        btnLoad.clicked.connect(self.btn_load)

        btnSegmentation = QPushButton('Segmentation')
        btnSegmentation.clicked.connect(self.btn_segmentation)

        label2 = QLabel('_______________')
        label2.setAlignment(Qt.AlignHCenter)

        label3 = QLabel('Curation')
        label3.setFont(font)

        btnUncertainty = QPushButton('Load uncertainty list')
        btnUncertainty.clicked.connect(self.btn_uncertainty)
        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(btnLoad)
        self.layout().addWidget(btnSegmentation)
        self.layout().addWidget(label2)
        self.layout().addWidget(label3)
        self.layout().addWidget(btnUncertainty)

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
        filename = QFileDialog.getOpenFileName(self, 'Input file', filter='*.nii')
        image_file = Path(filename[0])
        parent = image_file.parent        # directory
        uncertainty_file = parent / 'Uncertainty.nii'
        prediction_file = parent / 'Prediction.nii'

        # Load the data files
        image_data = sitk.ReadImage(image_file)
        image = sitk.GetArrayFromImage(image_data)
        print('Image')
        self.show_info(image)

        uncertainty_data = sitk.ReadImage(uncertainty_file)
        self.uncertainty = sitk.GetArrayFromImage(uncertainty_data)
        self.uncert_values, counts = np.unique(self.uncertainty, \
            return_counts=True)
        print('num uncert', np.size(self.uncert_values))
        print(self.uncert_values)
        print(counts)

        prediction_data = sitk.ReadImage(prediction_file)
        self.prediction = sitk.GetArrayFromImage(prediction_data)

        # Show the image in Napari
        self.viewer.add_image(image, name='Data')
        self.viewer.dims.ndisplay=3

    def btn_segmentation(self):
        # Show prediction and uncertainty in Napari
        self.viewer.add_labels(self.prediction, name='Prediction')
        self.viewer.add_image(self.uncertainty, name='Uncertainty', \
            blending='additive')

    def btn_uncertainty(self):
        # Define a pop-up window for the uncertainty-list
        self.uncertainty_list = QWidget()
        self.uncertainty_list.setWindowTitle('napari')

        # Define a group box inside the uncertanty list
        group_box = QGroupBox('Uncertainty List')

        # Define some buttons
        button1 = QPushButton('Area 1')
        button1.clicked.connect(self.button_clicked)
        button2 = QPushButton('Area 2')
        button2.clicked.connect(self.button_clicked)
        button3 = QPushButton('Area 51')
        button3.clicked.connect(self.button_clicked)

        # add widgets to the group box
        self.group_box_layout = QGridLayout()
        self.group_box_layout.addWidget(button1, 0, 0)
        self.group_box_layout.addWidget(QLabel('123.45'), 0, 1)
        self.group_box_layout.addWidget(button2, 1, 0)
        self.group_box_layout.addWidget(QLabel('234.56'), 1, 1)
        self.group_box_layout.addWidget(button3, 2, 0)
        self.group_box_layout.addWidget(QLabel('345.67'), 2, 1)
        group_box.setLayout(self.group_box_layout)

        uncertainty_list_layout = QVBoxLayout()
        uncertainty_list_layout.addWidget(group_box)
        self.uncertainty_list.setLayout(uncertainty_list_layout)

        # Show the pop-up window
        self.uncertainty_list.show()

    def button_clicked(self):
        text1 = self.sender().text()
        # print(self.group_box_layout.indexOf(self.sender()))
        index = self.group_box_layout.indexOf(self.sender())
        text2 = self.group_box_layout.itemAt(index + 1).widget().text()
        print(text2)

    def show_info(self, image):
        print('type',  type(image))
        print('dtype', image.dtype)
        print('size',  image.size)
        print('ndim',  image.ndim)
        print('shape', image.shape)
        print('---')

        print('min', np.min(image))
        print('median', np.median(image))
        print('max', np.max(image))
        print('mean', np.mean(image))
        print('std', np.std(image))

    def on_close(self):
        print("on_close")
        if hasattr(self, uncertainty_list):
            self.uncertainty_list.close()
