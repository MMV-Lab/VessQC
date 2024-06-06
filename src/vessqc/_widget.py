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
        self.image = sitk.GetArrayFromImage(image_data)
        # self.show_info(self.image)

        uncertainty_data = sitk.ReadImage(uncertainty_file)
        self.uncertainty = sitk.GetArrayFromImage(uncertainty_data)
        self.uncert_values, self.counts = np.unique(self.uncertainty, \
            return_counts=True)

        prediction_data = sitk.ReadImage(prediction_file)
        self.prediction = sitk.GetArrayFromImage(prediction_data)

        # Show the image in Napari
        self.viewer.add_image(self.image, name='Data')
        self.viewer.dims.ndisplay=3

    def btn_segmentation(self):
        # Show prediction and uncertainty in Napari
        self.viewer.add_labels(self.prediction, name='Prediction')
        self.viewer.add_image(self.uncertainty, name='Uncertainty', \
            blending='additive')

    def btn_uncertainty(self):
        # Define a pop-up window for the uncertainty list
        self.pop_up_window = QWidget()
        self.pop_up_window.setWindowTitle('napari')
        self.pop_up_window.setMinimumSize(QSize(300, 300))

        # define a scroll area inside the pop-up window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Define a group box inside the scroll area
        group_box = QGroupBox('Uncertainty List')
        scroll_area.setWidget(group_box)

        # add widgets to the group box
        self.group_box_layout = QGridLayout()
        self.group_box_layout.addWidget(QLabel('Area'), 0, 0)
        self.group_box_layout.addWidget(QLabel('Uncertainty'), 0, 1)
        self.group_box_layout.addWidget(QLabel('Counts'), 0, 2)

        # Define buttons and select uncertainty values for the labels
        n = len(self.uncert_values)
        for i in range(1, n):
            text1 = 'Area ' + str(i)
            button = QPushButton(text1)
            button.clicked.connect(self.button_clicked)
            value = self.uncert_values[i]
            label1 = QLabel(str(value))
            value = self.counts[i]
            label2 = QLabel(str(value))

            self.group_box_layout.addWidget(button, i, 0)
            self.group_box_layout.addWidget(label1, i, 1)
            self.group_box_layout.addWidget(label2, i, 2)

        group_box.setLayout(self.group_box_layout)

        pop_up_window_layout = QVBoxLayout()
        pop_up_window_layout.addWidget(scroll_area)
        self.pop_up_window.setLayout(pop_up_window_layout)

        # Show the pop-up window
        self.pop_up_window.show()

    def button_clicked(self):
        # index = self.group_box_layout.indexOf(self.sender())
        # text2 = self.group_box_layout.itemAt(index + 1).widget().text()

        text1 = self.sender().text()        # name of the button: Area nn
        index = int(text1[5:])              # number of the area
        value = self.uncert_values[index]   # uncertainty value of the area

        indices = np.where(self.uncertainty == value)   # find all data points
        area = np.zeros(self.uncertainty.shape, dtype=np.int8)
        area[indices] = index + 1           # build a new label layer
        self.viewer.add_labels(area, name=text1)

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
        print("Good by!")
        if hasattr(self, 'pop_up_window'):
            self.pop_up_window.close()
