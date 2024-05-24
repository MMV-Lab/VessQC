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

from pathlib import Path
import SimpleITK as sitk
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
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
        QApplication.instance().aboutToQuit.connect(self.on_close)

    def btn_load(self):
        file = QFileDialog.getOpenFileName(self, 'Input file', filter='*.nii')
        file = Path(file[0])
        parent = file.parent        # directory
        uncertainty_file = parent / 'Uncertainty.nii'
        prediction_file = parent / 'Prediction.nii'

        image_data = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image_data)
        uncertainty_data = sitk.ReadImage(uncertainty_file)
        self.uncertainty = sitk.GetArrayFromImage(uncertainty_data)
        prediction_data = sitk.ReadImage(prediction_file)
        self.prediction = sitk.GetArrayFromImage(prediction_data)

        self.viewer.add_image(image, name='Data')
        self.viewer.dims.ndisplay=3

    def btn_segmentation(self):
        self.viewer.add_labels(self.prediction, name='Prediction')
        self.viewer.add_image(self.uncertainty, name='Uncertainty', \
            blending='additive')

    def btn_uncertainty(self):
        self.uncertainty_list = QWidget()
        self.uncertainty_list.setWindowTitle('napari')
        label1 = QLabel('123.45')
        label2 = QLabel('234.56')
        label3 = QLabel('345.67')
        
        self.uncertainty_list.setLayout(QVBoxLayout())
        self.uncertainty_list.layout().addWidget(label1)
        self.uncertainty_list.layout().addWidget(label2)
        self.uncertainty_list.layout().addWidget(label3)
        self.uncertainty_list.show()

    def on_close(self):
        if self.uncertanty_list is not None:
            self.uncertanty_list.close()
            