import pytest
import numpy as np
import PyQt5
from unittest.mock import patch
from pathlib import Path
from tifffile import imread
from vessqc._widget import VessQC
# capsys is a pytest fixture that captures stdout and stderr output streams


# Class to save and share variables between tests
class ValueStorage():
    parent = Path(__file__).parent / "Data"
    image_path =  parent / "Image.tif"
    areas = None


# define a fixture for the image
@pytest.fixture
def image_data():
    image_path = ValueStorage.image_path
    return imread(image_path)

@pytest.fixture
def prediction_data():
    parent = ValueStorage.parent
    prediction_file = parent / 'Prediction.tif'
    return imread(prediction_file)

@pytest.fixture
def uncertainty_data():
    parent = ValueStorage.parent
    uncertainty_file = parent / 'Uncertainty.tif'
    return imread(uncertainty_file)

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
@pytest.fixture
def vess_qc(make_napari_viewer):
    viewer = make_napari_viewer()
    return VessQC(viewer)           # create a VessQC object and give it back


@pytest.mark.init
def test_init(vess_qc):
    assert str(type(vess_qc)) == "<class 'vessqc._widget.VessQC'>"
    assert vess_qc.start_multiple_viewer == True
    assert vess_qc.save_uncertainty == False


# Durch den Patch wird die Funktion getOpenFileName durch die Return-Values
# ersetzt
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(ValueStorage.image_path, None))


@pytest.mark.load
def test_btn_load(mock_get_open_file_name, vess_qc, image_data):
    viewer = vess_qc.viewer
    vess_qc.btn_load()

    mock_get_open_file_name.assert_called_once()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Image'
    assert np.array_equal(layer.data, image_data)
    assert vess_qc.parent == ValueStorage.parent
    assert vess_qc.suffix == '.tif'


@pytest.mark.segmentation
def test_btn_segmentation(vess_qc, prediction_data, uncertainty_data):
    viewer = vess_qc.viewer
    vess_qc.suffix = '.tif'
    vess_qc.parent = ValueStorage.parent
    vess_qc.is_tifffile = True
    vess_qc.areas = [None]
    vess_qc.btn_segmentation()

    assert len(viewer.layers) == 2
    layer0 = viewer.layers[0]
    layer1 = viewer.layers[1]
    assert layer0.name == 'Prediction'
    assert layer1.name == 'Uncertainty'
    assert np.array_equal(layer0.data, prediction_data)
    assert np.array_equal(layer1.data, uncertainty_data)


@pytest.mark.build_areas
def test_build_areas(vess_qc, uncertainty_data):
    # (17.09.2024)
    vess_qc.uncertainty = uncertainty_data
    vess_qc.build_areas()
    ValueStorage.areas = vess_qc.areas

    assert len(vess_qc.areas) == 10
    assert vess_qc.areas[1]['name'] == 'Area 1'
    assert vess_qc.areas[2]['unc_value'] == np.float32(0.2)
    assert vess_qc.areas[3]['counts'] == 34
    assert vess_qc.areas[4]['centroid'] == ()
    assert vess_qc.areas[5]['where'] == None
    assert vess_qc.areas[6]['done'] == False


@pytest.mark.uncertainty
def test_btn_uncertainty(vess_qc):
    # (17.09.2024)
    vess_qc.areas = ValueStorage.areas
    vess_qc.btn_uncertainty()

    assert str(type(vess_qc.popup_window)) == "<class 'PyQt5.QtWidgets.QWidget'>"
    assert vess_qc.popup_window.windowTitle() == 'napari'
    #assert vess_qc.popup_window.minimumSize == PyQt5.QtCore.QSize(350, 300)
