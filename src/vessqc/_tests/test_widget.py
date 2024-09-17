import pytest
import numpy as np
from unittest.mock import patch
from pathlib import Path
from tifffile import imread
from vessqc._widget import VessQC

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
# capsys is a pytest fixture that captures stdout and stderr output streams

parent = Path(__file__).parent / "Data"
image_path =  parent / "Image.tif"

# define a fixture for the image
@pytest.fixture
def image_data():
    return imread(image_path)

@pytest.fixture
def prediction_data():
    prediction_file = parent / 'Prediction.tif'
    return imread(prediction_file)

@pytest.fixture
def uncertainty_data():
    uncertainty_file = parent / 'Uncertainty.tif'
    return imread(uncertainty_file)

# define a fixture for the Napari viewer and the VessQC object
@pytest.fixture
def vess_qc(make_napari_viewer):
    viewer = make_napari_viewer()
    return VessQC(viewer)           # create a VessQC object and give it back


def test_init(vess_qc):
    assert str(type(vess_qc)) == "<class 'vessqc._widget.VessQC'>"
    assert vess_qc.start_multiple_viewer == True
    assert vess_qc.save_uncertainty == False


# Durch den Patch wird die Funktion getOpenFileName durch die Return-Values
# ersetzt
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(image_path, None))


def test_btn_load(mock_get_open_file_name, vess_qc, image_data):
    viewer = vess_qc.viewer
    vess_qc.btn_load()

    mock_get_open_file_name.assert_called_once()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Image'
    assert np.array_equal(layer.data, image_data)
    assert vess_qc.parent == parent
    assert vess_qc.suffix == '.tif'


def test_btn_segmentation(vess_qc, prediction_data, uncertainty_data):
    viewer = vess_qc.viewer
    vess_qc.suffix = '.tif'
    vess_qc.parent = parent
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
