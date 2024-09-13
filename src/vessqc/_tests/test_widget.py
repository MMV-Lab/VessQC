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
image = imread(image_path)

# Das Fixture definiert eine Funktion, die in allen test_Funktionen verwendet
# werden kann, um ein Objekt der Klasse VessQC zur Verfügung zu haben.
@pytest.fixture
def make_vqc(make_napari_viewer):
    yield VessQC(make_napari_viewer())


def test_init(make_vqc):
    vess_qc = make_vqc
    assert str(type(vess_qc)) == '<class \'vessqc._widget.VessQC\'>'


# Durch den Patch wird die Funktion getOpenFileName durch die Return-Values
# ersetzt
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(image_path, None))
def test_btn_load(mock_get_open_file_name, make_vqc):
    vess_qc = make_vqc
    viewer = vess_qc.viewer
    vess_qc.btn_load()

    mock_get_open_file_name.assert_called_once()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Image"
    assert np.array_equal(layer.data, image)


def test_btn_segmentation(make_vqc):
    vess_qc = make_vqc
    viewer = vess_qc.viewer
    vess_qc.parent = parent
    vess_qc.suffix = '.tif'
    vess_qc.is_tifffile = True
    vess_qc.areas = [None]
    vess_qc.btn_segmentation()
    