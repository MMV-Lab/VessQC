# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (12.09.2024)

import pytest
import napari
import numpy as np
import builtins
from unittest import mock
from unittest.mock import patch
from pathlib import Path
from tifffile import imread, imwrite
from vessqc._widget import VessQC
import qtpy
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

# A single constant
PARENT = Path(__file__).parent / 'data'

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed in your
# testing environment
@pytest.fixture
def vessqc(make_napari_viewer):
    # (12.09.2024)
    viewer = make_napari_viewer()
    vessqc = VessQC(viewer)     # create an Object of class VessQC
    return vessqc

# define fixtures for the image data
@pytest.fixture
def box32x32():
    return imread(PARENT / 'Box32x32_IM.tif')

@pytest.fixture
def segmentation_data():
    return imread(PARENT / 'Box32x32_segPred.tif')

@pytest.fixture
def segmentation_new():
    # (24.09.2024)
    return imread(PARENT / 'Box32x32_segNew.tif')

@pytest.fixture
def uncertainty_data():
    return imread(PARENT / 'Box32x32_uncertainty.tif')

@pytest.fixture
def uncertainty_new():
    # (26.09.2024)
    return imread(PARENT / 'Box32x32_uncNew.tif')

@pytest.fixture
def segment4_data():
    # (20.09.2024)
    return imread(PARENT / 'Segment_4.tif')

@pytest.fixture
def segment4_new():
    # (24.09.2024)
    return imread(PARENT / 'Segment_4_new.tif')

@pytest.fixture
def areas(vessqc, uncertainty_data):
    # (18.09.2024)
    vessqc.build_areas(uncertainty_data)
    return vessqc.areas

@pytest.mark.init
def test_init(vessqc):
    # (12.09.2024)
    assert isinstance(vessqc, VessQC)
    assert isinstance(vessqc.layout(), QVBoxLayout)
    assert vessqc.save_uncertainty == False


# The patch replaces the getOpenFileName() function with the return values
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(PARENT / 'Box32x32_IM.tif', None))
@pytest.mark.load_image
def test_load_image(mock_open_file, vessqc, box32x32):
    # (12.09.2024)
    viewer = vessqc.viewer
    vessqc.load_image()

    assert vessqc.areas == []
    mock_open_file.assert_called_once()
    assert vessqc.parent == PARENT
    assert vessqc.stem1 == 'Box32x32_IM'

    # Check the contents of the first Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, box32x32)


@pytest.mark.read_segmentation
def test_read_segmentation(vessqc, segmentation_data, uncertainty_data):
    # (13.09.2024)
    viewer = vessqc.viewer
    vessqc.stem1 = 'Box32x32_IM'
    vessqc.parent = PARENT
    vessqc.areas = []
    vessqc.read_segmentation()

    assert np.array_equal(vessqc.segmentation, segmentation_data)
    assert np.array_equal(vessqc.uncertainty,  uncertainty_data)

    # Check the contents of the next Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Segmentation'
    assert np.array_equal(layer.data, segmentation_data)

    assert isinstance(vessqc.areas, list)


@pytest.mark.build_areas
def test_build_areas(vessqc, uncertainty_data):
    # (17.09.2024)
    vessqc.build_areas(uncertainty_data)

    assert len(vessqc.areas) == 9
    assert vessqc.areas[0]['name'] == 'Segment_1'
    assert vessqc.areas[1]['label'] == 3
    assert vessqc.areas[2]['uncertainty'] == np.float32(0.4)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['com'] == None       # center of mass
    assert vessqc.areas[5]['site'] == None
    assert vessqc.areas[6]['done'] == False


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.popup_window
def test_popup_window(mock_show_widget, vessqc, areas):
    # (17.09.2024)
    vessqc.areas = areas
    vessqc.show_popup_window()

    popup_window = vessqc.popup_window
    assert isinstance(popup_window, QWidget)
    assert popup_window.windowTitle() == 'napari'
    assert popup_window.minimumSize() == qtpy.QtCore.QSize(350, 300)

    vbox_layout = popup_window.layout()
    assert isinstance(vbox_layout, QVBoxLayout)
    assert vbox_layout.count() == 1

    item0 = vbox_layout.itemAt(0)
    assert isinstance(item0, QWidgetItem)

    scroll_area = item0.widget()
    assert isinstance(scroll_area, QScrollArea)

    group_box = scroll_area.widget()
    assert isinstance(group_box, QGroupBox)
    assert group_box.title() == 'Uncertainty list'

    grid_layout = group_box.layout()
    assert isinstance(grid_layout, QGridLayout)
    assert grid_layout.rowCount() == 12
    assert grid_layout.columnCount() == 4

    item_0 = grid_layout.itemAtPosition(5, 0)
    item_1 = grid_layout.itemAtPosition(5, 1)
    item_2 = grid_layout.itemAtPosition(5, 2)
    item_3 = grid_layout.itemAtPosition(5, 3)
    assert item_0.widget().text() == 'Segment_5'
    assert item_1.widget().text() == '0.60000'
    assert item_2.widget().text() == '37'
    assert item_3.widget().text() == 'done'

    mock_show_widget.assert_called_once()


@pytest.mark.new_entry
def test_new_entry(vessqc, areas):
    # (18.09.2024)
    grid_layout = QGridLayout()
    vessqc.new_entry(areas[2], grid_layout, 3)
    item_0 = grid_layout.itemAtPosition(3, 0)
    item_1 = grid_layout.itemAtPosition(3, 1)
    item_2 = grid_layout.itemAtPosition(3, 2)
    item_3 = grid_layout.itemAtPosition(3, 3)

    assert grid_layout.rowCount() == 4
    assert grid_layout.columnCount() == 4
    assert isinstance(item_0, QWidgetItem)
    assert isinstance(item_0.widget(), QPushButton)
    assert isinstance(item_1.widget(), QLabel)
    assert isinstance(item_2.widget(), QLabel)
    assert isinstance(item_3.widget(), QPushButton)

    assert item_0.widget().text() == 'Segment_3'
    assert item_1.widget().text() == '0.40000'
    assert item_2.widget().text() == '35'
    assert item_3.widget().text() == 'done'


@pytest.mark.show_area
def test_show_area(vessqc, areas, segment4_data):
    # (20.09.2024)
    vessqc.areas = areas

    # Define button1 to call vessqc.show_area()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)
    button1 = QPushButton('Segment_4')
    button1.clicked.connect(vessqc.show_area)
    layout.addWidget(button1)

    # Here I simulate a mouse click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    segment = areas[3]
    com = (15, 16, 15)                  # center of mass
    assert segment['com'] == com
    assert vessqc.viewer.dims.current_step == com
    assert vessqc.viewer.camera.center == com

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)
    else:
        assert False

    # 2nd click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
    else:
        assert False


# QWidget.show is required to intercept vessqc.show_popup_window()
@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.transfer
def test_transfer(mock_show_widget, vessqc, segmentation_data, segmentation_new,
    uncertainty_data, uncertainty_new, areas, segment4_data, segment4_new):
    # (24.09.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    segmentation_layer = vessqc.viewer.add_labels(vessqc.segmentation,
        name='Segmentation')
    vessqc.areas = areas

    # Define three buttons to call vessqc.show_area(), vessqc.done() and
    # vessqc.restore()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    button1 = QPushButton('Segment_4')
    button1.clicked.connect(vessqc.show_area)
    layout.addWidget(button1)

    button2 = QPushButton('done', objectName='Segment_4')
    button2.clicked.connect(vessqc.done)
    layout.addWidget(button2)

    button3 = QPushButton('restore', objectName='Segment_4')
    button3.clicked.connect(vessqc.restore)
    layout.addWidget(button3)

    # press button1 'Segment_4' to call vessqc.show_area()
    QTest.mouseClick(button1, Qt.LeftButton)

    # search for the Napari layer "Segment_4" and change its data
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)

        # replace the data of the layer
        layer.data = segment4_new
    else:
        assert False

    # press button2 to call vessqc.done()
    QTest.mouseClick(button2, Qt.LeftButton)

    # the data in the Napari layers Prediction and Uncertainty should have
    # been changed by the function compare_and_transfer()
    assert np.array_equal(segmentation_layer.data, segmentation_new)
    assert np.array_equal(vessqc.segmentation, segmentation_new)
    assert np.array_equal(vessqc.uncertainty,  uncertainty_new)
    assert areas[3]['done'] == True

    # the Napari layer 'Segment_4' is removed
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        assert False

    # press button3 to call vessqc.restore()
    QTest.mouseClick(button3, Qt.LeftButton)
    assert areas[3]['done'] == False


# tmp_path is a pytest fixture (see lab book from 27.09.2024)
@pytest.mark.save
def test_save(tmp_path, vessqc, segmentation_data, uncertainty_data):
    # (27.09.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty  = uncertainty_data
    vessqc.parent       = tmp_path
    vessqc.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, segmentation_data)

    filename = tmp_path / '_Uncertainty.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, uncertainty_data)


@pytest.mark.save_with_exc
def test_save_with_exc(tmp_path, vessqc):
    # (27.09.2024)
    vessqc.segmentation = np.ones((3, 3, 3), dtype=int)
    vessqc.uncertainty  = np.ones((3, 3, 3))
    vessqc.parent       = tmp_path

    # Simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        vessqc.btn_save()

    assert mock_warning.call_count == 2

    filename = tmp_path / '_Segmentation.npy'
    assert not filename.exists()

    filename = tmp_path / '_Uncertainty.npy'
    assert not filename.exists()


@pytest.mark.reload
def test_reload(tmp_path, vessqc, segmentation_data, uncertainty_data):
    # (01.10.2024)
    vessqc.parent = tmp_path
    vessqc.areas = []

    # 1st: save the segmentation data
    filename = tmp_path / '_Segmentation.npy'
    try:
        file = open(filename, 'wb')
        np.save(file, segmentation_data)
    except BaseException as error:
        print('Error:', error)
        assert False
    finally:
        if 'file' in locals() and file:
            file.close()

    #2nd: save the uncertainty data
    filename = tmp_path / '_Uncertainty.npy'
    try:
        file = open(filename, 'wb')
        np.save(file, uncertainty_data)
    except BaseException as error:
        print('Error:', error)
        assert False
    finally:
        if 'file' in locals() and file:
            file.close()

    vessqc.reload()

    # Test the content of the Napari layer and vessqc nD arrays
    assert len(vessqc.viewer.layers) == 1
    layer = vessqc.viewer.layers[0]
    assert layer.name == 'Segmentation'
    np.testing.assert_array_equal(layer.data,          segmentation_data)
    np.testing.assert_array_equal(vessqc.segmentation, segmentation_data)
    np.testing.assert_array_equal(vessqc.uncertainty,  uncertainty_data)

    # Test vessqc.areas
    assert len(vessqc.areas) == 9
    assert vessqc.areas[0]['name'] == 'Segment_1'
    assert vessqc.areas[1]['label'] == 3
    assert vessqc.areas[2]['uncertainty'] == np.float32(0.4)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['com'] == None
    assert vessqc.areas[5]['site'] == None
    assert vessqc.areas[6]['done'] == False


@pytest.mark.reload_with_exc
def test_reload_with_exc(tmp_path, vessqc):
    # (01.10.2024)
    vessqc.segmentation = np.ones((3, 3, 3), dtype=int)
    vessqc.uncertainty  = np.ones((3, 3, 3))
    vessqc.parent       = tmp_path
    vessqc.areas        = []

    real_open = builtins.open       # Save original

    def open_side_effect(file, *args, **kwargs):
        # Suggestion from ChatGPT
        if '_Segmentation.npy' in str(file) or '_Uncertainty.npy' in str(file):
            raise OSError("File error")
        return real_open(file, *args, **kwargs)

    # simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=open_side_effect), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        vessqc.reload()

    assert mock_warning.call_count == 1
    assert len(vessqc.viewer.layers) == 0
    assert vessqc.areas == []


@pytest.mark.final_segmentation
def test_final_segmentation(tmp_path, vessqc, segmentation_data,
    uncertainty_data):
    # (01.10.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    vessqc.parent = tmp_path
    vessqc.save_uncertainty = True
    vessqc.stem1 = 'Box32x32_IM'
    output_file = str(tmp_path / 'Box32x32_segNew.tif')

    # call the function final_segmentation()
    with patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(output_file, None)):
        vessqc.final_segmentation()

    try:
        filename = tmp_path / 'Box32x32_segNew.tif'
        segNew_data = imread(filename)
        np.testing.assert_array_equal(segNew_data, segmentation_data)
    except BaseException as error:
        print('Error:', error)
        assert False

    try:
        filename = tmp_path / 'Box32x32_uncNew.tif'
        uncNew_data = imread(filename)
        np.testing.assert_array_equal(uncNew_data, uncertainty_data)
    except BaseException as error:
        print('Error:', error)
        assert False


@patch("vessqc._widget.imwrite", side_effect=BaseException("File error"))
@pytest.mark.final_seg_with_exc
def test_final_seg_with_exc(mock_imwrite, tmp_path, vessqc, segmentation_data,
    uncertainty_data):
    # (02.10.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    vessqc.parent = tmp_path
    vessqc.save_uncertainty = True
    vessqc.stem1 = 'Box32x32_IM'
    output_file = str(tmp_path / 'Box32x32_segNew.tif')

    # call the function final_segmentation()
    with patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(output_file, None)):
        vessqc.final_segmentation()

    filename = tmp_path / 'Box32x32_segNew.tif'
    assert not filename.exists()

    filename = tmp_path / 'Box32x32_uncNew.tif'
    assert not filename.exists()
