# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (12.09.2024)

import pytest
import napari
import numpy as np
import builtins
from unittest import mock
from pathlib import Path
from tifffile import imread, imwrite
from vessqc import ExampleQWidget
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
def vessqc(make_napari_viewer, qtbot):
    # Create an Object of class ExampleQWidget
    # (12.09.2024)
    widget = ExampleQWidget(make_napari_viewer())
    qtbot.addWidget(widget)             # Fixture from pytest-qt
    return widget

# define fixtures for the image data
@pytest.fixture
def box32x32():
    return imread(PARENT / 'Box32x32_IM.tif')

@pytest.fixture
def segmentation():
    return imread(PARENT / 'Box32x32_segPred.tif')

@pytest.fixture
def segmentation_new():
    # (24.09.2024)
    return imread(PARENT / 'Box32x32_segNew.tif')

@pytest.fixture
def uncertainty():
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
def areas(vessqc, uncertainty):
    # (18.09.2024)
    vessqc.build_areas(uncertainty)
    return vessqc.areas

@pytest.mark.init
def test_init(vessqc):
    # (12.09.2024)
    assert isinstance(vessqc, QWidget)          # Base class of ExampleQWidget
    assert isinstance(vessqc, ExampleQWidget)   # Class of vessqc
    assert issubclass(ExampleQWidget, QWidget)  # Is QWidget the base class?
    assert isinstance(vessqc.viewer, napari.Viewer)
    assert isinstance(vessqc.layout(), QVBoxLayout)
    assert vessqc.save_uncertainty == False


@pytest.mark.load_image
def test_load_image(vessqc, box32x32):
    # (12.09.2024)
    viewer = vessqc.viewer

    with mock.patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(PARENT / 'Box32x32_IM.tif', None)) as mock_open:
        vessqc.load_image()
        mock_open.assert_called_once()

    assert vessqc.areas == []
    assert vessqc.parent == PARENT
    assert vessqc.stem1 == 'Box32x32_IM'
    assert np.array_equal(vessqc.image, box32x32)

    # Check the contents of the first Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, box32x32)


@pytest.mark.read_segmentation
def test_read_segmentation(vessqc, segmentation, uncertainty):
    # (13.09.2024)
    viewer = vessqc.viewer
    vessqc.stem1 = 'Box32x32_IM'
    vessqc.parent = PARENT
    vessqc.areas = []
    vessqc.read_segmentation()

    assert np.array_equal(vessqc.segmentation, segmentation)
    assert np.array_equal(vessqc.uncertainty,  uncertainty)

    # Check the contents of the next Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Segmentation'
    assert np.array_equal(layer.data, segmentation)

    assert isinstance(vessqc.areas, list)


@pytest.mark.build_areas
def test_build_areas(vessqc, uncertainty):
    # (17.09.2024)
    vessqc.build_areas(uncertainty)

    assert len(vessqc.areas) == 9
    assert vessqc.areas[0]['name'] == 'Segment_1'
    assert vessqc.areas[1]['label'] == 3
    assert vessqc.areas[2]['uncertainty'] == np.float32(0.4)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['com'] == None       # center of mass
    assert vessqc.areas[5]['site'] == None
    assert vessqc.areas[6]['done'] == False


@pytest.mark.popup_window
def test_popup_window(vessqc, areas):
    # (17.09.2024)
    vessqc.areas = areas

    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        vessqc.show_popup_window()
        mock_show.assert_called_once()

    popup_window = vessqc.popup_window
    assert isinstance(popup_window, QWidget)
    assert popup_window.windowTitle() == 'napari'
    assert popup_window.minimumSize() == qtpy.QtCore.QSize(350, 300)

    layout = popup_window.layout()
    assert isinstance(layout, QVBoxLayout)
    assert layout.count() == 1

    item0 = layout.itemAt(0)
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


@pytest.mark.transfer
def test_transfer(vessqc, segmentation, segmentation_new, uncertainty,
    uncertainty_new, segment4_data, segment4_new, areas):
    # (24.09.2024)
    vessqc.segmentation = segmentation
    vessqc.uncertainty = uncertainty
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
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
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
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
        QTest.mouseClick(button3, Qt.LeftButton)

    assert areas[3]['done'] == False


# tmp_path is a pytest fixture (see lab book from 27.09.2024)
@pytest.mark.save
def test_save(tmp_path, vessqc, segmentation, uncertainty):
    # (27.09.2024)
    vessqc.segmentation = segmentation
    vessqc.uncertainty  = uncertainty
    vessqc.parent       = tmp_path
    vessqc.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, segmentation)

    filename = tmp_path / '_Uncertainty.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, uncertainty)


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
def test_reload(tmp_path, vessqc, segmentation, uncertainty):
    # (01.10.2024)
    vessqc.parent = tmp_path
    vessqc.areas = []

    # 1st: save the segmentation data
    filename = tmp_path / '_Segmentation.npy'
    try:
        file = open(filename, 'wb')
        np.save(file, segmentation)
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
        np.save(file, uncertainty)
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
    np.testing.assert_array_equal(layer.data,          segmentation)
    np.testing.assert_array_equal(vessqc.segmentation, segmentation)
    np.testing.assert_array_equal(vessqc.uncertainty,  uncertainty)

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
def test_final_segmentation(tmp_path, vessqc, segmentation, uncertainty):
    # (01.10.2024)
    vessqc.segmentation = segmentation
    vessqc.uncertainty = uncertainty
    vessqc.parent = tmp_path
    vessqc.stem1 = 'Box32x32_IM'
    vessqc.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    # call the function final_segmentation()
    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(filename1, None)) as mock_save:
        vessqc.final_segmentation()
        mock_save.assert_called_once()

    try:
        seg_saved_data = imread(filename1)
    except BaseException as error:
        print('Error:', error)
        assert False

    try:
        unc_saved_data = imread(filename2)
    except BaseException as error:
        print('Error:', error)
        assert False

    np.testing.assert_array_equal(seg_saved_data, segmentation)
    np.testing.assert_array_equal(unc_saved_data, uncertainty)


@pytest.mark.final_seg_with_exc
def test_final_seg_with_exc(tmp_path, vessqc, segmentation, uncertainty):
    # (02.10.2024)
    vessqc.segmentation = segmentation
    vessqc.uncertainty = uncertainty
    vessqc.parent = tmp_path
    vessqc.save_uncertainty = True
    vessqc.stem1 = 'Box32x32_IM'
    output_file = str(tmp_path / 'Box32x32_segNew.tif')

    # call the function final_segmentation()
    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(output_file, None)), \
        mock.patch("vessqc._widget.imwrite",
        side_effect=BaseException("File error")):
        vessqc.final_segmentation()

    filename = tmp_path / 'Box32x32_segNew.tif'
    assert not filename.exists()

    filename = tmp_path / 'Box32x32_uncNew.tif'
    assert not filename.exists()
