import pytest
import numpy as np
import napari
import qtpy
from qtpy.QtWidgets import QGridLayout
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from unittest.mock import patch
from pathlib import Path
from tifffile import imread
from vessqc._widget import VessQC
# capsys is a pytest fixture that captures stdout and stderr output streams

# Class to save and share variables between tests
class ValueStorage():
    parent = Path(__file__).parent / "Data"
    image_path =  parent / "Image.tif"


# define a fixture for the image
@pytest.fixture
def image_data():
    return imread(ValueStorage.image_path)

@pytest.fixture
def prediction_data():
    prediction_file = ValueStorage.parent / 'Prediction.tif'
    return imread(prediction_file)

@pytest.fixture
def uncertainty_data():
    uncertainty_file = ValueStorage.parent / 'Uncertainty.tif'
    return imread(uncertainty_file)

@pytest.fixture
def area5_data():
    area5_file = ValueStorage.parent / 'Area5.tif'
    return imread(area5_file)

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
@pytest.fixture
def vessqc(make_napari_viewer):
    viewer = make_napari_viewer()
    return VessQC(viewer)           # create a VessQC object and give it back

@pytest.fixture
def areas(vessqc, uncertainty_data):
    vessqc.uncertainty = uncertainty_data
    vessqc.build_areas()
    return vessqc.areas


@pytest.mark.init
def test_init(vessqc):
    assert str(type(vessqc)) == "<class 'vessqc._widget.VessQC'>"
    assert vessqc.start_multiple_viewer == True
    assert vessqc.save_uncertainty == False


# Durch den Patch wird die Funktion getOpenFileName durch die Return-Values
# ersetzt
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(ValueStorage.image_path, None))
@pytest.mark.load
def test_load(mock_get_open_file_name, vessqc, image_data):
    viewer = vessqc.viewer
    vessqc.btn_load()

    mock_get_open_file_name.assert_called_once()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Image'
    assert np.array_equal(layer.data, image_data)
    assert vessqc.parent == ValueStorage.parent
    assert vessqc.suffix == '.tif'


@pytest.mark.segmentation
def test_segmentation(vessqc, prediction_data, uncertainty_data):
    # (13.09.2024)
    viewer = vessqc.viewer
    vessqc.suffix = '.tif'
    vessqc.parent = ValueStorage.parent
    vessqc.is_tifffile = True
    vessqc.areas = [None]
    vessqc.btn_segmentation()

    assert len(viewer.layers) == 2
    layer0 = viewer.layers[0]
    layer1 = viewer.layers[1]
    assert layer0.name == 'Prediction'
    assert layer1.name == 'Uncertainty'
    assert np.array_equal(layer0.data, prediction_data)
    assert np.array_equal(layer1.data, uncertainty_data)


@pytest.mark.build_areas
def test_build_areas(vessqc, uncertainty_data):
    # (17.09.2024)
    vessqc.uncertainty = uncertainty_data
    vessqc.build_areas()
    ValueStorage.areas = vessqc.areas

    assert len(vessqc.areas) == 10
    assert vessqc.areas[1]['name'] == 'Area 1'
    assert vessqc.areas[2]['unc_value'] == np.float32(0.2)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['centroid'] == ()
    assert vessqc.areas[5]['where'] == None
    assert vessqc.areas[6]['done'] == False


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.uncertainty
def test_uncertainty(mock_widget_show, vessqc, areas):
    # (17.09.2024)
    vessqc.areas = areas
    vessqc.areas[7]['done'] == True
    vessqc.btn_uncertainty()
    popup_window = vessqc.popup_window

    assert str(type(popup_window)) == "<class 'PyQt5.QtWidgets.QWidget'>"
    assert popup_window.windowTitle() == 'napari'
    assert popup_window.minimumSize() == qtpy.QtCore.QSize(350, 300)

    vbox_layout = popup_window.layout()
    assert str(type(vbox_layout)) == "<class 'PyQt5.QtWidgets.QVBoxLayout'>"
    assert vbox_layout.count() == 1

    item0 = vbox_layout.itemAt(0)
    assert str(type(item0)) == "<class 'PyQt5.QtWidgets.QWidgetItem'>"

    scroll_area = item0.widget()
    assert str(type(scroll_area)) == "<class 'PyQt5.QtWidgets.QScrollArea'>"

    group_box = scroll_area.widget()
    assert str(type(group_box)) == "<class 'PyQt5.QtWidgets.QGroupBox'>"
    assert group_box.title() == 'Uncertainty list'

    grid_layout = group_box.layout()
    item_0 = grid_layout.itemAtPosition(5, 0)
    item_1 = grid_layout.itemAtPosition(5, 1)
    item_2 = grid_layout.itemAtPosition(5, 2)
    item_3 = grid_layout.itemAtPosition(5, 3)
    assert str(type(grid_layout)) == "<class 'PyQt5.QtWidgets.QGridLayout'>"
    assert grid_layout.rowCount() == 12
    assert grid_layout.columnCount() == 4
    assert item_0.widget().text() == areas[5]['name']
    assert item_1.widget().text() == '%.5f' % (areas[5]['unc_value'])
    assert item_2.widget().text() == '%d' % (areas[5]['counts'])
    assert item_3.widget().text() == 'done'

    mock_widget_show.assert_called_once()
    #assert False


@pytest.mark.new_entry
def test_new_entry(vessqc, areas):
    # (18.09.2024)
    area_i = areas[2]
    grid_layout = QGridLayout()
    vessqc.new_entry(area_i, grid_layout, 2)
    item_0 = grid_layout.itemAtPosition(2, 0)
    item_1 = grid_layout.itemAtPosition(2, 1)
    item_2 = grid_layout.itemAtPosition(2, 2)
    item_3 = grid_layout.itemAtPosition(2, 3)

    assert grid_layout.rowCount() == 3
    assert grid_layout.columnCount() == 4
    assert str(type(item_0)) == "<class 'PyQt5.QtWidgets.QWidgetItem'>"
    assert str(type(item_0.widget())) == "<class 'PyQt5.QtWidgets.QPushButton'>"
    assert str(type(item_1.widget())) == "<class 'PyQt5.QtWidgets.QLabel'>"
    assert str(type(item_2.widget())) == "<class 'PyQt5.QtWidgets.QLabel'>"
    assert str(type(item_3.widget())) == "<class 'PyQt5.QtWidgets.QPushButton'>"
    assert item_0.widget().text() == area_i['name']
    assert item_1.widget().text() == '%.5f' % (area_i['unc_value'])
    assert item_2.widget().text() == '%d' % (area_i['counts'])
    assert item_3.widget().text() == 'done'


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.show_area
def test_show_area(mock_widget_show, vessqc, uncertainty_data, area5_data,
    areas):
    # (20.09.2024)
    vessqc.uncertainty = uncertainty_data
    vessqc.areas = areas

    # In order to be able to define the value "name = self.sender().text()",
    # we take the way via the function self.btn_uncertainty()
    vessqc.btn_uncertainty()
    popup_window = vessqc.popup_window
    vbox_layout = popup_window.layout()
    scroll_area = vbox_layout.itemAt(0).widget()
    group_box = scroll_area.widget()
    grid_layout = group_box.layout()
    button5 = grid_layout.itemAtPosition(5, 0).widget()

    # Here I simulate a mouse click on the "Area 5" button
    QTest.mouseClick(button5, Qt.LeftButton)

    assert areas[5]['centroid'] == (15, 15, 15)
    assert vessqc.viewer.dims.current_step == (15, 15, 15)
    assert vessqc.viewer.camera.center == (15, 15, 15)

    if any(layer.name == 'Area 5' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Area 5']
        assert layer.name == 'Area 5'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, area5_data)
    else:
        assert False

    # 2nd click ob the "Area 5" button
    QTest.mouseClick(button5, Qt.LeftButton)

    if any(layer.name == 'Area 5' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Area 5']
        assert layer.name == 'Area 5'
    else:
        assert False
