# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (12.09.2024)

import pytest
import napari
import numpy as np
import qtpy
from qtpy.QtWidgets import QGridLayout
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from unittest import mock
from unittest.mock import patch
from pathlib import Path
from tifffile import imread, imwrite
from vessqc._widget import VessQC

# A single constant
PARENT = Path(__file__).parent / 'data'

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed in your
# testing environment
@pytest.fixture
def vessqc(make_napari_viewer):
    # (12.09.2024)
    viewer = make_napari_viewer()
    return VessQC(viewer)           # create a VessQC object and give it back

# define fixtures for the image data
@pytest.fixture
def image_data():
    return imread(PARENT / 'Box32x32_IM.tif')

@pytest.fixture
def segmentation_data():
    return imread(PARENT / 'Box32x32_segPred.tif')

@pytest.fixture
def segmentation_new_data():
    # (24.09.2024)
    return imread(PARENT / 'Box32x32_segNew.tif')

@pytest.fixture
def uncertainty_data():
    return imread(PARENT / 'Box32x32_uncertainty.tif')

@pytest.fixture
def uncertainty_new_data():
    # (26.09.2024)
    return imread(PARENT / 'Box32x32_uncNew.tif')

@pytest.fixture
def area5_data():
    # (20.09.2024)
    return imread(PARENT / 'Area5.tif')

@pytest.fixture
def area5_new_data():
    # (24.09.2024)
    return imread(PARENT / 'Area5_new.tif')

@pytest.fixture
def areas(vessqc, uncertainty_data):
    # (18.09.2024)
    vessqc.uncertainty = uncertainty_data
    vessqc.build_areas()
    return vessqc.areas

@pytest.mark.init
def test_init(vessqc):
    # (12.09.2024)
    assert str(type(vessqc)) == "<class 'vessqc._widget.VessQC'>"
    assert vessqc.save_uncertainty == False


# The patch replaces the getOpenFileName() function with the return values
@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
    return_value=(PARENT / 'Box32x32_IM.tif', None))
@pytest.mark.load_image
def test_load_image(mock_open_file_name, vessqc, image_data):
    # (12.09.2024)
    viewer = vessqc.viewer
    vessqc.load_image()

    mock_open_file_name.assert_called_once()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, image_data)
    assert vessqc.parent == PARENT


@pytest.mark.read_segmentation
def test_read_segmentation(vessqc, segmentation_data, uncertainty_data):
    # (13.09.2024)
    viewer = vessqc.viewer
    vessqc.stem1 = 'Box32x32_IM'
    vessqc.parent = PARENT
    vessqc.areas = [None]
    vessqc.read_segmentation()

    assert len(viewer.layers) == 2
    layer0 = viewer.layers[0]
    layer1 = viewer.layers[1]
    assert layer0.name == 'Segmentation'
    assert layer1.name == 'Uncertainty'
    assert np.array_equal(layer0.data, segmentation_data)
    assert np.array_equal(layer1.data, uncertainty_data)


@pytest.mark.build_areas
def test_build_areas(vessqc, uncertainty_data):
    # (17.09.2024)
    vessqc.uncertainty = uncertainty_data
    vessqc.build_areas()

    assert len(vessqc.areas) == 10
    assert vessqc.areas[1]['name'] == 'Area 1'
    assert vessqc.areas[2]['uncertainty'] == np.float32(0.2)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['centroid'] == None
    assert vessqc.areas[5]['where'] == None
    assert vessqc.areas[6]['done'] == False


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.popup_window
def test_popup_window(mock_widget_show, vessqc, areas):
    # (17.09.2024)
    vessqc.areas = areas
    vessqc.areas[7]['done'] == True
    vessqc.show_popup_window()
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
    assert str(type(grid_layout)) == "<class 'PyQt5.QtWidgets.QGridLayout'>"
    assert grid_layout.rowCount() == 12
    assert grid_layout.columnCount() == 4
    item_0 = grid_layout.itemAtPosition(5, 0)
    item_1 = grid_layout.itemAtPosition(5, 1)
    item_2 = grid_layout.itemAtPosition(5, 2)
    item_3 = grid_layout.itemAtPosition(5, 3)
    assert item_0.widget().text() == areas[5]['name']
    assert item_1.widget().text() == '%.5f' % (areas[5]['uncertainty'])
    assert item_2.widget().text() == '%d' % (areas[5]['counts'])
    assert item_3.widget().text() == 'done'

    mock_widget_show.assert_called_once()


@pytest.mark.new_entry
def test_new_entry(vessqc, areas):
    # (18.09.2024)
    grid_layout = QGridLayout()
    vessqc.new_entry(areas[2], grid_layout, 2)
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
    assert item_0.widget().text() == areas[2]['name']
    assert item_1.widget().text() == '%.5f' % (areas[2]['uncertainty'])
    assert item_2.widget().text() == '%d' % (areas[2]['counts'])
    assert item_3.widget().text() == 'done'


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.show_area
def test_show_area(mock_widget_show, vessqc, areas, area5_data):
    # (20.09.2024)
    vessqc.areas = areas

    # In order to be able to define the value "name = self.sender().text()",
    # we take the way via the function self.show_popup_window()
    grid_layout = get_grid_layout(vessqc)
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

    # 2nd click on the "Area 5" button
    QTest.mouseClick(button5, Qt.LeftButton)

    if any(layer.name == 'Area 5' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        layer = vessqc.viewer.layers['Area 5']
        assert layer.name == 'Area 5'
    else:
        assert False


def get_grid_layout(vessqc: VessQC) -> QGridLayout:
    # get the grid_layout from the popup-window of function show_popup_window
    vessqc.show_popup_window()
    popup_window = vessqc.popup_window
    vbox_layout = popup_window.layout()
    scroll_area = vbox_layout.itemAt(0).widget()
    group_box = scroll_area.widget()
    grid_layout = group_box.layout()
    return grid_layout


@patch("qtpy.QtWidgets.QWidget.show")
@pytest.mark.transfer
def test_transfer(mock_widget_show, vessqc, segmentation_data,
    segmentation_new_data, uncertainty_data, uncertainty_new_data, areas,
    area5_new_data):
    # (24.09.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    seg_layer = vessqc.viewer.add_labels(vessqc.segmentation, name='Segmentation')
    unc_layer =  vessqc.viewer.add_image(vessqc.uncertainty, name='Uncertainty')
    vessqc.areas = areas

    # search for the row with button "Area 5"
    grid_layout = get_grid_layout(vessqc)
    index1 = None
    n = grid_layout.rowCount()
    for i in range(n):
        widget0 = grid_layout.itemAtPosition(i, 0).widget()
        if str(type(widget0)) == "<class 'PyQt5.QtWidgets.QPushButton'>" and \
            widget0.text() == 'Area 5':
            index1 = i
            break

    assert index1 != None       # button "Area 5" has been found

    # press the button "Area 5" to call "show_area()"
    QTest.mouseClick(widget0, Qt.LeftButton)

    # search for the Napari layer "Area 5" and change this data
    if any(layer.name == 'Area 5' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        area5_layer = vessqc.viewer.layers['Area 5']
        assert area5_layer.name == 'Area 5'
        area5_layer.data = area5_new_data       # replace the data of the layer
    else:
        assert False

    # press the button done in row "Area 5" to call "compare_and_transfer()"
    widget3 = grid_layout.itemAtPosition(index1, 3).widget()
    assert str(type(widget3)) == "<class 'PyQt5.QtWidgets.QPushButton'>"
    assert widget3.text() == 'done'
    QTest.mouseClick(widget3, Qt.LeftButton)

    # the data in the Napari layers Prediction and Uncertainty should have
    # been changed by the function compare_and_transfer()
    assert np.array_equal(seg_layer.data, segmentation_new_data)
    assert np.array_equal(unc_layer.data, uncertainty_new_data)
    assert areas[5]['done'] == True

    # the Napari layer "Area 5" is removed
    if any(layer.name == 'Area 5' and isinstance(layer, napari.layers.Labels)
        for layer in vessqc.viewer.layers):
        assert False

    # find the new row of the button "Area 5"
    grid_layout = get_grid_layout(vessqc)
    index2 = None
    n = grid_layout.rowCount()
    for i in range(n):
        widget0 = grid_layout.itemAtPosition(i, 0).widget()
        if str(type(widget0)) == "<class 'PyQt5.QtWidgets.QPushButton'>" and \
            widget0.text() == 'Area 5':
            index2 = i
            break

    assert index2 != None       # button "Area 5" has been found
    assert index2 > index1      # "Area 5" is now at the end of the list
    assert widget0.isEnabled() == False     # the button "Area 5" is inactive

    # press the button restore to call the function restore()
    widget3 = grid_layout.itemAtPosition(index2, 3).widget()
    assert str(type(widget3)) == "<class 'PyQt5.QtWidgets.QPushButton'>"
    assert widget3.text() == 'restore'
    QTest.mouseClick(widget3, Qt.LeftButton)

    assert areas[5]['done'] == False

    # find the new row of the button "Area 5"
    grid_layout = get_grid_layout(vessqc)
    index3 = None
    n = grid_layout.rowCount()
    for i in range(n):
        widget0 = grid_layout.itemAtPosition(i, 0).widget()
        if str(type(widget0)) == "<class 'PyQt5.QtWidgets.QPushButton'>" and \
            widget0.text() == 'Area 5':
            index3 = i
            break

    assert index3 != None       # button "Area 5" has been found
    assert index3 < index2      # "Area 5" is now in the upper part of the list
    assert index3 == index1
    assert widget0.isEnabled() == True     # the button "Area 5" is active

    # check the label of the right button
    widget3 = grid_layout.itemAtPosition(index3, 3).widget()
    assert str(type(widget3)) == "<class 'PyQt5.QtWidgets.QPushButton'>"
    assert widget3.text() == 'done'


# tmp_path is a pytest fixture
@pytest.mark.save
def test_save(tmp_path, vessqc, segmentation_data, uncertainty_data):
    # (27.09.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    vessqc.parent = tmp_path
    vessqc.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, segmentation_data)

    filename = tmp_path / '_Uncertainty.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, uncertainty_data)


@pytest.mark.save_with_exc
def test_save_with_exc(tmp_path, vessqc, segmentation_data,
    uncertainty_data):
    # (27.09.2024)
    vessqc.segmentation = segmentation_data
    vessqc.uncertainty = uncertainty_data
    vessqc.parent = tmp_path

    # simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=OSError("File error")):
        vessqc.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    assert not filename.exists()

    filename = tmp_path / '_Uncertainty.npy'
    assert not filename.exists()


@pytest.mark.reload
def test_reload(tmp_path, vessqc, segmentation_data, uncertainty_data):
    # (01.10.2024)
    vessqc.parent = tmp_path
    vessqc.areas = [None]

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

    # test vessqc.areas
    assert len(vessqc.areas) == 10
    assert vessqc.areas[1]['name'] == 'Area 1'
    assert vessqc.areas[2]['uncertainty'] == np.float32(0.2)
    assert vessqc.areas[3]['counts'] == 34
    assert vessqc.areas[4]['centroid'] == None
    assert vessqc.areas[5]['where'] == None
    assert vessqc.areas[6]['done'] == False

    # test the content of the Napari layers
    assert len(vessqc.viewer.layers) == 2
    layer0 = vessqc.viewer.layers[0]
    layer1 = vessqc.viewer.layers[1]
    assert layer0.name == 'Segmentation'
    assert layer1.name == 'Uncertainty'
    np.testing.assert_array_equal(layer0.data, segmentation_data)
    np.testing.assert_array_equal(layer1.data, uncertainty_data)


@pytest.mark.reload_with_exc
def test_reload_with_exc(tmp_path, vessqc):
    # (01.10.2024)
    vessqc.parent = tmp_path
    vessqc.areas = [None]

    # simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=OSError("File error")):
        vessqc.reload()

    assert len(vessqc.viewer.layers) == 0
    assert vessqc.areas == [None]


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
