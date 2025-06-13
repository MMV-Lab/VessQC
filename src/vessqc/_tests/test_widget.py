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
    QMessageBox,
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
def vqc_object(make_napari_viewer, qtbot):
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
def areas(vqc_object, uncertainty):
    # (18.09.2024)
    vqc_object.build_areas(uncertainty)
    return vqc_object.areas

@pytest.mark.init
def test_init(vqc_object):
    # (12.09.2024)
    assert isinstance(vqc_object, QWidget)          # Base class of ExampleQWidget
    assert isinstance(vqc_object, ExampleQWidget)   # Class of vqc_object
    assert issubclass(ExampleQWidget, QWidget)      # Is QWidget the base class?
    assert isinstance(vqc_object.viewer, napari.Viewer)
    assert isinstance(vqc_object.layout(), QVBoxLayout)
    assert vqc_object.save_uncertainty == False


@pytest.mark.load_image
def test_load_image(vqc_object, box32x32):
    # (12.09.2024)
    viewer = vqc_object.viewer

    with mock.patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(PARENT / 'Box32x32_IM.tif', None)) as mock_open:
        vqc_object.load_image()
        mock_open.assert_called_once()

    assert vqc_object.areas == []
    assert vqc_object.parent == PARENT
    assert vqc_object.stem1 == 'Box32x32_IM'
    assert np.array_equal(vqc_object.image, box32x32)

    # Check the contents of the first Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, box32x32)


@pytest.mark.read_segmentation
def test_read_segmentation(vqc_object, segmentation, uncertainty):
    # (13.09.2024)
    viewer = vqc_object.viewer
    vqc_object.stem1 = 'Box32x32_IM'
    vqc_object.parent = PARENT
    vqc_object.areas = []
    vqc_object.read_segmentation()

    assert np.array_equal(vqc_object.segmentation, segmentation)
    assert np.array_equal(vqc_object.uncertainty,  uncertainty)

    # Check the contents of the next Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Segmentation'
    assert np.array_equal(layer.data, segmentation)

    assert isinstance(vqc_object.areas, list)


@pytest.mark.build_areas
def test_build_areas(vqc_object, uncertainty):
    # (17.09.2024)
    vqc_object.build_areas(uncertainty)

    assert len(vqc_object.areas) == 9
    assert vqc_object.areas[0]['name'] == 'Segment_1'
    assert vqc_object.areas[1]['label'] == 3
    assert vqc_object.areas[2]['uncertainty'] == np.float32(0.4)
    assert vqc_object.areas[3]['counts'] == 34
    assert vqc_object.areas[4]['com'] == None       # center of mass
    assert vqc_object.areas[5]['site'] == None
    assert vqc_object.areas[6]['done'] == False


@pytest.mark.popup_window
def test_popup_window(vqc_object, areas):
    # (17.09.2024)
    vqc_object.areas = areas

    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        vqc_object.show_popup_window()
        mock_show.assert_called_once()

    popup_window = vqc_object.popup_window
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
def test_new_entry(vqc_object, areas):
    # (18.09.2024)
    grid_layout = QGridLayout()
    vqc_object.new_entry(areas[2], grid_layout, 3)
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
def test_show_area(vqc_object, areas, segment4_data):
    # (20.09.2024)
    vqc_object.areas = areas

    # Define button1 to call vqc_object.show_area()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)
    button1 = QPushButton('Segment_4')
    button1.clicked.connect(vqc_object.show_area)
    layout.addWidget(button1)

    # Here I simulate a mouse click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    segment = areas[3]
    com = (15, 16, 15)                  # center of mass
    assert segment['com'] == com
    assert vqc_object.viewer.dims.current_step == com
    assert vqc_object.viewer.camera.center == com

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vqc_object.viewer.layers):
        layer = vqc_object.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)
    else:
        assert False

    # 2nd click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vqc_object.viewer.layers):
        layer = vqc_object.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
    else:
        assert False


@pytest.mark.transfer
def test_transfer(vqc_object, segmentation, segmentation_new, uncertainty,
    uncertainty_new, segment4_data, segment4_new, areas):
    # (24.09.2024)
    vqc_object.segmentation = segmentation
    vqc_object.uncertainty = uncertainty
    segmentation_layer = vqc_object.viewer.add_labels(vqc_object.segmentation,
        name='Segmentation')
    vqc_object.areas = areas

    # Define three buttons to call vqc_object.show_area(), vqc_object.done() and
    # vqc_object.restore()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    button1 = QPushButton('Segment_4')
    button1.clicked.connect(vqc_object.show_area)
    layout.addWidget(button1)

    button2 = QPushButton('done', objectName='Segment_4')
    button2.clicked.connect(vqc_object.done)
    layout.addWidget(button2)

    button3 = QPushButton('restore', objectName='Segment_4')
    button3.clicked.connect(vqc_object.restore)
    layout.addWidget(button3)

    # press button1 'Segment_4' to call vqc_object.show_area()
    QTest.mouseClick(button1, Qt.LeftButton)

    # search for the Napari layer "Segment_4" and change its data
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vqc_object.viewer.layers):
        layer = vqc_object.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)

        # replace the data of the layer
        layer.data = segment4_new
    else:
        assert False

    # press button2 to call vqc_object.done()
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
        QTest.mouseClick(button2, Qt.LeftButton)

    # the data in the Napari layers Prediction and Uncertainty should have
    # been changed by the function compare_and_transfer()
    assert np.array_equal(segmentation_layer.data, segmentation_new)
    assert np.array_equal(vqc_object.segmentation, segmentation_new)
    assert np.array_equal(vqc_object.uncertainty,  uncertainty_new)
    assert areas[3]['done'] == True

    # the Napari layer 'Segment_4' is removed
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in vqc_object.viewer.layers):
        assert False

    # press button3 to call vqc_object.restore()
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
        QTest.mouseClick(button3, Qt.LeftButton)

    assert areas[3]['done'] == False


# tmp_path is a pytest fixture (see lab book from 27.09.2024)
@pytest.mark.save
def test_save(tmp_path, vqc_object, segmentation, uncertainty):
    # (27.09.2024)
    vqc_object.segmentation = segmentation
    vqc_object.uncertainty  = uncertainty
    vqc_object.parent       = tmp_path
    vqc_object.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, segmentation)

    filename = tmp_path / '_Uncertainty.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, uncertainty)


@pytest.mark.save_with_exc
def test_save_with_exc(tmp_path, vqc_object):
    # (27.09.2024)
    vqc_object.segmentation = np.ones((3, 3, 3), dtype=int)
    vqc_object.uncertainty  = np.ones((3, 3, 3))
    vqc_object.parent       = tmp_path

    # Simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        vqc_object.btn_save()
        assert mock_warning.call_count == 2

    filename = tmp_path / '_Segmentation.npy'
    assert not filename.exists()

    filename = tmp_path / '_Uncertainty.npy'
    assert not filename.exists()


@pytest.mark.reload
def test_reload(tmp_path, vqc_object, segmentation, uncertainty):
    # (01.10.2024)
    vqc_object.parent = tmp_path
    vqc_object.areas = []

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

    vqc_object.reload()

    # Test the content of the Napari layer and vqc_object nD arrays
    assert len(vqc_object.viewer.layers) == 1
    layer = vqc_object.viewer.layers[0]
    assert layer.name == 'Segmentation'
    np.testing.assert_array_equal(layer.data,          segmentation)
    np.testing.assert_array_equal(vqc_object.segmentation, segmentation)
    np.testing.assert_array_equal(vqc_object.uncertainty,  uncertainty)

    # Test vqc_object.areas
    assert len(vqc_object.areas) == 9
    assert vqc_object.areas[0]['name'] == 'Segment_1'
    assert vqc_object.areas[1]['label'] == 3
    assert vqc_object.areas[2]['uncertainty'] == np.float32(0.4)
    assert vqc_object.areas[3]['counts'] == 34
    assert vqc_object.areas[4]['com'] == None
    assert vqc_object.areas[5]['site'] == None
    assert vqc_object.areas[6]['done'] == False


@pytest.mark.reload_with_exc
def test_reload_with_exc(tmp_path, vqc_object):
    # (01.10.2024)
    vqc_object.segmentation = np.ones((3, 3, 3), dtype=int)
    vqc_object.uncertainty  = np.ones((3, 3, 3))
    vqc_object.parent       = tmp_path
    vqc_object.areas        = []

    real_open = builtins.open       # Save original

    def open_side_effect(file, *args, **kwargs):
        # Suggestion from ChatGPT
        if '_Segmentation.npy' in str(file) or '_Uncertainty.npy' in str(file):
            raise OSError("File error")
        return real_open(file, *args, **kwargs)

    # simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=open_side_effect), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        vqc_object.reload()
        assert mock_warning.call_count == 1

    assert len(vqc_object.viewer.layers) == 0
    assert vqc_object.areas == []


@pytest.mark.final_segmentation
def test_final_segmentation(tmp_path, vqc_object, segmentation, uncertainty):
    # (01.10.2024)
    vqc_object.segmentation = segmentation
    vqc_object.uncertainty = uncertainty
    vqc_object.parent = tmp_path
    vqc_object.stem1 = 'Box32x32_IM'
    vqc_object.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    # call the function final_segmentation()
    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(filename1, None)) as mock_save:
        vqc_object.final_segmentation()
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


@pytest.mark.final_segmentation_write_error
def test_final_segmentation_write_error(tmp_path, vqc_object):
    # (13.06.2025)
    vqc_object.segmentation = np.ones((3, 3, 3), dtype=int)
    vqc_object.uncertainty  = np.ones((3, 3, 3))
    vqc_object.parent       = tmp_path
    vqc_object.stem1        = 'Box32x32_IM'
    vqc_object.save_uncertainty = False
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=BaseException("Segmentation error")), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:
        # Patch of the "warning" method of the "QMessageBox" class

        vqc_object.final_segmentation()

        mock_warning.assert_called_once()
        assert 'Segmentation error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()


@pytest.mark.final_uncertainty_write_error
def test_final_uncertainty_write_error(tmp_path, vqc_object):
    # (13.06.2025)
    vqc_object.segmentation = np.ones((3, 3, 3), dtype=int)
    vqc_object.uncertainty  = np.ones((3, 3, 3))
    vqc_object.parent       = tmp_path
    vqc_object.stem1        = 'Box32x32_IM'
    vqc_object.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=[None, BaseException("Uncertainty error")]), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:

        vqc_object.final_segmentation()

        assert mock_warning.call_count == 1
        assert 'Uncertainty error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()
        assert not Path(filename2).exists()


@pytest.mark.info_image
def test_btn_info_image_layer(vqc_object, capsys):
    # Image-Layer hinzufügen
    image = np.random.rand(3, 3, 3)
    layer = vqc_object.viewer.add_image(image, name="TestImage")
    vqc_object.viewer.layers.selection.active = layer

    vqc_object.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestImage" in captured.out
    assert "type:" in captured.out
    assert "min:" in captured.out
    assert "mean:" in captured.out


@pytest.mark.info_labels
def test_btn_info_labels_layer(vqc_object, capsys):
    labels = np.random.randint(0, high=10, size=(3, 3, 3), dtype=int)
    layer  = vqc_object.viewer.add_labels(labels, name="TestLabels")
    vqc_object.viewer.layers.selection.active = layer

    vqc_object.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestLabels" in captured.out
    assert "type:" in captured.out
    assert "values:" in captured.out
    assert "counts:" in captured.out
