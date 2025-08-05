# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (12.09.2024)

import builtins
# from deepdiff import DeepDiff
import json
import napari
import numpy as np
from pathlib import Path
import pytest
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
from tifffile import imread, imwrite
from unittest import mock
from vessqc import ExampleQWidget

# A constant with the _data path
DATA = Path(__file__).parent / '_data'


def normalize_for_json(data):
    # Suggestion from ChatGPT
    import numpy as np
    if isinstance(data, dict):
        return {k: normalize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_for_json(v) for v in data]
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed in your
# testing environment
@pytest.fixture
def widget0(make_napari_viewer, qtbot):
    # Create an Object of class ExampleQWidget
    # (12.09.2024)
    my_widget = ExampleQWidget(make_napari_viewer())
    qtbot.addWidget(my_widget)          # Fixture from pytest-qt
    return my_widget

# define fixtures for the image data
@pytest.fixture
def box32x32_IM():
    filename = DATA / 'Box32x32_IM.tif'
    return imread(filename)

@pytest.fixture
def segPred():
    filename = DATA / 'Box32x32_segPred.tif'
    return imread(filename)

@pytest.fixture
def segPredNew():
    # (24.09.2024)
    filename = DATA / 'Box32x32_segPredNew.tif'
    return imread(filename)

@pytest.fixture
def uncertainty():
    filename = DATA / 'Box32x32_uncertainty.tif'
    return imread(filename)

@pytest.fixture
def uncertaintyNew():
    # (26.09.2024)
    filename = DATA / 'Box32x32_uncertaintyNew.tif'
    return imread(filename)

@pytest.fixture
def labels():
    # (05.08.2024)
    filename = DATA / 'labels.tif'
    return imread(filename)

@pytest.fixture
def segments():
    # (01.08.2025)
    filename = DATA / 'segments.json'
    with filename.open('r', encoding='utf-8') as file:
        segments = json.load(file)
    return segments

@pytest.fixture
def segment4_data():
    # (20.09.2024)
    filename = DATA / 'Segment4.tif'
    return imread(filename)

@pytest.fixture
def segment4_new():
    # (24.09.2024)
    filename = DATA / 'Segment4New.tif'
    return imread(filename)

@pytest.fixture
def find_segments(widget0, uncertainty):
    # (18.09.2024)
    widget0.find_segments(uncertainty)
    return widget0.segments

@pytest.mark.init
def test_init(widget0):
    # (12.09.2024)
    assert isinstance(widget0, QWidget)             # Base class of ExampleQWidget
    assert isinstance(widget0, ExampleQWidget)      # Class of widget0
    assert issubclass(ExampleQWidget, QWidget)      # Is QWidget the base class?
    assert isinstance(widget0.viewer, napari.Viewer)
    assert isinstance(widget0.layout(), QVBoxLayout)
    assert widget0.save_uncertainty == False


@pytest.mark.load_image
def test_load_image(widget0, box32x32_IM):
    # (12.09.2024)
    viewer = widget0.viewer

    with mock.patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(DATA / 'Box32x32_IM.tif', None)) as mock_open:
        widget0.load_image()
        mock_open.assert_called_once()

    assert widget0.segments == []
    assert widget0.parent == DATA
    assert widget0.stem1 == 'Box32x32_IM'
    assert np.array_equal(widget0.image, box32x32_IM)

    # Check the contents of the first Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, box32x32_IM)


@pytest.mark.read_segPred
def test_read_segPred(widget0, segPred, uncertainty):
    # (13.09.2024)
    viewer = widget0.viewer
    widget0.stem1 = 'Box32x32_IM'
    widget0.parent = DATA
    widget0.segments = []
    widget0.read_segPred()

    assert widget0.stem2 == 'Box32x32_segPred'
    assert widget0.stem3 == 'Box32x32_uncertainty'
    assert np.array_equal(widget0.segPred, segPred)
    assert np.array_equal(widget0.uncertainty, uncertainty)

    # Check the contents of the next Napari layer
    assert len(viewer.layers) == 2
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_segPred'
    assert np.array_equal(layer.data, segPred)

    assert isinstance(widget0.segments, list)


@pytest.mark.find_segments
def test_find_segments(widget0, uncertainty, labels, segments):
    # (17.09.2024)
    viewer = widget0.viewer
    widget0.find_segments(uncertainty)

    # For comparison purposes, the data must be standardized.
    actual_segments = normalize_for_json(widget0.segments)

    regenerate_reference = False
    if regenerate_reference:
        # Save the reference data as a JSON file
        filename = DATA / 'segments.json'
        with filename.open('w', encoding='utf-8') as file:
            json.dump(actual_segments, file, indent=2)
        pytest.skip('Reference data has been regenerated.')

    assert np.array_equal(widget0.labels, labels)
    assert len(widget0.segments) == 9
    assert actual_segments == segments, \
        "The current data does not match the stored JSON."

    layer = viewer.layers['Segmentation']
    assert np.array_equal(layer.data, labels)


@pytest.mark.popup_window
def test_popup_window(widget0, segments):
    # (17.09.2024)
    widget0.segments = segments

    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget0.show_popup_window()
        mock_show.assert_called_once()

    popup_window = widget0.popup_window
    assert isinstance(popup_window, QWidget)
    assert popup_window.windowTitle() == 'Napari (Segment list)'
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
    assert group_box.title() == 'List of segments:'

    grid_layout = group_box.layout()
    assert isinstance(grid_layout, QGridLayout)
    assert grid_layout.rowCount() == 12
    assert grid_layout.columnCount() == 4

    item_0 = grid_layout.itemAtPosition(5, 0)
    item_1 = grid_layout.itemAtPosition(5, 1)
    item_2 = grid_layout.itemAtPosition(5, 2)
    item_3 = grid_layout.itemAtPosition(5, 3)
    assert item_0.widget().text() == 'Segment_5'
    assert item_1.widget().text() == '0.600'
    assert item_2.widget().text() == '37'
    assert item_3.widget().text() == 'done'


@pytest.mark.new_entry
def test_new_entry(widget0, segments):
    # (18.09.2024)
    grid_layout = QGridLayout()
    widget0.new_entry(segments[2], grid_layout, 3)
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
def test_show_area(widget0, segments, segment4_data):
    # (20.09.2024)
    widget0.segments = segments

    # Define button1 to call widget0.show_area()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)
    button1 = QPushButton('Segment_4')
    button1.clicked.connect(widget0.show_area)
    layout.addWidget(button1)

    # Here I simulate a mouse click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    segment = segments[3]
    com = (15, 16, 15)                  # center of mass
    assert segment['com'] == com
    assert widget0.viewer.dims.current_step == com
    assert widget0.viewer.camera.center == com

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in widget0.viewer.layers):
        layer = widget0.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)
    else:
        assert False

    # 2nd click on button1
    QTest.mouseClick(button1, Qt.LeftButton)

    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in widget0.viewer.layers):
        layer = widget0.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
    else:
        assert False


@pytest.mark.transfer
def test_transfer(widget0, segPred, segPredNew, uncertainty,
    uncertaintyNew, segment4_data, segment4_new, segments):
    # (24.09.2024)
    widget0.segPred = segPred
    widget0.uncertainty = uncertainty
    segPred_layer = widget0.viewer.add_labels(widget0.segPred,
        name='Segmentation')
    widget0.segments = segments

    # Define three buttons to call widget0.show_area(), widget0.done() and
    # widget0.restore()
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    button1 = QPushButton('Segment_4')
    button1.clicked.connect(widget0.show_area)
    layout.addWidget(button1)

    button2 = QPushButton('done', objectName='Segment_4')
    button2.clicked.connect(widget0.done)
    layout.addWidget(button2)

    button3 = QPushButton('restore', objectName='Segment_4')
    button3.clicked.connect(widget0.restore)
    layout.addWidget(button3)

    # press button1 'Segment_4' to call widget0.show_area()
    QTest.mouseClick(button1, Qt.LeftButton)

    # search for the Napari layer "Segment_4" and change its data
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in widget0.viewer.layers):
        layer = widget0.viewer.layers['Segment_4']
        assert layer.name == 'Segment_4'
        assert layer.selected_label == 6
        assert np.array_equal(layer.data, segment4_data)

        # replace the data of the layer
        layer.data = segment4_new
    else:
        assert False

    # press button2 to call widget0.done()
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
        QTest.mouseClick(button2, Qt.LeftButton)

    # the data in the Napari layers Prediction and Uncertainty should have
    # been changed by the function compare_and_transfer()
    assert np.array_equal(segPred_layer.data, segPredNew)
    assert np.array_equal(widget0.segPred, segPredNew)
    assert np.array_equal(widget0.uncertainty,  uncertaintyNew)
    assert segments[3]['done'] == True

    # the Napari layer 'Segment_4' is removed
    if any(layer.name == 'Segment_4' and isinstance(layer, napari.layers.Labels)
        for layer in widget0.viewer.layers):
        assert False

    # press button3 to call widget0.restore()
    with mock.patch("qtpy.QtWidgets.QWidget.show"):
        QTest.mouseClick(button3, Qt.LeftButton)

    assert segments[3]['done'] == False


# tmp_path is a pytest fixture (see lab book from 27.09.2024)
@pytest.mark.save
def test_save(tmp_path, widget0, segPred, uncertainty):
    # (27.09.2024)
    widget0.segPred = segPred
    widget0.uncertainty  = uncertainty
    widget0.parent       = tmp_path
    widget0.btn_save()

    filename = tmp_path / '_Segmentation.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, segPred)

    filename = tmp_path / '_Uncertainty.npy'
    loaded_data = np.load(str(filename))
    np.testing.assert_array_equal(loaded_data, uncertainty)


@pytest.mark.save_with_exc
def test_save_with_exc(tmp_path, widget0):
    # (27.09.2024)
    widget0.segPred = np.ones((3, 3, 3), dtype=int)
    widget0.uncertainty  = np.ones((3, 3, 3))
    widget0.parent       = tmp_path

    # Simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        widget0.btn_save()
        assert mock_warning.call_count == 2

    filename = tmp_path / '_Segmentation.npy'
    assert not filename.exists()

    filename = tmp_path / '_Uncertainty.npy'
    assert not filename.exists()


@pytest.mark.reload
def test_reload(tmp_path, widget0, segPred, uncertainty):
    # (01.10.2024)
    widget0.parent = tmp_path
    widget0.segments = []

    # 1st: save the segPred data
    filename = tmp_path / '_Segmentation.npy'
    try:
        file = open(filename, 'wb')
        np.save(file, segPred)
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

    widget0.reload()

    # Test the content of the Napari layer and widget0 nD arrays
    assert len(widget0.viewer.layers) == 1
    layer = widget0.viewer.layers[0]
    assert layer.name == 'Segmentation'
    np.testing.assert_array_equal(layer.data,          segPred)
    np.testing.assert_array_equal(widget0.segPred, segPred)
    np.testing.assert_array_equal(widget0.uncertainty,  uncertainty)

    # Test widget0.segments
    assert len(widget0.segments) == 9
    assert widget0.segments[0]['name'] == 'Segment_1'
    assert widget0.segments[1]['label'] == 3
    assert widget0.segments[2]['uncertainty'] == np.float32(0.4)
    assert widget0.segments[3]['counts'] == 34
    assert widget0.segments[4]['com'] == None
    assert widget0.segments[5]['site'] == None
    assert widget0.segments[6]['done'] == False


@pytest.mark.reload_with_exc
def test_reload_with_exc(tmp_path, widget0):
    # (01.10.2024)
    widget0.segPred = np.ones((3, 3, 3), dtype=int)
    widget0.uncertainty  = np.ones((3, 3, 3))
    widget0.parent       = tmp_path
    widget0.segments        = []

    real_open = builtins.open       # Save original

    def open_side_effect(file, *args, **kwargs):
        # Suggestion from ChatGPT
        if '_Segmentation.npy' in str(file) or '_Uncertainty.npy' in str(file):
            raise OSError("File error")
        return real_open(file, *args, **kwargs)

    # simulate an exception when opening the file
    with mock.patch("builtins.open", side_effect=open_side_effect), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        widget0.reload()
        assert mock_warning.call_count == 1

    assert len(widget0.viewer.layers) == 0
    assert widget0.segments == []


@pytest.mark.final_segPred
def test_final_segPred(tmp_path, widget0, segPred, uncertainty):
    # (01.10.2024)
    widget0.segPred = segPred
    widget0.uncertainty = uncertainty
    widget0.parent = tmp_path
    widget0.stem1 = 'Box32x32_IM'
    widget0.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    # call the function final_segPred()
    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(filename1, None)) as mock_save:
        widget0.final_segPred()
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

    np.testing.assert_array_equal(seg_saved_data, segPred)
    np.testing.assert_array_equal(unc_saved_data, uncertainty)


@pytest.mark.final_segPred_write_error
def test_final_segPred_write_error(tmp_path, widget0):
    # (13.06.2025)
    widget0.segPred = np.ones((3, 3, 3), dtype=int)
    widget0.uncertainty  = np.ones((3, 3, 3))
    widget0.parent       = tmp_path
    widget0.stem1        = 'Box32x32_IM'
    widget0.save_uncertainty = False
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=BaseException("Segmentation error")), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:
        # Patch of the "warning" method of the "QMessageBox" class

        widget0.final_segPred()

        mock_warning.assert_called_once()
        assert 'Segmentation error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()


@pytest.mark.final_uncertainty_write_error
def test_final_uncertainty_write_error(tmp_path, widget0):
    # (13.06.2025)
    widget0.segPred = np.ones((3, 3, 3), dtype=int)
    widget0.uncertainty  = np.ones((3, 3, 3))
    widget0.parent       = tmp_path
    widget0.stem1        = 'Box32x32_IM'
    widget0.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=[None, BaseException("Uncertainty error")]), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:

        widget0.final_segPred()

        assert mock_warning.call_count == 1
        assert 'Uncertainty error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()
        assert not Path(filename2).exists()


@pytest.mark.info_image
def test_btn_info_image_layer(widget0, capsys):
    # Image-Layer hinzufügen
    image = np.random.rand(3, 3, 3)
    layer = widget0.viewer.add_image(image, name="TestImage")
    widget0.viewer.layers.selection.active = layer

    widget0.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestImage" in captured.out
    assert "type:" in captured.out
    assert "min:" in captured.out
    assert "mean:" in captured.out


@pytest.mark.info_labels
def test_btn_info_labels_layer(widget0, capsys):
    labels = np.random.randint(0, high=10, size=(3, 3, 3), dtype=int)
    layer  = widget0.viewer.add_labels(labels, name="TestLabels")
    widget0.viewer.layers.selection.active = layer

    widget0.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestLabels" in captured.out
    assert "type:" in captured.out
    assert "values:" in captured.out
    assert "counts:" in captured.out
