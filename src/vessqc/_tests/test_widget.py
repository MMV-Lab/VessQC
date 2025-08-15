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
from qtpy.QtCore import QSize, Qt
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
import tempfile
from tifffile import imread, imwrite
from unittest import mock
from vessqc import ExampleQWidget

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

# Constants with the _data path and the TEMP directory
DATA = Path(__file__).parent / '_data'
tmp  = tempfile.gettempdir()
TEMP = Path(tmp)

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed in your
# testing environment
@pytest.fixture
def widget(make_napari_viewer, qtbot):
    # Create an Object of class ExampleQWidget
    # (12.09.2024)
    my_widget = ExampleQWidget(make_napari_viewer())
    qtbot.addWidget(my_widget)          # Fixture from pytest-qt
    return my_widget

# define fixtures for the image data
@pytest.fixture
def image():
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
def cropped_segPred():
    filename = DATA / 'Cropped_segPred.tif'
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
def labelsNew():
    # (05.08.2024)
    filename = DATA / 'labelsNew.tif'
    return imread(filename)

@pytest.fixture
def segment_4():
    # (20.09.2024)
    filename = DATA / 'Segment_4.tif'
    return imread(filename)

@pytest.fixture
def segment_4New():
    # (24.09.2024)
    filename = DATA / 'Segment_4New.tif'
    return imread(filename)

@pytest.fixture
def segments():
    # (01.08.2025)
    filename = DATA / 'segments.json'
    with filename.open('r', encoding='utf-8') as file:
        segments = json.load(file)
    return segments


@pytest.mark.init
def test_init(widget):
    # (12.09.2024)
    assert isinstance(widget, QWidget)              # Base class of ExampleQWidget
    assert isinstance(widget, ExampleQWidget)       # Class of widget
    assert issubclass(ExampleQWidget, QWidget)      # Is QWidget the base class?
    assert isinstance(widget.viewer, napari.Viewer)
    assert isinstance(widget.layout(), QVBoxLayout)
    assert isinstance(widget.segments, list)
    assert widget.save_uncertainty == False


@pytest.mark.load_image
def test_load_image(widget, image):
    # (12.09.2024)
    viewer = widget.viewer

    with mock.patch("qtpy.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(DATA / 'Box32x32_IM.tif', None)) as mock_open:
        widget.load_image()
        mock_open.assert_called_once()

    assert widget.segments == []
    assert widget.parent == DATA
    assert widget.stem1 == 'Box32x32_IM'
    assert np.array_equal(widget.image, image)

    # Check the contents of the first Napari layer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_IM'
    assert np.array_equal(layer.data, image)


@pytest.mark.read_segPred
def test_read_segPred(widget, segPred, uncertainty):
    # (13.09.2024)
    viewer = widget.viewer
    widget.stem1 = 'Box32x32_IM'
    widget.parent = DATA
    widget.segments = []
    widget.read_segPred()

    assert widget.stem2 == 'Box32x32_segPred'
    assert widget.stem3 == 'Box32x32_uncertainty'
    assert np.array_equal(widget.segPred, segPred)
    assert np.array_equal(widget.uncertainty, uncertainty)

    # Check the contents of the next Napari layer
    assert len(viewer.layers) == 2
    layer = viewer.layers[0]
    assert layer.name == 'Box32x32_segPred'
    assert np.array_equal(layer.data, segPred)


@pytest.mark.find_segments
def test_find_segments(widget, uncertainty, labels, segments):
    # (17.09.2024)
    viewer = widget.viewer
    widget.find_segments(uncertainty)

    # For comparison purposes, the data must be standardized.
    actual_segments = normalize_for_json(widget.segments)

    regenerate_reference = False
    if regenerate_reference:
        # Save the reference data as a JSON file
        filename = DATA / 'segments.json'
        with filename.open('w', encoding='utf-8') as file:
            json.dump(actual_segments, file, indent=2)
        pytest.skip('Reference data has been regenerated.')

    assert np.array_equal(widget.labels, labels)
    assert len(widget.segments) == 9
    assert actual_segments == segments, \
        "The current data does not match the stored JSON."

    layer = viewer.layers['Segmentation']
    assert np.array_equal(layer.data, labels)


@pytest.mark.popup_window
def test_popup_window(widget, segments):
    # (17.09.2024)
    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget.segments = segments
        widget.show_popup_window()
        popup_window = widget.popup_window
        mock_show.assert_called_once()

    assert isinstance(popup_window, QWidget)
    assert popup_window.windowTitle() == 'Napari (segment list)'
    assert popup_window.minimumSize() == QSize(350, 300)

    vbox_layout = popup_window.layout()
    assert isinstance(vbox_layout, QVBoxLayout)
    assert vbox_layout.count() == 1

    item0 = vbox_layout.itemAt(0)
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

    item_5_0 = grid_layout.itemAtPosition(5, 0)
    item_5_1 = grid_layout.itemAtPosition(5, 1)
    item_5_2 = grid_layout.itemAtPosition(5, 2)
    item_5_3 = grid_layout.itemAtPosition(5, 3)
    assert item_5_0.widget().text() == 'Segment_5'
    assert item_5_1.widget().text() == '0.600'
    assert item_5_2.widget().text() == '37'
    assert item_5_3.widget().text() == 'done'


@pytest.mark.new_entry
def test_new_entry(widget, segments):
    # (18.09.2024)
    grid_layout = QGridLayout()
    widget.new_entry(segments[2], grid_layout, 3)

    item_3_0 = grid_layout.itemAtPosition(3, 0)
    item_3_1 = grid_layout.itemAtPosition(3, 1)
    item_3_2 = grid_layout.itemAtPosition(3, 2)
    item_3_3 = grid_layout.itemAtPosition(3, 3)

    assert grid_layout.rowCount() == 4
    assert grid_layout.columnCount() == 4
    assert isinstance(item_3_0, QWidgetItem)
    assert isinstance(item_3_0.widget(), QPushButton)
    assert isinstance(item_3_1.widget(), QLabel)
    assert isinstance(item_3_2.widget(), QLabel)
    assert isinstance(item_3_3.widget(), QPushButton)

    assert item_3_0.widget().text() == 'Segment_3'
    assert item_3_1.widget().text() == '0.400'
    assert item_3_2.widget().text() == '35'
    assert item_3_3.widget().text() == 'done'


@pytest.mark.zoom_in
def test_zoom_in(widget, image, segPred, labels, cropped_segPred,
    segment_4, segments):
    # (06.08.2025)
    widget.image    = image
    widget.segPred  = segPred
    widget.labels   = labels
    widget.segments = segments
    widget.stem1    = 'Box32x32_IM'
    widget.stem2    = 'Box32x32_segPred'
    widget.zoom_in(segments[3], 0.75)

    name = 'Cropped Box32x32_segPred'
    layer = widget.viewer.layers[name]
    assert np.array_equal(layer.data, cropped_segPred)

    name = 'Segment_4'
    layer = widget.viewer.layers[name]
    assert np.array_equal(layer.data, segment_4)

    assert widget.segments[3]['coords'] == [[13, 13, 12], [18, 20, 19]]
    

@pytest.mark.done
def test_done(widget, image, segPred, segPredNew, uncertainty, uncertaintyNew,
    labels, labelsNew, segment_4New, segments):
    # (24.09.2024)
    widget.image       = image
    widget.segPred     = segPred
    widget.uncertainty = uncertainty
    widget.labels      = labels
    widget.segments    = segments
    widget.stem1       = 'Box32x32_IM'
    widget.stem2       = 'Box32x32_segPred'

    segment = segments[3]
    segment['coords'] = [[13, 13, 12], [18, 20, 19]]

    widget.viewer.add_labels(segment_4New, name='Segment_4')

    # Call the function done(segment)
    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget.done(segment)
        assert mock_show.call_count == 2

    # the data in widget.segPred and widget.labels should have been changed
    # by the function compare_and_transfer()
    assert np.array_equal(widget.labels,      labelsNew)
    assert np.array_equal(widget.segPred,     segPredNew)
    assert np.array_equal(widget.uncertainty, uncertaintyNew)
    assert segments[3]['done'] == True


@pytest.mark.restore
def test_restore(widget, segments):
    # (13.08.2025)
    widget.segments = segments
    segment = segments[3]

    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget.restore(segment)
        mock_show.assert_called_once()

    assert segments[3]['done'] == False


@pytest.mark.save_intermediate
def test_save_intermediate_data(widget, segPred, uncertainty, labels, segments):
    # (27.09.2024)
    widget.segPred      = segPred
    widget.uncertainty  = uncertainty
    widget.labels       = labels
    widget.segments     = segments
    widget.stem1        = 'Box32x32_IM'
    widget.stem2        = 'Box32x32_segPred'
    widget.stem3        = 'Box32x32_uncertainty'
    widget.save_intermediate_data()

    filename = TEMP.joinpath('Box32x32_segPred.npy')
    loaded_data = np.load(filename)
    assert np.array_equal(loaded_data, segPred)

    filename = filename.with_name('Box32x32_uncertainty.npy')
    loaded_data = np.load(filename)
    assert np.array_equal(loaded_data, uncertainty)

    filename = filename.with_name('Box32x32_labels.npy')
    loaded_data = np.load(filename)
    assert np.array_equal(loaded_data, labels)

    filename = filename.with_name('Box32x32_segments.json')
    with filename.open('r', encoding='utf-8') as file:
        loaded_data = json.load(file)
    assert loaded_data == segments


@pytest.mark.save_intermediate_with_exc
def test_save_intermediate_data_with_exc(widget, segments):
    # (27.09.2024)
    widget.segPred     = np.ones((3, 3, 3), dtype=int)
    widget.uncertainty = np.random.rand(3, 3, 3)
    widget.labels      = np.ones((3, 3, 3), dtype=int)
    widget.segments    = segments
    widget.stem1       = 'test_save_IM'
    widget.stem2       = 'test_save_segPred'
    widget.stem3       = 'test_save_uncertainty'

    # Simulate an exception when opening the file
    with mock.patch("pathlib.Path.open", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.save_intermediate_data()
        assert mock_warning.call_count == 4

    filename = TEMP.joinpath('test_save_segPred.npy')
    assert not filename.exists()

    filename = filename.with_name('test_save_uncertainty.npy')
    assert not filename.exists()

    filename = filename.with_name('test_save_labels.npy')
    assert not filename.exists()

    filename = filename.with_name('test_save_segments.json')
    assert not filename.exists()


@pytest.mark.load_intermediate
def test_load_intermediate_data(widget, image, segPred, uncertainty, labels,
    segments):
    # (01.10.2024)
    filename = TEMP.joinpath('Box32x32_segPred.npy')
    with filename.open('wb') as file:
        np.save(file, segPred)

    filename = filename.with_name('Box32x32_uncertainty.npy')
    with filename.open('wb') as file:
        np.save(file, uncertainty)

    filename = filename.with_name('Box32x32_labels.npy')
    with filename.open('wb') as file:
        np.save(file, labels)

    filename = filename.with_name('Box32x32_segments.json')
    with filename.open('w', encoding='utf-8') as file:
        json.dump(segments, file, indent=2)

    widget.image = image
    widget.stem1 = 'Box32x32_IM'
    widget.stem2 = 'Box32x32_segPred'
    widget.stem3 = 'Box32x32_uncertainty'
    widget.load_intermediate_data()

    assert np.array_equal(widget.segPred,     segPred)
    assert np.array_equal(widget.uncertainty, uncertainty)
    assert np.array_equal(widget.labels,      labels)
    assert widget.segments == segments

    # Test the content of the Napari layers
    assert len(widget.viewer.layers) == 3
    layer = widget.viewer.layers['Box32x32_IM']
    assert np.array_equal(layer.data, image)
    layer = widget.viewer.layers['Box32x32_segPred']
    assert np.array_equal(layer.data, segPred)
    layer = widget.viewer.layers['Segmentation']
    assert np.array_equal(layer.data, labels)


@pytest.mark.load_intermediate_with_exc
def test_load_intermediate_data_with_exc(widget):
    # (01.10.2024)
    widget.image       = np.random.rand(3, 3, 3)
    widget.segPred     = np.ones((3, 3, 3), dtype=int)
    widget.uncertainty = np.random.rand(3, 3, 3)
    widget.segments    = []
    widget.stem1       = 'test_save_IM'
    widget.stem2       = 'test_save_segPred'
    widget.stem3       = 'test_save_uncertainty'

    # Simulate an exception when opening the file
    with mock.patch("pathlib.Path.open", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.load_intermediate_data()
        assert mock_warning.call_count == 1

    assert len(widget.viewer.layers) == 0
    assert widget.segments == []


@pytest.mark.save_final
def test_save_final_result(widget, image, segPred, uncertainty, labels,
    segments):
    # (01.10.2024)
    widget.image            = image
    widget.segPred          = segPred
    widget.uncertainty      = uncertainty
    widget.labels           = labels
    widget.segments         = segments
    widget.parent           = TEMP
    widget.stem1            = 'Box32x32_IM'
    widget.stem2            = 'Box32x32_segPred'
    widget.save_uncertainty = True

    # call the function final_segPred()
    filename = TEMP.joinpath('Box32x32_segPredNew.tif')
    filename1 = str(filename)
    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
        return_value=(filename1, None)) as mock_save:
        widget.save_final_result()
        mock_save.assert_called_once()

    loaded_data = imread(filename)
    assert np.array_equal(loaded_data, segPred)

    filename = filename.with_name('Box32x32_uncertaintyNew.tif')
    loaded_data = imread(filename)
    assert np.array_equal(loaded_data, uncertainty)


@pytest.mark.save_final_with_exc
def test_save_final_result_with_exc(tmp_path, widget):
    # (13.06.2025)
    widget.segPred = np.ones((3, 3, 3), dtype=int)
    widget.uncertainty  = np.ones((3, 3, 3))
    widget.parent       = tmp_path
    widget.stem1        = 'Box32x32_IM'
    widget.save_uncertainty = False
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=BaseException("Segmentation error")), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:
        # Patch of the "warning" method of the "QMessageBox" class

        widget.final_segPred()

        mock_warning.assert_called_once()
        assert 'Segmentation error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()


@pytest.mark.save_uncertainty_with_exc
def test_save_uncertainty_with_exc(tmp_path, widget):
    # (13.06.2025)
    widget.segPred = np.ones((3, 3, 3), dtype=int)
    widget.uncertainty  = np.ones((3, 3, 3))
    widget.parent       = tmp_path
    widget.stem1        = 'Box32x32_IM'
    widget.save_uncertainty = True
    filename1 = str(tmp_path / 'Box32x32_segNew.tif')
    filename2 = str(tmp_path / 'Box32x32_uncNew.tif')

    with mock.patch("qtpy.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(filename1, None)), \
         mock.patch("vessqc._widget.imwrite",
            side_effect=[None, BaseException("Uncertainty error")]), \
         mock.patch.object(QMessageBox, "warning") as mock_warning:

        widget.final_segPred()

        assert mock_warning.call_count == 1
        assert 'Uncertainty error' in mock_warning.call_args[0][2]
        assert not Path(filename1).exists()
        assert not Path(filename2).exists()


@pytest.mark.info_image
def test_btn_info_image_layer(widget, capsys):
    # Image-Layer hinzufügen
    image = np.random.rand(3, 3, 3)
    layer = widget.viewer.add_image(image, name="TestImage")
    widget.viewer.layers.selection.active = layer

    widget.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestImage" in captured.out
    assert "type:" in captured.out
    assert "min:" in captured.out
    assert "mean:" in captured.out


@pytest.mark.info_labels
def test_btn_info_labels_layer(widget, capsys):
    labels = np.random.randint(0, high=10, size=(3, 3, 3), dtype=int)
    layer  = widget.viewer.add_labels(labels, name="TestLabels")
    widget.viewer.layers.selection.active = layer

    widget.btn_info()
    captured = capsys.readouterr()

    assert "layer: TestLabels" in captured.out
    assert "type:" in captured.out
    assert "values:" in captured.out
    assert "counts:" in captured.out
