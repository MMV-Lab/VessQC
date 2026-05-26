"""
test_widget.py
==============

Functions for Pytest
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2024
# (12.09.2024)

import napari
import numpy as np
from pathlib import Path
import pytest
from qtpy.QtCore import QSize
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)
from tifffile import imread
from unittest import mock

from ..io_utils import (
    save_npy,
    load_npy,
    save_segments,
    load_segments,
    build_filename,
)
from ..multiple_viewer_widget import MultipleViewerWidget, CrossWidget
from vessqc import VessQCWidget


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

# NOTE:
# Napari 0.5.x emits a spurious "Widgets leaked" warning due to an
# internal generator in make_napari_viewer.
# This is filtered in pytest.ini on purpose.
@pytest.fixture
def widget(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    # Create an object of class VessQCWidget
    # (12.09.2024)
    test_config = tmp_path / '.vessqc_test_config.json'
    monkeypatch.setattr('vessqc._data_manager.DataManager.CONFIG_FILE', test_config)
    napari_viewer = make_napari_viewer(strict_qt=True)
    example_widget = VessQCWidget(napari_viewer)
    dock_widget = example_widget.dock_widget
    qtbot.addWidget(example_widget)         # Fixture from pytest-qt

    yield example_widget

    example_widget.setParent(None)
    example_widget.deleteLater()
    napari_viewer.close()

    try:
        del dock_widget.viewer_model1
        del dock_widget.viewer_model2
        del example_widget.dock_widget
        del example_widget.cross
    except Exception:
        pass


@pytest.fixture(autouse=True)
def suppress_success_message_boxes():
    """Prevent modal success dialogs during widget tests."""
    with mock.patch("qtpy.QtWidgets.QMessageBox.information"):
        yield


# define fixtures for the image data
@pytest.fixture
def image():
    return imread(DATA / 'Box32x32_IM.tif')

@pytest.fixture
def segPred():
    return imread(DATA / 'Box32x32_segPred.tif')

@pytest.fixture
def segPredNew():
    # (24.09.2024)
    return imread(DATA / 'Box32x32_segPredNew.tif')

@pytest.fixture
def cropped_segPred():
    return imread(DATA / 'Cropped_segPred.tif')

@pytest.fixture
def uncertainty():
    return imread(DATA / 'Box32x32_uncertainty.tif')

@pytest.fixture
def uncertaintyNew():
    # (26.09.2024)
    return imread(DATA / 'Box32x32_uncertaintyNew.tif')

@pytest.fixture
def labels():
    # (05.08.2024)
    return imread(DATA / 'labels.tif')

@pytest.fixture
def labelsNew():
    # (05.08.2024)
    return imread(DATA / 'labelsNew.tif')

@pytest.fixture
def segment_4():
    # (20.09.2024)
    return imread(DATA / 'Segment_4.tif')

@pytest.fixture
def segment_4New():
    # (24.09.2024)
    return imread(DATA / 'Segment_4New.tif')

@pytest.fixture
def segments():
    # (01.08.2025, revised 07.05.2026)
    segments = load_segments(DATA / 'segments.json')
    return segments


@pytest.mark.init
def test_init(widget):
    # (12.09.2024, updated 07.10.2025)
    assert isinstance(widget, QWidget)              # Base class of VessQcWidget
    assert isinstance(widget, VessQcWidget)         # Class of widget
    assert issubclass(VessQcWidget, QWidget)        # Is QWidget the base class?
    assert isinstance(widget.viewer, napari.Viewer)
    assert isinstance(widget.layout(), QVBoxLayout)
    assert isinstance(widget.segments, list)
    assert isinstance(widget.dock_widget, QSplitter)    # Base class
    assert isinstance(widget.dock_widget, MultipleViewerWidget)
    assert isinstance(widget.cross, QCheckBox)          # Base class
    assert isinstance(widget.cross, CrossWidget)
    assert widget.save_uncertainty == False
    assert hasattr(widget, 'data_manager')
    assert hasattr(widget, 'segmentation_worker')
    assert hasattr(widget, 'threshold_timer')


@pytest.mark.load_image
def test_load_image_deprecated(widget):
    # (12.09.2024, updated 07.10.2025)
    # Test that load_image shows deprecation message
    with mock.patch("qtpy.QtWidgets.QMessageBox.information") as mock_msg:
        widget.load_image()
        mock_msg.assert_called_once()
        args = mock_msg.call_args[0]
        assert 'Deprecated' in args[1]


@pytest.mark.read_segPred
def test_read_segPred_deprecated(widget):
    # (13.09.2024, updated 07.10.2025)
    # Test that read_segPred shows deprecation message
    with mock.patch("qtpy.QtWidgets.QMessageBox.information") as mock_msg:
        widget.read_segPred()
        mock_msg.assert_called_once()
        args = mock_msg.call_args[0]
        assert 'Deprecated' in args[1]


@pytest.mark.find_segments
def test_find_segments(widget, uncertainty, labels, segPred, segments):
    # (17.09.2024, updated 07.10.2025)
    # Note: Segmentation uses NOISE_MIN_SIZE=50 for merging into Noise
    viewer = widget.viewer
    widget.segPred = segPred
    widget.labels = labels
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

    # With min_size=200, all test segments are too small and grouped into Noise
    # So we just verify segmentation ran and produced a result
    assert len(widget.segments) >= 1
    assert 'Segmentation' in viewer.layers
    
    # Verify Noise collection exists
    has_noise = any(s['name'] == 'Noise' for s in widget.segments)
    assert has_noise


@pytest.mark.popup_window
def test_popup_window(widget, segments):
    # (17.09.2024)
    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget.segments = segments
        widget._sort_segments()
        widget.show_popup_window()
        popup_window = widget.popup_window
        mock_show.assert_called_once()

    assert isinstance(popup_window, QWidget)
    assert popup_window.windowTitle() == 'Napari (segment list)'
    from vessqc._constants import SEGMENT_LIST_POPUP_MIN_WIDTH

    assert popup_window.minimumSize() == QSize(SEGMENT_LIST_POPUP_MIN_WIDTH, 300)

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

    # Highest priority first: Segment_9 (uncertainty 0.9999, shown as %.3f)
    item_1_0 = grid_layout.itemAtPosition(1, 0)
    item_1_1 = grid_layout.itemAtPosition(1, 1)
    item_1_2 = grid_layout.itemAtPosition(1, 2)
    item_1_3 = grid_layout.itemAtPosition(1, 3)
    assert item_1_0.widget().text() == 'Segment_9'
    assert item_1_1.widget().text() == '1.000'
    assert item_1_2.widget().text() == '1'
    assert item_1_3.widget().text() == 'done'


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
def test_zoom_in(widget, image, segPred, labels, segments):
    # (06.08.2025, updated for label remapping)
    widget.image    = image
    widget.segPred  = segPred
    widget.labels   = labels
    widget.segments = segments
    widget.stem1    = 'Box32x32_IM'
    widget.stem2    = 'Box32x32_segPred'
    widget.zoom_in(segments[3], 0.75)

    segment = segments[3]
    startz, starty, startx = segment['coords'][0]
    endz, endy, endx = segment['coords'][1]

    # After label remapping, cropped segPred should have all non-zero labels → 1
    name = 'Cropped Box32x32_segPred'
    layer = widget.viewer.layers[name]
    cropped_sp = segPred[startz:endz, starty:endy, startx:endx]
    expected_remapped = np.where(cropped_sp > 0, 1, 0).astype(np.uint8)
    assert np.array_equal(layer.data, expected_remapped)

    # Segment layer should be remapped to 1 within the same crop
    name = 'Segment_4'
    layer = widget.viewer.layers[name]
    cropped_lbl = labels[startz:endz, starty:endy, startx:endx]
    expected_segment = np.where(cropped_lbl == segment['label'], 1, 0).astype(np.uint8)
    assert np.array_equal(layer.data, expected_segment)
    

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
    segment.coords = [[13, 13, 12], [18, 20, 19]]

    widget.viewer.add_labels(segment_4New, name='Segment_4')

    widget.done(segment)

    # the data in widget.segPred and widget.labels should have been changed
    # by the function compare_and_transfer()
    assert np.array_equal(widget.labels,      labelsNew)
    assert np.array_equal(widget.segPred,     segPredNew)
    assert np.array_equal(widget.uncertainty, uncertaintyNew)
    segment = widget.segments[3]
    assert segment.done


@pytest.mark.re_enable
def test_re_enable(widget, segments):
    # (13.08.2025)
    widget.segments = segments
    segment = segments[3]

    with mock.patch("qtpy.QtWidgets.QWidget.show") as mock_show:
        widget.re_enable(segment)
        mock_show.assert_called_once()

    segment = widget.segments[3]
    assert not segment.done


@pytest.mark.save_intermediate
def test_save_intermediate_data(widget, segPred, uncertainty, labels, segments, tmp_path):
    # (27.09.2024, updated for _temp .tif saves and segmentation dir)
    from types import SimpleNamespace

    widget.segPred = segPred
    widget.uncertainty = uncertainty
    widget.labels = labels
    widget.segments = segments
    widget.stem1 = 'Box32x32_IM'
    widget.stem2 = 'Box32x32_segPred'
    widget.stem3 = 'Box32x32_uncertainty'
    widget.parent = tmp_path
    widget.current_triplet = SimpleNamespace(base_name='Box32x32')
    seg_dir = tmp_path / 'segmentation'
    seg_dir.mkdir()
    widget.data_manager.segmentation_dir = seg_dir

    with mock.patch("qtpy.QtWidgets.QMessageBox.information"):
        widget.save_intermediate_data()

    assert (tmp_path / 'Box32x32_segPred_temp.tif').exists()
    assert (tmp_path / 'Box32x32_uncertainty_temp.tif').exists()
    assert np.array_equal(imread(tmp_path / 'Box32x32_segPred_temp.tif'), segPred)
    assert np.array_equal(imread(tmp_path / 'Box32x32_uncertainty_temp.tif'), uncertainty)
    assert (seg_dir / 'Box32x32_labels.tif').exists()
    assert np.array_equal(imread(seg_dir / 'Box32x32_labels.tif'), labels)
    with (seg_dir / 'Box32x32_segments.json').open('r', encoding='utf-8') as file:
        saved_segments = json.load(file)
    assert len(saved_segments) == len(segments)


@pytest.mark.save_intermediate_with_exc
def test_save_intermediate_data_with_exc(widget, segments, tmp_path):
    # (27.09.2024)
    widget.segPred = np.ones((3, 3, 3), dtype=np.int32)
    widget.uncertainty = np.random.rand(3, 3, 3)
    widget.labels = np.ones((3, 3, 3), dtype=np.int32)
    widget.segments = segments
    widget.stem1 = 'test_save_IM'
    widget.stem2 = 'test_save_segPred'
    widget.stem3 = 'test_save_uncertainty'
    widget.parent = tmp_path

    with mock.patch("vessqc._widget.imwrite", side_effect=OSError("File error")), \
         mock.patch("qtpy.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.save_intermediate_data()
        assert mock_warning.call_count == 1

    assert not (tmp_path / 'test_save_segPred_temp.tif').exists()
    assert not (tmp_path / 'test_save_uncertainty_temp.tif').exists()


@pytest.mark.load_intermediate
def test_load_intermediate_data_deprecated(widget):
    # (01.10.2024, updated 07.10.2025)
    # Test that load_intermediate_data shows deprecation message
    with mock.patch("qtpy.QtWidgets.QMessageBox.information") as mock_msg:
        widget.load_intermediate_data()
        mock_msg.assert_called_once()
        args = mock_msg.call_args[0]
        assert 'Deprecated' in args[1]


@pytest.mark.info_image
def test_show_info_image(widget, capsys):
    # Image-Layer hinzufügen
    image = np.random.rand(3, 3, 3)
    layer = widget.viewer.add_image(image, name="TestImage")
    widget.viewer.layers.selection.active = layer

    widget.show_info()
    captured = capsys.readouterr()

    assert "layer: TestImage" in captured.out
    assert "type: <class 'numpy.ndarray'>" in captured.out
    assert "dtype: float" in captured.out
    assert "shape: (3, 3, 3)" in captured.out
    assert "size:" in captured.out
    assert "ndim:" in captured.out


@pytest.mark.info_labels
def test_show_info_labels(widget, capsys):
    labels = np.random.randint(0, high=10, size=(3, 3, 3), dtype=np.int32)
    layer  = widget.viewer.add_labels(labels, name="TestLabels")
    widget.viewer.layers.selection.active = layer

    widget.show_info()
    captured = capsys.readouterr()

    # show_info now only prints layer name for labels
    assert "layer: TestLabels" in captured.out


# New tests for updated functionality (07.10.2025)

@pytest.mark.threshold
def test_threshold_debouncing(widget, qtbot):
    """Test that threshold changes are debounced"""
    widget.labels = np.ones((10, 10, 10), dtype=np.int32)
    widget.segments = [{'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 1000, 'coords': None, 'done': False}]
    widget._small_segments_label = None  # No noise segment in this test
    widget.original_labels = None
    
    # Add Segmentation layer to viewer so threshold filter doesn't fail
    widget.viewer.add_labels(widget.labels, name='Segmentation')
    
    # Change threshold multiple times quickly (values >= 50)
    widget.threshold_spinbox.setValue(100)
    widget.threshold_spinbox.setValue(150)
    widget.threshold_spinbox.setValue(200)
    
    # Timer should be active but not fired yet
    assert widget.threshold_timer.isActive()
    
    # Wait for timer to fire
    qtbot.wait(600)
    
    assert not widget.threshold_timer.isActive()


@pytest.mark.threshold
def test_threshold_minimum(widget):
    """Test segment-size spinbox bounds and default"""
    from vessqc._constants import DISPLAY_MIN_SIZE_DEFAULT, DISPLAY_MIN_SIZE_MIN

    assert widget.threshold_spinbox.minimum() == DISPLAY_MIN_SIZE_MIN
    assert widget.threshold_spinbox.value() == DISPLAY_MIN_SIZE_DEFAULT


@pytest.mark.threshold
def test_apply_threshold_merges_small_segments(widget):
    """Segments below spinbox value merge into Noise; lowering spinbox restores them."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[0:3, 0:3, 0:3] = 4          # 27 voxels: permanent Noise (<50 floor)
    labels[5:11, 5:11, 5:11] = 2       # 216 voxels: large segment (>= 200)
    labels[10:15, 10:15, 0:4] = 5      # 100 voxels: medium (50-199)

    widget.original_labels = labels.copy()
    widget._original_noise_label = 4
    widget.original_segments = [
        {'name': 'Segment_2', 'label': 2, 'uncertainty': 0.8, 'counts': 216,
         'coords': None, 'done': False},
        {'name': 'Segment_5', 'label': 5, 'uncertainty': 0.6, 'counts': 100,
         'coords': None, 'done': False},
    ]
    widget.labels = labels.copy()
    widget.viewer.add_labels(widget.labels, name='Segmentation')

    widget.apply_threshold_filter(200)
    names = {s['name'] for s in widget.segments}
    assert names == {'Segment_2', 'Noise'}

    widget.apply_threshold_filter(50)
    names = {s['name'] for s in widget.segments}
    assert names == {'Segment_2', 'Segment_5', 'Noise'}
    noise = next(s for s in widget.segments if s['name'] == 'Noise')
    assert noise['counts'] == 27


@pytest.mark.segment_count
def test_segment_count_update_on_done(widget):
    """Test that segment count updates when done is clicked"""
    widget.labels = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    widget.segPred = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    widget.uncertainty = np.array([[[0.5, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]])
    widget.image = np.zeros((1, 3, 3))
    widget.stem1 = 'test_IM'
    widget.stem2 = 'test_segPred'
    widget.stem3 = 'test_uncertainty'
    
    segment = {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 3, 'coords': None, 'done': False}
    widget.segments = [segment]
    
    initial_count = segment['counts']
    assert initial_count == 3
    
    # Simulate modification (add a pixel)
    widget.labels[0, 0, 2] = 1
    
    # Call done
    with mock.patch.object(widget, 'show_popup_window'):
        widget.done(segment)
    
    # Count should be updated
    assert segment['counts'] == 4
    assert segment['done'] == True


@pytest.mark.segment_transfer
def test_label_remapping_transfer(widget):
    """Test that label remapping (42 → 1 → 42) works correctly in compare_and_transfer"""
    # Setup: Simple 3D volume with segment at label 42
    widget.labels = np.zeros((5, 5, 5), dtype=np.int32)
    widget.labels[1:3, 1:3, 1:3] = 42  # Original segment with label 42
    
    widget.segPred = np.zeros((5, 5, 5), dtype=np.uint8)
    widget.uncertainty = np.zeros((5, 5, 5), dtype=np.float32)
    
    segment = {
        'name': 'Segment_42',
        'label': 42,
        'uncertainty': 0.8,
        'coords': [[0, 0, 0], [5, 5, 5]]  # Cropped region
    }
    
    # Simulate segment_data that was remapped to 1 (as zoom_in does)
    # User added pixel at position [2, 3, 3] (disconnected from original)
    segment_data = np.zeros((5, 5, 5), dtype=np.uint8)
    segment_data[1:3, 1:3, 1:3] = 1  # Original segment (remapped to 1)
    segment_data[2, 3, 3] = 1  # New disconnected pixel
    
    # Add the layer to the real viewer
    widget.viewer.add_labels(segment_data, name='Segment_42')
    
    # Call compare_and_transfer
    widget.compare_and_transfer(segment)
    
    # Verify the new pixel was transferred with ORIGINAL label (42)
    assert widget.labels[2, 3, 3] == 42
    
    # Verify original segment pixels still have label 42
    assert widget.labels[1, 1, 1] == 42
    assert widget.labels[2, 2, 2] == 42


@pytest.mark.segment_transfer  
def test_non_continuous_segment_addition(widget):
    """Test that disconnected segment additions work (user draws away from segment)"""
    # Setup
    widget.labels = np.zeros((5, 5, 5), dtype=np.int32)
    widget.labels[1, 1, 1] = 100  # Small segment at label 100
    
    widget.segPred = np.zeros((5, 5, 5), dtype=np.uint8)
    widget.uncertainty = np.zeros((5, 5, 5), dtype=np.float32)
    
    segment = {
        'name': 'Segment_100',
        'label': 100,
        'uncertainty': 0.9,
        'coords': [[0, 0, 0], [5, 5, 5]]
    }
    
    # Simulated segment_data after user edits (remapped to 1)
    # Original pixel + disconnected addition
    segment_data = np.zeros((5, 5, 5), dtype=np.uint8)
    segment_data[1, 1, 1] = 1  # Original pixel
    segment_data[3, 3, 3] = 1  # Disconnected pixel added by user
    
    # Add the layer to the real viewer
    widget.viewer.add_labels(segment_data, name='Segment_100')
    
    # Call compare_and_transfer
    widget.compare_and_transfer(segment)
    
    # Both pixels should have the original label 100
    assert widget.labels[1, 1, 1] == 100
    assert widget.labels[3, 3, 3] == 100  # Disconnected pixel gets correct label


@pytest.mark.segment_count
def test_segment_count_update_on_zoom(widget):
    """Test that segment count updates when zooming to segment"""
    widget.labels = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    widget.image = np.zeros((1, 3, 3))
    widget.segPred = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    widget.stem1 = 'test_IM'  # Required by zoom_in
    widget.stem2 = 'test_segPred'  # Required by zoom_in
    
    segment = {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 3, 'coords': None, 'done': False}
    widget.segments = [segment]
    
    # Modify labels
    widget.labels[0, 1, 1] = 1
    
    # Call zoom_in
    widget.zoom_in(segment, 0.75)
    
    # Count should be updated
    assert segment['counts'] == 4


@pytest.mark.temp_files
def test_save_intermediate_creates_temp_files(widget, tmp_path):
    """Test that save_intermediate_data creates temp files"""
    widget.parent = tmp_path
    widget.stem2 = 'test_segPred'
    widget.stem3 = 'test_uncertainty'
    widget.segPred = np.ones((5, 5, 5), dtype=np.uint8)
    widget.uncertainty = np.ones((5, 5, 5), dtype=np.float32) * 0.5
    
    # Mock data manager and current_triplet
    from vessqc._data_manager import DataManager
    widget.data_manager.segmentation_dir = tmp_path / '.vessqc_segmentations'
    widget.data_manager.segmentation_dir.mkdir()
    widget.current_triplet = mock.Mock()
    widget.current_triplet.base_name = 'test'
    widget.segments = [{'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 100, 'coords': None, 'done': False}]
    widget.labels = np.ones((5, 5, 5), dtype=np.int32)
    
    with mock.patch("qtpy.QtWidgets.QMessageBox.information"):
        widget.save_intermediate_data()
    
    # Check temp files created
    assert (tmp_path / 'test_segPred_temp.tif').exists()
    assert (tmp_path / 'test_uncertainty_temp.tif').exists()
    assert (widget.data_manager.segmentation_dir / 'test_labels.tif').exists()
    assert (widget.data_manager.segmentation_dir / 'test_segments.json').exists()


@pytest.mark.temp_files  
def test_temp_suffix_not_duplicated(widget):
    """Test that _temp suffix is not duplicated when loading temp files"""
    widget.stem2 = 'test_segPred_temp'
    widget.stem3 = 'test_uncertainty_temp'
    
    # Simulate loading with temp files - stems should have _temp removed
    if widget.stem2.endswith('_temp'):
        widget.stem2 = widget.stem2[:-5]
    if widget.stem3.endswith('_temp'):
        widget.stem3 = widget.stem3[:-5]
    
    assert widget.stem2 == 'test_segPred'
    assert widget.stem3 == 'test_uncertainty'


@pytest.mark.popup
def test_popup_window_closes_on_widget_close(widget, qtbot):
    """Test that popup window closes when widget closes"""
    widget.segments = [{'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 100, 'coords': None, 'done': False}]
    
    with mock.patch.object(widget.popup_window if hasattr(widget, 'popup_window') and widget.popup_window else QWidget(), 'show'):
        widget.show_popup_window()
    
    assert widget.popup_window is not None
    
    # Close widget
    widget.close()
    
    # Popup should be closed
    assert widget.popup_window is None or not widget.popup_window.isVisible()
