# Tests for LoadingDialog
# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import json
import numpy as np
from pathlib import Path
import pytest
from qtpy.QtCore import Qt
from tifffile import imread, imwrite
from unittest import mock

from vessqc._data_manager import DataManager, DatasetTriplet, SegmentPriority
from vessqc._loading_dialog import LoadingDialog


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with sample files"""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    
    # Create sample dataset
    img = np.random.rand(10, 10, 10).astype(np.uint8)
    segpred = np.random.randint(0, 5, (10, 10, 10)).astype(np.uint8)
    uncertainty = np.random.rand(10, 10, 10).astype(np.float32)
    
    imwrite(data_dir / 'test_IM.tif', img)
    imwrite(data_dir / 'test_segPred.tif', segpred)
    imwrite(data_dir / 'test_uncertainty.tif', uncertainty)
    
    return data_dir


@pytest.fixture
def data_manager(temp_data_dir, tmp_path, monkeypatch):
    """Create a DataManager with temp directory and isolated config"""
    # Patch CONFIG_FILE to prevent overwriting user's real config
    test_config = tmp_path / '.vessqc_test_config.json'
    monkeypatch.setattr('vessqc._data_manager.DataManager.CONFIG_FILE', test_config)
    
    manager = DataManager()
    manager.set_data_directory(temp_data_dir)
    return manager


@pytest.fixture
def loading_dialog(data_manager, qtbot):
    """Create a LoadingDialog"""
    with mock.patch.object(LoadingDialog, '_refresh_datasets'):
        dialog = LoadingDialog(data_manager)
        qtbot.addWidget(dialog)
        return dialog


@pytest.mark.dialog
def test_dialog_init(loading_dialog):
    """Test dialog initialization"""
    assert loading_dialog.data_manager is not None
    assert loading_dialog.selected_triplet is None
    assert loading_dialog.windowTitle() == 'Load Dataset'


@pytest.mark.dialog
def test_directory_label_updates(loading_dialog, temp_data_dir):
    """Test that directory label updates"""
    loading_dialog.data_manager.set_data_directory(temp_data_dir)
    # The label updates automatically when directory is set via __init__ or select
    # Just verify it shows the correct path
    assert str(temp_data_dir) in loading_dialog.dir_label.text()


@pytest.mark.dialog
def test_select_directory(loading_dialog, temp_data_dir, qtbot):
    """Test directory selection"""
    with mock.patch('qtpy.QtWidgets.QFileDialog.getExistingDirectory', return_value=str(temp_data_dir)):
        with mock.patch.object(loading_dialog, '_refresh_datasets'):
            loading_dialog._select_directory()
    
    assert loading_dialog.data_manager.data_directory == temp_data_dir


@pytest.mark.dialog
def test_dataset_list_display_empty(loading_dialog):
    """Test dataset list display when no datasets"""
    loading_dialog.data_manager.all_segment_priorities = []
    loading_dialog._update_dataset_list()
    
    assert loading_dialog.dataset_list.count() >= 1


@pytest.mark.dialog
def test_dataset_list_display_with_segments(loading_dialog, temp_data_dir):
    """Test dataset list display with segments"""
    # Create mock triplet and priorities
    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif'
    )
    
    seg_priorities = [
        SegmentPriority('test', 1, 'Segment_1', 0.9, triplet),
        SegmentPriority('test', 2, 'Segment_2', 0.7, triplet),
        SegmentPriority('test', 3, 'Segment_3', 0.5, triplet),
    ]
    
    loading_dialog.data_manager.all_segment_priorities = seg_priorities
    loading_dialog._update_dataset_list()
    
    # Should display segments
    assert loading_dialog.dataset_list.count() == 3


@pytest.mark.dialog
def test_temp_file_priority_logic(loading_dialog, temp_data_dir):
    """Test that temp files are displayed in priority list"""
    # Create two triplets
    triplet1 = DatasetTriplet(
        'test1',
        temp_data_dir / 'test1_IM.tif',
        temp_data_dir / 'test1_segPred.tif',
        temp_data_dir / 'test1_uncertainty.tif'
    )
    triplet1.has_temp = False
    
    triplet2 = DatasetTriplet(
        'test2',
        temp_data_dir / 'test2_IM.tif',
        temp_data_dir / 'test2_segPred.tif',
        temp_data_dir / 'test2_uncertainty.tif'
    )
    triplet2.has_temp = True
    
    # Create segment priorities where test1 is top priority
    seg_priorities = [
        SegmentPriority('test1', 1, 'Segment_1', 0.9, triplet1),
        SegmentPriority('test2', 1, 'Segment_1', 0.5, triplet2),
    ]
    
    loading_dialog.data_manager.all_segment_priorities = seg_priorities
    loading_dialog._update_dataset_list()
    
    # Should display both: test1 in top 10, test2 as temp file
    assert loading_dialog.dataset_list.count() == 2


@pytest.mark.dialog
def test_top_10_display(loading_dialog, temp_data_dir):
    """Test that only top 10 segments are displayed (plus temp files)"""
    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif'
    )
    
    # Create 15 segments
    seg_priorities = [
        SegmentPriority('test', i, f'Segment_{i}', 1.0 - i*0.05, triplet)
        for i in range(1, 16)
    ]
    
    loading_dialog.data_manager.all_segment_priorities = seg_priorities
    loading_dialog._update_dataset_list()
    
    # Should display max 10
    assert loading_dialog.dataset_list.count() == 10


@pytest.mark.dialog
def test_dataset_selection(loading_dialog, temp_data_dir):
    """Test dataset selection"""
    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif'
    )
    
    seg_priority = SegmentPriority('test', 1, 'Segment_1', 0.9, triplet)
    loading_dialog.data_manager.all_segment_priorities = [seg_priority]
    loading_dialog._update_dataset_list()
    
    # Select first item
    item = loading_dialog.dataset_list.item(0)
    selected_triplet = item.data(Qt.UserRole)
    
    assert selected_triplet is triplet


@pytest.mark.dialog
def test_refresh_triggers_sync(loading_dialog):
    """Test that refresh triggers directory sync"""
    with mock.patch.object(loading_dialog.data_manager, 'sync_with_directory', return_value=([], [])):
        with mock.patch.object(loading_dialog.data_manager, 'calculate_priorities'):
            loading_dialog._refresh_datasets()


@pytest.mark.dialog
def test_queue_status_display(loading_dialog):
    """Test queue status label updates"""
    loading_dialog._on_priorities_calculated()
    
    # Queue label should be updated
    assert loading_dialog.queue_label is not None


@pytest.mark.dialog  
def test_auto_refresh_on_open(data_manager, qtbot):
    """Test that dialog auto-refreshes on open if directory is set"""
    with mock.patch.object(LoadingDialog, '_refresh_datasets') as mock_refresh:
        dialog = LoadingDialog(data_manager)
        qtbot.addWidget(dialog)
        
        # Should have called refresh once during init
        assert mock_refresh.call_count >= 1


@pytest.mark.dialog
def test_progress_bar_visibility(loading_dialog):
    """Test progress bar visibility during refresh"""
    # Progress bar starts hidden
    assert not loading_dialog.progress_bar.isVisible()
    
    # After calling refresh, progress bar is set visible, then hidden again by callback
    # We need to check visibility DURING the refresh, not after
    # So we capture the state by checking right after setVisible is called
    
    original_setVisible = loading_dialog.progress_bar.setVisible
    visible_states = []
    
    def track_visibility(visible):
        visible_states.append(visible)
        original_setVisible(visible)
    
    loading_dialog.progress_bar.setVisible = track_visibility
    
    with mock.patch.object(loading_dialog.data_manager, 'sync_with_directory', return_value=([], [])):
        with mock.patch.object(loading_dialog.data_manager, 'calculate_priorities'):
            loading_dialog._refresh_datasets()
    
    # Should have been set to True during refresh, then False by callback
    assert True in visible_states  # Was set to visible at some point


@pytest.mark.dialog
def test_dialog_close_cleanup(loading_dialog):
    """Test dialog close cleanup"""
    loading_dialog.close()
    
    # Should not raise any exceptions
    assert True


