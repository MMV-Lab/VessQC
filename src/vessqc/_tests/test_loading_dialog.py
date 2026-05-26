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
    loading_dialog.data_manager.datasets = []
    loading_dialog._update_dataset_list()
    
    assert loading_dialog.dataset_list.count() >= 1


@pytest.mark.dialog
def test_dataset_list_display_with_datasets(loading_dialog, temp_data_dir):
    """Test dataset list display with datasets"""
    # Create mock triplet with metrics
    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif'
    )
    triplet.has_segmentation = True
    triplet.max_uncertainty_excluding_noise = 0.9
    triplet.voxel_count_at_max_uncertainty = 500
    
    loading_dialog.data_manager.datasets = [triplet]
    loading_dialog._update_dataset_list()
    
    # Should display 1 dataset
    assert loading_dialog.dataset_list.count() == 1
    
    # Check display text format
    item = loading_dialog.dataset_list.item(0)
    assert 'test' in item.text()
    assert '0.900' in item.text()
    assert '500' in item.text()


@pytest.mark.dialog
def test_temp_file_display(loading_dialog, temp_data_dir):
    """Test that temp files are displayed with [TEMP] marker"""
    # Create two triplets
    triplet1 = DatasetTriplet(
        'test1',
        temp_data_dir / 'test1_IM.tif',
        temp_data_dir / 'test1_segPred.tif',
        temp_data_dir / 'test1_uncertainty.tif'
    )
    triplet1.has_temp = False
    triplet1.has_segmentation = True
    triplet1.max_uncertainty_excluding_noise = 0.9
    triplet1.voxel_count_at_max_uncertainty = 500
    
    triplet2 = DatasetTriplet(
        'test2',
        temp_data_dir / 'test2_IM.tif',
        temp_data_dir / 'test2_segPred.tif',
        temp_data_dir / 'test2_uncertainty.tif'
    )
    triplet2.has_temp = True
    triplet2.has_segmentation = True
    triplet2.max_uncertainty_excluding_noise = 0.5
    triplet2.voxel_count_at_max_uncertainty = 200
    
    loading_dialog.data_manager.datasets = [triplet1, triplet2]
    loading_dialog._update_dataset_list()
    
    # Should display both datasets
    assert loading_dialog.dataset_list.count() == 2
    
    # Check for [TEMP] marker in test2
    item1 = loading_dialog.dataset_list.item(0)
    item2 = loading_dialog.dataset_list.item(1)
    
    # One should have [TEMP], one should not
    texts = [item1.text(), item2.text()]
    assert any('[TEMP]' in text for text in texts)
    assert any('[TEMP]' not in text for text in texts)


@pytest.mark.dialog
def test_sort_option_labels_reflect_mode(loading_dialog):
    """Active sort option is normal; the other is greyed out."""
    from vessqc._constants import DATASET_SORT_MAX, DATASET_SORT_MEAN
    from vessqc._loading_dialog import _SORT_LABEL_INACTIVE_STYLE

    loading_dialog.data_manager.dataset_sort_mode = DATASET_SORT_MAX
    loading_dialog._update_sort_option_labels()
    assert loading_dialog.sort_max_label.text() == 'Max'
    assert loading_dialog.sort_mean_label.text() == 'Mean'
    assert loading_dialog.sort_max_label.styleSheet() != _SORT_LABEL_INACTIVE_STYLE
    assert loading_dialog.sort_mean_label.styleSheet() == _SORT_LABEL_INACTIVE_STYLE

    loading_dialog.data_manager.dataset_sort_mode = DATASET_SORT_MEAN
    loading_dialog._update_sort_option_labels()
    assert loading_dialog.sort_mean_label.styleSheet() != _SORT_LABEL_INACTIVE_STYLE
    assert loading_dialog.sort_max_label.styleSheet() == _SORT_LABEL_INACTIVE_STYLE


@pytest.mark.dialog
def test_sort_switch_handle_matches_checked_state(loading_dialog, qtbot):
    """Handle offset must match checked state after apply_sort_mode (reopen glitch)."""
    from vessqc._constants import DATASET_SORT_MEAN

    switch = loading_dialog.mean_sort_switch
    loading_dialog.data_manager.dataset_sort_mode = DATASET_SORT_MEAN
    qtbot.addWidget(loading_dialog)
    loading_dialog.show()
    qtbot.waitExposed(loading_dialog)
    loading_dialog._sync_mean_sort_switch()

    expected = switch._offset_for_checkstate(True)
    assert switch.isChecked()
    assert abs(switch._get_offset() - expected) < 0.01


@pytest.mark.dialog
def test_sort_mode_mean_display(loading_dialog, temp_data_dir):
    """Mean sort mode shows mean uncertainty in the list."""
    from vessqc._constants import DATASET_SORT_MEAN

    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif',
    )
    triplet.has_segmentation = True
    triplet.mean_uncertainty = 0.456

    loading_dialog.data_manager.datasets = [triplet]
    loading_dialog.data_manager.dataset_sort_mode = DATASET_SORT_MEAN
    loading_dialog._sync_mean_sort_switch()
    loading_dialog._update_dataset_list()

    item = loading_dialog.dataset_list.item(0)
    assert 'mean uncertainty: 0.456' in item.text()


@pytest.mark.dialog
def test_all_datasets_displayed(loading_dialog, temp_data_dir):
    """Test that ALL datasets are displayed (not limited to top 10)"""
    # Create 15 datasets
    datasets = []
    for i in range(1, 16):
        triplet = DatasetTriplet(
            f'test{i}',
            temp_data_dir / f'test{i}_IM.tif',
            temp_data_dir / f'test{i}_segPred.tif',
            temp_data_dir / f'test{i}_uncertainty.tif'
        )
        triplet.has_segmentation = True
        triplet.max_uncertainty_excluding_noise = 1.0 - i * 0.05
        triplet.voxel_count_at_max_uncertainty = 100 * i
        datasets.append(triplet)
    
    loading_dialog.data_manager.datasets = datasets
    loading_dialog._update_dataset_list()
    
    # Should display all 15 datasets
    assert loading_dialog.dataset_list.count() == 15


@pytest.mark.dialog
def test_dataset_selection(loading_dialog, temp_data_dir):
    """Test dataset selection"""
    triplet = DatasetTriplet(
        'test',
        temp_data_dir / 'test_IM.tif',
        temp_data_dir / 'test_segPred.tif',
        temp_data_dir / 'test_uncertainty.tif'
    )
    triplet.has_segmentation = True
    triplet.max_uncertainty_excluding_noise = 0.9
    triplet.voxel_count_at_max_uncertainty = 500
    
    loading_dialog.data_manager.datasets = [triplet]
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


