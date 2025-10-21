# Tests for DataManager
# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import json
import numpy as np
from pathlib import Path
import pytest
from tifffile import imread, imwrite
from unittest import mock

from vessqc._data_manager import DataManager, DatasetTriplet, SegmentPriority


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with sample files"""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    
    # Create sample dataset files
    img = np.random.rand(10, 10, 10).astype(np.uint8)
    segpred = np.random.randint(0, 5, (10, 10, 10)).astype(np.uint8)
    uncertainty = np.random.rand(10, 10, 10).astype(np.float32)
    
    imwrite(data_dir / 'test_IM.tif', img)
    imwrite(data_dir / 'test_segPred.tif', segpred)
    imwrite(data_dir / 'test_uncertainty.tif', uncertainty)
    
    return data_dir


@pytest.fixture
def data_manager(tmp_path, monkeypatch):
    """Create a DataManager instance with isolated config"""
    # Patch the CONFIG_FILE before creating DataManager
    test_config = tmp_path / '.vessqc_config.json'
    monkeypatch.setattr(DataManager, 'CONFIG_FILE', test_config)
    
    manager = DataManager()
    # Ensure clean state
    manager.data_directory = None
    manager.cache_file = None
    manager.segmentation_dir = None
    manager.datasets = []
    manager.all_segment_priorities = []
    return manager


@pytest.mark.data_manager
def test_data_manager_init(tmp_path, monkeypatch):
    """Test DataManager initialization"""
    # Use isolated config file
    test_config = tmp_path / '.vessqc_config.json'
    monkeypatch.setattr(DataManager, 'CONFIG_FILE', test_config)
    
    manager = DataManager()
    # Should start with clean state when no config exists
    assert manager.data_directory is None or isinstance(manager.data_directory, Path)
    assert manager.cache_file is None or isinstance(manager.cache_file, Path)
    assert manager.datasets == []
    assert manager.all_segment_priorities == []


@pytest.mark.data_manager
def test_set_data_directory(data_manager, temp_data_dir):
    """Test setting data directory"""
    result = data_manager.set_data_directory(temp_data_dir)
    
    assert result is True
    assert data_manager.data_directory == temp_data_dir
    assert data_manager.segmentation_dir == temp_data_dir / '.vessqc_segmentations'
    assert data_manager.segmentation_dir.exists()


@pytest.mark.data_manager
def test_detect_datasets_basic(data_manager, temp_data_dir):
    """Test basic dataset detection"""
    data_manager.set_data_directory(temp_data_dir)
    datasets = data_manager.detect_datasets()
    
    assert len(datasets) == 1
    assert datasets[0].base_name == 'test'
    assert datasets[0].raw_file.name == 'test_IM.tif'
    assert datasets[0].segpred_file.name == 'test_segPred.tif'
    assert datasets[0].uncertainty_file.name == 'test_uncertainty.tif'


@pytest.mark.data_manager
def test_detect_datasets_with_temp_files(data_manager, temp_data_dir):
    """Test detection of temp files"""
    data_manager.set_data_directory(temp_data_dir)
    
    # Create temp files
    segpred_temp = np.random.randint(0, 5, (10, 10, 10)).astype(np.uint8)
    uncertainty_temp = np.random.rand(10, 10, 10).astype(np.float32)
    imwrite(temp_data_dir / 'test_segPred_temp.tif', segpred_temp)
    imwrite(temp_data_dir / 'test_uncertainty_temp.tif', uncertainty_temp)
    
    datasets = data_manager.detect_datasets()
    
    assert len(datasets) == 1
    assert datasets[0].has_temp is True
    assert datasets[0].temp_segpred is not None
    assert datasets[0].temp_uncertainty is not None


@pytest.mark.data_manager
def test_detect_datasets_ome_tiff(tmp_path, monkeypatch):
    """Test detection of OME-TIFF files"""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    
    img = np.random.rand(10, 10, 10).astype(np.uint8)
    segpred = np.random.randint(0, 5, (10, 10, 10)).astype(np.uint8)
    uncertainty = np.random.rand(10, 10, 10).astype(np.float32)
    
    imwrite(data_dir / 'sample_IM.ome.tif', img)
    imwrite(data_dir / 'sample_segPred.tif', segpred)  # Use .tif to avoid duplication
    imwrite(data_dir / 'sample_uncertainty.tif', uncertainty)
    
    # Use isolated config
    test_config = tmp_path / '.vessqc_config.json'
    monkeypatch.setattr(DataManager, 'CONFIG_FILE', test_config)
    
    data_manager = DataManager()
    data_manager.set_data_directory(data_dir)
    datasets = data_manager.detect_datasets()
    
    # Should detect exactly 1 dataset
    assert len(datasets) == 1
    assert datasets[0].base_name == 'sample'
    assert 'ome' in str(datasets[0].raw_file).lower()  # Verify OME-TIFF was used


@pytest.mark.data_manager
def test_segmentation_detection(data_manager, temp_data_dir):
    """Test detection of existing segmentation"""
    data_manager.set_data_directory(temp_data_dir)
    
    # Create segmentation files
    seg_dir = data_manager.segmentation_dir
    labels = np.random.randint(0, 5, (10, 10, 10)).astype(np.int32)
    segments = [
        {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.5, 'counts': 100, 'coords': None, 'done': False},
        {'name': 'Segment_2', 'label': 2, 'uncertainty': 0.8, 'counts': 50, 'coords': None, 'done': False}
    ]
    
    imwrite(seg_dir / 'test_labels.tif', labels)
    with (seg_dir / 'test_segments.json').open('w') as f:
        json.dump(segments, f)
    
    datasets = data_manager.detect_datasets()
    
    assert len(datasets) == 1
    assert datasets[0].has_segmentation is True


@pytest.mark.data_manager
def test_load_existing_segmentation_filters_done(data_manager, temp_data_dir):
    """Test that done segments are filtered out"""
    data_manager.set_data_directory(temp_data_dir)
    
    # Create segmentation with done and undone segments
    seg_dir = data_manager.segmentation_dir
    labels = np.ones((10, 10, 10), dtype=np.int32)
    segments = [
        {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.9, 'counts': 100, 'coords': None, 'done': True},
        {'name': 'Segment_2', 'label': 2, 'uncertainty': 0.8, 'counts': 80, 'coords': None, 'done': False},
        {'name': 'Segment_3', 'label': 3, 'uncertainty': 0.7, 'counts': 60, 'coords': None, 'done': False}
    ]
    
    imwrite(seg_dir / 'test_labels.tif', labels)
    with (seg_dir / 'test_segments.json').open('w') as f:
        json.dump(segments, f)
    
    datasets = data_manager.detect_datasets()
    triplet = datasets[0]
    
    # Calculate priorities
    priority = data_manager._load_existing_segmentation(triplet)
    
    # Should only have 2 segment priorities (done segment filtered out)
    assert len(triplet.segment_priorities) == 2
    assert all(s.segment_name != 'Segment_1' for s in triplet.segment_priorities)
    assert triplet.segment_priorities[0].uncertainty == 0.8  # Highest undone


@pytest.mark.data_manager
def test_load_existing_segmentation_filters_small_segments(data_manager, temp_data_dir):
    """Test that small segments (0.9999) are filtered out"""
    data_manager.set_data_directory(temp_data_dir)
    
    seg_dir = data_manager.segmentation_dir
    labels = np.ones((10, 10, 10), dtype=np.int32)
    segments = [
        {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.8, 'counts': 100, 'coords': None, 'done': False},
        {'name': 'Small_Segments', 'label': 99, 'uncertainty': 0.9999, 'counts': 20, 'coords': None, 'done': False}
    ]
    
    imwrite(seg_dir / 'test_labels.tif', labels)
    with (seg_dir / 'test_segments.json').open('w') as f:
        json.dump(segments, f)
    
    datasets = data_manager.detect_datasets()
    triplet = datasets[0]
    
    priority = data_manager._load_existing_segmentation(triplet)
    
    # Should only have 1 segment priority (small segments filtered)
    assert len(triplet.segment_priorities) == 1
    assert triplet.segment_priorities[0].segment_name == 'Segment_1'


@pytest.mark.data_manager
def test_config_persistence(temp_data_dir, tmp_path, monkeypatch):
    """Test that config is saved and loaded"""
    test_config = tmp_path / '.vessqc_test_config.json'
    
    # Patch CONFIG_FILE for the whole test
    monkeypatch.setattr(DataManager, 'CONFIG_FILE', test_config)
    
    # Create manager with test config file
    manager1 = DataManager()
    manager1.set_data_directory(temp_data_dir)
    
    # Create new manager with same config file
    manager2 = DataManager()
    
    # Should load saved directory
    assert manager2.data_directory == temp_data_dir
    
    # Cleanup
    if test_config.exists():
        test_config.unlink()


@pytest.mark.data_manager
def test_priority_calculation_excludes_done(data_manager, temp_data_dir):
    """Test that priority calculation excludes done segments"""
    data_manager.set_data_directory(temp_data_dir)
    
    seg_dir = data_manager.segmentation_dir
    labels = np.ones((10, 10, 10), dtype=np.int32)
    segments = [
        {'name': 'Segment_1', 'label': 1, 'uncertainty': 0.9, 'counts': 100, 'coords': None, 'done': True},
        {'name': 'Segment_2', 'label': 2, 'uncertainty': 0.5, 'counts': 80, 'coords': None, 'done': False}
    ]
    
    imwrite(seg_dir / 'test_labels.tif', labels)
    with (seg_dir / 'test_segments.json').open('w') as f:
        json.dump(segments, f)
    
    datasets = data_manager.detect_datasets()
    data_manager.calculate_priorities(threaded=False)
    
    # Should only have undone segments
    assert len(data_manager.all_segment_priorities) == 1
    assert data_manager.all_segment_priorities[0].uncertainty == 0.5

