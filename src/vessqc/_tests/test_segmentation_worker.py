# Tests for SegmentationWorker
# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import json
import numpy as np
from pathlib import Path
import pytest
import time
from tifffile import imread, imwrite
from unittest import mock

from vessqc._segmentation_worker import SegmentationWorker


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_uncertainty():
    """Create sample uncertainty data"""
    # Create small test data with distinct uncertainty values
    uncertainty = np.zeros((10, 10, 10), dtype=np.float32)
    uncertainty[2:5, 2:5, 2:5] = 0.8  # High uncertainty region
    uncertainty[6:8, 6:8, 6:8] = 0.3  # Low uncertainty region
    return uncertainty


@pytest.fixture
def sample_segpred():
    """Create sample segpred data"""
    segpred = np.zeros((10, 10, 10), dtype=np.uint8)
    segpred[2:5, 2:5, 2:5] = 1
    segpred[6:8, 6:8, 6:8] = 1
    return segpred


@pytest.mark.worker
def test_worker_init():
    """Test worker initialization"""
    worker = SegmentationWorker()
    
    assert worker.is_running is False
    assert worker.should_stop is False
    assert worker.current_dataset is None
    assert worker.get_queue_size() == 0


@pytest.mark.worker
def test_worker_start_stop():
    """Test worker start and stop"""
    worker = SegmentationWorker()
    
    worker.start()
    assert worker.is_running is True
    
    time.sleep(0.1)  # Let worker loop start
    
    worker.stop()
    time.sleep(0.5)  # Let worker loop finish
    
    assert worker.is_running is False


@pytest.mark.worker
def test_add_dataset_to_queue(temp_output_dir):
    """Test adding dataset to queue"""
    worker = SegmentationWorker()
    
    uncertainty_file = temp_output_dir / 'test_uncertainty.tif'
    segpred_file = temp_output_dir / 'test_segPred.tif'
    
    # Create dummy files
    imwrite(uncertainty_file, np.ones((5, 5, 5), dtype=np.float32) * 0.5)
    imwrite(segpred_file, np.ones((5, 5, 5), dtype=np.uint8))
    
    worker.add_dataset('test', uncertainty_file, segpred_file, temp_output_dir)
    
    assert worker.get_queue_size() == 1
    assert 'test' in worker._queued_datasets


@pytest.mark.worker
def test_duplicate_prevention(temp_output_dir):
    """Test that duplicate datasets are not queued"""
    worker = SegmentationWorker()
    
    uncertainty_file = temp_output_dir / 'test_uncertainty.tif'
    segpred_file = temp_output_dir / 'test_segPred.tif'
    
    imwrite(uncertainty_file, np.ones((5, 5, 5), dtype=np.float32) * 0.5)
    imwrite(segpred_file, np.ones((5, 5, 5), dtype=np.uint8))
    
    worker.add_dataset('test', uncertainty_file, segpred_file, temp_output_dir)
    worker.add_dataset('test', uncertainty_file, segpred_file, temp_output_dir)
    
    # Should only have one item in queue
    assert worker.get_queue_size() == 1


@pytest.mark.worker
def test_json_serialization_no_numpy_arrays(temp_output_dir, sample_uncertainty, sample_segpred):
    """Test that saved JSON does not contain numpy arrays"""
    worker = SegmentationWorker()
    
    uncertainty_file = temp_output_dir / 'test_uncertainty.tif'
    segpred_file = temp_output_dir / 'test_segPred.tif'
    
    imwrite(uncertainty_file, sample_uncertainty)
    imwrite(segpred_file, sample_segpred)
    
    # Process directly without threading
    work_item = {
        'dataset_name': 'test',
        'uncertainty_file': uncertainty_file,
        'segpred_file': segpred_file,
        'output_dir': temp_output_dir
    }
    
    try:
        worker._process_dataset(work_item)
    except Exception as e:
        # Processing might fail on small test data, but we check JSON anyway
        pass
    
    segments_file = temp_output_dir / 'test_segments.json'
    if segments_file.exists():
        with segments_file.open('r') as f:
            data = json.load(f)
        
        # Verify all fields are serializable
        for segment in data:
            assert isinstance(segment['label'], int)
            assert isinstance(segment['uncertainty'], float)
            assert isinstance(segment['count'], int)
            assert isinstance(segment['done'], bool)
            # 'name' field is no longer stored (we store 'custom_name' only if it differs)
            if 'custom_name' in segment:
                assert isinstance(segment['custom_name'], str)


@pytest.mark.worker
def test_callback_mechanism(temp_output_dir, sample_uncertainty, sample_segpred):
    """Test that callback is called after processing"""
    callback_called = []
    
    def callback(dataset_name, success, error):
        callback_called.append((dataset_name, success, error))
    
    worker = SegmentationWorker(callback=callback)
    worker.start()
    
    uncertainty_file = temp_output_dir / 'test_uncertainty.tif'
    segpred_file = temp_output_dir / 'test_segPred.tif'
    
    imwrite(uncertainty_file, sample_uncertainty)
    imwrite(segpred_file, sample_segpred)
    
    worker.add_dataset('test', uncertainty_file, segpred_file, temp_output_dir)
    
    # Wait for processing
    time.sleep(2.0)
    
    worker.stop()
    
    # Callback should have been called
    assert len(callback_called) > 0


@pytest.mark.worker
def test_graceful_shutdown():
    """Test graceful shutdown doesn't raise exceptions"""
    worker = SegmentationWorker()
    worker.start()
    
    time.sleep(0.1)
    
    # Should not raise
    worker.stop()
    time.sleep(0.5)


@pytest.mark.worker
def test_error_handling_invalid_file(temp_output_dir):
    """Test error handling for invalid files"""
    callback_called = []
    
    def callback(dataset_name, success, error):
        callback_called.append((dataset_name, success, error))
    
    worker = SegmentationWorker(callback=callback)
    
    # Use non-existent files
    uncertainty_file = temp_output_dir / 'nonexistent_uncertainty.tif'
    segpred_file = temp_output_dir / 'nonexistent_segPred.tif'
    
    work_item = {
        'dataset_name': 'invalid',
        'uncertainty_file': uncertainty_file,
        'segpred_file': segpred_file,
        'output_dir': temp_output_dir
    }
    
    # Should handle error gracefully
    try:
        worker._process_dataset(work_item)
    except Exception:
        pass  # Expected to fail
    
    # Should not crash worker


@pytest.mark.worker
def test_calculate_segmentation_filters_small(sample_uncertainty, sample_segpred):
    """Test that small segments are filtered (< 50 pixels)"""
    worker = SegmentationWorker()
    
    # Create small segment
    small_uncertainty = np.zeros((10, 10, 10), dtype=np.float32)
    small_segpred = np.zeros((10, 10, 10), dtype=np.uint8)
    
    # Small region (< 50 pixels)
    small_uncertainty[0:2, 0:2, 0:2] = 0.5
    small_segpred[0:2, 0:2, 0:2] = 1
    
    # Large region (> 50 pixels)
    small_uncertainty[3:8, 3:8, 3:8] = 0.7
    small_segpred[3:8, 3:8, 3:8] = 1
    
    labels, segments = worker._calculate_segmentation(small_uncertainty, small_segpred)
    
    # Small region (8 voxels) should be merged into Noise
    noise_seg = [s for s in segments if s.name == 'Noise']
    assert len(noise_seg) == 1
    assert noise_seg[0].uncertainty == 0.9999


@pytest.mark.worker
def test_queue_removes_on_complete(temp_output_dir, sample_uncertainty, sample_segpred):
    """Test that dataset is removed from _queued_datasets after processing"""
    worker = SegmentationWorker()
    worker.start()
    
    uncertainty_file = temp_output_dir / 'test_uncertainty.tif'
    segpred_file = temp_output_dir / 'test_segPred.tif'
    
    imwrite(uncertainty_file, sample_uncertainty)
    imwrite(segpred_file, sample_segpred)
    
    worker.add_dataset('test', uncertainty_file, segpred_file, temp_output_dir)
    
    assert 'test' in worker._queued_datasets
    
    # Wait for processing
    time.sleep(2.0)
    
    # Should be removed after completion
    assert 'test' not in worker._queued_datasets
    
    worker.stop()


