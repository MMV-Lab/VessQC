"""
Background segmentation worker for VessQC

This module handles background segmentation calculation in a separate thread.

Imports
-------
threading, queue, numpy, pathlib, tifffile, scipy.ndimage, joblib

Exports
-------
SegmentationWorker
"""

# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import json
import numpy as np
from pathlib import Path
import queue
import threading
from typing import Optional, Callable, Dict, List
from tifffile import imread, imwrite
import SimpleITK as sitk
from scipy import ndimage
from joblib import Parallel, delayed
import time


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _label_value_sparse(uncertainty, uncert, tolerance, structure, value_idx,
    num_unique_uncert):
    """Worker function for parallel segmentation"""
    mask = np.abs(uncertainty - uncert) < tolerance
    if not np.any(mask):
        return None

    labeled, num = ndimage.label(mask, structure)
    if num == 0:
        return None

    # Calculate global unique labels directly
    labeled_global = (labeled - 1) * num_unique_uncert + (value_idx + 1)
    labeled_global[labeled == 0] = 0

    indices = np.where(mask)
    result = dict(
        indices       = indices,
        global_labels = labeled_global[indices],
        uncert        = uncert,
        num           = num
    )
    return result


class SegmentationWorker:
    """
    Background worker for calculating segmentations
    
    Attributes
    ----------
    work_queue : queue.Queue
        Queue of datasets to process
    is_running : bool
        Whether the worker is currently running
    current_dataset : str
        Name of dataset currently being processed
    """
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize the segmentation worker
        
        Parameters
        ----------
        callback : Optional[Callable]
            Function to call when segmentation is complete
            Should accept (dataset_name, success, error_message)
        """
        self.work_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.should_stop = False
        self.current_dataset: Optional[str] = None
        self.callback = callback
        self._worker_thread: Optional[threading.Thread] = None
        self._queued_datasets: set = set()  # Track which datasets are queued
        
    def start(self):
        """Start the background worker thread"""
        if self.is_running:
            print("Worker already running")
            return
        
        self.should_stop = False
        self.is_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("Segmentation worker started")
    
    def stop(self):
        """Stop the background worker thread"""
        print("Stopping segmentation worker...")
        self.should_stop = True
        self.is_running = False
        # Add sentinel to unblock queue.get()
        self.work_queue.put(None)
    
    def add_dataset(self, dataset_name: str, uncertainty_file: Path, 
                   segpred_file: Path, output_dir: Path):
        """
        Add a dataset to the processing queue
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        uncertainty_file : Path
            Path to uncertainty file
        segpred_file : Path
            Path to segmentation prediction file
        output_dir : Path
            Directory to save segmentation results
        """
        # Check if already queued
        if dataset_name in self._queued_datasets:
            print(f"DEBUG: {dataset_name} already in queue, skipping")
            return
        
        work_item = {
            'dataset_name': dataset_name,
            'uncertainty_file': uncertainty_file,
            'segpred_file': segpred_file,
            'output_dir': output_dir
        }
        self.work_queue.put(work_item)
        self._queued_datasets.add(dataset_name)
        print(f"DEBUG: Added {dataset_name} to segmentation queue (queue size: {self.work_queue.qsize()})")
    
    def get_queue_size(self) -> int:
        """Get the number of datasets waiting to be processed"""
        return self.work_queue.qsize()
    
    def _worker_loop(self):
        """Main worker loop - runs in background thread"""
        print("DEBUG: Segmentation worker loop started")
        
        while not self.should_stop:
            try:
                # Get next work item (blocks until available)
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Sentinel value
                    break
                
                if self.should_stop:  # Check again after getting item
                    break
                
                dataset_name = work_item['dataset_name']
                self.current_dataset = dataset_name
                
                print(f"\n{'='*60}")
                print(f"DEBUG: Processing segmentation for: {dataset_name}")
                print(f"DEBUG: Queue remaining: {self.work_queue.qsize()}")
                print(f"{'='*60}")
                
                try:
                    self._process_dataset(work_item)
                    
                    print(f"✓ Completed segmentation for {dataset_name}")
                    
                    # Clean up BEFORE callback so status is correct
                    print(f"DEBUG: Cleaning up {dataset_name}")
                    print(f"DEBUG:   Queue size before task_done: {self.work_queue.qsize()}")
                    
                    try:
                        self.work_queue.task_done()
                    except:
                        pass
                    
                    print(f"DEBUG:   Queue size after task_done: {self.work_queue.qsize()}")
                    
                    if dataset_name in self._queued_datasets:
                        self._queued_datasets.remove(dataset_name)
                        print(f"DEBUG:   Removed from _queued_datasets, now: {self._queued_datasets}")
                    
                    self.current_dataset = None
                    print(f"DEBUG:   current_dataset set to None")
                    
                    # Now call callback with correct queue status
                    print(f"DEBUG:   Calling callback - queue: {self.work_queue.qsize()}, current: {self.current_dataset}")
                    if self.callback and not self.should_stop:
                        self.callback(dataset_name, True, None)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Error processing {dataset_name}: {error_msg}")
                    
                    # Clean up BEFORE callback
                    try:
                        self.work_queue.task_done()
                    except:
                        pass
                    
                    if dataset_name in self._queued_datasets:
                        self._queued_datasets.remove(dataset_name)
                    
                    self.current_dataset = None
                    
                    # Now call callback
                    if self.callback and not self.should_stop:
                        self.callback(dataset_name, False, error_msg)
                    
            except queue.Empty:
                continue
            except Exception as e:
                if not self.should_stop:
                    print(f"Worker loop error: {e}")
        
        self.is_running = False
        print("DEBUG: Worker loop stopped")
    
    def _process_dataset(self, work_item: Dict):
        """
        Process a single dataset
        
        Parameters
        ----------
        work_item : Dict
            Work item containing file paths and parameters
        """
        dataset_name = work_item['dataset_name']
        uncertainty_file = work_item['uncertainty_file']
        segpred_file = work_item['segpred_file']
        output_dir = work_item['output_dir']
        
        t0 = time.time()
        
        # Load uncertainty data
        print(f"DEBUG: Loading uncertainty file: {uncertainty_file.name}")
        print(f"DEBUG: File size: {uncertainty_file.stat().st_size / 1024 / 1024:.2f} MB")
        suffix = uncertainty_file.suffix.lower()
        if suffix in ['.tif', '.tiff'] or suffix.endswith('.ome.tif') or suffix.endswith('.ome.tiff'):
            uncertainty = imread(uncertainty_file)
        elif suffix in ['.nii', '.gz']:
            sitk_image = sitk.ReadImage(str(uncertainty_file))
            uncertainty = sitk.GetArrayFromImage(sitk_image)
        else:
            raise ValueError(f"Unsupported file format: {uncertainty_file.suffix}")
        
        # Load segPred data
        print(f"DEBUG: Loading segPred file: {segpred_file.name}")
        print(f"DEBUG: File size: {segpred_file.stat().st_size / 1024 / 1024:.2f} MB")
        suffix = segpred_file.suffix.lower()
        if suffix in ['.tif', '.tiff'] or suffix.endswith('.ome.tif') or suffix.endswith('.ome.tiff'):
            segpred = imread(segpred_file)
        elif suffix in ['.nii', '.gz']:
            sitk_image = sitk.ReadImage(str(segpred_file))
            segpred = sitk.GetArrayFromImage(sitk_image)
        else:
            raise ValueError(f"Unsupported file format: {segpred_file.suffix}")
        
        # Calculate segmentation
        print(f"DEBUG: Data loaded, starting segmentation calculation...")
        print(f"DEBUG: Uncertainty shape: {uncertainty.shape}, dtype: {uncertainty.dtype}")
        print(f"DEBUG: SegPred shape: {segpred.shape}, dtype: {segpred.dtype}")
        labels, segments = self._calculate_segmentation(uncertainty, segpred)
        
        # Save results
        print(f"DEBUG: Segmentation complete, saving results...")
        self._save_results(dataset_name, labels, segments, output_dir)
        
        print(f"DEBUG: Processing completed in {time.time() - t0:.1f}s")
    
    def _calculate_segmentation(self, uncertainty: np.ndarray, 
                               segpred: np.ndarray) -> tuple:
        """
        Calculate segmentation from uncertainty data
        
        Parameters
        ----------
        uncertainty : np.ndarray
            Uncertainty data
        segpred : np.ndarray
            Segmentation prediction data
            
        Returns
        -------
        tuple
            (labels, segments) where labels is the label array and segments is metadata
        """
        # Find unique uncertainty values
        unique_uncertainties = np.unique(uncertainty)
        unique_uncertainties = unique_uncertainties[unique_uncertainties > 0]
        num_unique_uncert = len(unique_uncertainties)
        print(f"DEBUG: Found {num_unique_uncert} unique uncertainty values")
        print(f"DEBUG: Uncertainty range: {np.min(unique_uncertainties):.4f} to {np.max(unique_uncertainties):.4f}")
        
        tolerance = 1e-2
        structure = np.ones((3, 3, 3), dtype=int)
        
        # Parallel segmentation
        print(f"DEBUG: Running parallel segmentation with {num_unique_uncert} jobs...")
        try:
            results = Parallel(n_jobs=-1, verbose=0)(
                delayed(_label_value_sparse)(
                    uncertainty, uncert, tolerance, structure, idx,
                    num_unique_uncert
                )
                for idx, uncert in enumerate(unique_uncertainties)
            )
        except (RuntimeError, Exception) as e:
            # Handle shutdown errors gracefully
            if "shutdown" in str(e).lower() or "interpreter" in str(e).lower():
                print(f"DEBUG: Parallel processing interrupted by shutdown")
                raise RuntimeError("Processing interrupted by shutdown")
            raise
        
        # Assemble results
        print(f"DEBUG: Parallel processing complete, assembling results...")
        labels = np.zeros_like(uncertainty, dtype=int)
        uncert_values = {0: 0.0}
        non_null_results = sum(1 for r in results if r is not None)
        print(f"DEBUG: Got {non_null_results} non-null results out of {len(results)}")
        
        for result in results:
            if result is None:
                continue
            indices = result['indices']
            result_labels = result['global_labels']
            uncert = result['uncert']
            num = result['num']
            
            labels[indices] = result_labels
            
            keys = list(np.unique(result_labels))
            values = [uncert] * num
            u_values = dict(zip(keys, values))
            uncert_values = {**uncert_values, **u_values}
        
        # Filter small segments
        print(f"DEBUG: Filtering small segments...")
        min_size = 200
        counts = np.bincount(labels.ravel())
        small_labels = np.where(counts < min_size)[0]
        small_labels = small_labels[small_labels != 0]
        print(f"DEBUG: Found {len(small_labels)} small segments (< {min_size} pixels)")
        print(f"DEBUG: Total unique labels before filtering: {len(np.unique(labels)) - 1}")
        
        # Group small segments
        max_label = np.max(labels) + 1
        mask = np.isin(labels, small_labels)
        labels[mask] = max_label
        
        # Create segment metadata
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        counts = np.bincount(labels.ravel())
        uncert_values[max_label] = 0.9999
        
        segments = []
        for label in unique_labels:
            segment = {
                'name': '',
                'label': int(label),
                'uncertainty': float(uncert_values[label]),
                'counts': int(counts[label]),
                'coords': None,
                'done': False,
            }
            segments.append(segment)
        
        # Sort by uncertainty
        segments.sort(key=lambda x: x['uncertainty'])
        
        # Assign names using label IDs
        for i, segment in enumerate(segments, start=1):
            if segment['label'] == max_label:
                segment['name'] = "Small_Segments"
            else:
                segment['name'] = f"Segment_{segment['label']}"
        
        print(f"DEBUG: Created {len(segments)} segments")
        if len(segments) > 0:
            uncertainties = [s['uncertainty'] for s in segments if s['name'] != 'Small_Segments']
            if uncertainties:
                print(f"DEBUG: Uncertainty range in segments: {min(uncertainties):.4f} to {max(uncertainties):.4f}")
        
        return labels, segments
    
    def _save_results(self, dataset_name: str, labels: np.ndarray, 
                     segments: List[Dict], output_dir: Path):
        """
        Save segmentation results to disk
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        labels : np.ndarray
            Label array
        segments : List[Dict]
            Segment metadata
        output_dir : Path
            Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        labels_file = output_dir / f"{dataset_name}_labels.tif"
        print(f"DEBUG: Saving labels to: {labels_file.name}")
        imwrite(labels_file, labels.astype(np.int32))
        
        # Clean and save segments metadata (ensure JSON-serializable)
        segments_file = output_dir / f"{dataset_name}_segments.json"
        print(f"DEBUG: Saving segments metadata to: {segments_file.name}")
        
        clean_segments = []
        for seg in segments:
            # Clean coords - convert numpy arrays to lists
            coords_value = seg.get('coords')
            if coords_value is not None:
                # Convert [[np.int64, ...], [...]] to [[int, ...], [...]]
                coords_value = [[int(c) for c in coord] for coord in coords_value]
            
            # Determine default name based on label
            uncertainty = float(seg['uncertainty'])
            if uncertainty >= 0.999:
                default_name = 'Small_Segments'
            else:
                default_name = f"Segment_{int(seg['label'])}"
            
            clean_seg = {
                'label': int(seg['label']),
                'uncertainty': uncertainty,
                'counts': int(seg['counts']),
                'coords': coords_value,  # Now clean
                'done': bool(seg.get('done', False))
            }
            
            # Only store custom_name if it differs from the default
            if seg.get('name') and seg['name'] != default_name:
                clean_seg['custom_name'] = str(seg['name'])
            
            clean_segments.append(clean_seg)
        
        with segments_file.open('w', encoding='utf-8') as f:
            json.dump(clean_segments, f, indent=2, cls=NumpyEncoder)
        
        print(f"DEBUG: Saved {len(clean_segments)} segments successfully")
