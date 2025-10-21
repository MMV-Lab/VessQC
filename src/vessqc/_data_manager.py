"""
Data management module for VessQC

This module handles file detection, priority calculation, caching, and
directory management for vessel segmentation datasets.

Imports
-------
json, numpy, pathlib.Path, threading, tifffile

Exports
-------
DataManager, DatasetTriplet
"""

# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import json
import logging
import numpy as np
from pathlib import Path
import threading
from typing import Optional, List, Dict, Tuple, Callable
from tifffile import imread
import SimpleITK as sitk
import time

# Setup logging
log_file = Path.home() / '.vessqc_debug.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('vessqc')

def debug_print(msg: str):
    """Print and log debug message"""
    print(msg)
    logger.debug(msg)


class SegmentPriority:
    """
    Represents a segment with its priority
    
    Attributes
    ----------
    dataset_name : str
        Name of the dataset
    segment_label : int
        Label of the segment
    segment_name : str
        Name of the segment (e.g., "Segment_3")
    uncertainty : float
        Maximum uncertainty value in this segment
    triplet : DatasetTriplet
        Reference to the dataset triplet
    """
    
    def __init__(self, dataset_name: str, segment_label: int, segment_name: str,
                 uncertainty: float, triplet: 'DatasetTriplet'):
        self.dataset_name = dataset_name
        self.segment_label = segment_label
        self.segment_name = segment_name
        self.uncertainty = uncertainty
        self.triplet = triplet
    
    def __repr__(self):
        return f"{self.dataset_name} - {self.segment_name} ({self.uncertainty:.3f})"


class DatasetTriplet:
    """
    Represents a triplet of files: raw image, segmentation prediction, and uncertainty
    
    Attributes
    ----------
    base_name : str
        Base name of the dataset (without suffixes)
    raw_file : Path
        Path to the raw image file
    segpred_file : Path
        Path to the segmentation prediction file
    uncertainty_file : Path
        Path to the uncertainty file
    priority : float
        Priority value based on uncertainty (lower = higher priority)
    has_temp : bool
        Whether temporary files exist for this dataset
    segment_priorities : List[SegmentPriority]
        List of segment priorities for this dataset
    """
    
    def __init__(self, base_name: str, raw_file: Path, segpred_file: Path, 
                 uncertainty_file: Path):
        self.base_name = base_name
        self.raw_file = raw_file
        self.segpred_file = segpred_file
        self.uncertainty_file = uncertainty_file
        self.priority: Optional[float] = None
        self.has_temp = False
        self.temp_segpred: Optional[Path] = None
        self.temp_uncertainty: Optional[Path] = None
        self.segment_priorities: List[SegmentPriority] = []
        self.has_segmentation = False  # Whether segmentation has been calculated
        self.segmentation_dir: Optional[Path] = None  # Where segmentation is stored
        
    def __repr__(self):
        return f"DatasetTriplet('{self.base_name}', priority={self.priority})"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'base_name': self.base_name,
            'raw_file': str(self.raw_file),
            'segpred_file': str(self.segpred_file),
            'uncertainty_file': str(self.uncertainty_file),
            'priority': self.priority,
            'has_temp': self.has_temp,
            'temp_segpred': str(self.temp_segpred) if self.temp_segpred else None,
            'temp_uncertainty': str(self.temp_uncertainty) if self.temp_uncertainty else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetTriplet':
        """Create from dictionary"""
        triplet = cls(
            base_name=data['base_name'],
            raw_file=Path(data['raw_file']),
            segpred_file=Path(data['segpred_file']),
            uncertainty_file=Path(data['uncertainty_file'])
        )
        triplet.priority = data.get('priority')
        triplet.has_temp = data.get('has_temp', False)
        if data.get('temp_segpred'):
            triplet.temp_segpred = Path(data['temp_segpred'])
        if data.get('temp_uncertainty'):
            triplet.temp_uncertainty = Path(data['temp_uncertainty'])
        return triplet


class DataManager:
    """
    Manages dataset loading, caching, and priority calculation
    
    Attributes
    ----------
    data_directory : Path
        Directory containing the datasets
    cache_file : Path
        Path to the cache file storing priorities and metadata
    datasets : List[DatasetTriplet]
        List of detected dataset triplets
    all_segment_priorities : List[SegmentPriority]
        Flat list of all segments across all datasets, sorted by priority
    """
    
    CONFIG_FILE = Path.home() / '.vessqc_config.json'
    CACHE_FILENAME = '.vessqc_cache.json'
    SEGMENTATION_DIR = '.vessqc_segmentations'
    
    def __init__(self):
        self.data_directory: Optional[Path] = None
        self.cache_file: Optional[Path] = None
        self.segmentation_dir: Optional[Path] = None
        self.datasets: List[DatasetTriplet] = []
        self.all_segment_priorities: List[SegmentPriority] = []
        self._priority_thread: Optional[threading.Thread] = None
        self._priority_callback: Optional[Callable] = None
        self._load_config()
        
    def _load_config(self):
        """Load persistent configuration (data directory)"""
        if self.CONFIG_FILE.exists():
            try:
                with self.CONFIG_FILE.open('r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'data_directory' in config:
                        self.data_directory = Path(config['data_directory'])
                        if self.data_directory.exists():
                            self.cache_file = self.data_directory / self.CACHE_FILENAME
                            self.segmentation_dir = self.data_directory / self.SEGMENTATION_DIR
                            self.segmentation_dir.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
    
    def _save_config(self):
        """Save persistent configuration"""
        try:
            config = {}
            if self.data_directory:
                config['data_directory'] = str(self.data_directory)
            with self.CONFIG_FILE.open('w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def set_data_directory(self, directory: Path) -> bool:
        """
        Set the data directory and save to config
        
        Parameters
        ----------
        directory : Path
            Path to the data directory
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not directory.exists() or not directory.is_dir():
            return False
        
        self.data_directory = directory
        self.cache_file = directory / self.CACHE_FILENAME
        self.segmentation_dir = directory / self.SEGMENTATION_DIR
        self.segmentation_dir.mkdir(exist_ok=True)
        self._save_config()
        return True
    
    def get_data_directory(self) -> Optional[Path]:
        """Get the current data directory"""
        return self.data_directory
    
    def _find_matching_file(self, directory: Path, patterns: List[str], 
                           extensions: List[str]) -> Optional[Path]:
        """
        Find a file matching any of the patterns with any of the extensions
        
        Parameters
        ----------
        directory : Path
            Directory to search in
        patterns : List[str]
            List of filename patterns (without extension)
        extensions : List[str]
            List of file extensions to try
            
        Returns
        -------
        Optional[Path]
            Path to the found file, or None
        """
        for pattern in patterns:
            for ext in extensions:
                filepath = directory / f"{pattern}{ext}"
                if filepath.exists():
                    return filepath
        return None
    
    def detect_datasets(self) -> List[DatasetTriplet]:
        """
        Detect dataset triplets in the data directory
        
        Returns
        -------
        List[DatasetTriplet]
            List of detected dataset triplets
        """
        if not self.data_directory or not self.data_directory.exists():
            print(f"Data directory not set or does not exist: {self.data_directory}")
            return []
        
        print(f"Detecting datasets in: {self.data_directory}")
        datasets = []
        extensions = ['.tif', '.tiff', '.ome.tif', '.ome.tiff', '.nii', '.nii.gz']
        
        # Find all potential raw files (files without _segPred or _uncertainty suffixes)
        all_files = []
        for ext in extensions:
            all_files.extend(self.data_directory.glob(f'*{ext}'))
        
        # Deduplicate files (e.g., .ome.tif might be matched by both .tif and .ome.tif patterns)
        all_files = list(set(all_files))
        
        print(f"Found {len(all_files)} total files with supported extensions")
        if len(all_files) > 0:
            print(f"Sample files: {[f.name for f in all_files[:5]]}")
        
        # Filter to get potential raw files (must end with _IM)
        raw_files = []
        for f in all_files:
            stem = f.stem
            # Handle .nii.gz
            if stem.endswith('.nii'):
                stem = stem[:-4]
            # Handle .ome.tif(f)
            if stem.endswith('.ome'):
                stem = stem[:-4]
            if stem.endswith('_IM'):
                raw_files.append(f)
        
        print(f"Found {len(raw_files)} files ending with _IM")
        if len(raw_files) > 0:
            print(f"Raw files: {[f.name for f in raw_files[:5]]}")
        
        # For each raw file, try to find matching segPred and uncertainty files
        for raw_file in raw_files:
            base_stem = raw_file.stem
            # Handle .nii.gz
            if base_stem.endswith('.nii'):
                base_stem = base_stem[:-4]
            # Handle .ome.tif(f)
            if base_stem.endswith('.ome'):
                base_stem = base_stem[:-4]
            
            # Remove _IM suffix to get base name
            if base_stem.endswith('_IM'):
                base_name = base_stem[:-3]
            else:
                # Should not happen based on our filter, but handle it anyway
                base_name = base_stem
            
            print(f"\nProcessing raw file: {raw_file.name}")
            print(f"  Base name: {base_name}")
            
            # Try to find segPred file
            segpred_patterns = [
                f"{base_name}_segPred",
                f"{base_stem}_segPred",
            ]
            print(f"  Looking for segPred with patterns: {segpred_patterns}")
            segpred_file = self._find_matching_file(self.data_directory, 
                                                    segpred_patterns, extensions)
            if segpred_file:
                print(f"  Found segPred: {segpred_file.name}")
            else:
                print(f"  segPred not found")
            
            # Try to find uncertainty file
            uncertainty_patterns = [
                f"{base_name}_uncertainty",
                f"{base_stem}_uncertainty",
            ]
            print(f"  Looking for uncertainty with patterns: {uncertainty_patterns}")
            uncertainty_file = self._find_matching_file(self.data_directory, 
                                                       uncertainty_patterns, extensions)
            if uncertainty_file:
                print(f"  Found uncertainty: {uncertainty_file.name}")
            else:
                print(f"  uncertainty not found")
            
            # Fallback: If we didn't find both with naming conventions, 
            # look for ANY files containing the base name and use size to determine which is which
            if not (segpred_file and uncertainty_file):
                print(f"  Standard naming not found, searching for any files with base name...")
                candidates = []
                for f in all_files:
                    # Skip the raw file itself
                    if f == raw_file:
                        continue
                    # Check if filename contains the base name
                    if base_name in f.stem:
                        size = f.stat().st_size
                        candidates.append((f, size))
                        print(f"    Found candidate: {f.name} (size: {size})")
                
                if len(candidates) == 2:
                    # Exactly 2 candidates - we can confidently assign by size
                    # Sort by size: smaller = segPred (typically uint8/uint16), larger = uncertainty (float32)
                    candidates.sort(key=lambda x: x[1])
                    segpred_file = candidates[0][0]
                    uncertainty_file = candidates[1][0]
                    print(f"  ✓ Auto-assigned by size:")
                    print(f"    segPred: {segpred_file.name} ({candidates[0][1]} bytes)")
                    print(f"    uncertainty: {uncertainty_file.name} ({candidates[1][1]} bytes)")
                elif len(candidates) == 1:
                    print(f"  ⚠️  Only found 1 candidate file, need exactly 2 for auto-assignment")
                elif len(candidates) > 2:
                    print(f"  ⚠️  Found {len(candidates)} candidates, need exactly 2 for confident auto-assignment")
                    print(f"      Please use standard naming (_segPred and _uncertainty)")
            
            # If we found both, create a triplet
            if segpred_file and uncertainty_file:
                # Verify size constraint: uncertainty must be >= segPred
                # (uncertainty is float32, segPred is typically uint8/uint16)
                try:
                    segpred_size = segpred_file.stat().st_size
                    uncertainty_size = uncertainty_file.stat().st_size
                    print(f"  Size check: segPred={segpred_size}, uncertainty={uncertainty_size}")
                    
                    # Auto-swap if they're backwards (segPred should be smaller)
                    if segpred_size > uncertainty_size:
                        print(f"  ⚠️  Files appear backwards, swapping based on size")
                        segpred_file, uncertainty_file = uncertainty_file, segpred_file
                        segpred_size, uncertainty_size = uncertainty_size, segpred_size
                        print(f"  After swap: segPred={segpred_size}, uncertainty={uncertainty_size}")
                    
                    if uncertainty_size >= segpred_size:
                        triplet = DatasetTriplet(base_name, raw_file, 
                                                segpred_file, uncertainty_file)
                        print(f"  ✓ Created triplet for {base_name}")
                        
                        # Check for temp files
                        temp_segpred_patterns = [f"{base_name}_segPred_temp", 
                                                f"{base_stem}_segPred_temp"]
                        temp_segpred = self._find_matching_file(self.data_directory,
                                                               temp_segpred_patterns, 
                                                               extensions)
                        
                        temp_uncertainty_patterns = [f"{base_name}_uncertainty_temp",
                                                    f"{base_stem}_uncertainty_temp"]
                        temp_uncertainty = self._find_matching_file(self.data_directory,
                                                                   temp_uncertainty_patterns,
                                                                   extensions)
                        
                        if temp_segpred or temp_uncertainty:
                            triplet.has_temp = True
                            triplet.temp_segpred = temp_segpred
                            triplet.temp_uncertainty = temp_uncertainty
                        
                        # Check if segmentation exists
                        if self.segmentation_dir:
                            labels_file = self.segmentation_dir / f"{base_name}_labels.tif"
                            segments_file = self.segmentation_dir / f"{base_name}_segments.json"
                            if labels_file.exists() and segments_file.exists():
                                triplet.has_segmentation = True
                                triplet.segmentation_dir = self.segmentation_dir
                        
                        datasets.append(triplet)
                    else:
                        print(f"  ✗ Size check failed: uncertainty must be >= segPred")
                except Exception as e:
                    print(f"  ✗ Error verifying sizes for {base_name}: {e}")
            else:
                print(f"  ✗ Missing files (segPred={segpred_file is not None}, uncertainty={uncertainty_file is not None})")
        
        print(f"\nTotal datasets detected: {len(datasets)}")
        self.datasets = datasets
        return datasets
    
    def _calculate_priority_for_dataset(self, triplet: DatasetTriplet) -> float:
        """
        Calculate priority for a single dataset based on uncertainty values
        Also calculates per-segment priorities
        
        Parameters
        ----------
        triplet : DatasetTriplet
            Dataset triplet to calculate priority for
            
        Returns
        -------
        float
            Priority value (max uncertainty across segments)
        """
        try:
            # If segmentation already exists, load it
            if triplet.has_segmentation and triplet.segmentation_dir:
                print(f"Loading existing segmentation for {triplet.base_name}")
                return self._load_existing_segmentation(triplet)
            
            # Otherwise calculate from scratch
            print(f"Calculating priorities from scratch for {triplet.base_name}")
            # Use temp file if available, otherwise use original
            uncertainty_file = (triplet.temp_uncertainty if triplet.temp_uncertainty 
                              else triplet.uncertainty_file)
            segpred_file = (triplet.temp_segpred if triplet.temp_segpred 
                          else triplet.segpred_file)
            print(f"  Using uncertainty file: {uncertainty_file.name}")
            print(f"  Using segpred file: {segpred_file.name}")
            
            # Load uncertainty data
            suffix = uncertainty_file.suffix.lower()
            if suffix in ['.tif', '.tiff'] or suffix.endswith('.ome.tif') or suffix.endswith('.ome.tiff'):
                uncertainty = imread(uncertainty_file)
            elif suffix in ['.nii', '.gz']:
                sitk_image = sitk.ReadImage(str(uncertainty_file))
                uncertainty = sitk.GetArrayFromImage(sitk_image)
            else:
                return float('inf')
            
            # Load segmentation data to identify segments
            suffix = segpred_file.suffix.lower()
            if suffix in ['.tif', '.tiff'] or suffix.endswith('.ome.tif') or suffix.endswith('.ome.tiff'):
                segpred = imread(segpred_file)
            elif suffix in ['.nii', '.gz']:
                sitk_image = sitk.ReadImage(str(segpred_file))
                segpred = sitk.GetArrayFromImage(sitk_image)
            else:
                return float('inf')
            
            # Calculate per-segment priorities
            print(f"  Calculating per-segment priorities...")
            triplet.segment_priorities = []
            unique_labels = np.unique(segpred)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            print(f"  Found {len(unique_labels)} unique labels in segPred")
            
            # Calculate size threshold (200 pixels) to identify small segments
            min_size = 200
            counts = np.bincount(segpred.ravel())
            
            max_uncertainty = 0.0
            segments_processed = 0
            segments_skipped = 0
            
            for label in unique_labels:
                # Skip small segments
                if counts[label] < min_size:
                    segments_skipped += 1
                    continue
                
                # Get uncertainty values for this segment
                mask = segpred == label
                segment_uncertainty = uncertainty[mask]
                segment_uncertainty = segment_uncertainty[segment_uncertainty > 0]
                
                if len(segment_uncertainty) > 0:
                    # Use maximum uncertainty in segment
                    max_uncert = float(np.max(segment_uncertainty))
                    
                    # Skip if this looks like a grouped small segments value (0.9999)
                    if max_uncert >= 0.999:
                        print(f"  Skipping label {label}: uncertainty {max_uncert:.4f} (likely small segments)")
                        segments_skipped += 1
                        continue
                    
                    max_uncertainty = max(max_uncertainty, max_uncert)
                    
                    # Create segment priority entry
                    seg_priority = SegmentPriority(
                        dataset_name=triplet.base_name,
                        segment_label=int(label),
                        segment_name=f"Segment_{label}",  # Will be updated later
                        uncertainty=max_uncert,
                        triplet=triplet
                    )
                    triplet.segment_priorities.append(seg_priority)
                    segments_processed += 1
            
            print(f"  Processed {segments_processed} segments, skipped {segments_skipped} small segments")
            print(f"  Max uncertainty: {max_uncertainty:.4f}")
            
            # Sort segments by uncertainty (highest first)
            triplet.segment_priorities.sort(key=lambda x: x.uncertainty, reverse=True)
            if len(triplet.segment_priorities) > 0:
                print(f"  Top segment: {triplet.segment_priorities[0].segment_name} ({triplet.segment_priorities[0].uncertainty:.4f})")
            
            return max_uncertainty if max_uncertainty > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate priority for {triplet.base_name}: {e}")
            return float('inf')
    
    def _load_existing_segmentation(self, triplet: DatasetTriplet) -> float:
        """
        Load existing segmentation from disk and calculate priorities
        Uses cached segment uncertainties from JSON (fast, no file loading)
        
        Parameters
        ----------
        triplet : DatasetTriplet
            Dataset triplet with existing segmentation
            
        Returns
        -------
        float
            Priority value (max uncertainty across segments)
        """
        try:
            print(f"\nDEBUG: _load_existing_segmentation() called for {triplet.base_name}")
            print(f"DEBUG:   segmentation_dir: {triplet.segmentation_dir}")
            
            # Load segments metadata (contains cached uncertainties)
            segments_file = triplet.segmentation_dir / f"{triplet.base_name}_segments.json"
            print(f"DEBUG:   Segments file path: {segments_file}")
            print(f"DEBUG:   File exists: {segments_file.exists()}")
            
            if not segments_file.exists():
                print(f"DEBUG:   Segments file not found: {segments_file}")
                return float('inf')
            
            with segments_file.open('r', encoding='utf-8') as f:
                segments = json.load(f)
            
            print(f"DEBUG:   Loaded {len(segments)} segments from JSON")
            
            # Count done vs undone
            done_count = sum(1 for s in segments if s.get('done', False))
            print(f"DEBUG:   Segments status: {done_count} done, {len(segments) - done_count} undone")
            
            # Create segment priorities from cached data (no need to load large files!)
            triplet.segment_priorities = []
            max_uncertainty = 0.0
            min_size = 200
            segments_processed = 0
            segments_skipped = 0
            
            for segment in segments:
                label = segment['label']
                uncertainty_value = segment.get('uncertainty', 0.0)
                
                # Synthesize name from label (name field no longer stored)
                # Use custom_name if it exists, otherwise synthesize default
                if 'custom_name' in segment:
                    segment_name = segment['custom_name']
                elif uncertainty_value >= 0.999:
                    segment_name = 'Small_Segments'
                else:
                    segment_name = f'Segment_{label}'
                
                # Skip if already done
                if segment.get('done', False):
                    print(f"DEBUG:   Skipping {segment_name}: marked as done")
                    segments_skipped += 1
                    continue
                
                # Skip small segments collection
                if uncertainty_value >= 0.999:
                    segments_skipped += 1
                    continue
                
                # Skip if too small
                if segment.get('counts', 0) < min_size:
                    segments_skipped += 1
                    continue
                
                if uncertainty_value > 0:
                    max_uncertainty = max(max_uncertainty, uncertainty_value)
                    
                    # Create segment priority entry
                    seg_priority = SegmentPriority(
                        dataset_name=triplet.base_name,
                        segment_label=label,
                        segment_name=segment_name,  # Synthesized from label
                        uncertainty=uncertainty_value,
                        triplet=triplet
                    )
                    triplet.segment_priorities.append(seg_priority)
                    segments_processed += 1
                else:
                    segments_skipped += 1
            
            # Sort segments by uncertainty (highest first)
            triplet.segment_priorities.sort(key=lambda x: x.uncertainty, reverse=True)
            
            print(f"DEBUG:   Processed {segments_processed} segments, skipped {segments_skipped}")
            print(f"DEBUG:   Max uncertainty: {max_uncertainty:.4f}")
            if len(triplet.segment_priorities) > 0:
                print(f"DEBUG:   Top segment: {triplet.segment_priorities[0]}")
            
            return max_uncertainty if max_uncertainty > 0 else 0.0
            
        except Exception as e:
            print(f"DEBUG: Error loading existing segmentation for {triplet.base_name}: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def calculate_priorities(self, callback: Optional[Callable] = None, 
                           threaded: bool = True):
        """
        Calculate priorities for all datasets
        
        Parameters
        ----------
        callback : Optional[Callable]
            Function to call when priorities are calculated
        threaded : bool
            Whether to calculate in a separate thread
        """
        self._priority_callback = callback
        
        if threaded:
            if self._priority_thread and self._priority_thread.is_alive():
                print("Priority calculation already in progress")
                return
            
            self._priority_thread = threading.Thread(
                target=self._calculate_priorities_worker,
                daemon=True
            )
            self._priority_thread.start()
        else:
            self._calculate_priorities_worker()
    
    def _calculate_priorities_worker(self):
        """Worker function for priority calculation"""
        try:
            print(f"DEBUG: Starting priority calculation for {len(self.datasets)} datasets")
            
            # Only calculate priorities for datasets that have segmentation
            datasets_with_seg = [d for d in self.datasets if d.has_segmentation]
            datasets_without_seg = [d for d in self.datasets if not d.has_segmentation]
            
            print(f"DEBUG:   {len(datasets_with_seg)} with segmentation: {[d.base_name for d in datasets_with_seg]}")
            print(f"DEBUG:   {len(datasets_without_seg)} without segmentation (will be queued): {[d.base_name for d in datasets_without_seg]}")
            
            # Load cache first
            cached_priorities = self._load_cache()
            
            for triplet in datasets_with_seg:
                print(f"DEBUG: Processing {triplet.base_name}")
                # Always load/calculate to get segment priorities
                triplet.priority = self._calculate_priority_for_dataset(triplet)
            
            # Sort by priority (higher uncertainty = higher priority, so reverse sort)
            self.datasets.sort(key=lambda x: x.priority if x.priority is not None 
                             else float('-inf'), reverse=True)
            
            # Create flat list of all segment priorities across all datasets
            print(f"\nDEBUG: Building flat segment list...")
            self.all_segment_priorities = []
            for triplet in self.datasets:
                print(f"DEBUG:   Checking {triplet.base_name}: has_segmentation={triplet.has_segmentation}, segment_priorities length={len(triplet.segment_priorities)}")
                if triplet.has_segmentation:
                    if len(triplet.segment_priorities) > 0:
                        print(f"DEBUG:   Adding {len(triplet.segment_priorities)} segments from {triplet.base_name}")
                        self.all_segment_priorities.extend(triplet.segment_priorities)
                    else:
                        print(f"DEBUG:   WARNING: {triplet.base_name} has segmentation but no segment_priorities!")
                else:
                    print(f"DEBUG:   Skipping {triplet.base_name} (no segmentation yet)")
            
            # Sort all segments by uncertainty (highest first)
            self.all_segment_priorities.sort(key=lambda x: x.uncertainty, reverse=True)
            
            print(f"\nDEBUG: Total segments across all datasets: {len(self.all_segment_priorities)}")
            if len(self.all_segment_priorities) > 0:
                print(f"DEBUG: Top 5 segments:")
                for seg in self.all_segment_priorities[:5]:
                    print(f"  {seg}")
            else:
                print(f"DEBUG: No segments found - all datasets need segmentation calculation")
            
            # Save cache
            self._save_cache()
            
            # Call callback if provided
            if self._priority_callback:
                self._priority_callback()
                
        except Exception as e:
            print(f"DEBUG: Error calculating priorities: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_cache(self) -> Dict[str, float]:
        """Load cached priorities from file"""
        if not self.cache_file or not self.cache_file.exists():
            return {}
        
        try:
            with self.cache_file.open('r', encoding='utf-8') as f:
                cache_data = json.load(f)
                return cache_data.get('priorities', {})
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save priorities and metadata to cache file"""
        if not self.cache_file:
            return
        
        try:
            cache_data = {
                'priorities': {
                    triplet.base_name: triplet.priority 
                    for triplet in self.datasets 
                    if triplet.priority is not None
                },
                'datasets': [triplet.to_dict() for triplet in self.datasets],
                'last_updated': time.time()
            }
            
            with self.cache_file.open('w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def sync_with_directory(self) -> Tuple[List[str], List[str]]:
        """
        Synchronize cache with directory contents
        
        Returns
        -------
        Tuple[List[str], List[str]]
            Lists of (new_datasets, removed_datasets)
        """
        if not self.data_directory:
            return [], []
        
        # Load cached dataset list
        old_datasets = set()
        if self.cache_file and self.cache_file.exists():
            try:
                with self.cache_file.open('r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    old_datasets = set(d['base_name'] 
                                     for d in cache_data.get('datasets', []))
            except Exception:
                pass
        
        # Detect current datasets
        self.detect_datasets()
        new_datasets_set = set(d.base_name for d in self.datasets)
        
        # Find differences
        added = list(new_datasets_set - old_datasets)
        removed = list(old_datasets - new_datasets_set)
        
        return added, removed
    
    def move_to_finished(self, triplet: DatasetTriplet) -> bool:
        """
        Move a completed dataset to the 'finished' subdirectory
        
        Parameters
        ----------
        triplet : DatasetTriplet
            Dataset to move
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self.data_directory:
            return False
        
        finished_dir = self.data_directory / 'finished'
        finished_dir.mkdir(exist_ok=True)
        
        try:
            # Move all files
            files_to_move = [
                triplet.raw_file,
                triplet.segpred_file,
                triplet.uncertainty_file
            ]
            
            if triplet.temp_segpred:
                files_to_move.append(triplet.temp_segpred)
            if triplet.temp_uncertainty:
                files_to_move.append(triplet.temp_uncertainty)
            
            for file in files_to_move:
                if file.exists():
                    dest = finished_dir / file.name
                    file.rename(dest)
            
            # Remove from datasets list
            self.datasets.remove(triplet)
            
            # Update cache
            self._save_cache()
            
            return True
            
        except Exception as e:
            print(f"Error moving dataset to finished: {e}")
            return False
    
    def get_files_for_loading(self, triplet: DatasetTriplet) -> Tuple[Path, Path, Path]:
        """
        Get the appropriate files to load (prioritizing temp files)
        
        Parameters
        ----------
        triplet : DatasetTriplet
            Dataset triplet
            
        Returns
        -------
        Tuple[Path, Path, Path]
            (raw_file, segpred_file, uncertainty_file) to load
        """
        raw = triplet.raw_file
        segpred = triplet.temp_segpred if triplet.temp_segpred else triplet.segpred_file
        uncertainty = (triplet.temp_uncertainty if triplet.temp_uncertainty 
                      else triplet.uncertainty_file)
        
        return raw, segpred, uncertainty
