#!/usr/bin/env python
"""Script to create test data files for VessQC tests

Run this script from the project root:
    python src/vessqc/_tests/create_test_data.py
    
Or from the _tests directory:
    python create_test_data.py
"""

import sys
import numpy as np
import json
from pathlib import Path
from tifffile import imwrite

# Determine data directory based on where script is run from
if Path('_data').exists():
    data_dir = Path('_data')
else:
    data_dir = Path(__file__).parent / '_data'

data_dir.mkdir(exist_ok=True)

print(f"Creating test data in {data_dir.resolve()}...")

# 1. Create OME-TIFF files
print("\n1. Creating OME-TIFF files...")
img = np.random.rand(10, 10, 10).astype(np.uint8)
imwrite(data_dir / 'sample_IM.ome.tif', img)
print("  ✓ sample_IM.ome.tif")

segpred = np.random.randint(0, 3, (10, 10, 10)).astype(np.uint8)
imwrite(data_dir / 'sample_segPred.ome.tiff', segpred)
print("  ✓ sample_segPred.ome.tiff")

uncertainty = np.random.rand(10, 10, 10).astype(np.float32) * 0.5
imwrite(data_dir / 'sample_uncertainty.tif', uncertainty)
print("  ✓ sample_uncertainty.tif")

# 2. Create temp files
print("\n2. Creating temp files...")
segpred_temp = np.random.randint(0, 3, (10, 10, 10)).astype(np.uint8)
imwrite(data_dir / 'Box32x32_segPred_temp.tif', segpred_temp)
print("  ✓ Box32x32_segPred_temp.tif")

uncertainty_temp = np.random.rand(10, 10, 10).astype(np.float32) * 0.6
imwrite(data_dir / 'Box32x32_uncertainty_temp.tif', uncertainty_temp)
print("  ✓ Box32x32_uncertainty_temp.tif")

# 3. Create segments.json with done/undone segments
print("\n3. Creating segments_with_done.json...")
segments_with_done = [
    {
        "name": "Segment_1",
        "label": 1,
        "uncertainty": 0.9,
        "count": 150,
        "coords": None,
        "done": True
    },
    {
        "name": "Segment_2",
        "label": 2,
        "uncertainty": 0.8,
        "count": 120,
        "coords": None,
        "done": False
    },
    {
        "name": "Segment_3",
        "label": 3,
        "uncertainty": 0.7,
        "count": 100,
        "coords": None,
        "done": False
    },
    {
        "name": "Noise",
        "label": 99,
        "uncertainty": 0.9999,
        "count": 20,
        "coords": None,
        "done": False
    }
]

with (data_dir / 'segments_with_done.json').open('w') as f:
    json.dump(segments_with_done, f, indent=2)
print("  ✓ segments_with_done.json")

# 4. Create sample labels file
print("\n4. Creating sample_labels.tif...")
labels = np.zeros((10, 10, 10), dtype=np.int32)
labels[2:5, 2:5, 2:5] = 1  # Label 1
labels[6:8, 6:8, 6:8] = 2  # Label 2
imwrite(data_dir / 'sample_labels.tif', labels)
print("  ✓ sample_labels.tif")

# 5. Create sample cache.json
print("\n5. Creating sample_cache.json...")
cache_data = {
    "priorities": {
        "test_dataset": 0.85,
        "sample_dataset": 0.72
    },
    "datasets": [
        {
            "base_name": "test_dataset",
            "raw_file": "test_IM.tif",
            "segpred_file": "test_segPred.tif",
            "uncertainty_file": "test_uncertainty.tif",
            "priority": 0.85,
            "has_temp": False,
            "temp_segpred": None,
            "temp_uncertainty": None,
            "has_segmentation": True,
            "segmentation_dir": ".vessqc_segmentations"
        }
    ],
    "last_modified": 1696950000.0
}

with (data_dir / 'sample_cache.json').open('w') as f:
    json.dump(cache_data, f, indent=2)
print("  ✓ sample_cache.json")

print("\n" + "="*60)
print("✅ All test data files created successfully!")
print("="*60)
print(f"\nFiles created in: {data_dir.resolve()}")
print("\nYou can now run the tests:")
print("  pytest src/vessqc/_tests/ -v")

