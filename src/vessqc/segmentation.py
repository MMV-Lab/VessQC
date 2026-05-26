"""
segmentation.py
===============

Functions for uncertainty-based segmentation and segment processing.

Functions
---------
segment_uncertainties
    Segmend 3D voxels by unique uncertainty values
label_value_sparse
    Segments contiguous voxels with similar uncertainty and assigns them
    unique global labels.
merge_labels
    Merge results of segmentation into a single label volume
merge_small_segments
    Merge labels that occur less than 'min_size' times into a new label.
create_segments
    Create segment metadata from labels and uncertainties.
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2026
# (07.05.2026)

from joblib import Parallel, delayed
from .models import Segment
import numpy as np
from scipy import ndimage

def segment_uncertainties(uncertainty: np.ndarray):
    """
    Segmend 3D voxels by unique uncertainty values

    Parameters
    ----------
    uncertainty : np.ndarray
        3D array of uncertainty values.

    Returns
    -------
    result : list of dict
        Each dict contains 'incices', 'global_labels', 'uncert', 'num_features'
    """

    # (12.02.2026)
    unique_uncertainties = np.unique(uncertainty)
    unique_uncertainties = unique_uncertainties[unique_uncertainties > 0]
    num_uncert = len(unique_uncertainties)

    results = Parallel(n_jobs=-1)(
        delayed(label_value_sparse)(uncertainty, uncert, idx, num_uncert)
        for idx, uncert in enumerate(unique_uncertainties, start=1)
    )

    return [r for r in results if r is not None]

def label_value_sparse(uncertainty, uncert, idx, num_uncert):
    """
    Segments contiguous voxels with similar uncertainty and assigns them
    unique global labels.

    Parameters
    ----------
    uncertainty : np.ndarray
        3D array with information on the uncertainty of the calculated
        data points
    uncert : float
        Single uncertainty value
    idx : int
        Index of the unique uncertainty value
    num_uncert : int
        Number of unique uncertainty values

    Returns
    -------
    dict
        Dictionary with the keys:
        - indices : tuple of np.ndarray
            Indices of voxels belonging to the segment
        - global_labels : np.ndarray
            Global label values at the given indices
        - uncert : float
            Uncertainty value of the segment
        - num_features : int
            Number of found features

    None
        If no voxels match the criteria.
    """

    # (03.07.2025)
    tolerance = 1e-2
    structure = np.ones((3, 3, 3), dtype=int)       # Connectivity array
    mask = np.abs(uncertainty - uncert) < tolerance
    if not np.any(mask):
        return None

    labels, num_features = ndimage.label(mask, structure)    # Segmentation
    if num_features == 0:
        return None

    # Calculate global unique labels
    # labels = 1, 2, 3, ... num_features
    # e.g. global_labels = 3, 23, 43, ... for num_uncert = 20, idx = 3
    global_labels = idx + (labels - 1) * num_uncert
    global_labels[labels == 0] = 0

    indices = np.where(mask)
    result = dict(
        indices = indices,
        global_labels = global_labels[indices],
        uncert = uncert,
        num_features = num_features
    )
    return result

def merge_labels(results: list, shape: tuple):
    """
    Merge results of segmentation into a single label volume

    Parameters
    ----------
    results : list of dict
        Output of segment_uncertainties
    shape : tuple
        Shape of the original image

    Returns
    -------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping labels -> uncertainty
    """

    # (17.02.2026)
    labels = np.zeros(shape, dtype=np.int32)
    uncert_values = {0: 0.0}

    # Reconstruct der labels array with global labels
    for result in results:
        indices = result['indices']
        global_labels = result['global_labels']
        labels[indices] = global_labels

        unique_labels = np.unique(global_labels)
        unique_labels = unique_labels[unique_labels != 0]

        for lbl in unique_labels:
            uncert_values[lbl] = result['uncert']

    return labels, uncert_values

def merge_small_segments(labels: np.ndarray, uncert_values: dict, min_size: int):
    """
    Merge labels that occur less than 'min_size' times into a new label.

    Parameters
    ----------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping label -> uncertainty
    min_size : int
        Minimum voxel count to keep a segment

    Returns
    -------
    labels : np.ndarray
        Updated label volume
    uncert_values : dict
        Updated uncertainty dictionary
    """

    # (06.03.2026)
    # Find all labels that appear less than min_size times
    counts = np.bincount(labels.ravel())
    small_labels = np.where(counts < min_size)[0]
    small_labels = small_labels[small_labels != 0]

    # Replaces all labels that occur less than min_size with max_label
    if len(small_labels) > 0:
        max_label = np.max(labels) + 1
        mask = np.isin(labels, small_labels)

        labels[mask] = max_label
        uncert_values[max_label] = 0.9999

    return labels, uncert_values

def create_segments(labels: np.ndarray, uncert_values: dict):
    """
    Create segment metadata from labels and uncertainties.

    Parameters
    ----------
    labels : np.ndarray
        Label volume
    uncert_values : dict
        Dictionary mapping labels -> uncertainty

    Returns
    -------
    segments : List
        List of Segments
    """

    # (18.02.2026, revised 29.04.2026)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    counts = np.bincount(labels.ravel())

    segments = [
        Segment(
            name = f"Segment_{i}",
            label = int(label),
            uncertainty = float(uncert_values.get(label, 0.0)),
            count = int(counts[label]),
        )
        for i, label in enumerate(unique_labels, start=1)
    ]

    # Sort by uncertainty ascending
    segments.sort(key=lambda x: x.uncertainty)
    return segments
