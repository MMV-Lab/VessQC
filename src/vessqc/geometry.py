"""
geometry.py
===========

Functions for geometric operations on 3D image volumes and segments.

Functions
---------
compute_bbox
    Determine a bounding box
expand_bbox
    Enlarge the bounding box
crop_volumes
    Cropping the data
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2026
# (07.05.2026)

import numpy as np

def compute_bbox(mask: np.ndarray):
    """Determine a bounding box"""

    # (06.03.2026)
    coords = np.argwhere(mask)

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    return [[min_z, min_y, min_x], [max_z, max_y, max_x]]

def expand_bbox(b_box: list, shape: tuple, margin_factor: float):
    """
    Enlarge the bounding box

    Parameters
    ----------
    b_box : list
        Bounding box
    shape : tuple
        Shape of the image
    margin_factor : float
        Factor for enlarging the b_box

    Returns
    -------
    b_box : list
        Updated bounding box
    """

    # (06.03.2026)
    (min_z, min_y, min_x), (max_z, max_y, max_x) = b_box

    size_z = max_z - min_z + 1
    size_y = max_y - min_y + 1
    size_x = max_x - min_x + 1

    size = max(size_x, size_y, size_z)
    margin = int(size * margin_factor / 2)

    start_z = max(min_z - margin, 0)
    start_y = max(min_y - margin, 0)
    start_x = max(min_x - margin, 0)

    end_z   = min(max_z + margin + 1, shape[0])
    end_y   = min(max_y + margin + 1, shape[1])
    end_x   = min(max_x + margin + 1, shape[2])

    return [[start_z, start_y, start_x], [end_z, end_y, end_x]]

def crop_volumes(b_box: list, image: np.ndarray, segPred: np.ndarray,
    labels: np.ndarray, label: np.int32):
    """
    Cropping the data

    Parameters
    ----------
    b_box : list
        Bounding box
    image : np.ndarray
        3D array with image data
    segPred : np.ndarray
        3D array with the predicted segmentation data
    labels : np,ndarray
        3D array with segmentation labels
    label : np.int32
        Singel label

    Returns
    -------
    dict
        Dictionary with the keys:
        - image : np.ndarray
        - segPred : np.ndarray
        - labels : np.ndarray
    """

    # (06.03.2026)
    (min_z, min_y, min_x), (max_z, max_y, max_x) = b_box

    cropped_image   = image[  min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_segPred = segPred[min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_labels  = labels[ min_z:max_z, min_y:max_y, min_x:max_x]

    # Keep only inside the box
    masked_labels = np.where(cropped_labels == label, label, 0)

    return {
        "image": cropped_image,
        "segPred": cropped_segPred,
        "labels": masked_labels
    }
