"""
models.py
=========

Data class for the metadata of segmented vessel region.

Classes
-------
Segment
    Metadata for a segmented vessel region.
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2026
# (29.04.2024)

from dataclasses import dataclass
from typing import Optional

@dataclass
class Segment:
    """
    Metadata for a segmented vessel region.

    Attributes
    ----------
    name : str
        Display name of the segment.
    label : int
        Unique label value in the label volume.
    uncertainty : float
        Uncertainty assigned to the segment.
    count : int
        Number of voxels in the segment.
    coords : list | None
        Bounding box coordinates of the cropped region.
    done : bool
        True if the segment has already been processed.
    """

    # (29.04.2026)
    name: str
    label: str
    uncertainty: float
    count: int
    coords: Optional[list] = None
    done: bool = False
