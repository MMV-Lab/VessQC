"""
io_utils.py
===========

Functions for data input and output.

Functions
---------
save_npy
    Save the array in .npy format.
load_npy
    Load an data array from a .npy file
build_filename
    Generate a filename
"""

# Copyright © Peter Lampen, ISAS Dortmund, 2026
# (07.05.2026)

from dataclasses import asdict
import json
import numpy as np
from pathlib import Path
import tempfile
from typing import List

from .models import Segment

def save_npy(array: np.ndarray, filename: Path):
    """
    Save the array in .npy format
    
    Parameters
    ----------
    array : np.ndarray
        Data array
    filename : Path
        Output .npy file
    """

    # (13.03.2026)
    print('Save file', filename)

    with filename.open("wb") as f:
        np.save(f, array)

def load_npy(filename: Path):
    """
    Load an data array from a .npy file

    Parameters
    ----------
    filename : Path
        Input .npy file

    Returns
    -------
    np.ndarray
    """

    # (13.03.2026)
    print('Read file', filename)

    with filename.open("rb") as f:
        return np.load(f)

def save_segments(segments: List[Segment], filename: Path):
    """
    Save Segment objects as a JSON file.

    Parameters
    ----------
    segments : list of Segment
        Segment metadata
    filename : Path
        Output JSON file
    """

    # (07.05.2026)
    print('Save file', filename)

    with filename.open('w', encoding='utf-8') as f:
        # Convert dataclass into dictionaries before JSON export.
        data = [asdict(seg) for seg in segments]
        json.dump(data, f, indent=2)

def load_segments(filename: Path) -> List[Segment]:
    """
    Load Segment object from a JSON file.

    Parameters
    ----------
    filename : Path
        Input JSON file

    Returns
    -------
    list of Segment
        loaded segment metadata
    """

    # (07.06.2026)
    print('Read file', filename)

    with filename.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Reconstruct Segment objects from dictionaries.
    segments = [Segment(**seg) for seg in data]
    return segments

def build_filename(stem: str, suffix: str):
    """
    Generate a filename

    Parameters
    ----------
    stem : str
        Name of the temporary file
    suffix : str
        File name extension

    Returns
    -------
        Name of the temporary file
    """

    # (24.04.2026)
    temp = Path(tempfile.gettempdir())
    return temp.joinpath(stem).with_suffix(suffix)
