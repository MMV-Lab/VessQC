"""
Shared thresholds for segmentation and UI display.
"""

from typing import Union

from .models import Segment

# Voxels in segments smaller than this are merged into Noise at segmentation time.
NOISE_MIN_SIZE = 50

# Default spinbox value: segments below this are merged into Noise (reversible down to NOISE_MIN_SIZE).
DISPLAY_MIN_SIZE_DEFAULT = 200

# Lower bound for the spinbox (cannot merge segments already in the <50 px Noise floor).
DISPLAY_MIN_SIZE_MIN = NOISE_MIN_SIZE

# Qt treats '&' as mnemonic marker; use '&&' for a literal ampersand in labels.
TOP5_PANEL_TITLE = 'Top 5 segments by (uncertainty) && [size]'

# Minimum width for the segment list popup (avoids horizontal clipping).
SEGMENT_LIST_POPUP_MIN_WIDTH = 400

# Dataset ordering in the loading dialog
DATASET_SORT_MAX = 'max'
DATASET_SORT_MEAN = 'mean'

SegmentLike = Union[Segment, dict]


def segment_name(segment: SegmentLike) -> str:
    if isinstance(segment, Segment):
        return segment.name
    return str(segment.get('name', ''))


def set_segment_name(segment: SegmentLike, name: str) -> None:
    if isinstance(segment, Segment):
        segment.name = name
    else:
        segment['name'] = name


def segment_label(segment: SegmentLike) -> int:
    if isinstance(segment, Segment):
        return int(segment.label)
    return int(segment.get('label', 0))


def segment_uncertainty(segment: SegmentLike) -> float:
    if isinstance(segment, Segment):
        return float(segment.uncertainty)
    return float(segment.get('uncertainty', 0))


def segment_count(segment: SegmentLike) -> int:
    if isinstance(segment, Segment):
        return int(segment.count)
    return int(segment.get('count', segment.get('counts', 0)))


def set_segment_count(segment: SegmentLike, count: int) -> None:
    if isinstance(segment, Segment):
        segment.count = int(count)
    else:
        segment['count'] = int(count)
        segment['counts'] = int(count)


def segment_done(segment: SegmentLike) -> bool:
    if isinstance(segment, Segment):
        return bool(segment.done)
    return bool(segment.get('done', False))


def set_segment_done(segment: SegmentLike, done: bool) -> None:
    if isinstance(segment, Segment):
        segment.done = bool(done)
    else:
        segment['done'] = bool(done)


def segment_coords(segment: SegmentLike):
    if isinstance(segment, Segment):
        return segment.coords
    return segment.get('coords')


def set_segment_coords(segment: SegmentLike, coords) -> None:
    if isinstance(segment, Segment):
        segment.coords = coords
    else:
        segment['coords'] = coords


def segment_sort_key(segment: SegmentLike) -> tuple:
    """Sort segments: highest uncertainty, then most pixels, then lowest label id."""
    return (
        -segment_uncertainty(segment),
        -segment_count(segment),
        segment_label(segment),
    )


def short_segment_name(name: str, label: int) -> str:
    """Compact display name for segment rows (e.g. Segment_11 → S_11)."""
    if name == 'Noise':
        return name
    if name.startswith('Segment_'):
        return f'S_{label}'
    return name


def format_segment_row_label(segment: SegmentLike, *, short_name: bool = False) -> str:
    """e.g. S_11 (0.850) [1234]"""
    name = segment_name(segment)
    if short_name:
        name = short_segment_name(name, segment_label(segment))
    uncertainty = segment_uncertainty(segment)
    counts = segment_count(segment)
    return f'{name} ({uncertainty:.3f}) [{counts}]'
