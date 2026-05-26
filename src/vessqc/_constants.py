"""
Shared thresholds for segmentation and UI display.
"""

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


def segment_sort_key(segment: dict) -> tuple:
    """Sort segments: highest uncertainty, then most pixels, then lowest label id."""
    return (
        -float(segment.get('uncertainty', 0)),
        -int(segment.get('counts', 0)),
        int(segment.get('label', 0)),
    )


def short_segment_name(name: str, label: int) -> str:
    """Compact display name for segment rows (e.g. Segment_11 → S_11)."""
    if name == 'Noise':
        return name
    if name.startswith('Segment_'):
        return f'S_{label}'
    return name


def format_segment_row_label(segment: dict, *, short_name: bool = False) -> str:
    """e.g. S_11 (0.850) [1234]"""
    name = segment.get('name', '')
    if short_name:
        name = short_segment_name(name, int(segment.get('label', 0)))
    uncertainty = float(segment.get('uncertainty', 0))
    counts = int(segment.get('counts', 0))
    return f'{name} ({uncertainty:.3f}) [{counts}]'
