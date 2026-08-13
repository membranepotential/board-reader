"""board-reader-core — the shared contract (DESIGN.md §5, §4.5).

Imported by both the edge and the inference node. Contains no node-specific logic:
the L3 position protocol, the L1->L2 wire envelope, and sim helpers.
"""

from board_reader_core.corners import parse_corners
from board_reader_core.protocol import (
    CLASSES,
    EMPTY_PLACEMENT,
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
    placement_from_dist,
)
from board_reader_core.wire import Frame, decode_jpeg, encode_jpeg

__all__ = [
    "CLASSES",
    "EMPTY_PLACEMENT",
    "START_PLACEMENT",
    "ClockState",
    "Frame",
    "PositionEvent",
    "decode_jpeg",
    "dist_from_placement",
    "encode_jpeg",
    "iso_utc",
    "parse_corners",
    "placement_from_dist",
]
