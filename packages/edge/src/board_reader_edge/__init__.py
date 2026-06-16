"""board-reader-edge — the edge node (the "Pi"), DESIGN.md L1.

Captures frames, (later) detects + rectifies the board, measures stability, and
forwards frames over the wire. Depends ONLY on :mod:`board_reader_core`; it must
never import ``board_reader_inference`` — the two layers meet only on the wire
(and packaging enforces it: this package does not depend on the inference package).
"""

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.source import (
    FrameSource,
    ImageDirSource,
    RepeatSource,
    SyntheticSource,
    VideoFileSource,
)

__all__ = [
    "FrameSource",
    "VideoFileSource",
    "ImageDirSource",
    "RepeatSource",
    "SyntheticSource",
    "FileEdgeAdapter",
]
