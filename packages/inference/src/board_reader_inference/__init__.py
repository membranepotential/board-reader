"""board-reader-inference — the inference node, DESIGN.md L2.

Stateless per-frame position estimation, the position stream hub, and the WS
gateway. Depends ONLY on :mod:`board_reader_core`; it must never import
``board_reader_edge`` — the two layers meet only on the wire (and packaging
enforces it: this package does not depend on the edge package).
"""

from board_reader_inference.estimator import Estimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.server import create_app

__all__ = ["Estimator", "PositionHub", "StubEstimator", "create_app"]
