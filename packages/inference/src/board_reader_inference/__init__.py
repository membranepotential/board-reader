"""board-reader-inference — the inference node, DESIGN.md L2.

Stateless per-frame position reading, the position stream hub, and the WS
gateway. Depends ONLY on :mod:`board_reader_core`; it must never import
``board_reader_edge`` — the two layers meet only on the wire (and packaging
enforces it: this package does not depend on the edge package).
"""

from board_reader_inference.recognition.recognizer import Recognizer, StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

__all__ = ["PositionHub", "Recognizer", "StubRecognizer", "create_app"]
