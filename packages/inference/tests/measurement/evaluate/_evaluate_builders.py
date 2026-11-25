"""Builders for the evaluation tests: labeled boards, canned distributions, a fake recognizer.

Shared by every ``test_*.py`` here, which import it by name (the same arrangement
as ``tests/_sim_recognizer.py``), so keep the basename unique across the workspace.
"""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import CLASS_INDEX, GRID, NUM_CLASSES, dist_from_placement

from board_reader_inference.measurement.labels import LabeledBoard
from board_reader_inference.recognition.recognizer import Recognition

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"
CORNERS = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

# The fakes never look at pixels, but read_board decodes the photo like the real
# thing does — so every labeled board needs a file that actually exists. The
# ``_photo_on_disk`` fixture in conftest.py writes it and sets this.
PHOTO: object = "unused.jpg"


def labeled_board(board_id: str = "b0", placement: str = MIDGAME, image=None) -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id,
        image=image if image is not None else PHOTO,
        frame="chess",
        corners=None,
        placement=placement,
        source="confirmed",
    )


def corner_frame_board(board_id: str) -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id, image=PHOTO, frame="corners", corners=CORNERS,
        placement=MIDGAME, source="confirmed",
    )


def dist_with(placement: str, overrides: dict[str, tuple[str, float]]) -> np.ndarray:
    """A peaked distribution for ``placement``, then specific squares overwritten.

    ``overrides`` maps a square name to ``(class, confidence)`` — the lever these
    tests use to plant a *confidently wrong* square.
    """
    dist = dist_from_placement(placement, confidence=0.9).copy()
    for square, (cls, conf) in overrides.items():
        c, r = "abcdefgh".index(square[0]), GRID - int(square[1])
        dist[r, c] = (1.0 - conf) / (NUM_CLASSES - 1)
        dist[r, c, CLASS_INDEX[cls]] = conf
    return dist


class FakeRecognizer:
    """Answers with the canned distributions in order, one per read — no model.

    A recognizer is handed pixels, not a board id, so the fake hands out its
    distributions in the order the boards are read (``evaluate`` reads them in
    the order given).
    """

    name = "fake"
    frame = "chess"
    model_path = None

    def __init__(self, dists: dict[str, np.ndarray]) -> None:
        self._dists = iter(dists.values())
        self.reads = 0

    def read(self, image: np.ndarray) -> Recognition:
        self.reads += 1
        return Recognition(next(self._dists))


def chess_board(board_id: str = "c0") -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id, image=PHOTO, frame="chess", corners=None,
        placement=MIDGAME, source="curated",
    )
