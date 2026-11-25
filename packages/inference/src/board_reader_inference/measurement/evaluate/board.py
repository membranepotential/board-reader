"""Read one labeled photo and score its distribution against the truth, square by square."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_core.protocol import CLASSES, GRID

from board_reader_inference.measurement.evaluate.errors import SquareError, square_errors
from board_reader_inference.measurement.labels import LabeledBoard, best_rotation
from board_reader_inference.recognition.recognizer import Recognizer


def round_opt(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


@dataclass(frozen=True)
class BoardResult:
    board_id: str
    source: str
    truth: str
    predicted: str
    correct: int
    confidence: float  # mean per-square max probability
    errors: tuple[SquareError, ...]
    #: Quarter-turns under which the read agrees best with the truth; 0 = same frame.
    rotation: int = 0

    @property
    def perfect(self) -> bool:
        return self.correct == GRID * GRID

    def to_dict(self) -> dict:
        return {
            "board_id": self.board_id,
            "source": self.source,
            "truth": self.truth,
            "predicted": self.predicted,
            "correct": self.correct,
            "total": GRID * GRID,
            "perfect": self.perfect,
            "confidence": round(self.confidence, 4),
            "rotation": self.rotation,
            "errors": [
                {
                    "square": e.square,
                    "truth": e.truth,
                    "predicted": e.predicted,
                    "kind": e.kind,
                    "confidence": round_opt(e.confidence),
                }
                for e in self.errors
            ],
        }


def read_board(recognizer: Recognizer, board: LabeledBoard) -> np.ndarray:
    """Run the recognizer over one labeled photo -> its (8, 8, 13) distribution."""
    import cv2

    image = cv2.imread(str(board.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {board.image}")
    return np.asarray(recognizer.read(image).square_dist, dtype=np.float32)


def score_board(board: LabeledBoard, dist: np.ndarray) -> BoardResult:
    """Compare one distribution against its truth, square by square."""
    from board_reader_core.protocol import encode_placement

    idx = dist.argmax(axis=-1)
    conf = dist.max(axis=-1)
    predicted = encode_placement(
        [[CLASSES[int(idx[r, c])] for c in range(GRID)] for r in range(GRID)]
    )
    errors = square_errors(board.placement, predicted, conf)
    return BoardResult(
        board_id=board.board_id,
        source=board.source,
        truth=board.placement,
        predicted=predicted,
        correct=GRID * GRID - len(errors),
        confidence=float(conf.mean()),
        errors=errors,
        rotation=best_rotation(board.placement, predicted),
    )
