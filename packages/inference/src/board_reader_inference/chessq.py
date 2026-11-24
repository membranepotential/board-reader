"""ChessQ Lite: Lichess's end-to-end board recognizer as an opt-in estimator.

``BR_CHESSQ=<model.onnx>`` serves it (see :func:`board_reader_inference.server._default_estimator`).
The model reads the **whole frame** and finds the board and its orientation itself:
one 644x644 image in, a 13-class distribution for each of the 64 squares out. So
there is no square cropping, no rectification and no exemplar bank here — which is
why it sits beside the crop classifiers (:mod:`.trained`) rather than inside them.

Model contract (the release's descriptor JSON, ``preprocessing_version
rgb-imagenet-f32-bilinear-ignore-exif-v1``, and the reference port in
``scratch/experiments/chessq-lite-probe/cq_bench.py``):

* input ``image``: float32 NCHW ``(1, 3, 644, 644)`` — the whole frame, RGB,
  resized to 644x644 *ignoring* aspect ratio, ``/255``, ImageNet mean/std. That is
  :func:`board_reader_inference.trained.preprocess` on the resized frame, so the
  normalisation has one implementation. The resize is ``cv2.INTER_AREA``, what the
  benchmark measured (the descriptor says antialiased bilinear; on a downscale the
  two are the same filter family, and the benchmark's numbers are what we hold).
* output ``logits``: ``(1, 64, 13)``, squares in FEN order (a8, b8, ..., h8, a7,
  ..., h1), classes ``. P N B R Q K p n b r q k``. FEN order is exactly
  ``square_dist``'s row-major rank-8-first grid, and the class order is
  :data:`~board_reader_core.protocol.CLASSES` with ``.`` for ``empty`` — the
  mapping is still *derived* (:data:`_TO_PROTOCOL`), so a release that reordered
  its classes would fail the protocol check at import rather than serve a board
  with white and black swapped.

**Corners are not an input.** The frames the edge sends still carry them, and the
gateway still routes on them (:mod:`.routing`); this estimator declares the photo
path so it is handed the ORIGINAL pixels rather than a server-side warp, and then
ignores the quad. Cropping to the board was measured at ±0.5 points (no gain,
docs/plans/2026-10-09-chessqueries-handoff.md), and a frame without corners is read
the same way as one with them. Readiness does not change either: the estimator is
always ready, every routed frame is read, and every read is ``status="ok"`` (an
estimator never judges legality, DESIGN D1).

**Not calibrated.** The read is built straight from the model's softmax, NOT
through :func:`~board_reader_inference.estimator.read_event`: the black-bishop
demotion there was measured on our own trained stack's distribution, and its
absolute margin means nothing on a different model. And nothing calibrates it —
:class:`~board_reader_inference.session.EstimatorSession` refuses ``/calibrate``
while it is active, since an exemplar bank would replace it, not boost it.

License: the weights are PolyForm Noncommercial 1.0.0 (THIRD-PARTY.md); they live
under the gitignored ``models/`` and are never committed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    NUM_CLASSES,
    ClockState,
    PositionEvent,
    iso_utc,
)
from board_reader_core.wire import Frame

from board_reader_inference.onnxrt import cuda_providers
from board_reader_inference.routing import RECTIFIED_GEOMETRY
from board_reader_inference.squares import CropGeometry
from board_reader_inference.trained import LogitsFn, preprocess, softmax

#: Side of the square the model reads, in pixels (``input.shape`` in the descriptor).
INPUT_PX = 644

#: The release's class order (``output.classes``), ``.`` meaning an empty square.
CHESSQ_CLASSES = ".PNBRQKpnbrqk"

#: Column ``j`` of a protocol-ordered distribution is column ``_TO_PROTOCOL[j]`` of
#: the model's output. The identity today; derived so it cannot silently be wrong.
_TO_PROTOCOL = np.array(
    [CHESSQ_CLASSES.index("." if name == "empty" else name) for name in CLASSES]
)


def preprocess_frame(image: np.ndarray) -> np.ndarray:
    """A BGR uint8 frame -> the model's ``(1, 3, 644, 644)`` float32 input."""
    resized = cv2.resize(image, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA)
    return preprocess([resized])


def square_dist_from_logits(logits: np.ndarray) -> np.ndarray:
    """The model's ``(1, 64, 13)`` logits -> an ``(8, 8, 13)`` ``square_dist``."""
    arr = np.asarray(logits, dtype=np.float32)
    if arr.shape != (1, GRID * GRID, NUM_CLASSES):
        raise ValueError(
            f"ChessQ returned {arr.shape}, expected (1, {GRID * GRID}, {NUM_CLASSES})"
        )
    return softmax(arr[0][:, _TO_PROTOCOL]).reshape(GRID, GRID, NUM_CLASSES)


def _onnx_logits_fn(model_path: Path) -> LogitsFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=cuda_providers())

    def run(batch: np.ndarray) -> np.ndarray:
        # By name: the release has a second output (``direction``) beside it.
        return sess.run(["logits"], {"image": batch})[0]

    return run


class ChessQEstimator:
    """Stateless per-frame estimator over the ChessQ Lite ONNX model.

    ``model`` is a path to the ``.onnx`` file, or — for tests — a callable
    ``(1, 3, 644, 644) float32 -> (1, 64, 13) logits``.
    """

    # Declares the photo path so the gateway routes the original pixels here
    # (:func:`board_reader_inference.routing.photo_estimator`) instead of warping.
    crop_mode = "photo"
    # What ``?debug=1`` draws over the frame: the flat squares of the quad the
    # frame carries. The model cut no boxes, so this is the operator's view of
    # where the corners say the board is, labeled with what ChessQ read there.
    geometry: CropGeometry = RECTIFIED_GEOMETRY
    square_size: int | None = None

    def __init__(
        self, model: str | Path | LogitsFn, *, orientation: str = "a1_bottom_left"
    ) -> None:
        if callable(model):
            self.logits_fn: LogitsFn = model
            self.model_path: str | None = None
        else:
            self.logits_fn = _onnx_logits_fn(Path(model))
            self.model_path = str(model)
        self.orientation = orientation

    def square_dist(self, image: np.ndarray) -> np.ndarray:
        """The model's ``(8, 8, 13)`` distribution for a whole BGR frame."""
        return square_dist_from_logits(self.logits_fn(preprocess_frame(image)))

    def estimate(self, frame: Frame) -> PositionEvent:
        # Deliberately NOT read_event: demote_attractor's margin was fitted on our
        # trained stack's distribution, never on this model's (module docstring).
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=self.square_dist(frame.image),
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        """The photo path the gateway routes to; ``corners`` are not used.

        The model finds the board in the whole frame by itself (module docstring).
        """
        return self.estimate(frame)
