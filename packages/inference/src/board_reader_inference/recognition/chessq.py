"""ChessQ Lite: Lichess's end-to-end board recognizer, the core :class:`~.recognizer.Recognizer`.

``BR_CHESSQ=<model.onnx>`` serves it (:func:`board_reader_inference.recognition.recognizer.build_from_env`,
DESIGN D10). The model reads the **whole frame** and finds the board and its
orientation itself: one 644x644 image in, a 13-class distribution for each of the
64 squares out, in the chess frame. So there is no square cropping, no
rectification, no exemplar bank and no calibration here, and this module imports
nothing from the crop-classifier stack.

Model contract (the release's descriptor JSON, ``preprocessing_version
rgb-imagenet-f32-bilinear-ignore-exif-v1``, and the reference port in
``scratch/experiments/chessq-lite-probe/cq_bench.py``):

* input ``image``: float32 NCHW ``(1, 3, 644, 644)`` — the whole frame, RGB,
  resized to 644x644 *ignoring* aspect ratio, ``/255``, ImageNet mean/std. The
  resize is ``cv2.INTER_AREA``, what the benchmark measured (the descriptor says
  antialiased bilinear; on a downscale the two are the same filter family, and the
  benchmark's numbers are what we hold).
* output ``logits``: ``(1, 64, 13)``, squares in FEN order (a8, b8, ..., h8, a7,
  ..., h1), classes ``. P N B R Q K p n b r q k``. FEN order is exactly
  ``square_dist``'s row-major rank-8-first grid, and the class order is
  :data:`~board_reader_core.protocol.CLASSES` with ``.`` for ``empty`` — the
  mapping is still *derived* (:data:`_TO_PROTOCOL`), so a release that reordered
  its classes would fail the protocol check at import rather than serve a board
  with white and black swapped.
* output ``direction``: ``(1, 2)``, the unit vector from e1 toward e8 in the
  image as ``(cos, sin)`` with cos pointing up and sin pointing right (the
  author's PORTABLE.md). :func:`direction_degrees` turns it into the bearing
  :class:`~.recognizer.Recognition` carries; the ingest stream's orientation
  guard (:class:`~.ingest.OrientationGuard`) reads it.

**Not calibrated.** The distribution is the model's softmax, unchanged: the
black-bishop demotion the old crop-classifier stack applied was measured on that
stack's distribution, and its absolute margin means nothing on a different model.

License: the weights are PolyForm Noncommercial 1.0.0 (THIRD-PARTY.md); they live
under the gitignored ``models/`` and are never committed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import CLASSES, GRID, NUM_CLASSES

from board_reader_inference.recognition.onnxrt import cuda_providers
from board_reader_inference.recognition.recognizer import Recognition

#: ``(1, 3, 644, 644)`` float32 -> its ``(1, 64, 13)`` logits and ``(1, 2)``
#: direction, the release's two outputs: the model, or a test fake.
ModelFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

#: The release's output names, in the order :data:`ModelFn` returns them.
OUTPUTS = ["logits", "direction"]

#: Side of the square the model reads, in pixels (``input.shape`` in the descriptor).
INPUT_PX = 644

#: The release's class order (``output.classes``), ``.`` meaning an empty square.
CHESSQ_CLASSES = ".PNBRQKpnbrqk"

#: Column ``j`` of a protocol-ordered distribution is column ``_TO_PROTOCOL[j]`` of
#: the model's output. The identity today; derived so it cannot silently be wrong.
_TO_PROTOCOL = np.array(
    [CHESSQ_CLASSES.index("." if name == "empty" else name) for name in CLASSES]
)

#: ``input.normalization`` in the descriptor (ImageNet), per RGB channel.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(image: np.ndarray) -> np.ndarray:
    """A BGR uint8 frame -> the model's ``(1, 3, 644, 644)`` float32 input."""
    resized = cv2.resize(image, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA)
    rgb = resized[None, ..., ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - _MEAN) / _STD).transpose(0, 3, 1, 2))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def square_dist_from_logits(logits: np.ndarray) -> np.ndarray:
    """The model's ``(1, 64, 13)`` logits -> an ``(8, 8, 13)`` ``square_dist``."""
    arr = np.asarray(logits, dtype=np.float32)
    if arr.shape != (1, GRID * GRID, NUM_CLASSES):
        raise ValueError(
            f"ChessQ returned {arr.shape}, expected (1, {GRID * GRID}, {NUM_CLASSES})"
        )
    return _softmax(arr[0][:, _TO_PROTOCOL]).reshape(GRID, GRID, NUM_CLASSES)


def direction_degrees(direction: np.ndarray) -> float | None:
    """The model's ``(1, 2)`` ``(cos, sin)`` -> degrees clockwise from up.

    ``None`` when the vector is not finite or has no length, so no bearing at all
    is reported rather than an arbitrary one.
    """
    arr = np.asarray(direction, dtype=np.float64)
    if arr.shape != (1, 2):
        raise ValueError(f"ChessQ returned direction {arr.shape}, expected (1, 2)")
    cos, sin = float(arr[0, 0]), float(arr[0, 1])
    if not (math.isfinite(cos) and math.isfinite(sin)) or math.hypot(cos, sin) == 0.0:
        return None
    return math.degrees(math.atan2(sin, cos))


def _onnx_model_fn(model_path: Path) -> ModelFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=cuda_providers())

    def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # By name, never by position: what a release exports is its own business.
        logits, direction = sess.run(OUTPUTS, {"image": batch})
        return logits, direction

    return run


class ChessQRecognizer:
    """Stateless whole-frame recognizer over the ChessQ Lite ONNX model.

    ``model`` is a path to the ``.onnx`` file, or — for tests — a :data:`ModelFn`.
    """

    name = "chessq"

    def __init__(self, model: str | Path | ModelFn) -> None:
        if callable(model):
            self.model_fn: ModelFn = model
            self.model_path: str | None = None
        else:
            self.model_fn = _onnx_model_fn(Path(model))
            self.model_path = str(model)

    def read(self, image: np.ndarray) -> Recognition:
        """The model's chess-frame distribution and direction for a whole BGR frame."""
        logits, direction = self.model_fn(preprocess_frame(image))
        return Recognition(square_dist_from_logits(logits), direction_degrees(direction))
