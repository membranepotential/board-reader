"""Trained piece-classifier estimator (DESIGN §4.6, the zero-shot default).

Wraps the multi-style-trained ONNX crop classifier (2026-07-18 campaign:
92.5%/320 zero-shot on the held-out user set vs 91.9% calibrated kNN). Unlike
:class:`~board_reader_inference.estimator.CentroidEstimator` there is no
embedder/classifier split — the network maps a crop straight to 13 logits in
:data:`board_reader_core.protocol.CLASSES` order.

Preprocessing contract (must match training bit-for-bit — verified against a
fixture produced by the training pipeline): crops arrive as BGR uint8 (cv2),
are converted to RGB, scaled to [0,1], normalized with ImageNet mean/std, and
batched NCHW float32.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from board_reader_core.protocol import (
    NUM_CLASSES,
    ClockState,
    PositionEvent,
    iso_utc,
)
from board_reader_core.wire import Frame

from board_reader_inference.onnxrt import cuda_providers
from board_reader_inference.squares import (
    DEFAULT_GEOMETRY,
    LEGACY_GEOMETRY,
    BboxGeometry,
    CropGeometry,
    crop_squares_photo,
    full_frame_corners,
    geometry_from_dict,
)

# ImageNet normalization, RGB order — the torchvision transform the model
# trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Logits -> distribution temperature 1.0 (plain softmax): the network's own
# confidences are what the event's square_dist should carry.
LogitsFn = Callable[[np.ndarray], np.ndarray]


def preprocess(crops: list[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops -> NCHW float32 batch (RGB, [0,1], ImageNet-normalized)."""
    batch = np.stack([c[..., ::-1] for c in crops]).astype(np.float32) / 255.0
    batch = (batch - _MEAN) / _STD
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _onnx_logits_fn(model_path: str | Path) -> LogitsFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path),
        providers=cuda_providers(),
    )
    name = sess.get_inputs()[0].name

    def run(batch: np.ndarray) -> np.ndarray:
        return sess.run(None, {name: batch})[0]

    return run


class TrainedEstimator:
    """Stateless per-frame estimator backed by the trained ONNX crop classifier.

    ``logits_fn`` is injectable for tests (a callable ``(N,3,224,224) float32 ->
    (N,13) logits``); production use passes a model path.
    """

    # The photo-crop geometry the model was trained with (must match
    # tools/train/cropcache.py).
    crop_mode = "photo"

    def __init__(
        self,
        model: str | Path | LogitsFn,
        *,
        geometry: CropGeometry | None = None,
        square_size: int | None = None,
        orientation: str = "a1_bottom_left",
        batch_size: int = 64,
    ) -> None:
        # Crop geometry SHIPS WITH the model: a `<model>.meta.json` sidecar
        # written at export time carries the geometry and size the network
        # trained on (train/serve must crop identically — a geometry mismatch
        # silently decapitates crops and reads as "the model got worse").
        # An explicit `geometry=` overrides; without a sidecar the historical
        # v1 geometry applies.
        meta: dict = {}
        if not callable(model):
            meta_path = Path(f"{model}.meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            elif geometry is None:
                # Never fall back silently: this exact mismatch (a tall-crop
                # model served on v1's short geometry) shipped once and read as
                # "the model got worse" — kings clipped into bishops.
                print(
                    f"WARNING: no crop-geometry sidecar at {meta_path}; falling back to "
                    f"{LEGACY_GEOMETRY} (the superseded v1 geometry). A model trained on "
                    "tall crops will have its pieces decapitated. Export the sidecar or "
                    "pass geometry= explicitly.",
                    file=sys.stderr,
                )
        self.logits_fn: LogitsFn = (
            model if callable(model) else _onnx_logits_fn(model)
        )
        # Provenance: an accuracy number is only meaningful next to the weights
        # that produced it, and "which model was that?" has already been the
        # wrong answer once (the v1/d16 mix-up). None when driven by a callable.
        self.model_path: str | None = None if callable(model) else str(model)
        if geometry is not None:
            self.geometry: CropGeometry = geometry
        elif "crop_geometry" in meta:
            self.geometry = geometry_from_dict(meta["crop_geometry"])
        elif "headroom" in meta:  # pre-ADT sidecars carried bare scalars
            self.geometry = geometry_from_dict(meta)
        else:
            self.geometry = LEGACY_GEOMETRY
        self.square_size = (
            square_size if square_size is not None else int(meta.get("square_size", 224))
        )
        self.orientation = orientation
        self.batch_size = batch_size

    # Box padding as scalars, for the callers that build a (bbox-only) exemplar
    # bank alongside this model. Pose geometry has no such scalars, so they
    # report the deployed bank default rather than inventing a number.
    @property
    def headroom(self) -> float:
        geo = self.geometry if isinstance(self.geometry, BboxGeometry) else DEFAULT_GEOMETRY
        return geo.headroom

    @property
    def side_pad(self) -> float:
        geo = self.geometry if isinstance(self.geometry, BboxGeometry) else DEFAULT_GEOMETRY
        return geo.side_pad

    def _dist(self, crops: list[np.ndarray]) -> np.ndarray:
        outs = []
        for i in range(0, len(crops), self.batch_size):
            outs.append(self.logits_fn(preprocess(crops[i : i + self.batch_size])))
        logits = np.concatenate(outs, axis=0)
        if logits.shape != (len(crops), NUM_CLASSES):
            raise ValueError(f"model returned {logits.shape}, expected ({len(crops)}, {NUM_CLASSES})")
        return _softmax(logits).reshape(8, 8, NUM_CLASSES)

    def estimate(self, frame: Frame) -> PositionEvent:
        """Rectified-crop path: treat the whole frame as the board."""
        return self.estimate_photo(frame, full_frame_corners(frame.image))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        crops = crop_squares_photo(
            frame.image,
            corners,
            geometry=self.geometry,
            size=self.square_size,
        )
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=self._dist(crops),
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )
