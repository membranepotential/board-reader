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

:class:`CascadeEstimator` puts two such models behind the same seam: the second
one is consulted only on the squares the first was unsure about (2026-08-11
run_e ⊕ run_f experiment), which buys the full ensemble's accuracy at ~1.3× a
single model's cost.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

#: Where a :class:`TrainedEstimator`'s crop geometry came from. ``"fallback"`` is
#: the guessed one — good enough for a lone legacy model that predates sidecars,
#: never good enough for a cascade member (see :class:`CascadeEstimator`).
GeometrySource = Literal["sidecar", "explicit", "fallback"]


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
            self.geometry_source: GeometrySource = "explicit"
        elif "crop_geometry" in meta:
            self.geometry = geometry_from_dict(meta["crop_geometry"])
            self.geometry_source = "sidecar"
        elif "headroom" in meta:  # pre-ADT sidecars carried bare scalars
            self.geometry = geometry_from_dict(meta)
            self.geometry_source = "sidecar"
        else:
            self.geometry = LEGACY_GEOMETRY
            self.geometry_source = "fallback"
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
        return _softmax(logits)

    def square_dist(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        squares: Sequence[int] | None = None,
    ) -> np.ndarray:
        """``(N, 13)`` distributions for board ``squares`` (default all 64, L3 order).

        The crops are cut with **this model's own** geometry, which is the whole
        point of exposing the step below :meth:`estimate_photo`: an ensemble of
        two differently-trained models must crop the same photo twice, once per
        member, and a member asked about a subset of squares must pay for that
        subset only.
        """
        crops = crop_squares_photo(
            image,
            corners,
            geometry=self.geometry,
            size=self.square_size,
            squares=squares,
        )
        return self._dist(crops)

    def estimate(self, frame: Frame) -> PositionEvent:
        """Rectified-crop path: treat the whole frame as the board."""
        return self.estimate_photo(frame, full_frame_corners(frame.image))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        dist = self.square_dist(frame.image, corners)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=dist.reshape(8, 8, NUM_CLASSES),
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )


#: Squares whose max probability is below this go to the cascade's second model.
#: 0.95 is the measured operating point (2026-08-11 run_e⊕run_f experiment): it
#: reproduces the full two-model ensemble's accuracy on every set while deferring
#: under 30% of squares.
DEFAULT_CASCADE_TAU = 0.95


@dataclass
class CascadeMetrics:
    """Cumulative squares seen / deferred, for ``/healthz``.

    Counters only: the estimate itself stays a pure function of the frame
    (DESIGN D1). ``deferred`` is what makes the cascade observable in production
    — "is the second model actually being consulted, and how often?" is otherwise
    invisible from outside the process.
    """

    frames: int = 0
    squares: int = 0
    deferred: int = 0

    @property
    def deferred_fraction(self) -> float:
        return self.deferred / self.squares if self.squares else 0.0

    def observe(self, squares: int, deferred: int) -> None:
        self.frames += 1
        self.squares += squares
        self.deferred += deferred

    def as_dict(self) -> dict[str, float | int]:
        return {
            "frames": self.frames,
            "squares": self.squares,
            "deferred": self.deferred,
            "deferred_fraction": round(self.deferred_fraction, 4),
        }


class CascadeEstimator:
    """Two crop classifiers, each on its own geometry, gated by confidence.

    ``primary`` reads all 64 squares. Every square whose max probability lands
    below ``tau`` is re-read by ``secondary`` — cropped with *its* geometry, not
    the primary's — and the two distributions are averaged 50/50 for that square
    alone. Squares the primary was sure about keep the primary's distribution
    untouched, so per-square confidence and argmax always come from the
    distribution actually used.

    Measured 2026-08-11 (run_e bbox hr1.6/sp0.15 ⊕ run_f pose h1.9/r0.5, 60
    boards): the gate at τ=0.95 reproduces the unconditional 50/50 ensemble
    exactly — user_photos 0.938→0.956, chesscog_transfer 0.819→0.861, OVERALL
    0.696→0.718 — while invoking the second model on 29.7% of squares. The
    complementary-error win at ~1.3× the single model's cost, which is what makes
    it affordable on the blocking ``/estimate`` path.

    Sits exactly where a single :class:`TrainedEstimator` sits: it is the
    *trained component*, and an :class:`~board_reader_inference.estimator.EnsembleEstimator`
    still blends a calibration bank on top of it, unchanged.
    """

    crop_mode = "photo"

    def __init__(
        self,
        primary: TrainedEstimator,
        secondary: TrainedEstimator,
        *,
        tau: float = DEFAULT_CASCADE_TAU,
        metrics: CascadeMetrics | None = None,
    ) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError(
                "cascade tau must be in (0, 1]: at 0 the second model is never "
                "consulted (use the single model directly), above 1 the gate is "
                "meaningless (max probability cannot exceed 1)"
            )
        # Each member crops with its OWN geometry, so a member that does not know
        # its geometry poisons the ensemble in the one way that reads as "the
        # ensemble didn't help": silently decapitated crops from the guessed v1
        # box. A lone model may fall back with a warning; a cascade member may not.
        for role, member in (("primary", primary), ("secondary", secondary)):
            if member.geometry_source == "fallback":
                raise ValueError(
                    f"cascade {role} model {member.model_path!r} has no crop-geometry "
                    "sidecar (<model>.meta.json). The cascade's two members must each "
                    "crop with the geometry they trained on; guessing one decapitates "
                    "its crops. Export the sidecar, or pass geometry= explicitly."
                )
        self.primary = primary
        self.secondary = secondary
        self.tau = tau
        self.metrics = metrics if metrics is not None else CascadeMetrics()

    # Geometry/orientation delegate to the primary: it is the member that reads
    # every square, so it is the one ``?debug=1`` should draw and the one a
    # bbox exemplar bank should be built against.
    @property
    def geometry(self) -> CropGeometry:
        return self.primary.geometry

    @property
    def headroom(self) -> float:
        return self.primary.headroom

    @property
    def side_pad(self) -> float:
        return self.primary.side_pad

    @property
    def square_size(self) -> int:
        return self.primary.square_size

    @property
    def orientation(self) -> str:
        return self.primary.orientation

    @property
    def model_path(self) -> str | None:
        return self.primary.model_path

    def square_dist(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        squares: Sequence[int] | None = None,
    ) -> np.ndarray:
        """``(N, 13)`` cascaded distributions — the primary's, averaged where unsure."""
        dist = self.primary.square_dist(image, corners, squares)
        asked = np.arange(64) if squares is None else np.asarray(squares, dtype=int)
        unsure = np.flatnonzero(dist.max(axis=-1) < self.tau)
        self.metrics.observe(len(asked), len(unsure))
        if unsure.size == 0:
            return dist
        second = self.secondary.square_dist(image, corners, asked[unsure])
        dist = dist.copy()
        dist[unsure] = 0.5 * dist[unsure] + 0.5 * second
        return dist

    def estimate(self, frame: Frame) -> PositionEvent:
        return self.estimate_photo(frame, full_frame_corners(frame.image))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        dist = self.square_dist(frame.image, corners)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=dist.reshape(8, 8, NUM_CLASSES),
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )


#: The zero-shot trained component of the estimator stack: one ONNX crop
#: classifier, or the confidence-gated cascade over two of them. Both crop
#: original photos themselves and expose the bbox scalars an exemplar bank is
#: built against, so everything downstream (the booster blend, auto-calibration,
#: ``/calibrate``) treats them identically.
TrainedReader = TrainedEstimator | CascadeEstimator
