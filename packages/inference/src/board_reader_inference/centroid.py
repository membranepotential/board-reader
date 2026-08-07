"""Per-class centroid classifier + few-shot calibration (DESIGN.md §4.6, WP-E2/B3).

The position-agnostic few-shot path: the operator supplies **one** known position for
a single rectified frame; we embed its 64 squares, average the embeddings per class
into centroids, and classify every later square by nearest centroid. Because the
start position contains all 13 classes, one start-position frame calibrates the whole
vocabulary — but any labeled position works (the Chesscog trick, generalised so we are
not tied to the start position).

Output is a full 13-class distribution per square (softmax over cosine similarity to
the centroids), in :data:`board_reader_core.protocol.CLASSES` order, ready to drop
into a :class:`~board_reader_core.protocol.PositionEvent`.

:class:`ExemplarClassifier` is a kNN variant of the same idea: instead of averaging
each class's crops into one centroid, it keeps every calibration embedding as its
own exemplar and scores a query by its nearest exemplar per class, with per-board
transductive centering (both the bank and the query batch are re-centered on their
own means before the cosine step). Validated on labeled real photos (leave-one-photo-out),
this lifts accuracy 261->294/320 and cuts empty->piece hallucinations 28->6 versus
centroids — the accuracy gain comes from keeping the full within-class embedding
spread instead of collapsing it to a mean, plus the transductive step absorbing
per-photo lighting/backbone shift. Centroid files remain fully loadable
(:func:`load_classifier` dispatches on file kind).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    GRID,
    NUM_CLASSES,
    parse_placement,
)

from board_reader_inference.embed import Embedder, l2_normalize

# Cropping regimes a centroid file can be calibrated on (see CentroidClassifier).
CROP_MODES = ("rectified", "photo")


class CentroidClassifier:
    """Nearest-centroid classifier over the 13-class vocabulary.

    ``centroids`` is ``(13, dim)``; a row of NaN marks a class with no calibration
    example (it can never win and gets ~0 probability). ``temperature`` sets the
    softmax sharpness over cosine similarity — lower is peakier (more confident).
    """

    def __init__(
        self,
        centroids: np.ndarray,
        *,
        temperature: float = 0.1,
        backbone: str = "lite",
        crop_mode: str = "rectified",
        box_headroom: float = 1.6,
        box_side_pad: float = 0.15,
    ) -> None:
        c = np.asarray(centroids, dtype=np.float32)
        if c.shape[0] != NUM_CLASSES:
            raise ValueError(f"centroids must have {NUM_CLASSES} rows, got {c.shape[0]}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")
        if box_headroom < 0 or box_side_pad < 0:
            raise ValueError("box_headroom and box_side_pad must be >= 0")
        self.centroids = c
        self.temperature = temperature
        # Which embedder these centroids were built with; the gateway rebuilds the
        # matching one so dims agree (a dino centroid can't be scored by lite embeddings).
        self.backbone = backbone
        # Which cropping regime the calibration crops came from ("rectified" = flat
        # squares of the top-down warp, "photo" = tall full-piece boxes from the
        # original view). Classification is only *exact* when inference crops the
        # same way; the gateway uses this to pick the matching estimate path.
        self.crop_mode = crop_mode
        # Photo-mode box geometry the calibration crops were cut with; inference
        # must reuse these for exactness (irrelevant for crop_mode="rectified").
        self.box_headroom = float(box_headroom)
        self.box_side_pad = float(box_side_pad)
        self._present = ~np.isnan(c).any(axis=1)
        if not self._present.any():
            raise ValueError("no class centroids present")

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    def classify(self, embeddings: np.ndarray) -> np.ndarray:
        """Embeddings ``(N, dim)`` -> distributions ``(N, 13)`` over CLASSES."""
        e = l2_normalize(np.asarray(embeddings, dtype=np.float32), axis=1)
        c = np.nan_to_num(self.centroids, nan=0.0)
        sims = e @ c.T  # (N, 13), cosine since both sides L2-normalised
        logits = sims / self.temperature
        # Absent classes are never selectable.
        logits[:, ~self._present] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)  # stabilise softmax
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def classify_board(self, embeddings: np.ndarray) -> np.ndarray:
        """64 square embeddings -> an ``(8, 8, 13)`` ``square_dist`` (rank 8→1, file a→h)."""
        if embeddings.shape[0] != GRID * GRID:
            raise ValueError(f"expected {GRID * GRID} embeddings, got {embeddings.shape[0]}")
        return self.classify(embeddings).reshape(GRID, GRID, NUM_CLASSES)

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            centroids=self.centroids,
            temperature=np.float64(self.temperature),
            classes=np.array(CLASSES),
            backbone=np.array(self.backbone),
            crop_mode=np.array(self.crop_mode),
            box_headroom=np.float64(self.box_headroom),
            box_side_pad=np.float64(self.box_side_pad),
        )

    @classmethod
    def load(cls, path: str | Path) -> CentroidClassifier:
        data = np.load(Path(path), allow_pickle=False)
        if tuple(data["classes"].tolist()) != CLASSES:
            raise ValueError("centroid file classes do not match this protocol's CLASSES")
        backbone = str(data["backbone"]) if "backbone" in data else "lite"
        crop_mode = str(data["crop_mode"]) if "crop_mode" in data else "rectified"
        return cls(
            data["centroids"],
            temperature=float(data["temperature"]),
            backbone=backbone,
            crop_mode=crop_mode,
            box_headroom=float(data["box_headroom"]) if "box_headroom" in data else 1.6,
            box_side_pad=float(data["box_side_pad"]) if "box_side_pad" in data else 0.15,
        )


def _gamma(crop: np.ndarray, g: float) -> np.ndarray:
    lut = ((np.arange(256) / 255.0) ** g * 255.0).astype(np.uint8)
    return cv2.LUT(crop, lut)


def _grayworld(crop: np.ndarray) -> np.ndarray:
    f = crop.astype(np.float32)
    means = f.reshape(-1, f.shape[-1] if f.ndim == 3 else 1).mean(axis=0)
    f = f * (float(means.mean()) / np.maximum(means, 1e-6))
    return np.clip(f, 0, 255).astype(np.uint8)


def photometric_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Cheap lighting augmentation for calibration crops.

    Identity + darker/brighter gamma + gray-world white balance. Averaging the
    embeddings of all variants into the class centroids widens them against the
    lighting shifts that single-frame few-shot calibration is most brittle to
    (warm lamp vs daylight was the dominant cross-photo failure).
    """
    return [crop, _gamma(crop, 0.7), _gamma(crop, 1.4), _grayworld(crop)]


def build_centroids(
    embedder: Embedder,
    square_crops: list[np.ndarray],
    placement: str,
    *,
    temperature: float = 0.1,
    backbone: str = "lite",
    crop_mode: str = "rectified",
    augment: bool = False,
    box_headroom: float = 1.6,
    box_side_pad: float = 0.15,
) -> CentroidClassifier:
    """Calibrate centroids from one labeled position's 64 square crops.

    ``square_crops`` is the 64 crops (rank 8→1, file a→h, from
    :func:`~board_reader_inference.squares.crop_squares` or
    :func:`~board_reader_inference.squares.crop_squares_photo` — record which via
    ``crop_mode``); ``placement`` is their true FEN-field-1 labels. Per-class
    centroid = L2-normalised mean of that class's square embeddings; with
    ``augment`` every crop also contributes its :func:`photometric_variants`.
    Classes absent from the position stay NaN.
    """
    if len(square_crops) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} crops, got {len(square_crops)}")
    labels = [cell for row in parse_placement(placement) for cell in row]
    crops: list[np.ndarray] = []
    lab_expanded: list[str] = []
    for crop, lab in zip(square_crops, labels):
        variants = photometric_variants(crop) if augment else [crop]
        crops.extend(variants)
        lab_expanded.extend([lab] * len(variants))
    emb = l2_normalize(embedder.embed(crops), axis=1)
    centroids = np.full((NUM_CLASSES, emb.shape[1]), np.nan, dtype=np.float32)
    for ci, name in enumerate(CLASSES):
        mask = np.array([lab == name for lab in lab_expanded])
        if mask.any():
            centroids[ci] = l2_normalize(emb[mask].mean(axis=0), axis=0)
    return CentroidClassifier(
        centroids,
        temperature=temperature,
        backbone=backbone,
        crop_mode=crop_mode,
        box_headroom=box_headroom,
        box_side_pad=box_side_pad,
    )


class ExemplarClassifier:
    """kNN exemplar-bank classifier over the 13-class vocabulary.

    Where :class:`CentroidClassifier` collapses each class's calibration crops into
    one mean vector, this keeps every embedding as its own exemplar and scores a
    query by its **nearest exemplar per class** — cheap to compute (one matmul) and
    empirically stronger once combined with per-board transductive centering (see
    :meth:`classify`), because it preserves the within-class embedding spread a
    centroid mean throws away.

    ``exemplars`` is ``(N, dim)``, L2-normalised on ingest; ``labels`` names each
    row's class (must be one of :data:`board_reader_core.protocol.CLASSES`).
    """

    def __init__(
        self,
        exemplars: np.ndarray,
        labels: list[str],
        *,
        temperature: float = 0.1,
        backbone: str = "dinov2",
        crop_mode: str = "photo",
        box_headroom: float = 0.15,
        box_side_pad: float = 0.15,
    ) -> None:
        ex = np.asarray(exemplars, dtype=np.float32)
        if ex.ndim != 2 or ex.shape[0] == 0:
            raise ValueError("exemplars must be a non-empty (N, dim) array")
        labels = list(labels)
        if len(labels) != ex.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} does not match {ex.shape[0]} exemplars"
            )
        bad = sorted({lab for lab in labels if lab not in CLASS_INDEX})
        if bad:
            raise ValueError(f"unknown class label(s): {bad}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")
        if box_headroom < 0 or box_side_pad < 0:
            raise ValueError("box_headroom and box_side_pad must be >= 0")
        self.exemplars = l2_normalize(ex, axis=1)
        self.labels = np.array(labels)
        self.temperature = temperature
        # Which embedder/cropping regime this bank was built with; same role as on
        # CentroidClassifier (the gateway rebuilds the matching embedder and box
        # geometry so dims and crops agree).
        self.backbone = backbone
        self.crop_mode = crop_mode
        self.box_headroom = float(box_headroom)
        self.box_side_pad = float(box_side_pad)
        self._present = np.array([name in labels for name in CLASSES])
        class_mask = np.zeros((NUM_CLASSES, len(labels)), dtype=bool)
        for i, lab in enumerate(labels):
            class_mask[CLASS_INDEX[lab], i] = True
        self._class_mask = class_mask
        # The transductive bank only depends on the exemplars themselves, so it is
        # precomputed once here rather than on every classify() call.
        self._centered_bank = l2_normalize(
            self.exemplars - self.exemplars.mean(axis=0, keepdims=True), axis=1
        )

    @property
    def dim(self) -> int:
        return self.exemplars.shape[1]

    def classify(self, embeddings: np.ndarray, *, transduce: bool = False) -> np.ndarray:
        """Embeddings ``(N, dim)`` -> distributions ``(N, 13)`` over CLASSES.

        Per-class score is the MAX cosine similarity to that class's exemplars
        (nearest exemplar per class), then a stable softmax over score/temperature.

        ``transduce=True`` re-centers both the bank and the query batch on their
        own means before scoring — a per-board correction for the lighting/backbone
        shift between the calibration photo and the query frame that is only
        statistically valid over a full board's worth of squares at once (hence
        :meth:`classify_board` always uses it, while single-crop ad-hoc calls
        default to the plain untransduced score).
        """
        e = l2_normalize(np.asarray(embeddings, dtype=np.float32), axis=1)
        if transduce:
            bank = self._centered_bank
            e = l2_normalize(e - e.mean(axis=0, keepdims=True), axis=1)
        else:
            bank = self.exemplars
        sims = e @ bank.T  # (N, n_exemplars)
        masked = np.where(self._class_mask[None, :, :], sims[:, None, :], -np.inf)
        logits = masked.max(axis=2) / self.temperature  # (N, 13), nearest exemplar per class
        logits[:, ~self._present] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)  # stabilise softmax
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def classify_board(self, embeddings: np.ndarray) -> np.ndarray:
        """64 square embeddings -> an ``(8, 8, 13)`` ``square_dist`` (rank 8→1, file a→h)."""
        if embeddings.shape[0] != GRID * GRID:
            raise ValueError(f"expected {GRID * GRID} embeddings, got {embeddings.shape[0]}")
        return self.classify(embeddings, transduce=True).reshape(GRID, GRID, NUM_CLASSES)

    def extended(self, embeddings: np.ndarray, labels: list[str]) -> ExemplarClassifier:
        """Return a new bank with ``embeddings``/``labels`` appended to this one.

        Used by in-session calibration to ADD a photo to the bank rather than
        replace it — more exemplars (esp. under different lighting) only helps
        nearest-exemplar scoring. Metadata (temperature, backbone, crop geometry)
        carries over unchanged.
        """
        exemplars = np.concatenate(
            [self.exemplars, np.asarray(embeddings, dtype=np.float32)], axis=0
        )
        all_labels = list(self.labels) + list(labels)
        return ExemplarClassifier(
            exemplars,
            all_labels,
            temperature=self.temperature,
            backbone=self.backbone,
            crop_mode=self.crop_mode,
            box_headroom=self.box_headroom,
            box_side_pad=self.box_side_pad,
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            kind=np.array("exemplar"),
            exemplars=self.exemplars,
            labels=np.array(self.labels),
            classes=np.array(CLASSES),
            temperature=np.float64(self.temperature),
            backbone=np.array(self.backbone),
            crop_mode=np.array(self.crop_mode),
            box_headroom=np.float64(self.box_headroom),
            box_side_pad=np.float64(self.box_side_pad),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExemplarClassifier:
        data = np.load(Path(path), allow_pickle=False)
        if tuple(data["classes"].tolist()) != CLASSES:
            raise ValueError("exemplar file classes do not match this protocol's CLASSES")
        labels = [str(lab) for lab in data["labels"].tolist()]
        backbone = str(data["backbone"]) if "backbone" in data else "dinov2"
        crop_mode = str(data["crop_mode"]) if "crop_mode" in data else "photo"
        return cls(
            data["exemplars"],
            labels,
            temperature=float(data["temperature"]),
            backbone=backbone,
            crop_mode=crop_mode,
            box_headroom=float(data["box_headroom"]) if "box_headroom" in data else 0.15,
            box_side_pad=float(data["box_side_pad"]) if "box_side_pad" in data else 0.15,
        )


def build_exemplars(
    embedder: Embedder,
    square_crops: list[np.ndarray],
    placement: str,
    *,
    mask: np.ndarray | None = None,
    augment: bool = True,
    temperature: float = 0.1,
    backbone: str = "lite",
    box_headroom: float = 0.15,
    box_side_pad: float = 0.15,
) -> ExemplarClassifier:
    """Calibrate a kNN exemplar bank from one labeled position's 64 square crops.

    Mirrors :func:`build_centroids`, but keeps every embedding as its own exemplar
    instead of averaging into a per-class centroid mean — the accuracy-validated
    replacement (DESIGN.md §4.6). ``crop_mode`` is always ``"photo"``: the tall
    full-piece boxes this classifier was validated on. With ``augment`` (the
    default here, unlike centroids) every crop also contributes its
    :func:`photometric_variants`, widening the bank against lighting shift.

    ``mask`` (64 bools, cropped order) drops squares whose label is not trusted —
    start-position auto-calibration uses it to exclude squares whose occupancy
    disagrees with the start pattern (already-played moves, occupancy misreads).
    """
    if len(square_crops) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} crops, got {len(square_crops)}")
    keep = np.ones(GRID * GRID, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape != (GRID * GRID,):
        raise ValueError(f"mask must have {GRID * GRID} entries, got {keep.shape}")
    labels = [cell for row in parse_placement(placement) for cell in row]
    crops: list[np.ndarray] = []
    lab_expanded: list[str] = []
    for crop, lab, kept in zip(square_crops, labels, keep):
        if not kept:
            continue
        variants = photometric_variants(crop) if augment else [crop]
        crops.extend(variants)
        lab_expanded.extend([lab] * len(variants))
    emb = embedder.embed(crops)
    return ExemplarClassifier(
        emb,
        lab_expanded,
        temperature=temperature,
        backbone=backbone,
        crop_mode="photo",
        box_headroom=box_headroom,
        box_side_pad=box_side_pad,
    )


def load_classifier(path: str | Path) -> CentroidClassifier | ExemplarClassifier:
    """Load a calibration file, dispatching on kind (centroid mean vs kNN exemplar bank).

    Both file kinds carry the same ``classes``/``temperature``/``backbone``/
    ``crop_mode``/box-geometry metadata; only the payload differs (one row per
    class vs. one row per exemplar), which is what the ``exemplars``/``kind`` keys
    distinguish.
    """
    data = np.load(Path(path), allow_pickle=False)
    is_exemplar = "exemplars" in data.files or (
        "kind" in data.files and str(data["kind"]) == "exemplar"
    )
    return ExemplarClassifier.load(path) if is_exemplar else CentroidClassifier.load(path)
