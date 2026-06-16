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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board_reader_core.protocol import CLASSES, GRID, NUM_CLASSES, parse_placement

from board_reader_inference.embed import Embedder, l2_normalize


class CentroidClassifier:
    """Nearest-centroid classifier over the 13-class vocabulary.

    ``centroids`` is ``(13, dim)``; a row of NaN marks a class with no calibration
    example (it can never win and gets ~0 probability). ``temperature`` sets the
    softmax sharpness over cosine similarity — lower is peakier (more confident).
    """

    def __init__(
        self, centroids: np.ndarray, *, temperature: float = 0.1, backbone: str = "lite"
    ) -> None:
        c = np.asarray(centroids, dtype=np.float32)
        if c.shape[0] != NUM_CLASSES:
            raise ValueError(f"centroids must have {NUM_CLASSES} rows, got {c.shape[0]}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.centroids = c
        self.temperature = temperature
        # Which embedder these centroids were built with; the gateway rebuilds the
        # matching one so dims agree (a dino centroid can't be scored by lite embeddings).
        self.backbone = backbone
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
        )

    @classmethod
    def load(cls, path: str | Path) -> "CentroidClassifier":
        data = np.load(Path(path), allow_pickle=False)
        if tuple(data["classes"].tolist()) != CLASSES:
            raise ValueError("centroid file classes do not match this protocol's CLASSES")
        backbone = str(data["backbone"]) if "backbone" in data else "lite"
        return cls(
            data["centroids"], temperature=float(data["temperature"]), backbone=backbone
        )


def build_centroids(
    embedder: Embedder,
    square_crops: list[np.ndarray],
    placement: str,
    *,
    temperature: float = 0.1,
    backbone: str = "lite",
) -> CentroidClassifier:
    """Calibrate centroids from one labeled position's 64 square crops.

    ``square_crops`` is the 64 crops (rank 8→1, file a→h, e.g. from
    :func:`~board_reader_inference.squares.crop_squares`); ``placement`` is their true
    FEN-field-1 labels. Per-class centroid = L2-normalised mean of that class's square
    embeddings. Classes absent from the position stay NaN.
    """
    if len(square_crops) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} crops, got {len(square_crops)}")
    labels = [cell for row in parse_placement(placement) for cell in row]
    emb = l2_normalize(embedder.embed(square_crops), axis=1)
    centroids = np.full((NUM_CLASSES, emb.shape[1]), np.nan, dtype=np.float32)
    for ci, name in enumerate(CLASSES):
        mask = np.array([lab == name for lab in labels])
        if mask.any():
            centroids[ci] = l2_normalize(emb[mask].mean(axis=0), axis=0)
    return CentroidClassifier(centroids, temperature=temperature, backbone=backbone)
