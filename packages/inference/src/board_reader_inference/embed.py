"""Square-crop embedders (DESIGN.md WP-B2, §4.6).

An :class:`Embedder` maps square crops to fixed-length vectors that a downstream
classifier (per-class centroids in M1, an offline-trained head later) turns into the
13-class distribution. Per DESIGN §4.6 the embedding backbone deliberately sits
**behind an interface** so the default DINOv2 ViT-S/14 can be swapped (DINOv3, a
fine-tuned CNN, …) without touching the classifier or estimator.

M1 ships :class:`LiteEmbedder`, a dependency-light embedder good enough to drive the
centroid classifier on the synthetic feed and in tests. The real :class:`!DinoV2Embedder`
(ONNX Runtime) plugs into the same protocol next.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Square crops -> embeddings. The seam the recognition backbone plugs into."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed N crops -> an ``(N, dim)`` float32 array."""
        ...


def l2_normalize(x: np.ndarray, *, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation (so cosine similarity is a plain dot product)."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


class LiteEmbedder:
    """Cheap deterministic embedder: background-subtracted colour ink pattern.

    Not a learned model — it is the M1 stand-in that lets the full centroid pipeline
    (cropping → embed → classify → event) run and be tested end-to-end without the
    heavyweight backbone or any download.

    Per square it subtracts the square's own background colour (the median pixel,
    since most of a square is empty board) and keeps the signed colour *deviation* —
    the piece's ink. This is contrast-invariant (a low-contrast white piece on a light
    square and a high-contrast one on a dark square share a direction after
    normalisation) yet polarity- and colour-aware (white ink is +, black ink is −), so
    ``P`` vs ``p`` and piece type are all separable. A trailing **occupancy** dimension
    keeps empty squares (no ink) distinct from pieces under cosine similarity.

    On real angled photos this is *not* adequate — that is exactly what the DINOv2
    embedder replaces, behind this same interface.
    """

    def __init__(self, side: int = 16, *, occupancy_scale: float = 6.0) -> None:
        self.side = side
        self.occupancy_scale = occupancy_scale

    @property
    def dim(self) -> int:
        return self.side * self.side * 3 + 1  # colour pattern + occupancy bias

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(crops), self.dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            f = bgr.astype(np.float32)
            bg = np.median(f.reshape(-1, 3), axis=0)  # the flat board-square colour
            dev = cv2.resize(
                f - bg, (self.side, self.side), interpolation=cv2.INTER_AREA
            ).ravel()
            occ = float(np.abs(dev).mean())
            amp = np.tanh(occ / self.occupancy_scale)  # ~0 empty, ->1 occupied
            shape = dev / max(float(np.linalg.norm(dev)), 1e-6)
            # Occupied -> the ink pattern dominates; empty -> the occupancy dim does.
            out[i, :-1] = shape * amp
            out[i, -1] = 1.0 - amp
        return l2_normalize(out, axis=1)


# Default location of the converted DINOv2 ONNX (see tools/export_dinov2_onnx.py).
# Overridable via BR_DINOV2_ONNX so deployments can point elsewhere.
DEFAULT_DINOV2_ONNX = os.environ.get("BR_DINOV2_ONNX", "models/dinov2_small.onnx")

# ImageNet statistics DINOv2 was trained with (RGB).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoV2Embedder:
    """Frozen DINOv2 ViT-S/14 backbone (ONNX Runtime) → 384-d CLS embeddings.

    The DESIGN-default recognition backbone (§4.6): an Apache-2.0 DINOv2 ViT-S/14
    exported from **official Meta weights** to ONNX by ``tools/export_dinov2_onnx.py``
    (we convert ourselves; we never pull a third-party ONNX). Plugs into the same
    :class:`Embedder` protocol as :class:`LiteEmbedder`, so calibration, the centroid
    classifier, and the estimator are unchanged — only the embeddings get better on
    real angled photos.

    Square crops (BGR, any size) are resized to ``image_size`` and ImageNet-normalised
    to match the export preprocessing, then embedded in one batched session run.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_DINOV2_ONNX,
        *,
        image_size: int = 224,
        providers: list[str] | None = None,
        batch_size: int = 64,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DINOv2 ONNX not found at {self.model_path}; export it with "
                "tools/export_dinov2_onnx.py (see tools/README.md)"
            )
        self.image_size = image_size
        if providers is None:
            # Prefer the GPU when the CUDA EP is usable (~5x on a GTX 1650);
            # preload_dlls resolves the pip-installed nvidia-*-cu12 libs so no
            # LD_LIBRARY_PATH setup is needed. Silently fall back to CPU — the
            # session constructor itself downgrades if CUDA can't initialise.
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                try:
                    ort.preload_dlls()
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    pass
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        self._input = self.session.get_inputs()[0].name
        self._dim = int(self.session.get_outputs()[0].shape[-1])
        # Cap the per-run batch: a 64-crop board fits comfortably, but augmented
        # multi-photo calibration can hand us hundreds of crops at once, which
        # OOMs the ONNX arena (especially on small GPUs).
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._dim

    def _preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        n = len(crops)
        batch = np.empty((n, 3, self.image_size, self.image_size), dtype=np.float32)
        for i, crop in enumerate(crops):
            bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(
                bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            batch[i] = np.transpose(rgb, (2, 0, 1))
        return batch

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        outs = [
            self.session.run(
                None, {self._input: self._preprocess(crops[i : i + self.batch_size])}
            )[0]
            for i in range(0, len(crops), self.batch_size)
        ]
        out = np.vstack(outs)
        return l2_normalize(np.asarray(out, dtype=np.float32), axis=1)


# Named backbones, so a calibration file can record which embedder it was built with
# and the gateway can reconstruct the matching one (centroid dims must agree).
BACKBONES = ("lite", "dinov2")


def make_embedder(name: str) -> Embedder:
    """Construct an :class:`Embedder` by name (``lite`` | ``dinov2``)."""
    if name == "lite":
        return LiteEmbedder()
    if name == "dinov2":
        return DinoV2Embedder()
    raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")
