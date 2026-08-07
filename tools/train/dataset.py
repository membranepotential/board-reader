"""Torch dataset over the crop cache, with style/group-aware splitting.

The dataset holds JPEG bytes in memory and decodes lazily in ``__getitem__`` (a
full ChessReD cache is ~1.6 GB of JPEG, comfortably RAM-resident; decode is the
per-item cost). Splitting is by **style** (the leave-style-out protocol) or, when
only one style is available, by **group** (hold out whole games/cameras) so the
validation set is never the same viewpoint the model trained on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch
from augment import to_rgb_tensor
from cropcache import CACHE_ROOT, iter_shard, load_meta
from torch.utils.data import Dataset

from board_reader_core.protocol import CLASSES, NUM_CLASSES


@dataclass
class StyleData:
    """All cached crops for a set of styles, flattened and RAM-resident."""

    jpegs: list[bytes]
    labels: np.ndarray  # int64 (N,)
    group_keys: np.ndarray  # object (N,) "<style>/<group>"
    style_keys: np.ndarray  # object (N,) "<style>"

    def __len__(self) -> int:
        return len(self.jpegs)


def load_styles(names: list[str], cache_root: Path = CACHE_ROOT) -> StyleData:
    """Load the crop caches for ``names`` into one in-memory :class:`StyleData`."""
    jpegs: list[bytes] = []
    labels: list[int] = []
    group_keys: list[str] = []
    style_keys: list[str] = []
    for name in names:
        meta = load_meta(name, cache_root)
        groups = meta.group_names
        for shard in meta.shards:
            for jpeg, lab, gid in iter_shard(cache_root / name / shard):
                jpegs.append(jpeg)
                labels.append(lab)
                group_keys.append(f"{name}/{groups[gid]}")
                style_keys.append(name)
    return StyleData(
        jpegs=jpegs,
        labels=np.asarray(labels, dtype=np.int64),
        group_keys=np.asarray(group_keys, dtype=object),
        style_keys=np.asarray(style_keys, dtype=object),
    )


def loader_worker_init(_worker_id: int) -> None:
    """DataLoader worker init: one decode thread per worker.

    OpenCV spawns its own thread pool per process; with N workers that is N pools
    fighting over the same cores, and torch does the same for intra-op threads.
    The parallelism belongs to the workers, not inside them.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class CropDataset(Dataset):
    """Indexed view over :class:`StyleData`; decodes + transforms one crop."""

    def __init__(
        self,
        data: StyleData,
        indices: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        buf = np.frombuffer(self.data.jpegs[idx], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        x = self.transform(to_rgb_tensor(bgr))
        return x, int(self.data.labels[idx])

    def label_slice(self) -> np.ndarray:
        return self.data.labels[self.indices]


def split_by_style(data: StyleData, holdout: str) -> tuple[np.ndarray, np.ndarray]:
    """Train = every style except ``holdout``; val = ``holdout`` (leave-style-out)."""
    is_holdout = data.style_keys == holdout
    val = np.nonzero(is_holdout)[0]
    train = np.nonzero(~is_holdout)[0]
    if len(val) == 0:
        raise ValueError(f"holdout style {holdout!r} has no crops")
    return train, val


def split_by_group(
    data: StyleData, *, val_frac: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole groups (games/cameras) for validation — single-style fallback."""
    groups = np.unique(data.group_keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_frac))
    val_groups = set(groups[:n_val].tolist())
    is_val = np.array([g in val_groups for g in data.group_keys])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def subsample_val(
    val_idx: np.ndarray, labels: np.ndarray, cap: int, *, seed: int = 0
) -> np.ndarray:
    """Deterministic, class-stratified cap on a validation index set.

    Validation is ~25% of epoch wall time at the full 16k leave-style-out holdout,
    and a screening run only needs enough resolution to *rank* candidates, not to
    quote a deployable number. Capping to a few thousand crops buys that back.

    Stratified rather than a flat random draw: piece-type accuracy is decided by the
    rare classes (queens and kings are ~4% of a board each), and an unstratified
    sample lets their counts wobble enough to move the metric on sampling noise
    alone. Largest-remainder rounding keeps each class's share of the original set.

    Returns ``val_idx`` unchanged when it already fits under ``cap``.
    """
    if cap <= 0:
        raise ValueError(f"val_max must be positive, got {cap}")
    if cap >= len(val_idx):
        return val_idx
    rng = np.random.default_rng(seed)
    cls_labels = labels[val_idx]
    classes, counts = np.unique(cls_labels, return_counts=True)
    exact = counts * (cap / len(val_idx))
    quota = np.floor(exact).astype(int)
    spare = cap - int(quota.sum())
    if spare > 0:
        # Hand leftover slots to the largest fractional parts, ties by class order.
        quota[np.argsort(-(exact - quota), kind="stable")[:spare]] += 1
    quota = np.minimum(quota, counts)
    picked = [
        rng.choice(val_idx[cls_labels == c], size=q, replace=False)
        for c, q in zip(classes, quota)
        if q > 0
    ]
    return np.sort(np.concatenate(picked))


def class_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency weight per sample for a WeightedRandomSampler.

    Empty squares dominate (~half a board); without rebalancing the model just
    learns the empty prior. Weight = 1 / count(class).
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    return inv[labels]


def class_counts_str(labels: np.ndarray) -> str:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return " ".join(f"{CLASSES[i]}:{int(counts[i])}" for i in range(NUM_CLASSES) if counts[i])
