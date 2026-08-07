"""Tests for the split/sampling helpers in ``dataset.py``.

These run under the **torch venv**, not the workspace suite: the workspace stays
torch-free by design (CLAUDE.md), and ``dataset`` imports ``torch``. See the
README ("Tests") for the one-liner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import subsample_val

RNG = np.random.default_rng(1234)


def _labels(counts: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build (val_idx, labels) where class ``c`` appears ``counts[c]`` times.

    ``val_idx`` deliberately does NOT start at 0 — the real one indexes into the
    full corpus, so a helper that confused positions with indices would pass on a
    0-based fixture and corrupt a real run.
    """
    labels_list: list[int] = []
    for c, n in counts.items():
        labels_list.extend([c] * n)
    offset = 500
    labels = np.zeros(offset + len(labels_list), dtype=np.int64)
    labels[offset:] = labels_list
    val_idx = np.arange(offset, offset + len(labels_list))
    return val_idx, labels


def test_returns_input_when_already_under_cap():
    val_idx, labels = _labels({0: 10, 1: 5})
    out = subsample_val(val_idx, labels, cap=100, seed=0)
    assert out is val_idx


def test_caps_to_requested_size():
    val_idx, labels = _labels({0: 8000, 1: 4000, 2: 4000})
    out = subsample_val(val_idx, labels, cap=4000, seed=0)
    assert len(out) == 4000


def test_is_deterministic_for_a_seed_and_varies_across_seeds():
    val_idx, labels = _labels({0: 5000, 1: 3000, 2: 2000})
    a = subsample_val(val_idx, labels, cap=1000, seed=0)
    b = subsample_val(val_idx, labels, cap=1000, seed=0)
    c = subsample_val(val_idx, labels, cap=1000, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_preserves_class_proportions():
    """The point of stratifying: rare classes must not wobble.

    Queens/kings are ~4% of a board each, and piece-type accuracy turns on them.
    """
    counts = {0: 8000, 1: 1000, 2: 600, 3: 400}
    val_idx, labels = _labels(counts)
    out = subsample_val(val_idx, labels, cap=1000, seed=0)
    total = sum(counts.values())
    for cls, n in counts.items():
        got = int((labels[out] == cls).sum())
        assert got == pytest.approx(n / total * 1000, abs=1)


def test_returns_sorted_indices_drawn_from_the_input():
    val_idx, labels = _labels({0: 300, 1: 200})
    out = subsample_val(val_idx, labels, cap=100, seed=7)
    assert np.array_equal(out, np.sort(out))
    assert len(np.unique(out)) == len(out)
    assert set(out.tolist()) <= set(val_idx.tolist())


def test_never_over_draws_a_class_smaller_than_its_quota():
    """A class with fewer members than its share must not be sampled twice."""
    val_idx, labels = _labels({0: 999, 1: 1})
    out = subsample_val(val_idx, labels, cap=500, seed=0)
    assert len(np.unique(out)) == len(out)
    assert int((labels[out] == 1).sum()) <= 1


def test_rejects_a_non_positive_cap():
    val_idx, labels = _labels({0: 10})
    with pytest.raises(ValueError, match="must be positive"):
        subsample_val(val_idx, labels, cap=0, seed=0)
