"""Tests for confidence calibration (:mod:`board_reader_inference.measurement.evaluate.confidence`).
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_inference.measurement.evaluate.confidence import _ranking_auc, reliability
from board_reader_inference.measurement.evaluate.report import Report


def test_reliability_measures_the_gap_between_claimed_and_actual():
    # 10 squares claiming 0.95, only half of them right -> a 0.45 gap.
    conf = np.full(10, 0.95)
    hits = np.array([True] * 5 + [False] * 5)
    bins, ece = reliability(conf, hits)
    assert len(bins) == 1
    (b,) = bins
    assert b.count == 10
    assert b.mean_confidence == pytest.approx(0.95)
    assert b.accuracy == pytest.approx(0.5)
    assert b.gap == pytest.approx(0.45)
    assert ece == pytest.approx(0.45)


def test_reliability_counts_confidence_of_exactly_one():
    """A one-hot distribution must land in the top bin, not fall off the end."""
    bins, _ = reliability(np.array([1.0, 1.0]), np.array([True, False]))
    assert sum(b.count for b in bins) == 2


def test_perfectly_calibrated_input_has_near_zero_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=20000)
    hits = rng.uniform(size=conf.size) < conf
    _, ece = reliability(conf, hits)
    assert ece < 0.02


def test_ranking_auc_is_the_signal_the_confidence_gap_hides():
    """Temperature scaling compresses the perfect/imperfect confidence gap by
    construction, so the gap alone would report a harmless change as a loss. AUC
    is the rank-based number that actually decides whether a consumer can
    threshold on confidence."""
    assert _ranking_auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert _ranking_auc([0.2, 0.1], [0.9, 0.8]) == 0.0
    assert _ranking_auc([0.5], [0.5]) == 0.5  # ties are half a win
    # A monotone squeeze toward 1.0 keeps the ordering, so AUC is unmoved even
    # though the means have nearly converged.
    assert _ranking_auc([0.99, 0.98], [0.97, 0.96]) == 1.0
    assert _ranking_auc([], [0.5]) is None
    assert _ranking_auc([0.5], []) is None


def test_temperature_report_is_absent_unless_asked_for():
    """It costs a leave-one-out fit per board; the default eval must not pay it."""
    assert Report(boards=[], estimator="x").temperature is None
