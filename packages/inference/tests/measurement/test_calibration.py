"""Tests for temperature scaling of the per-square distribution (calibration.py)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import GRID, NUM_CLASSES

from board_reader_inference.measurement.calibration import (
    apply_temperature,
    fit_temperature,
    fit_temperature_over_boards,
    leave_one_out_scaled,
    negative_log_likelihood,
)


def _rng():
    return np.random.default_rng(0)


def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _synthetic(n=800, *, distortion=1.0, seed=0):
    """Miscalibrated distributions with a known, recoverable correction.

    Labels are **sampled from** ``softmax(logits)``, which is what makes that
    distribution the perfectly calibrated one for this data — without sampling
    there is no ground truth about what a confidence *should* be, and "recovers
    the right temperature" would mean nothing.

    The returned distribution is ``softmax(logits / distortion)``: ``distortion
    > 1`` is too soft (under-confident, the deployed ensemble's failure mode),
    ``< 1`` too sharp. Since ``apply_temperature`` divides ``log p`` by ``T``,
    the correcting temperature is ``1 / distortion``.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.5, size=(n, NUM_CLASSES))
    calibrated = _softmax(logits)
    truths = np.array([rng.choice(NUM_CLASSES, p=row) for row in calibrated])
    return _softmax(logits / distortion), truths


def test_temperature_never_changes_an_answer():
    """The whole safety argument: scaling is monotone per square, so the argmax —
    and therefore the published placement — cannot move."""
    dist, _ = _synthetic(400)
    base = dist.argmax(axis=-1)
    for t in (0.05, 0.2, 0.5, 0.999, 1.0, 2.0, 7.5, 20.0):
        assert (apply_temperature(dist, t).argmax(axis=-1) == base).all(), t


def test_temperature_one_is_the_identity():
    dist, _ = _synthetic(50)
    assert np.allclose(apply_temperature(dist, 1.0), dist, atol=1e-6)


def test_scaled_distributions_stay_distributions():
    dist, _ = _synthetic(100)
    for t in (0.1, 1.0, 5.0):
        out = apply_temperature(dist, t)
        assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-5)
        assert (out >= 0).all()


def test_low_temperature_sharpens_and_high_softens():
    dist, _ = _synthetic(200)
    conf = dist.max(axis=-1).mean()
    assert apply_temperature(dist, 0.5).max(axis=-1).mean() > conf
    assert apply_temperature(dist, 2.0).max(axis=-1).mean() < conf


def test_fit_recovers_a_known_temperature():
    """A distribution softened by 2.5x must be corrected by a fitted T ≈ 1/2.5."""
    dist, truths = _synthetic(6000, distortion=2.5, seed=1)
    got = fit_temperature(dist, truths)
    assert abs(got - 0.4) / 0.4 < 0.15, got


def test_fit_returns_one_for_already_calibrated_scores():
    dist, truths = _synthetic(6000, distortion=1.0, seed=2)
    assert abs(fit_temperature(dist, truths) - 1.0) < 0.15


def test_fit_minimises_nll():
    """Whatever the fit returns must beat its neighbours — that is what it claims."""
    dist, truths = _synthetic(2000, distortion=0.4, seed=3)
    t = fit_temperature(dist, truths)
    best = negative_log_likelihood(dist, truths, t)
    for other in (t * 0.7, t * 1.4, 1.0):
        assert best <= negative_log_likelihood(dist, truths, other) + 1e-9


def test_underconfident_scores_are_fixed_by_a_temperature_below_one():
    """The deployed ensemble's actual failure mode: claiming 35-75% while being
    right nearly always. The fit must respond by sharpening, i.e. T < 1."""
    dist, truths = _synthetic(6000, distortion=4.0, seed=4)
    # Generated "too soft": claimed confidence understates observed accuracy.
    accuracy = (dist.argmax(axis=-1) == truths).mean()
    assert dist.max(axis=-1).mean() < accuracy
    t = fit_temperature(dist, truths)
    assert t < 1.0, t  # sharpen
    fixed = apply_temperature(dist, t)
    gap_before = abs(dist.max(axis=-1).mean() - accuracy)
    gap_after = abs(fixed.max(axis=-1).mean() - accuracy)
    assert gap_after < gap_before


def test_leave_one_out_never_fits_a_board_on_itself():
    """Each fold's T must come from the other boards only — the eval-contamination
    lesson applied to calibration."""
    boards = []
    truths = []
    for i in range(6):
        d, t = _synthetic(GRID * GRID, distortion=2.0, seed=10 + i)
        boards.append(d.reshape(GRID, GRID, NUM_CLASSES))
        truths.append(t.reshape(GRID, GRID))
    scaled, temps = leave_one_out_scaled(boards, truths)
    assert len(scaled) == len(temps) == 6
    assert all(s.shape == (GRID, GRID, NUM_CLASSES) for s in scaled)
    # A fold's T must differ from the all-boards T (it saw a different sample)…
    pooled = fit_temperature_over_boards(boards, truths)
    assert any(abs(t - pooled) > 1e-6 for t in temps)
    # …but only slightly, which is the evidence the parameter is real, not noise.
    assert max(temps) / min(temps) < 1.5
    # And no fold may move an answer.
    for before, after in zip(boards, scaled):
        assert (before.argmax(axis=-1) == after.argmax(axis=-1)).all()


def test_leave_one_out_degrades_gracefully_on_a_single_board():
    d, t = _synthetic(GRID * GRID, seed=99)
    boards = [d.reshape(GRID, GRID, NUM_CLASSES)]
    scaled, temps = leave_one_out_scaled(boards, [t.reshape(GRID, GRID)])
    assert temps == [1.0]
    assert np.allclose(scaled[0], boards[0], atol=1e-6)


def test_zero_probabilities_do_not_produce_nan():
    """A hard zero in the true class would make NLL infinite; the clip prevents it."""
    dist = np.zeros((4, NUM_CLASSES), dtype=np.float32)
    dist[:, 0] = 1.0
    truths = np.array([1, 2, 3, 4])  # the true class has probability exactly 0
    assert np.isfinite(negative_log_likelihood(dist, truths, 1.0))
    assert np.isfinite(fit_temperature(dist, truths))
    assert np.isfinite(apply_temperature(dist, 0.3)).all()


def test_non_positive_temperature_is_rejected():
    dist, _ = _synthetic(10)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(dist, bad)


def test_empty_input_fits_the_identity():
    assert fit_temperature(np.empty((0, NUM_CLASSES)), np.empty(0, dtype=int)) == 1.0
    assert fit_temperature_over_boards([], []) == 1.0
