"""Temperature scaling for the per-square distribution (DESIGN §9, STATUS Next 3).

The estimator's confidences are measured and they are **not honest**: the
deployed ensemble is systematically *under*-confident (ECE 0.226 — bins from
0.3 to 0.8 are essentially always right while claiming 35–75%). DESIGN §9 flags
this as a top risk precisely because downstream smoothing consumes those
numbers, so a consumer that trusts them today is misled.

Temperature scaling is the standard one-parameter fix (Guo et al., 2017): rescale
the logits by a single ``T`` fitted to minimise negative log-likelihood on held-out
labels. Two properties make it the right first move here:

* **It cannot change any answer.** ``q ∝ p^(1/T)`` is a monotone transform of
  each square's distribution, so the argmax — and therefore ``placement`` — is
  bit-identical. Only the confidence moves. Nothing about recognition is at risk.
* **It is one parameter.** With 23 labelled boards, one parameter is roughly what
  the data can support; anything richer would fit noise.

The estimator hands us **probabilities**, not logits (an ensemble is an average
of distributions, and has no logits of its own). Working on ``log p`` is the
faithful generalisation: ``softmax(log(p)/T) == p^(1/T) / sum(p^(1/T))``, which is
exactly logit temperature scaling when ``p`` did come from a softmax.

Fitting and scoring on the same boards would be circular, so
:func:`leave_one_out_scaled` is what the eval reports: each board is scaled by a
``T`` fitted on *the other* boards only.
"""

from __future__ import annotations

import math

import numpy as np

from board_reader_core.protocol import GRID, NUM_CLASSES

#: Probabilities are clipped here before the log — a hard zero would make NLL
#: infinite and drag the fit to a degenerate T.
EPS = 1e-9

#: Search bounds for T. Below 1 sharpens (fixes under-confidence), above 1
#: softens (fixes over-confidence); the deployed ensemble needs the former.
T_MIN, T_MAX = 0.05, 20.0


def apply_temperature(dist: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a distribution: ``q ∝ p^(1/T)``, renormalised.

    Accepts any shape whose last axis is the class axis — one square, one board's
    ``(8, 8, 13)``, or a stack. ``T = 1`` is the identity. The argmax along the
    class axis is unchanged for every ``T > 0``, which is the whole safety
    argument for applying this to a served distribution.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    p = np.asarray(dist, dtype=np.float64)
    if temperature == 1.0:
        return p.astype(np.float32)
    z = np.log(np.clip(p, EPS, None)) / temperature
    z -= z.max(axis=-1, keepdims=True)  # stabilise before exp
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def negative_log_likelihood(
    dists: np.ndarray, truths: np.ndarray, temperature: float
) -> float:
    """Mean NLL of the true class over all squares, at this temperature.

    ``dists`` is ``(N, 13)`` and ``truths`` is ``(N,)`` of class indices.
    """
    scaled = apply_temperature(dists, temperature)
    picked = scaled[np.arange(len(truths)), truths]
    return float(-np.log(np.clip(picked, EPS, None)).mean())


def fit_temperature(
    dists: np.ndarray, truths: np.ndarray, *, iterations: int = 60
) -> float:
    """The ``T`` minimising NLL on these squares.

    Golden-section search over ``log T``. NLL as a function of temperature is
    smooth and unimodal in practice, and a bounded 1-D search needs no optimiser
    dependency — which keeps the inference package's runtime lean.
    """
    dists = np.asarray(dists, dtype=np.float64).reshape(-1, NUM_CLASSES)
    truths = np.asarray(truths).ravel()
    if len(truths) == 0:
        return 1.0

    lo, hi = math.log(T_MIN), math.log(T_MAX)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    fa = negative_log_likelihood(dists, truths, math.exp(a))
    fb = negative_log_likelihood(dists, truths, math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - inv_phi * (hi - lo)
            fa = negative_log_likelihood(dists, truths, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + inv_phi * (hi - lo)
            fb = negative_log_likelihood(dists, truths, math.exp(b))
        if hi - lo < 1e-6:
            break
    return float(math.exp((lo + hi) / 2.0))


def _flatten(board_dists: list[np.ndarray], board_truths: list[np.ndarray]):
    dists = np.concatenate(
        [np.asarray(d, dtype=np.float64).reshape(-1, NUM_CLASSES) for d in board_dists]
    )
    truths = np.concatenate([np.asarray(t).ravel() for t in board_truths])
    return dists, truths


def fit_temperature_over_boards(
    board_dists: list[np.ndarray], board_truths: list[np.ndarray]
) -> float:
    """Pooled ``T`` over every square of every board."""
    if not board_dists:
        return 1.0
    dists, truths = _flatten(board_dists, board_truths)
    return fit_temperature(dists, truths)


def leave_one_out_scaled(
    board_dists: list[np.ndarray], board_truths: list[np.ndarray]
) -> tuple[list[np.ndarray], list[float]]:
    """Each board scaled by a ``T`` fitted on **the other boards only**.

    This is the honest way to report a calibration gain on a set this small: a
    ``T`` fitted on all 23 boards and then scored on those same 23 would flatter
    itself, in exactly the way the eval-contamination incident already taught us
    once. Returns the scaled distributions and the per-board ``T`` that produced
    them, so the spread of ``T`` is visible too — a stable ``T`` across folds is
    itself evidence the parameter is real and not fitting noise.

    With fewer than two boards there is nothing to hold out; the temperatures come
    back as 1.0 and the distributions unchanged.
    """
    n = len(board_dists)
    if n < 2:
        return [np.asarray(d, dtype=np.float32) for d in board_dists], [1.0] * n
    scaled: list[np.ndarray] = []
    temps: list[float] = []
    for i in range(n):
        others_d = [d for j, d in enumerate(board_dists) if j != i]
        others_t = [t for j, t in enumerate(board_truths) if j != i]
        dists, truths = _flatten(others_d, others_t)
        t = fit_temperature(dists, truths)
        temps.append(t)
        scaled.append(
            apply_temperature(np.asarray(board_dists[i]), t).reshape(
                GRID, GRID, NUM_CLASSES
            )
        )
    return scaled, temps
