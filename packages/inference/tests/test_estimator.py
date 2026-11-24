"""Tests for the walking-skeleton estimator (DESIGN.md D1 / L2)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import (
    CLASS_INDEX,
    EMPTY_PLACEMENT,
    NUM_CLASSES,
    START_PLACEMENT,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import Estimator, StubEstimator


def _frame(seq=0, ts=0.0, stability=1.0, clock="none", image=None):
    return Frame(
        board_id="A1", seq=seq, capture_ts=ts,
        image=render_board() if image is None else image,
        stability=stability, clock=clock,
    )


def test_stub_estimator_satisfies_protocol():
    assert isinstance(StubEstimator(), Estimator)


def test_stub_emits_configured_placement_as_valid_event():
    ev = StubEstimator(START_PLACEMENT).estimate(_frame())
    ev.validate()
    assert ev.placement == START_PLACEMENT
    assert ev.status == "ok"


def test_stub_passes_through_frame_metadata():
    ev = StubEstimator().estimate(_frame(seq=11, ts=0.0, stability=0.25, clock="white"))
    assert ev.seq == 11
    assert ev.utc == iso_utc(0.0)
    assert ev.stability == 0.25
    assert ev.clock.running == "white"


def test_stub_confidence_is_configurable():
    ev = StubEstimator(EMPTY_PLACEMENT, confidence=0.6).estimate(_frame())
    assert 0.59 <= ev.confidence <= 0.61
    assert ev.placement == EMPTY_PLACEMENT


def test_stub_is_stateless_same_output_each_frame():
    est = StubEstimator()
    a = est.estimate(_frame(seq=0))
    b = est.estimate(_frame(seq=1))
    assert a.placement == b.placement  # ignores pixels, no game state


# --- CentroidEstimator (M1 recognition) ---------------------------------------


def _square(**probs: float) -> np.ndarray:
    """One square's distribution: named classes at their probability, rest flat."""
    row = np.full(NUM_CLASSES, (1.0 - sum(probs.values())) / (NUM_CLASSES - len(probs)))
    for cls, p in probs.items():
        row[CLASS_INDEX[cls]] = p
    return row.astype(np.float32)


def _board(**squares: np.ndarray) -> np.ndarray:
    """An 8x8x13 board of confident empties, with named squares overridden."""
    dist = np.asarray(dist_from_placement(EMPTY_PLACEMENT), dtype=np.float32)
    for name, row in squares.items():
        r, c = 8 - int(name[1]), ord(name[0]) - ord("a")
        dist[r, c] = row
    return dist
