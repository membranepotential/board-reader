"""Tests for the ported motion meter and restamping (:mod:`board_reader_inference.stream.schedule.meter`).

The ported meter is the **edge's meter**, value for value.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.recording import FrameEntry

from board_reader_inference.stream.schedule.constants import (
    DEFAULT_KEEPALIVE_S,
    ENTER_BURST,
    EXIT_BURST,
)
from board_reader_inference.stream.schedule.meter import (
    RESTAMPED_KEY,
    MotionMeter,
    restamped,
)
from board_reader_inference.stream.schedule.policies import BurstTriggered


def entries(stabilities, *, dt: float = 0.12, first_ts: float = 1_000.0):
    """One header per stability, ``dt`` recorded seconds apart."""
    return [
        FrameEntry(
            header={
                "board_id": "phone",
                "seq": i,
                "capture_ts": first_ts + i * dt,
                "stability": s,
            },
            jpeg_rel=f"jpeg/0000/{i:06d}.jpg",
            received_ts=first_ts + i * dt,
        )
        for i, s in enumerate(stabilities)
    ]


# --------------------------------------------------------------------------- #
# The ported meter
# --------------------------------------------------------------------------- #


def _frames(n: int, *, moving: bool = False, seed: int = 0) -> list[np.ndarray]:
    """A static scene with sensor noise, optionally with something crossing it."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        image = scene.astype(np.int16) + rng.integers(-2, 3, scene.shape)
        if moving:
            image[:, (i * 4) % 40 : (i * 4) % 40 + 8] = 0
        out.append(np.clip(image, 0, 255).astype(np.uint8))
    return out


def test_the_meter_reports_rest_on_the_first_frame_and_finds_the_floor():
    meter = MotionMeter()
    scores = [meter.update(f) for f in _frames(30)]
    assert scores[0] == 1.0
    assert scores[-1] > EXIT_BURST, "a still scene settles above the burst-exit line"


def test_the_meter_drops_on_motion_and_recovers():
    meter = MotionMeter()
    for f in _frames(30):
        meter.update(f)
    for f in _frames(10, moving=True, seed=1):
        meter.update(f)
    assert meter.stability < ENTER_BURST
    for f in _frames(30, seed=2):
        meter.update(f)
    assert meter.stability > EXIT_BURST


def test_the_ported_meter_matches_the_edges_value_for_value():
    """The mirror, pinned.

    :class:`MotionMeter` is a copy of ``board_reader_edge.motion.MotionMeter``
    because DESIGN §4 forbids this package from importing the edge's (see
    ``tests/test_layer_separation.py``, which scans ``src`` only — a test may
    import both, and this is exactly what that is for). A copy with no equality
    check is a fork waiting to happen, so: same frames, same numbers.
    """
    edge_motion = pytest.importorskip("board_reader_edge.motion")
    ours, theirs = MotionMeter(), edge_motion.MotionMeter()
    frames = _frames(20) + _frames(10, moving=True, seed=3) + _frames(20, seed=4)
    assert [ours.update(f) for f in frames] == [theirs.update(f) for f in frames]


# --------------------------------------------------------------------------- #
# Restamping
# --------------------------------------------------------------------------- #


def test_restamping_overrides_the_recorded_value_and_says_so():
    [entry] = entries([0.31])
    meter = MotionMeter()
    out = restamped(entry, meter, _frames(1)[0])

    assert out.header["stability"] == 1.0  # the meter's first-frame invariant
    assert out.header[RESTAMPED_KEY] is True
    assert out.seq == entry.seq and out.capture_ts == entry.capture_ts
    assert entry.header["stability"] == 0.31, "the recorded entry is not mutated"


def test_restamping_a_dead_meters_take_is_what_makes_it_schedulable():
    """The data caveat, end to end: the recorded field says "moving" on every
    frame of a still scene, and a policy reading it can only ever hit its floor.
    Re-metered, the same frames read as rest."""
    frames = _frames(40)
    index = entries([0.29] * len(frames))  # what the old meter actually wrote
    stale = BurstTriggered(keepalive_s=DEFAULT_KEEPALIVE_S)
    for e in index:
        stale.should_estimate(e)
    assert stale.bursting, "a still scene, reported as one long move"

    meter = MotionMeter()
    fresh = [restamped(e, meter, f) for e, f in zip(index, frames, strict=True)]
    live = BurstTriggered(keepalive_s=DEFAULT_KEEPALIVE_S)
    for e in fresh:
        live.should_estimate(e)

    assert all(e.header["stability"] > EXIT_BURST for e in fresh[5:])
    assert not live.bursting, "re-metered, the same frames read as rest"
