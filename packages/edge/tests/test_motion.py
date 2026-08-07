"""Tests for the edge motion meter (DESIGN.md D3, WP-C2).

Everything here is deterministic: fixed synthetic images, no clock, no randomness.
A "hand" is a bright rectangle swept over a rendered board.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from board_reader_core.sim import render_board

from board_reader_edge.motion import MotionMeter

BOARD = render_board(square_px=30)  # 240x240 BGR


def still() -> np.ndarray:
    return BOARD.copy()


def swept(step: int) -> np.ndarray:
    """The board with a hand-sized bright block at one of three positions."""
    img = still()
    x = 20 + 60 * (step % 3)
    img[80:160, x : x + 80] = 255
    return img


def test_first_frame_is_reported_at_rest():
    assert MotionMeter().update(still()) == 1.0


def test_identical_frames_stay_at_rest():
    meter = MotionMeter()
    scores = [meter.update(still()) for _ in range(5)]
    assert scores == [1.0] * 5


def test_sensor_noise_does_not_disturb_rest():
    meter = MotionMeter()
    meter.update(still())
    noisy = still().astype(np.int16) + 1  # a uniform 1/255 lift: below the floor
    assert meter.update(np.clip(noisy, 0, 255).astype(np.uint8)) > 0.95


def test_a_hand_sweep_drops_stability_well_below_half():
    meter = MotionMeter()
    meter.update(still())
    scores = [meter.update(swept(i)) for i in range(2)]
    assert scores[0] < 0.5  # within a single frame
    assert min(scores) < 0.3


def test_stability_recovers_towards_rest_after_motion_stops():
    meter = MotionMeter()
    meter.update(still())
    for i in range(4):
        meter.update(swept(i))
    assert meter.stability < 0.5
    # The first still frame is itself motion (the hand leaving), so recovery is
    # counted from there.
    recovery = [meter.update(still()) for _ in range(8)]
    assert recovery == sorted(recovery)  # monotonically rising
    assert recovery[4] > 0.85  # back over the forwarder's burst-exit threshold
    assert recovery[-1] > 0.95


def test_stability_property_tracks_the_last_update():
    meter = MotionMeter()
    for image in (still(), swept(0), swept(1), still()):
        scored = meter.update(image)
        assert meter.stability == scored


def test_reset_forgets_the_reference_frame():
    meter = MotionMeter()
    meter.update(still())
    meter.update(swept(0))
    assert meter.stability < 1.0
    meter.reset()
    assert meter.update(swept(1)) == 1.0  # no reference -> at rest again


def test_scoring_is_independent_of_capture_resolution():
    # The thumbnail is fixed-size, so the same motion scores the same whether the
    # camera hands us 240px or 720px frames.
    def run(scale: int) -> float:
        meter = MotionMeter()
        for _i, image in enumerate([still(), swept(0), swept(1)]):
            frame = cv2.resize(image, None, fx=scale, fy=scale)
            meter.update(frame)
        return meter.stability

    assert run(1) == pytest.approx(run(3), abs=0.05)


def test_two_meters_score_an_identical_stream_identically():
    images = [still(), swept(0), swept(1), still(), still()]
    a, b = MotionMeter(), MotionMeter()
    assert [a.update(i) for i in images] == [b.update(i) for i in images]


def test_grayscale_input_is_accepted():
    meter = MotionMeter()
    gray = cv2.cvtColor(still(), cv2.COLOR_BGR2GRAY)
    assert meter.update(gray) == 1.0
    assert meter.update(gray) == 1.0


@pytest.mark.parametrize("kwargs", [{"downscale": 1}, {"attack": 0.0}, {"release": 1.5}])
def test_rejects_nonsensical_configuration(kwargs):
    with pytest.raises(ValueError):
        MotionMeter(**kwargs)
