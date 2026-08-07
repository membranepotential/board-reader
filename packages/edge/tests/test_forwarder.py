"""Tests for the adaptive frame forwarder (DESIGN.md D4, WP-C3).

The whole point of D4's capture-time base is that these tests need no clock: the
frame stream is built by hand with explicit ``capture_ts`` arithmetic, and the
forwarder must make the same decisions a live edge would.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_edge import forwarder as forwarder_module
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig

CAPTURE_FPS = 30.0
BOARD = render_board(square_px=30)  # 240x240 BGR


def rest(n: int) -> list[np.ndarray]:
    """``n`` frames of a perfectly still board."""
    return [BOARD.copy() for _ in range(n)]


def motion(n: int) -> list[np.ndarray]:
    """``n`` frames of a hand-sized bright block sweeping across the board."""
    images = []
    for i in range(n):
        img = BOARD.copy()
        x = 20 + 60 * (i % 3)
        img[80:160, x : x + 80] = 255
        images.append(img)
    return images


def stream(images, *, base_ts: float = 0.0, fps: float = CAPTURE_FPS) -> list[Frame]:
    """Wrap images as a capture stream with evenly spaced capture timestamps."""
    return [
        Frame(board_id="A1", seq=i, capture_ts=base_ts + i / fps, image=image)
        for i, image in enumerate(images)
    ]


def test_the_first_frame_is_always_forwarded():
    frames = stream(rest(1))
    forwarded = AdaptiveForwarder().decide(frames[0])
    assert forwarded is not None
    assert forwarded.seq == 0


def test_at_rest_forwards_at_the_keepalive_rate():
    # 90 still frames = 3.0s of capture at 30 fps; 1 fps keepalive -> t=0,1,2.
    out = list(AdaptiveForwarder().forward(stream(rest(90))))
    assert [f.seq for f in out] == [0, 30, 60]


def test_at_rest_frames_carry_full_stability_and_no_burst_id():
    out = list(AdaptiveForwarder().forward(stream(rest(60))))
    assert all(f.stability == 1.0 and f.burst_id is None for f in out)


def test_the_keepalive_rate_is_configurable():
    config = ForwarderConfig(keepalive_fps=2.0)
    out = list(AdaptiveForwarder(config).forward(stream(rest(60))))
    assert [f.seq for f in out] == [0, 15, 30, 45]


def test_motion_bursts_at_the_burst_rate():
    at_rest = list(AdaptiveForwarder().forward(stream(rest(60))))
    moving = list(AdaptiveForwarder().forward(stream(motion(60))))
    # 2.0s at 12 fps burst, quantised to the 30 fps capture grid -> every 3rd frame.
    assert 18 <= len(moving) <= 21
    assert len(moving) > 5 * len(at_rest)
    gaps = [b.capture_ts - a.capture_ts for a, b in itertools.pairwise(moving)]
    assert max(gaps) <= 1 / 12.0 + 1 / CAPTURE_FPS


def test_a_burst_shares_one_id_across_all_its_frames():
    out = list(AdaptiveForwarder().forward(stream(motion(60))))
    ids = {f.burst_id for f in out[1:]}
    assert len(ids) == 1
    burst_id = ids.pop()
    assert burst_id is not None and burst_id.startswith("A1:")


def test_forwarded_frames_are_stamped_with_the_measured_stability():
    config = ForwarderConfig()
    out = list(AdaptiveForwarder(config).forward(stream(motion(60))))
    assert out[0].stability == 1.0  # the very first frame has no predecessor
    assert all(f.stability < config.enter_burst for f in out[1:])


def test_motion_between_keepalive_slots_starts_the_burst_immediately():
    # One still frame, then a sweep. The pending keepalive slot is a second away,
    # but the burst rate applies as soon as motion is detected.
    out = list(AdaptiveForwarder().forward(stream(rest(1) + motion(30))))
    assert out[1].capture_ts <= 1 / 12.0 + 1 / CAPTURE_FPS


def test_a_brief_calm_does_not_end_a_burst():
    # 10 still frames (0.33s) is long enough for stability to climb back over
    # exit_burst, but far short of the 0.7s hold — one burst, not two.
    frames = stream(motion(15) + rest(10) + motion(15))
    out = list(AdaptiveForwarder().forward(frames))
    ids = {f.burst_id for f in out[1:]}
    assert len(ids) == 1


def test_sustained_calm_ends_the_burst():
    forwarder = AdaptiveForwarder()
    out = list(forwarder.forward(stream(motion(15) + rest(60))))
    assert any(f.burst_id is not None for f in out)  # it did burst
    assert not forwarder.bursting
    assert forwarder.burst_id is None
    assert out[-1].burst_id is None
    assert out[-1].stability > ForwarderConfig().exit_burst
    # ...and once at rest it is back to the keepalive rate: the last two forwarded
    # frames are ~1s apart, not ~0.08s.
    assert out[-1].capture_ts - out[-2].capture_ts == pytest.approx(1.0, abs=0.05)


def test_a_second_burst_gets_a_new_id():
    frames = stream(motion(15) + rest(60) + motion(15))
    out = list(AdaptiveForwarder().forward(frames))
    ids = [f.burst_id for f in out if f.burst_id is not None]
    assert len(set(ids)) == 2


def test_capture_ts_is_the_time_base_not_the_iteration_speed():
    # The same images, timestamped as 30 fps capture vs. 1 fps capture. Nothing
    # about the loop changes — only capture_ts — and the cadence follows it.
    images = rest(60)
    fast = list(AdaptiveForwarder().forward(stream(images, fps=30.0)))
    slow = list(AdaptiveForwarder().forward(stream(images, fps=1.0)))
    assert [f.seq for f in fast] == [0, 30]
    assert len(slow) == 60


def test_forward_is_decide_over_a_stream():
    frames = stream(rest(5) + motion(20))
    one = list(AdaptiveForwarder().forward(frames))
    driver = AdaptiveForwarder()
    other = [f for f in (driver.decide(frame) for frame in frames) if f is not None]
    assert [(f.seq, f.stability, f.burst_id) for f in one] == [
        (f.seq, f.stability, f.burst_id) for f in other
    ]


def test_forwarding_does_not_mutate_the_captured_frame():
    frames = stream(motion(10))
    list(AdaptiveForwarder().forward(frames))
    assert all(f.stability == 1.0 and f.burst_id is None for f in frames)


def test_a_backwards_clock_step_does_not_stall_forwarding():
    # D5: capture_ts is wall clock and can step. Re-anchor rather than go silent.
    frames = stream(rest(3))
    frames[2].capture_ts = -10.0
    out = list(AdaptiveForwarder().forward(frames))
    assert [f.seq for f in out] == [0, 2]


def test_the_forwarder_never_reads_a_wall_clock():
    tree = ast.parse(Path(forwarder_module.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"time", "datetime"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keepalive_fps": 0.0},
        {"burst_fps": -1.0},
        {"enter_burst": 0.9, "exit_burst": 0.5},  # inverted hysteresis
        {"exit_hold_s": -0.1},
    ],
)
def test_config_rejects_nonsensical_settings(kwargs):
    with pytest.raises(ValueError):
        ForwarderConfig(**kwargs)
