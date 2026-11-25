"""Tests for the trial timing score (:mod:`board_reader_inference.measurement.trial.timing`).
"""

from __future__ import annotations

import pytest
from _trial_builders import (
    FLIPFLOP_EVENTS,
    event_line,
    photo_frames,
    record_session,
    truth_of,
)

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.measurement.trial.session import score_session


def test_timing_measures_input_rate_against_read_rate(tmp_path):
    """The 08-13 finding in miniature: frames arrive far faster than reads are
    published, and the report says so in one line instead of by hand."""
    frames = photo_frames(8)
    recording = record_session(tmp_path, FLIPFLOP_EVENTS, frames)
    t = score_session(recording, truth_of()).timing

    assert t.frames == 8
    assert t.events == 8  # every event on this board, guard-excluded ones included
    assert t.frames_per_read == pytest.approx(1.0)
    assert t.latencies and all(lat == pytest.approx(2.0) for lat in t.latencies)
    assert t.to_dict()["latency"]["median"] == pytest.approx(2.0)


def test_timing_says_nothing_rather_than_zero_when_a_session_has_one_frame(tmp_path):
    recording = record_session(tmp_path, [event_line(0, START_PLACEMENT)])
    t = score_session(recording, truth_of()).timing

    assert t.fps == 0.0 and t.reads_per_s == 0.0  # no span to divide by
    assert "timing" in score_session(recording, truth_of()).render()


def test_timing_counts_only_the_scored_board(tmp_path):
    frames = photo_frames(2)
    events = [
        event_line(0, START_PLACEMENT),
        event_line(1, START_PLACEMENT, board_id="other"),
    ]
    t = score_session(record_session(tmp_path, events, frames), truth_of()).timing

    assert t.events == 1
