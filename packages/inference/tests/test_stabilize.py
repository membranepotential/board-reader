"""Tests for the per-square debounce behind ``PositionEvent.settled``."""

from __future__ import annotations

import pytest

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.stabilize import BoardStabilizer

#: START_PLACEMENT after 1.e4 — one real move, two squares changed.
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: A phantom: the trial's hallucinated black rook on b6.
PHANTOM_B6 = "rnbqkbnr/pppppppp/1r6/8/8/8/PPPPPPPP/RNBQKBNR"


def test_first_read_is_adopted_wholesale():
    settled = BoardStabilizer().observe(START_PLACEMENT, 0.0)
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ()


def test_a_real_move_is_adopted_once_it_spans_the_window():
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(E4, 1.0).placement == START_PLACEMENT
    held = stab.observe(E4, 2.0)  # consistent, but only 1.0s into the window
    assert held.placement == START_PLACEMENT
    assert held.pending == ("e4", "e2")  # board order: rank 8 first
    done = stab.observe(E4, 3.6)  # 2.6s of unbroken E4: adopted
    assert done.placement == E4
    assert done.pending == ()


def test_wall_clock_alone_is_not_evidence():
    """One read followed by silence must not be adopted on its second sighting
    hours later — at least SETTLE_MIN_READS *consistent* reads are required."""
    stab = BoardStabilizer(settle_seconds=2.5, settle_min_reads=3)
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(E4, 10.0)
    assert stab.observe(E4, 20.0).placement == START_PLACEMENT  # 2 reads < 3
    assert stab.observe(E4, 20.5).placement == E4


def test_a_flicker_never_reaches_the_settled_board():
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 0.0)
    flickered = stab.observe(PHANTOM_B6, 1.0)
    assert flickered.placement == START_PLACEMENT
    assert flickered.pending == ("b6",)
    recovered = stab.observe(START_PLACEMENT, 1.5)
    assert recovered.placement == START_PLACEMENT
    assert recovered.pending == ()
    # the run restarted: the phantom's clock does not carry over
    stab.observe(PHANTOM_B6, 2.0)
    assert stab.observe(PHANTOM_B6, 4.0).placement == START_PLACEMENT


def test_alternating_reads_never_accumulate():
    """The trial's d8/e8 k-q coin flip: each disagreement restarts the run."""
    a = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # k on d8
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 0.0)
    for i in range(20):  # 10s of alternation, far beyond the window
        assert stab.observe(a if i % 2 else START_PLACEMENT, float(i)).placement == (
            START_PLACEMENT
        )


def test_a_backwards_timestamp_restarts_the_run():
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 100.0)
    stab.observe(E4, 101.0)
    stab.observe(E4, 90.0)  # edge clock stepped back: evidence discarded
    assert stab.observe(E4, 92.0).placement == START_PLACEMENT
    assert stab.observe(E4, 93.0).placement == E4  # 3.0s from the 90.0 restart


def test_zero_window_still_needs_min_reads():
    stab = BoardStabilizer(settle_seconds=0.0)
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(PHANTOM_B6, 1.0).placement == START_PLACEMENT
    assert stab.observe(PHANTOM_B6, 2.0).placement == PHANTOM_B6


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        BoardStabilizer(settle_seconds=-1.0)
    with pytest.raises(ValueError):
        BoardStabilizer(settle_min_reads=0)


def test_rejects_a_malformed_placement():
    with pytest.raises(ValueError):
        BoardStabilizer().observe("not/a/board", 0.0)
