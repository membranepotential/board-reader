"""Tests for scoring the raw and settled streams (:mod:`board_reader_inference.measurement.trial.score`).

The two streams are scored *apart*, so a repair that trades one right square for one wrong one shows up
as a different perfect-board rate rather than as a wash.
"""

from __future__ import annotations

from _trial_builders import E4, FLIPFLOP_EVENTS, record_session, truth_of

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.measurement.trial.score import Stream
from board_reader_inference.measurement.trial.session import score_session
from board_reader_inference.measurement.trial.truth import TruthSegment


def test_both_streams_score_differently_on_the_flipflop_session(tmp_path):
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), truth_of())

    assert report.events_total == 8
    assert report.events_scored == 6
    assert report.excluded_guard == 2
    assert report.excluded_untruthed == 0
    assert report.settled_derived is True  # no settled block on these events

    raw, settled = (report.stream(s) for s in Stream)
    # Raw is wrong on d8 and e8 for the two swapped reads: 62/64 each.
    assert (raw.correct, raw.perfect) == (4 * 64 + 2 * 62, 4)
    # The debounce holds the first board through the flip-flop AND takes the
    # move, so every scored board is right.
    assert (settled.correct, settled.perfect) == (6 * 64, 6)
    assert settled.derived is True

    assert raw.confusions == {"q->k": 2, "k->q": 2}
    assert raw.errors == {"type": 4}


def test_per_segment_breakdown_splits_the_move(tmp_path):
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), truth_of())
    before, after = report.stream(Stream.RAW).segments
    assert (before.from_seq, before.to_seq, before.events) == (0, 100, 4)
    assert (after.from_seq, after.to_seq, after.events) == (100, None, 2)
    assert after.perfect == 2  # the two post-guard reads of the new board


def test_reads_in_the_truths_frame_pass_the_frame_check(tmp_path):
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), truth_of())
    assert report.frame.by_rotation == (6, 0, 0, 0)
    assert report.frame.verdict == 0
    assert report.to_dict()["truth_frame"]["verdict_deg"] == 0
    assert "6/6 raw reads agree with the truth unturned" in report.render()


def test_truth_written_a_half_turn_off_is_flagged(tmp_path):
    from board_reader_inference.measurement.labels import rotate_placement

    turned = truth_of(
        segments=(
            TruthSegment(0, rotate_placement(START_PLACEMENT, 2)),
            TruthSegment(100, rotate_placement(E4, 2)),
        )
    )
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), turned)
    assert report.frame.verdict == 2
    assert "WARNING 6/6 raw reads agree with the truth best turned 180°" in report.render()
