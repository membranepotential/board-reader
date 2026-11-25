"""Tests for truth parsing and the guard window (:mod:`board_reader_inference.measurement.trial.truth`).

A truth file that cannot be trusted is rejected at the boundary rather than quietly mis-scoring.
"""

from __future__ import annotations

import json

import pytest
from _trial_builders import E4, truth_of

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.measurement.trial.truth import (
    DEFAULT_GUARD_SEQS,
    TRUTH_NAME,
    TruthError,
    TruthSegment,
    load_truth,
)


def _write_truth(path, **over) -> None:
    doc = {
        "format": 1,
        "board_id": "phone",
        "segments": [
            {"from_seq": 0, "placement": START_PLACEMENT},
            {"from_seq": 130, "placement": E4},
        ],
    }
    doc.update(over)
    path.write_text(json.dumps(doc))


def test_load_truth_reads_segments_and_defaults_the_guard(tmp_path):
    _write_truth(tmp_path / TRUTH_NAME)
    truth = load_truth(tmp_path / TRUTH_NAME)
    assert truth.board_id == "phone"
    assert truth.starts == (0, 130)
    assert truth.guard_seqs == DEFAULT_GUARD_SEQS
    assert truth.boundaries == 1


@pytest.mark.parametrize(
    "override, message",
    [
        ({"format": 2}, "unsupported truth format"),
        ({"board_id": ""}, "board_id"),
        ({"segments": []}, "segments"),
        (
            {"segments": [{"from_seq": 10, "placement": START_PLACEMENT},
                          {"from_seq": 10, "placement": E4}]},
            "strictly increasing",
        ),
        (
            {"segments": [{"from_seq": 5, "placement": START_PLACEMENT},
                          {"from_seq": 1, "placement": E4}]},
            "strictly increasing",
        ),
        ({"segments": [{"from_seq": 0, "placement": "not-a-placement"}]}, "segment 0"),
        ({"segments": [{"from_seq": -1, "placement": START_PLACEMENT}]}, "segment 0"),
        ({"segments": [{"placement": START_PLACEMENT}]}, "needs from_seq"),
        ({"guard_seqs": "soon"}, "guard_seqs"),
    ],
)
def test_load_truth_rejects_untrustworthy_files(tmp_path, override, message):
    _write_truth(tmp_path / TRUTH_NAME, **override)
    with pytest.raises(TruthError, match=message):
        load_truth(tmp_path / TRUTH_NAME)


def test_load_truth_reports_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(TruthError, match="cannot read"):
        load_truth(tmp_path / TRUTH_NAME)
    (tmp_path / TRUTH_NAME).write_text("{oops")
    with pytest.raises(TruthError, match="not valid JSON"):
        load_truth(tmp_path / TRUTH_NAME)


def test_segment_lookup_is_a_step_function_over_frame_seq():
    truth = truth_of(
        segments=(
            TruthSegment(10, START_PLACEMENT),
            TruthSegment(100, E4),
        )
    )
    assert truth.index_for(9) == -1  # before any truth
    assert truth.index_for(10) == 0
    assert truth.index_for(99) == 0
    assert truth.index_for(100) == 1
    assert truth.index_for(10_000) == 1


def test_guard_excludes_only_the_window_after_a_boundary():
    truth = truth_of(guard_seqs=5)  # segments start at 0 and 100
    # The session's opening board is not a move: nothing at the start is guarded.
    assert not truth.in_guard(0)
    assert not truth.in_guard(4)
    assert not truth.in_guard(99)
    assert truth.in_guard(100)
    assert truth.in_guard(104)
    assert not truth.in_guard(105)


def test_guard_of_zero_excludes_nothing():
    truth = truth_of(guard_seqs=0)
    assert not truth.in_guard(100)


def test_truth_must_cover_the_stream_it_scores():
    truth_of(segments=(TruthSegment(0, START_PLACEMENT),)).check_covers(7)  # 0 always ok
    truth_of(segments=(TruthSegment(7, START_PLACEMENT),)).check_covers(7)  # or the first seq
    with pytest.raises(TruthError, match="have no truth"):
        truth_of(segments=(TruthSegment(50, START_PLACEMENT),)).check_covers(7)
