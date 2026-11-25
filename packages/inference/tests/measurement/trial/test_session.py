"""Tests for scoring a whole recording (:mod:`board_reader_inference.measurement.trial.session`).

Joins events to truth, writes the report artifact, and is what ``eval-session`` runs.
"""

from __future__ import annotations

import json

import pytest
from _trial_builders import (
    E4,
    FLIPFLOP_EVENTS,
    event_line,
    photo_frames,
    record_session,
    truth_of,
)

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.cli import main
from board_reader_inference.measurement.trial.session import REPORT_NAME, run, score_session
from board_reader_inference.measurement.trial.truth import (
    DEFAULT_GUARD_SEQS,
    TRUTH_NAME,
    TruthError,
    TruthSegment,
)


def test_events_for_another_board_are_counted_not_scored(tmp_path):
    events = [
        event_line(0, START_PLACEMENT),
        event_line(1, START_PLACEMENT, board_id="other"),
    ]
    report = score_session(record_session(tmp_path, events), truth_of())
    assert (report.events_total, report.events_scored) == (2, 1)
    assert report.excluded_other_board == 1


def test_a_board_with_no_events_is_an_error_that_names_what_was_recorded(tmp_path):
    recording = record_session(tmp_path, [event_line(0, START_PLACEMENT, board_id="other")])
    with pytest.raises(TruthError, match="recorded \\['other'\\]"):
        score_session(recording, truth_of())


def test_cli_prints_a_fillable_skeleton_when_truth_is_missing(tmp_path, capsys):
    recording = record_session(tmp_path, FLIPFLOP_EVENTS, photo_frames(1))
    assert main(["eval-session", str(recording.path)]) == 2
    out = capsys.readouterr()
    assert TRUTH_NAME in out.err
    skeleton = json.loads(out.out)
    assert skeleton["board_id"] == "phone"  # taken from the recording, not guessed
    assert skeleton["guard_seqs"] == DEFAULT_GUARD_SEQS


def test_cli_scores_a_session_and_writes_json(tmp_path, capsys):
    recording = record_session(tmp_path / "rec", FLIPFLOP_EVENTS)
    (recording.path / TRUTH_NAME).write_text(json.dumps(truth_of().to_dict()))
    report_json = tmp_path / "report.json"

    assert main(["eval-session", str(recording.path), "--json", str(report_json)]) == 0
    assert "per-square" in capsys.readouterr().out
    assert json.loads(report_json.read_text())["events"]["scored"] == 6


def test_cli_fails_on_a_session_with_nothing_left_to_score(tmp_path, capsys):
    # Every event lands inside the guard after the boundary at seq 100.
    events = [event_line(100, E4), event_line(101, E4)]
    recording = record_session(tmp_path / "rec", events)
    truth = truth_of(
        segments=(TruthSegment(0, START_PLACEMENT), TruthSegment(100, E4)), guard_seqs=50
    )
    (recording.path / TRUTH_NAME).write_text(json.dumps(truth.to_dict()))
    assert main(["eval-session", str(recording.path)]) == 1
    assert "no events left to score" in capsys.readouterr().out


def test_cli_reports_a_bad_truth_file_in_one_line(tmp_path, capsys):
    recording = record_session(tmp_path / "rec", FLIPFLOP_EVENTS)
    (recording.path / TRUTH_NAME).write_text('{"format": 9}')
    assert main(["eval-session", str(recording.path)]) == 2
    assert "unsupported truth format" in capsys.readouterr().err


def test_cli_reports_a_directory_that_is_not_a_recording(tmp_path, capsys):
    assert main(["eval-session", str(tmp_path)]) == 2
    assert "cannot score" in capsys.readouterr().err


def test_scoring_always_leaves_an_artifact_beside_the_recording(tmp_path):
    """A number quoted from a run that left no artifact cannot be checked later —
    this project has already lost two headline figures that way."""
    recording = record_session(tmp_path, FLIPFLOP_EVENTS)
    report = run(recording, truth=truth_of())

    written = recording.path / REPORT_NAME
    assert report.written_to == written
    assert json.loads(written.read_text())["events"]["scored"] == 6
    assert str(written) in report.render()


def test_an_explicit_json_path_wins_over_the_default(tmp_path):
    recording = record_session(tmp_path / "rec", FLIPFLOP_EVENTS)
    out = tmp_path / "elsewhere.json"
    report = run(recording, truth=truth_of(), json_out=str(out))

    assert report.written_to == out
    assert out.exists()
    assert not (recording.path / REPORT_NAME).exists()


def test_an_unwritable_report_does_not_cost_the_score(tmp_path):
    recording = record_session(tmp_path / "rec", FLIPFLOP_EVENTS)
    report = run(
        recording, truth=truth_of(), json_out=str(tmp_path / "no" / "such" / "dir.json")
    )

    assert report.written_to is None
    assert report.events_scored == 6
