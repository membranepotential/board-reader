"""Tests for the evaluation report (:mod:`board_reader_inference.measurement.evaluate.report`).

The report counts boards, not squares, and a confident wrong answer shows up as a calibration gap.
"""

from __future__ import annotations

import json

from _evaluate_builders import (
    MIDGAME,
    FakeRecognizer,
    chess_board,
    corner_frame_board,
    dist_with,
    labeled_board,
)

from board_reader_core.protocol import dist_from_placement

from board_reader_inference.measurement.evaluate.run import evaluate
from board_reader_inference.recognition.recognizer import StubRecognizer


def test_report_headline_counts_boards_not_squares():
    dists = {
        "clean": dist_from_placement(MIDGAME),
        "one_off": dist_with(MIDGAME, {"g1": ("B", 0.9)}),
        "two_off": dist_with(MIDGAME, {"g1": ("B", 0.9), "d2": ("P", 0.9)}),
    }
    boards = [labeled_board(k) for k in dists]
    report = evaluate(FakeRecognizer(dists), boards)

    assert report.perfect_boards == 1
    assert len(report.boards) == 3
    # 3 wrong squares out of 192 reads as 98.4% per-square but only 33% of boards.
    assert report.squares == 192 and report.correct == 189
    assert report.errors_by_kind() == {"type": 2, "hallucination": 1}
    assert report.confusions()["K->B"] == 2

    rendered = report.render()
    assert "PERFECT BOARDS   1/3" in rendered
    assert "per-square       189/192" in rendered
    assert "K->B" in rendered


def test_report_is_json_serialisable_and_round_trips_the_numbers(tmp_path):
    dists = {"one_off": dist_with(MIDGAME, {"g1": ("B", 0.9)})}
    report = evaluate(FakeRecognizer(dists), [labeled_board("one_off")])
    out = json.loads(json.dumps(report.to_dict()))
    assert out["boards"] == 1 and out["perfect_boards"] == 0
    assert out["correct_squares"] == 63
    assert out["per_board"][0]["errors"][0]["square"] == "g1"


def test_empty_input_reports_rather_than_crashes():
    report = evaluate(FakeRecognizer({}), [])
    assert report.boards == [] and report.perfect_boards == 0
    assert "no labeled boards" in report.render()


def test_a_confidently_wrong_board_shows_up_as_a_gap_not_an_average():
    """The 2026-08-06 field failure: two white kings, no black king, at 0.97."""
    broken = dist_with(MIDGAME, {"e8": ("K", 0.97), "g1": ("K", 0.97)})
    report = evaluate(FakeRecognizer({"broken": broken}), [labeled_board("broken")])
    assert not report.boards[0].perfect
    assert report.boards[0].confidence > 0.85  # the board still looks confident
    top = [b for b in report.bins if b.lo >= 0.9]
    assert top and any(b.gap > 0 for b in top)
    assert "confidence reliability" in report.render()


def test_report_separates_confidence_of_right_and_wrong_boards():
    dists = {
        "clean": dist_from_placement(MIDGAME, confidence=0.9),
        "broken": dist_with(MIDGAME, {"e8": ("K", 0.97)}),
    }
    report = evaluate(FakeRecognizer(dists), [labeled_board(k) for k in dists])
    rendered = report.render()
    assert "board confidence:" in rendered
    assert "perfect" in rendered and "imperfect" in rendered


def test_a_recognizer_reads_chess_labels_whole_and_skips_corner_frame_ones():
    report = evaluate(
        StubRecognizer(MIDGAME), [chess_board(), corner_frame_board("legacy")]
    )
    assert [b.board_id for b in report.boards] == ["c0"]
    assert report.perfect_boards == 1 and report.skipped == ["legacy"]
    assert any("corner-frame label(s) skipped" in w for w in report.warnings)
    assert report.estimator == "StubRecognizer"
    assert report.to_dict()["rotated_boards"] == []


def test_a_turned_read_is_flagged_by_the_frame_diagnostic():
    from board_reader_inference.measurement.labels import rotate_placement

    report = evaluate(StubRecognizer(rotate_placement(MIDGAME, 2)), [chess_board()])
    assert report.boards[0].rotation == 2
    assert report.to_dict()["rotated_boards"] == [{"board_id": "c0", "rotation": 180}]
    assert "TURNED: c0 180°" in report.render()
