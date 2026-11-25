"""Tests for the evaluation entry point (:mod:`board_reader_inference.measurement.evaluate.run`).

The whole path: a confirmed label on disk, a decoded photo, a score.
"""

from __future__ import annotations

import json

from _evaluate_builders import MIDGAME, FakeRecognizer, dist_with

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg

from board_reader_inference.measurement.evaluate.run import run
from board_reader_inference.measurement.labels import save_label


def test_run_loads_labels_from_disk_and_reads_the_photos(tmp_path):
    """The whole path: a confirmed label on disk -> a decoded photo -> a score."""
    stem = "20260806-210118-0000-estimate"
    (tmp_path / f"{stem}.jpg").write_bytes(encode_jpeg(render_board(MIDGAME)))
    save_label(
        tmp_path, stem,
        frame="chess", placement=MIDGAME, corners=None,
        predicted=MIDGAME, corrections=1, autorotated=0,
    )
    est = FakeRecognizer({stem: dist_with(MIDGAME, {"g1": ("B", 0.9)})})
    report = run([str(tmp_path)], recognizer=est, json_out=str(tmp_path / "out.json"))

    assert est.reads == 1
    assert len(report.boards) == 1 and report.correct == 63
    assert json.loads((tmp_path / "out.json").read_text())["correct_squares"] == 63


def test_run_reads_through_the_recognizer_the_environment_builds(tmp_path, monkeypatch):
    """Without an injected recognizer, ``eval`` scores what ``serve`` would serve."""
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    stem = "20260806-210118-0000-estimate"
    (tmp_path / f"{stem}.jpg").write_bytes(encode_jpeg(render_board(START_PLACEMENT)))
    save_label(
        tmp_path, stem, frame="chess", placement=START_PLACEMENT, corners=None,
        predicted=START_PLACEMENT, corrections=0, autorotated=0,
    )
    report = run([str(tmp_path)])
    assert report.estimator == "StubRecognizer"
    assert report.perfect_boards == 1  # the stub reads the start position


def test_run_on_a_directory_with_nothing_labeled_is_not_an_error(tmp_path):
    report = run([str(tmp_path)], recognizer=FakeRecognizer({}))
    assert report.boards == []
