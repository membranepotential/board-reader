"""Tests for the trial report (:mod:`board_reader_inference.measurement.trial.session_report`).
"""

from __future__ import annotations

import json

from _trial_builders import FLIPFLOP_EVENTS, record_session, truth_of

from board_reader_inference.measurement.trial.score import Stream
from board_reader_inference.measurement.trial.session import score_session


def test_report_renders_and_serialises(tmp_path):
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), truth_of())
    text = report.render()
    assert "DERIVED" in text
    assert "per segment" in text
    for stream in Stream:
        assert stream.value in text
    doc = report.to_dict()
    assert doc["events"] == {
        "total": 8,
        "scored": 6,
        "excluded_guard": 2,
        "excluded_untruthed": 0,
        "excluded_other_board": 0,
    }
    assert [s["stream"] for s in doc["streams"]] == ["raw", "settled"]
    json.dumps(doc)  # the --json path must not choke on a Counter or a Path
