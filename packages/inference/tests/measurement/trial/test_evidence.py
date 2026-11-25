"""Tests for the wrong-square evidence score (:mod:`board_reader_inference.measurement.trial.evidence`).
"""

from __future__ import annotations

import pytest
from _trial_builders import (
    FLIPFLOP_EVENTS,
    TWO_KINGS,
    digest,
    event_line,
    record_session,
    truth_of,
)

from board_reader_inference.measurement.trial.session import score_session


def test_evidence_separates_a_coin_flip_from_a_conviction(tmp_path):
    """The distinction the whole digest exists for: d8 read k over q by 0.04 (the
    crop carries the answer, the model nearly found it), while b6 hallucinated a
    rook with truth nowhere in the top 2 (the evidence is simply not there)."""
    events = [
        event_line(
            0,
            TWO_KINGS,
            evidence=digest(TWO_KINGS, d8=("q", 0.51, 0.47)),
        ),
        event_line(
            10,
            TWO_KINGS,
            evidence=digest(TWO_KINGS, d8=("n", 0.90, 0.05)),
        ),
    ]
    report = score_session(record_session(tmp_path, events), truth_of())
    e = report.evidence

    assert e.recorded and e.events == 2 and e.events_total == 2
    assert e.wrong == 2  # d8 in both events
    assert e.truth_runner_up == 1  # only the first had q second
    assert e.truth_ranked == 1
    assert e.margins == [pytest.approx(0.04, abs=1e-6)]
    assert [(c.pair, c.n, c.truth_ranked) for c in e.by_confusion] == [("q->k", 2, 1)]


def test_evidence_is_absent_not_zero_for_a_session_that_recorded_none(tmp_path):
    """Old recordings predate the digest. They score exactly as before, and the
    report says the evidence is missing rather than implying it was empty."""
    report = score_session(record_session(tmp_path, FLIPFLOP_EVENTS), truth_of())

    assert not report.evidence.recorded
    assert report.evidence.wrong == 0
    assert report.evidence.to_dict()["margin_median"] is None
    assert "predates the top-2 digest" in report.render()


def test_evidence_survives_a_malformed_digest(tmp_path):
    """A digest is diagnostic, never load-bearing: garbage there must not cost a
    session its score."""
    events = [
        event_line(0, TWO_KINGS, evidence=[["k", 0.5]]),  # wrong shape
        event_line(
            10, TWO_KINGS, evidence=digest(TWO_KINGS, d8=("q", 0.6, 0.3))
        ),
    ]
    report = score_session(record_session(tmp_path, events), truth_of())

    assert report.events_scored == 2  # both scored
    assert report.evidence.events == 1  # only one carried a readable digest
    assert report.evidence.wrong == 1


def test_evidence_renders_its_margins(tmp_path):
    events = [
        event_line(
            0, TWO_KINGS, evidence=digest(TWO_KINGS, d8=("q", 0.51, 0.47))
        )
    ]
    text = score_session(record_session(tmp_path, events), truth_of()).render()

    assert "truth was the runner-up" in text
    assert "margin p(read)-p(truth)" in text
    assert "q->k" in text
