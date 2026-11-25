"""Tests for reading recorded events (:mod:`board_reader_inference.measurement.trial.events`).

Three line generations are read, and a settled board is derived only when the recording never had one.
"""

from __future__ import annotations

from _trial_builders import (
    E4,
    SWAPPED,
    TWO_KINGS,
    digest,
    event_line,
    record_session,
    truth_of,
)

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.measurement.trial.events import read_events
from board_reader_inference.measurement.trial.score import Stream
from board_reader_inference.measurement.trial.session import score_session


def test_read_events_skips_junk_and_keeps_order(tmp_path):
    recording = record_session(tmp_path, [event_line(0, START_PLACEMENT), event_line(5, E4)])
    with (recording.path / "events.ndjson").open("a") as fh:
        fh.write("{truncated\n\n")
    events = read_events(recording)
    assert [e.frame_seq for e in events] == [0, 5]
    assert events[0].settled is None
    assert events[0].read == START_PLACEMENT


def test_a_one_placement_line_reads_the_settled_board_and_the_read_from_its_digest(tmp_path):
    """Since 2026-11-24 a line carries one board, the settled one, with ``pending``;
    the read is the top class of each square's evidence."""
    line = event_line(0, START_PLACEMENT, evidence=digest(TWO_KINGS, d8=("q", 0.51, 0.47)))
    line["pending"] = ["d8"]
    (event,) = read_events(record_session(tmp_path, [line]))
    assert (event.read, event.settled) == (TWO_KINGS, START_PLACEMENT)

    report = score_session(record_session(tmp_path / "again", [line]), truth_of())
    assert report.settled_derived is False
    assert report.stream(Stream.RAW).perfect == 0  # the read had two kings
    assert report.stream(Stream.SETTLED).perfect == 1


def test_recorded_settled_is_used_as_is(tmp_path):
    """A line from before the flat event: ``placement`` is the read and the
    settled board rides in its own block."""
    events = [
        event_line(0, TWO_KINGS, settled=START_PLACEMENT),
        event_line(10, TWO_KINGS, settled=SWAPPED),
    ]
    report = score_session(record_session(tmp_path, events), truth_of())
    assert report.settled_derived is False
    settled = report.stream(Stream.SETTLED)
    assert settled.derived is False
    # SWAPPED taken verbatim — a derivation would have held START through it.
    assert settled.perfect == 1
