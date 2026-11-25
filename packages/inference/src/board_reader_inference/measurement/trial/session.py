"""Score a recorded live session against its ``truth.json``.

Every served run records itself (:mod:`board_reader_core.recording`): the frames
that crossed the wire, and the ``PositionEvent``\\ s the gateway published from
them. That makes a *trial* — a person sitting at a real board with a phone —
into eval data, but only once someone writes down what was actually on the board.

Scoring reads only ``events.ndjson`` — no model, no GPU, a fraction of a second.

    board-reader-inference eval-session recordings/20260813-193308
"""

from __future__ import annotations

import json
from pathlib import Path

from board_reader_core.recording import Recording

from board_reader_inference.measurement.trial.events import derive_settled, read_events
from board_reader_inference.measurement.trial.evidence import score_evidence
from board_reader_inference.measurement.trial.score import (
    ScoredRead,
    Stream,
    score_stream,
    score_truth_frame,
)
from board_reader_inference.measurement.trial.session_report import SessionReport
from board_reader_inference.measurement.trial.timing import measure_timing
from board_reader_inference.measurement.trial.truth import (
    TRUTH_NAME,
    SessionTruth,
    TruthError,
    load_truth,
)

#: Where a session's score lands by default — beside the recording that produced
#: it. Scoring writes it every time on purpose: a number quoted from a run that
#: left no artifact cannot be checked later, and this project has already lost
#: two headline figures that way.
REPORT_NAME = "report.json"


def score_session(recording: Recording, truth: SessionTruth) -> SessionReport:
    """Join a recording's events to ``truth`` and score both streams."""
    all_events = read_events(recording)
    mine = [e for e in all_events if e.board_id == truth.board_id]
    if not mine:
        boards = sorted({e.board_id for e in all_events})
        raise TruthError(
            f"no events for board {truth.board_id!r} in {recording.path}"
            + (f" — it recorded {boards}" if boards else " — it recorded no events")
        )
    truth.check_covers(mine[0].frame_seq)

    # Derived as one homogeneous stream or not at all: a settled row that is half
    # recorded and half replayed would not be a measurement of either.
    settled_derived = any(e.settled is None for e in mine)
    settled = derive_settled(mine) if settled_derived else [e.settled or e.read for e in mine]

    reads: list[ScoredRead] = []
    excluded_guard = excluded_untruthed = 0
    for event, settled_placement in zip(mine, settled):
        index = truth.index_for(event.frame_seq)
        if index < 0:
            excluded_untruthed += 1
            continue
        if truth.in_guard(event.frame_seq):
            excluded_guard += 1
            continue
        reads.append(
            ScoredRead(
                event=event,
                segment=index,
                raw=event.read,
                settled=settled_placement,
            )
        )

    streams = [
        score_stream(Stream.RAW, reads, truth, derived=False),
        score_stream(Stream.SETTLED, reads, truth, derived=settled_derived),
    ]
    return SessionReport(
        session=str(recording.path),
        board_id=truth.board_id,
        truth=truth,
        streams=streams,
        reads=reads,
        events_total=len(all_events),
        excluded_guard=excluded_guard,
        excluded_untruthed=excluded_untruthed,
        excluded_other_board=len(all_events) - len(mine),
        settled_derived=settled_derived,
        evidence=score_evidence(reads, truth),
        # Timed over every event this board published, not just the scored ones:
        # a guard-excluded read still cost the estimator the same work.
        timing=measure_timing(recording, truth.board_id, mine),
        frame=score_truth_frame(reads, truth),
    )


def run(
    recording: Recording,
    *,
    truth: SessionTruth | None = None,
    json_out: str | None = None,
) -> SessionReport:
    """Score ``recording`` against its ``truth.json``.

    The full report is always written as JSON — to ``json_out`` when given, else
    to :data:`REPORT_NAME` inside the recording. Failing to write it is not
    failing to score: the rendered report still returns.
    """
    report = score_session(recording, truth or load_truth(recording.path / TRUTH_NAME))
    report.written_to = Path(json_out) if json_out else recording.path / REPORT_NAME
    try:
        report.written_to.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
    except OSError as e:
        report.written_to = None
        print(f"could not write the report JSON: {e}")
    return report
