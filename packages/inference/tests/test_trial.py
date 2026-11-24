"""Tests for the trial scorer (:mod:`board_reader_inference.trial`).

The load-bearing claims, in order: a truth file that cannot be trusted is
rejected at the boundary rather than quietly mis-scoring; the guard window drops
exactly the events a hand was over the board for; and the three streams are
scored *apart*, so a repair that trades one right square for one wrong one shows
up as a different perfect-board rate rather than as a wash.
"""

from __future__ import annotations

import json

import pytest

from board_reader_core.protocol import GRID, START_PLACEMENT, parse_placement
from board_reader_core.recording import FrameRecorder, Recording
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.cli import main
from board_reader_inference.trial import (
    DEFAULT_GUARD_SEQS,
    REPORT_NAME,
    SQUARE_INDEX,
    TRUTH_NAME,
    SessionTruth,
    Stream,
    TruthError,
    TruthSegment,
    load_truth,
    read_events,
    run,
    score_session,
)

# Rank 8 read with two black kings — the 08-13 trial's actual raw failure. One
# square wrong (d8), and impossible, so the prior decoder has something to repair.
TWO_KINGS = "rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
# What the coin-flip repair produces on the other half of its flips: possible,
# still wrong, and wrong on *two* squares.
SWAPPED = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


# --------------------------------------------------------------------------- #
# Building a session on disk
# --------------------------------------------------------------------------- #


def _event(
    frame_seq: int,
    placement: str,
    *,
    prior: str | None = None,
    settled: str | None = None,
    board_id: str = "phone",
    evidence: list | None = None,
) -> dict:
    """One events.ndjson line, in the shape ``server._record_published`` writes."""
    line: dict = {
        "board_id": board_id,
        "seq": frame_seq,
        "utc": "2026-08-13T17:33:47.820Z",
        "placement": placement,
        "confidence": 0.78,
        "status": "ok",
        "frame_seq": frame_seq,
        "capture_ts": 1786642427.0 + frame_seq,
        "published_ts": 1786642429.0 + frame_seq,
    }
    if prior is not None:
        line["prior"] = {"placement": prior, "flips": [], "residual": {}}
    if settled is not None:
        line["settled"] = {"placement": settled, "pending": []}
    if evidence is not None:
        line["evidence"] = evidence
    return line


def _digest(placement: str, **rivals: tuple[str, float, float]) -> list:
    """A top-2 digest agreeing with ``placement``, board order.

    Each keyword names a square whose read was close: ``d8=("q", 0.51, 0.47)``
    means the winner (whatever ``placement`` says) took 0.51 and ``q`` 0.47.
    Every other square is a confident, uncontested read.
    """
    grid = parse_placement(placement)
    squares = [
        [[grid[r][c], 0.99], ["empty" if grid[r][c] != "empty" else "P", 0.005]]
        for r in range(GRID)
        for c in range(GRID)
    ]
    for name, (runner_up, won, lost) in rivals.items():
        i = SQUARE_INDEX[name]
        squares[i] = [[squares[i][0][0], won], [runner_up, lost]]
    return squares


def _session(tmp_path, events: list[dict], frames: list[tuple[dict, bytes]] = ()) -> Recording:
    recorder = FrameRecorder(tmp_path, session_id="s", writer="test")
    for header, jpeg in frames:
        recorder.record(header, jpeg)
    for line in events:
        recorder.record_event(line)
    recorder.stop()
    return Recording(recorder.path)


def _photo_frames(n: int, placement: str = START_PLACEMENT) -> list[tuple[dict, bytes]]:
    """``n`` recorded frames of ``placement``, one second apart."""
    photo = render_board(placement)
    return [
        Frame(board_id="phone", seq=seq, capture_ts=1786642427.0 + seq, image=photo).encode()
        for seq in range(n)
    ]


def _truth(**over) -> SessionTruth:
    return SessionTruth(
        board_id=over.pop("board_id", "phone"),
        segments=over.pop(
            "segments",
            (TruthSegment(0, START_PLACEMENT), TruthSegment(100, E4)),
        ),
        guard_seqs=over.pop("guard_seqs", 5),
    )


#: The 08-13 failure in miniature: a persistently wrong raw board, a repair that
#: alternates between right and a two-square-wrong possibility, and a move at
#: seq 100 whose first two reads land inside the guard.
FLIPFLOP_EVENTS = [
    _event(0, TWO_KINGS, prior=START_PLACEMENT),
    _event(10, TWO_KINGS, prior=SWAPPED),
    _event(20, TWO_KINGS, prior=START_PLACEMENT),
    _event(30, TWO_KINGS, prior=SWAPPED),
    _event(100, E4, prior=E4),  # in the guard
    _event(103, E4, prior=E4),  # in the guard
    _event(110, E4, prior=E4),
    _event(120, E4, prior=E4),
]


# --------------------------------------------------------------------------- #
# Truth parsing
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Segment lookup and the guard window
# --------------------------------------------------------------------------- #


def test_segment_lookup_is_a_step_function_over_frame_seq():
    truth = _truth(
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
    truth = _truth(guard_seqs=5)  # segments start at 0 and 100
    # The session's opening board is not a move: nothing at the start is guarded.
    assert not truth.in_guard(0)
    assert not truth.in_guard(4)
    assert not truth.in_guard(99)
    assert truth.in_guard(100)
    assert truth.in_guard(104)
    assert not truth.in_guard(105)


def test_guard_of_zero_excludes_nothing():
    truth = _truth(guard_seqs=0)
    assert not truth.in_guard(100)


def test_truth_must_cover_the_stream_it_scores():
    _truth(segments=(TruthSegment(0, START_PLACEMENT),)).check_covers(7)  # 0 always ok
    _truth(segments=(TruthSegment(7, START_PLACEMENT),)).check_covers(7)  # or the first seq
    with pytest.raises(TruthError, match="have no truth"):
        _truth(segments=(TruthSegment(50, START_PLACEMENT),)).check_covers(7)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_read_events_skips_junk_and_keeps_order(tmp_path):
    recording = _session(tmp_path, [_event(0, START_PLACEMENT), _event(5, E4)])
    with (recording.path / "events.ndjson").open("a") as fh:
        fh.write("{truncated\n\n")
    events = read_events(recording)
    assert [e.frame_seq for e in events] == [0, 5]
    assert events[0].prior_placement is None


def test_three_streams_score_differently_on_the_flipflop_session(tmp_path):
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), _truth())

    assert report.events_total == 8
    assert report.events_scored == 6
    assert report.excluded_guard == 2
    assert report.excluded_untruthed == 0
    assert report.settled_derived is True  # no settled block on these events

    raw, prior, settled = (report.stream(s) for s in Stream)
    # Raw is wrong on d8 for the four start-position reads: 63/64 each.
    assert (raw.correct, raw.perfect) == (4 * 63 + 2 * 64, 2)
    # The repair fixes two of those outright and breaks two squares on the other
    # two — the same square total, a very different board total. This is exactly
    # why the streams are not summed together.
    assert (prior.correct, prior.perfect) == (4 * 64 + 2 * 62, 4)
    assert prior.correct == raw.correct
    # The debounce holds the majority board through the flip-flop AND through the
    # move, so every scored board is right.
    assert (settled.correct, settled.perfect) == (6 * 64, 6)
    assert settled.derived is True

    assert raw.confusions == {"q->k": 4}
    assert prior.confusions == {"q->k": 2, "k->q": 2}
    assert raw.errors == {"type": 4}


def test_per_segment_breakdown_splits_the_move(tmp_path):
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), _truth())
    before, after = report.stream(Stream.RAW).segments
    assert (before.from_seq, before.to_seq, before.events) == (0, 100, 4)
    assert (after.from_seq, after.to_seq, after.events) == (100, None, 2)
    assert after.perfect == 2  # the two post-guard reads of the new board


def test_recorded_settled_is_used_as_is(tmp_path):
    events = [
        _event(0, TWO_KINGS, prior=START_PLACEMENT, settled=START_PLACEMENT),
        _event(10, TWO_KINGS, prior=SWAPPED, settled=SWAPPED),
    ]
    report = score_session(_session(tmp_path, events), _truth())
    assert report.settled_derived is False
    settled = report.stream(Stream.SETTLED)
    assert settled.derived is False
    # SWAPPED taken verbatim — a derivation would have held START through it.
    assert settled.perfect == 1


def test_events_for_another_board_are_counted_not_scored(tmp_path):
    events = [
        _event(0, START_PLACEMENT, prior=START_PLACEMENT),
        _event(1, START_PLACEMENT, prior=START_PLACEMENT, board_id="other"),
    ]
    report = score_session(_session(tmp_path, events), _truth())
    assert (report.events_total, report.events_scored) == (2, 1)
    assert report.excluded_other_board == 1


def test_a_board_with_no_events_is_an_error_that_names_what_was_recorded(tmp_path):
    recording = _session(tmp_path, [_event(0, START_PLACEMENT, board_id="other")])
    with pytest.raises(TruthError, match="recorded \\['other'\\]"):
        score_session(recording, _truth())


def test_an_event_without_a_repair_falls_back_to_the_raw_board(tmp_path):
    report = score_session(_session(tmp_path, [_event(0, TWO_KINGS)]), _truth())
    assert report.prior_missing == 1
    assert report.stream(Stream.PRIOR).correct == report.stream(Stream.RAW).correct


def test_report_renders_and_serialises(tmp_path):
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), _truth())
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
        "prior_missing": 0,
    }
    assert [s["stream"] for s in doc["streams"]] == ["raw", "prior", "settled"]
    json.dumps(doc)  # the --json path must not choke on a Counter or a Path


def test_cli_prints_a_fillable_skeleton_when_truth_is_missing(tmp_path, capsys):
    recording = _session(tmp_path, FLIPFLOP_EVENTS, _photo_frames(1))
    assert main(["eval-session", str(recording.path)]) == 2
    out = capsys.readouterr()
    assert TRUTH_NAME in out.err
    skeleton = json.loads(out.out)
    assert skeleton["board_id"] == "phone"  # taken from the recording, not guessed
    assert skeleton["guard_seqs"] == DEFAULT_GUARD_SEQS


def test_cli_scores_a_session_and_writes_json(tmp_path, capsys):
    recording = _session(tmp_path / "rec", FLIPFLOP_EVENTS)
    (recording.path / TRUTH_NAME).write_text(json.dumps(_truth().to_dict()))
    report_json = tmp_path / "report.json"

    assert main(["eval-session", str(recording.path), "--json", str(report_json)]) == 0
    assert "per-square" in capsys.readouterr().out
    assert json.loads(report_json.read_text())["events"]["scored"] == 6


def test_cli_fails_on_a_session_with_nothing_left_to_score(tmp_path, capsys):
    # Every event lands inside the guard after the boundary at seq 100.
    events = [_event(100, E4, prior=E4), _event(101, E4, prior=E4)]
    recording = _session(tmp_path / "rec", events)
    truth = _truth(
        segments=(TruthSegment(0, START_PLACEMENT), TruthSegment(100, E4)), guard_seqs=50
    )
    (recording.path / TRUTH_NAME).write_text(json.dumps(truth.to_dict()))
    assert main(["eval-session", str(recording.path)]) == 1
    assert "no events left to score" in capsys.readouterr().out


def test_cli_reports_a_bad_truth_file_in_one_line(tmp_path, capsys):
    recording = _session(tmp_path / "rec", FLIPFLOP_EVENTS)
    (recording.path / TRUTH_NAME).write_text('{"format": 9}')
    assert main(["eval-session", str(recording.path)]) == 2
    assert "unsupported truth format" in capsys.readouterr().err


def test_cli_reports_a_directory_that_is_not_a_recording(tmp_path, capsys):
    assert main(["eval-session", str(tmp_path)]) == 2
    assert "cannot score" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Evidence — how close the estimator came on the squares it got wrong
# --------------------------------------------------------------------------- #


def test_evidence_separates_a_coin_flip_from_a_conviction(tmp_path):
    """The distinction the whole digest exists for: d8 read k over q by 0.04 (the
    crop carries the answer, the model nearly found it), while b6 hallucinated a
    rook with truth nowhere in the top 2 (the evidence is simply not there)."""
    events = [
        _event(
            0,
            TWO_KINGS,
            prior=TWO_KINGS,
            evidence=_digest(TWO_KINGS, d8=("q", 0.51, 0.47)),
        ),
        _event(
            10,
            TWO_KINGS,
            prior=TWO_KINGS,
            evidence=_digest(TWO_KINGS, d8=("n", 0.90, 0.05)),
        ),
    ]
    report = score_session(_session(tmp_path, events), _truth())
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
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), _truth())

    assert not report.evidence.recorded
    assert report.evidence.wrong == 0
    assert report.evidence.to_dict()["margin_median"] is None
    assert "predates the top-2 digest" in report.render()


def test_evidence_survives_a_malformed_digest(tmp_path):
    """A digest is diagnostic, never load-bearing: garbage there must not cost a
    session its score."""
    events = [
        _event(0, TWO_KINGS, prior=TWO_KINGS, evidence=[["k", 0.5]]),  # wrong shape
        _event(
            10, TWO_KINGS, prior=TWO_KINGS, evidence=_digest(TWO_KINGS, d8=("q", 0.6, 0.3))
        ),
    ]
    report = score_session(_session(tmp_path, events), _truth())

    assert report.events_scored == 2  # both scored
    assert report.evidence.events == 1  # only one carried a readable digest
    assert report.evidence.wrong == 1


def test_evidence_renders_its_margins(tmp_path):
    events = [
        _event(
            0, TWO_KINGS, prior=TWO_KINGS, evidence=_digest(TWO_KINGS, d8=("q", 0.51, 0.47))
        )
    ]
    text = score_session(_session(tmp_path, events), _truth()).render()

    assert "truth was the runner-up" in text
    assert "margin p(read)-p(truth)" in text
    assert "q->k" in text


# --------------------------------------------------------------------------- #
# Timing — the rates a placement score cannot show
# --------------------------------------------------------------------------- #


def test_timing_measures_input_rate_against_read_rate(tmp_path):
    """The 08-13 finding in miniature: frames arrive far faster than reads are
    published, and the report says so in one line instead of by hand."""
    frames = _photo_frames(8)
    recording = _session(tmp_path, FLIPFLOP_EVENTS, frames)
    t = score_session(recording, _truth()).timing

    assert t.frames == 8
    assert t.events == 8  # every event on this board, guard-excluded ones included
    assert t.frames_per_read == pytest.approx(1.0)
    assert t.latencies and all(lat == pytest.approx(2.0) for lat in t.latencies)
    assert t.to_dict()["latency"]["median"] == pytest.approx(2.0)


def test_timing_says_nothing_rather_than_zero_when_a_session_has_one_frame(tmp_path):
    recording = _session(tmp_path, [_event(0, START_PLACEMENT, prior=START_PLACEMENT)])
    t = score_session(recording, _truth()).timing

    assert t.fps == 0.0 and t.reads_per_s == 0.0  # no span to divide by
    assert "timing" in score_session(recording, _truth()).render()


def test_timing_counts_only_the_scored_board(tmp_path):
    frames = _photo_frames(2)
    events = [
        _event(0, START_PLACEMENT, prior=START_PLACEMENT),
        _event(1, START_PLACEMENT, prior=START_PLACEMENT, board_id="other"),
    ]
    t = score_session(_session(tmp_path, events, frames), _truth()).timing

    assert t.events == 1


# --------------------------------------------------------------------------- #
# The report artifact
# --------------------------------------------------------------------------- #


def test_scoring_always_leaves_an_artifact_beside_the_recording(tmp_path):
    """A number quoted from a run that left no artifact cannot be checked later —
    this project has already lost two headline figures that way."""
    recording = _session(tmp_path, FLIPFLOP_EVENTS)
    report = run(recording, truth=_truth())

    written = recording.path / REPORT_NAME
    assert report.written_to == written
    assert json.loads(written.read_text())["events"]["scored"] == 6
    assert str(written) in report.render()


def test_an_explicit_json_path_wins_over_the_default(tmp_path):
    recording = _session(tmp_path / "rec", FLIPFLOP_EVENTS)
    out = tmp_path / "elsewhere.json"
    report = run(recording, truth=_truth(), json_out=str(out))

    assert report.written_to == out
    assert out.exists()
    assert not (recording.path / REPORT_NAME).exists()


def test_an_unwritable_report_does_not_cost_the_score(tmp_path):
    recording = _session(tmp_path / "rec", FLIPFLOP_EVENTS)
    report = run(
        recording, truth=_truth(), json_out=str(tmp_path / "no" / "such" / "dir.json")
    )

    assert report.written_to is None
    assert report.events_scored == 6


def test_reads_in_the_truths_frame_pass_the_frame_check(tmp_path):
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), _truth())
    assert report.frame.by_rotation == (6, 0, 0, 0)
    assert report.frame.verdict == 0
    assert report.to_dict()["truth_frame"]["verdict_deg"] == 0
    assert "6/6 raw reads agree with the truth unturned" in report.render()


def test_truth_written_a_half_turn_off_is_flagged(tmp_path):
    from board_reader_inference.labels import rotate_placement

    turned = _truth(
        segments=(
            TruthSegment(0, rotate_placement(START_PLACEMENT, 2)),
            TruthSegment(100, rotate_placement(E4, 2)),
        )
    )
    report = score_session(_session(tmp_path, FLIPFLOP_EVENTS), turned)
    assert report.frame.verdict == 2
    assert "WARNING 6/6 raw reads agree with the truth best turned 180°" in report.render()
