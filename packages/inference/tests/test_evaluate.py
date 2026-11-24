"""Tests for the whole-board evaluation harness (:mod:`board_reader_inference.evaluate`).

The harness's whole point is to report the *unflattering* number, so these tests
pin the arithmetic that makes it unflattering: one wrong square costs a whole
board, error kinds are told apart, and a confident wrong answer shows up as a
calibration gap rather than being averaged away.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASS_INDEX,
    GRID,
    NUM_CLASSES,
    dist_from_placement,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg

from board_reader_inference.evaluate import (
    error_kind,
    evaluate,
    reliability,
    run,
    score_board,
)
from board_reader_inference.labels import LabeledBoard, save_label
from board_reader_inference.recognizer import StubRecognizer

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"
CORNERS = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


# The fakes never look at pixels, but read_board decodes the photo like the real
# thing does — so every labeled board needs a file that actually exists.
_PHOTO: object = "unused.jpg"


@pytest.fixture(autouse=True, scope="module")
def _photo_on_disk(tmp_path_factory):
    global _PHOTO
    path = tmp_path_factory.mktemp("photos") / "board.jpg"
    path.write_bytes(encode_jpeg(render_board(MIDGAME)))
    _PHOTO = path
    yield


def _board(board_id: str = "b0", placement: str = MIDGAME, image=None) -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id,
        image=image if image is not None else _PHOTO,
        frame="chess",
        corners=None,
        placement=placement,
        source="confirmed",
    )


def _corner_frame_board(board_id: str) -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id, image=_PHOTO, frame="corners", corners=CORNERS,
        placement=MIDGAME, source="confirmed",
    )


def _dist_with(placement: str, overrides: dict[str, tuple[str, float]]) -> np.ndarray:
    """A peaked distribution for ``placement``, then specific squares overwritten.

    ``overrides`` maps a square name to ``(class, confidence)`` — the lever these
    tests use to plant a *confidently wrong* square.
    """
    dist = dist_from_placement(placement, confidence=0.9).copy()
    for square, (cls, conf) in overrides.items():
        c, r = "abcdefgh".index(square[0]), GRID - int(square[1])
        dist[r, c] = (1.0 - conf) / (NUM_CLASSES - 1)
        dist[r, c, CLASS_INDEX[cls]] = conf
    return dist


class FakeRecognizer:
    """Answers with the canned distributions in order, one per read — no model.

    A recognizer is handed pixels, not a board id, so the fake hands out its
    distributions in the order the boards are read (``evaluate`` reads them in
    the order given).
    """

    name = "fake"
    frame = "chess"
    model_path = None

    def __init__(self, dists: dict[str, np.ndarray]) -> None:
        self._dists = iter(dists.values())
        self.reads = 0

    def read(self, image: np.ndarray) -> np.ndarray:
        self.reads += 1
        return next(self._dists)


# ---- error taxonomy ----------------------------------------------------------


@pytest.mark.parametrize(
    ("truth", "predicted", "kind"),
    [
        ("empty", "P", "hallucination"),
        ("P", "empty", "missed"),
        ("P", "p", "colour"),
        ("k", "K", "colour"),
        ("Q", "K", "type"),
        ("r", "p", "type"),
    ],
)
def test_error_kind_separates_occupancy_from_recognition(truth, predicted, kind):
    assert error_kind(truth, predicted) == kind


# ---- scoring -----------------------------------------------------------------


def test_a_flawless_read_is_a_perfect_board():
    result = score_board(_board(), dist_from_placement(MIDGAME))
    assert result.perfect and result.correct == 64 and not result.errors
    assert result.predicted == MIDGAME


def test_one_wrong_square_costs_the_whole_board():
    """The headline metric's entire reason for existing."""
    result = score_board(_board(), _dist_with(MIDGAME, {"g1": ("B", 0.9)}))
    assert not result.perfect
    assert result.correct == 63
    assert [str(e) for e in result.errors] == ["g1 K->B"]
    assert result.errors[0].kind == "type"


def test_errors_carry_their_square_and_confidence():
    # d2 is empty in MIDGAME (rank 2 is PPP2PPP) — a pawn there is a hallucination.
    result = score_board(_board(), _dist_with(MIDGAME, {"d2": ("P", 0.97)}))
    (err,) = result.errors
    assert err.square == "d2" and err.truth == "empty" and err.predicted == "P"
    assert err.kind == "hallucination"
    assert err.confidence == pytest.approx(0.97, abs=1e-6)


def test_square_naming_matches_the_placement_frame():
    """Row 0 is rank 8 and column 0 is file a — an inverted axis would rename
    every error and make reports point at the wrong squares."""
    truth = parse_placement(MIDGAME)
    assert truth[0][0] == "r"  # a8, first rank listed / first file
    assert truth[GRID - 1][0] == "R"  # a1, last rank listed / first file
    assert truth[GRID - 1][GRID - 2] == "K"  # g1
    result = score_board(_board(), _dist_with(MIDGAME, {"a8": ("empty", 0.9)}))
    assert str(result.errors[0]) == "a8 r->empty"


# ---- report ------------------------------------------------------------------


def test_report_headline_counts_boards_not_squares():
    dists = {
        "clean": dist_from_placement(MIDGAME),
        "one_off": _dist_with(MIDGAME, {"g1": ("B", 0.9)}),
        "two_off": _dist_with(MIDGAME, {"g1": ("B", 0.9), "d2": ("P", 0.9)}),
    }
    boards = [_board(k) for k in dists]
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
    dists = {"one_off": _dist_with(MIDGAME, {"g1": ("B", 0.9)})}
    report = evaluate(FakeRecognizer(dists), [_board("one_off")])
    out = json.loads(json.dumps(report.to_dict()))
    assert out["boards"] == 1 and out["perfect_boards"] == 0
    assert out["correct_squares"] == 63
    assert out["per_board"][0]["errors"][0]["square"] == "g1"


def test_empty_input_reports_rather_than_crashes():
    report = evaluate(FakeRecognizer({}), [])
    assert report.boards == [] and report.perfect_boards == 0
    assert "no labeled boards" in report.render()


# ---- confidence calibration --------------------------------------------------


def test_reliability_measures_the_gap_between_claimed_and_actual():
    # 10 squares claiming 0.95, only half of them right -> a 0.45 gap.
    conf = np.full(10, 0.95)
    hits = np.array([True] * 5 + [False] * 5)
    bins, ece = reliability(conf, hits)
    assert len(bins) == 1
    (b,) = bins
    assert b.count == 10
    assert b.mean_confidence == pytest.approx(0.95)
    assert b.accuracy == pytest.approx(0.5)
    assert b.gap == pytest.approx(0.45)
    assert ece == pytest.approx(0.45)


def test_reliability_counts_confidence_of_exactly_one():
    """A one-hot distribution must land in the top bin, not fall off the end."""
    bins, _ = reliability(np.array([1.0, 1.0]), np.array([True, False]))
    assert sum(b.count for b in bins) == 2


def test_perfectly_calibrated_input_has_near_zero_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=20000)
    hits = rng.uniform(size=conf.size) < conf
    _, ece = reliability(conf, hits)
    assert ece < 0.02


def test_a_confidently_wrong_board_shows_up_as_a_gap_not_an_average():
    """The 2026-08-06 field failure: two white kings, no black king, at 0.97."""
    broken = _dist_with(MIDGAME, {"e8": ("K", 0.97), "g1": ("K", 0.97)})
    report = evaluate(FakeRecognizer({"broken": broken}), [_board("broken")])
    assert not report.boards[0].perfect
    assert report.boards[0].confidence > 0.85  # the board still looks confident
    top = [b for b in report.bins if b.lo >= 0.9]
    assert top and any(b.gap > 0 for b in top)
    assert "confidence reliability" in report.render()


def test_report_separates_confidence_of_right_and_wrong_boards():
    dists = {
        "clean": dist_from_placement(MIDGAME, confidence=0.9),
        "broken": _dist_with(MIDGAME, {"e8": ("K", 0.97)}),
    }
    report = evaluate(FakeRecognizer(dists), [_board(k) for k in dists])
    rendered = report.render()
    assert "board confidence:" in rendered
    assert "perfect" in rendered and "imperfect" in rendered


# ---- the opt-in hard-constraint prior ----------------------------------------


def _dist_with_runner_up(
    placement: str, square: str, top: tuple[str, float], second: tuple[str, float]
) -> np.ndarray:
    """Plant a square whose argmax is ``top`` and whose second choice is ``second``.

    The decoder relabels to the *cheapest alternative*, so a synthetic board with a
    flat tail would always decode to ``empty``. Giving the runner-up real mass is
    what makes these tests exercise the decision the real distribution makes.
    """
    dist = dist_from_placement(placement, confidence=0.95).copy()
    c, r = "abcdefgh".index(square[0]), GRID - int(square[1])
    rest = (1.0 - top[1] - second[1]) / (NUM_CLASSES - 2)
    dist[r, c] = rest
    dist[r, c, CLASS_INDEX[top[0]]] = top[1]
    dist[r, c, CLASS_INDEX[second[0]]] = second[1]
    return dist


def test_the_prior_is_on_by_default_because_it_is_what_serves():
    """The harness scores what /estimate serves, and the decode now serves."""
    dists = {"two_kings": _dist_with(MIDGAME, {"e8": ("K", 0.6)})}
    report = evaluate(FakeRecognizer(dists), [_board("two_kings")])
    assert report.prior != []
    assert "hard-constraint prior" in report.render()


def test_the_prior_can_be_isolated_away():
    """``--no-prior`` still gives the raw estimator on its own."""
    dists = {"two_kings": _dist_with(MIDGAME, {"e8": ("K", 0.6)})}
    report = evaluate(FakeRecognizer(dists), [_board("two_kings")], prior=False)
    assert report.prior == []
    assert "hard-constraint prior" not in report.render()


def test_the_prior_fixes_a_two_king_board_and_says_which_square():
    """e8 truly holds a black king; the estimator calls it white at a thin margin."""
    dist = _dist_with_runner_up(MIDGAME, "e8", ("K", 0.50), ("k", 0.45))
    report = evaluate(FakeRecognizer({"b": dist}), [_board("b")], prior=True)

    assert report.perfect_boards == 0 and report.correct == 63
    (p,) = report.prior
    assert p.decode.placement == MIDGAME and p.perfect_after
    assert [str(f) for f in p.decode.flips] == ["e8 K->k"]
    assert [str(c) for c in p.fixed] == ["e8 K->k (truth k)"]
    assert p.broken == ()
    assert report.prior_fixed == 1 and report.prior_broken == 0
    assert report.prior_perfect_boards == 1 and report.prior_correct == 64


def test_the_prior_can_break_a_square_it_was_right_about():
    """The failure mode that must stay visible: a hallucinated second king that is
    *more* confident than the real one drags the real one out instead."""
    dist = _dist_with(MIDGAME, {"e8": ("k", 0.40), "a5": ("k", 0.90)})
    report = evaluate(FakeRecognizer({"b": dist}), [_board("b")], prior=True)

    (p,) = report.prior
    assert [f.square for f in p.decode.flips] == ["e8"]
    assert [c.square for c in p.broken] == ["e8"]
    assert p.fixed == ()
    assert report.prior_broken == 1 and report.prior_correct == report.correct - 1


def test_a_board_with_nothing_to_repair_is_reported_as_untouched():
    report = evaluate(
        FakeRecognizer({"b": dist_from_placement(MIDGAME)}), [_board("b")], prior=True
    )
    (p,) = report.prior
    assert p.decode.flips == () and p.changes == ()
    assert p.correct_before == p.correct_after == 64
    rendered = report.render()
    assert "boards touched   0/1" in rendered
    assert "nothing to decode" in rendered


def test_the_prior_block_prints_both_scores_and_every_flip():
    dist = _dist_with_runner_up(MIDGAME, "e8", ("K", 0.50), ("k", 0.45))
    report = evaluate(FakeRecognizer({"b": dist}), [_board("b")], prior=True)
    rendered = report.render()
    assert "hard-constraint prior" in rendered
    assert "PERFECT BOARDS   0/1 -> 1/1" in rendered
    assert "per-square       63/64 -> 64/64   (fixed 1 · broken 0)" in rendered
    assert "e8 K->k" in rendered and "kings" in rendered and "FIXED" in rendered
    # The headline block above it is untouched by the prior.
    assert "PERFECT BOARDS   0/1" in rendered


def test_the_prior_survives_json_round_tripping():
    dist = _dist_with_runner_up(MIDGAME, "e8", ("K", 0.50), ("k", 0.45))
    report = evaluate(FakeRecognizer({"b": dist}), [_board("b")], prior=True)
    out = json.loads(json.dumps(report.to_dict()))
    assert out["prior"]["fixed"] == 1 and out["prior"]["broken"] == 0
    assert out["prior"]["per_board"][0]["flips"][0]["square"] == "e8"
    assert out["prior"]["per_board"][0]["changes"][0]["verdict"] == "fixed"
    assert out["correct_squares"] == 63  # the primary number never moves


def test_the_eval_cli_defaults_the_prior_on():
    from board_reader_inference.cli import build_parser

    assert build_parser().parse_args(["eval"]).no_prior is False
    assert build_parser().parse_args(["eval", "--no-prior", "data/uploads"]).no_prior is True


def test_the_prior_never_rewrites_what_the_estimator_serves():
    """DESIGN D1: the decode is a value beside the read, not a correction to it."""
    dist = _dist_with_runner_up(MIDGAME, "e8", ("K", 0.50), ("k", 0.45))
    before = dist.copy()
    report = evaluate(FakeRecognizer({"b": dist}), [_board("b")], prior=True)
    assert np.array_equal(dist, before)
    assert report.boards[0].predicted != report.prior[0].decode.placement


# ---- provenance and the contamination warning --------------------------------


def test_run_loads_labels_from_disk_and_reads_the_photos(tmp_path):
    """The whole path: a confirmed label on disk -> a decoded photo -> a score."""
    stem = "20260806-210118-0000-estimate"
    (tmp_path / f"{stem}.jpg").write_bytes(encode_jpeg(render_board(MIDGAME)))
    save_label(
        tmp_path, stem,
        frame="chess", placement=MIDGAME, corners=None,
        predicted=MIDGAME, corrections=1, autorotated=0,
    )
    est = FakeRecognizer({stem: _dist_with(MIDGAME, {"g1": ("B", 0.9)})})
    report = run([str(tmp_path)], recognizer=est, json_out=str(tmp_path / "out.json"))

    assert est.reads == 1
    assert len(report.boards) == 1 and report.correct == 63
    assert json.loads((tmp_path / "out.json").read_text())["correct_squares"] == 63


def test_run_on_a_directory_with_nothing_labeled_is_not_an_error(tmp_path):
    report = run([str(tmp_path)], recognizer=FakeRecognizer({}))
    assert report.boards == []


def test_ranking_auc_is_the_signal_the_confidence_gap_hides():
    """Temperature scaling compresses the perfect/imperfect confidence gap by
    construction, so the gap alone would report a harmless change as a loss. AUC
    is the rank-based number that actually decides whether a consumer can
    threshold on confidence."""
    from board_reader_inference.evaluate import _ranking_auc

    assert _ranking_auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert _ranking_auc([0.2, 0.1], [0.9, 0.8]) == 0.0
    assert _ranking_auc([0.5], [0.5]) == 0.5  # ties are half a win
    # A monotone squeeze toward 1.0 keeps the ordering, so AUC is unmoved even
    # though the means have nearly converged.
    assert _ranking_auc([0.99, 0.98], [0.97, 0.96]) == 1.0
    assert _ranking_auc([], [0.5]) is None
    assert _ranking_auc([0.5], []) is None


def test_temperature_report_is_absent_unless_asked_for():
    """It costs a leave-one-out fit per board; the default eval must not pay it."""
    from board_reader_inference.evaluate import Report

    assert Report(boards=[], estimator="x").temperature is None


# --------------------------------------------------------------------------- #
# Recognizers and the frame diagnostic (DESIGN D10)
# --------------------------------------------------------------------------- #


def _chess_board(board_id: str = "c0") -> LabeledBoard:
    return LabeledBoard(
        board_id=board_id, image=_PHOTO, frame="chess", corners=None,
        placement=MIDGAME, source="curated",
    )


def test_a_recognizer_reads_chess_labels_whole_and_skips_corner_frame_ones():
    report = evaluate(
        StubRecognizer(MIDGAME), [_chess_board(), _corner_frame_board("legacy")], prior=False
    )
    assert [b.board_id for b in report.boards] == ["c0"]
    assert report.perfect_boards == 1 and report.skipped == ["legacy"]
    assert any("corner-frame label(s) skipped" in w for w in report.warnings)
    assert report.estimator == "StubRecognizer"
    assert report.to_dict()["rotated_boards"] == []


def test_a_turned_read_is_flagged_by_the_frame_diagnostic():
    from board_reader_inference.labels import rotate_placement

    report = evaluate(StubRecognizer(rotate_placement(MIDGAME, 2)), [_chess_board()], prior=False)
    assert report.boards[0].rotation == 2
    assert report.to_dict()["rotated_boards"] == [{"board_id": "c0", "rotation": 180}]
    assert "TURNED: c0 180°" in report.render()
