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
    START_PLACEMENT,
    dist_from_placement,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg

from board_reader_inference.evaluate import (
    describe_estimator,
    error_kind,
    evaluate,
    has_bank,
    reliability,
    run,
    score_board,
)
from board_reader_inference.labels import LabeledBoard, save_label

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
        corners=CORNERS,
        placement=placement,
        source="confirmed",
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


class FakeEstimator:
    """Answers with a canned distribution per board id — no model, no weights."""

    crop_mode = "photo"
    headroom = 1.6
    side_pad = 0.15
    square_size = 224

    def __init__(self, dists: dict[str, np.ndarray]) -> None:
        self.dists = dists
        self.seen: list[str] = []

    def estimate_photo(self, frame, corners):
        from board_reader_core.protocol import PositionEvent

        self.seen.append(frame.board_id)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc="1970-01-01T00:00:00Z",
            square_dist=self.dists[frame.board_id],
        )


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
    report = evaluate(FakeEstimator(dists), boards)

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
    report = evaluate(FakeEstimator(dists), [_board("one_off")])
    out = json.loads(json.dumps(report.to_dict()))
    assert out["boards"] == 1 and out["perfect_boards"] == 0
    assert out["correct_squares"] == 63
    assert out["per_board"][0]["errors"][0]["square"] == "g1"


def test_empty_input_reports_rather_than_crashes():
    report = evaluate(FakeEstimator({}), [])
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
    report = evaluate(FakeEstimator({"broken": broken}), [_board("broken")])
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
    report = evaluate(FakeEstimator(dists), [_board(k) for k in dists])
    rendered = report.render()
    assert "board confidence:" in rendered
    assert "perfect" in rendered and "imperfect" in rendered


# ---- provenance and the contamination warning --------------------------------


def test_describe_estimator_names_the_model_and_the_bank():
    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import TrainedEstimator

    def logits(batch):
        return np.zeros((batch.shape[0], NUM_CLASSES), dtype=np.float32)

    trained = TrainedEstimator(logits, headroom=1.6, side_pad=0.15)
    trained.model_path = "models/d16/model.onnx"
    ensemble = EnsembleEstimator(trained, FakeEstimator({}), weight=0.3)
    described = describe_estimator(ensemble)
    assert "models/d16/model.onnx" in described
    assert "x0.30" in described and "x0.70" in described


def test_bank_backed_stacks_are_flagged_as_contaminated():
    """A bank scores near-perfectly on the boards it was built from; a report that
    does not say so is the exact mistake this harness exists to stop repeating."""

    class WithBank(FakeEstimator):
        class classifier:  # noqa: N801 — stands in for an ExemplarClassifier
            exemplars = np.zeros((4096, 8), dtype=np.float32)
            backbone = "dinov2"

    assert has_bank(WithBank({}))
    assert not has_bank(FakeEstimator({}))
    report = evaluate(WithBank({"b0": dist_from_placement(MIDGAME)}), [_board()])
    assert report.warnings and "NOT held out" in report.warnings[0]
    assert "warning:" in report.render()


def test_evaluation_refuses_an_estimator_with_no_photo_path():
    from board_reader_inference.estimator import StubEstimator

    with pytest.raises(ValueError, match="no photo path"):
        evaluate(StubEstimator(START_PLACEMENT), [_board()])


# ---- end to end over real files ----------------------------------------------


def test_run_loads_labels_from_disk_and_reads_the_photos(tmp_path):
    """The whole path: a confirmed label on disk -> a decoded photo -> a score."""
    stem = "20260806-210118-0000-estimate"
    (tmp_path / f"{stem}.jpg").write_bytes(encode_jpeg(render_board(MIDGAME)))
    save_label(
        tmp_path, stem,
        placement=MIDGAME, corners="0,0,100,0,100,100,0,100",
        predicted=MIDGAME, corrections=1, autorotated=0,
    )
    est = FakeEstimator({stem: _dist_with(MIDGAME, {"g1": ("B", 0.9)})})
    report = run([str(tmp_path)], estimator=est, json_out=str(tmp_path / "out.json"))

    assert est.seen == [stem]
    assert len(report.boards) == 1 and report.correct == 63
    assert json.loads((tmp_path / "out.json").read_text())["correct_squares"] == 63


def test_run_undoes_autorotation_so_truth_survives_the_orientation_guess(tmp_path):
    """A read confirmed after a 180-degree autorotate is stored in the corner frame;
    re-reading it in that frame must score perfect."""
    from board_reader_inference.labels import rotate_placement

    stem = "20260806-210118-0000-estimate"
    (tmp_path / f"{stem}.jpg").write_bytes(encode_jpeg(render_board(MIDGAME)))
    shown = rotate_placement(MIDGAME, 2)
    save_label(
        tmp_path, stem, placement=shown, corners="0,0,100,0,100,100,0,100",
        predicted=shown, corrections=0, autorotated=180,
    )
    est = FakeEstimator({stem: dist_from_placement(MIDGAME)})
    report = run([str(tmp_path)], estimator=est)
    assert report.perfect_boards == 1


def test_run_on_a_directory_with_nothing_labeled_is_not_an_error(tmp_path):
    report = run([str(tmp_path)], estimator=FakeEstimator({}))
    assert report.boards == []
