"""Tests for the stub and centroid estimators, and the read seam they share\n(DESIGN.md D1 / L2).\n"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    EMPTY_PLACEMENT,
    NUM_CLASSES,
    START_PLACEMENT,
    dist_from_placement,
    iso_utc,
    placement_from_dist,
    square_confidences,
    square_evidence,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.centroid import build_centroids
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.estimator import (
    ATTRACTOR_MARGIN,
    CentroidEstimator,
    Estimator,
    StubEstimator,
    demote_attractor,
    read_event,
)
from board_reader_inference.squares import crop_squares


def _frame(seq=0, ts=0.0, stability=1.0, clock="none", image=None):
    return Frame(
        board_id="A1", seq=seq, capture_ts=ts,
        image=render_board() if image is None else image,
        stability=stability, clock=clock,
    )


def _centroid_estimator(placement=START_PLACEMENT):
    emb = LiteEmbedder()
    crops = crop_squares(render_board(placement, square_px=48), inset=0.1)
    classifier = build_centroids(emb, crops, placement)
    return CentroidEstimator(emb, classifier, inset=0.1)


def test_stub_estimator_satisfies_protocol():
    assert isinstance(StubEstimator(), Estimator)


def test_stub_emits_configured_placement_as_valid_event():
    ev = StubEstimator(START_PLACEMENT).estimate(_frame())
    ev.validate()
    assert ev.placement == START_PLACEMENT
    assert ev.status == "ok"


def test_stub_passes_through_frame_metadata():
    ev = StubEstimator().estimate(_frame(seq=11, ts=0.0, stability=0.25, clock="white"))
    assert ev.seq == 11
    assert ev.utc == iso_utc(0.0)
    assert ev.stability == 0.25
    assert ev.clock.running == "white"


def test_stub_confidence_is_configurable():
    ev = StubEstimator(EMPTY_PLACEMENT, confidence=0.6).estimate(_frame())
    assert 0.59 <= ev.confidence <= 0.61
    assert ev.placement == EMPTY_PLACEMENT


def test_stub_is_stateless_same_output_each_frame():
    est = StubEstimator()
    a = est.estimate(_frame(seq=0))
    b = est.estimate(_frame(seq=1))
    assert a.placement == b.placement  # ignores pixels, no game state


# --- CentroidEstimator (M1 recognition) ---------------------------------------


def test_centroid_estimator_satisfies_protocol():
    assert isinstance(_centroid_estimator(), Estimator)


def test_centroid_estimator_reads_the_rectified_board():
    est = _centroid_estimator(START_PLACEMENT)
    board = render_board(START_PLACEMENT, square_px=48)
    ev = est.estimate(_frame(seq=7, ts=1.5, image=board)).validate()
    assert ev.placement == START_PLACEMENT  # self-consistent on the calibration frame
    assert ev.seq == 7 and ev.utc == iso_utc(1.5)
    assert ev.confidence > 0.5


def test_centroid_estimator_tracks_pixels_not_state():
    # Unlike the stub, output follows the input frame's pixels.
    est = _centroid_estimator(START_PLACEMENT)
    start = est.estimate(_frame(image=render_board(START_PLACEMENT, square_px=48)))
    empty = est.estimate(_frame(image=render_board(EMPTY_PLACEMENT, square_px=48)))
    assert start.placement == START_PLACEMENT
    assert empty.placement == EMPTY_PLACEMENT


# --- the black-bishop demotion (docs/reports/2026-08-27-evidence-rerank.md) ----
#
# The shipped calibration: when the BLACK bishop wins a square by less than
# ATTRACTOR_MARGIN, the runner-up is the right answer. These pin the rule itself;
# test_ensemble.py pins that a blend applies it once, to the blend.


def _square(**probs: float) -> np.ndarray:
    """One square's distribution: named classes at their probability, rest flat."""
    row = np.full(NUM_CLASSES, (1.0 - sum(probs.values())) / (NUM_CLASSES - len(probs)))
    for cls, p in probs.items():
        row[CLASS_INDEX[cls]] = p
    return row.astype(np.float32)


def _board(**squares: np.ndarray) -> np.ndarray:
    """An 8x8x13 board of confident empties, with named squares overridden."""
    dist = np.asarray(dist_from_placement(EMPTY_PLACEMENT), dtype=np.float32)
    for name, row in squares.items():
        r, c = 8 - int(name[1]), ord(name[0]) - ord("a")
        dist[r, c] = row
    return dist


def test_a_narrow_black_bishop_win_goes_to_the_runner_up():
    dist = _board(c8=_square(b=0.40, q=0.28))
    out = demote_attractor(dist)

    assert CLASSES[int(out[0, 2].argmax())] == "q"
    # A permutation of the two: the row still sums to one and the square's max —
    # hence the event's confidence — is untouched.
    assert out[0, 2, CLASS_INDEX["q"]] == pytest.approx(0.40)
    assert out[0, 2, CLASS_INDEX["b"]] == pytest.approx(0.28)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(out.max(axis=-1), np.asarray(dist).max(axis=-1), atol=1e-7)


def test_a_clear_black_bishop_win_is_left_alone():
    """The threshold has a real gap under it — the lowest margin at which a top-1
    'b' was actually right is 0.210 — so what matters is that a margin ABOVE 0.20
    survives and one below does not. The boundary itself is not pinned: at
    exactly 0.20 the comparison is float32 noise, and no distribution lands there.
    """
    for margin in (ATTRACTOR_MARGIN + 0.01, 0.25, 0.6):
        dist = _board(c8=_square(b=0.30 + margin, q=0.30))
        assert CLASSES[int(demote_attractor(dist)[0, 2].argmax())] == "b"
    below = _board(c8=_square(b=0.30 + ATTRACTOR_MARGIN - 0.01, q=0.30))
    assert CLASSES[int(demote_attractor(below)[0, 2].argmax())] == "q"


def test_the_white_bishop_is_not_an_attractor():
    """Lowercase only: top-1 'B' had 1 fixable square in 8512 (report §6)."""
    dist = _board(c1=_square(B=0.40, Q=0.28))
    assert CLASSES[int(demote_attractor(dist)[7, 2].argmax())] == "B"


@pytest.mark.parametrize("winner", [c for c in CLASSES if c != "b"])
def test_no_other_class_is_demoted_however_narrow_its_win(winner):
    runner_up = "b" if winner != "b" else "q"
    dist = _board(c8=_square(**{winner: 0.40, runner_up: 0.39}))
    assert CLASSES[int(demote_attractor(dist)[0, 2].argmax())] == winner


def test_the_demotion_is_idempotent():
    """After a swap the attractor is no longer top-1, so a second pass is a no-op.
    Nothing should apply this twice — but if something ever does, it cannot
    ping-pong a square between two classes."""
    dist = _board(c8=_square(b=0.40, q=0.28), f8=_square(b=0.31, r=0.30))
    once = demote_attractor(dist)
    np.testing.assert_array_equal(demote_attractor(once), once)


def test_a_demoted_read_is_demoted_everywhere_the_event_is_derived():
    """placement, evidence and per-square confidence are all derived from the one
    distribution read_event calibrated, so none of them can disagree with it."""
    dist = _board(c8=_square(b=0.40, q=0.28))
    event = read_event(_frame(seq=4), dist, orientation="a1_bottom_left").validate()

    assert event.placement.split("/")[0] == "2q5"  # c8, demoted, on the board
    assert placement_from_dist(event.square_dist) == event.placement
    assert square_evidence(event.square_dist)[2] == [("q", 0.4), ("b", 0.28)]
    assert square_confidences(event.square_dist)[0, 2] == pytest.approx(0.40)
