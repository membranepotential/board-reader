"""Tests for the stub estimator (DESIGN.md D1 / L2)."""

from __future__ import annotations

from board_reader_core.protocol import EMPTY_PLACEMENT, START_PLACEMENT, iso_utc
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame
from board_reader_inference.centroid import build_centroids
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.estimator import (
    CentroidEstimator,
    Estimator,
    StubEstimator,
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
