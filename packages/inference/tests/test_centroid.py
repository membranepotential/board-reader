"""Tests for the centroid classifier + few-shot calibration (WP-E2/B3)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import (
    CLASS_INDEX,
    EMPTY_PLACEMENT,
    NUM_CLASSES,
    START_PLACEMENT,
    parse_placement,
    placement_from_dist,
)
from board_reader_core.sim import render_board
from board_reader_inference.centroid import CentroidClassifier, build_centroids
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.squares import crop_squares

EMBEDDER = LiteEmbedder()


def _calibrate(placement: str) -> CentroidClassifier:
    crops = crop_squares(render_board(placement, square_px=48), inset=0.1)
    return build_centroids(EMBEDDER, crops, placement)


def _square_accuracy(classifier: CentroidClassifier, placement: str) -> float:
    crops = crop_squares(render_board(placement, square_px=48), inset=0.1)
    dist = classifier.classify_board(EMBEDDER.embed(crops))
    truth = parse_placement(placement)
    pred = parse_placement(placement_from_dist(dist))
    correct = sum(
        pred[r][c] == truth[r][c] for r in range(8) for c in range(8)
    )
    return correct / 64.0


def test_distributions_are_valid_probabilities():
    classifier = _calibrate(START_PLACEMENT)
    crops = crop_squares(render_board(START_PLACEMENT, square_px=48), inset=0.1)
    dist = classifier.classify_board(EMBEDDER.embed(crops))
    assert dist.shape == (8, 8, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)
    assert dist.min() >= 0.0 and dist.max() <= 1.0


def test_self_consistency_recovers_the_calibration_position():
    # Calibrating and classifying the same frame must reproduce the labels exactly.
    classifier = _calibrate(START_PLACEMENT)
    assert _square_accuracy(classifier, START_PLACEMENT) == 1.0


def test_generalises_to_a_different_position_with_seen_classes():
    classifier = _calibrate(START_PLACEMENT)  # start has all 13 classes
    midgame = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    assert _square_accuracy(classifier, midgame) > 0.9


def test_absent_classes_get_near_zero_probability():
    classifier = _calibrate(EMPTY_PLACEMENT)  # only 'empty' present
    crops = crop_squares(render_board(EMPTY_PLACEMENT, square_px=48), inset=0.1)
    dist = classifier.classify_board(EMBEDDER.embed(crops))
    # Every square should read 'empty'; piece-class mass is ~0.
    assert placement_from_dist(dist) == EMPTY_PLACEMENT
    assert dist[..., CLASS_INDEX["Q"]].max() < 1e-6


def test_save_load_round_trips(tmp_path):
    classifier = _calibrate(START_PLACEMENT)
    path = tmp_path / "centroids.npz"
    classifier.save(path)
    loaded = CentroidClassifier.load(path)
    np.testing.assert_array_equal(
        np.nan_to_num(classifier.centroids), np.nan_to_num(loaded.centroids)
    )
    assert loaded.temperature == classifier.temperature
    assert _square_accuracy(loaded, START_PLACEMENT) == 1.0


def test_build_centroids_rejects_wrong_crop_count():
    try:
        build_centroids(EMBEDDER, [render_board()], START_PLACEMENT)
    except ValueError:
        return
    raise AssertionError("expected ValueError for != 64 crops")
