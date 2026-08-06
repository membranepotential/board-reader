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
from board_reader_inference.centroid import (
    CentroidClassifier,
    ExemplarClassifier,
    build_centroids,
    build_exemplars,
    load_classifier,
)
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.squares import crop_squares, crop_squares_photo, full_frame_corners

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


def test_augmented_calibration_stays_self_consistent():
    crops = crop_squares(render_board(START_PLACEMENT, square_px=48), inset=0.1)
    classifier = build_centroids(EMBEDDER, crops, START_PLACEMENT, augment=True)
    assert _square_accuracy(classifier, START_PLACEMENT) == 1.0


def test_crop_mode_round_trips_and_is_validated(tmp_path):
    crops = crop_squares(render_board(START_PLACEMENT, square_px=48), inset=0.1)
    classifier = build_centroids(
        EMBEDDER, crops, START_PLACEMENT, crop_mode="photo"
    )
    path = tmp_path / "c.npz"
    classifier.save(path)
    assert CentroidClassifier.load(path).crop_mode == "photo"
    try:
        build_centroids(EMBEDDER, crops, START_PLACEMENT, crop_mode="sideways")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown crop_mode")


# --- ExemplarClassifier (kNN bank + transductive centering) ---------------------


def _photo_bank(placement: str, **kw) -> ExemplarClassifier:
    """Build a bank from flat (zero-headroom) photo boxes so LiteEmbedder can read it."""
    photo = render_board(placement, square_px=48)
    crops = crop_squares_photo(
        photo, full_frame_corners(photo), headroom=0.0, side_pad=0.0, size=64
    )
    return build_exemplars(EMBEDDER, crops, placement, **kw)


def _photo_crops(placement: str) -> list[np.ndarray]:
    photo = render_board(placement, square_px=48)
    return crop_squares_photo(
        photo, full_frame_corners(photo), headroom=0.0, side_pad=0.0, size=64
    )


def test_exemplar_classifier_rejects_invalid_construction():
    dim = 4
    ex = np.ones((3, dim), dtype=np.float32)
    labels = ["empty", "P", "p"]

    def _expect_value_error(**overrides):
        kwargs = dict(exemplars=ex, labels=labels)
        kwargs.update(overrides)
        try:
            ExemplarClassifier(kwargs.pop("exemplars"), kwargs.pop("labels"), **kwargs)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError for {overrides}")

    _expect_value_error(exemplars=np.zeros((0, dim), dtype=np.float32), labels=[])
    _expect_value_error(labels=["empty", "P"])  # length mismatch
    _expect_value_error(labels=["empty", "P", "nope"])  # unknown class
    _expect_value_error(temperature=0.0)
    _expect_value_error(crop_mode="sideways")
    _expect_value_error(box_headroom=-1.0)
    _expect_value_error(box_side_pad=-1.0)


def test_exemplar_classify_board_round_trips_calibration_position():
    bank = _photo_bank(START_PLACEMENT)
    emb = EMBEDDER.embed(_photo_crops(START_PLACEMENT))
    dist = bank.classify_board(emb)
    assert dist.shape == (8, 8, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)
    assert placement_from_dist(dist) == START_PLACEMENT


def test_exemplar_absent_classes_get_near_zero_probability():
    bank = _photo_bank(EMPTY_PLACEMENT)
    emb = EMBEDDER.embed(_photo_crops(EMPTY_PLACEMENT))
    dist = bank.classify_board(emb)
    assert placement_from_dist(dist) == EMPTY_PLACEMENT
    assert dist[..., CLASS_INDEX["Q"]].max() < 1e-6


def test_exemplar_save_load_round_trip_preserves_predictions_and_metadata(tmp_path):
    bank = _photo_bank(START_PLACEMENT, temperature=0.2, backbone="lite")
    path = tmp_path / "exemplars.npz"
    bank.save(path)
    loaded = ExemplarClassifier.load(path)
    assert loaded.temperature == bank.temperature
    assert loaded.backbone == bank.backbone
    assert loaded.crop_mode == bank.crop_mode
    assert loaded.box_headroom == bank.box_headroom
    assert loaded.box_side_pad == bank.box_side_pad
    np.testing.assert_allclose(loaded.exemplars, bank.exemplars, atol=1e-6)
    emb = EMBEDDER.embed(_photo_crops(START_PLACEMENT))
    np.testing.assert_allclose(
        loaded.classify_board(emb), bank.classify_board(emb), atol=1e-6
    )
    assert placement_from_dist(loaded.classify_board(emb)) == START_PLACEMENT


def test_exemplar_extended_grows_bank_and_keeps_metadata():
    bank = _photo_bank(EMPTY_PLACEMENT, augment=False)
    n0 = bank.exemplars.shape[0]
    labels = [cell for row in parse_placement(START_PLACEMENT) for cell in row]
    emb = EMBEDDER.embed(_photo_crops(START_PLACEMENT))
    grown = bank.extended(emb, labels)
    assert grown.exemplars.shape[0] == n0 + len(labels)
    assert grown.temperature == bank.temperature
    assert grown.backbone == bank.backbone
    assert grown.crop_mode == bank.crop_mode
    # The newly-added position is now classified correctly, without discarding
    # the original bank's calibration.
    assert placement_from_dist(grown.classify_board(emb)) == START_PLACEMENT


def test_load_classifier_dispatches_on_file_kind(tmp_path):
    centroid_path = tmp_path / "centroids.npz"
    _calibrate(START_PLACEMENT).save(centroid_path)
    assert isinstance(load_classifier(centroid_path), CentroidClassifier)

    exemplar_path = tmp_path / "exemplars.npz"
    _photo_bank(START_PLACEMENT, augment=False).save(exemplar_path)
    assert isinstance(load_classifier(exemplar_path), ExemplarClassifier)
