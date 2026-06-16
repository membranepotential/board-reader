"""Tests for the L3 position protocol (DESIGN.md §5)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    NUM_CLASSES,
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    encode_placement,
    iso_utc,
    parse_placement,
    placement_from_dist,
    position_confidence,
)


def test_classes_vocabulary_is_13_and_empty_is_index_0():
    assert NUM_CLASSES == 13
    assert CLASSES[0] == "empty"
    assert set(CLASSES[1:]) == set("PNBRQKpnbrqk")


@pytest.mark.parametrize(
    "placement",
    [
        START_PLACEMENT,
        "8/8/8/8/8/8/8/8",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        "4k3/8/8/8/8/8/8/4K3",
    ],
)
def test_parse_encode_placement_roundtrip(placement):
    grid = parse_placement(placement)
    assert len(grid) == GRID and all(len(row) == GRID for row in grid)
    assert encode_placement(grid) == placement


def test_parse_placement_rejects_malformed():
    with pytest.raises(ValueError):
        parse_placement("8/8/8/8/8/8/8")  # only 7 ranks
    with pytest.raises(ValueError):
        parse_placement("9/8/8/8/8/8/8/8")  # rank overflows
    with pytest.raises(ValueError):
        parse_placement("xxxxxxxx/8/8/8/8/8/8/8")  # invalid symbol


def test_dist_from_placement_is_valid_distribution_and_argmax_consistent():
    dist = dist_from_placement(START_PLACEMENT, confidence=0.9)
    assert dist.shape == (GRID, GRID, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)
    # argmax of the synthesised distribution recovers the placement exactly.
    assert placement_from_dist(dist) == START_PLACEMENT


def test_dist_from_placement_validates_confidence():
    with pytest.raises(ValueError):
        dist_from_placement(START_PLACEMENT, confidence=0.0)
    with pytest.raises(ValueError):
        dist_from_placement(START_PLACEMENT, confidence=1.5)


def test_position_confidence_mean_and_min():
    dist = dist_from_placement(START_PLACEMENT, confidence=0.8)
    assert position_confidence(dist, "mean") == pytest.approx(0.8, abs=1e-5)
    assert position_confidence(dist, "min") == pytest.approx(0.8, abs=1e-5)
    with pytest.raises(ValueError):
        position_confidence(dist, "median")


def test_iso_utc_format():
    # epoch 0 -> 1970-01-01T00:00:00.000Z
    assert iso_utc(0.0) == "1970-01-01T00:00:00.000Z"
    assert iso_utc(1.25).endswith("Z")
    assert "T" in iso_utc(1.25)


def test_clockstate_validation():
    assert ClockState("white").running == "white"
    assert ClockState.from_dict(None).running == "none"
    with pytest.raises(ValueError):
        ClockState("sideways")


def test_position_event_from_square_dist_derives_placement_and_confidence():
    dist = dist_from_placement(START_PLACEMENT, confidence=0.9)
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=7, utc=iso_utc(0.0), square_dist=dist
    )
    assert ev.placement == START_PLACEMENT
    assert ev.confidence == pytest.approx(0.9, abs=1e-5)
    ev.validate()


def test_position_event_json_roundtrip_preserves_placement_and_meta():
    dist = dist_from_placement(START_PLACEMENT, confidence=0.95)
    ev = PositionEvent.from_square_dist(
        board_id="B2",
        seq=42,
        utc=iso_utc(1.5),
        square_dist=dist,
        stability=0.3,
        clock=ClockState("black"),
    )
    back = PositionEvent.from_json(ev.to_json())
    assert back.board_id == "B2"
    assert back.seq == 42
    assert back.placement == ev.placement
    assert back.clock.running == "black"
    assert back.stability == pytest.approx(0.3, abs=1e-4)
    back.validate()


def test_to_dict_has_protocol_shape():
    dist = dist_from_placement(START_PLACEMENT)
    d = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    ).to_dict()
    for key in (
        "board_id", "utc", "seq", "orientation", "classes",
        "placement", "square_dist", "confidence", "stability", "clock", "status",
    ):
        assert key in d
    assert d["classes"] == list(CLASSES)
    assert np.asarray(d["square_dist"]).shape == (GRID, GRID, NUM_CLASSES)
    assert d["status"] == "ok"  # never "illegal" — legality is not ours to judge


def test_validate_rejects_bad_shape():
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0),
        square_dist=dist_from_placement(START_PLACEMENT),
    )
    ev.square_dist = np.zeros((8, 8, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        ev.validate()


def test_validate_rejects_non_normalised_distribution():
    bad = np.zeros((GRID, GRID, NUM_CLASSES), dtype=np.float32)  # rows sum to 0
    ev = PositionEvent(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=bad,
        placement=placement_from_dist(bad), confidence=0.0,
    )
    with pytest.raises(ValueError):
        ev.validate()


def test_validate_rejects_placement_not_matching_argmax():
    dist = dist_from_placement(START_PLACEMENT)
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    )
    ev.placement = "8/8/8/8/8/8/8/8"  # lie about the board
    with pytest.raises(ValueError):
        ev.validate()


def test_validate_rejects_out_of_range_and_bad_enums():
    dist = dist_from_placement(START_PLACEMENT)
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    )
    ev.stability = 1.5
    with pytest.raises(ValueError):
        ev.validate()
    ev.stability = 1.0
    ev.status = "illegal"  # deliberately not a valid status
    with pytest.raises(ValueError):
        ev.validate()


def test_from_dict_rejects_mismatched_classes():
    dist = dist_from_placement(START_PLACEMENT)
    d = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    ).to_dict()
    d["classes"] = list(reversed(CLASSES))
    with pytest.raises(ValueError):
        PositionEvent.from_dict(d)
