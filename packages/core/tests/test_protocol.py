"""Tests for the L3 position protocol (DESIGN.md §5)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    EVIDENCE_TOP_K,
    GRID,
    NUM_CLASSES,
    START_PLACEMENT,
    BoardName,
    ClockState,
    MotionUpdate,
    NamesMessage,
    PositionEvent,
    dist_from_placement,
    encode_placement,
    iso_utc,
    parse_placement,
    parse_viewer_message,
    placement_from_dist,
    position_confidence,
    square_evidence,
    square_name_at,
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
    with pytest.raises(ValueError):
        parse_placement("08/8/8/8/8/8/8/8")  # a run is 1 to 8 squares; 0 is no run


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
    assert ev.placement == START_PLACEMENT == ev.raw_placement
    assert ev.pending == ()  # a one-shot read publishes itself
    assert ev.changed_utc == ev.utc  # ... and its board changed when it was read
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
        "changed_utc", "placement", "pending", "square_dist", "confidence", "stability",
        "clock", "status", "obscured",
    ):
        assert key in d
    assert d["classes"] == list(CLASSES)
    assert np.asarray(d["square_dist"]).shape == (GRID, GRID, NUM_CLASSES)
    assert d["status"] == "ok"  # never "illegal" — legality is not ours to judge
    assert d["obscured"] is False  # a one-shot read is never judged obscured


def test_an_event_recorded_before_the_obscured_flag_parses_as_clear():
    line = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist_from_placement(START_PLACEMENT)
    ).to_dict()
    del line["obscured"]
    assert PositionEvent.from_dict(line).obscured is False


def test_an_event_refuses_a_distribution_of_the_wrong_shape_or_range():
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0),
        square_dist=dist_from_placement(START_PLACEMENT),
    )
    with pytest.raises(ValueError, match="square_dist must be"):
        ev.square_dist = np.zeros((8, 8, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="probabilities"):
        ev.square_dist = np.full((GRID, GRID, NUM_CLASSES), 1.5, dtype=np.float32)
    nested = ev.to_dict()["square_dist"]
    nested[0][0][0] = True  # a bool is not a probability, though numpy would say 1.0
    with pytest.raises(ValueError):
        ev.square_dist = nested


def test_validate_rejects_non_normalised_distribution():
    bad = np.zeros((GRID, GRID, NUM_CLASSES), dtype=np.float32)  # rows sum to 0
    ev = PositionEvent(
        board_id="A1", seq=0, utc=iso_utc(0.0), changed_utc=iso_utc(0.0), square_dist=bad,
        placement=placement_from_dist(bad), confidence=0.0,
    )
    with pytest.raises(ValueError):
        ev.validate()


def test_validate_rejects_pending_that_does_not_match_the_argmax():
    dist = dist_from_placement(START_PLACEMENT)
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    )
    ev.placement = "8/8/8/8/8/8/8/8"  # a board with no pending squares to show for it
    with pytest.raises(ValueError, match="pending"):
        ev.validate()


def test_an_event_refuses_out_of_range_values_and_bad_enums():
    dist = dist_from_placement(START_PLACEMENT)
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    )
    with pytest.raises(ValueError):
        ev.stability = 1.5
    with pytest.raises(ValueError):
        ev.status = "illegal"  # deliberately not a valid status
    with pytest.raises(ValueError):
        ev.seq = -1
    assert (ev.stability, ev.status, ev.seq) == (1.0, "ok", 0)


def test_from_dict_rejects_mismatched_classes():
    dist = dist_from_placement(START_PLACEMENT)
    d = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist
    ).to_dict()
    d["classes"] = list(reversed(CLASSES))
    with pytest.raises(ValueError):
        PositionEvent.from_dict(d)


def test_a_settled_board_is_the_placement_and_the_read_stays_in_the_dist():
    """One placement on the wire: the settled board, with ``pending`` naming where
    this read disagrees, and the read itself recoverable as the argmax."""
    read = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"  # 1.e4, not settled yet
    ev = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist_from_placement(read)
    ).with_placement(START_PLACEMENT, changed_utc=iso_utc(-5.0), obscured=True)
    assert ev.placement == START_PLACEMENT
    assert ev.changed_utc == "1969-12-31T23:59:55.000Z"  # the board held since then
    assert ev.obscured
    assert ev.raw_placement == read
    assert ev.pending == ("e4", "e2")  # board order: rank 8 first
    ev.validate()

    back = PositionEvent.from_json(ev.to_json())
    assert back.to_dict()["pending"] == ["e4", "e2"]
    assert back.placement == START_PLACEMENT
    back.validate()

    # The constructor takes the settled board too, and derives the same pending.
    built = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0), square_dist=dist_from_placement(read),
        placement=START_PLACEMENT,
    )
    assert built.pending == ev.pending


def test_an_old_event_with_a_settled_block_is_refused():
    """The wire refuses the block it no longer has; recordings are read apart."""
    d = PositionEvent.from_square_dist(
        board_id="A1", seq=0, utc=iso_utc(0.0),
        square_dist=dist_from_placement(START_PLACEMENT),
    ).to_dict()
    d["settled"] = {"placement": START_PLACEMENT, "pending": []}
    with pytest.raises(ValueError):
        PositionEvent.from_dict(d)


# --------------------------------------------------------------------------- #
# The evidence digest — what the recorder keeps of a distribution it drops
# --------------------------------------------------------------------------- #


def test_evidence_names_the_winner_and_its_nearest_rival():
    dist = dist_from_placement(START_PLACEMENT, confidence=0.6)
    # Make a1's runner-up unambiguous: the rook wins, the queen is second.
    dist[7, 0, CLASS_INDEX["Q"]] = 0.3
    evidence = square_evidence(dist)

    assert len(evidence) == GRID * GRID
    a1 = evidence[next(i for i in range(GRID * GRID) if square_name_at(i) == "a1")]
    assert [c for c, _ in a1] == ["R", "Q"]
    assert a1[0][1] == pytest.approx(0.6, abs=1e-3)
    assert a1[1][1] == pytest.approx(0.3, abs=1e-3)


def test_evidence_agrees_with_the_placement_it_ships_beside():
    """The digest's top class per square IS the argmax — the same read, digested."""
    dist = dist_from_placement(START_PLACEMENT)
    tops = [c for square in square_evidence(dist) for c, _ in square[:1]]
    grid = parse_placement(placement_from_dist(dist))
    assert tops == [grid[r][c] for r in range(GRID) for c in range(GRID)]


def test_evidence_keeps_as_many_classes_as_asked():
    dist = dist_from_placement(START_PLACEMENT)
    assert all(len(square) == 5 for square in square_evidence(dist, top_k=5))
    assert all(len(square) == 1 for square in square_evidence(dist, top_k=1))


def test_evidence_rejects_a_shape_or_a_depth_it_cannot_honour():
    dist = dist_from_placement(START_PLACEMENT)
    with pytest.raises(ValueError, match="square_dist must be"):
        square_evidence(dist[:4])
    with pytest.raises(ValueError, match="top_k must be"):
        square_evidence(dist, top_k=0)
    with pytest.raises(ValueError, match="top_k must be"):
        square_evidence(dist, top_k=NUM_CLASSES + 1)


def test_evidence_ties_break_the_same_way_every_run():
    """A flat square is the worst case for a digest: it must still be stable."""
    flat = np.full((GRID, GRID, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    once = square_evidence(flat)
    assert once == square_evidence(flat)
    assert [c for c, _ in once[0]] == list(CLASSES[:EVIDENCE_TOP_K])


def test_viewer_messages_parse_by_their_type():
    """A position event has no `type`; a motion update says `motion`."""
    event = PositionEvent.from_square_dist(
        board_id="A1", seq=2, utc=iso_utc(0.0), square_dist=dist_from_placement(START_PLACEMENT)
    )
    assert "type" not in event.to_dict()
    assert parse_viewer_message(event.to_json()).placement == START_PLACEMENT
    motion = MotionUpdate(board_id="A1", moving=False, stability=0.93, ts=1_700_000_000.25)
    assert motion.to_dict() == {
        "type": "motion", "board_id": "A1", "moving": False, "stability": 0.93,
        "ts": 1_700_000_000.25,
    }
    assert parse_viewer_message(motion.to_json()) == motion
    names = NamesMessage(names=(BoardName("board-2", "kitchen table"),))
    assert names.to_dict() == {
        "type": "names", "names": [{"board_id": "board-2", "name": "kitchen table"}],
    }
    assert parse_viewer_message(names.to_json()) == names
    with pytest.raises(ValueError, match="unknown /viewer message type"):
        parse_viewer_message('{"type": "gossip"}')
