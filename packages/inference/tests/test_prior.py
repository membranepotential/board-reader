"""Tests for the hard-constraint decoder (:mod:`board_reader_inference.prior`).

Two things must hold at once: an *impossible* board gets repaired at its weakest
square, and a board the constraints have nothing to say about is returned byte-for-byte
as the estimator's own argmax. The second half is the dangerous one — a decoder that
edits possible boards would silently corrupt good reads — so identity is tested as
hard as repair, and the occluded zero-king board (DESIGN D8) is tested explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASS_INDEX,
    GRID,
    NUM_CLASSES,
    START_PLACEMENT,
    PositionEvent,
    dist_from_placement,
    encode_placement,
    parse_placement,
    placement_from_dist,
)

from board_reader_inference.prior import (
    MAX_FLIPS,
    Constraint,
    Tally,
    decode,
    square_name,
    tally,
    violation,
)

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"


def _dist(placement: str, overrides: dict[str, tuple[str, float]] | None = None) -> np.ndarray:
    """A peaked distribution for ``placement``, with named squares overwritten.

    Overriding a square to ``(class, confidence)`` spreads the remaining mass
    uniformly, so every alternative class there costs the same and the tie-break
    (lowest class index) decides — which keeps these expectations exact.
    """
    dist = np.asarray(dist_from_placement(placement, confidence=0.95), dtype=np.float64)
    for square, (cls, conf) in (overrides or {}).items():
        c, r = "abcdefgh".index(square[0]), GRID - int(square[1])
        dist[r, c] = (1.0 - conf) / (NUM_CLASSES - 1)
        dist[r, c, CLASS_INDEX[cls]] = conf
    return dist


def _counts(**pieces: int) -> Tally:
    per_class = [0] * NUM_CLASSES
    for sym, n in pieces.items():
        per_class[CLASS_INDEX[sym]] = n
    per_class[0] = GRID * GRID - sum(per_class)
    return Tally(per_class=tuple(per_class), back_rank_pawns=0)


# ---- the violation function ---------------------------------------------------


def test_a_legal_army_has_no_violation():
    assert violation(tally(np.zeros((GRID, GRID), dtype=int))).satisfied
    start = _dist(START_PLACEMENT).argmax(axis=-1)
    assert violation(tally(start)).satisfied
    assert str(violation(tally(start))) == "possible"


@pytest.mark.parametrize(
    ("pieces", "constraint", "excess"),
    [
        ({"K": 2, "k": 1}, Constraint.KINGS, 1),
        ({"K": 3}, Constraint.KINGS, 2),
        ({"K": 1, "k": 2}, Constraint.KINGS, 1),
        ({"Q": 3, "P": 8}, Constraint.PROMOTION_BUDGET, 2),  # 2 promotions + 8 pawns
        ({"N": 4, "R": 3, "P": 8}, Constraint.PROMOTION_BUDGET, 3),
        ({"q": 2, "p": 8}, Constraint.PROMOTION_BUDGET, 1),
    ],
)
def test_each_constraint_counts_its_own_excess(pieces, constraint, excess):
    v = violation(_counts(**pieces))
    assert v[constraint] == excess
    assert v.total == excess  # and nothing else fires


def test_a_pawn_on_a_back_rank_is_counted_wherever_it_sits():
    for square in ("a8", "h8", "a1", "e1"):
        for pawn in ("P", "p"):
            grid = _dist(START_PLACEMENT, {square: (pawn, 0.9)}).argmax(axis=-1)
            assert violation(tally(grid)).back_rank_pawns >= 1
    # rank 2..7 is where pawns belong; nothing fires.
    assert violation(tally(_dist(MIDGAME).argmax(axis=-1))).satisfied


def test_the_kings_constraint_is_at_most_one_never_exactly_one():
    """An occluded back rank hides a king (DESIGN D8); zero kings is not a violation."""
    assert violation(_counts(P=8, k=1)).satisfied
    assert violation(_counts(P=8)).satisfied  # neither side's king visible


def test_piece_count_fires_only_alongside_a_king_error():
    """The promotion budget already implies <=16 pieces when a side has one king,
    so this constraint can only bind on its own when the king count is also wrong."""
    # 15 officers + 2 kings = 17 white pieces, spending exactly the 8 promotions
    # the budget allows — so only the count and the king are wrong.
    v = violation(_counts(K=2, N=4, B=4, R=4, Q=3))
    assert v.piece_count == 1 and v.kings == 1 and v.promotion_budget == 0
    # With one king, 17 pieces is impossible to reach without breaking the budget.
    assert violation(_counts(K=1, P=9, N=2, B=2, R=2, Q=1)).promotion_budget > 0


# ---- identity: a possible board is never touched ------------------------------


def test_a_possible_board_comes_back_unchanged_with_no_flips():
    dist = _dist(MIDGAME)
    result = decode(dist)
    assert result.placement == MIDGAME == placement_from_dist(dist)
    assert result.flips == ()
    assert result.residual.satisfied and not result.repaired


def test_the_start_position_is_left_alone():
    result = decode(_dist(START_PLACEMENT))
    assert result.placement == START_PLACEMENT and result.flips == ()


def test_an_occluded_board_with_no_kings_is_left_alone():
    """The failure mode an ``== 1`` king constraint would cause: inventing a king
    by stealing a neighbouring square's label exactly when the photo is worst."""
    no_kings = "r2q3r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1R2"
    result = decode(_dist(no_kings))
    assert result.placement == no_kings and result.flips == ()


def test_the_distribution_is_never_mutated():
    dist = _dist(MIDGAME, {"e8": ("K", 0.6)})
    before = dist.copy()
    decode(dist)
    assert np.array_equal(dist, before)


def test_the_event_keeps_the_raw_argmax_so_the_wire_stays_honest():
    """D1: the decode is a value beside the event, never written into it."""
    dist = _dist(MIDGAME, {"e8": ("K", 0.6)})
    event = PositionEvent.from_square_dist(
        board_id="b", seq=0, utc="1970-01-01T00:00:00Z", square_dist=dist
    ).validate()
    result = decode(event.square_dist)
    assert result.placement != event.placement
    assert event.placement == placement_from_dist(dist)


# ---- repair -------------------------------------------------------------------


def test_the_weaker_king_flips_not_the_confident_one():
    # MIDGAME already has a white king on g1 (0.95). A second one on e8 at 0.60.
    result = decode(_dist(MIDGAME, {"e8": ("K", 0.60)}))
    (flip,) = result.flips
    assert flip.square == "e8" and flip.from_class == "K"
    assert flip.constraint is Constraint.KINGS
    assert flip.cost == pytest.approx(np.log(0.60) - np.log(0.40 / 12))
    assert result.residual.satisfied
    assert "K" not in result.placement.split("/")[0]  # rank 8 lost its white king
    assert result.placement.split("/")[7] == "R2Q1RK1"  # g1 kept its


def test_a_confident_second_king_still_loses_the_less_confident_one():
    """Both kings wrong-ish: the king *evidence* decides, not the rank they sit on."""
    result = decode(_dist(MIDGAME, {"e8": ("K", 0.99), "g1": ("K", 0.55)}))
    (flip,) = result.flips
    assert flip.square == "g1"


def test_surplus_kings_resolve_by_king_evidence_not_by_margin():
    """The 08-13 trial failure, in miniature. The estimator read two black kings
    all session; the *true* king (e8) carried a fat queen tail, which made its
    k-q margin the smallest on 55% of frames — so the margin rule demoted the
    real king and the board's k/q flip-flopped. Which square keeps the king is a
    question about evidence *for the king*, so the weaker ``p(k)`` square is the
    one demoted, even when its margin to the runner-up is the larger."""
    dist = _dist(MIDGAME)
    d8 = np.full(NUM_CLASSES, 0.30 / 11)  # k 0.60, thin q tail: margin 1.79
    d8[CLASS_INDEX["k"]], d8[CLASS_INDEX["q"]] = 0.60, 0.10
    e8 = np.full(NUM_CLASSES, 0.04 / 11)  # k 0.66, fat q tail: margin 0.79
    e8[CLASS_INDEX["k"]], e8[CLASS_INDEX["q"]] = 0.66, 0.30
    dist[0, 3], dist[0, 4] = d8, e8
    result = decode(dist)
    (flip,) = result.flips
    assert flip.square == "d8" and flip.from_class == "k" and flip.to_class == "q"
    assert flip.constraint is Constraint.KINGS
    assert result.residual.satisfied
    assert result.placement.split("/")[0] == "r2qk2r"  # e8 kept the king


def test_a_pawn_on_rank_eight_is_relabelled_when_the_frame_is_known():
    result = decode(_dist(MIDGAME, {"a8": ("p", 0.7)}), quarter_turns=0)
    (flip,) = result.flips
    assert flip.square == "a8" and flip.from_class == "p"
    assert flip.constraint is Constraint.BACK_RANK_PAWNS
    assert result.residual.satisfied


def test_the_cheapest_back_rank_pawn_is_the_one_that_moves():
    result = decode(_dist(MIDGAME, {"a8": ("p", 0.9), "h8": ("p", 0.4)}), quarter_turns=0)
    assert [f.square for f in result.flips] == ["h8", "a8"]
    assert result.residual.satisfied


def test_three_queens_and_eight_pawns_spend_more_promotions_than_exist():
    board = "4k3/8/8/8/3Q4/8/PPPPPPPP/3QKQ2"
    assert violation(tally(_dist(board).argmax(axis=-1))).promotion_budget == 2
    result = decode(_dist(board, {"d4": ("Q", 0.35), "a2": ("P", 0.40)}))
    assert [str(f) for f in result.flips] == ["d4 Q->empty", "a2 P->empty"]
    assert all(f.constraint is Constraint.PROMOTION_BUDGET for f in result.flips)
    assert result.residual.satisfied
    assert result.placement == "4k3/8/8/8/8/8/1PPPPPPP/3QKQ2"


def test_every_surplus_king_is_removed_one_flip_at_a_time():
    board = "kkkkkkkk/8/8/8/8/8/8/KKKKKKKK"  # 8 kings a side
    result = decode(_dist(board))
    assert result.residual.satisfied
    assert len(result.flips) == 14  # 7 surplus kings per colour
    assert {f.constraint for f in result.flips} == {Constraint.KINGS}
    # Every margin is identical, so the deterministic tie-break (lowest square
    # index) empties a8..g8 and leaves the last king standing on h8/h1.
    assert result.placement == "7k/8/8/8/8/8/8/7K"


def test_one_flip_can_relieve_two_constraints_and_is_credited_by_drop_size():
    """17 white pieces including two kings: removing the surplus king fixes the
    count as well, so a single flip lands. Equal drops are credited to whichever
    constraint is declared first, and kings is."""
    board = "4k3/8/8/8/8/K7/NNNNBBBB/RRRRQQQK"
    assert violation(tally(_dist(board).argmax(axis=-1))).total == 2
    result = decode(_dist(board))
    (flip,) = result.flips
    assert flip.square == "a3" and flip.from_class == "K" and flip.to_class == "empty"
    assert flip.constraint is Constraint.KINGS
    assert result.residual.satisfied


# ---- orientation: only the back-rank constraint cares -------------------------


def _rot(placement: str, k: int) -> str:
    from board_reader_inference.labels import rotate_placement

    return rotate_placement(placement, k)


def test_an_unknown_frame_costs_no_pawn_on_a_quarter_turned_board():
    """The defect this default exists to prevent: 16 of 21 confirmed labels are
    stored a quarter- or half-turn off, and in that frame the a- and h-file pawns
    look promoted. With no frame to stand on, the constraint must sit out rather
    than relabel perfectly good squares."""
    sideways = _rot(START_PLACEMENT, 1)
    counted = tally(_dist(sideways).argmax(axis=-1))  # oriented=True: the wrong frame
    assert violation(counted).back_rank_pawns == 4

    unknown = tally(_dist(sideways).argmax(axis=-1), oriented=False)
    assert unknown.back_rank_pawns is None
    assert violation(unknown).satisfied

    result = decode(_dist(sideways))  # quarter_turns defaults to None
    assert result.placement == sideways and result.flips == ()
    assert result.residual.satisfied


def test_a_known_frame_re_enables_the_back_rank_constraint():
    # ``quarter_turns`` is the turn to APPLY to reach upright, so a board rotated
    # by +1 is described by -1 — the same convention ``resolve_rotation`` returns.
    sideways = _rot(START_PLACEMENT, 1)
    assert decode(_dist(sideways), quarter_turns=-1).flips == ()  # correct frame: clean
    # A frame that is genuinely a1-bottom-left, with a genuinely promoted pawn.
    result = decode(_dist(MIDGAME, {"h1": ("P", 0.7)}), quarter_turns=0)
    assert [str(f) for f in result.flips] == ["h1 P->empty"]
    assert result.flips[0].constraint is Constraint.BACK_RANK_PAWNS


def test_an_inferred_rotation_is_never_what_the_eval_hands_the_decoder():
    """``resolve_rotation`` minimises exactly what BACK_RANK_PAWNS measures, so
    feeding it in is circular: it fires only where the guess was wrong."""
    from board_reader_inference.autocalib import resolve_rotation

    sideways = _dist(_rot(START_PLACEMENT, 1))
    guess, upright = resolve_rotation(sideways)
    # The guesser gets this one right (undoing a +1 turn takes 3), and in doing so
    # has already driven the constraint to zero — there is nothing left for it to say.
    assert guess == 3
    assert violation(tally(upright.argmax(axis=-1))).satisfied
    assert decode(sideways, quarter_turns=guess).flips == ()


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_repair_is_reported_in_the_frame_it_was_handed(turns):
    """A half-turned board's flips must not come back naming the mirrored square."""
    board = _rot(MIDGAME, turns)
    square = {0: "e8", 1: "a5", 2: "d1", 3: "h4"}[turns]  # where MIDGAME's e8 lands
    result = decode(_dist(board, {square: ("K", 0.6)}), quarter_turns=-turns)
    (flip,) = result.flips
    assert flip.square == square and flip.from_class == "K"
    # The surplus king is emptied and nothing else moves, in the caller's frame.
    grid = parse_placement(MIDGAME)
    grid[0][4] = "empty"  # e8
    assert result.placement == _rot(encode_placement(grid), turns)


def test_the_count_constraints_are_rotation_invariant():
    board = "kkkkkkkk/8/8/8/8/8/8/KKKKKKKK"
    for turns in range(4):
        result = decode(_dist(_rot(board, turns)), quarter_turns=-turns)
        assert result.residual.satisfied
        assert len(result.flips) == 14


# ---- determinism, termination, guards -----------------------------------------


def test_the_same_input_always_decodes_the_same_way():
    dist = _dist(MIDGAME, {"e8": ("K", 0.6), "a8": ("p", 0.6), "d5": ("Q", 0.6)})
    first = decode(dist)
    for _ in range(4):
        assert decode(dist.copy()) == first


def test_an_all_queens_board_terminates_and_reports_what_it_could_not_fix():
    dist = _dist("QQQQQQQQ/" * 7 + "QQQQQQQQ")
    result = decode(dist)
    assert result.residual.satisfied
    assert len(result.flips) <= MAX_FLIPS
    assert result.placement.count("Q") == 9  # 1 free queen + 8 promotions


def test_a_flip_cap_returns_the_best_so_far_with_a_residual():
    dist = _dist("QQQQQQQQ/" * 7 + "QQQQQQQQ")
    result = decode(dist, max_flips=3)
    assert len(result.flips) == 3
    assert not result.residual.satisfied
    assert result.residual.promotion_budget > 0
    assert "promotion_budget" in str(result.residual)


def test_a_zero_probability_class_does_not_produce_an_infinite_cost():
    dist = np.zeros((GRID, GRID, NUM_CLASSES), dtype=np.float64)
    dist[..., CLASS_INDEX["K"]] = 1.0  # 64 white kings, every alternative at p=0
    result = decode(dist)
    assert all(np.isfinite(f.cost) for f in result.flips)
    assert result.residual.satisfied


def test_a_wrongly_shaped_distribution_is_rejected():
    with pytest.raises(ValueError, match="square_dist must be"):
        decode(np.zeros((GRID, GRID)))


def test_square_names_match_the_placement_frame():
    assert square_name(0) == "a8"
    assert square_name(GRID - 1) == "h8"
    assert square_name(GRID * GRID - 1) == "h1"
    assert square_name(GRID * (GRID - 1)) == "a1"
