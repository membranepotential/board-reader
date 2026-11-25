"""Tests for the per-square debounce behind ``PositionEvent.settled``."""

from __future__ import annotations

import pytest

from board_reader_core.protocol import (
    START_PLACEMENT,
    dist_from_placement,
    encode_placement,
    parse_placement,
    pending_squares,
)

from board_reader_inference.stream.rules import violation
from board_reader_inference.stream.stabilize import (
    OBSCURED_ENERGY,
    BoardStabilizer,
    SettleConfig,
    balanced_squares,
)

#: START_PLACEMENT after 1.e4 — one real move, two squares changed.
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: A phantom: the trial's hallucinated black rook on b6. Note it is *impossible*
#: — 17 black pieces — so the guard keeps it off the settled board.
PHANTOM_B6 = "rnbqkbnr/pppppppp/1r6/8/8/8/PPPPPPPP/RNBQKBNR"
#: The other flicker a hand makes, and a possible board: b8 briefly occluded.
GONE_B8 = "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: The k/q swap the 08-27 control take coin-flipped into: d8 k, e8 q.
#: Possible on its own — it is only ever *half* of it that is not.
SWAP = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: The same reading with d8 misread as empty: e8 still k, d8 no longer q.
D8_EMPTY = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: Two black kings — what composing halves of START and SWAP produces.
TWO_KINGS = "rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _pending(read: str, settled: str) -> tuple[str, ...]:
    """The squares, board order, where ``read`` differs from the settled board:
    the event's ``pending`` (``protocol.pending_squares``) for that read."""
    return pending_squares(dist_from_placement(read), settled)


def is_possible(placement: str) -> bool:
    """Does this board break none of the guard's conditions? (the invariant)"""
    return violation(placement).total == 0


def test_first_read_is_adopted_wholesale():
    settled = BoardStabilizer().observe(START_PLACEMENT, 0.0)
    assert settled == START_PLACEMENT
    assert _pending(START_PLACEMENT, settled) == ()


def test_a_real_move_is_adopted_once_it_spans_the_window():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(E4, 1.0) == START_PLACEMENT
    held = stab.observe(E4, 2.0)  # consistent, but only 1.0s into the window
    assert held == START_PLACEMENT
    assert _pending(E4, held) == ("e4", "e2")  # board order: rank 8 first
    done = stab.observe(E4, 3.6)  # 2.6s of unbroken E4: adopted
    assert done == E4
    assert _pending(E4, done) == ()


def test_changed_ts_is_the_first_read_that_showed_the_settled_board():
    """The board changes at the start of the run that settled it, not at the read
    that finished the window; a read that commits nothing leaves it alone."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    assert stab.changed_ts is None
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.changed_ts == 0.0
    stab.observe(E4, 1.0)  # the move is first read here
    stab.observe(E4, 2.0)
    assert stab.changed_ts == 0.0  # not settled yet: still the start position's time
    assert stab.observe(E4, 3.6) == E4
    assert stab.changed_ts == 1.0
    stab.observe(E4, 9.0)
    assert stab.changed_ts == 1.0


def test_a_commit_of_several_squares_changes_at_the_latest_run_start():
    """A drag adopts a partner whose run began earlier; the board as it now
    stands was first read only once the later square read its new class too."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    _swap_stream(stab, START_PLACEMENT)  # d8 from 1.0 (dragged), e8 from 2.0
    assert stab.changed_ts == 2.0


def test_wall_clock_alone_is_not_evidence():
    """One read followed by silence must not be adopted on its second sighting
    hours later — at least SETTLE_MIN_READS *consistent* reads are required."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5, min_reads=3))
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(E4, 10.0)
    assert stab.observe(E4, 20.0) == START_PLACEMENT  # 2 reads < 3
    assert stab.observe(E4, 20.5) == E4


def test_a_flicker_never_reaches_the_settled_board():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    flickered = stab.observe(GONE_B8, 1.0)
    assert flickered == START_PLACEMENT
    assert _pending(GONE_B8, flickered) == ("b8",)
    recovered = stab.observe(START_PLACEMENT, 1.5)
    assert recovered == START_PLACEMENT
    assert _pending(START_PLACEMENT, recovered) == ()
    # the run restarted: the flicker's clock does not carry over
    stab.observe(GONE_B8, 2.0)
    assert stab.observe(GONE_B8, 4.0) == START_PLACEMENT


def test_alternating_reads_never_accumulate():
    """The trial's d8/e8 k-q coin flip: each disagreement restarts the run.

    Deliberate, and re-measured when the coherence rule below was added: letting
    a persistent minority reading accumulate votes instead (majority or
    long-horizon attrition over a contested episode) was scored offline on all
    four 08-27 takes x six read policies and lost on every pooled axis — settled
    perfect 37.7 % -> 33-36 %, and the settled board *changed* more often, which
    is the one thing a debounce exists to stop. A coin-flipping square is not
    evidence that anything moved.
    """
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    for i in range(20):  # 10s of alternation, far beyond the window
        assert stab.observe(SWAP if i % 2 else START_PLACEMENT, float(i)) == (
            START_PLACEMENT
        )


def test_a_backwards_timestamp_restarts_the_run():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 100.0)
    stab.observe(E4, 101.0)
    stab.observe(E4, 90.0)  # edge clock stepped back: evidence discarded
    assert stab.observe(E4, 92.0) == START_PLACEMENT
    assert stab.observe(E4, 93.0) == E4  # 3.0s from the 90.0 restart


def test_zero_window_still_needs_min_reads():
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(GONE_B8, 1.0) == START_PLACEMENT
    assert stab.observe(GONE_B8, 2.0) == GONE_B8


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        SettleConfig(seconds=-1.0)
    with pytest.raises(ValueError):
        SettleConfig(seconds=float("nan"))
    with pytest.raises(ValueError):
        SettleConfig(min_reads=0)
    with pytest.raises(ValueError):
        SettleConfig(stale_windows=0)
    for energy in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            SettleConfig(obscured_energy=energy)


def test_settle_config_from_env_keeps_unset_knobs_at_their_defaults():
    assert SettleConfig.from_env({"BR_SETTLE_SECONDS": "1.5", "BR_SETTLE_MIN_READS": ""}) == (
        SettleConfig(seconds=1.5)
    )
    assert SettleConfig.from_env({}) == SettleConfig()
    assert SettleConfig.from_env(
        {"BR_SETTLE_SECONDS": "0.5", "BR_SETTLE_MIN_READS": "1", "BR_SETTLE_STALE_WINDOWS": "3"}
    ) == SettleConfig(seconds=0.5, min_reads=1, stale_windows=3)
    assert SettleConfig.from_env({"BR_OBSCURED_ENERGY": "-11.5"}) == (
        SettleConfig(obscured_energy=-11.5)
    )
    assert SettleConfig.from_env({"BR_OBSCURED_ENERGY": "off"}) == (
        SettleConfig(obscured_energy=None)
    )


@pytest.mark.parametrize(
    "env",
    [
        {"BR_SETTLE_SECONDS": "fast"},
        {"BR_SETTLE_MIN_READS": "1.5"},
        {"BR_SETTLE_STALE_WINDOWS": "0"},
        {"BR_OBSCURED_ENERGY": "high"},
        {"BR_OBSCURED_ENERGY": "nan"},
    ],
)
def test_a_malformed_settle_env_fails_loudly(env):
    with pytest.raises(ValueError):
        SettleConfig.from_env(env)


def test_settle_config_names_itself_for_a_replay_directory():
    assert SettleConfig().is_default
    config = SettleConfig(seconds=1.5, min_reads=1, stale_windows=3)
    assert not config.is_default
    assert config.slug == "settle1.5s-1r-3w"
    assert config.to_dict() == {
        "settle_seconds": 1.5,
        "settle_min_reads": 1,
        "stale_windows": 3,
        "obscured_energy": OBSCURED_ENERGY,
    }
    assert SettleConfig(obscured_energy=-12.0).slug == "settle0.3s-2r-2w-e-12"
    assert SettleConfig(obscured_energy=-12.25).slug == "settle0.3s-2r-2w-e-12.25"
    assert SettleConfig(obscured_energy=None).slug == "settle0.3s-2r-2w-eoff"


def test_a_read_is_obscured_only_above_the_threshold():
    config = SettleConfig(obscured_energy=-12.4)
    assert config.obscures(-12.0)
    assert not config.obscures(-12.4)  # strictly above, as the probe gated
    assert not config.obscures(-13.0)
    assert not config.obscures(None)  # a recognizer that reports no energy
    assert not SettleConfig(obscured_energy=None).obscures(0.0)  # the gate is off


def test_a_piece_that_only_vanished_never_commits_on_an_obscured_read():
    """The resting hand: reads under it agree on a wrong board for longer than the
    window, and none of them commits, because the knight left b8 for no other
    square. The first clear read that agrees with the challenger they started then
    commits at once: their evidence counted."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    for ts in (1.0, 1.5, 2.0, 3.0, 5.0):  # four seconds of the hand, ripe many times over
        assert stab.observe(GONE_B8, ts, obscured=True) == START_PLACEMENT
    assert stab.changed_ts == 0.0
    assert stab.observe(GONE_B8, 5.3) == GONE_B8
    assert stab.changed_ts == 1.0  # the run began under the hand


def test_a_clear_read_after_the_hand_resets_what_the_hand_started():
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    for ts in (1.0, 1.5, 2.0):
        stab.observe(GONE_B8, ts, obscured=True)
    assert stab.observe(START_PLACEMENT, 2.5) == START_PLACEMENT
    assert stab.observe(GONE_B8, 2.8) == START_PLACEMENT  # a fresh run, not ripe yet


def test_an_obscured_read_cannot_take_the_stale_fall_through():
    """The deadline escape is a commit too: under the hand it waits."""
    stab = BoardStabilizer(SettleConfig(seconds=1.0, stale_windows=1))
    stab.observe(START_PLACEMENT, 0.0)
    # d8 alone reads k (two black kings if committed alone): ripe, blocked.
    for i in range(1, 5):
        stab.observe(SWAP if i % 2 else TWO_KINGS, i * 0.5)
    # Clear, this stream falls through to SWAP at 2.5 s; obscured, it never does.
    for i in range(5, 40):
        assert stab.observe(SWAP if i % 2 else TWO_KINGS, i * 0.5, obscured=True) == (
            START_PLACEMENT
        )


def test_an_obscured_first_read_is_still_adopted():
    """No settled board exists to hold, so the first read is the board (and the
    clear reads that follow displace whatever it got wrong)."""
    assert BoardStabilizer().observe(GONE_B8, 0.0, obscured=True) == GONE_B8


#: E4 with a hand over b8: the move in plain view, one piece hidden elsewhere.
E4_GONE_B8 = "r1bqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: E4 with a hand over a2: the hidden piece is a white pawn, like the moved one.
E4_GONE_A2 = "rnbqkbnr/pppppppp/8/8/4P3/8/1PPP1PPP/RNBQKBNR"
#: After 1.e4 d5, then 2.exd5, and that capture half done (the pawn in the hand).
E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR"
EXD5 = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR"
E4_LIFTED = "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPP1PPP/RNBQKBNR"
#: White ready to castle short, castled, and the king moved with the rook to follow.
CAN_CASTLE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R"
CASTLED = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1"
KING_ONLY = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ2KR"
#: White's e5 pawn takes d5 en passant, and a white pawn promotes on a8.
EP_BEFORE = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR"
EP_AFTER = "rnbqkbnr/ppp1pppp/3P4/8/8/8/PPPP1PPP/RNBQKBNR"
PROMOTE_BEFORE = "4k3/P7/8/8/8/8/8/4K3"
PROMOTE_AFTER = "Q3k3/8/8/8/8/8/8/4K3"


def _squares(*names: str) -> list[tuple[int, int]]:
    """Algebraic names -> the stabilizer's (row, column) squares."""
    return [(8 - int(name[1]), "abcdefgh".index(name[0])) for name in names]


def _balanced(settled: str, read: str) -> set[tuple[int, int]]:
    """``balanced_squares`` with every differing square ripe."""
    was, now = parse_placement(settled), parse_placement(read)
    ripe = [(r, c) for r in range(8) for c in range(8) if was[r][c] != now[r][c]]
    return set(balanced_squares(was, now, ripe))


@pytest.mark.parametrize(
    ("settled", "read", "squares"),
    [
        (START_PLACEMENT, E4, ("e2", "e4")),  # a move
        (E4_D5, EXD5, ("e4", "d5")),  # a capture: the pawn left e4 and stands on d5
        (CAN_CASTLE, CASTLED, ("e1", "g1", "h1", "f1")),  # castling: two groups
        (START_PLACEMENT, GONE_B8, ()),  # a piece that only vanished
        (GONE_B8, START_PLACEMENT, ()),  # a piece that only appeared
        (START_PLACEMENT, SWAP, ()),  # two pieces misread as each other: no square emptied
        (START_PLACEMENT, E4_GONE_B8, ("e2", "e4")),  # the move, not the hidden knight
        (START_PLACEMENT, E4_GONE_A2, ()),  # two pawns left, one arrived: which one moved?
        (CAN_CASTLE, KING_ONLY, ("e1", "g1")),  # the king's group balances on its own
        (EP_BEFORE, EP_AFTER, ("e5", "d6")),  # en passant: the captured pawn only vanished
        (PROMOTE_BEFORE, PROMOTE_AFTER, ()),  # a promotion: a pawn vanished, a queen appeared
    ],
)
def test_balanced_squares_are_the_pieces_that_left_one_square_for_another(
    settled, read, squares
):
    assert _balanced(settled, read) == set(_squares(*squares))


def test_balanced_squares_looks_at_ripe_squares_only():
    """A group is judged on what is ripe: the move's target alone is a piece that
    only appeared, however clearly the read shows the source empty too."""
    was, now = parse_placement(START_PLACEMENT), parse_placement(E4)
    assert balanced_squares(was, now, _squares("e4")) == []
    assert balanced_squares(was, now, []) == []


def test_committing_balanced_squares_never_raises_the_excess():
    """Why an obscured commit needs no guard: per piece the count stays the same
    and a capture only removes a piece. Seeded boards from a small alphabet, so
    that kings, pawns and queens sit at their limits; each read makes a few moves
    and a few misreads. Adopting the whole read does raise the excess on the same
    boards, so the generator reaches the conditions."""
    import random

    rng = random.Random(20261125)
    alphabet = ["empty"] * 6 + list("KkPpQqNn")
    cells = [(r, c) for r in range(2) for c in range(8)]
    pad = [["empty"] * 8 for _ in range(6)]

    def total(grid: list[list[str]]) -> int:
        return violation(encode_placement(grid)).total

    committed = whole_read_raises = 0
    for _ in range(1000):
        was = [[rng.choice(alphabet) for _ in range(8)] for _ in range(2)] + pad
        now = [row[:] for row in was]
        for _ in range(rng.randint(1, 3)):  # moves and captures
            (r0, c0), (r1, c1) = rng.sample(cells, 2)
            now[r1][c1], now[r0][c0] = now[r0][c0], "empty"
        for _ in range(rng.randint(0, 2)):  # misreads
            r, c = rng.choice(cells)
            now[r][c] = rng.choice(alphabet)
        ripe = [(r, c) for r, c in cells if was[r][c] != now[r][c]]
        taken = balanced_squares(was, now, ripe)
        after = [row[:] for row in was]
        for r, c in taken:
            after[r][c] = now[r][c]
        committed += bool(taken)
        whole_read_raises += total(now) > total(was)
        assert total(after) <= total(was)
    assert committed > 300
    assert whole_read_raises > 20


def test_a_move_in_plain_view_commits_on_obscured_reads():
    """The hand rests elsewhere on the board: the move shows after the usual
    window, and the hidden knight stays on the settled board."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(E4_GONE_B8, 1.0, obscured=True) == START_PLACEMENT
    assert stab.observe(E4_GONE_B8, 1.3, obscured=True) == START_PLACEMENT  # not ripe yet
    assert stab.observe(E4_GONE_B8, 1.6, obscured=True) == E4
    assert stab.changed_ts == 1.0
    assert _pending(E4_GONE_B8, E4) == ("b8",)
    for ts in (2.0, 4.0, 6.0):  # the hand stays: the knight never leaves
        assert stab.observe(E4_GONE_B8, ts, obscured=True) == E4
    assert stab.observe(E4, 6.3) == E4  # the hand is gone, so is its challenger
    assert stab.changed_ts == 1.0


def test_one_obscured_read_is_still_no_quorum():
    stab = BoardStabilizer(SettleConfig(seconds=0.0, min_reads=2))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(E4, 1.0, obscured=True) == START_PLACEMENT
    assert stab.observe(E4, 1.1, obscured=True) == E4


def test_a_capture_commits_on_obscured_reads_but_not_its_first_half():
    """The pawn in the hand: e4 reads empty and nothing gained a white pawn, so
    the half-made capture waits. Set down on d5, both squares commit together."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(E4_D5, 0.0)
    for ts in (1.0, 1.6, 2.2):
        assert stab.observe(E4_LIFTED, ts, obscured=True) == E4_D5
    assert stab.observe(EXD5, 2.8, obscured=True) == E4_D5  # d5 has one read
    assert stab.observe(EXD5, 3.4, obscured=True) == EXD5
    assert stab.changed_ts == 2.8  # the board as it stands was first read then


def test_a_hidden_piece_of_the_moved_kind_holds_the_move_until_a_clear_read():
    """Two white pawns left their squares and one arrived: the count cannot say
    which moved, so nothing commits under the hand. A clear read settles it."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    for ts in (1.0, 1.6, 2.2, 4.2):
        assert stab.observe(E4_GONE_A2, ts, obscured=True) == START_PLACEMENT
    assert stab.observe(E4, 4.5) == E4


def test_a_swap_never_commits_on_obscured_reads():
    """No square emptied, so no group balances: k and q misread as each other
    under a hand wait for a clear read, however long they persist."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    for i in range(1, 20):
        assert stab.observe(SWAP, i * 0.5, obscured=True) == START_PLACEMENT


def test_stale_windows_sets_how_long_a_blocked_square_waits():
    """A square blocked by the coherence rule falls through after
    ``seconds * stale_windows`` ripe seconds — one window sooner with one fewer."""

    def settle_time(stale_windows: int) -> float:
        stab = BoardStabilizer(SettleConfig(seconds=1.0, stale_windows=stale_windows))
        stab.observe(START_PLACEMENT, 0.0)
        # d8 alone reads k: committing it alone makes two black kings, and its
        # partner (e8) never reaches quorum, so only the deadline can land it.
        two_kings = "rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        partner = ["rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", two_kings]
        for i in range(1, 60):
            ts = i * 0.25
            board = partner[0] if i % 2 else two_kings
            if stab.observe(board, ts) != START_PLACEMENT:
                return ts
        return float("inf")

    assert settle_time(1) < settle_time(2) < float("inf")


def test_rejects_a_malformed_placement():
    with pytest.raises(ValueError):
        BoardStabilizer().observe("not/a/board", 0.0)


# --------------------------------------------------------------------------- #
# Coherence: the board the per-square settlements compose must still be a board
# --------------------------------------------------------------------------- #


def _swap_stream(stab: BoardStabilizer, d8_second: str) -> list:
    """The 08-27 control take's failure, in four reads.

    The first read is the swap, so the settled board starts with ``k`` on d8 and
    ``q`` on e8. The next three read e8 as ``k`` — an unbroken 2.6 s run that
    goes ripe — while d8's run is restarted by ``d8_second``. Adopting e8 alone
    would put a second black king on the board.
    """
    return [
        stab.observe(SWAP, 0.0),
        stab.observe(D8_EMPTY, 1.0),
        stab.observe(START_PLACEMENT, 2.0),
        stab.observe(d8_second, 3.6),
    ]


def test_a_swap_is_displaced_as_a_swap():
    """d8's quorum is dragged into e8's settlement rather than left behind.

    This is the 08-27 control take's lock-in: settled per-square 96.1 % *below*
    its own 97.2 % repaired input, 0/253 perfect boards, 0/7 truth segments ever
    settled — because 69 % of its settled events carried two black kings, a board
    no read ever claimed and no truth segment can equal.
    """
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    settled = _swap_stream(stab, START_PLACEMENT)
    assert settled[:3] == [SWAP, SWAP, SWAP]
    assert settled[3] == START_PLACEMENT  # both halves of the swap, together
    assert all(is_possible(board) for board in settled)


def test_a_sub_quorum_square_is_not_dragged_and_the_commit_is_held():
    """Dragging is not a free pass: the dragged square owes ``settle_min_reads``
    consistent reads. Without them the ripe displacement waits instead of
    landing on a board nobody read."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    boards = _swap_stream(stab, D8_EMPTY)  # d8's run restarts on the last read
    assert boards == [SWAP] * 4  # ripe, held, still possible
    assert _pending(D8_EMPTY, boards[-1]) == ("d8", "e8")  # both in flux, evidence kept
    # One more read gives d8 its quorum, and now the pair moves together.
    assert stab.observe(D8_EMPTY, 3.7) == D8_EMPTY


def test_the_settled_board_never_becomes_less_possible():
    """The invariant, over a stream that mixes both readings of the swap with
    noise. Measured the same way over all four 08-27 takes x six read policies:
    0 impossible settled boards, against 233/337 on the control take before."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stream = [SWAP, D8_EMPTY, START_PLACEMENT, SWAP, START_PLACEMENT, START_PLACEMENT,
              D8_EMPTY, START_PLACEMENT, START_PLACEMENT, SWAP, SWAP, START_PLACEMENT]
    for i, board in enumerate(stream):
        settled = stab.observe(board, float(i) * 0.9)
        assert settled != TWO_KINGS
        assert is_possible(settled)


def test_an_impossible_read_is_never_adopted():
    """A hallucinated 17th piece survives the whole window and still does not
    land: the settled board may not be made less possible than it is."""
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(PHANTOM_B6, 1.0)
    settled = stab.observe(PHANTOM_B6, 5.0)
    assert settled == START_PLACEMENT
    assert _pending(PHANTOM_B6, settled) == ("b6",)


def test_free_rules_settle_an_impossible_read_like_any_other():
    """``free`` turns the guard off for a board that holds extra material or plays
    a variant: the same phantom 17th piece now settles once it spans the window."""
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0, "free")
    stab.observe(PHANTOM_B6, 1.0, "free")
    assert stab.observe(PHANTOM_B6, 5.0, "free") == PHANTOM_B6


def test_the_guard_follows_the_setting_from_one_read_to_the_next():
    """The node reads the board's setting per read, so switching back to
    ``standard`` mid-stream guards the very next commit."""
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0, "free")
    stab.observe(PHANTOM_B6, 1.0, "free")
    assert stab.observe(PHANTOM_B6, 5.0, "standard") == START_PLACEMENT


def test_an_impossible_first_board_is_still_escapable():
    """The first read is adopted wholesale even if it is impossible. The rule is
    *never less possible*, not *never commit*, so the stream still recovers."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    assert stab.observe(TWO_KINGS, 0.0) == TWO_KINGS
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6) == START_PLACEMENT


def test_a_wrong_first_board_is_displaced_like_any_other():
    """Wholesale adoption is not adoption forever: the square the first read got
    wrong needs the same window as every other square, and gets it."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(SWAP, 0.0)
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6) == START_PLACEMENT


def test_pending_lists_every_square_reading_differently():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    settled = stab.observe(SWAP, 1.0)
    assert settled == START_PLACEMENT
    assert _pending(SWAP, settled) == ("d8", "e8")


def test_two_identical_streams_settle_identically():
    """Deterministic on capture_ts alone — the harness the replay depends on."""
    stream = [(SWAP, 0.0), (D8_EMPTY, 1.0), (START_PLACEMENT, 2.0),
              (START_PLACEMENT, 3.6), (E4, 4.0), (E4, 7.0), (PHANTOM_B6, 7.5)]
    runs = [
        [stab.observe(p, t) for p, t in stream]
        for stab in (BoardStabilizer(), BoardStabilizer())
    ]
    assert runs[0] == runs[1]

def _noisy_stream(rng, reads: int, *, classes: str, churn: int):
    """A stream that actually ripens things: a board that drifts a few squares at
    a time out of a small alphabet, so runs form and commits happen."""
    grid = [list(row) for row in parse_placement(START_PLACEMENT)]
    for i in range(reads):
        for _ in range(rng.randrange(churn + 1)):
            r, c = rng.randrange(8), rng.randrange(8)
            grid[r][c] = rng.choice(classes)
        yield encode_placement(grid), i * 1.0


class _UnguardedStabilizer(BoardStabilizer):
    """The pre-fix rule: commit every ripe square, coherence be damned."""

    def _commit(self, grid, ripe, contested, ts, rules):
        return ripe


def _stabilizer(window: float) -> BoardStabilizer:
    return BoardStabilizer(SettleConfig(seconds=window))


#: Two windows, because they exercise different commit paths: 1.0 s reaches the
#: deadline fall-through, 1.5 s completes drags. Both do partial and empty commits.
_PROPERTY_WINDOWS = (1.0, 1.5)

#: Seeds for the property sweep — enough streams that every commit path fires.
_PROPERTY_SEEDS = 40


def _excess_rises(build) -> bool:
    """Does any seeded stream ever make the settled board less possible?"""
    import random

    from board_reader_inference.stream.stabilize import _excess

    for window in _PROPERTY_WINDOWS:
        for seed in range(_PROPERTY_SEEDS):
            rng = random.Random(seed)
            stab = build(window)
            previous = None
            for placement, ts in _noisy_stream(rng, 60, classes="pqkPQK1", churn=3):
                excess = _excess(parse_placement(stab.observe(placement, ts)), "standard")
                if previous is not None and excess > previous:
                    return True
                previous = excess
    return False


def test_the_excess_never_rises_over_a_noisy_stream():
    """The invariant, over streams that really do commit — the earlier version of
    this test drew from all 13 classes at 0.4 s spacing, so nothing ever ripened
    and it passed against a stabilizer with no coherence check at all."""
    assert not _excess_rises(_stabilizer)


def test_that_property_test_can_actually_fail():
    """The mutation: strip the coherence guard and the invariant must break, or
    the test above is measuring nothing."""
    assert _excess_rises(lambda w: _UnguardedStabilizer(SettleConfig(seconds=w)))


def test_the_property_stream_exercises_every_commit_path():
    """Guard the guard: if the generator stops ripening squares the invariant
    test goes quiet without failing, which is how the first version of it came
    to pass against no coherence check at all."""
    import random

    seen: set[str] = set()
    original = BoardStabilizer._commit

    def spy(self, grid, ripe, contested, ts, rules):
        taken = original(self, grid, ripe, contested, ts, rules)
        if ripe:
            if taken == ripe:
                seen.add("uncontested")
            elif set(taken) == set(contested) and len(contested) > len(ripe):
                seen.add("deadline")
            elif len(taken) > len(ripe):
                seen.add("drag")
            elif taken:
                seen.add("partial")
            else:
                seen.add("empty")
        return taken

    BoardStabilizer._commit = spy
    try:
        for window in _PROPERTY_WINDOWS:
            for seed in range(_PROPERTY_SEEDS):
                stab = BoardStabilizer(SettleConfig(seconds=window))
                for p, t in _noisy_stream(random.Random(seed), 60,
                                          classes="pqkPQK1", churn=3):
                    stab.observe(p, t)
    finally:
        BoardStabilizer._commit = original
    assert seen == {"uncontested", "drag", "deadline", "partial", "empty"}


# --------------------------------------------------------------------------- #
# Bounded staleness, and the order the drag picks partners in
# --------------------------------------------------------------------------- #

#: d8 q, e8 empty — the partner square flickering below the quorum.
E8_EMPTY = "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_a_blocked_square_is_not_starved_forever():
    """Coherence alone starves: settled holds `k` on d8, every read says `q`, and
    committing d8 alone would compose two black queens while e8 flickers below
    the quorum and can never be dragged. Before the deadline this held `k` for
    200 reads / 260 s. It must now land within one window to ripen plus
    STALE_WINDOWS to time out."""
    from board_reader_inference.stream.stabilize import STALE_WINDOWS

    window = 2.5
    stab = BoardStabilizer(SettleConfig(seconds=window))
    stab.observe(SWAP, 0.0)
    first_disagreement = 1.0
    landed = None
    for i in range(200):
        ts = first_disagreement + i * 1.3
        settled = stab.observe(START_PLACEMENT if i % 2 else E8_EMPTY, ts)
        if parse_placement(settled)[0][3] == "q":  # d8
            landed = ts
            break
    assert landed is not None, "d8 never landed: the debounce starved"
    bound = window * (1 + STALE_WINDOWS) + 1.3  # + one read interval to notice
    assert landed - first_disagreement <= bound


def test_the_deadline_still_refuses_a_less_possible_read():
    """The escape is a fall-through to the read, not a surrender: a read that is
    itself less possible than the settled board never lands, deadline or not."""
    stab = BoardStabilizer(SettleConfig(seconds=0.5))
    stab.observe(START_PLACEMENT, 0.0)
    for i in range(30):
        settled = stab.observe(PHANTOM_B6, 1.0 + i)
    assert settled == START_PLACEMENT
    assert _pending(PHANTOM_B6, settled) == ("b6",)


def test_the_drag_takes_the_best_evidenced_partner():
    """Two partners could each relieve the same blocked commit; the one with more
    consistent reads is taken. Board order would pick b7, evidence order picks
    g7 — this is the only test that tells those two apart."""

    # a6 grows a black rook (17 black pieces, 9 promotions): blocked. Removing a
    # black *pawn* is what relieves both counts, so b7/g7 are the two partners.
    def board(a6: str, b7: str, g7: str) -> str:
        grid = [list(row) for row in parse_placement(START_PLACEMENT)]
        grid[2][0] = a6  # a6
        grid[1][1] = b7  # b7
        grid[1][6] = g7  # g7
        return encode_placement(grid)

    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(board("r", "p", "p"), 1.0)  # a6 rook: run starts
    stab.observe(board("r", "p", "empty"), 2.0)  # g7 goes: 1 read
    stab.observe(board("r", "p", "empty"), 2.5)  # g7: 2 reads
    stab.observe(board("r", "empty", "empty"), 3.0)  # g7: 3, b7 joins: 1
    settled = stab.observe(board("r", "empty", "empty"), 3.6)  # a6 ripe; g7 4, b7 2
    grid = parse_placement(settled)
    assert grid[2][0] == "r"  # the blocked square landed
    assert grid[1][6] == "empty"  # dragged: g7, the better-evidenced partner
    assert grid[1][1] == "p"  # b7 was not touched, though it also would have done


def test_the_deadline_takes_only_what_it_needs():
    """The escape waives the quorum, not the debounce: it adds partners in
    evidence order and stops as soon as the board is no worse, so a square that
    merely happens to be in flux at that moment is not swept up with it."""

    def board(e8: str, h2: str) -> str:
        grid = [list(row) for row in parse_placement(SWAP)]
        grid[0][3], grid[0][4], grid[6][7] = "q", e8, h2
        return encode_placement(grid)

    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(SWAP, 0.0)  # settled: d8 k, e8 q
    settled = None
    for i in range(9):
        # d8 reads q on every read and goes ripe at 3.5 s, but committing it
        # alone composes two black queens; e8 flickers below the quorum, so only
        # the 9.0 s deadline can land it. h2 goes into flux on that very read.
        settled = stab.observe(
            board("k" if i % 2 == 0 else "q", "empty" if i == 8 else "P"), 1.0 + i
        )
    grid = parse_placement(settled)
    assert grid[0][3] == "q"  # the blocked square landed, on the deadline
    assert grid[0][4] == "k"  # with the one partner that unblocked it
    assert grid[6][7] == "P"  # h2 was in flux on the same read, and left alone
