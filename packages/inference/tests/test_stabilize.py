"""Tests for the per-square debounce behind ``PositionEvent.settled``."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASSES,
    START_PLACEMENT,
    encode_placement,
    parse_placement,
)

from board_reader_inference.prior import tally, violation
from board_reader_inference.stabilize import BoardStabilizer, SettleConfig

#: START_PLACEMENT after 1.e4 — one real move, two squares changed.
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: A phantom: the trial's hallucinated black rook on b6. Note it is *impossible*
#: — 17 black pieces — which is the whole reason the prior repair runs first.
PHANTOM_B6 = "rnbqkbnr/pppppppp/1r6/8/8/8/PPPPPPPP/RNBQKBNR"
#: The other flicker a hand makes, and a possible board: b8 briefly occluded.
GONE_B8 = "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: The k/q swap the 08-27 control take's repair coin-flipped into: d8 k, e8 q.
#: Possible on its own — it is only ever *half* of it that is not.
SWAP = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: The same reading with d8 misread as empty: e8 still k, d8 no longer q.
D8_EMPTY = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: Two black kings — what composing halves of START and SWAP produces.
TWO_KINGS = "rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def is_possible(placement: str) -> bool:
    """Does this board satisfy the prior's hard constraints? (the invariant)"""
    index = {symbol: i for i, symbol in enumerate(CLASSES)}
    labels = np.array([[index[cls] for cls in row] for row in parse_placement(placement)])
    return violation(tally(labels, oriented=False)).satisfied


def test_first_read_is_adopted_wholesale():
    settled = BoardStabilizer().observe(START_PLACEMENT, 0.0)
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ()


def test_a_real_move_is_adopted_once_it_spans_the_window():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(E4, 1.0).placement == START_PLACEMENT
    held = stab.observe(E4, 2.0)  # consistent, but only 1.0s into the window
    assert held.placement == START_PLACEMENT
    assert held.pending == ("e4", "e2")  # board order: rank 8 first
    done = stab.observe(E4, 3.6)  # 2.6s of unbroken E4: adopted
    assert done.placement == E4
    assert done.pending == ()


def test_wall_clock_alone_is_not_evidence():
    """One read followed by silence must not be adopted on its second sighting
    hours later — at least SETTLE_MIN_READS *consistent* reads are required."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5, min_reads=3))
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(E4, 10.0)
    assert stab.observe(E4, 20.0).placement == START_PLACEMENT  # 2 reads < 3
    assert stab.observe(E4, 20.5).placement == E4


def test_a_flicker_never_reaches_the_settled_board():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    flickered = stab.observe(GONE_B8, 1.0)
    assert flickered.placement == START_PLACEMENT
    assert flickered.pending == ("b8",)
    recovered = stab.observe(START_PLACEMENT, 1.5)
    assert recovered.placement == START_PLACEMENT
    assert recovered.pending == ()
    # the run restarted: the flicker's clock does not carry over
    stab.observe(GONE_B8, 2.0)
    assert stab.observe(GONE_B8, 4.0).placement == START_PLACEMENT


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
        assert stab.observe(SWAP if i % 2 else START_PLACEMENT, float(i)).placement == (
            START_PLACEMENT
        )


def test_a_backwards_timestamp_restarts_the_run():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 100.0)
    stab.observe(E4, 101.0)
    stab.observe(E4, 90.0)  # edge clock stepped back: evidence discarded
    assert stab.observe(E4, 92.0).placement == START_PLACEMENT
    assert stab.observe(E4, 93.0).placement == E4  # 3.0s from the 90.0 restart


def test_zero_window_still_needs_min_reads():
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(GONE_B8, 1.0).placement == START_PLACEMENT
    assert stab.observe(GONE_B8, 2.0).placement == GONE_B8


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        SettleConfig(seconds=-1.0)
    with pytest.raises(ValueError):
        SettleConfig(seconds=float("nan"))
    with pytest.raises(ValueError):
        SettleConfig(min_reads=0)
    with pytest.raises(ValueError):
        SettleConfig(stale_windows=0)


def test_settle_config_from_env_keeps_unset_knobs_at_their_defaults():
    assert SettleConfig.from_env({}) == SettleConfig()
    assert SettleConfig.from_env({"BR_SETTLE_SECONDS": "1.5", "BR_SETTLE_MIN_READS": ""}) == (
        SettleConfig(seconds=1.5)
    )
    assert SettleConfig.from_env(
        {"BR_SETTLE_SECONDS": "0.5", "BR_SETTLE_MIN_READS": "1", "BR_SETTLE_STALE_WINDOWS": "3"}
    ) == SettleConfig(seconds=0.5, min_reads=1, stale_windows=3)


@pytest.mark.parametrize(
    "env",
    [{"BR_SETTLE_SECONDS": "fast"}, {"BR_SETTLE_MIN_READS": "1.5"}, {"BR_SETTLE_STALE_WINDOWS": "0"}],
)
def test_a_malformed_settle_env_fails_loudly(env):
    with pytest.raises(ValueError):
        SettleConfig.from_env(env)


def test_settle_config_names_itself_for_a_replay_directory():
    assert SettleConfig().is_default
    config = SettleConfig(seconds=1.5, min_reads=1, stale_windows=3)
    assert not config.is_default
    assert config.slug == "settle1.5s-1r-3w"
    assert config.to_dict() == {"settle_seconds": 1.5, "settle_min_reads": 1, "stale_windows": 3}


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
            if stab.observe(board, ts).placement != START_PLACEMENT:
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
    its own 97.2 % prior input, 0/253 perfect boards, 0/7 truth segments ever
    settled — because 69 % of its settled events carried two black kings, a board
    no read ever claimed and no truth segment can equal.
    """
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    settled = [s.placement for s in _swap_stream(stab, START_PLACEMENT)]
    assert settled[:3] == [SWAP, SWAP, SWAP]
    assert settled[3] == START_PLACEMENT  # both halves of the swap, together
    assert all(is_possible(board) for board in settled)


def test_a_sub_quorum_square_is_not_dragged_and_the_commit_is_held():
    """Dragging is not a free pass: the dragged square owes ``settle_min_reads``
    consistent reads. Without them the ripe displacement waits instead of
    landing on a board nobody read."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    boards = _swap_stream(stab, D8_EMPTY)  # d8's run restarts on the last read
    assert [b.placement for b in boards] == [SWAP] * 4  # ripe, held, still possible
    assert boards[-1].pending == ("d8", "e8")  # both in flux, evidence kept
    # One more read gives d8 its quorum, and now the pair moves together.
    assert stab.observe(D8_EMPTY, 3.7).placement == D8_EMPTY


def test_the_settled_board_never_becomes_less_possible():
    """The invariant, over a stream that mixes both readings of the swap with
    noise. Measured the same way over all four 08-27 takes x six read policies:
    0 impossible settled boards, against 233/337 on the control take before."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stream = [SWAP, D8_EMPTY, START_PLACEMENT, SWAP, START_PLACEMENT, START_PLACEMENT,
              D8_EMPTY, START_PLACEMENT, START_PLACEMENT, SWAP, SWAP, START_PLACEMENT]
    for i, board in enumerate(stream):
        settled = stab.observe(board, float(i) * 0.9)
        assert settled.placement != TWO_KINGS
        assert is_possible(settled.placement)


def test_an_impossible_read_is_never_adopted():
    """A hallucinated 17th piece survives the whole window and still does not
    land: the settled board may not be made less possible than it is. In the
    live path the prior repair has already removed it — this is the backstop."""
    stab = BoardStabilizer(SettleConfig(seconds=0.0))
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(PHANTOM_B6, 1.0)
    settled = stab.observe(PHANTOM_B6, 5.0)
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ("b6",)


def test_an_impossible_first_board_is_still_escapable():
    """The first read is adopted wholesale even if it is impossible (a caller
    feeding the raw argmax rather than the repair). The rule is *never less
    possible*, not *never commit*, so the stream still recovers."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    assert stab.observe(TWO_KINGS, 0.0).placement == TWO_KINGS
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6).placement == START_PLACEMENT


def test_a_wrong_first_board_is_displaced_like_any_other():
    """Wholesale adoption is not adoption forever: the square the first read got
    wrong needs the same window as every other square, and gets it."""
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(SWAP, 0.0)
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6).placement == START_PLACEMENT


def test_pending_lists_every_square_reading_differently():
    stab = BoardStabilizer(SettleConfig(seconds=2.5))
    stab.observe(START_PLACEMENT, 0.0)
    settled = stab.observe(SWAP, 1.0)
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ("d8", "e8")


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

    def _commit(self, grid, ripe, contested, ts):
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

    from board_reader_inference.stabilize import _excess

    for window in _PROPERTY_WINDOWS:
        for seed in range(_PROPERTY_SEEDS):
            rng = random.Random(seed)
            stab = build(window)
            previous = None
            for placement, ts in _noisy_stream(rng, 60, classes="pqkPQK1", churn=3):
                excess = _excess(parse_placement(stab.observe(placement, ts).placement))
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

    def spy(self, grid, ripe, contested, ts):
        taken = original(self, grid, ripe, contested, ts)
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
    from board_reader_inference.stabilize import STALE_WINDOWS

    window = 2.5
    stab = BoardStabilizer(SettleConfig(seconds=window))
    stab.observe(SWAP, 0.0)
    first_disagreement = 1.0
    landed = None
    for i in range(200):
        ts = first_disagreement + i * 1.3
        settled = stab.observe(START_PLACEMENT if i % 2 else E8_EMPTY, ts)
        if parse_placement(settled.placement)[0][3] == "q":  # d8
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
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ("b6",)


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
    grid = parse_placement(settled.placement)
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
    grid = parse_placement(settled.placement)
    assert grid[0][3] == "q"  # the blocked square landed, on the deadline
    assert grid[0][4] == "k"  # with the one partner that unblocked it
    assert grid[6][7] == "P"  # h2 was in flux on the same read, and left alone
