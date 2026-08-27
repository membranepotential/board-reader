"""Tests for the per-square debounce behind ``PositionEvent.settled``."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import CLASSES, START_PLACEMENT, parse_placement

from board_reader_inference.prior import tally, violation
from board_reader_inference.stabilize import BoardStabilizer

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
    stab = BoardStabilizer(settle_seconds=2.5)
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
    stab = BoardStabilizer(settle_seconds=2.5, settle_min_reads=3)
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(E4, 10.0)
    assert stab.observe(E4, 20.0).placement == START_PLACEMENT  # 2 reads < 3
    assert stab.observe(E4, 20.5).placement == E4


def test_a_flicker_never_reaches_the_settled_board():
    stab = BoardStabilizer(settle_seconds=2.5)
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
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 0.0)
    for i in range(20):  # 10s of alternation, far beyond the window
        assert stab.observe(SWAP if i % 2 else START_PLACEMENT, float(i)).placement == (
            START_PLACEMENT
        )


def test_a_backwards_timestamp_restarts_the_run():
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(START_PLACEMENT, 100.0)
    stab.observe(E4, 101.0)
    stab.observe(E4, 90.0)  # edge clock stepped back: evidence discarded
    assert stab.observe(E4, 92.0).placement == START_PLACEMENT
    assert stab.observe(E4, 93.0).placement == E4  # 3.0s from the 90.0 restart


def test_zero_window_still_needs_min_reads():
    stab = BoardStabilizer(settle_seconds=0.0)
    stab.observe(START_PLACEMENT, 0.0)
    assert stab.observe(GONE_B8, 1.0).placement == START_PLACEMENT
    assert stab.observe(GONE_B8, 2.0).placement == GONE_B8


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        BoardStabilizer(settle_seconds=-1.0)
    with pytest.raises(ValueError):
        BoardStabilizer(settle_min_reads=0)


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
    stab = BoardStabilizer(settle_seconds=2.5)
    settled = [s.placement for s in _swap_stream(stab, START_PLACEMENT)]
    assert settled[:3] == [SWAP, SWAP, SWAP]
    assert settled[3] == START_PLACEMENT  # both halves of the swap, together
    assert all(is_possible(board) for board in settled)


def test_a_sub_quorum_square_is_not_dragged_and_the_commit_is_held():
    """Dragging is not a free pass: the dragged square owes ``settle_min_reads``
    consistent reads. Without them the ripe displacement waits instead of
    landing on a board nobody read."""
    stab = BoardStabilizer(settle_seconds=2.5)
    boards = _swap_stream(stab, D8_EMPTY)  # d8's run restarts on the last read
    assert [b.placement for b in boards] == [SWAP] * 4  # ripe, held, still possible
    assert boards[-1].pending == ("d8", "e8")  # both in flux, evidence kept
    # One more read gives d8 its quorum, and now the pair moves together.
    assert stab.observe(D8_EMPTY, 3.7).placement == D8_EMPTY


def test_the_settled_board_never_becomes_less_possible():
    """The invariant, over a stream that mixes both readings of the swap with
    noise. Measured the same way over all four 08-27 takes x six read policies:
    0 impossible settled boards, against 233/337 on the control take before."""
    stab = BoardStabilizer(settle_seconds=2.5)
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
    stab = BoardStabilizer(settle_seconds=0.0)
    stab.observe(START_PLACEMENT, 0.0)
    stab.observe(PHANTOM_B6, 1.0)
    settled = stab.observe(PHANTOM_B6, 5.0)
    assert settled.placement == START_PLACEMENT
    assert settled.pending == ("b6",)


def test_an_impossible_first_board_is_still_escapable():
    """The first read is adopted wholesale even if it is impossible (a caller
    feeding the raw argmax rather than the repair). The rule is *never less
    possible*, not *never commit*, so the stream still recovers."""
    stab = BoardStabilizer(settle_seconds=2.5)
    assert stab.observe(TWO_KINGS, 0.0).placement == TWO_KINGS
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6).placement == START_PLACEMENT


def test_a_wrong_first_board_is_displaced_like_any_other():
    """Wholesale adoption is not adoption forever: the square the first read got
    wrong needs the same window as every other square, and gets it."""
    stab = BoardStabilizer(settle_seconds=2.5)
    stab.observe(SWAP, 0.0)
    stab.observe(START_PLACEMENT, 1.0)
    assert stab.observe(START_PLACEMENT, 3.6).placement == START_PLACEMENT


def test_pending_lists_every_square_reading_differently():
    stab = BoardStabilizer(settle_seconds=2.5)
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
