"""``core.sim.SAMPLE_GAME_PLACEMENTS`` is the game its comment names, move by move.

Checked with python-chess here, at the workspace root, so ``board-reader-core``
keeps no runtime dependency on it (THIRD-PARTY.md §5).
"""

from __future__ import annotations

import chess

from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS

#: Légal's Mate, as the comment above SAMPLE_GAME_PLACEMENTS spells it.
LEGALS_MATE = "e4 e5 Bc4 d6 Nf3 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#"


def test_the_sample_game_is_legals_mate_played_out():
    board = chess.Board()
    played = [board.board_fen()]
    for san in LEGALS_MATE.split():
        board.push_san(san)
        played.append(board.board_fen())
    assert tuple(played) == SAMPLE_GAME_PLACEMENTS
    assert board.is_checkmate()
