"""Boards the settle tests share: a few placements, as FEN piece-placement strings.

Imported by name by ``test_stabilizer.py`` and ``test_balance.py`` (the same arrangement
as ``tests/measurement/trial/_trial_builders.py``), so keep the basename unique across
the workspace.
"""

#: START_PLACEMENT after 1.e4 — one real move, two squares changed.
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: The other flicker a hand makes, and a possible board: b8 briefly occluded.
GONE_B8 = "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: The k/q swap the 08-27 control take coin-flipped into: d8 k, e8 q.
#: Possible on its own — it is only ever *half* of it that is not.
SWAP = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
#: E4 with a hand over b8: the move in plain view, one piece hidden elsewhere.
E4_GONE_B8 = "r1bqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
#: E4 with a hand over a2: the hidden piece is a white pawn, like the moved one.
E4_GONE_A2 = "rnbqkbnr/pppppppp/8/8/4P3/8/1PPP1PPP/RNBQKBNR"
#: After 1.e4 d5, then 2.exd5, and that capture half done (the pawn in the hand).
E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR"
EXD5 = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR"
E4_LIFTED = "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPP1PPP/RNBQKBNR"
