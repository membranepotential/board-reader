# Partial commit under a hand: may an obscured read commit a complete move? (2026-11-25)

**Question.** In the live trial 20261125-024330 the board did not update until the
hand had left the frame. Can the node update the unobstructed part of the board while
a hand is still in view?

**Answer.** Yes. The energy gate (DESIGN D16) blocked every commit while
a read was obscured. A narrower rule keeps most of its protection: an obscured read may
commit a **balanced piece group**, and nothing else. On the trial that raised the
question, 9 moves show 0.3 to 12.7 s earlier. On the seven gate takes the settled
row goes from 350 to 353 of 405 perfect. The cost is one phantom board, held for 1 s,
in nine takes. The rule shipped the same day (D16 amended, see
[Shipped](#shipped-the-rule-through-the-changed-node-2026-11-25)); the tables below
are the offline simulation that led to it, and the gate run reproduced them.

## What happened in the trial

`events.ndjson` of 20261125-024330: 273 reads, 56 obscured, 55 of them during the
140 s of play. A contact sheet of twelve obscured reads shows the cause: a hand rests
at the far edge of the board, over one to four squares, or its fingers only touch the
far rim. The moved piece's squares are in plain view, and the reads show the move
correctly and repeatedly:

| move | first read that shows it | committed | obscured reads in between |
|---|---|---|---|
| Ng8-g6 | 54.7 s | 60.8 s (g8 empty), 64.2 s (g6) | 5 |
| c2-c4 | 74.6 s | 79.2 s | 4 |
| Nb1-c3 | 86.1 s | 89.2 s | 5 |
| Bf8-b4 (set down) | 92.7 s | 99.5 s | 7 |
| Bf1-d3 | 126.3 s | 139.6 s | 9 |

So the schedule did read the board; the gate held the commits. The gate works as
designed: it judges the frame, not the square.

## The rule

A square is *ripe* when it has read the same new class across the settle window (0.3 s,
2 reads), as today. On an obscured read, group the ripe squares by piece: the squares
that lose piece P to "empty", and the squares that gain P (from empty or by capturing).
A group is **balanced** when both counts are equal. Balanced groups commit, if the
commit does not raise the board's excess (the existing guard). Everything else waits
for a clear read, as today.

Why this shape: a covered piece reads as empty, so a hand produces losses without
gains, which never balance. A move, a capture and castling balance. A piece lifted and
still in the hand does not balance, so a lifted piece's square does not empty on an
obscured read. A promotion does not balance either and waits for a clear read. The
groups are per piece, so the rule does not see moves: castling can show its king before
its rook (below), and en passant shows the capturing pawn's move while the captured
pawn, which only vanished, stays until a clear read.

This differs from the rejected "hold squares with high energy" of the
[hand signal probe](2026-11-24-hand-signal-probe.md): that variant trusted the
per-square energy to say which squares are covered (it cannot: a covered square looks
normal) and committed half-moves.

## Results

`tools/probes/partial_commit_sim.py` re-runs the settle debounce over the recorded reads
of each take's `replays/latency3` (raw board, capture time, obscured flag). `base` is
the debounce as it shipped before (an obscured read commits nothing); it
reproduces every recorded settled board (the seven gate takes: 350/405, as in the
gate table; the trial: 273/273). `off` is the gate switched off.

"Not-lag wrong" counts settled boards that are neither the truth nor the previous
truth. Lag is the time from a truth boundary to the first settled board that equals it.

| take | base | off | balanced |
|---|---|---|---|
| 193308 | 53/55, 2 wrong | 53/55, 2 | 53/55, 2 |
| 163352 | 20/25, 0 | 20/25, 1 | 20/25, 0 |
| 170333 | 61/69, 1 | 61/69, 1 | 61/69, 1 |
| 172736 | 112/126, 5 | 109/126, 8 | 112/126, 5 |
| 175539 | 11/15, 2 | 11/15, 2 | 11/15, 2 |
| 181802 | 54/67, 9 | 55/67, 7 | 55/67, 7 |
| 111421 | 39/48, 3 | 40/48, 5 | 41/48, 4 |
| **seven gate takes** | **350/405, 22, mean lag 2.67 s** | 349/405, 26, 2.60 s | **353/405, 21, 2.60 s** |
| 221950 (resting hand) | 91/96, 4, mean lag 1.99 s | 76/96, 20, 2.30 s | 90/96, 6, 0.93 s |

No gate take loses a perfect board. The reads where `balanced` scores worse than `base`:

- **111421 seq 117** (1 read): Bf1-c4 committed on an obscured read while the truth
  still holds the bishop on f1. Not checked by eye; it is the next move, so most likely
  the commit leads the truth boundary.
- **221950 seq 39 to 57** (5 reads): the pieces are being put back to the starting
  position by hand. `balanced` follows piece by piece; the truth file has one boundary
  at seq 68, when the hands are gone. Checked on the frames: each committed board is
  what stood on the board.
- **221950 seq 382** (1 read): a real phantom. With a hand over the centre, b8 reads
  empty and the pawn on d5 reads as a black knight, both for two reads: one loss and
  one gain of `n`, so the group balances. The next clear read, 1.0 s later, reverts it.

The resting-hand stretch of 221950 (rows 510 to 572) stays free of wrong boards.

On the trial 20261125-024330 (no truth; commit timelines printed by the script and
read by eye): every commit `balanced` makes on an obscured read is a move that was
played. Earlier than `base`: Ng8-g6 5.8 s (and as one move, not as an empty g8 first),
c2-c4 4.3 s, Nb1-c3 2.8 s, Qd1-c2 1.9 s, O-O 0.6 s, Ng1-f3 0.3 s, e2-e3 2.1 s,
Bf1-d3 12.7 s, d7-d6 3.7 s. Not helped: a piece set down from the hand (Bb4: its
source square emptied earlier, on a clear read) and White's castling, where the king
committed 3.1 s before the rook: the rook's two squares were not ripe together on
any obscured read.

## Limits

- **One phantom in nine takes is not zero.** A hand that hides a piece while another
  square reads as the same piece balances by accident. Asking three reads of an
  obscured commit removes it in the simulation (221950: 92/96, 4 not-lag wrong, all in
  the reset by hand) but gives back two of the three gained gate boards (351/405, 22
  not-lag wrong) and delays the trial's moves by 0.3 to 1.2 s; not shipped. Untested:
  refuse groups that touch a square with a high square energy (needs the per-square
  energy on `Recognition`).
- **A second lost piece of the same kind blocks the group.** A hand over a pawn holds
  back every pawn move until it leaves. That is today's behaviour, not a regression.
- **The balance test counts pieces, nothing more.** It consults no move rules, like the
  count guard (D14).

## Shipped: the rule through the changed node (2026-11-25)

`BoardStabilizer.observe` now commits `stabilize.balanced_squares` on an obscured read.
The shipped rule is the simulated one with one simplification: it needs no excess
check, because a balanced commit cannot raise a piece count (pinned by a seeded test).
The script's `balanced` rows now come from the shipped class and are unchanged.

`make gate GATE_NAME=balanced` against `latency3`:

| take | raw perfect | settled perfect |
|---|---|---|
| 20260813-193308 | 48/55 | 53/55 |
| 20260827-175539 | 15/15 | 11/15 |
| 20260828-111421 | 40/48 | 39/48 → 41/48 |
| 20260827-181802 | 60/67 | 54/67 → 55/67 |
| 20260827-163352 | 25/25 | 20/25 |
| 20260827-170333 | 68/69 | 61/69 |
| 20260827-172736 | 118/126 | 112/126 |
| **total** | 374/405 | 350/405 → 353/405 |

Live take 20261124-221950, replayed with the same schedule (`replays/balanced`
against `replays/latency3`, both `report.json`): the same 124 reads, raw unchanged
(73/96 perfect), settled perfect 91 → 90 of 96, settled per square 99.71 % → 99.25 %.
The per-square drop is the reset by hand at seq 39 to 57, where the settled board
follows the pieces while the truth holds the old position.

## Live check: take 20261125-035108 (2026-11-25)

A game played against the changed node (1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O b5
6.Bb3 Be7 7.Re1 O-O; 156 reads over 247 s, 64 obscured). No truth file; read off
`events.ndjson` and a contact sheet. The shipped rule reproduces 154 of the 156 recorded
settled boards from the recorded reads (the other two are the first two reads, where the
node still held the previous game's board); the old rule reproduces 129.

- **Every settled change is a move of that game.** No phantom, nothing reverted.
- **8 of the 18 commits landed on an obscured read** (e2-e4, Ng8-f6, both halves of
  White's castling, Bf8-e7, Rf1-e1, both halves of Black's castling), with a hand in the
  frame each time. Replayed under the old rule, the same reads show these eight 2.1 to
  21.7 s later (Ng8-f6 21.7 s, White's castling 15 s).
- **Wait from the first read that shows a change to its commit:** median 0.6 s, mean
  1.0 s, max 2.1 s, against median 1.2 s, mean 2.1 s, max 6.9 s on 20261125-024330
  under the old rule. Two different games, so the replay above is the cleaner number.
- **What still lags is the schedule, not the gate.** 6 of the 18 commits waited 2.1 s:
  the change's second read came with the 2 s keepalive, because no burst-end run was
  scheduled there (every read frame at stability 1.000). In four moves (Nb8-c6, Bf1-b5,
  Bb5-a4, Ba4-b3) the source square emptied on a clear read and the piece showed on its
  target one read later, 0.3 to 2.1 s; the old rule does the same.
- **Castling per piece, as designed:** White's king and rook committed 0.3 s apart.
  Black moved the rook about 2 s after the king, and its reads flickered under the hand,
  so the board stood half-castled for 6.9 s (4.2 s under the old rule, starting 4.8 s
  later).

## Reproduce

```bash
uv run python tools/probes/partial_commit_sim.py            # the tables + the trial's commit timelines
```
