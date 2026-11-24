# When did the board change? `changed_utc` against the truth boundaries (2026-11-24)

## Outcome

A position event now carries `changed_utc`: the capture time of the earliest read that
showed the settled board as it now stands (DESIGN §5, D15). On the seven gate takes it
sits a median **1.45 s** after the truth boundary (p90 2.76 s), where the `utc` of the
event that first publishes the new board sits 3.12 s after it (p90 4.32 s). A
consumer that timestamps moves (a PGN with clock times) should take `changed_utc` of
the first event that carries the new board. Since 2026-11-25 `changed_utc` is also
back-dated by pixels over the frames between reads, which brings the gate median to
1.35 s and live take 2's to 0.57 s ([Back-dating by pixels](#back-dating-by-pixels-2026-11-25)).

The other candidate, the start of the calm stretch that ended the motion burst (the
hand leaving the board), lands closer in the median (−0.66 s) but much worse in the
tail (p90 6.45 s, max 26 s), because a burst does not map to one move reliably: 13 of
54 changes had no calm start between the previous boundary and the read, and spurious
bursts give early matches. It is not used.

## Method

- The `changed-utc` gate replay (`make gate`, fp32 V6, the `serve` schedule; settled
  boards identical to `guard-only` on all 446 events). Its recorded `changed_utc`
  equals an offline `BoardStabilizer` rerun over the same reads on every event.
- A truth boundary is a segment's `from_seq` in `truth.json`, timed by that frame's
  `capture_ts`. Truth is authored from ChessQ's read of *every* frame, so a boundary is
  the first frame that shows the new position, typically as the hand leaves.
- For each boundary, the matching change is the first event after the previous
  boundary whose settled board equals the new segment's board. 54 boundaries matched;
  27 never settled on the truth board inside their segment (short segments, wrong
  settles) and are left out.
- Calm starts come from the shipped burst-end policy run over the replay's frame
  index (restamped stability where the gate restamps).
- Script: a scratch runner over `trial.read_events`, `BoardStabilizer.changed_ts` and
  the policy's burst tracking; no product code changed for the measurement.

## Results

Seconds after the truth boundary (negative = before it), over the 54 matched changes:

| instant | signed median | \|d\| median | \|d\| p90 | \|d\| max | before the boundary |
|---|---|---|---|---|---|
| `utc` of the first event with the new board | +3.12 | 3.12 | 4.32 | 6.12 | 2 |
| **`changed_utc`** (first read showing it) | **+1.45** | 1.61 | 2.76 | 5.04 | 7 |
| calm start before that read | −0.66 | 2.03 | 6.45 | 26.10 | 29 |

Why `changed_utc` still trails the boundary by about 1.5 s: the node does not read
every frame. The burst-end policy reads 0.7 s after the motion meter calms down, and
on these takes, recorded before the edge's burst tail existed, the next frame often
came only with the 1 fps keepalive. The truth boundary is the first frame of all
frames, read or not. A faster first read after a move would move `changed_utc` closer;
the event's own `utc` additionally waits for the settle window.

## Back-dating by pixels (2026-11-25)

### Outcome

`changed_utc` is now back-dated over the frames the node received but did not read
(`stream/backdate.py`, DESIGN §5). On live take 2 (20261124-221950, the phone with the
edge's 1 s burst tail) the median lag behind the truth boundary drops from **0.94 s to
0.57 s**; on the seven gate takes from 1.46 s to 1.35 s (|d| median 1.56 → 1.44 s).
Settled and raw boards are unchanged: the `backtrack` gate prints 341/361 raw and
314/361 settled perfect, every cell equal to `guard-only`, and each take's
`events.ndjson` equals the `changed-utc` replay line for line on (`frame_seq`,
`placement`, `pending`, `evidence`). No recognizer read is added: 0 extra reads per
move, 0 GPU time. The target of about one frame interval (0.1 s on take 2) is not
reached; the rest of the lag is a hand still on the moved piece (below).

### How it works

Every arriving frame leaves a grey copy, 160 px wide, blurred and scaled to a fixed
median against the phone's auto-exposure, in a per-board ring (10 s of capture time,
at most 160 frames). When the settled board changes, `ChangeClock` takes two
reference frames from its read log: the last read that still showed the old board on
the changed squares, and the read that first showed the new board (the stabilizer's
`changed_ts`). The pixels where the two references differ by more than 25 grey levels
stand for the changed squares; ChessQ reports no square coordinates, so there is no
square grid to use. A held frame between the two references counts as new when 70 %
of those pixels lie within 15 levels of the new reference. The search walks back from
the new reference and stops at the first frame that is not new, so the result is the
start of the unbroken run of new-looking frames that ends at the read. A hand over
the square matches neither reference and ends the run; a frame that looks new before
the hand has left does not count.

It runs inline in the read, right after the stabilizer, so the event that first
publishes the new board already carries the back-dated time: the field keeps its
meaning ("when the current settled board first appeared") and no second event is
needed. Replay decodes and pushes every recorded frame, read or not, to the same ring,
so the gate measures the same code; a replay-vs-live test pins that.

### Results

Seconds after the truth boundary, same method as above (a change counts only if its
`changed_utc` lies after the previous boundary, which drops one of the 54 earlier
matches, a reset on 170333 that the settled board never left; the base row is the
`changed-utc` replay, the new row `backtrack`):

| take | n | signed median | \|d\| median | \|d\| p90 | \|d\| max | before the boundary |
|---|---|---|---|---|---|---|
| 20260813-193308 | 5 | −0.31 → −0.42 | 0.31 → 0.42 | 1.83 → 2.04 | 2.08 → 2.08 | 3 → 5 |
| 20260827-175539 | 3 | +2.28 → +1.92 | 2.28 → 1.92 | 2.66 → 2.59 | 2.76 → 2.76 | 0 → 0 |
| 20260828-111421 | 9 | +0.00 → +0.00 | 1.04 → 1.04 | 2.10 → 2.10 | 2.60 → 2.60 | 2 → 2 |
| 20260827-181802 | 8 | +1.29 → +1.29 | 1.29 → 1.29 | 2.26 → 2.26 | 2.76 → 2.76 | 1 → 1 |
| 20260827-163352 | 5 | +1.66 → +1.66 | 1.66 → 1.66 | 2.22 → 1.89 | 2.39 → 1.97 | 0 → 0 |
| 20260827-170333 | 9 | +2.04 → +1.68 | 2.04 → 1.68 | 2.81 → 2.81 | 3.00 → 3.00 | 0 → 0 |
| 20260827-172736 | 14 | +1.98 → +1.62 | 1.98 → 1.62 | 3.18 → 2.77 | 5.04 → 4.92 | 0 → 0 |
| **gate, pooled** | 53 | **+1.46 → +1.35** | 1.56 → 1.44 | 2.76 → 2.74 | 5.04 → 4.92 | 6 → 8 |
| **take 2** | 10 | **+0.94 → +0.57** | 0.94 → 0.57 | 1.93 → 1.90 | 8.11 → 7.80 | 0 → 0 |

The gate takes gain little although several hold dense frames during a move: the
burst there is the hand, and the frames between the hand leaving and the first read
are few. The two new early matches are both on 193308 (boundaries 285 and 330, now
0.21 s and 0.42 s early, 2–4 frames). Those frames hold no hand, and the change is a
single square; either the pixels see the piece land before ChessQ reads it, or the
boundary is a few frames late. They are not explained here.

On take 2 the remaining lag is the hand. The truth boundary is the first frame ChessQ
reads as the new board, often with the fingers still on the piece; the pixels wait
until the hand has left. Per boundary (before → after): 1.e4 1.04 → 0.21, the king set
on g5 0.62 → 0.21, the reset 0.31 → 0.21; 1...e5 0.42, 2...Nf6 0.94, 3...d5 1.04 and
4...Nxe4 0.73 stay where the hand stays on the piece; 2.Nc3 is 0.00 already (the move
fell between two 1 fps keepalive frames, the motion meter saw no burst); the knight
stood back on e4 settles 7.8 s late because a hand then rests on the board for 6 s,
and the pixels stop at it.

**One confirming ChessQ read was evaluated offline and is not needed.** Checking the
candidate frame against ChessQ's cached read of every frame: 30 of 41 back-dated
candidates read as the new board on the changed squares, and rejecting the other 11
changes no matched boundary's number (pooled 1.35 s either way, one "before" fewer).
The 11 include frames the truth itself counts as new (172736's 4...O-O and Re8). So
there is no confirm read, and no GPU cost.

**Cost**, re-driven offline over the same replays (equal to the replay's output on
every line): the grey copy costs 0.6 ms per arriving frame on the phone's 960 × 520
crop (1.1 ms on the old 960 × ~780 takes), run on the event loop in
`FrameIngest.submit`; the search costs 0.27 ms per settled change in the worker
thread (~20 µs per searched frame). The ring held at most 1.3 MB on take 2 and
2.1 MB (its 160-frame cap) on the old takes. Back-dating moved 9 of 21 settled
changes on take 2 and 29 of 89 on the gate takes.

### Take 2's truth

`recordings/20261124-221950/truth.json` (git-ignored) has 14 segments, authored from
ChessQ's read of every frame (`scratch/experiments/changed-utc/`) and placements pushed
with python-chess: a middlegame (0), the pieces reset to the start by hand (68), 1.e4
(90) e5 (117) 2.Nc3 (144) Nf6 (203) 3.f4 (246) d5 (247) 4.fxe5 (256) Nxe4 (258), then
two tests that are not moves: the black king set on g5 (383) and back on e8 (436), and
the e4 knight laid on its side off the squares (489, e4 empty) and stood back (501).
Frames 510–572 hold a resting hand over the board on the Nxe4 position. Every boundary
was checked by eye on the JPEGs; every long ChessQ run agrees with its segment. The
live run scores 69/86 raw and 72/86 settled perfect boards (98.3 % / 99.0 % per
square).

Take 2 does not join the gate: it holds two staged non-moves (an illegal king, a
lying knight), and adding it would change the gate's 361-board totals that the next
candidates (the hand-energy gate) are compared against. Its replays sit beside the
gate's (`replays/changed-utc`, `replays/backtrack`) for the changed_utc measurement.
Note that 16 frames (seq 597–612) at rest on the same position were appended to the
recording on 2026-11-25 00:15, two hours after the take, so the `backtrack` replay
holds 117 events against 109 in `changed-utc`; every boundary lies before them.

Scripts: `scratch/experiments/changed-utc/` (`analyze.py`, `compare.py`, `proto.py`,
`cost.py`, `cache_reads.py`, `build_truth.py`, `sheet.py`).
