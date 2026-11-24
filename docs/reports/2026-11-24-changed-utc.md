# When did the board change? `changed_utc` against the truth boundaries (2026-11-24)

## Outcome

A position event now carries `changed_utc`: the capture time of the earliest read that
showed the settled board as it now stands (DESIGN §5, D15). On the seven gate takes it
sits a median **1.45 s** after the truth boundary (p90 2.76 s), where the `utc` of the
event that first publishes the new board sits 3.12 s after it (p90 4.32 s). A
consumer that timestamps moves (a PGN with clock times) should take `changed_utc` of
the first event that carries the new board.

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
