# Several boards and input spikes: latency grows, results do not (2026-11-25)

**Question.** Does the inference node keep each board's results when it serves several
boards at once or receives a burst of frames faster than it can read them? The user's
requirement: "real-time latency can increase, but the ultimate results and timestamps
should not." Results here means, per board, which frames are read, the settled board
after each read, and every event's `utc` and `changed_utc`.

**Answer: it did not; now it does.** Before this change the node decided each read when
a board's worker took a frame out of its latest-wins inbox, so load changed which frames
were read. Three boards under a slow model read 8 of 17 frames per board, lost settled
changes, and moved `changed_utc` by up to 1.5 s. Now every read is decided when its frame
arrives, and the frame waits in one read queue until the recognizer is free (DESIGN
D17). On real data, two gate takes streamed at once into one node with the fp32 model
publish exactly their solo `latency3` replay, line for line, at 1x and 4x speed. The
gate is identical to `latency3`.

## What the node did before

Each board had its own latest-wins inbox (one frame) and its own worker task. The
`/ingest` handler fed every frame to the policy's `observe()`, but the worker asked
`due()` only on the frame it took out of the inbox. When the worker was busy, newer
frames displaced older ones, so a burst-end read landed on whatever frame was newest
when the worker came free, not on the frame an idle node reads. A scheduled read whose
frame was displaced was simply lost.

Measured with the new load test on the old code (3 boards, 146 frames each, flooded
into a recognizer that takes 0.03 s per read):

| board | reads alone | reads loaded | settled changes alone | loaded, flood | loaded, clumps |
|---|---|---|---|---|---|
| B0 | 17 | 8 | 7 | 7, two of them up to 1.5 s later | 4 |
| B3 | 17 | 8 | 5 | 5, two 0.2-0.3 s later | 5, all later |
| B6 | 17 | 8 | 5 | 5, two 0.3 s later | 5, all later |

421 of 438 frames were dropped from the inboxes.

## The answers to the four questions

**1. Per-board isolation.** Before: each board had its own inbox, worker, scheduler,
stabilizer, guard, ring and change clock. The recognizer (one onnxruntime session) was
shared, and reads were not serialised: each board's worker called it through
`asyncio.to_thread`, so N boards ran up to N `sess.run` calls at once, in no order. No
board could starve another (each had at most one read in flight), but load on board A
slowed board B's reads, and slower reads changed which of B's frames were read.

Now: the per-board stream state is unchanged (`_Board`: policy, ring, stabilizer, guard,
change clock, motion throttle). All boards share one read queue, served oldest first by
one worker, one recognizer call at a time. Load on A delays B's reads by A's reads that
arrived earlier, never more, and never changes which frames B reads. Starvation cannot
happen: oldest first serves every read in arrival order, and the scheduler bounds each
board's demand (three burst-end reads per move, one keepalive per 2 s). In the load test
the first 30 reads split 10/10/10 across three flooded boards.

**2. Spikes.** The scheduler saw every frame header in order before too (`observe()`
in `submit`), but `due()` was asked only on the frames the inbox kept, so a superseded
burst-end read moved to a later frame or vanished, and with it the settled sequence and
`changed_utc` (table above). Now `submit` asks `observe()` then `due()` on every frame,
as replay's `should_estimate` does, and the scheduled frame itself is queued. The
`FrameRing` is fed on submit for every frame, as before; but before, the worker took its
snapshot at read time, so a long backlog could evict the frame a back-dating needs
(`HOLD_SECONDS` = 10 s, `HOLD_FRAMES` = 160). Now the snapshot is taken at arrival, as
replay takes it. Recording still happens in the `/ingest` handler before `submit`, for
every board (`test_two_boards_on_one_socket_record_every_frame_under_a_slow_recognizer`).

**3. Timestamps.** Every event's `utc` is the read frame's `capture_ts`
(`event_from_dist`), and `changed_utc` is the stabilizer's `changed_ts` (a read's
`capture_ts`) or a held frame's `capture_ts` from the back-dating: all edge capture time.
That holds on every path: burst-end reads, keepalive reads, backstopped reads (the node
clock only decides *that* a frame is read, the event still carries its capture time) and
obscured reads (published with `obscured`, `changed_utc` unchanged). Motion updates
carry the frame's `capture_ts`. The node clock appears only in `published_ts` and in the
new drop line's `waited_s`, both by design. Since `rules` is now also taken at arrival,
a `/control` change applies from the board's next frame rather than its next read.

**4. Invariants.** Each board's events publish in arrival order, so `seq` is monotone
per board (one worker, a FIFO queue). Nothing is published out of order. Memory is
bounded: at most `READ_BACKLOG` = 32 waiting reads per board (~1.5 MB each for the
phone's crop, ~50 MB per board) plus the ring (~2.2 MB per board); the hub's subscriber
queues were already bounded (256). The GPU arena rule holds: every read has the one
input shape `preprocess_frame` produces, and now only one call runs at a time, so the
arena holds one call's activations, not one per board.

## What changed

- **Decide at arrival** (`stream/ingest.py`, `FrameIngest.submit`). `observe()`,
  `due()`, the ring snapshot and the board's `rules` all happen on arrival. The
  wall-clock backstop also moved to arrival, so a slow recognizer no longer looks like a
  stalled edge clock.
- **One read queue, oldest first, one call at a time** (`FrameIngest._next`, the one
  place that takes reads off the queue, where batching would go; `_serve`). The debounce
  runs in the same worker thread right after the read.
- **The overload rule.** A keepalive read that waited `MAX_KEEPALIVE_WAIT_S` = 5 s is
  dropped instead of read. A board with 32 reads waiting sheds its oldest keepalive read,
  and a burst-end read only when everything it has waiting is one.
  `FramePolicy.due()` now returns a `ReadKind` (`BURST_END` or `KEEPALIVE`) instead of a
  bool, so the node can tell them apart. Every drop is counted (`/healthz`
  `ingest.dropped`, `ingest.dropped_burst_end`), logged at WARNING, and written to the
  session's `events.ndjson` as a `{"type": "read_dropped", ...}` line (`drop_line`),
  which the trial scorer skips and `recorded_rules` still reads.
- **`/healthz` `ingest.queue`**: `waiting`, `max_waiting`, `served`, `wait_mean_s`,
  `wait_max_s`.
- **Removed:** the per-board inbox and `prefer_frame`, its sharpness preference. It only
  acted when the node was behind, so it made results depend on load, and the gate never
  measured it (replay never had an inbox).
- `_read_and_settle(read, ...)` became `_settle_read(reading, ...)`; the live worker and
  replay still share it.

The design passed through two variants on the way, both rejected. In the first, each
board's request in a shared queue was replaced by its next one (latest-wins). That drops
burst-end reads under load, which is the bug itself. The second served boards
round-robin. Oldest first gives the same mean wait when reads take about equally long,
a smaller worst wait, and keeps each board's order with one simple queue.

## Evidence

**Unit load test** (`packages/inference/tests/test_multiboard_load.py`, in `make test`).
Three `core.sim` games with hands over each move. A scripted `FrameRead` returns
occasional misreads, so a read on a different frame changes the lines. The real
`FrameRing` back-dates on the rendered pixels. Each board's lines through a loaded node
(flooded, or in clumps of ten per board; a recognizer at 0.01 s per read) equal its solo
run, compared as JSON (`frame_seq`, `placement`, `pending`, `utc`, `changed_utc`,
evidence). The recognizer serves reads in arrival order, and nothing is dropped.
Measured on the same scenario:

| read time | arrival | reads | identical to solo | dropped | queue wait mean / max |
|---|---|---|---|---|---|
| 0.01 s | flood | 51 | yes | 0 | 0.23 s / 0.46 s |
| 0.01 s | clumps | 51 | yes | 0 | 0.23 s / 0.46 s |
| 0.2 s | flood | 45 | no | 6 keepalive, 0 burst-end | 4.4 s / 8.7 s |

The last row is a sustained overload by construction: 45 s of capture time from three
boards arrives in one instant, about 10 s of reads. The rule then drops keepalive reads
only. The test pins this with the wait cap lowered to 0.15 s: only keepalive reads drop,
every drop is counted, and every burst-end read is still read.

**Real data, two boards on one node** (fp32 ChessQ V6 on the GTX 1650, the shipped
schedule, `serve` with recording, under the GPU lock). Takes `20260828-111421` (243
frames, 92 s) and `20260827-181802` (301 frames, 142 s) are streamed at once as boards T1
and T2: stored JPEG bytes and recorded headers, paced to each take's own capture clock
divided by the speed. Each board's recorded events are compared with the take's
`latency3` replay on `frame_seq`, `placement`, `pending`, `utc`, `changed_utc`, the read
(evidence top row) and `obscured`.

| speed | reads T1 / T2 | identical to `latency3` | settled changes T1 / T2 | dropped | queue wait mean / max | most waiting |
|---|---|---|---|---|---|---|
| 1x (real time) | 63 / 86 | yes / yes | 13 / 18 | 0 | 0.015 s / 0.41 s | 1 |
| 4x | 63 / 86 | yes / yes | 13 / 18 | 0 | 2.1 s / 4.1 s | 20 |
| 8x | 53 / 52 | no / no (from read 36 / 33) | 10 / 14 | 44 keepalive (10 / 34), 0 burst-end | 3.1 s / 5.4 s | 56 |

At 4x the two boards ask for 149 reads in 36 s (4.1 reads/s) against ~4.2 reads/s the
card sustains, so the queue builds up to 20 reads and the waits reach 4 s, but nothing
changes. At 8x they ask for ~8 reads/s, twice what the card sustains: a sustained
overload. The rule drops 44 keepalive reads and no burst-end read; each is a WARNING in
the node's log and a `read_dropped` line in `events.ndjson` (10 for T1, 34 for T2), and
`/healthz` counts them. The results then differ, as the rule allows: T1 settles 10 of its
13 changes, T2 14 of 18. The driver script was a one-off and is not committed.

**Gate** (`make gate GATE_NAME=multiboard` against `latency3`): 374/405 raw perfect,
350/405 settled perfect, the same as the base, and every event line identical on all
seven takes on (`frame_seq`, `placement`, `pending`, `utc`, `changed_utc`, the evidence
digest, `obscured`).

## Limits

- **Latency grows with boards × reads.** A read waits for every read that arrived before
  it, ~0.24 s each on the GTX 1650. At rest a board asks for one read per 2 s, so the
  card serves ~8 resting boards. Every board finishing a move at once queues three reads
  per board: the last of B boards waits ~0.7 s × B. Changes publish that much later,
  with the same `changed_utc`.
- **Results can still change under a sustained overload**, by design: past 5 s of
  waiting the node drops keepalive reads, and a dropped read can change a later settle.
  On the gate takes this costs more than it would on a live phone: most of their moves
  are first read by a keepalive, not a burst end (latency round 3), so at 8x the two
  takes lost 7 of 31 settled changes.
  It is counted, logged and recorded, never silent. Burst-end reads drop only when one
  board has 32 of them waiting (`dropped_burst_end`; 0 in every run here).
- **The wall-clock backstop and the overload rule use the node's clock**, so they are not
  replayed. On a healthy stream neither fires (`backstopped` 0, `dropped` 0 in the real
  runs above).
- **No batching yet.** One call at a time leaves the GPU idle between reads. Batching
  would go into `FrameIngest._next`, padded to a fixed batch (the arena rule).
- **A board never leaves.** `FrameIngest` keeps a board's state until `close()`. With
  board ids assigned per device this grows by ~2 MB per device that ever connected.
- **The sharpness preference is gone.** Live runs recorded before this change read a
  blur-filtered subset when the node was behind. Since this change a live run reads
  the frames a replay reads.
