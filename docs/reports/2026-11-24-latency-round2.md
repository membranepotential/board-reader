# Latency round 2: forwarder tail, 0.3 s burst-end reads, 0.5 s settle (2026-11-24)

**Question.** After the blind-meter fix ([[2026-11-24-live-latency]]), live take 2
(`recordings/20261124-204025`) settled a move ~1.0–1.3 s after its burst ended. Why, and
how much lower can it go without letting a hand settle?

**Cause.** After a burst the phone's `AdaptiveForwarder` fell back to its 1 fps rest
rate, so the node's burst-end reads (3 × 0.6 s) had no frame for ~1.04 s; then the
1.0 s / 2-read settle window.

**Change (commit 5c1b9a0).** Both forwarders (Python and the phone's `motion.ts`) keep
the burst rate for `tail_s = 1 s` after a burst. The node reads a burst end three times
0.3 s apart and settles on a 0.5 s window, so a move settles ~0.6 s after its burst ends.

**Method.** `scratch/experiments/latency2/` (gitignored): the recorded stream run
through the real `AdaptiveForwarder`, the old 1 fps gaps filled with synthetic rest
frames that reuse the nearest recorded read. On recorded frames alone it reproduces the
real replays event for event. Grid: spacing 0.2–0.6 s × 2–4 reads × settle 0.3–1.0 s ×
min reads 2–3, on recorded frames ("real") and with the tail simulated ("tail").

| config | gate perfect (real) | gate perfect (tail) | hand leaks (real/tail) | settled ≤ 1 s after the hand leaves (tail) | mean latency (tail) | held-out perfect (real) |
|---|---|---|---|---|---|---|
| before: 3 × 0.6 s, 1.0 s | 138/154 | 155/181 | 0/0 | 8 of 27 | 2.06 s | 107/202 |
| **shipped: 3 × 0.3 s, 0.5 s** | **140/156** | 165/192 | **0/0** | **11 of 27** | **1.70 s** | **123/212** |
| 3 × 0.4 s, 0.4 s | 140/156 | 165/191 | 0/0 | 11 | 1.63 s | 121/208 |
| 4 × 0.3 s, 0.4 s | 151/167 | 183/212 | 0/0 | 10 | 1.67 s | 141/239 |

Real `make gate GATE_NAME=sp03-settle05` vs `ka2-tuned`: raw perfect 142/154 → 145/156,
settled perfect 138/154 → 140/156; the 16 settled-wrong boards are the same 11 lag and
5 partly adopted moves as before, no hand.

**Findings.**
- A settle window at or below the burst-end spacing (two reads span it) let a hand settle
  in 19 of 30 swept configs; a three-read window in 2 of 30. Keep
  `spacing < window ≤ (reads − 1) × spacing` (now a rule in `.claude/rules/stream.md`).
- Latency splits in two groups: moves whose burst fired settle in 0.3–1.0 s; moves the
  meter missed wait for the 2 s keepalive (2–4 s). A median between them means little.
- Live take 2, hand-free frame → settled (projected for the new config): e4 1.66 → 0.72 s,
  Nc3 1.77 → 0.83 s, Nb1 2.71 → 1.32 s. **Measured only after the next live take.**

**Open — not fixed by any settle setting.**
- **A hand resting still** over the board reads as rest to the motion meter: on live take 2
  a hand held over f6 for ~2.5 s (frames 146–152) settled under every config, including the
  one that ran live. Only a direct hand signal would stop it.
- **Missed bursts:** d3 and Nc3 opened no burst; hands at the far board edge often do not
  register, and the camera seems to move near frame 203.
