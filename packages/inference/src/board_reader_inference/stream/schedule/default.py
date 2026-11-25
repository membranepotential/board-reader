"""The spec string the live gateway schedules with, and why it is what it is."""

from __future__ import annotations

#: What the live gateway schedules with unless ``BR_SCHEDULE`` says otherwise —
#: ``BurstTriggered(keepalive_s=2.0, burst_end_reads=3, spacing_s=0.3,
#: exit_hold_s=0.3)``. Only the
#: non-default parameters appear, because a spec states what was *asked for* (see
#: ``canonical_spec``); the class defaults (``DEFAULT_KEEPALIVE_S`` among them) stay
#: put so every recorded spec string still rebuilds the run it names.
#:
#: The burst ends after 0.3 s of calm, and three reads 0.3 s apart follow. Two of
#: them span ``SETTLE_SECONDS=0.3``, so a completed move settles on the burst
#: end's own evidence 0.3 s after the burst ends — provided the frames are there:
#: the edge forwarder keeps its burst rate through its own 0.7 s hold
#: (``EXIT_HOLD_S``) and then ``tail_s`` (1 s), because at the keepalive's
#: 1 fps each of these reads waited ~1 s for a frame (2026-11-24 live take 2).
#:
#: **Round 3 (2026-11-25, docs/reports/2026-11-25-latency-round3.md)** shortened
#: the hold from 0.7 s to 0.3 s and the settle window from 0.5 s to 0.3 s: live
#: take 2's median move-to-settle fell from 1.66 s to 0.94 s with no hand settling
#: and no wrong board in its resting-hand stretch. **This relies on the energy gate**
#: (``BR_OBSCURED_ENERGY``, DESIGN D16): with the gate off, the same settings let
#: hands settle (6 on the gate takes, 12 wrong boards with a hand resting on take
#: 2). The node does not fall back by itself; whoever turns the gate off must also
#: set ``BR_SCHEDULE`` and ``BR_SETTLE_SECONDS`` back to 0.7 s / 0.5 s.
#:
#: Round 2's numbers, for the 0.7 s hold and 0.5 s window:
#: Swept 2026-11-24 (round 2: spacing 0.2-0.6 s x 2-4 reads x settle 0.3-1.0 s,
#: on cached reads with the tail simulated): with the tail, the gate settles 11 of
#: 27 reached boundaries within 1 s of the hand leaving instead of 8, the real
#: gate (no tail) holds 140/156 settled perfect against 138/154, and no hand
#: settles on the gate. Before that: ``spacing_s=0.6`` against a 1.0 s window
#: (docs/reports/2026-11-24-settle-tuning.md), and ``burst_end_reads=3`` at
#: 1.3 s spacing against 2.5 s (docs/reports/2026-08-27-move-trigger-sweep.md §8).
#: ``BR_SCHEDULE=every-frame`` is the way back to reading everything.
#:
#: The keepalive is 2 s, not 3 s, since the 2026-11-24 live take: its phone's
#: meter never opened a burst, so every move waited on the keepalive and took
#: 3-4 s to settle. Replayed on cached reads, 2 s cut the gate's median
#: move-to-settle from 3.0 s to 2.0 s with no hand settling (104/119 -> 138/154
#: settled perfect, +29 % reads) and the live take's from 3.1 s to 2.1 s. 1.5 s
#: had the same medians at +48 % reads; 1.0 s let a hand settle on two gate takes
#: (a lifted b8 knight, a c8 bishop under the hand). The keepalive is the floor
#: when the meter is blind, so it is sized for that case.
DEFAULT_POLICY_SPEC = (
    "burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3,exit_hold_s=0.3"
)
