"""The numbers every policy shares: the edge's burst thresholds and the read floor.

Two of them (:data:`ENTER_BURST` and :data:`EXIT_BURST`, :data:`EXIT_HOLD_S`) are
mirrored value for value from the edge's forwarder, because a policy that
segments bursts differently from the edge that forwarded them would be reasoning
about a stream nobody sent.
"""

from __future__ import annotations

#: Stability under this starts a burst; it must climb back over
#: :data:`EXIT_BURST` and stay there for :data:`EXIT_HOLD_S` to end one.
#: Mirrored value for value from the browser edge's ``AdaptiveForwarder``
#: (``packages/viewer/src/lib/edge/motion.ts``) — those are the thresholds that
#: decided which frames were *sent*, so segmenting the arriving stream by any
#: other pair would describe bursts that never happened.
ENTER_BURST = 0.7
EXIT_BURST = 0.85
#: Sustained calm before the *edge* declares a burst over — the phone's
#: ``EXIT_HOLD_S`` in ``motion.ts`` and the Python edge's
#: ``ForwarderConfig.exit_hold_s``, mirrored here as :class:`BurstTriggered`'s
#: default. The edge's hold only decides how long it keeps sending at burst rate:
#: through this hold and then its 1 s tail. The shipped node holds for a shorter
#: 0.3 s (``exit_hold_s`` in :data:`DEFAULT_POLICY_SPEC`), so its burst-end run
#: lands inside that dense stretch and a move settles ~0.4 s sooner. The hold has
#: a floor: at 0.1 s a burst ends in the pause between two strides of one move
#: (a lifted king settled on live take 2), which is what this hold exists for
#: (docs/reports/2026-11-25-latency-round3.md).
EXIT_HOLD_S = 0.7

#: Float slack, so an interval due "exactly now" is not missed by a rounding bit
#: (the same guard the forwarder's rate check uses).
DUE_EPS = 1e-6

#: The keepalive floor every policy carries unless told otherwise. Three seconds
#: is two settle windows: slow enough that a resting board is nearly free, fast
#: enough that a scheduler which has gone blind still assembles a settled board
#: within a handful of seconds.
DEFAULT_KEEPALIVE_S = 3.0
