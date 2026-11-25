"""The seam every policy implements, and the one place a header becomes two floats.

**What a policy may look at.** The header only — ``seq``, ``capture_ts``,
``stability``, ``sharpness``, ``burst_id`` — never the pixels: that is what lets a
declined frame cost nothing at all (its JPEG is never read off disk), and it is
all a scheduler needs. And **only ``capture_ts``**, never the wall clock: a policy
is replayed offline at whatever speed the GPU manages, so a decision that consults
``time.time()`` would answer differently on every run and take the replay
harness's determinism down with it.

That purity has a live cost, and the live path pays it rather than this package:
a camera whose ``capture_ts`` freezes or steps backwards stalls a capture-time
floor, so the ingest worker carries a **wall-clock backstop** on top of the
policy's own. In the policies, both clock pathologies are made to converge on the
floor rather than on either extreme — a backwards step re-anchors without becoming
due, and an unreadable interval reads as "not yet".
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol, runtime_checkable

from board_reader_core.recording import FrameEntry


class ReadKind(Enum):
    """Why a policy asked for a read, which says what losing it costs.

    The live node reads every scheduled frame, in order, however far behind it
    runs; only a sustained overload makes it shed reads, and it sheds
    :attr:`KEEPALIVE` reads first (:mod:`.ingest`).
    """

    #: Owed by a burst end: the move-complete evidence the debounce settles on.
    BURST_END = "burst-end"
    #: The floor, or a fixed-rate read: the next one says the same thing.
    KEEPALIVE = "keepalive"


@runtime_checkable
class FramePolicy(Protocol):
    """Which frames are worth an estimate.

    Stateful by design: burst end is only recognisable from the *run* of frames —
    a burst's last frame looks exactly like its first until the ones after it
    stay calm — so the policy must see every frame's header, in order.

    That is why the seam is **two** calls rather than one, and the split is what
    lets the same policy drive the offline replay and the live gateway:

    :meth:`observe`
        Fold one arriving frame's header into the burst state. Called for
        **every** frame, in capture order, exactly once each — including frames
        that are never estimated. A policy blinded to part of the stream segments
        bursts that nobody sent.

    :meth:`due`
        Decide whether *this* frame is the one to spend an estimate on, and
        record the read if so (its :class:`ReadKind`). Called once per frame,
        right after :meth:`observe`.

    Offline (:mod:`board_reader_inference.measurement.replay`) the two collapse into
    :meth:`should_estimate`. Live
    (:class:`~board_reader_inference.stream.ingest.FrameIngest`) ``submit`` calls
    both on every arriving frame, in the same order, so the node reads the frames
    a replay reads; a scheduled frame then waits in its board's read queue. The
    decision used to wait until a worker drained a latest-wins inbox, which let
    load change which frame a burst-end read landed on, and so the settled board
    and its ``changed_utc`` (docs/reports/2026-11-25-multiboard-spikes.md).

    :attr:`name` is the policy's canonical spec string: it round-trips through
    :func:`parse_policy`, so the ``policy`` field of a replay manifest is literally
    the ``--policy`` argument that reproduces that run, and ``BR_SCHEDULE`` is the
    same string again.
    """

    #: Canonical, re-parseable spec — ``"burst-end:keepalive_s=5"`` and friends.
    name: str
    #: The floor, in seconds of capture time (D3). Part of the seam because the
    #: live worker backstops it against the *wall* clock, and needs the number
    #: from whatever policy it was handed.
    keepalive_s: float

    def observe(self, capture_ts: float, stability: float) -> None:
        """Fold one arriving frame's header in. Every frame, in capture order."""

    def due(self, capture_ts: float, stability: float) -> ReadKind | None:
        """Whether to spend an estimate on this frame (its :class:`ReadKind`, or
        ``None``) — and, if so, record it.

        Not a pure query: answering a kind consumes the read (the keepalive
        clock restarts, a burst end's owed read is struck off), so it must be
        asked exactly once per frame the caller can estimate, and never for a
        frame the caller then declines to read for its own reasons.
        """

    def should_estimate(self, entry: FrameEntry) -> bool:
        """:meth:`observe` then :meth:`due` for a recorded frame — the replay path.

        Called once per frame, in seq order.
        """


def stability_of(raw: object) -> float:
    """The stability a policy reads, from whatever a header actually carried.

    The one coercion boundary, shared by the recorded path (:func:`header_signal`) and
    the live one (``ingest._schedule_signal``), because a scheduler that reasons
    about ``None`` or a NaN is a scheduler with two behaviours depending on which
    edge connected. :meth:`FramePolicy.observe` and :meth:`FramePolicy.due` are
    entitled to a finite number.

    Anything not a finite number reads as **1.0, fully at rest** — the D3 default.
    A JSON ``null`` from a hand-rolled edge, a missing field, and a NaN from a
    meter that divided by zero all mean "this stream does not tell me about
    motion", and the honest answer to that is the keepalive floor, not an
    exception in the ``/ingest`` handler and not a burst that ends spuriously
    (NaN loses every comparison, so it would slip through the hysteresis' calm
    branch and declare a move complete in the middle of a hand).
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) else 1.0


def header_signal(entry: FrameEntry) -> tuple[float, float]:
    """The two header fields a policy reads: capture time and stability."""
    return entry.capture_ts, stability_of(entry.header.get("stability"))
