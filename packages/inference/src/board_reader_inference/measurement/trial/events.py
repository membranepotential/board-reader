"""The events a recording published, parsed down to what scoring needs.

Three line generations are read (:func:`_boards`), and a recording made before the
node settled gets its settled board **derived** here, by replaying the same
:class:`~board_reader_inference.stream.settle.stabilizer.BoardStabilizer` the gateway now
runs live; the report says so, because a replayed number is not a recorded one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from board_reader_core.protocol import GRID, encode_placement
from board_reader_core.recording import EVENTS_NAME, Recording

from board_reader_inference.stream.settle.stabilizer import BoardStabilizer

SQUARES = GRID * GRID


@dataclass(frozen=True)
class RecordedEvent:
    """One line of ``events.ndjson``, parsed down to what scoring needs.

    ``square_dist`` is stripped by the writer; what survives it is ``evidence``,
    the top-2-per-square digest (:func:`~board_reader_core.protocol.square_evidence`).
    Sessions recorded before that digest existed carry ``None`` there and are
    scored exactly as before — the boards alone.

    ``read`` is the read's own board and ``settled`` the board the stream settled
    on, or ``None`` for a recording made before the node settled. Three line
    generations are read (:func:`_boards`).
    """

    board_id: str
    seq: int
    frame_seq: int
    capture_ts: float
    published_ts: float
    confidence: float
    read: str
    settled: str | None
    #: 64 squares in board order, each a tuple of (class, probability) pairs
    #: ordered most-likely first. ``None`` when the session recorded no digest.
    evidence: tuple[tuple[tuple[str, float], ...], ...] | None = None

    @property
    def latency(self) -> float | None:
        """Capture-to-publish seconds, or ``None`` if the session did not stamp both."""
        if self.capture_ts <= 0.0 or self.published_ts <= 0.0:
            return None
        return self.published_ts - self.capture_ts


def read_events(recording: Recording, *, board_id: str | None = None) -> list[RecordedEvent]:
    """Every published event in a recording, in the order it was published.

    Unparseable lines are skipped like :meth:`Recording.frames` skips them — a
    session killed mid-write is still worth scoring.
    """
    path = recording.path / EVENTS_NAME
    if not path.exists():
        return []
    out: list[RecordedEvent] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                evidence = _parse_evidence(d.get("evidence"))
                read, settled = _boards(d, evidence)
                event = RecordedEvent(
                    board_id=str(d["board_id"]),
                    seq=int(d["seq"]),
                    frame_seq=int(d["frame_seq"]),
                    capture_ts=float(d.get("capture_ts", 0.0)),
                    published_ts=float(d.get("published_ts", 0.0)),
                    confidence=float(d.get("confidence", 0.0)),
                    read=read,
                    settled=settled,
                    evidence=evidence,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if board_id is not None and event.board_id != board_id:
                continue
            out.append(event)
    return out


def _parse_evidence(block: object) -> tuple[tuple[tuple[str, float], ...], ...] | None:
    """The recorded top-k digest, or ``None`` if this line has none worth reading.

    Anything malformed is treated as absent rather than raised: the digest is
    diagnostic, and a session is still fully scoreable without it.
    """
    if not isinstance(block, list) or len(block) != SQUARES:
        return None
    squares: list[tuple[tuple[str, float], ...]] = []
    for entry in block:
        if not isinstance(entry, list) or not entry:
            return None
        pairs: list[tuple[str, float]] = []
        for pair in entry:
            if not isinstance(pair, list) or len(pair) != 2:
                return None
            try:
                pairs.append((str(pair[0]), float(pair[1])))
            except (TypeError, ValueError):
                return None
        squares.append(tuple(pairs))
    return tuple(squares)


def _boards(
    line: dict, evidence: tuple[tuple[tuple[str, float], ...], ...] | None
) -> tuple[str, str | None]:
    """``(read, settled)`` of one recorded line. Raises ``KeyError``/``ValueError``.

    * Since 2026-11-24 an event carries one board: ``placement`` is the settled
      board, ``pending`` is always there, and the read is the argmax of
      ``square_dist``, which the line keeps as the top class of each square's
      ``evidence``.
    * Before that, ``placement`` was the read, and the settled board rode in a
      ``settled`` block.
    * Before the node settled at all, there is no ``settled`` block: the settled
      row is then derived (:func:`derive_settled`).
    """
    placement = str(line["placement"])
    if "pending" in line:
        if evidence is None:
            raise ValueError("a one-placement line without its evidence has no read")
        top = [square[0][0] for square in evidence]
        read = encode_placement([top[r * GRID : (r + 1) * GRID] for r in range(GRID)])
        return read, placement
    block = line.get("settled")
    settled = block.get("placement") if isinstance(block, dict) else None
    return placement, (str(settled) if isinstance(settled, str) else None)


def derive_settled(events: list[RecordedEvent]) -> list[str]:
    """Replay :class:`BoardStabilizer` over a board's events, in order.

    What the gateway does live, for recordings that predate it doing so. Fed the
    read, exactly as the live path feeds it (see :mod:`~board_reader_inference.stream.settle.stabilizer`), and
    fed **every** event including the ones scoring will exclude — the gateway saw
    those too, and a debounce that skipped them would settle differently.
    """
    stabilizer = BoardStabilizer()
    return [stabilizer.observe(e.read, e.capture_ts) for e in events]
