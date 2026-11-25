"""Frame recording — the on-disk capture of an L1->L2 stream (DESIGN.md WP-H2).

A run is only reproducible if the *input* to it was kept. This module defines the
recording format so a stream that crossed the wire once can be replayed byte for
byte later: the inference node writes it at its ``/ingest`` chokepoint, the edge
reads it back as a frame source. Like :mod:`board_reader_core.wire` it is shared
contract and holds no node-specific behaviour.

Layout — a *session* directory, append-only::

    <root>/<session_id>/manifest.json      # format, session id, start time, writer
    <root>/<session_id>/frames.ndjson      # one JSON line per frame, in arrival order
    <root>/<session_id>/jpeg/0000/000000.jpg

Each index line is the frame's **wire header verbatim** plus where its bytes went::

    {"type": "frame", "board_id": "A1", "seq": 7, "capture_ts": 1754..., ...,
     "received_ts": 1754..., "jpeg": "jpeg/0000/000007.jpg", "bytes": 48231}

Storing the header verbatim is what makes replay exact — the replayed frame is not
*reconstructed* from parsed fields, it is the same header dict and the same JPEG
bytes that arrived. Nothing is re-encoded on either side of the round trip.

One directory is one **streaming session**. An edge that reconnects restarts its
``seq`` at 0, so when a frame's ``seq`` stops advancing the writer *rolls*: it
closes the current directory's manifest (``closed_reason``) and continues in a
sibling with a fresh timestamp id. Historical recordings, written before the
recorder rolled, may still hold several sessions with colliding seqs in one
directory — which is why readers (:mod:`board_reader_inference.measurement.replay`) still
detect session boundaries themselves.

Two properties the writer maintains, both for crash-safety: the JPEG file is
written **before** the line that references it (so an index line never points at
missing or half-written bytes), and every line is flushed as it is appended (so a
killed process loses at most the frame in flight). The reader is correspondingly
forgiving: a truncated trailing line is skipped, not fatal.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from board_reader_core.wire import HEADER_KEYS, Frame, decode_jpeg

FORMAT = 1

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "frames.ndjson"
#: Published position events (one JSON line each, ``square_dist`` stripped by the
#: writer — the distribution is reproducible by replaying ``frames.ndjson``
#: through the same estimator, and at burst rate it would dominate the session's
#: size). The *outputs* of a run, beside the inputs that produced them.
EVENTS_NAME = "events.ndjson"
JPEG_DIR = "jpeg"

# JPEGs are sharded this many per subdirectory. A three-hour burst-heavy session
# is O(100k) frames, and one flat directory of that size makes every ``ls`` and
# every glob painful. The index carries each frame's relative path, so the reader
# never needs to know the sharding rule.
_SHARD = 1000


def _shard_path(index: int) -> str:
    return f"{JPEG_DIR}/{index // _SHARD:04d}/{index:06d}.jpg"


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """One ``frames.ndjson`` line: the wire header and arrival time, no pixels.

    The cheap half of :class:`RecordedFrame`. A session's index is a few hundred
    kB where its JPEGs are hundreds of MB, so anything that only needs timing,
    rates or board ids reads this and never touches the disk again.
    """

    header: dict
    jpeg_rel: str
    received_ts: float

    @property
    def board_id(self) -> str:
        return self.header["board_id"]

    @property
    def seq(self) -> int:
        return int(self.header["seq"])

    @property
    def capture_ts(self) -> float:
        return float(self.header["capture_ts"])


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    """One frame as it was recorded: the wire header, the bytes, the arrival time.

    ``header`` and ``jpeg`` are exactly what crossed the wire, so they can be
    re-sent verbatim. :meth:`frame` decodes instead, for consumers that want
    pixels (an offline estimator run, a re-rectification experiment).
    """

    header: dict
    jpeg: bytes
    received_ts: float

    @property
    def board_id(self) -> str:
        return self.header["board_id"]

    @property
    def seq(self) -> int:
        return int(self.header["seq"])

    @property
    def capture_ts(self) -> float:
        return float(self.header["capture_ts"])

    def frame(self) -> Frame:
        """Decode to the in-memory :class:`~board_reader_core.wire.Frame`.

        Only the wire's own keys are parsed: a replay marks the headers it
        restamps (``stability_restamped``), and the header model refuses keys the
        wire does not have. The sender's ``device_id`` is left out too: a replay is
        not the camera that recorded it, and under that camera's id it would stream
        into the camera's live board, or be refused there (DESIGN D18).
        """
        wire = {
            key: value
            for key, value in self.header.items()
            if key in HEADER_KEYS and key != "device_id"
        }
        return Frame.from_header(wire, decode_jpeg(self.jpeg))


def _board_seq(header: dict) -> tuple[str, int] | None:
    """The ``(board_id, seq)`` a roll decision needs, or ``None`` if unreadable."""
    try:
        return str(header["board_id"]), int(header["seq"])
    except (KeyError, TypeError, ValueError):
        return None


def _capture_ts(header: dict, now: Callable[[], float]) -> float:
    """A frame's capture time, falling back to the clock for a header without one."""
    try:
        return float(header["capture_ts"])
    except (KeyError, TypeError, ValueError):
        return now()


class FrameRecorder:
    """Append-only writer for the streaming sessions of one run.

    The session directory is created **lazily**, on the first recorded frame or
    event: a gateway that runs all day without an edge ever connecting leaves no
    empty directories behind.

    **One directory per streaming session.** An edge that reconnects restarts its
    ``seq`` at 0, so two sessions written into one directory collide: their
    frames interleave in the index under the same sequence numbers and nothing
    downstream can tell which read belongs to which take. A frame whose ``seq``
    does not advance the last one recorded for its board therefore *rolls* the
    recorder onto a sibling directory with a fresh timestamp id, closing the old
    directory's manifest with the reason. Rolling is entirely a writer-side
    concern — what a reader sees is one session per directory, so
    :class:`Recording` needs no notion of it.

    **Late events.** Frames arrive in order on one socket, but the events they
    produce do not: an estimate begun before a roll can be published after it,
    and ``seq``/``frame_seq`` cannot say which session it came from (both start
    at 0). ``capture_ts`` can — sessions are strictly time-ordered — so an
    event older than the current session's first frame is written to the
    previous directory, whose events file is kept open for exactly that. One
    level back is enough: a read takes a second or two where sessions are
    minutes apart. An event with no usable ``capture_ts`` goes to the current
    directory.

    Recording is best-effort infrastructure, never a reason to fail the stream it
    observes. An :class:`OSError` (disk full, permissions) stops the recorder and
    is reported through :attr:`stopped_reason`; it does not propagate.

    Parameters
    ----------
    root:
        Directory to create the session directories inside.
    session_id:
        Name of the *first* session directory. Defaults to a sortable
        ``YYYYmmdd-HHMMSS`` stamp, so listing a root directory lists sessions in
        chronological order; rolled directories are always named that way.
    writer:
        Free-form name of what is recording, stored in the manifest.
    max_bytes:
        Stop recording once this many JPEG bytes have been written **across the
        whole run**, every session directory included. The cap is a disk-fill
        stop, and a per-directory budget would be no cap at all — it would start
        over with every reconnect and grow without bound. The recording is
        *truncated*, never rotated: losing the tail of a session is obvious and
        recoverable, silently losing its start is neither.
    now:
        Injectable clock, for deterministic tests.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        session_id: str | None = None,
        writer: str = "unknown",
        max_bytes: int | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._now = now
        self.root = Path(root)
        self.session_id = session_id or self._stamp()
        self.path = self.root / self.session_id
        self.writer = writer
        self.max_bytes = max_bytes
        #: Run totals, across every session directory this recorder has written.
        self.frames_written = 0
        self.bytes_written = 0
        self.events_written = 0
        #: Session directories opened so far, the current one included.
        self.sessions = 1
        self.stopped_reason: str | None = None
        self._opened = False  # whether the current session dir exists on disk
        self._index: TextIO | None = None
        self._events: TextIO | None = None  # opened on the first event
        self._rolled_from: str | None = None
        # Per-session state, reset by every roll.
        self._session_frames = 0  # also the JPEG shard index within this dir
        self._session_start_ts: float | None = None  # first frame's capture_ts
        self._last_seq: dict[str, int] = {}
        # The session before the last roll: where a late event belongs.
        self._prev_path: Path | None = None
        self._prev_events: TextIO | None = None

    @property
    def active(self) -> bool:
        return self.stopped_reason is None

    def record(self, header: dict, jpeg: bytes) -> bool:
        """Append one frame. Returns whether it was written.

        ``header`` should be a frame's own :meth:`~board_reader_core.wire.Frame.header`
        rather than the raw JSON that arrived, so the recording holds values the
        wire has already validated and normalised.

        A ``seq`` that does not advance the last one recorded for this board is a
        new streaming session, and rolls the recorder onto a fresh directory.
        """
        if self.stopped_reason is not None:
            return False
        board_seq = _board_seq(header)
        try:
            if board_seq is not None and self._starts_a_new_session(*board_seq):
                self._roll()
            if self._session_frames == 0:
                # The roll boundary that routes late events (see the class docstring).
                self._session_start_ts = _capture_ts(header, self._now)
            index = self._index or self._open()
            rel = _shard_path(self._session_frames)
            blob = self.path / rel
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(jpeg)
            # Bytes first, then the reference to them: a reader that crashes into
            # the end of this file never sees a line pointing at nothing.
            line = {**header, "received_ts": self._now(), "jpeg": rel, "bytes": len(jpeg)}
            index.write(json.dumps(line) + "\n")
            index.flush()
        except OSError as e:
            self.stop(f"{type(e).__name__}: {e}")
            return False
        self.frames_written += 1
        self._session_frames += 1
        self.bytes_written += len(jpeg)
        if board_seq is not None:
            self._last_seq[board_seq[0]] = board_seq[1]
        # The cap stops the recorder *after* a frame lands, so a session may
        # overshoot by one frame but can never come out empty — a limit smaller
        # than a single JPEG would otherwise silently record nothing at all.
        if self.max_bytes is not None and self.bytes_written >= self.max_bytes:
            self.stop(f"max_bytes reached ({self.max_bytes} B)")
        return True

    def record_event(self, line: dict) -> bool:
        """Append one published position event. Returns whether it was written.

        The line lands in the directory of the session it was estimated from,
        which is the current one unless its ``capture_ts`` predates the last roll
        (see the class docstring).

        Same crash-safety contract as :meth:`record` (flushed per line) and the
        same failure semantics: an ``OSError`` stops the whole recorder, because
        a session with silently-partial outputs would be mistaken for one where
        the estimator went quiet.
        """
        if self.stopped_reason is not None:
            return False
        try:
            events = self._event_writer(line)
            events.write(json.dumps(line) + "\n")
            events.flush()
        except OSError as e:
            self.stop(f"{type(e).__name__}: {e}")
            return False
        self.events_written += 1
        return True

    def stop(self, reason: str = "closed") -> None:
        """Stop recording. Idempotent; keeps the first reason."""
        if self.stopped_reason is None:
            self.stopped_reason = reason
            self._close_manifest(reason)
        for handle in (self._index, self._events, self._prev_events):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._index = None
        self._events = None
        self._prev_events = None

    close = stop

    def stats(self) -> dict:
        """JSON-safe summary — what a health endpoint should report.

        ``frames``/``events``/``bytes`` are run totals; ``path`` and
        ``session_id`` name the directory currently being written.
        """
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "frames": self.frames_written,
            "events": self.events_written,
            "bytes": self.bytes_written,
            "sessions": self.sessions,
            "active": self.active,
            "stopped_reason": self.stopped_reason,
        }

    # ----------------------------------------------------------------- rolling

    def _starts_a_new_session(self, board_id: str, seq: int) -> bool:
        """Whether this frame's ``seq`` went backwards, i.e. a stream reconnected.

        Per board: two boards streaming into one recorder each count from 0, and
        the second board's first frame is not a new session.
        """
        last = self._last_seq.get(board_id)
        return last is not None and seq <= last

    def _roll(self) -> None:
        """Finish the current directory and continue in a fresh sibling."""
        self._close_manifest("rolled: new streaming session")
        for handle in (self._index, self._prev_events):  # one level back is enough
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._index = None
        # The session that just ended keeps its events file: an estimate started
        # before the roll can still publish into it.
        self._prev_path, self._prev_events = self.path, self._events
        self._events = None
        self._rolled_from = self.session_id
        self.session_id = self._unique_session_id()
        self.path = self.root / self.session_id
        self._opened = False
        self._session_frames = 0
        self._session_start_ts = None
        self._last_seq.clear()
        self.sessions += 1

    def _stamp(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))

    def _unique_session_id(self) -> str:
        """A fresh session id, suffixed if this second already has a directory.

        Two rolls within one second (or a rerun into a populated root) must not
        append into an existing session. The suffix sorts after the bare stamp
        and before the next second's, so chronological order survives.
        """
        stamp = self._stamp()
        candidate = stamp
        n = 1
        while candidate == self.session_id or (self.root / candidate).exists():
            n += 1
            candidate = f"{stamp}-{n}"
        return candidate

    def _event_writer(self, line: dict) -> TextIO:
        """The events file this line belongs in, opened if it is not yet."""
        prev = self._prev_path
        if prev is not None and self._belongs_to_previous(line):
            if self._prev_events is None:
                self._prev_events = (prev / EVENTS_NAME).open("a", encoding="utf-8")
            return self._prev_events
        if self._index is None:
            self._open()
        if self._events is None:
            self._events = (self.path / EVENTS_NAME).open("a", encoding="utf-8")
        return self._events

    def _belongs_to_previous(self, line: dict) -> bool:
        """Whether this event was estimated from a frame of the previous session."""
        if self._prev_path is None or self._session_start_ts is None:
            return False
        try:
            return float(line["capture_ts"]) < self._session_start_ts
        except (KeyError, TypeError, ValueError):
            return False

    def _open(self) -> TextIO:
        """Create the current session directory and open its index."""
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": FORMAT,
            "session_id": self.session_id,
            "started_ts": self._now(),
            "writer": self.writer,
        }
        if self._rolled_from is not None:
            manifest["rolled_from"] = self._rolled_from
        (self.path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        self._index = (self.path / INDEX_NAME).open("a", encoding="utf-8")
        self._opened = True
        return self._index

    def _close_manifest(self, reason: str) -> None:
        """Stamp how the current directory ended, for whoever finds it later.

        Best-effort like everything else here: a manifest that cannot be rewritten
        costs a reader some provenance, never the recording.
        """
        if not self._opened:
            return
        path = self.path / MANIFEST_NAME
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["closed_ts"] = self._now()
            manifest["closed_reason"] = reason
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError):
            pass


class Recording:
    """Read-only view of a recorded session.

    Iterating is lazy and streams from disk — a session is far larger than memory
    by design, so nothing here materialises the frames.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        index = self.path / INDEX_NAME
        if not index.exists():
            raise FileNotFoundError(f"not a recording (no {INDEX_NAME}): {self.path}")

    @classmethod
    def open(cls, path: str | Path) -> Recording:
        """Open a session directory, or the newest session inside a root directory.

        Session ids are timestamp-prefixed, so "newest" is simply the last one in
        sorted order. This is what makes ``--recording recordings/`` mean "replay
        the run that just happened".
        """
        p = Path(path)
        if (p / INDEX_NAME).exists():
            return cls(p)
        sessions = sorted(d for d in p.glob("*") if (d / INDEX_NAME).exists())
        if not sessions:
            raise FileNotFoundError(f"no recordings found in {p}")
        return cls(sessions[-1])

    @property
    def manifest(self) -> dict:
        try:
            return json.loads((self.path / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def frame_index(self, *, board_id: str | None = None) -> Iterator[FrameEntry]:
        """Yield the index entries in arrival order, without reading any JPEG.

        Unparseable lines are skipped — a session killed mid-write is still worth
        reading. This is the pass to take for timing, rates and board ids; take
        :meth:`frames` only when you actually want the bytes.
        """
        with (self.path / INDEX_NAME).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    header = json.loads(line)
                    rel = header.pop("jpeg")
                    received_ts = float(header.pop("received_ts", header["capture_ts"]))
                    header.pop("bytes", None)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if board_id is not None and header.get("board_id") != board_id:
                    continue
                yield FrameEntry(header=header, jpeg_rel=rel, received_ts=received_ts)

    def frames(self, *, board_id: str | None = None) -> Iterator[RecordedFrame]:
        """Yield recorded frames in arrival order, optionally for one board only.

        Entries that cannot be honoured — an unparseable line (a session killed
        mid-write), a JPEG that is gone — are skipped rather than raised: a
        partial recording is still a usable one.
        """
        for entry in self.frame_index(board_id=board_id):
            try:
                jpeg = (self.path / entry.jpeg_rel).read_bytes()
            except OSError:
                continue
            yield RecordedFrame(
                header=entry.header, jpeg=jpeg, received_ts=entry.received_ts
            )

    def board_ids(self) -> list[str]:
        """Every board that appears in the recording, in first-seen order."""
        seen: dict[str, None] = {}
        for entry in self.frame_index():
            seen.setdefault(entry.board_id, None)
        return list(seen)

    def __iter__(self) -> Iterator[RecordedFrame]:
        return self.frames()
