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

from board_reader_core.wire import Frame, decode_jpeg

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
        """Decode to the in-memory :class:`~board_reader_core.wire.Frame`."""
        return Frame.from_header(self.header, decode_jpeg(self.jpeg))


class FrameRecorder:
    """Append-only writer for one recording session.

    The session directory is created **lazily**, on the first recorded frame: a
    gateway that runs all day without an edge ever connecting leaves no empty
    directories behind.

    Recording is best-effort infrastructure, never a reason to fail the stream it
    observes. An :class:`OSError` (disk full, permissions) stops the recorder and
    is reported through :attr:`stopped_reason`; it does not propagate.

    Parameters
    ----------
    root:
        Directory to create the session directory inside.
    session_id:
        Session directory name. Defaults to a sortable ``YYYYmmdd-HHMMSS`` stamp,
        so listing a root directory lists sessions in chronological order.
    writer:
        Free-form name of what is recording, stored in the manifest.
    max_bytes:
        Stop recording once this many JPEG bytes have been written. The recording
        is *truncated*, never rotated — losing the tail of a session is obvious
        and recoverable, silently losing its start is neither.
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
        self.session_id = session_id or time.strftime(
            "%Y%m%d-%H%M%S", time.localtime(now())
        )
        self.path = self.root / self.session_id
        self.writer = writer
        self.max_bytes = max_bytes
        self.frames_written = 0
        self.bytes_written = 0
        self.events_written = 0
        self.stopped_reason: str | None = None
        self._index = None  # opened on the first frame
        self._events = None  # opened on the first event

    @property
    def active(self) -> bool:
        return self.stopped_reason is None

    def record(self, header: dict, jpeg: bytes) -> bool:
        """Append one frame. Returns whether it was written.

        ``header`` should be a frame's own :meth:`~board_reader_core.wire.Frame.header`
        rather than the raw JSON that arrived, so the recording holds values the
        wire has already validated and normalised.
        """
        if self.stopped_reason is not None:
            return False
        try:
            if self._index is None:
                self._open()
            rel = _shard_path(self.frames_written)
            blob = self.path / rel
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(jpeg)
            # Bytes first, then the reference to them: a reader that crashes into
            # the end of this file never sees a line pointing at nothing.
            line = {**header, "received_ts": self._now(), "jpeg": rel, "bytes": len(jpeg)}
            self._index.write(json.dumps(line) + "\n")
            self._index.flush()
        except OSError as e:
            self.stop(f"{type(e).__name__}: {e}")
            return False
        self.frames_written += 1
        self.bytes_written += len(jpeg)
        # The cap stops the recorder *after* a frame lands, so a session may
        # overshoot by one frame but can never come out empty — a limit smaller
        # than a single JPEG would otherwise silently record nothing at all.
        if self.max_bytes is not None and self.bytes_written >= self.max_bytes:
            self.stop(f"max_bytes reached ({self.max_bytes} B)")
        return True

    def record_event(self, line: dict) -> bool:
        """Append one published position event. Returns whether it was written.

        Same crash-safety contract as :meth:`record` (flushed per line) and the
        same failure semantics: an ``OSError`` stops the whole recorder, because
        a session with silently-partial outputs would be mistaken for one where
        the estimator went quiet.
        """
        if self.stopped_reason is not None:
            return False
        try:
            if self._index is None and self._events is None:
                self._open()
            if self._events is None:
                self._events = (self.path / EVENTS_NAME).open("a", encoding="utf-8")
            self._events.write(json.dumps(line) + "\n")
            self._events.flush()
        except OSError as e:
            self.stop(f"{type(e).__name__}: {e}")
            return False
        self.events_written += 1
        return True

    def stop(self, reason: str = "closed") -> None:
        """Stop recording. Idempotent; keeps the first reason."""
        if self.stopped_reason is None:
            self.stopped_reason = reason
        for handle in (self._index, self._events):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._index = None
        self._events = None

    close = stop

    def stats(self) -> dict:
        """JSON-safe summary — what a health endpoint should report."""
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "frames": self.frames_written,
            "events": self.events_written,
            "bytes": self.bytes_written,
            "active": self.active,
            "stopped_reason": self.stopped_reason,
        }

    def _open(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "session_id": self.session_id,
                    "started_ts": self._now(),
                    "writer": self.writer,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._index = (self.path / INDEX_NAME).open("a", encoding="utf-8")


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

    def frames(self, *, board_id: str | None = None) -> Iterator[RecordedFrame]:
        """Yield recorded frames in arrival order, optionally for one board only.

        Entries that cannot be honoured — an unparseable line (a session killed
        mid-write), a JPEG that is gone — are skipped rather than raised: a
        partial recording is still a usable one.
        """
        with (self.path / INDEX_NAME).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    rel = entry.pop("jpeg")
                    received_ts = float(entry.pop("received_ts", entry["capture_ts"]))
                    entry.pop("bytes", None)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if board_id is not None and entry.get("board_id") != board_id:
                    continue
                try:
                    jpeg = (self.path / rel).read_bytes()
                except OSError:
                    continue
                yield RecordedFrame(header=entry, jpeg=jpeg, received_ts=received_ts)

    def board_ids(self) -> list[str]:
        """Every board that appears in the recording, in first-seen order."""
        seen: dict[str, None] = {}
        for rec in self.frames():
            seen.setdefault(rec.board_id, None)
        return list(seen)

    def __iter__(self) -> Iterator[RecordedFrame]:
        return self.frames()
