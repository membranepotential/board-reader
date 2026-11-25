"""The node's own streams: each a ``board-reader-edge camera`` process it starts,
watches, restarts and stops (DESIGN D22).

/control adds a video or live stream by URL (``add_stream``). The supervisor gives it
a stream id and a device id of its own, keeps both (:mod:`.store`) and starts the
edge's ``camera`` command on it (:mod:`.command`), pointed at this node. The process
registers on ``/cameras/{device_id}`` like any camera, so the registry assigns it a
board id and Auto-frame frames it; the supervisor only runs it:

* its output lines go to the node's log, each prefixed with the stream id;
* one that exits with an error is started again after a delay that doubles with each
  quick failure, up to :attr:`Backoff.cap_s`; one that ran for
  :attr:`Backoff.healthy_s` starts from the first delay again. One that exits cleanly
  (a video that ended) stays stopped;
* removing a stream (``remove_stream``) stops its process and forgets its camera
  (:meth:`~board_reader_inference.control.camera.registry.CameraRegistry.forget`), so
  its board id is free again;
* the node's shutdown stops every process (SIGTERM to its process group, SIGKILL
  after :data:`GRACE_S`) and keeps the cameras, which the next start runs again
  under the same device ids, so they get their board ids back.

Every process runs in a session of its own, so a Ctrl-C on the node's terminal
reaches the node only, and the stop signal reaches whatever the edge started too.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from board_reader_core.control import (
    CamerasMessage,
    StreamEntry,
    StreamsMessage,
    StreamState,
)

from board_reader_inference.control.streams.command import EdgeCommand, NoEdge
from board_reader_inference.control.streams.store import (
    StreamRecord,
    load_streams,
    save_streams,
    state_file,
    write_state_file,
)

log = logging.getLogger(__name__)

#: How long a stopped process may take to exit before it is killed.
GRACE_S = 5.0
#: How long a removed stream's camera may take to disconnect before it is forgotten.
FORGET_WAIT_S = 5.0
#: The longest error a stream reports; a traceback's last line is enough.
MAX_ERROR_CHARS = 500


@dataclass(frozen=True, slots=True)
class Backoff:
    """When a crashed stream starts again: ``first_s``, doubling with each quick
    failure, at most ``cap_s``; a run of ``healthy_s`` or longer was no quick failure."""

    first_s: float = 1.0
    cap_s: float = 60.0
    healthy_s: float = 60.0

    def delay(self, failures: int) -> float:
        return min(self.cap_s, self.first_s * 2 ** max(0, failures - 1))


BACKOFF = Backoff()


class Cameras(Protocol):
    """What the supervisor needs of the camera registry."""

    def board_of(self, device_id: str) -> str | None: ...

    def cameras(self, frame_age: Callable[[str], float | None]) -> CamerasMessage: ...

    def forget(self, device_id: str) -> None: ...


#: What a stream's process is doing, as far as the supervisor knows; ``alive`` is
#: ``starting`` or ``running`` by whether its camera is connected.
Phase = Literal["alive", "crashed", "stopped"]


@dataclass(slots=True)
class _Stream:
    record: StreamRecord
    phase: Phase = "alive"
    error: str | None = None


def _no_age(_board_id: str) -> None:
    return None


class StreamSupervisor:
    """Every stream the node runs, and its process.

    ``streams_file`` keeps the list across restarts (``None``: in memory, with the
    ``--state`` files in a temporary directory). :meth:`start` and :meth:`stop` run
    in the app's lifespan; :meth:`add` and :meth:`remove` on the event loop.
    """

    def __init__(
        self,
        cameras: Cameras,
        *,
        command: EdgeCommand | NoEdge,
        node: str,
        streams_file: Path | None,
        backoff: Backoff = BACKOFF,
        grace_s: float = GRACE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cameras = cameras
        self._command = command
        self._node = node
        self._file = streams_file
        self._backoff = backoff
        self._grace_s = grace_s
        self._clock = clock
        loaded = () if streams_file is None else load_streams(streams_file)
        self._streams: dict[str, _Stream] = {r.id: _Stream(r) for r in loaded}
        #: Never reused while the node runs, so a removed stream's teardown cannot
        #: touch a new stream's state file.
        self._next = 1 + max((_number(r.id) for r in loaded), default=0)
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._teardowns: set[asyncio.Task[None]] = set()
        self._lists: set[asyncio.Event] = set()
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    # ---- the lifespan ------------------------------------------------------------

    def start(self) -> None:
        """Start every saved stream."""
        if self._streams and isinstance(self._command, NoEdge):
            log.error("not starting %d saved stream(s): %s", len(self._streams), self._command.reason)
        for stream in self._streams.values():
            self._launch(stream)

    async def stop(self) -> None:
        """Stop every process (the streams stay saved, their cameras known)."""
        pending = [*self._runs.values(), *self._teardowns]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    # ---- what /control asks ------------------------------------------------------

    def add(self, url: str, start_s: float | None) -> str:
        """Run ``url`` as a new camera; its stream id. ``ValueError`` when this node
        has no edge command."""
        if isinstance(self._command, NoEdge):
            raise ValueError(self._command.reason)
        stream_id = f"stream-{self._next}"
        self._next += 1
        device_id = _new_device_id()
        record = StreamRecord(stream_id, url, start_s or None, device_id, time.time())
        stream = self._streams[stream_id] = _Stream(record)
        self._save()
        log.info("%s: added %s as camera %s", stream_id, url, device_id[:8])
        self._launch(stream)
        return stream_id

    def remove(self, stream_id: str) -> None:
        """Stop ``stream_id`` and forget its camera; ``ValueError`` for an unknown one.

        Returns at once; the process stops and the camera is forgotten in a task of
        the supervisor's, which a /streams socket that hangs up does not cancel.
        """
        stream = self._streams.pop(stream_id, None)
        if stream is None:
            raise ValueError(f"no stream {stream_id}")
        self._save()
        self._tell_lists()
        log.info("%s: removing %s", stream_id, stream.record.url)
        task = asyncio.get_running_loop().create_task(self._teardown(stream.record))
        self._teardowns.add(task)
        task.add_done_callback(self._teardowns.discard)

    def streams(self) -> StreamsMessage:
        """Every stream, in the order added, with its camera's board id once known."""
        connected = {
            entry.device_id for entry in self._cameras.cameras(_no_age).cameras if entry.connected
        }
        return StreamsMessage(
            streams=tuple(
                StreamEntry(
                    id=stream.record.id,
                    url=stream.record.url,
                    start_s=stream.record.start_s,
                    state=_state(stream.phase, stream.record.device_id in connected),
                    device_id=stream.record.device_id,
                    board_id=self._cameras.board_of(stream.record.device_id),
                    error=stream.error,
                )
                for stream in self._streams.values()
            )
        )

    def connect_list(self) -> asyncio.Event:
        """A wake-up for one open /streams socket, set whenever a stream changes."""
        changed = asyncio.Event()
        self._lists.add(changed)
        return changed

    def disconnect_list(self, changed: asyncio.Event) -> None:
        self._lists.discard(changed)

    # ---- the processes -------------------------------------------------------------

    def _launch(self, stream: _Stream) -> None:
        if isinstance(self._command, NoEdge):
            self._set(stream, "stopped", self._command.reason)
            return
        command = self._command
        task = asyncio.get_running_loop().create_task(
            self._run(stream, command), name=stream.record.id
        )
        self._runs[stream.record.id] = task
        task.add_done_callback(lambda _: self._runs.pop(stream.record.id, None))

    async def _run(self, stream: _Stream, command: EdgeCommand) -> None:
        """Run ``stream``'s process until it exits cleanly; start it again after an error."""
        record = stream.record
        state = state_file(self._state_root(), record.id)
        argv = command.camera(record.url, node=self._node, state=state, start_s=record.start_s)
        failures = 0
        while True:
            write_state_file(state, record.device_id)
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except OSError as e:
                log.error("%s: cannot start %s: %s", record.id, argv[0], e)
                self._set(stream, "stopped", f"cannot start {argv[0]}: {e}")
                return
            log.info("%s: started (pid %d): %s", record.id, process.pid, shlex.join(argv))
            self._set(stream, "alive", stream.error)
            started = self._clock()
            try:
                assert process.stdout is not None
                last = await _relay(record.id, process.stdout)
                code = await process.wait()
            finally:
                await _end(process, self._grace_s)
            if code == 0:
                log.info("%s: the stream ended", record.id)
                self._set(stream, "stopped", stream.error)
                return
            failures = 1 if self._clock() - started >= self._backoff.healthy_s else failures + 1
            delay = self._backoff.delay(failures)
            error = f"exited with code {code}" + (f": {last}" if last else "")
            log.warning("%s %s; starting it again in %.0f s", record.id, error, delay)
            self._set(stream, "crashed", error[:MAX_ERROR_CHARS])
            await asyncio.sleep(delay)

    async def _teardown(self, record: StreamRecord) -> None:
        run = self._runs.get(record.id)
        if run is not None:
            run.cancel()
            await asyncio.wait([run])
        await self._forget(record.device_id)
        state_file(self._state_root(), record.id).unlink(missing_ok=True)

    async def _forget(self, device_id: str) -> None:
        """Forget the camera once its socket is closed (the registry refuses a
        connected one); a camera that never registered has nothing to forget."""
        if self._cameras.board_of(device_id) is None:
            return
        deadline = self._clock() + FORGET_WAIT_S
        while any(
            e.device_id == device_id and e.connected for e in self._cameras.cameras(_no_age).cameras
        ):
            if self._clock() >= deadline:
                log.warning("camera %s is still connected; not forgotten", device_id[:8])
                return
            await asyncio.sleep(0.05)
        try:
            self._cameras.forget(device_id)
        except ValueError as e:
            log.warning("cannot forget camera %s: %s", device_id[:8], e)

    # ---- bookkeeping -------------------------------------------------------------

    def _set(self, stream: _Stream, phase: Phase, error: str | None) -> None:
        stream.phase, stream.error = phase, error
        self._tell_lists()

    def _tell_lists(self) -> None:
        for changed in self._lists:
            changed.set()

    def _state_root(self) -> Path:
        if self._file is not None:
            return self._file.parent
        if self._scratch is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="board-reader-streams-")
        return Path(self._scratch.name)

    def _save(self) -> None:
        if self._file is None:
            return
        try:
            save_streams(self._file, tuple(s.record for s in self._streams.values()))
        except OSError as e:
            log.warning("cannot save the streams to %s: %s", self._file, e)


def _state(phase: Phase, connected: bool) -> StreamState:
    if phase != "alive":
        return phase
    return "running" if connected else "starting"


def _number(stream_id: str) -> int:
    return int(stream_id.removeprefix("stream-"))


def _new_device_id() -> str:
    return str(uuid.uuid4())


async def _relay(stream_id: str, output: asyncio.StreamReader) -> str | None:
    """Log every line the process writes, prefixed; its last non-empty line."""
    last: str | None = None
    while True:
        try:
            raw = await output.readline()
        except ValueError:  # a line past the reader's limit: dropped, the rest goes on
            continue
        if not raw:
            return last
        line = raw.decode(errors="replace").rstrip()
        if line:
            log.info("[%s] %s", stream_id, line)
            last = line


def _signal(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """``sig`` to the process's whole group (it leads a session of its own)."""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


async def _end(process: asyncio.subprocess.Process, grace_s: float) -> None:
    """Stop ``process`` if it still runs: SIGTERM, then SIGKILL after ``grace_s``
    (or at once when the wait itself is cancelled), and reap it."""
    if process.returncode is not None:
        return
    _signal(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), grace_s)
        return
    except TimeoutError:
        pass
    except asyncio.CancelledError:
        _signal(process, signal.SIGKILL)
        raise
    _signal(process, signal.SIGKILL)
    await process.wait()
