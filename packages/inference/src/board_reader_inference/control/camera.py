"""Camera control on the inference node: per-board config, phone reports, previews.

The node holds, per ``board_id``, what the control UI chose for that board's camera
(:class:`~board_reader_core.control.CameraConfig`), the last thing the camera said
about itself (:class:`~board_reader_core.control.CameraReport`) and the latest
full-frame preview, so the control UI can show the camera's whole field of view
while the stream itself carries only the crop.

Each phone is known by its **device id**, a random id its browser keeps, and the
registry **assigns** each new device a board id (the lowest free ``board-N``) once
and keeps that pair for good (DESIGN D18), so the same phone streams as the same
board after a reload, a reconnect or a node restart, and every event, recording and
PGN stays under one id. The control UI may give a camera a **name**, an alias the
live page, /control and the phone show instead of the id; a rename changes nothing
else. Names are unique among the known cameras and never another camera's board id,
so no two cards read alike. The control UI may also **forget** a disconnected
camera: its board id and config are dropped and the id is free again; the device,
if it comes back, is a new camera.

The registry also decides which ``/ingest`` frames a board id accepts
(:meth:`CameraRegistry.ingest_refusal`): a board id a camera holds takes frames from
that camera only, so a stale page or a replay cannot stream into a camera's board.

The configs, the device -> board pairs and the names can outlive a restart: with
``BR_CAMERA_CONFIG=<path>`` set (``serve`` sets it), every change is written to that
JSON file and read back at startup, so a node restarted mid-game gives each phone
its board id back and pushes the same crop, lens, resolution and streaming switch to
it when it reconnects, and settles each board under the same rules. Unset,
everything is in memory. Reports, previews and the board direction are always in
memory: the phone and the recognizer send them again.

The registry also follows each board's **direction** in the image, from the
recognizer's reads (:meth:`CameraRegistry.observe_direction`): which side White sits
on, and whether the camera moved since the angle settled.

The registry is the single source of truth, and it **pushes**: every change is
posted at once to every socket that cares (the message types and their directions
are :mod:`board_reader_core.control`'s):

* ``WS /cameras/{device_id}`` — the phone. It hears its board id, its config and
  whether to send previews; it says what the camera is and sends previews. One
  socket per device: a newer one replaces the older, which hears ``replaced``.
* ``WS /cameras`` — the control UI's list of every known camera; it sends renames
  and forgets.
* ``WS /boards/{board_id}/control`` — each open control page. It hears the state,
  the previews, the latest received frame and a health tick; it sends configs.

No polling, so no HTTP request per message and nothing to log per message: a
socket costs one access-log line when it opens.

Each socket gets a :class:`Mailbox` that keeps only the **latest message of each
type**. Every message here is a snapshot (a config, a state, a picture, a health
reading), so a slow client that misses an intermediate one loses nothing it needs,
and no client can make the node buffer without bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import statistics
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, ClassVar, Protocol

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from board_reader_core.control import (
    DEFAULT_RULES,
    AssignedMessage,
    BoardDirection,
    CameraBoardId,
    CameraConfig,
    CameraEntry,
    CameraName,
    CameraReport,
    CamerasDown,
    CamerasMessage,
    ConfigMessage,
    ControlDown,
    DeviceId,
    ForgetMessage,
    FrameMessage,
    HealthMessage,
    PhoneDown,
    PreviewMessage,
    PreviewsMessage,
    RefusedMessage,
    ReplacedMessage,
    Rules,
    StateMessage,
    WhiteSide,
    camera_name,
    parse_cameras_message,
    parse_control_message,
    parse_phone_message,
)
from board_reader_core.control import (
    device_id as checked_device_id,
)
from board_reader_core.protocol import BoardName, NamesMessage
from board_reader_core.wire import decode_jpeg
from board_reader_core.wiremodel import parse

log = logging.getLogger(__name__)

#: A preview is a small thumbnail; anything bigger is not one.
MAX_PREVIEW_BYTES = 1_000_000

#: How often a control page hears the health tick (and the latest received frame).
HEALTH_EVERY_S = 1.0

#: Where each board's config is kept across restarts; unset or empty, nowhere.
CAMERA_CONFIG_ENV = "BR_CAMERA_CONFIG"


class _Message(Protocol):
    TYPE: ClassVar[str]

    def to_json(self) -> str: ...


class Mailbox[M: _Message]:
    """The messages one socket has yet to send: the latest of each type.

    Posting a type again replaces the pending one and moves it to the back, so
    messages leave in the order they were last posted.
    """

    def __init__(self) -> None:
        self._pending: dict[str, M] = {}
        self._ready = asyncio.Event()

    def post(self, message: M) -> None:
        self._pending.pop(message.TYPE, None)
        self._pending[message.TYPE] = message
        self._ready.set()

    def take_nowait(self) -> list[M]:
        taken = list(self._pending.values())
        self._pending.clear()
        self._ready.clear()
        return taken

    async def take(self) -> list[M]:
        await self._ready.wait()
        return self.take_nowait()


# ---- the configs on disk --------------------------------------------------------------

#: The board id a new camera gets: this prefix and the lowest free number.
AUTO_BOARD_PREFIX = "board-"


@dataclass(frozen=True, slots=True)
class Saved:
    """What the node keeps across restarts: each board's config, each device's board
    and each named device's name."""

    boards: Mapping[str, CameraConfig] = field(default_factory=dict)
    cameras: Mapping[str, str] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)


def _parse_saved(raw: object) -> Saved:
    if (
        not isinstance(raw, dict)
        or "boards" not in raw
        or not set(raw) <= {"boards", "cameras", "names"}
    ):
        raise ValueError(
            'expected {"boards": {<board id>: <config>}, "cameras": {<device id>: <board id>}, '
            '"names": {<device id>: <name>}}'
        )
    if not isinstance(raw["boards"], dict):
        raise ValueError('"boards" is not an object')
    boards = {str(board_id): CameraConfig.from_dict(config) for board_id, config in raw["boards"].items()}
    # A file from before the cameras were known by device has no "cameras": none known
    # yet; one from before the names has no "names": none named yet.
    for key in ("cameras", "names"):
        if not isinstance(raw.get(key, {}), dict):
            raise ValueError(f'"{key}" is not an object')
    cameras: dict[str, str] = parse(dict[DeviceId, CameraBoardId], raw.get("cameras", {}))
    if len(set(cameras.values())) != len(cameras):
        raise ValueError("two cameras hold one board id")
    names: dict[str, str] = parse(dict[DeviceId, CameraName], raw.get("names", {}))
    if not set(names) <= set(cameras):
        raise ValueError("a name for a camera that is not known")
    if len(set(names.values())) != len(names):
        raise ValueError("two cameras have one name")
    return Saved(boards, cameras, names)


def load_state(path: Path) -> Saved:
    """What ``save_state`` wrote; nothing saved if there is no file.

    A file that does not parse in full is ignored with a WARNING, never half
    applied: the node then starts with the defaults (streaming off, no camera
    known), the same as a node that never had a file.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Saved()
    except OSError as e:
        log.warning("ignoring the camera config %s: %s", path, e)
        return Saved()
    try:
        return _parse_saved(json.loads(text))
    except ValueError as e:
        log.warning("ignoring the corrupt camera config %s: %s", path, e)
        return Saved()


def save_state(path: Path, saved: Saved) -> None:
    """Write everything to ``path`` at once: a temporary file, then a rename.

    A crash mid-write leaves the old file whole, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "boards": {board_id: config.to_dict() for board_id, config in saved.boards.items()},
        "cameras": dict(saved.cameras),
        "names": dict(saved.names),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    os.replace(tmp, path)


def _config_path() -> Path | None:
    raw = os.environ.get(CAMERA_CONFIG_ENV)
    return Path(raw) if raw else None


# ---- which way the board lies ------------------------------------------------------------

#: The running angle is the median of this many latest reads, so one or two odd
#: reads (a hand over the board) do not move it.
DIRECTION_READS = 5
#: The angle settles once that many reads span at least this many seconds of
#: capture time, so a burst of reads of one scene cannot settle it alone.
DIRECTION_SETTLE_S = 2.0
#: A running angle this far from the settled one means the camera moved. ChessQ
#: reads a fixed camera within about 2 degrees.
CAMERA_MOVED_DEGREES = 15.0

#: The angle at which White sits on each side (DESIGN D11; :class:`BoardDirection`).
_SIDES: dict[WhiteSide, float] = {"bottom": 0.0, "left": 90.0, "top": 180.0, "right": -90.0}


def _wrap(degrees: float) -> float:
    """``degrees`` folded into (-180, 180]."""
    folded = math.remainder(degrees, 360.0)
    return 180.0 if folded == -180.0 else folded


def white_side(degrees: float) -> WhiteSide:
    """The side White sits on for a board direction: the nearest of the four."""
    return min(_SIDES, key=lambda side: abs(_wrap(degrees - _SIDES[side])))


@dataclass(frozen=True, slots=True)
class _Read:
    degrees: float
    ts: float


@dataclass(slots=True)
class DirectionTrack:
    """One board's direction: the latest reads and the angle they settled on."""

    reads: deque[_Read] = field(default_factory=lambda: deque(maxlen=DIRECTION_READS))
    settled: float | None = None

    def observe(self, degrees: float, ts: float) -> None:
        self.reads.append(_Read(_wrap(degrees), ts))
        if self.settled is None and len(self.reads) == DIRECTION_READS:
            stamps = [read.ts for read in self.reads]
            if max(stamps) - min(stamps) >= DIRECTION_SETTLE_S:
                self.settled = self.running()

    def running(self) -> float | None:
        """The median of the latest reads, taken around the newest so it never wraps."""
        if not self.reads:
            return None
        newest = self.reads[-1].degrees
        offset = statistics.median(_wrap(read.degrees - newest) for read in self.reads)
        return _wrap(newest + offset)

    def view(self) -> BoardDirection | None:
        running = self.running()
        if running is None:
            return None
        moved = (
            self.settled is not None
            and abs(_wrap(running - self.settled)) > CAMERA_MOVED_DEGREES
        )
        return BoardDirection(
            degrees=round(running, 1), white_side=white_side(running), camera_moved=moved
        )


def _reframed(old: CameraConfig, new: CameraConfig) -> bool:
    """Whether ``new`` changes what the camera films (not just whether it streams)."""
    return (old.crop, old.device, old.resolution) != (new.crop, new.device, new.resolution)


# ---- the registry --------------------------------------------------------------------------


@dataclass(slots=True)
class BoardCamera:
    """One board's camera state; it moves with the board id when a camera is renamed."""

    config: CameraConfig = field(default_factory=CameraConfig)
    report: CameraReport | None = None
    preview: PreviewMessage | None = None
    direction: DirectionTrack = field(default_factory=DirectionTrack)
    watchers: set[Mailbox[ControlDown]] = field(default_factory=set)


@dataclass(eq=False, slots=True)
class PhoneLink:
    """One open phone socket: the device it speaks for and its outbox."""

    device_id: str
    mailbox: Mailbox[PhoneDown] = field(default_factory=Mailbox)


def _natural(board_id: str) -> list[int | str]:
    """``board-2`` before ``board-10``: digit runs compare as numbers."""
    return [int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", board_id)]


class CameraRegistry:
    """Every board's camera state, every known camera, and the sockets they are pushed to.

    A board exists from the first time anyone asks. A camera (a device id) exists
    from its first connect and keeps its board id until it is renamed. Methods are
    synchronous and run on the event loop; they only post to mailboxes (and wake the
    camera lists), and each socket's own task sends.

    With ``BR_CAMERA_CONFIG`` set, the configs and the cameras are read from that
    file here and written back on every change (see the module docstring). A saved
    config no camera holds (a board named by hand before the device ids) is dropped
    at load, so no new camera can inherit it.

    ``on_names`` hears the whole :class:`NamesMessage` now and on every change of a
    name (the gateway hands it to ``/viewer``); ``on_forget`` hears the board id of
    each forgotten camera (the gateway frees that board's read state).
    """

    def __init__(
        self,
        *,
        on_names: Callable[[NamesMessage], None] | None = None,
        on_forget: Callable[[str], None] | None = None,
    ) -> None:
        self._path = _config_path()
        saved = Saved() if self._path is None else load_state(self._path)
        held = set(saved.cameras.values())
        orphans = sorted(set(saved.boards) - held)
        if orphans:
            log.info(
                "dropping the saved config of %s: no camera holds %s",
                ", ".join(orphans), "it" if len(orphans) == 1 else "them",
            )
        self._boards: dict[str, BoardCamera] = {
            board_id: BoardCamera(config=config)
            for board_id, config in saved.boards.items()
            if board_id in held
        }
        #: device id -> board id, one board id per camera, for good (persisted).
        self._cameras: dict[str, str] = dict(saved.cameras)
        #: device id -> its name, for the named cameras (persisted).
        self._names: dict[str, str] = dict(saved.names)
        #: device id -> its one open socket.
        self._phones: dict[str, PhoneLink] = {}
        #: One wake-up per open camera list: something it shows changed.
        self._lists: set[asyncio.Event] = set()
        self._on_names = on_names
        self._on_forget = on_forget
        self._announce_names()

    def board(self, board_id: str) -> BoardCamera:
        return self._boards.setdefault(board_id, BoardCamera())

    def rules(self, board_id: str) -> Rules:
        """Which chess ``board_id`` plays, as its settle guard reads it.

        A board nobody configured plays standard chess; asking does not create it.
        """
        board = self._boards.get(board_id)
        return DEFAULT_RULES if board is None else board.config.rules

    def board_of(self, device_id: str) -> str | None:
        """The board id ``device_id`` streams as; ``None`` for a device never seen."""
        return self._cameras.get(device_id)

    def _holder(self, board_id: str) -> str | None:
        return next((d for d, b in self._cameras.items() if b == board_id), None)

    def _phone_on(self, board_id: str) -> PhoneLink | None:
        holder = self._holder(board_id)
        return None if holder is None else self._phones.get(holder)

    def _state(self, board_id: str) -> StateMessage:
        board = self.board(board_id)
        return StateMessage(
            config=board.config,
            report=board.report,
            phone_connected=self._phone_on(board_id) is not None,
        )

    def _tell_watchers(self, board_id: str, message: ControlDown | None = None) -> None:
        """Post ``message`` (the board's state by default) to its control pages."""
        board = self.board(board_id)
        for watcher in board.watchers:
            watcher.post(self._state(board_id) if message is None else message)

    def _tell_phone(self, board_id: str, message: PhoneDown) -> None:
        phone = self._phone_on(board_id)
        if phone is not None:
            phone.mailbox.post(message)

    def _tell_lists(self) -> None:
        for changed in self._lists:
            changed.set()

    def _current(self, link: PhoneLink) -> bool:
        """Whether ``link`` is still its device's socket (not replaced by a newer one)."""
        return self._phones.get(link.device_id) is link

    # ---- naming the cameras ------------------------------------------------------

    def _free_board_id(self) -> str:
        """The lowest ``board-N`` no camera holds, none is named, and no config sits under."""
        taken = set(self._cameras.values()) | set(self._names.values())
        for n in count(1):
            board_id = f"{AUTO_BOARD_PREFIX}{n}"
            board = self._boards.get(board_id)
            if board_id not in taken and (board is None or board.config == CameraConfig()):
                return board_id
        raise AssertionError("unreachable")

    def label(self, board_id: str) -> str:
        """What people call ``board_id``: its camera's name, or the id itself."""
        holder = self._holder(board_id)
        return board_id if holder is None else self._names.get(holder, board_id)

    def names(self) -> NamesMessage:
        """Every named camera's name, by board id, in board id order."""
        named = sorted(
            (BoardName(self._cameras[device_id], name) for device_id, name in self._names.items()),
            key=lambda entry: _natural(entry.board_id),
        )
        return NamesMessage(names=tuple(named))

    def _announce_names(self) -> None:
        if self._on_names is not None:
            self._on_names(self.names())

    def rename(self, device_id: str, name: str | None) -> None:
        """Give a known camera a name (``None``, or its own board id, clears it).

        Only the alias changes: the board id, its config, its events and recordings
        stay. Refused (``ValueError``) for an unknown device, a name that is no
        :data:`CameraName`, or one another camera has as its name or board id.
        """
        board_id = self._cameras.get(device_id)
        if board_id is None:
            raise ValueError(f"no camera {device_id[:8]} is known")
        wanted = None if name is None or name == board_id else camera_name(name)
        if wanted is not None:
            for other, other_board in self._cameras.items():
                if other != device_id and wanted in (other_board, self._names.get(other)):
                    raise ValueError(f"{wanted} is taken by camera {self.label(other_board)}")
        if wanted == self._names.get(device_id):
            return
        if wanted is None:
            del self._names[device_id]
        else:
            self._names[device_id] = wanted
        self._save()
        phone = self._phones.get(device_id)
        if phone is not None:
            phone.mailbox.post(AssignedMessage(board_id, wanted))
        self._tell_lists()
        self._announce_names()

    def forget(self, device_id: str) -> None:
        """Drop a disconnected camera: its board id, name and config; the id is free.

        The board's control pages stay open on a board with the default config. A
        device that comes back is a new camera and gets the lowest free ``board-N``.
        Refused (``ValueError``) for an unknown device or a connected one: a
        connected phone would only be handed a new board id at once.
        """
        board_id = self._cameras.get(device_id)
        if board_id is None:
            raise ValueError(f"no camera {device_id[:8]} is known")
        if device_id in self._phones:
            raise ValueError(
                f"camera {self.label(board_id)} is connected: close its /camera page, "
                "then forget it"
            )
        del self._cameras[device_id]
        named = self._names.pop(device_id, None) is not None
        watchers = self.board(board_id).watchers
        if watchers:
            self._boards[board_id] = BoardCamera(watchers=watchers)
        else:
            del self._boards[board_id]
        self._save()
        log.info("forgot camera %s and its board %s", device_id[:8], board_id)
        self._tell_watchers(board_id)
        self._tell_lists()
        if named:
            self._announce_names()
        if self._on_forget is not None:
            self._on_forget(board_id)

    # ---- the frames --------------------------------------------------------------

    def ingest_refusal(self, board_id: str, device_id: str | None) -> str | None:
        """Why ``/ingest`` must refuse a frame under ``board_id`` from ``device_id``.

        ``None`` accepts it. A board id a camera holds takes frames from that camera
        only. A frame from a device the node does not know (forgotten, or one that
        never opened its control socket) is refused, because its board id was never
        handed to it. An edge without a device id (``board-reader-edge feed``, a
        replayed recording) may stream under any board id no camera holds.
        """
        holder = self._holder(board_id)
        if device_id is None:
            if holder is None:
                return None
            return (
                f"board {board_id} belongs to camera {self.label(board_id)}; an edge "
                "without a device id may stream only under a board id no camera holds"
            )
        own = self._cameras.get(device_id)
        if own is None:
            return f"camera {device_id[:8]} is not known to the node (forgotten?): reload /camera"
        if own != board_id:
            return f"camera {device_id[:8]} streams as {own}, not {board_id}: reload /camera"
        return None

    # ---- the phone ---------------------------------------------------------------

    def connect_phone(self, device_id: str) -> PhoneLink:
        """Open ``device_id``'s socket; a socket it already had is replaced.

        A device seen for the first time gets the lowest free ``board-N``, for good.
        """
        previous = self._phones.get(device_id)
        if previous is not None:
            previous.mailbox.post(ReplacedMessage())
        link = PhoneLink(device_id)
        self._phones[device_id] = link
        if device_id not in self._cameras:
            self._cameras[device_id] = self._free_board_id()
            self._save()
        board_id = self._cameras[device_id]
        board = self.board(board_id)
        link.mailbox.post(AssignedMessage(board_id, self._names.get(device_id)))
        link.mailbox.post(ConfigMessage(board.config))
        link.mailbox.post(PreviewsMessage(on=bool(board.watchers)))
        self._tell_watchers(board_id)
        self._tell_lists()
        return link

    def disconnect_phone(self, link: PhoneLink) -> None:
        if not self._current(link):
            return  # replaced: the newer socket is the device's now
        del self._phones[link.device_id]
        self._tell_watchers(self._cameras[link.device_id])
        self._tell_lists()

    def report(self, link: PhoneLink, report: CameraReport) -> None:
        if not self._current(link):
            return
        board_id = self._cameras[link.device_id]
        self.board(board_id).report = report
        self._tell_watchers(board_id)
        self._tell_lists()

    def preview(self, link: PhoneLink, preview: PreviewMessage) -> None:
        if not self._current(link):
            return
        board_id = self._cameras[link.device_id]
        self.board(board_id).preview = preview
        self._tell_watchers(board_id, preview)

    # ---- the camera lists --------------------------------------------------------

    def connect_list(self) -> asyncio.Event:
        """A wake-up for one open camera list, set whenever what it shows changes."""
        changed = asyncio.Event()
        self._lists.add(changed)
        return changed

    def disconnect_list(self, changed: asyncio.Event) -> None:
        self._lists.discard(changed)

    def cameras(self, frame_age: Callable[[str], float | None]) -> CamerasMessage:
        """Every known camera, ordered by board id; ``frame_age`` reads a board's."""
        entries = []
        for device_id, board_id in sorted(self._cameras.items(), key=lambda kv: _natural(kv[1])):
            board = self._boards.get(board_id)
            phone = self._phones.get(device_id)
            entries.append(
                CameraEntry(
                    device_id=device_id,
                    board_id=board_id,
                    name=self._names.get(device_id),
                    connected=phone is not None,
                    report=None if board is None else board.report,
                    frame_age_s=frame_age(board_id),
                )
            )
        return CamerasMessage(cameras=tuple(entries))

    # ---- the control pages -------------------------------------------------------

    def connect_watcher(self, board_id: str) -> Mailbox[ControlDown]:
        board = self.board(board_id)
        mailbox: Mailbox[ControlDown] = Mailbox()
        board.watchers.add(mailbox)
        mailbox.post(self._state(board_id))
        if board.preview is not None:
            mailbox.post(board.preview)
        if len(board.watchers) == 1:
            self._tell_phone(board_id, PreviewsMessage(on=True))
        return mailbox

    def disconnect_watcher(self, board_id: str, mailbox: Mailbox[ControlDown]) -> None:
        board = self.board(board_id)
        board.watchers.discard(mailbox)
        if not board.watchers:
            self._tell_phone(board_id, PreviewsMessage(on=False))

    def configure(self, board_id: str, config: CameraConfig) -> None:
        board = self.board(board_id)
        if _reframed(board.config, config):
            board.direction = DirectionTrack()
        changed = config != board.config
        board.config = config
        if changed:
            self._save()
        self._tell_phone(board_id, ConfigMessage(config))
        self._tell_watchers(board_id)

    def _save(self) -> None:
        if self._path is None:
            return
        configured = {
            board_id: board.config
            for board_id, board in self._boards.items()
            if board.config != CameraConfig()
        }
        try:
            save_state(self._path, Saved(configured, dict(self._cameras), dict(self._names)))
        except OSError as e:
            # The state still applies; only the next restart forgets it.
            log.warning("cannot save the camera config to %s: %s", self._path, e)

    # ---- the recognizer --------------------------------------------------------------

    def observe_direction(self, board_id: str, degrees: float, ts: float) -> None:
        """One live read's board direction (see :class:`BoardDirection` for the angle).

        ``ts`` is the frame's capture time (seconds); it only decides when the angle
        has been read long enough to settle. Control pages hear the result on their
        next health tick.
        """
        self.board(board_id).direction.observe(degrees, ts)

    def direction(self, board_id: str) -> BoardDirection | None:
        return self.board(board_id).direction.view()


# ---- what the node says about itself on the health tick ----------------------------


@dataclass(frozen=True, slots=True)
class StreamedFrame:
    """The last frame ``/ingest`` received for a board, as the control UI needs it."""

    seq: int
    #: Node clock, seconds.
    received_at: float
    width: int
    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class NodeStatus:
    """The node-side facts a health tick reports, for one board."""

    recognizer: str
    fallback: str | None
    received: int
    refused: int
    frame: StreamedFrame | None


@dataclass(frozen=True, slots=True)
class StreamSample:
    """One reading of the stream's position, for its rate between two ticks."""

    seq: int
    at: float


def frame_rate(previous: StreamSample | None, current: StreamSample | None) -> float | None:
    """Frames per second between two samples, from the frame sequence number.

    ``None`` when there is nothing to measure: no earlier sample, no time between
    them, or a sequence that went backwards (the phone reloaded and began at 0).
    """
    if previous is None or current is None:
        return None
    dt = current.at - previous.at
    frames = current.seq - previous.seq
    return frames / dt if dt > 0 and frames >= 0 else None


# ---- the sockets -------------------------------------------------------------------


async def _deliver[M: _Message](ws: WebSocket, mailbox: Mailbox[M]) -> None:
    while True:
        for message in await mailbox.take():
            await ws.send_text(message.to_json())
            if isinstance(message, ReplacedMessage):
                # The device's newer socket took over: this one closes, said aloud,
                # because a handler that only returns leaves the socket open.
                await ws.close(code=1000, reason="replaced")
                return


class _HungUp(Exception):
    """One of a socket's loops ended: the client hung up (or a send failed)."""


async def _ending(loop: Coroutine[Any, Any, None]) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        await loop
    raise _HungUp


async def _until_first_ends(*loops: Coroutine[Any, Any, None]) -> None:
    """Run a socket's loops together; the first to end (a hang-up) ends them all.

    A ``TaskGroup``, not ``asyncio.wait`` plus a hand-rolled cancel: the server (or
    Starlette's test client) may cancel the handler from outside at any moment, and
    only the task group keeps that cancellation intact while it stops the loops.
    """
    try:
        async with asyncio.TaskGroup() as group:
            for loop in loops:
                group.create_task(_ending(loop))
    except* _HungUp:
        pass


class _Refusals:
    """Refuse one message on one socket: tell the client, log each reason once."""

    def __init__(self, socket: str, mailbox: Mailbox[Any]) -> None:
        self._socket = socket
        self._mailbox = mailbox
        self._logged: set[str] = set()

    def __call__(self, reason: str) -> None:
        self._mailbox.post(RefusedMessage(reason))
        if reason not in self._logged:
            self._logged.add(reason)
            log.warning("refused a message on %s: %s", self._socket, reason)


def _parse_preview(jpeg: bytes) -> PreviewMessage:
    if len(jpeg) > MAX_PREVIEW_BYTES:
        raise ValueError(f"preview larger than {MAX_PREVIEW_BYTES} bytes")
    try:
        image = decode_jpeg(jpeg)
    except ValueError as e:
        raise ValueError("preview is not a JPEG") from e
    height, width = image.shape[:2]
    return PreviewMessage(jpeg=jpeg, width=width, height=height)


def router(
    registry: CameraRegistry,
    status: Callable[[str], NodeStatus],
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """The control sockets. ``status`` reads the node's own facts per board."""
    api = APIRouter()

    @api.websocket("/cameras/{device_id}")
    async def phone(ws: WebSocket, device_id: str) -> None:
        await ws.accept()
        try:
            checked_device_id(device_id)
        except ValueError as e:
            await ws.send_text(RefusedMessage(str(e)).to_json())
            await ws.close(code=1008)
            return
        link = registry.connect_phone(device_id)
        refuse = _Refusals(f"/cameras/{device_id[:8]}", link.mailbox)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    jpeg = message.get("bytes")
                    if jpeg is not None:
                        registry.preview(link, _parse_preview(jpeg))
                    else:
                        report = parse_phone_message(message.get("text") or "").report
                        registry.report(link, report)
                except ValueError as e:
                    refuse(str(e))

        try:
            await _until_first_ends(receive(), _deliver(ws, link.mailbox))
        finally:
            registry.disconnect_phone(link)

    @api.websocket("/boards/{board_id}/camera")
    async def outdated_phone(ws: WebSocket, board_id: str) -> None:
        """A /camera page from before the phones had device ids: tell it to reload."""
        await ws.accept()
        await ws.send_text(RefusedMessage("this /camera page is out of date: reload it").to_json())
        await ws.close(code=1008)

    @api.websocket("/cameras")
    async def camera_list(ws: WebSocket) -> None:
        await ws.accept()
        changed = registry.connect_list()
        mailbox: Mailbox[CamerasDown] = Mailbox()
        refuse = _Refusals("/cameras", mailbox)

        def frame_age(board_id: str) -> float | None:
            frame = status(board_id).frame
            return None if frame is None else max(0.0, clock() - frame.received_at)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    asked = parse_cameras_message(message.get("text") or "")
                    if isinstance(asked, ForgetMessage):
                        registry.forget(asked.device_id)
                    else:
                        registry.rename(asked.device_id, asked.name)
                except ValueError as e:
                    refuse(str(e))

        async def push() -> None:
            """The list on connect, on every change, and once a second for the ages."""
            while True:
                changed.clear()
                mailbox.post(registry.cameras(frame_age))
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(HEALTH_EVERY_S):
                        await changed.wait()

        try:
            await _until_first_ends(receive(), _deliver(ws, mailbox), push())
        finally:
            registry.disconnect_list(changed)

    @api.websocket("/boards/{board_id}/control")
    async def control(ws: WebSocket, board_id: str) -> None:
        await ws.accept()
        mailbox = registry.connect_watcher(board_id)
        refuse = _Refusals(f"/boards/{board_id}/control", mailbox)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    refuse("the control socket takes JSON text messages only")
                    continue
                try:
                    registry.configure(board_id, parse_control_message(text).config)
                except ValueError as e:
                    refuse(str(e))

        async def tick() -> None:
            sample: StreamSample | None = None
            shown: float | None = None  # received_at of the last frame sent
            while True:
                node = status(board_id)
                now = clock()
                frame = node.frame
                current = None if frame is None else StreamSample(seq=frame.seq, at=now)
                mailbox.post(
                    HealthMessage(
                        recognizer=node.recognizer,
                        fallback=node.fallback,
                        received=node.received,
                        refused=node.refused,
                        frame_age_s=None if frame is None else max(0.0, now - frame.received_at),
                        fps=frame_rate(sample, current),
                        direction=registry.direction(board_id),
                    )
                )
                sample = current
                if frame is not None and frame.received_at != shown:
                    shown = frame.received_at
                    mailbox.post(
                        FrameMessage(
                            seq=frame.seq, jpeg=frame.jpeg, width=frame.width, height=frame.height
                        )
                    )
                await asyncio.sleep(HEALTH_EVERY_S)

        try:
            await _until_first_ends(receive(), _deliver(ws, mailbox), tick())
        finally:
            registry.disconnect_watcher(board_id, mailbox)

    return api
