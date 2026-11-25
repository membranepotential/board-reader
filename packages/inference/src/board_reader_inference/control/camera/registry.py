"""Every board's camera state, every known camera, and who may stream as which board.

The node holds, per ``board_id``, what the control UI chose for that board's camera
(:class:`~board_reader_core.control.CameraConfig`), the last thing the camera said
about itself (:class:`~board_reader_core.control.CameraReport`) and the latest
full-frame preview, so the control UI can show the camera's whole field of view
while the stream itself carries only the crop.

Each phone is known by its **device id**, a random id its browser keeps, and its
board id **is** that device id (DESIGN D18), so the same phone streams as the same
board after a reload, a reconnect or a node restart, and every event, recording and
PGN stays under one id. Every camera has a **name**, an alias the live page,
/control and the phone show instead of the id: a new camera gets the lowest
``board-N`` no other camera's name uses, and the control UI may rename it; a rename
changes nothing else. Names are unique among the known cameras and never another
camera's board id, so no two cards read alike. The control UI may also **forget** a
disconnected camera: its name and config are dropped; the device, if it comes back,
is a new camera under the same board id, with a new default name.

The registry also decides which ``/ingest`` frames a board id accepts
(:meth:`CameraRegistry.ingest_refusal`): a camera's board id takes frames from that
camera only, so a stale page or a replay cannot stream into a camera's board.

The configs and the names can outlive a restart
(:mod:`~board_reader_inference.control.camera.persistence`).

The registry also follows each board's **direction** in the image, from the
recognizer's reads (:meth:`CameraRegistry.observe_direction`): which side White sits
on, and whether the camera moved since the angle settled.

And it runs **Auto-frame** for each board (DESIGN D21): it steps the board's
:class:`~.autoframe.state.AutoFrame` controller on the camera's events (registered,
streaming, disconnected, a crop set by hand, the switch on /control, a new lens or
frame size, the node's ~1 s tick) and carries out its actions: it asks the phone for
a full frame, pushes the crop the node found, and tells /control the status. Finding
the board needs the recognizer, which ``control`` never imports, so the full frame
goes to the gateway (``on_full_frame``), which locates it on the read worker and
hands the result back (:meth:`CameraRegistry.autoframe_result`), together with the
read to publish when there is one.

The registry is the single source of truth, and it **pushes**: every change is
posted at once to the :class:`~board_reader_inference.control.camera.mailbox.Mailbox`
of every socket that cares (the sockets are
:mod:`board_reader_inference.control.camera.router`'s).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import replace
from itertools import count

from board_reader_core.control import (
    DEFAULT_RULES,
    MAX_FULL_FRAME_BYTES,
    AssignedMessage,
    BoardDirection,
    CameraConfig,
    CameraEntry,
    CameraReport,
    CamerasMessage,
    ConfigMessage,
    ControlDown,
    FullFrameMessage,
    PhoneDown,
    PreviewMessage,
    PreviewsMessage,
    ReplacedMessage,
    Rules,
    SendFullFrameMessage,
    StateMessage,
    camera_name,
)
from board_reader_core.protocol import BoardName, NamesMessage

from board_reader_inference.control.autoframe.actions import (
    PublishRead,
    PushCrop,
    ReportStatus,
    RequestFullFrame,
)
from board_reader_inference.control.autoframe.events import (
    AutoToggled,
    CameraRegistered,
    Disconnected,
    Event,
    FullFrameFailed,
    FullFrameLocated,
    ManualCropSet,
    PreviewLocated,
    Reframed,
    StreamingStarted,
    Tick,
)
from board_reader_inference.control.autoframe.geometry import Located
from board_reader_inference.control.autoframe.machine import (
    status,
    step,
    wants_preview,
    watching,
)
from board_reader_inference.control.autoframe.state import AutoFrame
from board_reader_inference.control.autoframe.tuning import TUNING, Tuning
from board_reader_inference.control.camera.direction import DirectionTrack
from board_reader_inference.control.camera.mailbox import Mailbox
from board_reader_inference.control.camera.persistence import (
    Saved,
    config_path,
    load_state,
    save_state,
)
from board_reader_inference.control.camera.records import BoardCamera, FullFrame, PhoneLink

log = logging.getLogger(__name__)

#: The name a new camera gets: this prefix and the lowest number no other camera's
#: name uses.
DEFAULT_NAME_PREFIX = "board-"


def _reframed(old: CameraConfig, new: CameraConfig) -> bool:
    """Whether ``new`` changes what the camera films (not just whether it streams)."""
    return (old.crop, old.device, old.resolution) != (new.crop, new.device, new.resolution)


def _new_picture(before: CameraReport | None, after: CameraReport) -> bool:
    """Whether the camera's whole frame became another picture: a live camera
    switched lens or frame size (a new resolution, the phone turned upright)."""
    if before is None or before.frame is None or after.frame is None:
        return False
    return (before.device, before.frame) != (after.device, after.frame)


def _natural(name: str) -> list[int | str]:
    """``board-2`` before ``board-10``: digit runs compare as numbers."""
    return [int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", name)]


class CameraRegistry:
    """Every board's camera state, every known camera, and the sockets they are pushed to.

    A board exists from the first time anyone asks. A camera (a device id, also its
    board id) exists from its first connect until it is forgotten. Methods are
    synchronous and run on the event loop; they only post to mailboxes (and wake the
    camera lists), and each socket's own task sends.

    With ``BR_CAMERA_CONFIG`` set, the configs and the cameras are read from that
    file here and written back on every change (see the module docstring). A saved
    config no camera holds (a board named by hand before the device ids) is dropped
    at load, so no new camera can inherit it.

    ``on_names`` hears the whole :class:`NamesMessage` now and on every change of a
    name or camera (the gateway hands it to ``/viewer``); ``on_forget`` hears the board id of
    each forgotten camera (the gateway frees that board's read state);
    ``on_full_frame`` hears each full frame Auto-frame asked for (the gateway locates
    the board in it and answers with :meth:`autoframe_result`); ``on_preview`` hears
    each preview worth locating for a camera move (the gateway answers with
    :meth:`preview_result`). ``clock`` is the monotonic clock Auto-frame's timeouts,
    retries and preview pacing run on; ``tuning`` its thresholds.
    """

    def __init__(
        self,
        *,
        on_names: Callable[[NamesMessage], None] | None = None,
        on_forget: Callable[[str], None] | None = None,
        on_full_frame: Callable[[FullFrame], None] | None = None,
        on_preview: Callable[[str, PreviewMessage], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        tuning: Tuning = TUNING,
    ) -> None:
        self._path = config_path()
        saved = Saved() if self._path is None else load_state(self._path)
        held = set(saved.names)
        orphans = sorted(set(saved.boards) - held)
        if orphans:
            log.info(
                "dropping the saved config of %s: no camera holds %s",
                ", ".join(orphans), "it" if len(orphans) == 1 else "them",
            )
        self._boards: dict[str, BoardCamera] = {
            board_id: BoardCamera(config=config, autoframe=AutoFrame(auto=config.auto))
            for board_id, config in saved.boards.items()
            if board_id in held
        }
        #: device id (= board id) -> its name, one entry per known camera (persisted).
        self._names: dict[str, str] = dict(saved.names)
        #: device id -> its one open socket.
        self._phones: dict[str, PhoneLink] = {}
        #: One wake-up per open camera list: something it shows changed.
        self._lists: set[asyncio.Event] = set()
        self._on_names = on_names
        self._on_forget = on_forget
        self._on_full_frame = on_full_frame
        self._on_preview = on_preview
        self._clock = clock
        self._tuning = tuning
        self._announce_names()

    def board(self, board_id: str) -> BoardCamera:
        return self._boards.setdefault(board_id, BoardCamera())

    def rules(self, board_id: str) -> Rules:
        """Which chess ``board_id`` plays, as its settle guard reads it.

        A board nobody configured plays standard chess; asking does not create it.
        """
        board = self._boards.get(board_id)
        return DEFAULT_RULES if board is None else board.config.rules

    def name_of(self, device_id: str) -> str | None:
        """What ``device_id``'s camera is called; ``None`` for a device not known."""
        return self._names.get(device_id)

    def _phone_on(self, board_id: str) -> PhoneLink | None:
        return self._phones.get(board_id)

    def _state(self, board_id: str) -> StateMessage:
        board = self.board(board_id)
        return StateMessage(
            config=board.config,
            report=board.report,
            phone_connected=self._phone_on(board_id) is not None,
            autoframe=status(board.autoframe, self._tuning).value,
            watching=watching(board.autoframe, self._tuning),
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

    def _taken(self, device_id: str) -> set[str]:
        """What no other camera than ``device_id`` may be called: the others' names
        and board ids."""
        others = {other: name for other, name in self._names.items() if other != device_id}
        return set(others) | set(others.values())

    def _default_name(self, device_id: str) -> str:
        """The lowest ``board-N`` no other camera is called."""
        taken = self._taken(device_id)
        return next(
            name for n in count(1) if (name := f"{DEFAULT_NAME_PREFIX}{n}") not in taken
        )

    def label(self, board_id: str) -> str:
        """What people call ``board_id``: its camera's name, or the id itself."""
        return self._names.get(board_id, board_id)

    def log_label(self, board_id: str) -> str:
        """``board_id`` for a log line: the whole id, then its camera's name, if any,
        so a log can be searched by id and still read by name."""
        name = self._names.get(board_id)
        return board_id if name is None else f"{board_id} ({name})"

    def names(self) -> NamesMessage:
        """Every camera's name, by board id, in name order."""
        return NamesMessage(
            names=tuple(
                BoardName(device_id, name)
                for device_id, name in sorted(self._names.items(), key=lambda kv: _natural(kv[1]))
            )
        )

    def _announce_names(self) -> None:
        if self._on_names is not None:
            self._on_names(self.names())

    def rename(self, device_id: str, name: str | None) -> None:
        """Give a known camera a name; ``None`` gives it a default name again (the
        lowest ``board-N`` no other camera is called).

        Only the alias changes: the board id, its config, its events and recordings
        stay. Refused (``ValueError``) for an unknown device, a name that is no
        :data:`CameraName`, or one another camera has as its name or board id.
        """
        current = self._names.get(device_id)
        if current is None:
            raise ValueError(f"no camera {device_id[:8]} is known")
        wanted = self._default_name(device_id) if name is None else camera_name(name)
        if wanted in self._taken(device_id):
            holder = next(d for d, n in self._names.items() if wanted in (d, n))
            raise ValueError(f"{wanted} is taken by camera {self.label(holder)}")
        if wanted == current:
            return
        self._names[device_id] = wanted
        self._save()
        phone = self._phones.get(device_id)
        if phone is not None:
            phone.mailbox.post(AssignedMessage(device_id, wanted))
        self._tell_lists()
        self._announce_names()

    def forget(self, device_id: str) -> None:
        """Drop a disconnected camera: its name and config.

        The board's control pages stay open on a board with the default config. A
        device that comes back is a new camera under the same board id, with the
        default config and a new default name. Refused (``ValueError``) for an
        unknown device or a connected one: a connected phone would only be known
        again at once.
        """
        if device_id not in self._names:
            raise ValueError(f"no camera {device_id[:8]} is known")
        if device_id in self._phones:
            raise ValueError(
                f"camera {self.label(device_id)} is connected: close its /camera page, "
                "then forget it"
            )
        name = self._names.pop(device_id)
        board_id = device_id
        watchers = self.board(board_id).watchers
        if watchers:
            self._boards[board_id] = BoardCamera(watchers=watchers)
        else:
            del self._boards[board_id]
        self._save()
        log.info("forgot camera %s (%s)", device_id, name)
        self._tell_watchers(board_id)
        self._tell_lists()
        self._announce_names()
        if self._on_forget is not None:
            self._on_forget(board_id)

    # ---- the frames --------------------------------------------------------------

    def ingest_refusal(self, board_id: str, device_id: str | None) -> str | None:
        """Why ``/ingest`` must refuse a frame under ``board_id`` from ``device_id``.

        ``None`` accepts it. A camera's board id (its device id) takes frames from
        that camera only. A frame from a device the node does not know (forgotten, or
        one that never opened its control socket) is refused, because its board id
        was never handed to it. An edge without a device id (``board-reader-edge
        feed``, a replayed recording) may stream under any board id no camera holds.
        """
        if device_id is None:
            if board_id not in self._names:
                return None
            return (
                f"board {board_id[:8]} belongs to camera {self.label(board_id)}; an edge "
                "without a device id may stream only under a board id no camera holds"
            )
        if device_id not in self._names:
            return f"camera {device_id[:8]} is not known to the node (forgotten?): reload /camera"
        if device_id != board_id:
            return (
                f"camera {self.label(device_id)} streams as its device id, not {board_id}: "
                "reload /camera"
            )
        return None

    # ---- the phone ---------------------------------------------------------------

    def connect_phone(self, device_id: str) -> PhoneLink:
        """Open ``device_id``'s socket; a socket it already had is replaced.

        A device seen for the first time is a new camera: it gets the default name
        (the lowest ``board-N`` no other camera is called).
        """
        previous = self._phones.get(device_id)
        if previous is not None:
            previous.mailbox.post(ReplacedMessage())
        link = PhoneLink(device_id)
        self._phones[device_id] = link
        board_id = device_id
        if device_id not in self._names:
            self._names[device_id] = self._default_name(device_id)
            self._save()
            self._announce_names()
        board = self.board(board_id)
        link.mailbox.post(AssignedMessage(board_id, self._names[device_id]))
        link.mailbox.post(ConfigMessage(board.config))
        board.previews = self._previews_wanted(board)
        link.mailbox.post(PreviewsMessage(on=board.previews))
        self._tell_watchers(board_id)
        self._tell_lists()
        if previous is not None:
            # The replaced socket cannot answer the full frame it was asked for.
            self._step(board_id, Disconnected())
        self._step(board_id, CameraRegistered())
        return link

    def disconnect_phone(self, link: PhoneLink) -> None:
        if not self._current(link):
            return  # replaced: the newer socket is the device's now
        del self._phones[link.device_id]
        board_id = link.device_id
        self._tell_watchers(board_id)
        self._tell_lists()
        self._step(board_id, Disconnected())

    def report(self, link: PhoneLink, report: CameraReport) -> None:
        if not self._current(link):
            return
        board_id = link.device_id
        board = self.board(board_id)
        before, board.report = board.report, report
        self._tell_watchers(board_id)
        self._tell_lists()
        if _new_picture(before, report):
            self._step(board_id, Reframed())
        if report.ingest == "streaming" and (before is None or before.ingest != "streaming"):
            self._step(board_id, StreamingStarted())

    def full_frame(self, link: PhoneLink, header: FullFrameMessage, jpeg: bytes) -> None:
        """The phone's answer to a full-frame request; handed to ``on_full_frame``.

        An answer to a request that is no longer the one in flight (late, or made
        stale by a new lens) is dropped here, before it costs a locate.
        """
        if len(jpeg) > MAX_FULL_FRAME_BYTES:
            raise ValueError(f"full frame larger than {MAX_FULL_FRAME_BYTES} bytes")
        if not self._current(link):
            return
        board_id = link.device_id
        inflight = self.board(board_id).autoframe.inflight
        if inflight is None or inflight.request != header.request:
            log.info(
                "board %s: dropping full frame %d, not the one asked for",
                self.log_label(board_id),
                header.request,
            )
            return
        if self._on_full_frame is not None:
            self._on_full_frame(FullFrame(board_id, header.request, header.capture_ts, jpeg))

    def preview(self, link: PhoneLink, preview: PreviewMessage) -> None:
        if not self._current(link):
            return
        board_id = link.device_id
        board = self.board(board_id)
        board.preview = preview
        self._tell_watchers(board_id, preview)
        if (
            self._on_preview is not None
            and not board.locating_preview
            and wants_preview(board.autoframe, self._clock(), self._tuning)
        ):
            board.locating_preview = True
            self._on_preview(board_id, preview)

    # ---- the camera lists --------------------------------------------------------

    def connect_list(self) -> asyncio.Event:
        """A wake-up for one open camera list, set whenever what it shows changes."""
        changed = asyncio.Event()
        self._lists.add(changed)
        return changed

    def disconnect_list(self, changed: asyncio.Event) -> None:
        self._lists.discard(changed)

    def cameras(self, frame_age: Callable[[str], float | None]) -> CamerasMessage:
        """Every known camera, ordered by name; ``frame_age`` reads a board's."""
        entries = []
        for device_id, name in sorted(self._names.items(), key=lambda kv: _natural(kv[1])):
            board = self._boards.get(device_id)
            entries.append(
                CameraEntry(
                    device_id=device_id,
                    name=name,
                    connected=device_id in self._phones,
                    report=None if board is None else board.report,
                    frame_age_s=frame_age(device_id),
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
        self._sync_previews(board_id)
        return mailbox

    def disconnect_watcher(self, board_id: str, mailbox: Mailbox[ControlDown]) -> None:
        board = self.board(board_id)
        board.watchers.discard(mailbox)
        self._sync_previews(board_id)

    def configure(self, board_id: str, config: CameraConfig) -> None:
        """/control set ``config``. A new crop there is a crop set by hand: it
        switches Auto-frame off, whatever ``auto`` says; otherwise ``auto`` is the
        switch."""
        old = self.board(board_id).config
        event: ManualCropSet | AutoToggled | None = None
        if config.crop != old.crop:
            config, event = replace(config, auto=False), ManualCropSet()
        elif config.auto != old.auto:
            event = AutoToggled(on=config.auto)
        self._set_config(board_id, config)
        if event is not None:
            self._step(board_id, event)

    def _set_config(self, board_id: str, config: CameraConfig) -> None:
        board = self.board(board_id)
        if _reframed(board.config, config):
            board.direction = DirectionTrack()
        changed = config != board.config
        board.config = config
        if changed:
            self._save()
        self._tell_phone(board_id, ConfigMessage(config))
        self._tell_watchers(board_id)

    # ---- Auto-frame ------------------------------------------------------------------

    def autoframe_result[R](
        self, board_id: str, result: FullFrameLocated[R] | FullFrameFailed
    ) -> list[PublishRead[R]]:
        """What the gateway found in a full frame of ``board_id``; returns the reads
        to publish (none, or Auto-frame's last read when it saw the board large
        enough). Nothing for a board forgotten meanwhile."""
        if board_id not in self._names:
            return []
        return self._step(board_id, result)

    def reframe(self, board_id: str) -> None:
        """/control asked to find the board again now; Auto-frame is switched on
        first if a crop set by hand had switched it off."""
        if self.board(board_id).config.auto:
            self._step(board_id, Reframed())
        else:
            self.configure(board_id, replace(self.board(board_id).config, auto=True))

    def preview_result(self, board_id: str, board: Located | None, aspect: float) -> None:
        """Where the gateway found the board in the preview :meth:`preview` handed
        it (``None``: no trusted board); ``aspect`` is the preview's width over its
        height."""
        camera = self._boards.get(board_id)
        if camera is None or board_id not in self._names:
            return  # forgotten meanwhile
        camera.locating_preview = False
        self._step(board_id, PreviewLocated(board, aspect))

    def preview_skipped(self, board_id: str) -> None:
        """The gateway located nothing for the preview (the read queue was busy):
        the next preview may go."""
        camera = self._boards.get(board_id)
        if camera is not None:
            camera.locating_preview = False

    def _previews_wanted(self, board: BoardCamera) -> bool:
        """Previews are wanted while a control page watches or Auto-frame watches
        for a camera move."""
        return bool(board.watchers) or watching(board.autoframe, self._tuning)

    def _sync_previews(self, board_id: str) -> None:
        board = self.board(board_id)
        wanted = self._previews_wanted(board)
        if wanted != board.previews:
            board.previews = wanted
            self._tell_phone(board_id, PreviewsMessage(on=wanted))

    def tick(self) -> None:
        """Let time pass for every camera: a retry or a timeout may be due."""
        for board_id in list(self._boards):
            if self._boards[board_id].autoframe.connected:
                self._step(board_id, Tick())

    def _step[R](self, board_id: str, event: Event[R]) -> list[PublishRead[R]]:
        """Step ``board_id``'s Auto-frame on ``event``, carry out its actions and
        return the reads it wants published."""
        board = self.board(board_id)
        board.autoframe, actions = step(board.autoframe, event, self._clock(), self._tuning)
        if board.autoframe.auto != board.config.auto:
            self._set_config(board_id, replace(board.config, auto=board.autoframe.auto))
        published: list[PublishRead[R]] = []
        for action in actions:
            match action:
                case RequestFullFrame(request=request):
                    self._tell_phone(board_id, SendFullFrameMessage(request))
                case PushCrop(crop=crop):
                    # The node's own crop: not a crop set by hand, Auto-frame stays on.
                    self._set_config(board_id, replace(board.config, crop=crop))
                case ReportStatus():
                    self._tell_watchers(board_id)
                case PublishRead():
                    published.append(action)
        self._sync_previews(board_id)
        return published

    def _save(self) -> None:
        if self._path is None:
            return
        configured = {
            board_id: board.config
            for board_id, board in self._boards.items()
            if board.config != CameraConfig()
        }
        try:
            save_state(self._path, Saved(configured, dict(self._names)))
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
