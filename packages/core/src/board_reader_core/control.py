"""The camera control channel — how the inference node configures an edge camera.

DESIGN D11 "dumb camera" (2026-11-24): the phone only films and streams. What it
films — which lens, at what resolution, which part of the frame is sent, and whether
it streams at all — is set from the control UI on the laptop, held per board on the
inference node, and **pushed to the edge, which applies it**. The edge still crops
and measures motion on its own crop, so the L1->L2 wire (:mod:`board_reader_core.wire`)
is unchanged: frames arrive already cropped, exactly as before.

Three shapes make up the camera's state, each validated here once (they are wire
models, :mod:`.schema`) so neither end trusts the other's JSON:

* :class:`CameraConfig` — what the control UI chose.
* :class:`CameraReport` — what the camera actually is (its lenses, the frame it
  delivers, the state of its stream), so the control UI can offer real choices and
  say when the phone is held upright.
* :class:`CropRect` — the rectangle itself, in **fractions of the full camera
  frame**, so it survives a resolution change unchanged. Also the Python edge's
  static ``--roi``.

A phone is known by its **device id** (:data:`DeviceId`), a random id (a UUID) it
keeps in its browser, and a camera's **board id** is its device id: what the phone's
frames name on ``/ingest``, what every event, recording and per-board socket below is
keyed by. It never changes. Each camera also has a **name** (:data:`CameraName`), an
alias people read: the node names a new camera ``board-N``, the lowest N no other
camera's name uses, and the control UI may rename it; a rename changes nothing else.

They travel on four WebSockets. Every text message is a JSON object whose ``type``
names it:

``WS /cameras/{device_id}`` — the phone.
  * up: :class:`ReportMessage` (text), sent on connect and whenever the report
    changes; a whole-frame preview JPEG (binary), sent only while the node has
    asked for previews; the answer to :class:`SendFullFrameMessage`, a
    :class:`FullFrameMessage` (text) followed at once by its JPEG (binary). A binary
    message is that full frame when a :class:`FullFrameMessage` came right before
    it, and a preview otherwise.
  * down: :class:`AssignedMessage` (its board id and name) on connect and on every
    rename;
    :class:`ConfigMessage` on connect and on every change;
    :class:`PreviewsMessage` on connect and whenever the first /control page opens
    or the last one closes; :class:`SendFullFrameMessage` when Auto-frame wants
    the whole picture; :class:`ReplacedMessage` when a newer socket of the same
    device took over (the node then closes this one); :class:`RefusedMessage` for a
    message the node refused.

``WS /cameras`` — the control UI's list of every known camera.
  * up: :class:`RenameMessage`, :class:`ForgetMessage` (:data:`CamerasUp`).
  * down: :class:`CamerasMessage` on connect, on every change and about once a
    second (for the frame ages); :class:`RefusedMessage`.

``WS /boards/{id}/control`` — a control page.
  * up: :class:`ConfigureMessage`, the whole new config; :class:`ReframeMessage`,
    "re-frame now" (:data:`ControlUp`).
  * down: :class:`StateMessage` on connect and on every change of config, report or
    phone link; :class:`PreviewMessage` and :class:`FrameMessage` (images as
    base64); :class:`HealthMessage` about once a second; :class:`RefusedMessage`.

``WS /streams`` — the control UI's videos and live streams, each run by the node
as a ``board-reader-edge camera`` process that registers as a camera above.
  * up: :class:`AddStreamMessage`, :class:`RemoveStreamMessage` (:data:`StreamsUp`).
  * down: :class:`StreamsMessage` on connect, on every change and about once a
    second; :class:`RefusedMessage`.

Previews and frames carry their JPEG as standard base64 text, because a page can
show that as a ``data:`` URL as it stands; binary would need a second message to say
which picture it is.

Like the rest of core this is contract only: no transport, no storage.
"""

from __future__ import annotations

import base64
from typing import Annotated, Literal, Self

from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    StrictBool,
    StrictBytes,
    StrictFloat,
    StrictStr,
    ValidationError,
    WithJsonSchema,
    model_validator,
)

from board_reader_core.wiremodel import (
    BoardId,
    Message,
    Natural,
    Positive,
    Wire,
    adapter,
    checked,
    message,
    parse,
    tag,
    wire,
)

#: Capture resolutions the control UI offers, by name. The edge asks for these as
#: ``ideal`` constraints, so a camera that cannot deliver one gets the nearest it
#: can; the report then states what was delivered.
Resolution = Literal["720p", "1080p", "2160p"]
RESOLUTIONS: dict[Resolution, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2160p": (3840, 2160),
}
DEFAULT_RESOLUTION: Resolution = "1080p"

#: Which chess the board plays, for the settle guard on the inference node.
#: ``standard`` keeps a settled board from becoming more impossible than it is
#: (two kings of a colour, more than 16 pieces of a side, more promoted pieces than
#: pawns can have made); ``free`` settles whatever the reads say, for a position
#: with extra material or a variant. The raw read is the same either way.
Rules = Literal["standard", "free"]
DEFAULT_RULES: Rules = "standard"

#: The smallest side a crop may have, as a fraction of the frame. Below this the
#: rectangle is a slip of the mouse, not a board.
MIN_CROP_SIDE = 0.05

#: How the edge reports its stream (the browser edge's ``IngestStatus``).
IngestState = Literal["idle", "connecting", "streaming", "reconnecting"]

#: What kind of camera reports: a phone's ``/camera`` page, whose lens and
#: resolution /control chooses, or a stream (``board-reader-edge camera``: a video or
#: a live stream), which has one picture at one size, so /control offers neither.
CameraKind = Literal["phone", "stream"]

#: How many lenses a report may list; more is not a phone.
MAX_DEVICES = 32

#: A non-empty piece of text: a lens label, a reason.
Text = Annotated[StrictStr, Field(min_length=1)]
#: A fraction of the frame along one axis.
Start = Annotated[StrictFloat, Field(ge=0, le=1)]
Side = Annotated[StrictFloat, Field(ge=MIN_CROP_SIDE, le=1)]


@wire(json_schema_extra=checked("cropInside"))
class CropRect(Wire):
    """An axis-aligned rectangle in fractions of the full camera frame.

    ``x, y`` is the top-left corner; the rectangle lies wholly inside the frame
    (the ``cropInside`` check: ``x + w <= 1`` and ``y + h <= 1``).
    """

    x: Start
    y: Start
    w: Side
    h: Side

    @model_validator(mode="after")
    def _inside(self) -> Self:
        if self.x + self.w > 1 or self.y + self.h > 1:
            raise ValueError("crop must end inside the frame (x + w <= 1, y + h <= 1)")
        return self

    @classmethod
    def parse(cls, spec: str) -> CropRect:
        """``"x,y,w,h"`` -> a rectangle, as the edge CLI's ``--roi`` spells it."""
        parts = [p for p in spec.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError("needs 4 numbers: x,y,w,h (fractions of the frame)")
        try:
            x, y, w, h = (float(p) for p in parts)
        except ValueError as e:
            raise ValueError("needs 4 numbers: x,y,w,h (fractions of the frame)") from e
        return parse(cls, {"x": x, "y": y, "w": w, "h": h})

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` in pixels of a ``width`` x ``height`` frame.

        Rounded outward-safe: never empty, never past the frame's edge.
        """
        x0 = min(round(self.x * width), width - 1)
        y0 = min(round(self.y * height), height - 1)
        x1 = max(x0 + 1, min(round((self.x + self.w) * width), width))
        y1 = max(y0 + 1, min(round((self.y + self.h) * height), height))
        return x0, y0, x1, y1


@wire
class CameraConfig(Wire):
    """What the control UI chose for one board's camera.

    * ``crop`` — the part of the frame the edge sends; ``None`` sends the whole frame.
    * ``device`` — the lens, by the **label** the edge reported. Labels are the one
      identity a browser keeps across reloads (device ids are regenerated per
      session); ``None`` leaves the edge on its default rear camera.
    * ``resolution`` — the capture resolution asked for.
    * ``streaming`` — whether the edge streams frames to ``/ingest``.
    * ``rules`` — which chess the board plays (:data:`Rules`). The node reads it,
      the edge ignores it.
    * ``auto`` — whether the node frames the camera by itself (Auto-frame): it asks
      for a full frame, finds the board and sets ``crop``. A crop set by hand on
      /control switches it off. The node reads it, the edge ignores it.

    Omitted fields take their defaults, so ``{}`` is the default config, and a
    config saved before a field existed still loads.
    """

    crop: CropRect | None = None
    device: Text | None = None
    resolution: Resolution = DEFAULT_RESOLUTION
    streaming: StrictBool = False
    rules: Rules = DEFAULT_RULES
    auto: StrictBool = True


@wire
class CameraDevice(Wire):
    """One lens the edge can open."""

    id: StrictStr
    label: StrictStr


@wire
class FrameSize(Wire):
    """A frame's size in whole pixels."""

    width: Positive
    height: Positive

    @property
    def portrait(self) -> bool:
        """Held upright: the framing measurement lost a quarter of perfect boards."""
        return self.height > self.width


@wire
class CameraReport(Wire):
    """What the edge says about its camera.

    ``device`` is the label of the lens actually open and ``frame`` the size it
    actually delivers — both ``None`` until the camera is live. ``kind`` says which
    choices it offers (:data:`CameraKind`); a report without it is a phone's.
    """

    devices: Annotated[tuple[CameraDevice, ...], Field(max_length=MAX_DEVICES)] = ()
    device: StrictStr | None = None
    frame: FrameSize | None = None
    ingest: IngestState = "idle"
    kind: CameraKind = "phone"


# ---- messages on the two control sockets -------------------------------------


@message
class ReportMessage(Message):
    """Phone -> node: what the camera is now."""

    type: Literal["report"] = tag("report")
    report: CameraReport


#: The long side a full frame is sent at, at most: enough for ChessQ to find a board
#: that fills a sixth of the picture, and a few hundred kB as a JPEG.
FULL_FRAME_LONG_SIDE = 1920

#: A full frame larger than this is not one.
MAX_FULL_FRAME_BYTES = 8_000_000


@message
class FullFrameMessage(Message):
    """Phone -> node: the answer to :class:`SendFullFrameMessage` ``request``; the
    socket's next message is its JPEG (binary).

    The whole camera frame, uncropped, its long side at most
    :data:`FULL_FRAME_LONG_SIDE`, and ``capture_ts`` its capture time in epoch
    seconds, on the same clock as the frames on ``/ingest``.
    """

    type: Literal["full_frame"] = tag("full_frame")
    request: Natural
    capture_ts: StrictFloat


#: What the phone sends as text.
PhoneUp = Annotated[ReportMessage | FullFrameMessage, Field(discriminator="type")]


def parse_phone_message(text: str) -> ReportMessage | FullFrameMessage:
    """One text message from the phone's socket; ``ValueError`` says why not."""
    return parse(PhoneUp, text)


@message
class ConfigureMessage(Message):
    """Control page -> node: replace the board's config with this whole one."""

    type: Literal["configure"] = tag("configure")
    config: CameraConfig


@message
class ReframeMessage(Message):
    """Control page -> node: find the board again now (Auto-frame), and switch
    Auto-frame on if a crop set by hand had switched it off."""

    type: Literal["reframe"] = tag("reframe")


#: What a control page sends.
ControlUp = Annotated[ConfigureMessage | ReframeMessage, Field(discriminator="type")]


def parse_control_message(text: str) -> ConfigureMessage | ReframeMessage:
    """One text message from a control page's socket; ``ValueError`` says why not."""
    return parse(ControlUp, text)


@message
class ConfigMessage(Message):
    """Node -> phone: the config to apply."""

    type: Literal["config"] = tag("config")
    config: CameraConfig


@message
class PreviewsMessage(Message):
    """Node -> phone: send whole-frame previews (``on``) or stop.

    On exactly while at least one control page watches the board, so an
    unwatched phone spends no uplink on pictures nobody sees.
    """

    type: Literal["previews"] = tag("previews")
    on: StrictBool


@message
class SendFullFrameMessage(Message):
    """Node -> phone: send one whole camera frame now (:class:`FullFrameMessage`).

    Auto-frame asks for it when the camera registers or starts streaming, and when
    the camera moved. A phone whose camera is not live yet answers once it is; only
    the latest request needs an answer.
    """

    type: Literal["send_full_frame"] = tag("send_full_frame")
    request: Natural


@message
class RefusedMessage(Message):
    """Node -> either client: one message was refused; the socket stays open."""

    type: Literal["refused"] = tag("refused")
    reason: Text


#: What Auto-frame is doing for one camera, as /control shows it: nothing yet
#: (``idle``), waiting for the full frame or reading it (``locating``), the crop is
#: set (``framed``), framed but the recent previews showed no board
#: (``not-visible``: a person in view, a camera knocked away), the last full frame
#: showed no board and a retry waits (``no-board``), or off because the crop was
#: set by hand (``manual``).
AutoFrameStatus = Literal["idle", "locating", "framed", "not-visible", "no-board", "manual"]


@message
class StateMessage(Message):
    """Node -> control page: the board's camera as the node holds it.

    ``report`` is the phone's last report (kept after it disconnects, so the lens
    list survives a reload); ``phone_connected`` says whether its socket is open;
    ``autoframe`` what Auto-frame is doing (:data:`AutoFrameStatus`); ``watching``
    whether the node watches the camera's previews for a move, which it does once
    framed and only while the board fills enough of the picture to be found on a
    preview.
    """

    type: Literal["state"] = tag("state")
    config: CameraConfig
    report: CameraReport | None
    phone_connected: StrictBool
    autoframe: AutoFrameStatus
    watching: StrictBool


def _from_base64(value: object) -> object:
    return base64.b64decode(value, validate=True) if isinstance(value, str) else value


#: A JPEG's bytes; standard base64 text on the wire (see the module note).
Jpeg = Annotated[
    StrictBytes,
    BeforeValidator(_from_base64),
    Field(min_length=1),
    PlainSerializer(lambda jpeg: base64.b64encode(jpeg).decode("ascii"), when_used="json"),
    WithJsonSchema({"type": "string", "contentEncoding": "base64", "minLength": 1}),
]


@message
class PreviewMessage(Message):
    """Node -> control page: the phone's latest whole-frame preview."""

    type: Literal["preview"] = tag("preview")
    jpeg: Jpeg
    width: Positive
    height: Positive


@message
class FrameMessage(Message):
    """Node -> control page: the latest frame ``/ingest`` received for the board."""

    type: Literal["frame"] = tag("frame")
    seq: Natural
    jpeg: Jpeg
    width: Positive
    height: Positive


#: Which edge of the image White's back rank lies along.
WhiteSide = Literal["bottom", "left", "top", "right"]


@wire
class BoardDirection(Wire):
    """Which way the board lies in the camera's image, as the recognizer reads it.

    ``degrees`` is the direction of the board's e1->e8 axis in the image, counted
    so that 0 means White sits at the bottom (e1->e8 points up), 90 White on the
    left (e1->e8 points right), 180 (or -180) White at the top and -90 White on the
    right; ``white_side`` is the nearest of those four. ``camera_moved`` says the
    angle has drifted well away from where it settled after the last crop, lens or
    resolution change: the camera (or the board) was moved.
    """

    degrees: Annotated[StrictFloat, Field(ge=-180, le=180)]
    white_side: WhiteSide
    camera_moved: StrictBool


#: A duration or a rate, never negative.
NonNegative = Annotated[StrictFloat, Field(ge=0)]


@message
class HealthMessage(Message):
    """Node -> control page, about once a second: the recognizer and the stream.

    Ages and rates are measured on the node's clock, so the page needs none: the
    phone, the laptop and the node need not agree on the time.
    """

    type: Literal["health"] = tag("health")
    recognizer: StrictStr
    #: Why the stub serves instead of the model, when it does (DESIGN D2).
    fallback: StrictStr | None
    #: Node-wide ``/ingest`` counters, as ``/healthz`` has them.
    received: Natural
    refused: Natural
    #: Seconds since the board's last frame arrived; ``None`` before the first.
    frame_age_s: NonNegative | None
    #: Frames per second received since the previous health message, if measurable.
    fps: NonNegative | None
    #: How the board lies in the image; ``None`` before the first read with one.
    direction: BoardDirection | None


#: What the node sends a control page.
ControlDown = Annotated[
    StateMessage | PreviewMessage | FrameMessage | HealthMessage | RefusedMessage,
    Field(discriminator="type"),
]


# ---- camera identity and the list of cameras ----------------------------------

#: A phone's own id: random, made and kept by its browser (``crypto.randomUUID``), and
#: the board id the camera streams as. 8 to 64 letters, digits or ``-``, starting with
#: a letter or a digit, so every device id is also a :data:`CameraBoardId`.
DeviceId = Annotated[StrictStr, Field(pattern=r"^[0-9A-Za-z][0-9A-Za-z-]{7,63}$")]

#: A board id a control page may open: a :data:`BoardId` that is also one plain URL
#: path segment (letters, digits, ``.``, ``_``, ``-``; up to 64), because every
#: per-board route (``/boards/{id}/control``, ``/boards/{id}/frame``) carries it. A
#: camera's board id is its :data:`DeviceId`; an edge without a device id
#: (``board-reader-edge feed --board X``) names its own.
CameraBoardId = Annotated[
    BoardId, Field(max_length=64, pattern=r"^[0-9A-Za-z][0-9A-Za-z._-]*$")
]


#: What people call a camera (DESIGN D18): an alias for its board id, shown on the live
#: page, on /control and in the PGN. A new camera gets ``board-N`` (the lowest N no
#: other camera's name uses). Not part of any route, so it may hold spaces:
#: up to 32 letters, digits, spaces, ``.``, ``_`` or ``-``, starting and ending with
#: no space. ASCII only, because the same pattern must judge alike in Python and in
#: the browser (no Unicode classes there).
CameraName = Annotated[
    StrictStr,
    Field(min_length=1, max_length=32, pattern=r"^[0-9A-Za-z._-]([0-9A-Za-z ._-]*[0-9A-Za-z._-])?$"),
]


def device_id(raw: str) -> str:
    """``raw`` as a :data:`DeviceId`; ``ValueError`` if it is none."""
    try:
        return adapter(DeviceId).validate_python(raw)
    except ValidationError:
        raise ValueError(f"{raw!r} is not a device id") from None


def camera_board_id(name: str) -> str:
    """``name`` as a :data:`CameraBoardId`; ``ValueError`` says why not, in words.

    The reason goes to an operator who typed a name, so it states the rule rather
    than the pattern.
    """
    try:
        return adapter(CameraBoardId).validate_python(name)
    except ValidationError:
        raise ValueError(
            f"{name!r} is not a board id: use up to 64 letters, digits, '.', '_' or '-', "
            "starting with a letter or a digit"
        ) from None


def camera_name(name: str) -> str:
    """``name`` as a :data:`CameraName`; ``ValueError`` says why not, in words."""
    try:
        return adapter(CameraName).validate_python(name)
    except ValidationError:
        raise ValueError(
            f"{name!r} is not a camera name: use up to 32 letters, digits, spaces, "
            "'.', '_' or '-', with no space at either end"
        ) from None


@message
class AssignedMessage(Message):
    """Node -> phone: the board id this phone streams as (its device id), and its name.

    The board id is the device's for good; ``name`` changes when /control renames it.
    """

    type: Literal["assigned"] = tag("assigned")
    board_id: DeviceId
    name: CameraName


@message
class ReplacedMessage(Message):
    """Node -> phone: a newer socket of this same device took over; this one closes.

    The phone stops streaming and does not reconnect, or two pages of one phone
    would take the camera from each other forever.
    """

    type: Literal["replaced"] = tag("replaced")


@wire
class CameraEntry(Wire):
    """One known camera, as the control UI lists it.

    ``device_id`` is also its board id; ``name`` what people call it; ``report`` the
    phone's last report (kept after it disconnects); ``frame_age_s`` the seconds since
    its board's last frame, on the node's clock.
    """

    device_id: DeviceId
    name: CameraName
    connected: StrictBool
    report: CameraReport | None
    frame_age_s: NonNegative | None


@message
class CamerasMessage(Message):
    """Node -> control UI: every known camera, ordered by name."""

    type: Literal["cameras"] = tag("cameras")
    cameras: tuple[CameraEntry, ...]


@message
class RenameMessage(Message):
    """Control UI -> node: give this device's camera a name; ``None`` gives it a
    default name again (the lowest free ``board-N``).

    The board id stays. ``name`` is checked by the node (:data:`CameraName`, not
    another camera's name or board id), which refuses with a reason an operator can
    read.
    """

    type: Literal["rename"] = tag("rename")
    device_id: DeviceId
    name: Text | None


@message
class ForgetMessage(Message):
    """Control UI -> node: forget this disconnected camera, its name and config.

    Refused while the camera is connected. The device, if it comes back, is a new
    camera under the same board id, with the defaults and a new default name.
    """

    type: Literal["forget"] = tag("forget")
    device_id: DeviceId


#: What the control UI's camera list sends.
CamerasUp = Annotated[RenameMessage | ForgetMessage, Field(discriminator="type")]


def parse_cameras_message(text: str) -> RenameMessage | ForgetMessage:
    """One text message from a ``/cameras`` socket; ``ValueError`` says why not."""
    return parse(CamerasUp, text)


#: What the node sends the phone.
PhoneDown = Annotated[
    AssignedMessage
    | ConfigMessage
    | PreviewsMessage
    | SendFullFrameMessage
    | ReplacedMessage
    | RefusedMessage,
    Field(discriminator="type"),
]
#: What the node sends the control UI's camera list.
CamerasDown = Annotated[CamerasMessage | RefusedMessage, Field(discriminator="type")]


# ---- streams the node runs as cameras ------------------------------------------

#: The longest stream URL the node takes.
MAX_STREAM_URL = 2048

#: A video or live stream page the node hands ``board-reader-edge camera`` (yt-dlp
#: resolves it): ``http`` or ``https`` only, printable ASCII without spaces (a browser
#: percent-encodes the rest), up to :data:`MAX_STREAM_URL` characters.
StreamUrl = Annotated[
    StrictStr, Field(max_length=MAX_STREAM_URL, pattern=r"^https?://[!-~]+$")
]

#: A stream's id on the node: ``stream-1``, ``stream-2``, ...
StreamId = Annotated[StrictStr, Field(pattern=r"^stream-[1-9][0-9]{0,8}$")]

#: What a stream's process is doing: started and not yet registered as a camera
#: (``starting``), registered and connected (``running``), exited with an error and
#: waiting to be started again (``crashed``), or not running and not restarted
#: (``stopped``: the video ended, or the node has no edge command to run).
StreamState = Literal["starting", "running", "crashed", "stopped"]


@wire
class StreamEntry(Wire):
    """One stream the node runs as a camera.

    ``start_s`` is a video's start position in seconds (``None``: the beginning, and
    the only choice for a live stream); ``device_id`` the camera the stream registers
    as (and so its board id); ``name`` that camera's name (``None`` until it first
    registered); ``error`` why it last exited or could not start.
    """

    id: StreamId
    url: StreamUrl
    start_s: NonNegative | None
    state: StreamState
    device_id: DeviceId
    name: CameraName | None
    error: Text | None


@message
class AddStreamMessage(Message):
    """Control UI -> node: run this video or live stream as a new camera."""

    type: Literal["add_stream"] = tag("add_stream")
    url: StreamUrl
    start_s: NonNegative | None = None


@message
class RemoveStreamMessage(Message):
    """Control UI -> node: stop this stream and forget its camera (its name and
    config)."""

    type: Literal["remove_stream"] = tag("remove_stream")
    stream_id: StreamId


@message
class StreamsMessage(Message):
    """Node -> control UI: every stream the node runs, in the order they were added."""

    type: Literal["streams"] = tag("streams")
    streams: tuple[StreamEntry, ...]


#: What the control UI sends on ``WS /streams``.
StreamsUp = Annotated[AddStreamMessage | RemoveStreamMessage, Field(discriminator="type")]
#: What the node sends on ``WS /streams``.
StreamsDown = Annotated[StreamsMessage | RefusedMessage, Field(discriminator="type")]


def parse_streams_message(text: str) -> AddStreamMessage | RemoveStreamMessage:
    """One text message from a ``/streams`` socket; ``ValueError`` says why not."""
    return parse(StreamsUp, text)
