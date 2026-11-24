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

They travel on two WebSockets, one per board and per kind of client. Every text
message is a JSON object whose ``type`` names it:

``WS /boards/{id}/camera`` — the phone.
  * up: :class:`ReportMessage` (text), sent on connect and whenever the report
    changes; a whole-frame preview JPEG (binary, the socket's only binary message),
    sent only while the node has asked for previews.
  * down: :class:`ConfigMessage` on connect and on every change;
    :class:`PreviewsMessage` on connect and whenever the first /control page opens
    or the last one closes; :class:`RefusedMessage` for a message the node refused.

``WS /boards/{id}/control`` — a control page.
  * up: :class:`ConfigureMessage`, the whole new config.
  * down: :class:`StateMessage` on connect and on every change of config, report or
    phone link; :class:`PreviewMessage` and :class:`FrameMessage` (images as
    base64); :class:`HealthMessage` about once a second; :class:`RefusedMessage`.

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
    WithJsonSchema,
    model_validator,
)

from board_reader_core.wiremodel import (
    Message,
    Natural,
    Positive,
    Wire,
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

    Omitted fields take their defaults, so ``{}`` is the default config, and a
    config saved before a field existed still loads.
    """

    crop: CropRect | None = None
    device: Text | None = None
    resolution: Resolution = DEFAULT_RESOLUTION
    streaming: StrictBool = False
    rules: Rules = DEFAULT_RULES


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
    actually delivers — both ``None`` until the camera is live.
    """

    devices: Annotated[tuple[CameraDevice, ...], Field(max_length=MAX_DEVICES)] = ()
    device: StrictStr | None = None
    frame: FrameSize | None = None
    ingest: IngestState = "idle"


# ---- messages on the two control sockets -------------------------------------


@message
class ReportMessage(Message):
    """Phone -> node: what the camera is now."""

    type: Literal["report"] = tag("report")
    report: CameraReport


def parse_phone_message(text: str) -> ReportMessage:
    """One text message from the phone's socket; ``ValueError`` says why not."""
    return parse(ReportMessage, text)


@message
class ConfigureMessage(Message):
    """Control page -> node: replace the board's config with this whole one."""

    type: Literal["configure"] = tag("configure")
    config: CameraConfig


def parse_control_message(text: str) -> ConfigureMessage:
    """One text message from a control page's socket; ``ValueError`` says why not."""
    return parse(ConfigureMessage, text)


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
class RefusedMessage(Message):
    """Node -> either client: one message was refused; the socket stays open."""

    type: Literal["refused"] = tag("refused")
    reason: Text


@message
class StateMessage(Message):
    """Node -> control page: the board's camera as the node holds it.

    ``report`` is the phone's last report (kept after it disconnects, so the lens
    list survives a reload); ``phone_connected`` says whether its socket is open.
    """

    type: Literal["state"] = tag("state")
    config: CameraConfig
    report: CameraReport | None
    phone_connected: StrictBool


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


#: What the node sends the phone.
PhoneDown = Annotated[ConfigMessage | PreviewsMessage | RefusedMessage, Field(discriminator="type")]
#: What the node sends a control page.
ControlDown = Annotated[
    StateMessage | PreviewMessage | FrameMessage | HealthMessage | RefusedMessage,
    Field(discriminator="type"),
]
