"""The camera control channel — how the inference node configures an edge camera.

DESIGN D11 "dumb camera" (2026-11-24): the phone only films and streams. What it
films — which lens, at what resolution, which part of the frame is sent, and whether
it streams at all — is set from the control UI on the laptop, held per board on the
inference node, and **pushed to the edge, which applies it**. The edge still crops
and measures motion on its own crop, so the L1->L2 wire (:mod:`board_reader_core.wire`)
is unchanged: frames arrive already cropped, exactly as before.

Three shapes make up the camera's state, each validated here once so neither end
trusts the other's JSON:

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

Like the rest of core this is contract only: no transport, no storage.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import ClassVar, Literal, cast, get_args

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

#: The smallest side a crop may have, as a fraction of the frame. Below this the
#: rectangle is a slip of the mouse, not a board.
MIN_CROP_SIDE = 0.05

#: How the edge reports its stream (the browser edge's ``IngestStatus``).
IngestState = Literal["idle", "connecting", "streaming", "reconnecting"]


def _number(raw: object, name: str) -> float:
    # ``bool`` is an ``int`` in Python; ``true`` is not a coordinate.
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{name} must be a number")
    return float(raw)


def _record(raw: object, name: str, keys: frozenset[str]) -> dict[str, object]:
    """``raw`` as a JSON object with no keys outside ``keys``.

    Unknown keys are refused rather than ignored: a misspelt ``"streming": true``
    silently doing nothing is exactly the bug a strict boundary exists to catch.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(raw) - keys
    if unknown:
        raise ValueError(f"{name}: unknown field(s) {', '.join(sorted(map(str, unknown)))}")
    return cast(dict[str, object], raw)


@dataclass(frozen=True, slots=True)
class CropRect:
    """An axis-aligned rectangle in fractions of the full camera frame.

    ``x, y`` is the top-left corner; the rectangle lies wholly inside the frame.
    """

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "w", "h"):
            _number(getattr(self, name), f"crop.{name}")
        if self.x < 0 or self.y < 0:
            raise ValueError("crop must start inside the frame (x, y >= 0)")
        if self.w < MIN_CROP_SIDE or self.h < MIN_CROP_SIDE:
            raise ValueError(f"crop sides must be at least {MIN_CROP_SIDE} of the frame")
        if self.x + self.w > 1 or self.y + self.h > 1:
            raise ValueError("crop must end inside the frame (x + w <= 1, y + h <= 1)")

    @classmethod
    def from_dict(cls, raw: object) -> CropRect:
        data = _record(raw, "crop", frozenset({"x", "y", "w", "h"}))
        missing = {"x", "y", "w", "h"} - set(data)
        if missing:
            raise ValueError(f"crop: missing field(s) {', '.join(sorted(missing))}")
        return cls(
            x=_number(data["x"], "crop.x"),
            y=_number(data["y"], "crop.y"),
            w=_number(data["w"], "crop.w"),
            h=_number(data["h"], "crop.h"),
        )

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
        return cls(x=x, y=y, w=w, h=h)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` in pixels of a ``width`` x ``height`` frame.

        Rounded outward-safe: never empty, never past the frame's edge.
        """
        x0 = min(round(self.x * width), width - 1)
        y0 = min(round(self.y * height), height - 1)
        x1 = max(x0 + 1, min(round((self.x + self.w) * width), width))
        y1 = max(y0 + 1, min(round((self.y + self.h) * height), height))
        return x0, y0, x1, y1


_CONFIG_KEYS = frozenset({"crop", "device", "resolution", "streaming"})


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """What the control UI chose for one board's camera.

    * ``crop`` — the part of the frame the edge sends; ``None`` sends the whole frame.
    * ``device`` — the lens, by the **label** the edge reported. Labels are the one
      identity a browser keeps across reloads (device ids are regenerated per
      session); ``None`` leaves the edge on its default rear camera.
    * ``resolution`` — the capture resolution asked for.
    * ``streaming`` — whether the edge streams frames to ``/ingest``.
    """

    crop: CropRect | None = None
    device: str | None = None
    resolution: Resolution = DEFAULT_RESOLUTION
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.resolution not in RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {', '.join(RESOLUTIONS)}, not {self.resolution!r}"
            )
        if self.device is not None and (not isinstance(self.device, str) or not self.device):
            raise ValueError("device must be a non-empty label, or null")
        if not isinstance(self.streaming, bool):
            raise ValueError("streaming must be true or false")

    @classmethod
    def from_dict(cls, raw: object) -> CameraConfig:
        """A complete config; omitted fields take their defaults."""
        data = _record(raw, "config", _CONFIG_KEYS)
        crop = data.get("crop")
        resolution = data.get("resolution", DEFAULT_RESOLUTION)
        if resolution not in get_args(Resolution):
            raise ValueError(f"resolution must be one of {', '.join(RESOLUTIONS)}")
        device = data.get("device")
        if device is not None and not isinstance(device, str):
            raise ValueError("device must be a string label, or null")
        return cls(
            crop=None if crop is None else CropRect.from_dict(crop),
            device=device,
            resolution=cast(Resolution, resolution),
            streaming=cast(bool, data.get("streaming", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "crop": None if self.crop is None else self.crop.to_dict(),
            "device": self.device,
            "resolution": self.resolution,
            "streaming": self.streaming,
        }


@dataclass(frozen=True, slots=True)
class CameraDevice:
    """One lens the edge can open."""

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class FrameSize:
    width: int
    height: int

    @property
    def portrait(self) -> bool:
        """Held upright: the framing measurement lost a quarter of perfect boards."""
        return self.height > self.width


_REPORT_KEYS = frozenset({"devices", "device", "frame", "ingest"})
_MAX_DEVICES = 32


def _text(raw: object, name: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string")
    return raw


@dataclass(frozen=True, slots=True)
class CameraReport:
    """What the edge says about its camera.

    ``device`` is the label of the lens actually open and ``frame`` the size it
    actually delivers — both ``None`` until the camera is live.
    """

    devices: tuple[CameraDevice, ...]
    device: str | None
    frame: FrameSize | None
    ingest: IngestState

    @classmethod
    def from_dict(cls, raw: object) -> CameraReport:
        data = _record(raw, "report", _REPORT_KEYS)
        devices_raw = data.get("devices", [])
        if not isinstance(devices_raw, list) or len(devices_raw) > _MAX_DEVICES:
            raise ValueError(f"devices must be a list of at most {_MAX_DEVICES}")
        devices = []
        for entry in devices_raw:
            item = _record(entry, "device", frozenset({"id", "label"}))
            devices.append(
                CameraDevice(
                    id=_text(item.get("id"), "device.id"),
                    label=_text(item.get("label"), "device.label"),
                )
            )
        device = data.get("device")
        frame_raw = data.get("frame")
        frame = None
        if frame_raw is not None:
            size = _record(frame_raw, "frame", frozenset({"width", "height"}))
            width, height = size.get("width"), size.get("height")
            if (
                isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, int) or not isinstance(height, int)
                or width <= 0 or height <= 0
            ):
                raise ValueError("frame must be {width, height} in positive whole pixels")
            frame = FrameSize(width=width, height=height)
        ingest = data.get("ingest", "idle")
        if ingest not in get_args(IngestState):
            raise ValueError(f"ingest must be one of {', '.join(get_args(IngestState))}")
        return cls(
            devices=tuple(devices),
            device=None if device is None else _text(device, "device"),
            frame=frame,
            ingest=cast(IngestState, ingest),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "devices": [{"id": d.id, "label": d.label} for d in self.devices],
            "device": self.device,
            "frame": (
                None
                if self.frame is None
                else {"width": self.frame.width, "height": self.frame.height}
            ),
            "ingest": self.ingest,
        }


# ---- messages on the two control sockets -------------------------------------


def _message(text: str, kinds: dict[str, frozenset[str]]) -> tuple[str, dict[str, object]]:
    """``text`` as a JSON object of one of ``kinds`` (type -> its other keys)."""
    try:
        raw = json.loads(text)
    except ValueError as e:
        raise ValueError(f"message is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("message must be an object")
    kind = raw.get("type")
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError(f"message type must be one of {', '.join(kinds)}, not {kind!r}")
    data = _record(raw, f"{kind} message", kinds[kind] | {"type"})
    missing = kinds[kind] - set(data)
    if missing:
        raise ValueError(f"{kind} message: missing field(s) {', '.join(sorted(missing))}")
    return kind, data


def _jpeg_text(jpeg: bytes) -> str:
    # A page can show base64 as a data: URL as it stands; binary would need a
    # second message to say which picture it is.
    return base64.b64encode(jpeg).decode("ascii")


@dataclass(frozen=True, slots=True)
class ReportMessage:
    """Phone -> node: what the camera is now."""

    TYPE: ClassVar[str] = "report"
    report: CameraReport


def parse_phone_message(text: str) -> ReportMessage:
    """One text message from the phone's socket; ``ValueError`` says why not."""
    _, data = _message(text, {ReportMessage.TYPE: frozenset({"report"})})
    return ReportMessage(report=CameraReport.from_dict(data["report"]))


@dataclass(frozen=True, slots=True)
class ConfigureMessage:
    """Control page -> node: replace the board's config with this whole one."""

    TYPE: ClassVar[str] = "configure"
    config: CameraConfig


def parse_control_message(text: str) -> ConfigureMessage:
    """One text message from a control page's socket; ``ValueError`` says why not."""
    _, data = _message(text, {ConfigureMessage.TYPE: frozenset({"config"})})
    return ConfigureMessage(config=CameraConfig.from_dict(data["config"]))


@dataclass(frozen=True, slots=True)
class ConfigMessage:
    """Node -> phone: the config to apply."""

    TYPE: ClassVar[str] = "config"
    config: CameraConfig

    def to_json(self) -> str:
        return json.dumps({"type": self.TYPE, "config": self.config.to_dict()})


@dataclass(frozen=True, slots=True)
class PreviewsMessage:
    """Node -> phone: send whole-frame previews (``on``) or stop.

    On exactly while at least one control page watches the board, so an
    unwatched phone spends no uplink on pictures nobody sees.
    """

    TYPE: ClassVar[str] = "previews"
    on: bool

    def to_json(self) -> str:
        return json.dumps({"type": self.TYPE, "on": self.on})


@dataclass(frozen=True, slots=True)
class RefusedMessage:
    """Node -> either client: one message was refused; the socket stays open."""

    TYPE: ClassVar[str] = "refused"
    reason: str

    def to_json(self) -> str:
        return json.dumps({"type": self.TYPE, "reason": self.reason})


@dataclass(frozen=True, slots=True)
class StateMessage:
    """Node -> control page: the board's camera as the node holds it.

    ``report`` is the phone's last report (kept after it disconnects, so the lens
    list survives a reload); ``phone_connected`` says whether its socket is open.
    """

    TYPE: ClassVar[str] = "state"
    config: CameraConfig
    report: CameraReport | None
    phone_connected: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.TYPE,
                "config": self.config.to_dict(),
                "report": None if self.report is None else self.report.to_dict(),
                "phone_connected": self.phone_connected,
            }
        )


@dataclass(frozen=True, slots=True)
class PreviewMessage:
    """Node -> control page: the phone's latest whole-frame preview."""

    TYPE: ClassVar[str] = "preview"
    jpeg: bytes
    width: int
    height: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.TYPE,
                "width": self.width,
                "height": self.height,
                "jpeg": _jpeg_text(self.jpeg),
            }
        )


@dataclass(frozen=True, slots=True)
class FrameMessage:
    """Node -> control page: the latest frame ``/ingest`` received for the board."""

    TYPE: ClassVar[str] = "frame"
    seq: int
    jpeg: bytes
    width: int
    height: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.TYPE,
                "seq": self.seq,
                "width": self.width,
                "height": self.height,
                "jpeg": _jpeg_text(self.jpeg),
            }
        )


@dataclass(frozen=True, slots=True)
class HealthMessage:
    """Node -> control page, about once a second: the recognizer and the stream.

    Ages and rates are measured on the node's clock, so the page needs none: the
    phone, the laptop and the node need not agree on the time.
    """

    TYPE: ClassVar[str] = "health"
    recognizer: str
    #: Why the stub serves instead of the model, when it does (DESIGN D2).
    fallback: str | None
    #: Node-wide ``/ingest`` counters, as ``/healthz`` has them.
    received: int
    refused: int
    #: Seconds since the board's last frame arrived; ``None`` before the first.
    frame_age_s: float | None
    #: Frames per second received since the previous health message, if measurable.
    fps: float | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.TYPE,
                "recognizer": self.recognizer,
                "fallback": self.fallback,
                "received": self.received,
                "refused": self.refused,
                "frame_age_s": self.frame_age_s,
                "fps": self.fps,
            }
        )


#: What the node sends the phone.
PhoneDown = ConfigMessage | PreviewsMessage | RefusedMessage
#: What the node sends a control page.
ControlDown = StateMessage | PreviewMessage | FrameMessage | HealthMessage | RefusedMessage
