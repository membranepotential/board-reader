"""The camera control channel — how the inference node configures an edge camera.

DESIGN D10 "dumb camera" (2026-11-24): the phone only films and streams. What it
films — which lens, at what resolution, which part of the frame is sent, and whether
it streams at all — is set from the control UI on the laptop, held per board on the
inference node, and **pushed to the edge, which applies it**. The edge still crops
and measures motion on its own crop, so the L1->L2 wire (:mod:`board_reader_core.wire`)
is unchanged: frames arrive already cropped, exactly as before.

Three shapes cross this channel, each validated here once so neither node trusts
the other's JSON:

* :class:`CameraConfig` — node -> edge: what the control UI chose.
* :class:`CameraReport` — edge -> node: what the camera actually is (its lenses,
  the frame it delivers, the state of its stream), so the control UI can offer real
  choices and say when the phone is held upright.
* :class:`CropRect` — the rectangle itself, in **fractions of the full camera
  frame**, so it survives a resolution change unchanged. Also the Python edge's
  static ``--roi``.

Like the rest of core this is contract only: no transport, no storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

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
    """What the edge says about its camera, on every poll.

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
