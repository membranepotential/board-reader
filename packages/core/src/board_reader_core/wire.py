"""L1 -> L2 frame envelope — the shared edge/inference wire format (DESIGN.md §4.5).

The edge sends, per frame, a small JSON *header* plus a binary JPEG of what its
camera sees (cropped to the board's rectangle, DESIGN D11 "dumb camera"). This
module defines that envelope and the in-memory :class:`Frame` that both sides
speak. Like :mod:`board_reader_core.protocol` it is shared, so it must not contain
node-specific behaviour — only the wire shape and JPEG (de)serialisation.

**``corners`` is a legacy field.** Older edges (the edge rectifier's
``--forward-original`` path and the phone tracker) stamped the board's
playing-surface quad on every frame, the JPEG then being the original camera
image. No producer sets it any more, but it still parses, because old recordings
carry it and replay through this class unchanged. :data:`CONTROL` stays reserved.

**The header is a wire model** (:class:`FrameHeader`, :mod:`.schema`), so its rules
hold in the browser edge too. The JPEG is not in it: on ``WS /ingest`` every header
is one TEXT message and its JPEG the BINARY message right after it, and the node
reads the two in lockstep. Nothing may be sent between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import cv2
import numpy as np
from pydantic import Field, StrictFloat, StrictStr
from pydantic.json_schema import SkipJsonSchema

from board_reader_core.protocol import Running
from board_reader_core.wiremodel import (
    BoardId,
    Natural,
    Unit,
    Wire,
    absent,
    tag,
    wire,
)

FRAME = "frame"  # header "type" discriminator on the ingress socket
CONTROL = "control"  # reserved for hello/control messages (later milestones)


@wire
class FrameHeader(Wire):
    """The JSON header of one ingest frame; its JPEG follows as the next message.

    ``capture_ts`` is epoch seconds, stamped at capture on the edge (DESIGN D5).
    ``sharpness`` is the edge's focus score (tenengrad), unitless and comparable
    only between neighbouring frames of one board; ``None`` from an edge that
    does not measure it. ``corners`` is the legacy quad (see the module note):
    parsed for old recordings, never sent, and not in the JSON Schema.
    """

    type: Literal["frame"] = tag(FRAME)
    board_id: BoardId
    seq: Natural
    capture_ts: StrictFloat
    stability: Unit = 1.0
    sharpness: Annotated[StrictFloat, Field(ge=0)] | None = None
    clock: Running = "none"
    burst_id: StrictStr | None = None
    corners: SkipJsonSchema[
        Annotated[tuple[StrictFloat, ...], Field(min_length=8, max_length=8)] | None
    ] = Field(default=None, exclude_if=absent)


#: The keys a header may carry. A recording index line adds its own (where the JPEG
#: went, when it arrived, a replay's restamp marker), which are not the wire's.
HEADER_KEYS = frozenset(FrameHeader.__dataclass_fields__)


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR image to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()


def decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode JPEG bytes back to a BGR image."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode failed (not a valid JPEG?)")
    return image


@dataclass
class Frame:
    """A single captured frame crossing the L1->L2 boundary.

    Produced by the edge (camera/file), consumed by the inference estimator. Carries
    the edge capture timestamp and a per-board monotonic ``seq`` (DESIGN.md D5).

    ``corners`` (legacy, see the module docstring) set means ``image`` is the
    original photo and the board is the quad; no current edge sets it.
    """

    board_id: str
    seq: int
    capture_ts: float  # epoch seconds, wall clock, stamped at capture on the edge
    image: np.ndarray  # BGR
    stability: float = 1.0
    #: Edge-measured focus score (tenengrad on the tracked frame), unitless and
    #: resolution-dependent — only comparable between neighbouring frames of the
    #: same board, which is exactly how the ingest pre-filter uses it. ``None``
    #: from edges that do not measure it.
    sharpness: float | None = None
    clock: str = "none"
    burst_id: str | None = None
    corners: np.ndarray | None = None  # (4,2) float32 in ``image`` px: a8,h8,h1,a1

    def __post_init__(self) -> None:
        # Normalise here so every construction path — literal, from_header, replace —
        # yields the same (4,2) float32 quad, and a malformed one never gets further.
        if self.corners is not None:
            quad = np.asarray(self.corners, dtype=np.float32)
            if quad.size != 8:
                raise ValueError("corners must be 4 (x, y) pairs, in a8,h8,h1,a1 order")
            self.corners = quad.reshape(4, 2)

    def header(self) -> dict:
        """JSON-safe header (everything except the image bytes), validated."""
        corners = None if self.corners is None else tuple(self.corners.reshape(-1).tolist())
        return FrameHeader(
            board_id=self.board_id,
            seq=self.seq,
            capture_ts=self.capture_ts,
            stability=self.stability,
            sharpness=self.sharpness,
            clock=self.clock,  # type: ignore[arg-type]  # validated by the model
            burst_id=self.burst_id,
            corners=corners,
        ).to_dict()

    @classmethod
    def from_header(cls, header: dict, image: np.ndarray) -> Frame:
        """The frame a header describes.

        Still the lenient parse (``int("4")`` is a seq), not :class:`FrameHeader`:
        ``test_recording_ingest`` pins that coercion, and refusing it is an open
        decision (DESIGN.md §4.5).
        """
        return cls(
            board_id=header["board_id"],
            seq=int(header["seq"]),
            capture_ts=float(header["capture_ts"]),
            image=image,
            stability=float(header.get("stability", 1.0)),
            sharpness=(
                float(header["sharpness"])
                if header.get("sharpness") is not None
                else None
            ),
            clock=header.get("clock", "none"),
            burst_id=header.get("burst_id"),
            corners=header.get("corners"),
        )

    def encode(self, quality: int = 90) -> tuple[dict, bytes]:
        """Split into (header dict, JPEG bytes) ready for a two-part WS send."""
        return self.header(), encode_jpeg(self.image, quality)

    @classmethod
    def decode(cls, header: dict, jpeg: bytes) -> Frame:
        return cls.from_header(header, decode_jpeg(jpeg))
