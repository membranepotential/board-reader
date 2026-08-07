"""L1 -> L2 frame envelope — the shared edge/inference wire format (DESIGN.md §4.5).

The edge sends, per frame, a small JSON *header* plus a binary JPEG of the
(eventually rectified) board crop. This module defines that envelope and the
in-memory :class:`Frame` that both sides speak. Like :mod:`board_reader_core.protocol`
it is shared, so it must not contain node-specific behaviour — only the wire shape
and JPEG (de)serialisation.

A frame may carry the board's playing-surface ``corners``, in which case the JPEG
is the ORIGINAL camera image rather than a rectified crop. Corners ride on **every**
frame by design: there is no ``set_corners`` control message and no per-connection
corner state on the inference node, so a reconnecting edge, a restarted gateway and
a mid-stream recalibration all need zero handshake to agree. :data:`CONTROL` stays
reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FRAME = "frame"  # header "type" discriminator on the ingress socket
CONTROL = "control"  # reserved for hello/set_corners/calibrate (later milestones)


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

    ``corners`` set means ``image`` is the original photo and the board is the quad;
    unset means ``image`` is already the rectified board crop.
    """

    board_id: str
    seq: int
    capture_ts: float  # epoch seconds, wall clock, stamped at capture on the edge
    image: np.ndarray  # BGR
    stability: float = 1.0
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
        """JSON-safe header (everything except the image bytes)."""
        header = {
            "type": FRAME,
            "board_id": self.board_id,
            "seq": self.seq,
            "capture_ts": self.capture_ts,
            "stability": self.stability,
            "clock": self.clock,
            "burst_id": self.burst_id,
        }
        if self.corners is not None:
            header["corners"] = self.corners.reshape(-1).tolist()
        return header

    @classmethod
    def from_header(cls, header: dict, image: np.ndarray) -> "Frame":
        return cls(
            board_id=header["board_id"],
            seq=int(header["seq"]),
            capture_ts=float(header["capture_ts"]),
            image=image,
            stability=float(header.get("stability", 1.0)),
            clock=header.get("clock", "none"),
            burst_id=header.get("burst_id"),
            corners=header.get("corners"),
        )

    def encode(self, quality: int = 90) -> tuple[dict, bytes]:
        """Split into (header dict, JPEG bytes) ready for a two-part WS send."""
        return self.header(), encode_jpeg(self.image, quality)

    @classmethod
    def decode(cls, header: dict, jpeg: bytes) -> "Frame":
        return cls.from_header(header, decode_jpeg(jpeg))
