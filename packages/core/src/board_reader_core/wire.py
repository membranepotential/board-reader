"""L1 -> L2 frame envelope — the shared edge/inference wire format (DESIGN.md §4.5).

The edge sends, per frame, a small JSON *header* plus a binary JPEG of the
(eventually rectified) board crop. This module defines that envelope and the
in-memory :class:`Frame` that both sides speak. Like :mod:`board_reader_core.protocol`
it is shared, so it must not contain node-specific behaviour — only the wire shape
and JPEG (de)serialisation.
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
    """

    board_id: str
    seq: int
    capture_ts: float  # epoch seconds, wall clock, stamped at capture on the edge
    image: np.ndarray  # BGR
    stability: float = 1.0
    clock: str = "none"
    burst_id: str | None = None

    def header(self) -> dict:
        """JSON-safe header (everything except the image bytes)."""
        return {
            "type": FRAME,
            "board_id": self.board_id,
            "seq": self.seq,
            "capture_ts": self.capture_ts,
            "stability": self.stability,
            "clock": self.clock,
            "burst_id": self.burst_id,
        }

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
        )

    def encode(self, quality: int = 90) -> tuple[dict, bytes]:
        """Split into (header dict, JPEG bytes) ready for a two-part WS send."""
        return self.header(), encode_jpeg(self.image, quality)

    @classmethod
    def decode(cls, header: dict, jpeg: bytes) -> "Frame":
        return cls.from_header(header, decode_jpeg(jpeg))
