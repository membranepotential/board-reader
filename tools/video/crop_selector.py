#!/usr/bin/env python3
"""Record the board crop for a downloaded broadcast, by hand.

This is the **first** step for every new stream and the only one that needs a
human: drag a box around the playing surface on one preview frame, and the
region is appended to ``crops.csv``. Nothing is re-encoded — every other tool
crops on the fly from that row.

The crop comes before the automatic angle check on purpose: board detection is
far more reliable on the cropped board than on a cluttered full broadcast frame
(measured 2026-08-07 — full-frame detection failed on 3 of 6 videos and latched
onto a 2D *overlay* board on a 4th, while the cropped frames detected fine).

Usage::

    python3 tools/video/crop_selector.py [--at TS] [--table PATH] VIDEO

Keys: ``c`` confirm · ``r`` reset · ``q`` cancel.

**Interpreter: system ``python3``, not ``uv run``.** This needs a GUI-enabled
cv2 for ``imshow``; the workspace venv has ``opencv-python-headless``, where
``imshow`` does nothing. Arch's ``python-opencv`` (pacman) is Qt-enabled, so run
this on the system interpreter — which in turn means this module must never
import ``board_reader_core`` or anything else from the workspace.

The recorded crop is in pixels of *this* download's resolution; grabbing the
same video at another resolution means re-running this tool. New rows are
written as ``status=keep`` with no median aspect — run ``check_angle.py`` next
and edit the row to ``rejected_low_angle`` if the camera sits too low.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from streams import (
    DEFAULT_TABLE,
    CropBox,
    Stream,
    StreamStatus,
    StreamsError,
    append_stream,
    slug_from_title,
    title_from_path,
    video_id_from_path,
)

WINDOW = "Select Crop Area"


@dataclass(slots=True)
class RectSelector:
    """Mouse-drag rectangle state for one selection window.

    Owns what used to be four module-level globals; cv2 hands it back through
    the callback rather than the callback reaching out for it.
    """

    frame: np.ndarray
    origin: tuple[int, int] | None = None
    corner: tuple[int, int] | None = None
    dragging: bool = False

    @property
    def box(self) -> CropBox | None:
        """The selection as a crop box, with even extents, or ``None``.

        Even width/height keep the crop compatible with chroma-subsampled
        encoders and with ffmpeg's ``crop`` filter on yuv420p input.
        """
        if self.origin is None or self.corner is None:
            return None
        width = abs(self.corner[0] - self.origin[0])
        height = abs(self.corner[1] - self.origin[1])
        width -= width % 2
        height -= height % 2
        if width == 0 or height == 0:
            return None
        return CropBox(
            x=min(self.origin[0], self.corner[0]),
            y=min(self.origin[1], self.corner[1]),
            width=width,
            height=height,
        )

    # cv2 calls this positionally with a fixed 5-argument signature; the last
    # two carry nothing we need now that the state lives on the instance.
    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.origin = self.corner = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.corner = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.corner = (x, y)
        else:
            return
        self.show()

    def show(self) -> None:
        canvas = self.frame.copy()
        if self.origin is not None and self.corner is not None:
            cv2.rectangle(canvas, self.origin, self.corner, (0, 255, 0), 2)
        cv2.imshow(WINDOW, canvas)

    def reset(self) -> None:
        self.origin = self.corner = None
        self.show()


def parse_time(value: str) -> float:
    """Parse ``HH:MM:SS``, ``MM:SS`` or a plain seconds value into seconds."""
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def grab_frame(video_path: Path, seconds: float) -> np.ndarray:
    """Decode one frame at the given offset."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open video file: {video_path}")
    if seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Error: could not read a frame at {seconds:.1f}s")
    return frame


def select_crop(frame: np.ndarray) -> CropBox:
    """Show the frame and block until the operator confirms a rectangle."""
    selector = RectSelector(frame)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, selector.on_mouse)

    print("Instructions:")
    print("- Click and drag to select the crop area")
    print("- Press 'c' to confirm selection")
    print("- Press 'r' to reset selection")
    print("- Press 'q' to quit without recording")
    selector.show()

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                box = selector.box
                if box is None:
                    print("Please select an area first!")
                    continue
                return box
            if key == ord("r"):
                selector.reset()
            elif key == ord("q"):
                raise SystemExit("Cancelled by user")
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the board crop region for a downloaded broadcast.",
    )
    parser.add_argument("video", help="Downloaded video file (yt-dlp naming)")
    parser.add_argument(
        "--at",
        default="0",
        help='Timestamp of the frame shown for selection (default: "0", the first frame)',
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help=f"CSV table the crop is appended to (default: {DEFAULT_TABLE})",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Error: input file not found: {video_path}")

    try:
        video_id = video_id_from_path(video_path)
    except StreamsError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    preview_at = parse_time(args.at)
    print(f"Loading frame at {preview_at:.1f}s from: {video_path}")
    frame = grab_frame(video_path, preview_at)
    box = select_crop(frame)

    title = title_from_path(video_path)
    height, width = frame.shape[:2]
    stream = Stream(
        video_id=video_id,
        slug=slug_from_title(title),
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        crop=box,
        frame_width=width,
        frame_height=height,
        preview_at_s=preview_at,
        median_aspect=None,
        status=StreamStatus.KEEP,
        recorded_at=datetime.now().replace(microsecond=0),
    )

    append_stream(stream, args.table)

    print(f"Selected crop: {box.ffmpeg_filter}  (frame {width}x{height})")
    print(f"Recorded {stream.video_id} as slug {stream.slug!r}")
    print("Next: uv run python tools/video/check_angle.py <video>")


if __name__ == "__main__":
    sys.exit(main())
