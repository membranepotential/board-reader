#!/usr/bin/env python3
"""Camera-elevation QC for a broadcast, against the operating envelope.

Detects the board in a handful of frames, measures the projected board aspect
(short axis / long axis of the corner quad) and turns the **median** into a
verdict. Exits non-zero when the stream should not be used, so it works as a
gate in front of frame extraction:

* ``>= 0.73``                — good band, the camera looks down on the board;
* ``>= LOW_ANGLE_ASPECT``    — degraded but usable;
* below that                 — too shallow, reject the stream;
* nothing detected at all    — ``unknown``, which is **not** a pass.

Usage::

    uv run python tools/video/check_angle.py (VIDEO | --frames DIR) [-n N] [--json]

Many frames legitimately yield no detection — clutter, camera cuts, replays,
graphics — so a partial detection rate is expected rather than an error.

Sampling a *video* needs that video's crop from ``crops.csv``, which is why the
manual crop comes first and this check second: board detection is much more
reliable on the cropped board than on a cluttered full broadcast frame
(measured 2026-08-07 — full-frame detection failed on 3 of 6 videos and latched
onto a 2D *overlay* board on a 4th, while the cropped frames detected fine).

**Interpreter: ``uv run python``, not the system one.** The board detector
(:mod:`boarddetect`, next to this file) needs the workspace venv's numpy and
headless cv2; this tool never opens a window, so headless cv2 is exactly right.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from boarddetect import LOW_ANGLE_ASPECT, detect_board_quad, quad_aspect
from extract_frames import even_times, format_timestamp, probe_duration
from streams import CropBox, StreamsError, stream_for_video

# Above this the camera has enough elevation for per-square crops to contain
# their own piece; see docs/operating-envelope.md and LOW_ANGLE_ASPECT.
GOOD_ASPECT = 0.73

# Fraction of the video skipped at each end, so intros and end screens (which
# never show a board) do not eat the sample.
SAMPLE_MARGIN = 0.02

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class Verdict(Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    TOO_SHALLOW = "too_shallow"
    UNKNOWN = "unknown"

    @property
    def exit_code(self) -> int:
        """0 only for verdicts that clear the gate."""
        return {
            Verdict.GOOD: 0,
            Verdict.DEGRADED: 0,
            Verdict.TOO_SHALLOW: 1,
            Verdict.UNKNOWN: 2,
        }[self]

    @property
    def summary(self) -> str:
        return {
            Verdict.GOOD: "good — camera elevation is inside the operating envelope",
            Verdict.DEGRADED: "degraded — usable, but the camera is low; expect errors",
            Verdict.TOO_SHALLOW: "TOO SHALLOW — reject this stream, it is near board level",
            Verdict.UNKNOWN: "UNKNOWN — no board detected in any frame; cannot clear the gate",
        }[self]


@dataclass(frozen=True, slots=True)
class FrameAspect:
    """One sampled frame: its label, and its board aspect if a board was found."""

    label: str
    aspect: float | None


@dataclass(frozen=True, slots=True)
class AngleReport:
    """The aggregate verdict over a sample of frames."""

    source: str
    frames: tuple[FrameAspect, ...]

    @classmethod
    def from_quads(
        cls, source: str, quads: Iterable[tuple[str, np.ndarray | None]]
    ) -> AngleReport:
        return cls(
            source=source,
            frames=tuple(
                FrameAspect(label, None if quad is None else quad_aspect(quad))
                for label, quad in quads
            ),
        )

    @property
    def aspects(self) -> tuple[float, ...]:
        return tuple(f.aspect for f in self.frames if f.aspect is not None)

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return len(self.aspects) / len(self.frames)

    @property
    def median_aspect(self) -> float | None:
        return median(self.aspects) if self.aspects else None

    @property
    def verdict(self) -> Verdict:
        med = self.median_aspect
        if med is None:
            return Verdict.UNKNOWN
        if med >= GOOD_ASPECT:
            return Verdict.GOOD
        if med >= LOW_ANGLE_ASPECT:
            return Verdict.DEGRADED
        return Verdict.TOO_SHALLOW

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "frames": [{"label": f.label, "aspect": f.aspect} for f in self.frames],
            "sampled": len(self.frames),
            "detected": len(self.aspects),
            "detection_rate": self.detection_rate,
            "median_aspect": self.median_aspect,
            "verdict": self.verdict.value,
            "good_aspect": GOOD_ASPECT,
            "low_angle_aspect": LOW_ANGLE_ASPECT,
        }


def _grab_cropped_frame(
    video_path: Path, seconds: float, crop: CropBox
) -> np.ndarray | None:
    """Decode one cropped frame straight into memory via an ffmpeg pipe."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-filter:v",
        crop.ffmpeg_filter,
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        raise SystemExit("Error: ffmpeg not found. Please install ffmpeg first.") from None
    if out.returncode != 0 or not out.stdout:
        return None
    return cv2.imdecode(np.frombuffer(out.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)


def quads_from_video(
    video_path: Path, count: int
) -> Iterator[tuple[str, np.ndarray | None]]:
    """Sample ``count`` frames evenly across the video and detect a board in each."""
    try:
        stream = stream_for_video(video_path)
    except StreamsError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    duration = probe_duration(video_path)
    for t in even_times(duration, count, SAMPLE_MARGIN):
        image = _grab_cropped_frame(video_path, t, stream.crop)
        label = f"t={format_timestamp(t)}"
        yield label, None if image is None else detect_board_quad(image)


def quads_from_frames(frames_dir: Path) -> Iterator[tuple[str, np.ndarray | None]]:
    """Detect a board in every already-cropped image in a directory.

    All of them: a frame set is small and hand-curated, so subsampling it would
    only make the verdict noisier.
    """
    paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in FRAME_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"Error: no images in {frames_dir}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        yield path.name, None if image is None else detect_board_quad(image)


def render(report: AngleReport) -> str:
    lines = [f"Source: {report.source}", ""]
    for frame in report.frames:
        shown = "no board detected" if frame.aspect is None else f"{frame.aspect:.3f}"
        lines.append(f"  {frame.label:<28} {shown}")
    detected, sampled = len(report.aspects), len(report.frames)
    med = report.median_aspect
    shown_median = "n/a" if med is None else f"{med:.3f}"
    bands = f"(good >= {GOOD_ASPECT}, reject < {LOW_ANGLE_ASPECT})"
    lines += [
        "",
        f"Detected:       {detected}/{sampled} ({report.detection_rate:.0%})",
        f"Median aspect:  {shown_median}   {bands}",
        f"Verdict:        {report.verdict.summary}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a broadcast's camera elevation against the operating envelope.",
    )
    # Not a mutually-exclusive group: an ``nargs="?"`` positional always counts
    # as "seen", so argparse would neither enforce the group nor allow --frames.
    parser.add_argument("video", nargs="?", help="Downloaded video file (yt-dlp naming)")
    parser.add_argument(
        "--frames",
        type=Path,
        help="Directory of already-cropped frames to check instead of a video",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=12,
        help="Frames sampled from a video (default: 12; ignored with --frames)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON on stdout"
    )
    args = parser.parse_args(argv)

    if (args.video is None) == (args.frames is None):
        parser.error("give exactly one of VIDEO or --frames DIR")

    if args.frames is not None:
        if not args.frames.is_dir():
            raise SystemExit(f"Error: not a directory: {args.frames}")
        report = AngleReport.from_quads(
            str(args.frames), quads_from_frames(args.frames)
        )
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Error: input file not found: {video_path}")
        report = AngleReport.from_quads(
            video_path.name, quads_from_video(video_path, args.count)
        )

    print(json.dumps(report.as_dict(), indent=2) if args.json else render(report))
    return report.verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
