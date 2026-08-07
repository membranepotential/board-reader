#!/usr/bin/env python3
"""Play a registered broadcast in mpv with its board crop applied live.

Nothing is re-encoded: mpv gets the crop as a video filter, so pressing ``s``
saves a screenshot that is already cropped exactly like ``extract_frames.py``
output. Use it to hand-pick interesting positions instead of sampling at random.

Usage::

    python3 tools/video/watch_cropped.py VIDEO [--outdir D] [--start TS] \\
        [--format png|jpg|webp]

Unrecognised arguments are passed straight through to mpv.

**Interpreter: system ``python3``, not ``uv run``.** mpv is a system program and
this tool is the sibling of ``crop_selector.py`` in the hand-operated half of
the toolset; like it, this module must never import ``board_reader_core`` or
anything else from the workspace venv.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from streams import StreamsError, stream_for_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a video cropped in mpv; press 's' to save a cropped frame.",
        epilog="Unrecognised arguments are passed straight through to mpv.",
    )
    parser.add_argument("video", help="Downloaded video file (yt-dlp naming)")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        help="Screenshot directory (default: data/streams/frames/<slug>_<video_id>)",
    )
    parser.add_argument(
        "-s", "--start", help='Start position (e.g. "01:42:46"), passed to mpv --start'
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "jpg", "webp"],
        help="Screenshot format (default: png)",
    )
    args, mpv_args = parser.parse_known_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Error: input file not found: {video_path}")

    try:
        stream = stream_for_video(video_path)
    except StreamsError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    outdir = args.outdir if args.outdir is not None else stream.frames_dir
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mpv",
        f"--vf={stream.crop.ffmpeg_filter}",
        f"--screenshot-format={args.format}",
        f"--screenshot-dir={outdir}",
        # Milliseconds keep frame-stepped shots within one second distinct.
        f"--screenshot-template={stream.slug}_t%wHh%wMm%wSs%wTms",
        "--screenshot-png-compression=7",
        "--hr-seek=yes",  # exact seeking, so a grabbed frame is the one you saw
        "--keep-open=yes",
    ]
    if args.start:
        cmd.append(f"--start={args.start}")
    cmd.extend(mpv_args)
    cmd.append(str(video_path))

    print(f"{stream.slug} ({stream.video_id}) — crop {stream.crop.ffmpeg_filter}")
    print(f"Screenshots -> {outdir}/{stream.slug}_tHHhMMmSSsMMMms.{args.format}\n")
    print("Keys:  s = save frame   space = pause   . / , = frame step")
    print("       <- -> = seek 5s  up/down = seek 60s  q = quit\n")

    try:
        sys.exit(subprocess.run(cmd, check=False).returncode)
    except FileNotFoundError:
        raise SystemExit("Error: mpv not found. Please install mpv first.") from None


if __name__ == "__main__":
    sys.exit(main())
