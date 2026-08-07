#!/usr/bin/env python3
"""Extract cropped still frames from a registered broadcast.

Timestamps are either drawn at random across the video (``-n``) or given
explicitly (``--times``). The crop comes from ``crops.csv`` (see
``crop_selector.py``), so nothing is ever re-encoded — ffmpeg seeks, crops and
writes one PNG per timestamp.

Usage::

    uv run python tools/video/extract_frames.py VIDEO (-n N | --times T [T...]) \\
        [--outdir D] [--prefix P] [--seed S] [--margin F]

**Interpreter: either one.** This module is pure stdlib plus ``ffmpeg``/
``ffprobe`` subprocesses, so it runs under ``uv run python`` *and* under the
system ``python3`` — keep it that way, ``watch_cropped.py`` (system-only) reuses
its helpers.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

from streams import CropBox, Stream, StreamsError, stream_for_video


def probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("Error: ffprobe not found. Please install ffmpeg first.") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Error: ffprobe failed: {exc.stderr.strip()}") from exc
    return float(out.stdout.strip())


def parse_time(value: str) -> float:
    """Parse ``HH:MM:SS``, ``MM:SS`` or a plain seconds value into seconds."""
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HHhMMmSSs`` for use in filenames."""
    total = int(seconds)
    return f"{total // 3600:02d}h{(total % 3600) // 60:02d}m{total % 60:02d}s"


def random_times(
    duration: float, count: int, margin: float, seed: int | None
) -> list[float]:
    """Draw ``count`` sorted timestamps, skipping ``margin`` of each end.

    The margin keeps intros, sponsor cards and end screens out of the sample.
    ``seed`` makes a draw reproducible, which is how a frame set is re-created.
    """
    if not 0.0 <= margin < 0.5:
        raise SystemExit(f"Error: --margin must be in [0, 0.5), got {margin}")
    rng = random.Random(seed)
    low = duration * margin
    high = duration * (1 - margin)
    return sorted(rng.uniform(low, high) for _ in range(count))


def even_times(duration: float, count: int, margin: float) -> list[float]:
    """``count`` timestamps evenly spread over the video, minus the margins."""
    if count < 1:
        raise SystemExit(f"Error: need at least one sample, got {count}")
    low = duration * margin
    high = duration * (1 - margin)
    if count == 1:
        return [0.5 * (low + high)]
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def extract_frame(
    video_path: Path, seconds: float, crop: CropBox, output_path: Path
) -> None:
    """Write one cropped frame at the given timestamp."""
    cmd = [
        "ffmpeg",
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-filter:v",
        crop.ffmpeg_filter,
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("Error: ffmpeg not found. Please install ffmpeg first.") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Error: ffmpeg failed for t={seconds:.1f}s: {exc.stderr.strip()}"
        ) from exc


def warn_if_rejected(stream: Stream) -> None:
    """Shout when frames are being pulled from a stream we already ruled out."""
    if not stream.is_rejected:
        return
    aspect = "unknown" if stream.median_aspect is None else f"{stream.median_aspect:.3f}"
    print(
        f"\n!! {stream.video_id} ({stream.slug}) is marked"
        f" {stream.status.value}: median board aspect {aspect}.\n"
        "!! The camera sits too close to board level — pieces occlude whole\n"
        "!! ranks and recognition collapses. These frames are NOT usable as\n"
        "!! training or eval data. Extracting anyway.\n",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cropped still frames from a registered broadcast.",
    )
    parser.add_argument("video", help="Downloaded video file (yt-dlp naming)")
    when = parser.add_mutually_exclusive_group(required=True)
    when.add_argument(
        "-n", "--count", type=int, help="Number of frames drawn at random times"
    )
    when.add_argument(
        "--times",
        nargs="+",
        help='Explicit timestamps instead of random ones (e.g. "01:42:46" or 6166)',
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        help="Output directory (default: data/streams/frames/<slug>_<video_id>)",
    )
    parser.add_argument(
        "--prefix", help="Output filename prefix (default: the stream's slug)"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible times")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Fraction of the video skipped at each end (default: 0.02)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Error: input file not found: {video_path}")

    try:
        stream = stream_for_video(video_path)
    except StreamsError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    warn_if_rejected(stream)

    duration = probe_duration(video_path)
    if args.times:
        times = sorted(parse_time(t) for t in args.times)
        if times[-1] >= duration:
            raise SystemExit(
                f"Error: timestamp {times[-1]:.1f}s is past the end of the video"
                f" ({duration:.1f}s)"
            )
    else:
        times = random_times(duration, args.count, args.margin, args.seed)

    outdir = args.outdir if args.outdir is not None else stream.frames_dir
    prefix = args.prefix if args.prefix is not None else stream.slug
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{stream.slug} ({stream.video_id}) — duration {duration:.1f}s")
    print(f"Crop: {stream.crop.ffmpeg_filter}\n")

    for i, t in enumerate(times, start=1):
        output_path = outdir / f"{prefix}_{i}_t{format_timestamp(t)}.png"
        extract_frame(video_path, t, stream.crop, output_path)
        print(f"[{i}/{len(times)}] t={format_timestamp(t)} -> {output_path}")


if __name__ == "__main__":
    sys.exit(main())
