#!/usr/bin/env python3
"""Typed registry over ``crops.csv`` — the board crop recorded for each stream.

A *stream* is one downloaded broadcast video. Its row records where the board
sits in the frame, so every other tool here (frame extraction, cropped
playback, angle QC) crops on the fly instead of re-encoding anything.

**Rows are keyed on the 11-char YouTube video id**, never on the mp4 basename:
the file gets renamed, re-downloaded at another resolution, or deleted, while
the id is stable. A video path is resolved to its row through the ``[<id>]``
suffix yt-dlp puts in the filename.

**The crop is in pixels of a download at that exact resolution.** Two of the
five rows were selected on 1080p downloads and three on 720p ones, so a crop is
only meaningful together with the ``frame_width``/``frame_height`` it was picked
at; :func:`load_streams` rejects any crop that does not fit inside them. Grab a
stream at a different resolution and you must re-run ``crop_selector.py``.

Interpreter: **stdlib only, no cv2, no workspace imports** — this module is
imported both by the system-``python3`` tools (``crop_selector.py``,
``watch_cropped.py``) and by the ``uv run`` ones (``check_angle.py``). Keep it
that way. See ``tools/video/README.md`` for why the split exists.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NewType

# ``tools/video/`` -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = Path(__file__).resolve().parent / "crops.csv"
FRAMES_ROOT = REPO_ROOT / "data" / "streams" / "frames"

COLUMNS = (
    "video_id",
    "slug",
    "title",
    "url",
    "x",
    "y",
    "width",
    "height",
    "frame_width",
    "frame_height",
    "preview_at_s",
    "median_aspect",
    "status",
    "recorded_at",
)

VideoId = NewType("VideoId", str)

# YouTube ids are exactly 11 chars of the URL-safe base64 alphabet. yt-dlp
# appends them in brackets: "Some Title [JTmovXvZnPM].mp4".
_VIDEO_ID = r"[0-9A-Za-z_-]{11}"
_VIDEO_ID_RE = re.compile(rf"^{_VIDEO_ID}$")
_BRACKETED_ID_RE = re.compile(rf"\[({_VIDEO_ID})\]")

# yt-dlp substitutes look-alike characters for the ones a filename cannot hold,
# so a title recovered from a basename has to be translated back.
_FILENAME_UNMANGLE = str.maketrans(
    {
        "｜": "|",
        "⧸": "/",
        "＼": "\\",
        "：": ":",
        "＊": "*",
        "？": "?",
        "＂": '"',
        "＜": "<",
        "＞": ">",
    }
)


class StreamsError(Exception):
    """Base class for every failure this module reports."""


class CropTableError(StreamsError):
    """The crop table is malformed, or a row's crop is not usable."""


class UnknownStreamError(StreamsError):
    """A video could not be resolved to a row in the crop table."""


class StreamStatus(Enum):
    """Whether a stream is usable as training/eval data.

    Rejection is a camera-angle verdict, not a download failure: below
    ``boarddetect.LOW_ANGLE_ASPECT`` the camera sits so close
    to board level that pieces occlude whole ranks. ``check_angle.py`` produces
    the verdict.
    """

    KEEP = "keep"
    REJECTED_LOW_ANGLE = "rejected_low_angle"


@dataclass(frozen=True, slots=True)
class CropBox:
    """A crop rectangle in source-frame pixels, top-left origin."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CropTableError(
                f"crop must have positive extent, got {self.width}x{self.height}"
            )
        if self.x < 0 or self.y < 0:
            raise CropTableError(
                f"crop origin must be non-negative, got ({self.x}, {self.y})"
            )

    @property
    def ffmpeg_filter(self) -> str:
        """The ``crop`` video filter for ffmpeg/mpv (``-filter:v`` / ``--vf``)."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    def fits_within(self, frame_width: int, frame_height: int) -> bool:
        return (
            self.x + self.width <= frame_width
            and self.y + self.height <= frame_height
        )


@dataclass(frozen=True, slots=True)
class Stream:
    """One registered broadcast: its identity, its board crop, its verdict."""

    video_id: VideoId
    slug: str
    title: str
    url: str
    crop: CropBox
    frame_width: int
    frame_height: int
    preview_at_s: float
    median_aspect: float | None
    status: StreamStatus
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.crop.fits_within(self.frame_width, self.frame_height):
            raise CropTableError(
                f"{self.video_id}: crop {self.crop.width}x{self.crop.height}"
                f"+{self.crop.x}+{self.crop.y} falls outside the"
                f" {self.frame_width}x{self.frame_height} frame it was recorded"
                " at — re-run crop_selector.py on this download"
            )

    @property
    def is_rejected(self) -> bool:
        return self.status is StreamStatus.REJECTED_LOW_ANGLE

    @property
    def frames_dir(self) -> Path:
        """Where this stream's extracted frames live: ``<slug>_<video_id>``."""
        return FRAMES_ROOT / f"{self.slug}_{self.video_id}"

    def to_row(self) -> dict[str, str]:
        return {
            "video_id": self.video_id,
            "slug": self.slug,
            "title": self.title,
            "url": self.url,
            "x": str(self.crop.x),
            "y": str(self.crop.y),
            "width": str(self.crop.width),
            "height": str(self.crop.height),
            "frame_width": str(self.frame_width),
            "frame_height": str(self.frame_height),
            "preview_at_s": f"{self.preview_at_s:.1f}",
            "median_aspect": "" if self.median_aspect is None else f"{self.median_aspect:.3f}",
            "status": self.status.value,
            "recorded_at": self.recorded_at.isoformat(timespec="seconds"),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Stream:
        missing = [c for c in COLUMNS if c not in row or row[c] is None]
        if missing:
            raise CropTableError(f"crop row is missing columns: {', '.join(missing)}")
        video_id = parse_video_id(row["video_id"])
        try:
            status = StreamStatus(row["status"])
        except ValueError:
            allowed = ", ".join(s.value for s in StreamStatus)
            raise CropTableError(
                f"{video_id}: unknown status {row['status']!r} (expected one of: {allowed})"
            ) from None
        try:
            median = row["median_aspect"].strip()
            return cls(
                video_id=video_id,
                slug=row["slug"],
                title=row["title"],
                url=row["url"],
                crop=CropBox(
                    x=int(row["x"]),
                    y=int(row["y"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                ),
                frame_width=int(row["frame_width"]),
                frame_height=int(row["frame_height"]),
                preview_at_s=float(row["preview_at_s"]),
                median_aspect=float(median) if median else None,
                status=status,
                recorded_at=datetime.fromisoformat(row["recorded_at"]),
            )
        except ValueError as exc:
            raise CropTableError(f"{video_id}: unreadable crop row: {exc}") from exc


def parse_video_id(text: str) -> VideoId:
    """Validate an 11-char YouTube id."""
    candidate = text.strip()
    if not _VIDEO_ID_RE.match(candidate):
        raise CropTableError(f"not a YouTube video id: {text!r}")
    return VideoId(candidate)


def video_id_from_path(path: Path | str) -> VideoId:
    """Pull the ``[<id>]`` yt-dlp stamps into a downloaded file's name.

    The last bracketed group wins — broadcast titles contain brackets of their
    own, and yt-dlp always appends the id last.
    """
    name = Path(path).name
    matches = _BRACKETED_ID_RE.findall(name)
    if not matches:
        raise UnknownStreamError(
            f"no YouTube id in {name!r} — expected a yt-dlp filename ending in"
            " '[<11-char id>].mp4'. Re-download with yt-dlp's default template,"
            " or rename the file to include the id."
        )
    return VideoId(matches[-1])


def title_from_path(path: Path | str) -> str:
    """Recover the broadcast title from a yt-dlp filename."""
    stem = Path(path).stem
    stem = _BRACKETED_ID_RE.sub("", stem).strip()
    return stem.translate(_FILENAME_UNMANGLE).strip()


def slug_from_title(title: str, max_length: int = 24) -> str:
    """A short, filesystem-safe handle for a stream, derived from its title.

    Only a starting point: the slug names the frames directory, so the operator
    is expected to shorten it by hand in ``crops.csv`` (``hexamind``,
    ``wtr-day2``) before extracting frames.
    """
    words = re.findall(r"[0-9a-z]+", title.lower())
    if not words:
        return "stream"
    slug = words[0]
    for word in words[1:]:
        if len(slug) + 1 + len(word) > max_length:
            break
        slug = f"{slug}-{word}"
    return slug


def load_streams(table: Path = DEFAULT_TABLE) -> Mapping[VideoId, Stream]:
    """Read the crop table, keyed by video id.

    Later rows win: ``crop_selector.py`` appends, so re-selecting a crop for a
    video supersedes the earlier attempt without rewriting the file.
    """
    if not table.exists():
        raise CropTableError(
            f"crop table not found: {table} — record a crop with crop_selector.py first"
        )
    with table.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or tuple(reader.fieldnames) != COLUMNS:
            raise CropTableError(
                f"{table}: unexpected header {reader.fieldnames!r};"
                f" expected {list(COLUMNS)}"
            )
        streams: dict[VideoId, Stream] = {}
        for row in reader:
            stream = Stream.from_row(row)
            streams[stream.video_id] = stream
    return streams


def stream_for_video(
    video_path: Path | str, table: Path = DEFAULT_TABLE
) -> Stream:
    """Resolve a downloaded video file to its registered stream."""
    video_id = video_id_from_path(video_path)
    streams = load_streams(table)
    try:
        return streams[video_id]
    except KeyError:
        raise UnknownStreamError(
            f"no crop recorded for video id {video_id} in {table} —"
            f" run: python3 tools/video/crop_selector.py {Path(video_path).name!r}"
        ) from None


def append_stream(stream: Stream, table: Path = DEFAULT_TABLE) -> None:
    """Append a stream to the crop table, writing the header if it is new."""
    is_new = not table.exists()
    with table.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(stream.to_row())
