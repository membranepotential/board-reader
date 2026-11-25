"""A video stream as a camera: a YouTube broadcast, a live stream or a file.

:func:`resolve` turns what the operator typed into something OpenCV opens. A web page
(``http(s)://``: YouTube and the other sites yt-dlp knows, or a direct media link)
goes through yt-dlp, which picks one format of at most ``max_height`` lines (H.264
first, because every FFmpeg build decodes it) and says whether it is live. Anything
else (``rtsp://``, a local file) goes to OpenCV unchanged.

:class:`StreamSource` reads it with OpenCV's FFmpeg backend, which reads YouTube's
progressive https formats and its live HLS playlists as they are, and samples it at
``fps`` (the phone's ``CAPTURE_FPS``), because the motion meter is tuned to the
phone's sampling rate. The two kinds differ in time:

* **A video (VOD)** plays at real time, paced by its own timestamps, from ``start_s``,
  once. ``capture_ts`` is the video's own timeline mapped onto wall time when playback
  (re)starts: ``capture_ts = wall_at_open + (pts - pts_at_open)``. So a frame's time
  is when it plays, and the gaps are the video's, even when decoding falls behind.
  A read that fails before the end (an expired or dropped URL) resolves again and
  resumes where it stopped.
* **A live stream** is read as fast as it arrives and reconnects (resolving again)
  whenever it drops. ``capture_ts`` is the wall time at decode. Right after opening,
  FFmpeg reads the playlist's backlog (about 25 s on YouTube) faster than real time,
  so for a few seconds the frames come denser than they were filmed.

Everything here blocks; it runs on the camera thread (:mod:`.run`), never on the
event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: The phone's sampling rate (``CAPTURE_FPS`` in ``lib/edge/motion.ts``): the motion
#: meter's gains and the forwarder's hysteresis are tuned for frames this far apart.
CAPTURE_FPS = 20.0

#: Slack on the sampling grid, so a frame exactly on its due time is not skipped.
_DUE_EPS = 1e-6

#: How long OpenCV may wait to open a stream or to read one frame from it. A
#: stalled read then fails, and the source reconnects, instead of hanging forever.
OPEN_TIMEOUT_MS = 15_000
READ_TIMEOUT_MS = 15_000

#: A VOD read that fails this close to the end is the end.
_END_SLACK_S = 1.0
#: Reopens of a VOD in a row that bring no new frame before it is given up on.
_MAX_VOD_RETRIES = 3


@dataclass(frozen=True)
class Resolved:
    """What to open: the media URL (or path), whether it is live, and its name."""

    media: str
    live: bool
    title: str


@dataclass(frozen=True)
class Captured:
    """One sampled frame: when it was captured (epoch seconds) and its pixels."""

    capture_ts: float
    image: np.ndarray


class CaptureSource(Protocol):
    """What the camera thread reads: sampled frames until the end or :meth:`stop`."""

    def captures(self) -> Iterator[Captured]: ...

    def stop(self) -> None:
        """End :meth:`captures` from another thread."""
        ...


def format_selector(max_height: int) -> str:
    """yt-dlp's format choice: one format with video, at most ``max_height`` lines.

    ``bv*`` is the best format that has video (with or without audio; OpenCV ignores
    the audio). H.264 first, because every FFmpeg build decodes it, then any codec.
    No frame-rate preference: the source samples at :data:`CAPTURE_FPS` anyway, a
    1080p60 decode keeps up (~95 fps measured), and lines are what the reader needs
    (preferring 30 fps picked 480p on a 1080p60 news stream).
    """
    h = f"[height<={max_height}]"
    return f"bv*{h}[vcodec^=avc1]/bv*{h}/b{h}"


def _yt_dlp_info(url: str, selector: str) -> dict[str, Any]:
    """yt-dlp's metadata for ``url`` with one format chosen; the only yt-dlp call."""
    import yt_dlp  # imported here: it takes a moment, and `feed` never needs it

    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": selector}
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


def is_web_page(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve(
    source: str,
    *,
    max_height: int = 1080,
    info: Callable[[str, str], dict[str, Any]] = _yt_dlp_info,
) -> Resolved:
    """What ``source`` is and how to open it; ``ValueError`` says why it cannot be.

    ``info`` is yt-dlp's metadata lookup, injectable so tests need no network.
    """
    if not is_web_page(source):
        path = Path(source)
        if path.exists():
            return Resolved(media=str(path), live=False, title=path.name)
        return Resolved(media=source, live=True, title=source)  # rtsp://, srt://, ...
    try:
        meta = info(source, format_selector(max_height))
    except Exception as e:  # yt-dlp raises its own DownloadError for everything
        raise ValueError(f"yt-dlp could not resolve {source}: {e}") from e
    if meta.get("_type") == "playlist":
        raise ValueError(f"{source} is a playlist or a channel; give one video or stream")
    media = meta.get("url")
    if not isinstance(media, str):
        raise ValueError(f"yt-dlp found no single video URL for {source}")
    title = meta.get("title")
    return Resolved(
        media=media,
        live=meta.get("is_live") is True,
        title=title if isinstance(title, str) and title else source,
    )


def open_capture(media: str) -> cv2.VideoCapture:
    """``media`` opened with OpenCV's FFmpeg backend; ``OSError`` if it does not open."""
    cap = cv2.VideoCapture(
        media,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
        ],
    )
    if not cap.isOpened():
        cap.release()
        raise OSError("OpenCV (FFmpeg) could not open the stream")
    return cap


def sample(
    cap: cv2.VideoCapture, fps: float, stopped: Callable[[], bool]
) -> Iterator[tuple[float, np.ndarray]]:
    """``(pts, image)`` at about ``fps``, by the stream's own timestamps (seconds).

    Frames between two due times are grabbed but never converted to BGR, which is
    most of what a skipped frame would cost. A stream whose timestamps do not advance
    is timed by its frame rate instead.
    """
    step = 1.0 / fps
    frame_s = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    due: float | None = None
    last: float | None = None
    while not stopped() and cap.grab():
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if last is not None and pts <= last:
            pts = last + frame_s
        last = pts
        if due is not None and pts < due - _DUE_EPS:
            continue
        ok, image = cap.retrieve()
        if not ok:
            return
        due = (pts if due is None else due) + step
        if due <= pts:
            due = pts + step  # a gap in the stream: restart the grid after it
        yield pts, image


class StreamSource:
    """Sample a resolved stream as :class:`Captured` frames (see the module note).

    ``reresolve`` gives a fresh media URL on every reopen, because a stream page's
    URL expires (YouTube's after about six hours). :meth:`stop` ends the iteration
    from another thread, within one frame or one wait.
    """

    def __init__(
        self,
        first: Resolved,
        *,
        reresolve: Callable[[], Resolved] | None = None,
        fps: float = CAPTURE_FPS,
        start_s: float = 0.0,
        opener: Callable[[str], cv2.VideoCapture] = open_capture,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
        retry_s: float = 1.0,
        max_retry_s: float = 30.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.first = first
        self._reresolve = reresolve
        self.fps = fps
        self.start_s = start_s
        self._opener = opener
        self._clock = clock
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._sleep = sleep if sleep is not None else self._stop.wait
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s

    @property
    def live(self) -> bool:
        return self.first.live

    def stop(self) -> None:
        self._stop.set()

    def _open(self, attempt: int) -> cv2.VideoCapture:
        media = self.first.media
        if attempt > 0 and self._reresolve is not None:
            media = self._reresolve().media
        return self._opener(media)

    def captures(self) -> Iterator[Captured]:
        yield from (self._live() if self.live else self._vod())

    def _backoff(self, failures: int) -> None:
        self._sleep(min(self._retry_s * 2 ** max(failures - 1, 0), self._max_retry_s))

    def _live(self) -> Iterator[Captured]:
        attempt = 0
        failures = 0
        while not self._stop.is_set():
            try:
                cap = self._open(attempt)
            except (OSError, ValueError) as e:
                failures += 1
                log.warning("cannot open the live stream (%s); retrying", e)
                self._backoff(failures)
                attempt += 1
                continue
            try:
                for _, image in sample(cap, self.fps, self._stop.is_set):
                    failures = 0
                    yield Captured(capture_ts=self._clock(), image=image)
            finally:
                cap.release()
            if self._stop.is_set():
                return
            failures += 1
            log.warning("the live stream dropped; reconnecting")
            self._backoff(failures)
            attempt += 1

    def _vod(self) -> Iterator[Captured]:
        position = self.start_s  # where the next open starts, video seconds
        played_to: float | None = None  # pts of the last frame yielded
        attempt = 0
        idle = 0  # reopens in a row that brought no new frame
        while not self._stop.is_set():
            try:
                cap = self._open(attempt)
            except (OSError, ValueError) as e:
                if attempt == 0:
                    raise
                log.warning("cannot reopen the video (%s)", e)
                idle += 1
                if idle >= _MAX_VOD_RETRIES:
                    return
                self._backoff(idle)
                attempt += 1
                continue
            progressed = False
            try:
                if position > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
                duration = _duration_s(cap)
                anchor: tuple[float, float, float] | None = None  # pts, wall, monotonic
                for pts, image in sample(cap, self.fps, self._stop.is_set):
                    if played_to is not None and pts <= played_to:
                        continue  # a seek lands on the keyframe before the position
                    if anchor is None:
                        anchor = (pts, self._clock(), self._monotonic())
                    offset = pts - anchor[0]
                    ahead = anchor[2] + offset - self._monotonic()
                    if ahead > 0:
                        self._sleep(ahead)
                        if self._stop.is_set():
                            return
                    progressed = True
                    played_to = position = pts
                    yield Captured(capture_ts=anchor[1] + offset, image=image)
            finally:
                cap.release()
            if self._stop.is_set():
                return
            if duration is None or (played_to or 0.0) >= duration - _END_SLACK_S:
                return  # the end of the video
            idle = 0 if progressed else idle + 1
            if idle >= _MAX_VOD_RETRIES:
                log.warning("the video stopped at %.0f s of %.0f s; giving up", position, duration)
                return
            log.warning("the video dropped at %.0f s; resuming", position)
            attempt += 1


def _duration_s(cap: cv2.VideoCapture) -> float | None:
    """The video's length, if the container says (frames / fps)."""
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frames / fps if frames > 0 and fps > 0 else None
