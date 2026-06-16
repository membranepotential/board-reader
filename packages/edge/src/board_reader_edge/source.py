"""Frame sources for the edge (DESIGN.md §6.1 — file-based bring-up first).

A ``FrameSource`` yields :class:`~board_reader_core.wire.Frame` objects. The walking
skeleton (M0) develops everything offline against recorded clips and image
directories before the browser ``getUserMedia`` capture path (WP-C1) exists.

Each source stamps a per-board monotonic ``seq`` and a wall-clock ``capture_ts``.
``capture_ts`` is injectable so tests are deterministic and so a real edge can
later substitute the sensor capture timestamp (DESIGN.md D5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, runtime_checkable

import cv2
import numpy as np

from board_reader_core.wire import Frame

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@runtime_checkable
class FrameSource(Protocol):
    """Anything that can be iterated into a stream of :class:`Frame`."""

    def frames(self) -> Iterator[Frame]:
        ...


class VideoFileSource:
    """Replay a video file (e.g. a ``yt-dlp`` OTB clip) as a frame stream.

    ``capture_ts`` is reconstructed as ``base_ts + frame_index / fps`` so a replayed
    clip carries plausible, monotonic, evenly spaced capture times regardless of how
    fast we actually decode it.
    """

    def __init__(
        self,
        path: str | Path,
        board_id: str = "A1",
        *,
        base_ts: float = 0.0,
        stability: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.board_id = board_id
        self.base_ts = base_ts
        self.stability = stability
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            seq = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(
                    board_id=self.board_id,
                    seq=seq,
                    capture_ts=self.base_ts + seq / fps,
                    image=image,
                    stability=self.stability,
                )
                seq += 1
        finally:
            cap.release()


class ImageDirSource:
    """Replay a directory of images (sorted by filename) as a frame stream."""

    def __init__(
        self,
        directory: str | Path,
        board_id: str = "A1",
        *,
        fps: float = 2.0,
        base_ts: float = 0.0,
        stability: float = 1.0,
    ) -> None:
        self.directory = Path(directory)
        self.board_id = board_id
        self.fps = fps
        self.base_ts = base_ts
        self.stability = stability
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)

    def _paths(self) -> list[Path]:
        return sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )

    def frames(self) -> Iterator[Frame]:
        for seq, path in enumerate(self._paths()):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield Frame(
                board_id=self.board_id,
                seq=seq,
                capture_ts=self.base_ts + seq / self.fps,
                image=image,
                stability=self.stability,
            )


class RepeatSource:
    """Yield a single image ``count`` times, lazily (O(1) memory).

    The synthetic demo feed: a continuous stream of one rendered board without
    materialising every frame. ``count=None`` streams forever.
    """

    def __init__(
        self,
        image: np.ndarray,
        board_id: str = "A1",
        *,
        count: int | None = None,
        fps: float = 2.0,
        base_ts: float = 0.0,
        stability: float = 1.0,
    ) -> None:
        self.image = image
        self.board_id = board_id
        self.count = count
        self.fps = fps
        self.base_ts = base_ts
        self.stability = stability

    def frames(self) -> Iterator[Frame]:
        seq = 0
        while self.count is None or seq < self.count:
            yield Frame(
                board_id=self.board_id,
                seq=seq,
                capture_ts=self.base_ts + seq / self.fps,
                image=self.image,
                stability=self.stability,
            )
            seq += 1


class SyntheticSource:
    """In-memory frame source over pre-built images — the test/sim workhorse.

    ``clock`` defaults to evenly-spaced synthetic times so tests need no real clock.
    """

    def __init__(
        self,
        images: Iterable[np.ndarray],
        board_id: str = "A1",
        *,
        fps: float = 2.0,
        base_ts: float = 0.0,
        stability: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.images = list(images)
        self.board_id = board_id
        self.fps = fps
        self.base_ts = base_ts
        self.stability = stability
        self.clock = clock

    def frames(self) -> Iterator[Frame]:
        for seq, image in enumerate(self.images):
            ts = self.clock() if self.clock is not None else self.base_ts + seq / self.fps
            yield Frame(
                board_id=self.board_id,
                seq=seq,
                capture_ts=ts,
                image=image,
                stability=self.stability,
            )
