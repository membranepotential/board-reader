"""Frame sources for the edge (DESIGN.md §6.1 — file-based bring-up first).

A ``FrameSource`` yields :class:`~board_reader_core.wire.Frame` objects. The walking
skeleton (M0) develops everything offline against recorded clips and image
directories before the browser ``getUserMedia`` capture path (WP-C1) exists.

Each source stamps a per-board monotonic ``seq`` and a wall-clock ``capture_ts``.
``capture_ts`` is injectable so tests are deterministic and so a real edge can
later substitute the sensor capture timestamp (DESIGN.md D5).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS, render_game_frames
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


class GameSource:
    """Replay a synthetic *recorded game* as a frame stream — the M2 demo feed.

    Wraps :func:`board_reader_core.sim.render_game_frames`: settled positions
    interleaved with transitions in which a hand sweeps over the changed squares.
    That is the whole point of M2 — the edge sees real motion and real occlusion,
    so :class:`~board_reader_edge.motion.MotionMeter` has something to measure and
    :class:`~board_reader_edge.forwarder.AdaptiveForwarder` something to burst on —
    with zero external files.

    ``fps`` is the *capture* rate: it only spaces ``capture_ts``, which is the time
    base every downstream rate decision uses (DESIGN D5). Frames are rendered as
    they are pulled, never accumulated, so a long game costs O(1) memory.

    Deliberately no ``stability`` knob, unlike the other sources: on a stream that
    actually moves, a constant would be a lie. The forwarder stamps the measured
    value.
    """

    def __init__(
        self,
        placements: Sequence[str] = SAMPLE_GAME_PLACEMENTS,
        board_id: str = "A1",
        *,
        fps: float = 12.0,
        base_ts: float = 0.0,
        frames_per_position: int = 30,
        transition_frames: int = 4,
        hand: bool = True,
        angled: bool = False,
        seed: int = 0,
        square_px: int = 48,
    ) -> None:
        self.placements = tuple(placements)
        self.board_id = board_id
        self.fps = fps
        self.base_ts = base_ts
        self.frames_per_position = frames_per_position
        self.transition_frames = transition_frames
        self.hand = hand
        self.angled = angled
        self.seed = seed
        self.square_px = square_px

    def frames(self) -> Iterator[Frame]:
        images = render_game_frames(
            self.placements,
            frames_per_position=self.frames_per_position,
            transition_frames=self.transition_frames,
            hand=self.hand,
            angled=self.angled,
            seed=self.seed,
            square_px=self.square_px,
        )
        for seq, image in enumerate(images):
            yield Frame(
                board_id=self.board_id,
                seq=seq,
                capture_ts=self.base_ts + seq / self.fps,
                image=image,
            )


class LoopSource:
    """Repeat another source forever, re-stamping ``seq`` and ``capture_ts``.

    A finite source (a clip, a directory, a game) becomes an endless feed — what
    the ``docker compose up`` demo and any long soak run want. The restamping is
    the point: each pass restarts its own numbering at zero, and a ``seq`` that
    jumps backwards would make the gateway's per-board monotone ordering (D5) a
    lie. Capture time continues across the seam at the stream's own last observed
    inter-frame gap.

    Lazy, like everything else in this module: one pass is never buffered.
    """

    def __init__(self, source: FrameSource) -> None:
        self.source = source

    def frames(self) -> Iterator[Frame]:
        seq = 0
        resume_at = 0.0
        while True:
            offset = 0.0
            prev_ts: float | None = None
            step = 0.0
            for frame in self.source.frames():
                if prev_ts is None:
                    offset = resume_at - frame.capture_ts
                ts = frame.capture_ts + offset
                if prev_ts is not None:
                    step = ts - prev_ts
                yield dataclasses.replace(frame, seq=seq, capture_ts=ts)
                seq += 1
                prev_ts = ts
            if prev_ts is None:
                # An empty source would spin this loop at full speed forever.
                raise RuntimeError(f"cannot loop an empty source: {self.source!r}")
            resume_at = prev_ts + step


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
