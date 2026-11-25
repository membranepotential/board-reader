"""The stream source: what a URL resolves to, sampling, VOD pacing, live reconnects.

No network: yt-dlp's lookup is a fake, and the videos are written here with OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_edge.camera.stream import (
    Resolved,
    StreamSource,
    format_selector,
    open_capture,
    resolve,
    sample,
)


def _video(path: Path, *, frames: int, fps: float) -> Path:
    """A small MJPG clip whose frame ``i`` is uniformly grey level ``20 + 3 * i``."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), 20 + 3 * i, np.uint8))
    writer.release()
    return path


def _index(image: np.ndarray) -> int:
    return round((float(image.mean()) - 20) / 3)


class FakeClock:
    """Wall and monotonic time that move only when the source sleeps."""

    def __init__(self, wall: float = 1_000.0) -> None:
        self.wall = wall
        self.mono = 0.0
        self.slept = 0.0

    def sleep(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds
        self.slept += seconds


# ---- resolve -----------------------------------------------------------------


def test_a_web_page_goes_through_yt_dlp():
    asked: list[tuple[str, str]] = []

    def info(url: str, selector: str) -> dict:
        asked.append((url, selector))
        return {"url": "https://media.example/v.mp4", "is_live": True, "title": "Round 5"}

    got = resolve("https://www.youtube.com/watch?v=abc", max_height=720, info=info)
    assert got == Resolved(media="https://media.example/v.mp4", live=True, title="Round 5")
    assert asked == [("https://www.youtube.com/watch?v=abc", format_selector(720))]
    assert "[height<=720]" in asked[0][1]


def test_a_finished_broadcast_is_a_video():
    got = resolve("https://x.example/p", info=lambda u, s: {"url": "m", "is_live": False})
    assert got.live is False
    assert got.title == "https://x.example/p"  # no title: the page names it


@pytest.mark.parametrize(
    "meta",
    [{"_type": "playlist", "entries": []}, {"title": "needs merging, no single url"}],
)
def test_what_is_no_single_video_is_refused(meta):
    with pytest.raises(ValueError):
        resolve("https://x.example/p", info=lambda u, s: meta)


def test_a_yt_dlp_failure_reads_as_a_value_error():
    def info(url: str, selector: str) -> dict:
        raise RuntimeError("Video unavailable")

    with pytest.raises(ValueError, match="Video unavailable"):
        resolve("https://x.example/p", info=info)


def test_a_file_is_a_video_and_any_other_url_is_live(tmp_path):
    clip = _video(tmp_path / "clip.avi", frames=2, fps=10)
    assert resolve(str(clip)) == Resolved(media=str(clip), live=False, title="clip.avi")
    assert resolve("rtsp://cam.local/stream").live is True


# ---- sampling and pacing -----------------------------------------------------


def test_sampling_thins_a_30_fps_clip_to_the_meter_rate(tmp_path):
    cap = open_capture(str(_video(tmp_path / "c.avi", frames=60, fps=30)))
    try:
        taken = [_index(image) for _, image in sample(cap, 20.0, lambda: False)]
    finally:
        cap.release()
    assert len(taken) == 40  # 2 s at 20 fps: two of every three frames
    assert taken[:6] == [0, 2, 3, 5, 6, 8]


def test_a_video_plays_at_real_time_from_its_start(tmp_path):
    clip = _video(tmp_path / "c.avi", frames=30, fps=10)  # 3 s
    clock = FakeClock()
    source = StreamSource(
        resolve(str(clip)),
        fps=10,
        start_s=1.0,
        clock=lambda: clock.wall,
        monotonic=lambda: clock.mono,
        sleep=clock.sleep,
    )
    got = list(source.captures())
    assert [_index(c.image) for c in got] == list(range(10, 30))
    # The video's own timeline, mapped onto the wall time when playback started.
    assert got[0].capture_ts == pytest.approx(1_000.0)
    assert got[-1].capture_ts == pytest.approx(1_001.9)
    assert clock.slept == pytest.approx(1.9)  # paced, not decoded flat out


def test_a_live_stream_reconnects_and_stamps_wall_time(tmp_path):
    clip = str(_video(tmp_path / "c.avi", frames=3, fps=10))
    clock = FakeClock()
    opens: list[str] = []

    def opener(media: str) -> cv2.VideoCapture:
        opens.append(media)
        if len(opens) == 2:
            raise OSError("dropped")
        if len(opens) == 4:
            source.stop()
        return open_capture(clip)

    source = StreamSource(
        Resolved(media="first", live=True, title="live"),
        reresolve=lambda: Resolved(media="fresh", live=True, title="live"),
        fps=10,
        opener=opener,
        clock=lambda: clock.wall,
        sleep=clock.sleep,
    )
    got = list(source.captures())
    assert opens == ["first", "fresh", "fresh", "fresh"]  # a new URL on every reopen
    assert len(got) == 6  # two whole reads; the stop ends the fourth
    # Decode time. The fake wall moves only in the backoff sleeps: 1 s after the
    # first read ended, 2 s after the failed open.
    assert [c.capture_ts for c in got] == [1_000.0] * 3 + [1_003.0] * 3
