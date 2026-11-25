"""The camera run loop: the pipeline's config handling, and the whole camera on a fake node."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator

import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from board_reader_core.control import (
    AssignedMessage,
    CameraConfig,
    ConfigMessage,
    CropRect,
    PreviewsMessage,
    SendFullFrameMessage,
)
from board_reader_core.wire import decode_jpeg

from board_reader_edge.camera.run import (
    SEND_LONG_SIDE,
    CameraPipeline,
    Settings,
    StreamCamera,
    fit_long_side,
)
from board_reader_edge.camera.stream import Captured

DEVICE = "study-camera-id"
CROP = CropRect(0.25, 0.0, 0.5, 1.0)
STREAMING = Settings(board_id="board-2", config=CameraConfig(crop=CROP, streaming=True))


def _frame(t: float, level: int = 120, size: tuple[int, int] = (200, 400)) -> Captured:
    return Captured(capture_ts=t, image=np.full((*size, 3), level, np.uint8))


def test_nothing_is_sent_until_streaming_with_a_board():
    pipeline = CameraPipeline(DEVICE)
    off = Settings(board_id="board-2", config=CameraConfig(crop=CROP, streaming=False))
    no_board = Settings(board_id=None, config=CameraConfig(streaming=True))
    assert pipeline.process(_frame(0.0), off) is None
    assert pipeline.process(_frame(0.1), no_board) is None


def test_a_streaming_frame_is_cropped_and_names_the_assigned_board():
    pipeline = CameraPipeline(DEVICE)
    frame = pipeline.process(_frame(5.0), STREAMING)
    assert frame is not None
    assert frame.board_id == "board-2"
    assert frame.device_id == DEVICE  # the node takes board-2 only from its camera
    assert frame.capture_ts == 5.0
    assert frame.image.shape[:2] == (200, 200)  # the crop's half of a 400-wide frame


def test_a_new_crop_is_not_motion():
    """Two crops of one still scene differ, yet the switch must not open a burst."""
    pipeline = CameraPipeline(DEVICE)
    scene = np.zeros((200, 400, 3), np.uint8)
    scene[:, 200:] = 255  # left half black, right half white
    left = Settings(board_id="b", config=CameraConfig(crop=CropRect(0.0, 0.0, 0.5, 1.0), streaming=True))
    right = Settings(board_id="b", config=CameraConfig(crop=CropRect(0.5, 0.0, 0.5, 1.0), streaming=True))
    for i in range(5):
        pipeline.process(Captured(i * 0.05, scene), left)
    forwarded = [pipeline.process(Captured(0.25 + i * 0.05, scene), right) for i in range(5)]
    assert pipeline.forwarder.burst_id is None
    assert all(f is None or f.stability == 1.0 for f in forwarded)


def test_seq_never_repeats_across_a_streaming_toggle():
    pipeline = CameraPipeline(DEVICE)
    first = pipeline.process(_frame(0.0), STREAMING)
    pipeline.process(_frame(0.05), Settings(board_id="board-2"))
    second = pipeline.process(_frame(0.1), STREAMING)
    assert first is not None and second is not None
    assert second.seq > first.seq


def test_frames_are_shrunk_to_the_phone_send_size():
    big = np.zeros((1080, 1920, 3), np.uint8)
    assert fit_long_side(big, SEND_LONG_SIDE).shape[:2] == (540, 960)
    small = np.zeros((100, 50, 3), np.uint8)
    assert fit_long_side(small, SEND_LONG_SIDE) is small  # never enlarged


class Endless:
    """A live-like source: a still frame every 20 ms until stopped."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._t = 0.0

    def captures(self) -> Iterator[Captured]:
        while not self._stop.wait(0.02):
            self._t += 0.05
            yield _frame(self._t, size=(240, 320))

    def stop(self) -> None:
        self._stop.set()


def test_the_camera_streams_under_its_assigned_board_with_its_crop():
    asyncio.run(_camera_against_a_fake_node())


async def _camera_against_a_fake_node() -> None:
    """The node assigns a board and a crop, asks for a full frame, turns streaming
    and previews on; the full frame, previews and frames follow."""
    heard: dict[str, object] = {"reports": []}
    got_frame = asyncio.Event()

    async def node(ws: ServerConnection) -> None:
        if ws.request.path.startswith("/cameras/"):
            await ws.send(AssignedMessage(board_id=DEVICE, name="board-5").to_json())
            # Asked before any frame exists: answered with the first one.
            await ws.send(SendFullFrameMessage(request=3).to_json())
            await ws.send(ConfigMessage(CameraConfig(crop=CROP, streaming=True)).to_json())
            await ws.send(PreviewsMessage(on=True).to_json())
            full_frame: dict | None = None
            async for message in ws:
                if isinstance(message, bytes):
                    shape = decode_jpeg(message).shape[:2]
                    if full_frame is not None:
                        heard["full_frame"] = (full_frame["request"], shape)
                        full_frame = None
                    else:
                        heard["preview"] = shape
                elif (parsed := json.loads(message))["type"] == "full_frame":
                    full_frame = parsed
                else:
                    heard["reports"].append(parsed["report"])  # type: ignore[union-attr]
                if {"preview", "full_frame"} <= heard.keys() and got_frame.is_set():
                    return
        else:  # /ingest
            header = json.loads(await ws.recv())
            jpeg = await ws.recv()
            await ws.send(json.dumps({"ack": header["seq"], "board_id": header["board_id"]}))
            heard["header"] = header
            heard["frame"] = decode_jpeg(jpeg).shape[:2]
            got_frame.set()
            await ws.wait_closed()

    async with asyncio.timeout(10), serve(node, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        camera = StreamCamera(
            Endless(),
            node=f"ws://127.0.0.1:{port}",
            device_id=DEVICE,
            title="Round 5",
            preview_s=0.1,
        )
        run = asyncio.create_task(camera.run())
        await got_frame.wait()
        while not {"preview", "full_frame"} <= heard.keys():
            await asyncio.sleep(0.02)
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)

    header = heard["header"]
    assert isinstance(header, dict)
    assert header["board_id"] == DEVICE
    assert header["device_id"] == DEVICE
    assert heard["frame"] == (240, 160)  # cropped to the middle half
    assert heard["preview"] == (240, 320)  # the whole frame, already under the preview size
    assert heard["full_frame"] == (3, (240, 320))  # uncropped
    reports = heard["reports"]
    assert isinstance(reports, list)
    last = reports[-1]
    assert last["devices"] == [{"id": "stream", "label": "Round 5"}]
    assert last["frame"] == {"width": 320, "height": 240}
    assert last["kind"] == "stream"
    assert "streaming" in {r["ingest"] for r in reports}
