"""The edge's control link: its device id and the ``/cameras/{device_id}`` socket."""

from __future__ import annotations

import asyncio
import json

from websockets.asyncio.server import ServerConnection, serve

from board_reader_core.control import (
    AssignedMessage,
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    PreviewsMessage,
    ReplacedMessage,
    SendFullFrameMessage,
)

from board_reader_edge.camera.link import (
    CameraLink,
    cameras_url,
    default_state_path,
    load_device_id,
)

# Non-hex letters on purpose: test_commit_refs reads 8+ hex characters as a commit.
DEVICE = "kitchen-camera-id"


def test_a_new_device_id_is_kept_and_reused(tmp_path):
    path = tmp_path / "state" / "edge-device-id"
    made = load_device_id(path, new_id=lambda: DEVICE)
    assert made == DEVICE
    assert load_device_id(path, new_id=lambda: "never-made-again") == DEVICE


def test_a_bad_device_id_file_is_replaced(tmp_path):
    path = tmp_path / "edge-device-id"
    path.write_text("not a device id!\n")
    assert load_device_id(path, new_id=lambda: DEVICE) == DEVICE
    assert path.read_text().strip() == DEVICE


def test_the_state_file_follows_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_state_path() == tmp_path / "board-reader" / "edge-device-id"


def test_the_link_applies_the_node_and_stops_when_replaced():
    asyncio.run(_link_against_a_fake_node())


async def _link_against_a_fake_node() -> None:
    """Report on connect, apply assigned/config, send a preview and a full frame when
    asked, stop on replaced."""
    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True)
    heard: dict[str, object] = {}

    async def node(ws: ServerConnection) -> None:
        heard["path"] = ws.request.path
        heard["report"] = json.loads(await ws.recv())
        await ws.send(AssignedMessage(board_id="board-3").to_json())
        await ws.send(ConfigMessage(config).to_json())
        await ws.send(PreviewsMessage(on=True).to_json())
        heard["preview"] = await ws.recv()
        await ws.send(SendFullFrameMessage(request=7).to_json())
        heard["full_frame"] = (json.loads(await ws.recv()), await ws.recv())
        await ws.send(ReplacedMessage().to_json())

    assigned: list[str] = []
    configs: list[CameraConfig] = []
    previews: list[bool] = []
    async with asyncio.timeout(10), serve(node, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        link = CameraLink(
            cameras_url(f"ws://127.0.0.1:{port}/", DEVICE),
            on_assigned=assigned.append,
            on_config=configs.append,
            on_previews=lambda on: (previews.append(on), on and link.offer_preview(b"jpeg")),
            on_full_frame=lambda request: link.send_full_frame(request, 12.5, b"whole"),
        )
        link.report(CameraReport(ingest="idle"))
        await link.run()  # returns only on `replaced`

    assert heard["path"] == f"/cameras/{DEVICE}"
    assert heard["report"] == {"type": "report", "report": CameraReport().to_dict()}
    assert assigned == ["board-3"]
    assert configs == [config]
    assert heard["preview"] == b"jpeg"
    # The header, then its JPEG at once: the node pairs them.
    assert heard["full_frame"] == (
        {"type": "full_frame", "request": 7, "capture_ts": 12.5},
        b"whole",
    )
    assert previews == [True, False]  # off again once the socket is gone
