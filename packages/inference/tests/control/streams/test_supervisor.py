"""The stream supervisor, with ``fake_edge.py`` in place of ``board-reader-edge``."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from board_reader_inference.control.camera.registry import CameraRegistry
from board_reader_inference.control.streams.command import EdgeCommand, NoEdge
from board_reader_inference.control.streams.store import state_file
from board_reader_inference.control.streams.supervisor import Backoff, StreamSupervisor

FAKE = EdgeCommand((sys.executable, str(Path(__file__).with_name("fake_edge.py"))))
NODE = "ws://127.0.0.1:8765"
QUICK = Backoff(first_s=0.05, cap_s=0.2, healthy_s=60.0)


@pytest.fixture(autouse=True)
def _in_memory_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BR_CAMERA_CONFIG", raising=False)


def run(test: Callable[[], Coroutine[Any, Any, None]]) -> None:
    async def guarded() -> None:
        async with asyncio.timeout(10):
            await test()

    asyncio.run(guarded())


async def until(condition: Callable[[], bool], what: str) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.02)
    assert condition(), what


def alive(pid: int) -> bool:
    """Whether ``ps`` lists ``pid`` as a live process (a zombie is not)."""
    stat = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def supervisor(tmp_path: Path, registry: CameraRegistry, **kw: Any) -> StreamSupervisor:
    return StreamSupervisor(
        registry,
        command=kw.pop("command", FAKE),
        node=NODE,
        streams_file=tmp_path / "streams.json",
        backoff=kw.pop("backoff", QUICK),
        grace_s=1.0,
        **kw,
    )


def runs(tmp_path: Path, stream_id: str) -> list[list[str]]:
    path = Path(f"{state_file(tmp_path, stream_id)}.runs")
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_stream_runs_the_edge_camera_command_and_stops_without_orphans(tmp_path):
    async def go() -> None:
        streams = supervisor(tmp_path, CameraRegistry())
        streams.start()
        stream_id = streams.add("https://video.test/sleep", 3600.0)
        pids = Path(f"{state_file(tmp_path, stream_id)}.pids")
        await until(pids.exists, "the fake edge started")
        edge, grandchild = (int(p) for p in pids.read_text().split())
        [entry] = streams.streams().streams
        assert (entry.id, entry.state, entry.name) == (stream_id, "starting", None)
        assert runs(tmp_path, stream_id) == [[
            "camera", "https://video.test/sleep", "--node", NODE,
            "--state", str(state_file(tmp_path, stream_id)), "--start", "3600.0",
        ]]
        assert state_file(tmp_path, stream_id).read_text() == entry.device_id + "\n"
        assert alive(edge) and alive(grandchild)
        await streams.stop()
        await until(lambda: not alive(edge) and not alive(grandchild), "both processes gone")

    run(go)


def test_a_crashed_stream_starts_again_and_says_why(tmp_path, caplog):
    async def go() -> None:
        streams = supervisor(tmp_path, CameraRegistry())
        streams.start()
        stream_id = streams.add("https://video.test/crash", None)
        await until(lambda: len(runs(tmp_path, stream_id)) >= 3, "three starts")
        [entry] = streams.streams().streams
        assert entry.error == "exited with code 3: boom: no such video"
        assert "--start" not in runs(tmp_path, stream_id)[0]
        await streams.stop()

    with caplog.at_level(logging.INFO):
        run(go)
    assert "[stream-1] boom: no such video" in caplog.text


def test_the_restart_delay_doubles_up_to_the_cap():
    backoff = Backoff(first_s=1.0, cap_s=10.0)
    assert [backoff.delay(n) for n in range(1, 7)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


def test_a_stream_that_ends_stays_stopped(tmp_path):
    async def go() -> None:
        streams = supervisor(tmp_path, CameraRegistry())
        streams.start()
        streams.add("https://video.test/end", None)
        await until(lambda: streams.streams().streams[0].state == "stopped", "stopped")
        # Its task returned rather than waiting to start it again.
        await until(lambda: asyncio.all_tasks() == {asyncio.current_task()}, "no task left")
        assert len(runs(tmp_path, "stream-1")) == 1
        await streams.stop()

    run(go)


def test_the_streams_start_again_as_the_same_cameras_after_a_restart(tmp_path):
    async def first() -> str:
        streams = supervisor(tmp_path, CameraRegistry())
        streams.start()
        streams.add("https://video.test/sleep", 90.0)
        await until(lambda: len(runs(tmp_path, "stream-1")) == 1, "started")
        await streams.stop()
        return streams.streams().streams[0].device_id

    async def second(device_id: str) -> None:
        streams = supervisor(tmp_path, CameraRegistry())
        [entry] = streams.streams().streams
        assert (entry.id, entry.url, entry.start_s, entry.device_id) == (
            "stream-1", "https://video.test/sleep", 90.0, device_id,
        )
        streams.start()
        await until(lambda: len(runs(tmp_path, "stream-1")) == 2, "started again")
        assert state_file(tmp_path, "stream-1").read_text() == device_id + "\n"
        assert streams.add("https://video.test/sleep", None) == "stream-2"
        await streams.stop()

    device: list[str] = []

    async def both() -> None:
        device.append(await first())
        await second(device[0])

    run(both)


def test_removing_a_stream_stops_it_and_forgets_its_camera(tmp_path):
    async def go() -> None:
        registry = CameraRegistry()
        streams = supervisor(tmp_path, registry)
        streams.start()
        stream_id = streams.add("https://video.test/sleep", None)
        pids = Path(f"{state_file(tmp_path, stream_id)}.pids")
        await until(pids.exists, "the fake edge started")
        edge = int(pids.read_text().split()[0])
        device_id = streams.streams().streams[0].device_id
        # The edge registers on /cameras/{device_id}; here the test does it for it.
        link = registry.connect_phone(device_id)
        assert streams.streams().streams[0].state == "running"
        assert streams.streams().streams[0].name == "board-1"
        streams.remove(stream_id)
        assert streams.streams().streams == ()
        await until(lambda: not alive(edge), "the process stopped")
        registry.disconnect_phone(link)  # its socket closes with the process
        await until(lambda: registry.name_of(device_id) is None, "the camera forgotten")
        assert not state_file(tmp_path, stream_id).exists()
        await streams.stop()

    run(go)


def test_a_node_without_the_edge_refuses_to_add_a_stream(tmp_path):
    async def go() -> None:
        streams = supervisor(tmp_path, CameraRegistry(), command=NoEdge("no edge here"))
        streams.start()
        with pytest.raises(ValueError, match="no edge here"):
            streams.add("https://video.test/sleep", None)
        assert streams.streams().streams == ()
        await streams.stop()

    run(go)
