"""Which command runs a stream."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from board_reader_inference.control.streams.command import EdgeCommand, NoEdge, edge_command


def executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_edge_on_path_runs_the_streams(tmp_path):
    script = executable(tmp_path / "board-reader-edge")
    assert edge_command({"PATH": str(tmp_path)}, python="/nowhere/python") == EdgeCommand(
        (str(script),)
    )


def test_the_edge_beside_this_python_when_path_has_none(tmp_path):
    script = executable(tmp_path / "board-reader-edge")
    found = edge_command({"PATH": os.devnull}, python=str(tmp_path / "python"))
    assert found == EdgeCommand((str(script),))


def test_br_edge_command_overrides_and_is_split_like_a_shell(tmp_path):
    executable(tmp_path / "uv")
    environ = {"PATH": str(tmp_path), "BR_EDGE_COMMAND": "uv run 'board-reader-edge'"}
    assert edge_command(environ) == EdgeCommand(("uv", "run", "board-reader-edge"))


def test_no_edge_says_what_to_do(tmp_path):
    missing = edge_command({"PATH": str(tmp_path)}, python=str(tmp_path / "python"))
    assert isinstance(missing, NoEdge)
    assert "BR_EDGE_COMMAND" in missing.reason
    wrong = edge_command({"PATH": str(tmp_path), "BR_EDGE_COMMAND": "nope"})
    assert isinstance(wrong, NoEdge) and "nope is not a command here" in wrong.reason


def test_the_camera_argv_passes_a_start_only_when_there_is_one():
    edge = EdgeCommand(("board-reader-edge",))
    argv = edge.camera("https://v.test/x", node="ws://n:1", state=Path("/s"), start_s=None)
    assert argv == (
        "board-reader-edge", "camera", "https://v.test/x", "--node", "ws://n:1", "--state", "/s",
    )
    assert edge.camera("https://v.test/x", node="ws://n:1", state=Path("/s"), start_s=75.5)[-2:] == (
        "--start", "75.5",
    )
