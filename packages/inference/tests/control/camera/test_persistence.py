"""The configs on disk: a restarted node gives each phone its board, name and config back."""

from __future__ import annotations

import json
import logging
import os

import pytest

from board_reader_core.control import (
    AssignedMessage,
    CameraConfig,
    ConfigMessage,
    SendFullFrameMessage,
)
from board_reader_core.protocol import BoardName, NamesMessage

from board_reader_inference.control.camera.persistence import CAMERA_CONFIG_ENV
from board_reader_inference.control.camera.registry import CameraRegistry

CONFIG = {
    "crop": {"x": 0.2, "y": 0.1, "w": 0.55, "h": 0.8},
    "device": "Back Ultra Wide Camera",
    "resolution": "2160p",
    "streaming": True,
    # A crop drawn on /control: Auto-frame is off for it.
    "auto": False,
}
#: ``CONFIG`` as the node sends it: every key, the defaulted ``rules`` included.

PHONE = "kitchen-phone-id"
OTHER = "garden-phone-id"


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "camera-config.json"
    monkeypatch.setenv(CAMERA_CONFIG_ENV, str(path))
    return path


def test_a_restarted_node_gives_each_phone_its_board_name_and_config_back(config_file):
    config = CameraConfig.from_dict(CONFIG)
    registry = CameraRegistry()
    registry.connect_phone(OTHER)
    registry.connect_phone(PHONE)
    registry.rename(PHONE, "left")
    registry.configure("board-2", config)
    assert config_file.exists()
    assert not list(config_file.parent.glob(".*.tmp")), "the temporary file was renamed"

    heard: list[NamesMessage] = []
    restarted = CameraRegistry(on_names=heard.append)
    phone = restarted.connect_phone(PHONE)
    assert phone.mailbox.take_nowait()[:2] == [
        AssignedMessage("board-2", "left"), ConfigMessage(config),
    ]
    assert restarted.board_of(OTHER) == "board-1"
    assert heard == [NamesMessage(names=(BoardName("board-2", "left"),))]


def test_a_corrupt_config_file_is_ignored_with_a_warning(config_file, caplog):
    config_file.parent.mkdir(parents=True)
    bad = {"boards": {"phone": {**CONFIG, "resolution": "8k"}}}
    twice = {"boards": {}, "cameras": {PHONE: "left", OTHER: "left"}}
    stray_name = {"boards": {}, "cameras": {PHONE: "left"}, "names": {OTHER: "garden"}}
    for raw in ("{not json", bad, {"phone": CONFIG}, twice, stray_name):
        config_file.write_text(raw if isinstance(raw, str) else json.dumps(raw))
        with caplog.at_level(logging.WARNING):
            registry = CameraRegistry()
        assert registry.board("phone").config == CameraConfig()
        assert registry.board_of(PHONE) is None
        assert "corrupt camera config" in caplog.text
        caplog.clear()

    # The next change replaces the corrupt file with a whole one.
    registry.connect_phone(PHONE)
    registry.configure("board-1", CameraConfig(streaming=True))
    assert CameraRegistry().board("board-1").config == CameraConfig(streaming=True)


def test_a_file_from_before_the_device_ids_loads_and_drops_its_orphaned_configs(
    config_file, caplog
):
    """Such a file has only ``boards`` (``phone`` was every phone's default id): no
    camera holds them, so they are dropped with one INFO line, and no new camera can
    inherit a crop or ``streaming: true`` from them."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"boards": {"phone": CONFIG, "board-1": CONFIG}}))
    with caplog.at_level(logging.INFO):
        registry = CameraRegistry()
    assert [r.getMessage() for r in caplog.records] == [
        "dropping the saved config of board-1, phone: no camera holds them"
    ]
    assert registry.board("phone").config == CameraConfig()
    phone = registry.connect_phone(PHONE)
    assert phone.mailbox.take_nowait()[:2] == [AssignedMessage("board-1"), ConfigMessage(CameraConfig())]
    registry.rename(PHONE, "phone")
    assert registry.board("board-1").config == CameraConfig()


def test_a_file_from_before_the_names_keeps_every_board_id(config_file):
    """A camera renamed before names existed kept the new name as its board id; it
    stays that camera's board id, so nothing recorded under it changes."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps({"boards": {"left": CONFIG}, "cameras": {PHONE: "left", OTHER: "board-1"}})
    )
    registry = CameraRegistry()
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[:2] == [
        AssignedMessage("left"), ConfigMessage(CameraConfig.from_dict(CONFIG)),
    ]
    assert (registry.label("left"), registry.label("board-1")) == ("left", "board-1")


def test_a_config_saved_before_the_rules_existed_loads_as_standard(config_file):
    """A config file from before the switch has no ``rules`` key: still standard."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"boards": {"board-1": CONFIG}, "cameras": {PHONE: "board-1"}}))
    registry = CameraRegistry()
    assert registry.board("board-1").config == CameraConfig.from_dict(CONFIG)
    assert registry.rules("board-1") == "standard"


def test_the_rules_are_per_board_and_survive_a_restart(config_file):
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.configure("board-1", CameraConfig(rules="free"))
    restarted = CameraRegistry()
    assert restarted.rules("board-1") == "free"
    assert restarted.rules("phone") == "standard"  # never configured: standard
    assert "phone" not in restarted._boards  # asking did not create the board


def test_without_the_env_var_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(CAMERA_CONFIG_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    CameraRegistry().configure("phone", CameraConfig.from_dict(CONFIG))
    assert list(tmp_path.rglob("*")) == []
    assert CameraRegistry().board("phone").config == CameraConfig()


def test_serve_keeps_the_configs_on_disk_by_default(monkeypatch):
    import uvicorn

    from board_reader_inference import cli

    # A private environment: monkeypatch cannot restore a key the code under test sets.
    env = dict(os.environ)
    env.pop(CAMERA_CONFIG_ENV, None)
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env[CAMERA_CONFIG_ENV] == cli.DEFAULT_CAMERA_CONFIG

    env[CAMERA_CONFIG_ENV] = ""
    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env[CAMERA_CONFIG_ENV] == "", "an operator's choice (memory only) outranks it"


def test_a_crop_saved_before_auto_frame_existed_stays_manual(config_file):
    config_file.parent.mkdir(parents=True)
    old = {k: v for k, v in CONFIG.items() if k != "auto"}
    config_file.write_text(json.dumps({"boards": {"board-1": old}, "cameras": {PHONE: "board-1"}}))
    registry = CameraRegistry()
    assert registry.board("board-1").config.auto is False
    phone = registry.connect_phone(PHONE)
    assert SendFullFrameMessage(0) not in phone.mailbox.take_nowait()
