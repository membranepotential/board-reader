"""The configs on disk: a restarted node gives each phone its name and config back."""

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
    registry.configure(PHONE, config)
    assert config_file.exists()
    assert not list(config_file.parent.glob(".*.tmp")), "the temporary file was renamed"
    assert json.loads(config_file.read_text()) == {
        "boards": {PHONE: CameraConfig.from_dict(CONFIG).to_dict()},
        "names": {OTHER: "board-1", PHONE: "left"},
    }

    heard: list[NamesMessage] = []
    restarted = CameraRegistry(on_names=heard.append)
    phone = restarted.connect_phone(PHONE)
    assert phone.mailbox.take_nowait()[:2] == [AssignedMessage(PHONE, "left"), ConfigMessage(config)]
    assert restarted.name_of(OTHER) == "board-1"
    assert heard == [NamesMessage(names=(BoardName(OTHER, "board-1"), BoardName(PHONE, "left")))]


def test_a_corrupt_config_file_is_ignored_with_a_warning(config_file, caplog):
    config_file.parent.mkdir(parents=True)
    bad = {"boards": {"phone": {**CONFIG, "resolution": "8k"}}}
    one_name = {"boards": {}, "names": {PHONE: "left", OTHER: "left"}}
    bad_device = {"boards": {}, "names": {"-starts-with-a-dash": "left"}}
    # Files from when the node handed out board-N board ids.
    one_board = {"boards": {}, "cameras": {PHONE: "left", OTHER: "left"}}
    stray_name = {"boards": {}, "cameras": {PHONE: "left"}, "names": {OTHER: "garden"}}
    # The old name of one camera is the old board id the other is named after now.
    clash = {"boards": {}, "cameras": {PHONE: "board-1", OTHER: "board-2"}, "names": {PHONE: "board-2"}}
    for raw in ("{not json", bad, {"phone": CONFIG}, one_name, bad_device, one_board, stray_name, clash):
        config_file.write_text(raw if isinstance(raw, str) else json.dumps(raw))
        with caplog.at_level(logging.WARNING):
            registry = CameraRegistry()
        assert registry.board("phone").config == CameraConfig()
        assert registry.name_of(PHONE) is None
        assert "corrupt camera config" in caplog.text, raw
        caplog.clear()

    # The next change replaces the corrupt file with a whole one.
    registry.connect_phone(PHONE)
    registry.configure(PHONE, CameraConfig(streaming=True))
    assert CameraRegistry().board(PHONE).config == CameraConfig(streaming=True)


def test_a_file_from_the_board_numbers_loads_under_device_ids_and_old_labels(config_file, caplog):
    """A file from when the node assigned ``board-N`` board ids: each camera's board id
    is its device id now, its config moves with it, and it keeps the label the
    operator read (its name, else its old board id). A config no camera held stays
    dropped."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps(
            {
                "boards": {"board-2": CONFIG, "board-9": CONFIG},
                "cameras": {PHONE: "board-1", OTHER: "board-2"},
                "names": {PHONE: "kitchen"},
            }
        )
    )
    with caplog.at_level(logging.INFO):
        registry = CameraRegistry()
    assert [r.getMessage() for r in caplog.records] == [
        "camera config: 2 camera(s) now under their device ids, named as before",
        "dropping the saved config of board-9: no camera holds it",
    ]
    assert registry.names() == NamesMessage(
        names=(BoardName(OTHER, "board-2"), BoardName(PHONE, "kitchen"))
    )
    assert registry.connect_phone(OTHER).mailbox.take_nowait()[:2] == [
        AssignedMessage(OTHER, "board-2"), ConfigMessage(CameraConfig.from_dict(CONFIG)),
    ]
    # The next new camera takes board-1, which no camera is called any more.
    assert registry.connect_phone("third-phone-id").mailbox.take_nowait()[0].name == "board-1"
    # Written back in today's shape.
    assert set(json.loads(config_file.read_text())) == {"boards", "names"}


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
    assert phone.mailbox.take_nowait()[:2] == [AssignedMessage(PHONE, "board-1"), ConfigMessage(CameraConfig())]


def test_a_config_saved_before_the_rules_existed_loads_as_standard(config_file):
    """A config file from before the switch has no ``rules`` key: still standard."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"boards": {PHONE: CONFIG}, "names": {PHONE: "board-1"}}))
    registry = CameraRegistry()
    assert registry.board(PHONE).config == CameraConfig.from_dict(CONFIG)
    assert registry.rules(PHONE) == "standard"


def test_the_rules_are_per_board_and_survive_a_restart(config_file):
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.configure(PHONE, CameraConfig(rules="free"))
    restarted = CameraRegistry()
    assert restarted.rules(PHONE) == "free"
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
    assert registry.board(PHONE).config.auto is False
    phone = registry.connect_phone(PHONE)
    assert SendFullFrameMessage(0) not in phone.mailbox.take_nowait()
