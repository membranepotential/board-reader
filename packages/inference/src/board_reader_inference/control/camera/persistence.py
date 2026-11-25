"""The camera configs on disk, so a node restarted mid-game knows its cameras again.

With ``BR_CAMERA_CONFIG=<path>`` set (``serve`` sets it), every change is written to
that JSON file and read back at startup, so a restarted node knows each camera's name
and pushes the same crop, lens, resolution and streaming switch to it when it
reconnects, and settles each board under the same rules. Unset, everything is in
memory. Reports, previews and the board direction are always in memory: the phone and
the recognizer send them again.

The file is ``{"boards": {<board id>: <config>}, "names": {<device id>: <name>}}``:
every known camera has a name, so ``names`` lists the cameras, and a camera's board id
is its device id. A file written while the node still handed out ``board-N`` board ids
also has ``"cameras": {<device id>: <board id>}``; it loads as if written now: each
config moves to its camera's device id, and a camera without a name is named after
its old board id, so the operator still reads the same label.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from board_reader_core.control import CameraBoardId, CameraConfig, CameraName, DeviceId
from board_reader_core.wiremodel import parse

log = logging.getLogger(__name__)

#: Where each board's config is kept across restarts; unset or empty, nowhere.
CAMERA_CONFIG_ENV = "BR_CAMERA_CONFIG"


@dataclass(frozen=True, slots=True)
class Saved:
    """What the node keeps across restarts: each board's config and each known
    camera's name, by device id (which is also the camera's board id)."""

    boards: Mapping[str, CameraConfig] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)


def _parse_saved(raw: object) -> Saved:
    if (
        not isinstance(raw, dict)
        or "boards" not in raw
        or not set(raw) <= {"boards", "cameras", "names"}
    ):
        raise ValueError(
            'expected {"boards": {<board id>: <config>}, "names": {<device id>: <name>}}'
        )
    # A file from before the cameras were known by device has only "boards": no
    # camera known yet.
    for key in ("boards", "cameras", "names"):
        if not isinstance(raw.get(key, {}), dict):
            raise ValueError(f'"{key}" is not an object')
    boards = {
        str(board_id): CameraConfig.from_dict(_manual_if_cropped(config))
        for board_id, config in raw["boards"].items()
    }
    names: dict[str, str] = parse(dict[DeviceId, CameraName], raw.get("names", {}))
    if "cameras" in raw:
        boards, names = _from_board_numbers(boards, raw["cameras"], names)
    if len(set(names.values())) != len(names):
        raise ValueError("two cameras have one name")
    return Saved(boards, names)


def _from_board_numbers(
    boards: Mapping[str, CameraConfig], raw_cameras: object, names: Mapping[str, str]
) -> tuple[dict[str, CameraConfig], dict[str, str]]:
    """A file from when the node assigned each device a ``board-N`` board id, as the
    configs and names it means now: a config moves from the old board id to its
    camera's device id, and each camera is named by its old name, else its old board
    id. A config no camera held keeps its id (the registry drops it)."""
    cameras: dict[str, str] = parse(dict[DeviceId, CameraBoardId], raw_cameras)
    if len(set(cameras.values())) != len(cameras):
        raise ValueError("two cameras hold one board id")
    if not set(names) <= set(cameras):
        raise ValueError("a name for a camera that is not known")
    device_of = {board_id: device for device, board_id in cameras.items()}
    moved = {device_of.get(board_id, board_id): config for board_id, config in boards.items()}
    named = {device: names.get(device, board_id) for device, board_id in cameras.items()}
    if cameras:
        log.info(
            "camera config: %d camera(s) now under their device ids, named as before",
            len(cameras),
        )
    return moved, named


def _manual_if_cropped(config: object) -> object:
    """A config saved before Auto-frame existed (no ``"auto"``) with a crop: that crop
    was drawn by hand, so it loads with Auto-frame off rather than being replaced."""
    if isinstance(config, dict) and "auto" not in config and config.get("crop") is not None:
        return {**config, "auto": False}
    return config


def load_state(path: Path) -> Saved:
    """What ``save_state`` wrote; nothing saved if there is no file.

    A file that does not parse in full is ignored with a WARNING, never half
    applied: the node then starts with the defaults (streaming off, no camera
    known), the same as a node that never had a file. A file from before the board id
    was the device id loads migrated (see the module docstring).
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Saved()
    except OSError as e:
        log.warning("ignoring the camera config %s: %s", path, e)
        return Saved()
    try:
        return _parse_saved(json.loads(text))
    except ValueError as e:
        log.warning("ignoring the corrupt camera config %s: %s", path, e)
        return Saved()


def save_state(path: Path, saved: Saved) -> None:
    """Write everything to ``path`` at once: a temporary file, then a rename.

    A crash mid-write leaves the old file whole, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "boards": {board_id: config.to_dict() for board_id, config in saved.boards.items()},
        "names": dict(saved.names),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    os.replace(tmp, path)


def config_path() -> Path | None:
    raw = os.environ.get(CAMERA_CONFIG_ENV)
    return Path(raw) if raw else None
