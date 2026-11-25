"""The camera configs on disk, so a node restarted mid-game gives each phone its board id back.

With ``BR_CAMERA_CONFIG=<path>`` set (``serve`` sets it), every change is written to
that JSON file and read back at startup, so a restarted node gives each phone its
board id back and pushes the same crop, lens, resolution and streaming switch to it
when it reconnects, and settles each board under the same rules. Unset, everything is
in memory. Reports, previews and the board direction are always in memory: the phone
and the recognizer send them again.
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
    """What the node keeps across restarts: each board's config, each device's board
    and each named device's name."""

    boards: Mapping[str, CameraConfig] = field(default_factory=dict)
    cameras: Mapping[str, str] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)


def _parse_saved(raw: object) -> Saved:
    if (
        not isinstance(raw, dict)
        or "boards" not in raw
        or not set(raw) <= {"boards", "cameras", "names"}
    ):
        raise ValueError(
            'expected {"boards": {<board id>: <config>}, "cameras": {<device id>: <board id>}, '
            '"names": {<device id>: <name>}}'
        )
    if not isinstance(raw["boards"], dict):
        raise ValueError('"boards" is not an object')
    boards = {
        str(board_id): CameraConfig.from_dict(_manual_if_cropped(config))
        for board_id, config in raw["boards"].items()
    }
    # A file from before the cameras were known by device has no "cameras": none known
    # yet; one from before the names has no "names": none named yet.
    for key in ("cameras", "names"):
        if not isinstance(raw.get(key, {}), dict):
            raise ValueError(f'"{key}" is not an object')
    cameras: dict[str, str] = parse(dict[DeviceId, CameraBoardId], raw.get("cameras", {}))
    if len(set(cameras.values())) != len(cameras):
        raise ValueError("two cameras hold one board id")
    names: dict[str, str] = parse(dict[DeviceId, CameraName], raw.get("names", {}))
    if not set(names) <= set(cameras):
        raise ValueError("a name for a camera that is not known")
    if len(set(names.values())) != len(names):
        raise ValueError("two cameras have one name")
    return Saved(boards, cameras, names)


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
    known), the same as a node that never had a file.
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
        "cameras": dict(saved.cameras),
        "names": dict(saved.names),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    os.replace(tmp, path)


def config_path() -> Path | None:
    raw = os.environ.get(CAMERA_CONFIG_ENV)
    return Path(raw) if raw else None
