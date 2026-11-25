"""The streams on disk, so a restarted node starts them again as the same cameras.

The list sits beside the camera config (``BR_CAMERA_CONFIG``), in ``streams.json``,
under the same rules: written whole as tmp + ``os.replace``, and on load taken in full
or ignored with a WARNING. A file of its own rather than a key in the camera config,
because the camera registry writes that file whole and knows nothing of streams.
Each stream's device id is kept here; its ``--state`` file is only how the edge
process is handed that id, written before every start (:func:`state_file`).
Unset, nothing is written.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, StrictFloat

from board_reader_core.control import DeviceId, NonNegative, StreamId, StreamUrl
from board_reader_core.wiremodel import Wire, parse, wire

log = logging.getLogger(__name__)

#: The streams' file, beside the camera config.
STREAMS_FILE = "streams.json"
#: The directory beside it that holds each stream's ``--state`` file.
STATE_DIR = "streams"


@wire
class StreamRecord(Wire):
    """One stream as the node keeps it: what to run, and as which camera.

    ``created`` is when it was added, in epoch seconds.
    """

    id: StreamId
    url: StreamUrl
    start_s: NonNegative | None
    device_id: DeviceId
    created: Annotated[StrictFloat, Field(ge=0)]


def streams_path(camera_config: Path) -> Path:
    return camera_config.with_name(STREAMS_FILE)


def state_file(directory: Path, stream_id: str) -> Path:
    """Where stream ``stream_id``'s device id is handed to its edge process."""
    return directory / STATE_DIR / f"{stream_id}.device-id"


def write_state_file(path: Path, device_id: str) -> None:
    """The edge's ``--state`` file: the device id on one line (what
    ``board_reader_edge.camera.link.load_device_id`` reads), tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(device_id + "\n")
    os.replace(tmp, path)


def _parse(raw: object) -> tuple[StreamRecord, ...]:
    if not isinstance(raw, dict) or set(raw) != {"streams"}:
        raise ValueError('expected {"streams": [<stream>, ...]}')
    records: tuple[StreamRecord, ...] = parse(tuple[StreamRecord, ...], raw["streams"])
    for key in ("id", "device_id"):
        values = [getattr(record, key) for record in records]
        if len(set(values)) != len(values):
            raise ValueError(f"two streams have one {key}")
    return records


def load_streams(path: Path) -> tuple[StreamRecord, ...]:
    """What :func:`save_streams` wrote; none if there is no file, or it does not parse
    in full (with a WARNING)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return ()
    except OSError as e:
        log.warning("ignoring the streams file %s: %s", path, e)
        return ()
    try:
        return _parse(json.loads(text))
    except ValueError as e:
        log.warning("ignoring the corrupt streams file %s: %s", path, e)
        return ()


def save_streams(path: Path, records: tuple[StreamRecord, ...]) -> None:
    """Write every stream to ``path`` at once: a temporary file, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"streams": [record.to_dict() for record in records]}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    os.replace(tmp, path)
