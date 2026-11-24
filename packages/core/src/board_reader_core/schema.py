"""The JSON Schema of every message that crosses to TypeScript (the browser).

The viewer's TypeScript types and validators are generated from this document, so
the Python models (:mod:`.protocol`, :mod:`.control`, :mod:`.wire`, built with
:mod:`.wiremodel`) are the single source of truth for both ends.
``python -m board_reader_core.schema`` prints it.

What the document holds:

* ``$defs`` — one entry per message and per block it contains, named as the
  Python class is, plus the unions each socket carries. Sorted by name, so a given
  model gives a byte-identical document.
* Serialization mode with defaults required: each entry says what the node
  *sends*, every key present. A key with a Python default may be left out when the
  node *parses* a message, so a sender that writes every key is always accepted.
* ``x-check`` — names a rule no JSON Schema keyword states (``isPlacement``: FEN
  field 1, eight ranks of eight squares; ``cropInside``: ``x + w <= 1`` and
  ``y + h <= 1``). The Python validator is the definition.
* Binary messages are not in it: on ``/ingest`` each ``FrameHeader`` (text) is
  followed by its JPEG (binary); the phone's previews go up as bare binary JPEGs.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from board_reader_core import control, protocol, wire
from board_reader_core.wiremodel import CHECK_KEYWORD, adapter

#: The document's dialect: what pydantic writes.
DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _roots() -> dict[str, Any]:
    """Every message the browser sends or receives, by the name its ``$defs`` uses."""
    return {
        # WS /viewer (DESIGN §5)
        "PositionEvent": protocol.PositionEvent,
        "MotionUpdate": protocol.MotionUpdate,
        "ViewerMessage": protocol.ViewerMessage,
        # WS /ingest: each header, then its JPEG as the next (binary) message
        "FrameHeader": wire.FrameHeader,
        # WS /boards/{id}/camera: up (text; previews go up as binary) and down
        "ReportMessage": control.ReportMessage,
        "PhoneDown": control.PhoneDown,
        # WS /boards/{id}/control: up and down
        "ConfigureMessage": control.ConfigureMessage,
        "ControlDown": control.ControlDown,
    }


def json_schema() -> dict[str, Any]:
    """The document (see the module note).

    The unions (``ViewerMessage``, ``PhoneDown``, ``ControlDown``) are ``$defs`` of
    their own. The socket unions discriminate on ``type``; a position event has no
    ``type``, and is told from a motion update by its absence.
    """
    keyed, document = TypeAdapter.json_schemas(
        [(name, "serialization", adapter(tp)) for name, tp in _roots().items()],
        ref_template="#/$defs/{model}",
    )
    defs = dict(document["$defs"])
    for (name, _), schema in keyed.items():
        if schema.get("$ref") != f"#/$defs/{name}":
            defs[name] = schema
    return {
        "$schema": DIALECT,
        "title": "board-reader wire messages",
        "description": (
            "Generated from board_reader_core (python -m board_reader_core.schema). "
            f"A rule no keyword states is named by '{CHECK_KEYWORD}'."
        ),
        "$defs": {name: defs[name] for name in sorted(defs)},
    }


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
