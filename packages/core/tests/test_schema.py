"""The JSON Schema export (``board_reader_core.schema``): what TypeScript is generated from."""

from __future__ import annotations

import json
import subprocess
import sys

from board_reader_core import control, protocol, wire
from board_reader_core.schema import json_schema
from board_reader_core.wiremodel import CHECK_KEYWORD


def _walk(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_the_document_names_every_message_and_is_stable():
    document = json_schema()
    defs = document["$defs"]
    for name in (
        "PositionEvent", "MotionUpdate", "ViewerMessage", "FrameHeader",
        "ReportMessage", "PhoneDown", "ConfigureMessage", "ControlDown",
        "CameraConfig", "CropRect", "CameraReport", "BoardDirection", "HealthMessage",
    ):
        assert name in defs
    assert list(defs) == sorted(defs)
    assert json.dumps(json_schema()) == json.dumps(document)
    refs = {n["$ref"] for n in _walk(document) if "$ref" in n}
    assert refs <= {f"#/$defs/{name}" for name in defs}


def test_each_socket_union_discriminates_on_the_messages_own_type():
    defs = json_schema()["$defs"]
    for union, members in (
        ("PhoneDown", (control.ConfigMessage, control.PreviewsMessage, control.RefusedMessage)),
        ("ControlDown", (control.StateMessage, control.PreviewMessage, control.FrameMessage,
                         control.HealthMessage, control.RefusedMessage)),
    ):
        discriminator = defs[union]["discriminator"]
        assert discriminator["propertyName"] == "type"
        assert discriminator["mapping"] == {
            m.TYPE: f"#/$defs/{m.__name__}" for m in members
        }
    # A position event is told from a motion update by having no `type` at all.
    assert "type" not in defs["PositionEvent"]["properties"]
    assert defs["MotionUpdate"]["properties"]["type"]["const"] == protocol.MOTION_TYPE


def test_it_states_what_the_node_sends_every_key_present():
    defs = json_schema()["$defs"]
    event = defs["PositionEvent"]
    # Defaulted keys are always sent; the optional blocks are left out when absent.
    assert {"stability", "status", "clock", "classes"} <= set(event["required"])
    assert not {"prior", "settled"} & set(event["required"])
    assert set(defs["CameraConfig"]["required"]) == {"crop", "device", "resolution", "streaming"}
    # The legacy corners parse in Python only; no edge sends them.
    assert "corners" not in defs["FrameHeader"]["properties"]


def test_the_rules_no_keyword_states_are_named_checks():
    document = json_schema()
    named = {n[CHECK_KEYWORD] for n in _walk(document) if CHECK_KEYWORD in n}
    assert named == {"isPlacement", "cropInside"}
    assert document["$defs"]["CropRect"][CHECK_KEYWORD] == "cropInside"


def test_the_module_prints_the_document():
    printed = subprocess.run(
        [sys.executable, "-W", "error", "-m", "board_reader_core.schema"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert json.loads(printed) == json_schema()


def test_the_python_edge_sends_a_header_the_schema_describes():
    import numpy as np

    header = wire.Frame(
        board_id="A1", seq=3, capture_ts=1.5, image=np.zeros((2, 2, 3), np.uint8)
    ).header()
    schema = json_schema()["$defs"]["FrameHeader"]
    assert set(header) == set(schema["required"]) == set(schema["properties"])
