"""The TypeScript wire types stay what the Python models say (``tools/gen_protocol.py``).

``packages/viewer/src/lib/protocol/generated.ts`` is generated from
``board_reader_core.schema``; these tests fail when it is stale, when the schema
names a check the TypeScript side lacks, and when pydantic judges a shared sample
differently from its label. ``generated.test.ts`` runs the same samples through the
generated validators, so both ends accept and refuse alike.

Python-side, so ``make test`` catches a model change without a Node toolchain.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from board_reader_core import control, protocol, wire
from board_reader_core.schema import json_schema
from board_reader_core.wiremodel import CHECK_KEYWORD, parse

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "gen_protocol.py"
_TS = _ROOT / "packages" / "viewer" / "src" / "lib" / "protocol"


def _load():
    spec = importlib.util.spec_from_file_location("gen_protocol", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load()


def test_the_generated_module_is_current() -> None:
    assert gen.OUT == _TS / "generated.ts"
    assert gen.OUT.read_text() == gen.generate(), (
        "protocol/generated.ts is stale for the Python models: run `make protocol`"
    )


def _checks(node: object) -> set[str]:
    if isinstance(node, dict):
        found = {node[CHECK_KEYWORD]} if CHECK_KEYWORD in node else set()
        return found.union(*(_checks(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_checks(v) for v in node))
    return set()


def test_every_check_the_schema_names_has_a_typescript_twin() -> None:
    named = _checks(json_schema())
    exported = set(re.findall(r"^export function (\w+)", (_TS / "checks.ts").read_text(), re.M))
    assert named, "the schema names no check; this test would prove nothing"
    assert named <= exported, f"write {sorted(named - exported)} in protocol/checks.ts"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "format": "email"},  # a keyword it does not translate
        {"type": "object", "properties": {"a": {"type": "string"}}},  # extra keys allowed
        {"$ref": "#/$defs/Missing"},
    ],
)
def test_the_generator_refuses_what_it_cannot_translate_exactly(schema: dict) -> None:
    with pytest.raises(gen.Unsupported):
        gen.render({"$defs": {"X": schema}})


def _samples() -> list[dict]:
    return json.loads((_TS / "samples.json").read_text())


def _model(name: str) -> object:
    for module in (protocol, control, wire):
        if hasattr(module, name):
            return getattr(module, name)
    raise LookupError(f"no wire type {name} in board_reader_core")


@pytest.mark.parametrize("sample", _samples(), ids=lambda s: f"{s['message']}: {s['note']}")
def test_pydantic_judges_each_shared_sample_as_labelled(sample: dict) -> None:
    try:
        parse(_model(sample["message"]), sample["value"])
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is sample["valid"]
