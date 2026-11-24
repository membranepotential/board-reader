"""Generate the viewer's wire types and validators from core's JSON Schema.

``board_reader_core.schema.json_schema()`` is the one source of truth for every
message the browser sends or receives. This tool turns it into
``packages/viewer/src/lib/protocol/generated.ts``: one zod 4 schema per ``$defs``
entry, and its (deeply readonly) type under the same name. ``make protocol`` runs it;
``tests/test_protocol_ts.py`` fails when the committed file is not what it writes.

It translates exactly the JSON Schema subset pydantic writes for our models, and
**refuses any keyword it does not translate**, so a new constraint in core can
never be silently dropped on the TypeScript side. A rule no keyword states is named
by ``x-check`` and becomes a ``.refine`` with its twin in ``protocol/checks.ts``.

Why a tool of our own rather than a package: the one dedicated JSON Schema -> zod
code generator (``json-schema-to-zod``) was deprecated in March 2026 and its
repository archived, and zod's own ``z.fromJSONSchema()`` is experimental and
builds a schema at runtime only, with no static type. The subset here is small and
fixed by our own models; translating it exactly, deterministically and in Python
lets ``make test`` check for drift without a Node toolchain.

Usage::

    uv run python tools/gen_protocol.py          # write the file (make protocol)
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages" / "viewer" / "src" / "lib" / "protocol" / "generated.ts"

Schema = Mapping[str, Any]

#: The keyword that names a rule's TypeScript twin in ``protocol/checks.ts``.
CHECK_KEYWORD = "x-check"
#: Keywords that only document a schema; they translate to nothing.
_ANNOTATIONS = frozenset({"title", "description", "default"})
_NUMBER_BOUNDS = {
    "minimum": "min",
    "maximum": "max",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
}
_REF = re.compile(r"^#/\$defs/(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
#: A list whose one-line form is longer than this is written one item per line.
_FLAT_LIST = 40


class Unsupported(ValueError):
    """The schema uses something this generator does not translate."""


# ---- literals ------------------------------------------------------------------


def _str(value: str) -> str:
    """``value`` as a single-quoted TypeScript string (the viewer's style)."""
    inner = json.dumps(value)[1:-1].replace('\\"', '"').replace("'", "\\'")
    return f"'{inner}'"


def _number(value: object) -> str:
    """A JSON number as a TypeScript literal; ``1.0`` is written ``1``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Unsupported(f"not a number: {value!r}")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def _list(items: list[str], indent: str) -> str:
    """``[a, b]`` on one line when it is short, else one item per line."""
    flat = f"[{', '.join(items)}]"
    if len(flat) <= _FLAT_LIST and "\n" not in flat:
        return flat
    inner = indent + "  "
    return "[\n" + ",\n".join(inner + item for item in items) + f"\n{indent}]"


def _key(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) else _str(name)


def _ref_name(ref: str) -> str:
    match = _REF.match(ref)
    if match is None:
        raise Unsupported(f"only local $defs references are supported, not {ref!r}")
    return match["name"]


# ---- one schema node -> one zod expression ---------------------------------------


def _consume(schema: Schema, allowed: set[str]) -> None:
    unknown = set(schema) - allowed - _ANNOTATIONS
    if unknown:
        raise Unsupported(f"keyword(s) {sorted(unknown)} not translated in {json.dumps(schema)}")


def _bounds(schema: Schema, methods: Mapping[str, str]) -> str:
    """``.min(…)``-style calls for the bound keywords ``schema`` has, in ``methods`` order."""
    return "".join(
        f".{method}({_number(schema[key])})" for key, method in methods.items() if key in schema
    )


def translate(schema: Schema, indent: str = "") -> str:
    """One JSON Schema node as a zod expression, for a line indented by ``indent``."""
    check = schema.get(CHECK_KEYWORD)
    out = _node({k: v for k, v in schema.items() if k != CHECK_KEYWORD}, indent)
    if check is None:
        return out
    if not isinstance(check, str) or not re.fullmatch(r"[a-z][A-Za-z0-9]*", check):
        raise Unsupported(f"{CHECK_KEYWORD} must name a function, not {check!r}")
    return f"{out}.refine(checks.{check}, {{ message: {_str(f'fails {check}')} }})"


def _node(schema: Schema, indent: str) -> str:
    if "$ref" in schema:
        _consume(schema, {"$ref"})
        return _ref_name(schema["$ref"])
    if "anyOf" in schema:
        return _any_of(schema, indent)
    if "oneOf" in schema:
        return _one_of(schema, indent)
    if "const" in schema:
        _consume(schema, {"const", "type"})
        return _literal(schema["const"], indent)
    if "enum" in schema:
        _consume(schema, {"enum", "type"})
        values = schema["enum"]
        if not all(isinstance(v, str) for v in values):
            raise Unsupported(f"only string enums are supported: {values}")
        return f"z.enum({_list([_str(v) for v in values], indent)})"
    match schema.get("type"):
        case "string":
            return _string(schema)
        case "integer" | "number" as kind:
            _consume(schema, {"type", *_NUMBER_BOUNDS})
            base = "z.int()" if kind == "integer" else "z.number()"
            return base + _bounds(schema, _NUMBER_BOUNDS)
        case "boolean":
            _consume(schema, {"type"})
            return "z.boolean()"
        case "null":
            _consume(schema, {"type"})
            return "z.null()"
        case "array":
            return _array(schema, indent)
        case "object":
            return _object(schema, indent)
    raise Unsupported(f"no translation for {json.dumps(schema)}")


def _literal(value: Any, indent: str) -> str:
    """A zod schema accepting exactly the JSON value ``value``."""
    if isinstance(value, list):
        return f"z.tuple({_list([_literal(v, indent + '  ') for v in value], indent)})"
    if isinstance(value, str):
        return f"z.literal({_str(value)})"
    if isinstance(value, bool) or value is None:
        return f"z.literal({json.dumps(value)})"
    return f"z.literal({_number(value)})"


def _string(schema: Schema) -> str:
    _consume(schema, {"type", "minLength", "maxLength", "pattern", "contentEncoding"})
    encoding = schema.get("contentEncoding")
    if encoding not in (None, "base64"):
        raise Unsupported(f"contentEncoding {encoding!r}")
    out = "z.base64()" if encoding == "base64" else "z.string()"
    out += _bounds(schema, {"minLength": "min", "maxLength": "max"})
    if "pattern" in schema:
        out += ".regex(/" + schema["pattern"].replace("/", r"\/") + "/)"
    return out


def _array(schema: Schema, indent: str) -> str:
    _consume(schema, {"type", "items", "minItems", "maxItems"})
    if "items" not in schema:
        raise Unsupported(f"an array needs items: {json.dumps(schema)}")
    item = translate(schema["items"], indent)
    low, high = schema.get("minItems"), schema.get("maxItems")
    if low is not None and low == high:
        return f"fixed({item}, {_number(low)})"
    return f"z.array({item})" + _bounds(schema, {"minItems": "min", "maxItems": "max"})


def _object(schema: Schema, indent: str) -> str:
    _consume(schema, {"type", "properties", "required", "additionalProperties"})
    extra = schema.get("additionalProperties", True)
    properties: Mapping[str, Schema] = schema.get("properties", {})
    if not properties:
        if not isinstance(extra, Mapping):
            raise Unsupported(f"an object with no properties needs a value schema: {schema}")
        return f"z.record(z.string(), {translate(extra, indent)})"
    if extra is not False:
        raise Unsupported(f"an object with properties must forbid other keys: {schema}")
    required = set(schema.get("required", ()))
    if required - set(properties):
        raise Unsupported(f"required names no property: {sorted(required - set(properties))}")
    inner = indent + "  "
    lines = [
        f"{inner}{_key(name)}: {translate(prop, inner)}{'' if name in required else '.optional()'}"
        for name, prop in properties.items()
    ]
    return "z.strictObject({\n" + ",\n".join(lines) + f"\n{indent}}})"


def _any_of(schema: Schema, indent: str) -> str:
    _consume(schema, {"anyOf"})
    options = schema["anyOf"]
    others = [o for o in options if o != {"type": "null"}]
    if len(options) == 2 and len(others) == 1:
        return f"{translate(others[0], indent)}.nullable()"
    return f"z.union({_list([translate(o, indent + '  ') for o in options], indent)})"


def _one_of(schema: Schema, indent: str) -> str:
    _consume(schema, {"oneOf", "discriminator"})
    options = _list([translate(o, indent + "  ") for o in schema["oneOf"]], indent)
    discriminator = schema.get("discriminator")
    if discriminator is None:
        # oneOf: exactly one option may match, which is what z.xor checks.
        return f"z.xor({options})"
    if set(discriminator) - {"propertyName", "mapping"}:
        raise Unsupported(f"discriminator {discriminator}")
    return f"z.discriminatedUnion({_str(discriminator['propertyName'])}, {options})"


# ---- the module ------------------------------------------------------------------


def _refs(schema: Any) -> set[str]:
    if isinstance(schema, Mapping):
        found = {_ref_name(schema["$ref"])} if "$ref" in schema else set()
        return found.union(*(_refs(v) for v in schema.values()))
    if isinstance(schema, list):
        return set().union(*(_refs(v) for v in schema))
    return set()


def _ordered(defs: Mapping[str, Schema]) -> list[str]:
    """The names in ``defs``, each after every one it references; ties by name."""
    pending = {name: _refs(schema) for name, schema in defs.items()}
    for name, deps in pending.items():
        if deps - set(defs):
            raise Unsupported(f"{name} references undefined {sorted(deps - set(defs))}")
    order: list[str] = []
    while pending:
        ready = sorted(name for name, deps in pending.items() if deps <= set(order))
        if not ready:
            raise Unsupported(f"recursive definitions are not supported: {sorted(pending)}")
        order.append(ready[0])
        del pending[ready[0]]
    return order


def _doc(text: str | None) -> str:
    """A Python docstring as a JSDoc block, with reST roles and double backticks dropped."""
    if not text:
        return ""
    text = re.sub(r":[a-z]+:`~?([^`]+)`", r"`\1`", text).replace("``", "`").replace("*/", "*\\/")
    body = "\n".join(f" * {line}".rstrip() for line in text.strip().splitlines())
    return f"/**\n{body}\n */\n"


HEADER = """\
// GENERATED by tools/gen_protocol.py from board_reader_core's JSON Schema
// (python -m board_reader_core.schema). Do not edit: change the Python models in
// packages/core and regenerate with `make protocol`.
//
// One zod schema per message and block, and its readonly type under the same
// name. A rule the schema cannot state (`x-check`) is a refinement from
// ./checks.

import { z } from 'zod';

import * as checks from './checks';

/** `T` with every key and every array readonly: a value off the wire is never mutated. */
type Immutable<T> = T extends object ? { readonly [K in keyof T]: Immutable<T[K]> } : T;

/** `T` repeated `N` times, as a tuple type. */
type Tuple<T, N extends number, Acc extends T[] = []> = Acc['length'] extends N
  ? Acc
  : Tuple<T, N, [...Acc, T]>;

/** A JSON array of exactly `n` items: checked by its length, typed as a tuple. */
function fixed<T extends z.ZodType, N extends number>(item: T, n: N) {
  return z.array(item).length(n) as unknown as z.ZodType<
    Tuple<z.output<T>, N>,
    Tuple<z.input<T>, N>
  >;
}
"""


def render(document: Schema) -> str:
    """The TypeScript module for ``document`` (a ``board_reader_core.schema`` document)."""
    defs: Mapping[str, Schema] = document["$defs"]
    blocks = [
        f"{_doc(defs[name].get('description'))}"
        f"export const {name} = {translate(defs[name])};\n"
        f"export type {name} = Immutable<z.infer<typeof {name}>>;\n"
        for name in _ordered(defs)
    ]
    return HEADER + "\n" + "\n".join(blocks)


def generate() -> str:
    """The module for the Python models as they are now."""
    from board_reader_core.schema import json_schema

    return render(json_schema())


def main() -> None:
    text = generate()
    # Write only on a change: every build runs this, and rewriting an unchanged file
    # would reload a running `vite dev` for nothing.
    if OUT.exists() and OUT.read_text() == text:
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
