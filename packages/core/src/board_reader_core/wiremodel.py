"""How every message that crosses a wire is modelled: strict pydantic dataclasses.

The wire contracts (:mod:`.protocol`, :mod:`.control`, :mod:`.wire`) are the one
source of truth for the JSON both ends speak. Their models validate what they are
built from and what they parse, and their JSON Schema (:mod:`.schema`) is what the
viewer's TypeScript types and validators are generated from. So a rule written in a
model holds on both sides.

The models are dataclasses (pydantic's), not ``BaseModel``\\ s, because the code that
uses them builds them positionally, compares them, and passes them to
``dataclasses.replace``; pydantic validates every one of those constructions.

What a model may say about a value:

* **Types are strict.** Scalars are pydantic's ``Strict*`` types, so ``"1"`` is not
  an int and ``1`` is not a bool, as in the generated zod. Containers stay lax, so
  a JSON object becomes a nested model and a JSON array a tuple.
* **Numbers are finite** (``allow_inf_nan=False``): JSON cannot carry NaN or
  infinity, and ``json.dumps`` would write them anyway.
* **Unknown keys are refused** (``extra="forbid"``): a misspelt ``"streming": true``
  silently doing nothing is exactly the bug a strict boundary exists to catch.
* **A rule the schema cannot state** (two fields together, a FEN rank's length) is a
  *check*: a pydantic validator plus an ``x-check`` keyword naming it, so the
  TypeScript side can attach its twin.

The JSON Schema is taken in serialization mode with defaults required: it states
what the node *sends*, every key present. A sender that writes every key is
therefore always accepted here, where a defaulted key may also be left out.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Self, dataclass_transform

from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic.dataclasses import dataclass

#: The JSON Schema keyword that names a check's TypeScript twin.
CHECK_KEYWORD = "x-check"

#: The largest integer a JavaScript number holds exactly (``Number.MAX_SAFE_INTEGER``),
#: which zod's ``z.int()`` enforces; so every wire integer is bounded by it here too.
SAFE_INT_MAX = 2**53 - 1

#: A count or a sequence number.
Natural = Annotated[StrictInt, Field(ge=0, le=SAFE_INT_MAX)]
#: A size in pixels.
Positive = Annotated[StrictInt, Field(gt=0, le=SAFE_INT_MAX)]
#: A value on the unit interval: a probability, a stability.
Unit = Annotated[StrictFloat, Field(ge=0, le=1)]
#: Which board a message is about. Any non-empty name the edge chose.
BoardId = Annotated[StrictStr, Field(min_length=1)]

WIRE_CONFIG = ConfigDict(
    extra="forbid",
    allow_inf_nan=False,
    validate_by_name=True,
    validate_by_alias=True,
    serialize_by_alias=True,
    validate_assignment=True,
    json_schema_serialization_defaults_required=True,
)


def checked(name: str) -> dict[str, Any]:
    """The JSON Schema extra that names check ``name`` (see the module note)."""
    return {CHECK_KEYWORD: name}


@functools.cache
def adapter(tp: Any) -> TypeAdapter[Any]:
    """The (cached) pydantic adapter for a wire type: a model or a union of them."""
    return TypeAdapter(tp)


def reason(error: ValidationError) -> str:
    """A validation error in one line, each problem prefixed with where it is."""
    parts = []
    for problem in error.errors(include_url=False):
        where = ".".join(str(part) for part in problem["loc"])
        # A ValueError raised by one of our validators reads as its own words.
        said = problem.get("ctx", {}).get("error")
        msg = str(said) if problem["type"] == "value_error" and said else problem["msg"]
        parts.append(f"{where}: {msg}" if where else msg)
    return "; ".join(parts)


def parse[T](tp: type[T] | Any, raw: object) -> T:
    """``raw`` (decoded JSON, or a JSON string) as ``tp``; ``ValueError`` says why not.

    Raises a plain ``ValueError`` carrying :func:`reason`, not pydantic's
    multi-line ``ValidationError``, because the reason travels to an operator (a
    ``refused`` message, an ingest refusal) and must read as one line.
    """
    try:
        if isinstance(raw, str | bytes):
            return adapter(tp).validate_json(raw)
        return adapter(tp).validate_python(raw)
    except ValidationError as e:
        raise ValueError(reason(e)) from None


def dump(value: object, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """``value`` as the JSON object it travels as."""
    return adapter(type(value)).dump_python(value, mode="json", context=context)


class Wire:
    """The (de)serialisation every wire model shares."""

    __slots__ = ()

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        return parse(cls, raw)

    @classmethod
    def from_json(cls, text: str | bytes) -> Self:
        return parse(cls, text)

    def to_dict(self) -> dict[str, Any]:
        return dump(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def wire[T](
    cls: type[T] | None = None, *, frozen: bool = True, kw_only: bool = False, **extra: Any
) -> Any:
    """``cls`` as a wire model: a pydantic dataclass under :data:`WIRE_CONFIG`.

    ``extra`` adds to the config (a check's ``json_schema_extra``).
    """

    def build(target: type[T]) -> type[T]:
        config = ConfigDict(**{**WIRE_CONFIG, **extra})  # type: ignore[typeddict-item]
        return dataclass(frozen=frozen, slots=True, kw_only=kw_only, config=config)(target)

    return build if cls is None else build(cls)


def tag(kind: str) -> Any:
    """The ``type`` field of a message: fixed, defaulted, and keyword-only.

    Keyword-only so the message's own fields keep their positions
    (``ConfigMessage(config)``).
    """
    return dataclasses.field(default=kind, kw_only=True)


class Message(Wire):
    """A wire model with a ``type`` field; :data:`TYPE` is that field's value.

    ``TYPE`` is what a socket's outbox keys its latest-per-type slots by.
    """

    __slots__ = ()
    TYPE: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _names_its_type(cls, data: Any) -> Any:
        # Built in Python the type is implied; read off a wire it must be said.
        if isinstance(data, dict) and "type" not in data:
            raise ValueError(f"{cls.__name__} needs its type")
        return data


@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def message[M: Message](cls: type[M]) -> type[M]:
    """``cls`` as a frozen wire model whose ``TYPE`` is its ``type`` default."""
    built: type[M] = wire(cls)
    default = built.__dataclass_fields__["type"].default  # type: ignore[attr-defined]
    if not isinstance(default, str):
        raise TypeError(f"{cls.__name__}.type needs a string default (use tag())")
    built.TYPE = default
    return built


def rounded(ndigits: int) -> Callable[[float], float]:
    """A float validator that rounds to ``ndigits``, as the value is built.

    A wire value carries no more precision than its consumer reads, and the
    rounding keeps lines short. It runs on construction rather than as a
    serializer, so the JSON Schema keeps the field's range.
    """
    return lambda value: round(value, ndigits)


def absent(value: object) -> bool:
    """``exclude_if`` for an optional block: left out of the JSON when ``None``."""
    return value is None
