"""A policy's canonical spec string, and the directory name made from it."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import fields

from board_reader_inference.stream.schedule.seam import FramePolicy


def _number(value: float, kind: type) -> str:
    """A parameter value rendered so that ``parse_policy`` reads it back exactly.

    ``%g`` — what this used to use — is a *display* format and loses on both
    sides: it renders ``burst_end_reads=1000000`` as ``1e+06``, which the int
    coercion in ``parse_policy`` then refuses outright, and it rounds floats
    to six significant digits, so two policies differing past that share one name
    and one replay directory. Ints print as ints; floats print at ``repr``'s
    shortest round-tripping precision, with a bare ``.0`` trimmed because
    ``keepalive_s=5`` is how anyone would write it and ``float("5")`` is the same
    number.
    """
    if kind is int:
        return str(int(value))
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def canonical_spec(policy: object, name: str, params: Mapping[str, type]) -> str:
    """``name:key=value,…`` for the parameters that differ from the defaults.

    Only the differences, so the common case reads as the bare policy name and a
    manifest is not a wall of restated defaults — and the defaults themselves are
    free to be re-tuned without every old spec string silently meaning something
    new, because an old spec that *pinned* a value still says so.
    """
    defaults = {f.name: f.default for f in fields(policy) if f.init}  # type: ignore[arg-type]
    parts = [
        f"{key}={_number(getattr(policy, key), kind)}"
        for key, kind in params.items()
        if getattr(policy, key) != defaults[key]
    ]
    return f"{name}:{','.join(parts)}" if parts else name


def policy_slug(policy: FramePolicy) -> str:
    """A filesystem-safe rendering of a policy's spec, for a replay directory name.

    Lossless enough to read back at a glance (``burst-end-keepalive_s5``), which is
    the point: a sweep leaves a dozen sibling directories and their names are the
    only index anyone reads.
    """
    return re.sub(r"[^0-9A-Za-z._]+", "-", policy.name.replace("=", "")).strip("-")
