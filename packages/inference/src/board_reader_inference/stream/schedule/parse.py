"""Policy spec strings: the ``--policy`` / ``BR_SCHEDULE`` grammar ``name[:key=value,...]``."""

from __future__ import annotations

from collections.abc import Mapping

from board_reader_inference.stream.schedule.policies import (
    BurstTriggered,
    EveryFrame,
    WhenStill,
)
from board_reader_inference.stream.schedule.seam import FramePolicy

#: Every policy the ``--policy`` flag accepts, by spec name.
POLICIES: Mapping[str, type] = {
    "every-frame": EveryFrame,
    "burst-end": BurstTriggered,
    "when-still": WhenStill,
}


def parse_policy(spec: str) -> FramePolicy:
    """Build a policy from ``name[:key=value,…]`` — the ``--policy`` surface.

    Deliberately tiny: this is an experiment knob, and every value it accepts is a
    number naming a field of the policy it configures, so there is nothing to
    invent. Unknown names and unknown keys raise with the list of what exists,
    because the alternative — silently ignoring a typo'd parameter — is a whole
    GPU sweep measuring the defaults under another policy's name.

    Round-trips: ``parse_policy(p.name) == p`` for every policy here.
    """
    head, _, rest = spec.strip().partition(":")
    head = head.strip()
    cls = POLICIES.get(head)
    if cls is None:
        raise ValueError(f"unknown policy {head!r}; known policies: {', '.join(POLICIES)}")
    params: dict[str, float | int] = {}
    for item in (p.strip() for p in rest.split(",")):
        if not item:
            continue
        key, sep, raw = item.partition("=")
        key = key.strip()
        coerce = cls.PARAMS.get(key)
        if not sep or coerce is None:
            known = ", ".join(cls.PARAMS) or "(none — it takes no parameters)"
            raise ValueError(f"policy {head!r} has no parameter {key!r}; it takes: {known}")
        try:
            params[key] = coerce(raw.strip())
        except ValueError as e:
            raise ValueError(
                f"policy {head!r}: {key}={raw.strip()!r} is not a {coerce.__name__}"
            ) from e
    return cls(**params)  # type: ignore[return-value]
