"""Validation of policy intervals: the checks that keep the read floor from being switched off."""

from __future__ import annotations

import math


def finite_positive(name: str, value: float, because: str) -> None:
    """Reject the two spellings of "no limit" that would switch a floor off.

    ``inf`` passes a bare ``> 0`` check and turns a keepalive into "never" — one
    read in the 60 recorded seconds of the dead-meter incident. NaN passes it too
    and is worse: every comparison against NaN is ``False``, so the rate limit
    inverts and the policy reads *every* frame. Both are the incident this package
    exists to prevent, arriving through a config string instead of a broken
    sensor, so an interval is a finite positive number of seconds and nothing else.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0 and finite, got {value!r}: {because}")


#: Why ``keepalive_s`` has no "off" spelling.
FLOOR_REASON = (
    "the keepalive is a floor under the read rate (D3), and a policy without one "
    "goes blind the moment the motion meter does"
)
#: Why a spacing of zero is not a thing either.
SPACING_REASON = (
    "it is the interval between two reads, and an interval of zero (or of none at "
    "all) is a burst of estimates at stream rate, which is what the floor exists "
    "to bound"
)


def under_the_floor(name: str, spacing_s: float, keepalive_s: float) -> None:
    """A tighter interval may govern over the floor; a looser one may not.

    ``spacing_s`` is the *tight* interval — the beat of a burst-end run, or the
    calm-frame rate — and the whole promise of D3 is that no configuration makes
    the node read more slowly than ``keepalive_s``. A spacing above the floor
    quietly breaks that promise (``when-still`` with ``keepalive_s=3,
    spacing_s=10`` leaves 10 s gaps under a claimed 3 s floor), and it does so in
    the one direction nobody watches: fewer reads, which looks like a quiet board.
    """
    if spacing_s > keepalive_s:
        raise ValueError(
            f"{name}={spacing_s!r} is looser than keepalive_s={keepalive_s!r}: the "
            "spacing is the tight interval and may only ever read MORE often than "
            "the floor, never less (D3). Raise keepalive_s if longer gaps are "
            "really what you want."
        )
