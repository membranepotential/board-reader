"""Small order statistics shared by the evidence and timing scores."""

from __future__ import annotations


def quantile(values: list[float], q: float) -> float:
    """The ``q``-quantile by nearest rank. Small samples, no numpy, no interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[i]
