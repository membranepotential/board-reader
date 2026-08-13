"""Corner-quad parsing shared by every ``--corners``/``corners`` consumer.

A board's four corners are given in the same textual shape everywhere they
appear — a CLI flag, a wire-frame header, a label sidecar — always
``x0,y0,...,x3,y3`` in a8, h8, h1, a1 order. Parsing that string lives here
once so the edge CLI, the inference CLI, the gateway and the eval/labels
reader all agree on what counts as well-formed input.

Validating that the resulting quad actually *looks like a board* — convex,
big enough, not absurdly foreshortened — is a separate, edge-only concern
(:mod:`board_reader_edge.quadcheck`) that runs on top of this, not instead of
it: it needs the operator-facing warnings/errors that only the edge CLI
gives, so it stays there rather than folding into this shared parser.
"""

from __future__ import annotations

import numpy as np


def parse_corners(spec: str) -> np.ndarray:
    """``x0,y0,...,x3,y3`` (``;`` also accepted as a separator) -> a 4x2 float32 quad.

    Raises ``ValueError`` on anything malformed, including a non-``str`` spec —
    some callers read corners out of JSON, where the field may be ``null``.
    """
    if not isinstance(spec, str):
        raise ValueError("corners must be a comma-separated string")
    nums = [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]
    if len(nums) != 8:
        raise ValueError("needs 8 numbers: x0,y0,...,x3,y3")
    return np.array(nums, dtype=np.float32).reshape(4, 2)
