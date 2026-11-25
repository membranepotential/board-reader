"""Recomputing the signal the policies read: the inference-side port of the edge's motion meter."""

from __future__ import annotations

import cv2
import numpy as np

from board_reader_core.recording import FrameEntry

#: Meter constants, mirrored from :mod:`board_reader_edge.motion` (itself mirrored
#: in ``packages/viewer/src/lib/edge/motion.ts``). The three meters are one
#: algorithm in three places on purpose: the edge/inference split (DESIGN §4,
#: pinned by ``tests/test_layer_separation.py``) forbids this package from
#: importing the edge's, and the browser's is in another language entirely. Any
#: change here is a change in all three — they are compared by their numbers.
MOTION_DOWNSCALE = 32
MOTION_GAIN = 1.5
MOTION_ATTACK = 0.7
MOTION_RELEASE = 0.4
MOTION_TOLERANCE = 2.0
MOTION_FLOOR_UP = 0.02
MOTION_FLOOR_DOWN = 0.08
MOTION_NOISE_FLOOR = 0.0015
MOTION_FLOOR_NEAR = 4.0
MOTION_FLOOR_FAR_DIVISOR = 5.0


class MotionMeter:
    """Mean absolute inter-frame difference of a tiny thumbnail -> stability [0,1].

    A port, not a variant: same thumbnail, same adaptive floor, same asymmetric
    smoother, same constants as :class:`board_reader_edge.motion.MotionMeter`. See
    :data:`MOTION_DOWNSCALE` for why it is copied rather than imported.

    The first frame scores 1.0 — there is nothing to difference against, and an
    unknown board reports at rest (D3: a hint, never a gate).
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._floor: float | None = None
        self._stability = 1.0

    @property
    def stability(self) -> float:
        """The current smoothed stability, without consuming a frame."""
        return self._stability

    def update(self, image: np.ndarray) -> float:
        """Fold one decoded frame in and return the stability to stamp on it."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        thumb = (
            cv2.resize(
                gray,
                (MOTION_DOWNSCALE, MOTION_DOWNSCALE),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            / 255.0
        )
        prev, self._prev = self._prev, thumb
        if prev is None:
            self._stability = 1.0
            return self._stability
        diff = float(np.abs(thumb - prev).mean())
        if self._floor is None:
            self._floor = MOTION_NOISE_FLOOR
        if diff < self._floor:
            step = MOTION_FLOOR_DOWN
        elif diff / self._floor <= MOTION_FLOOR_NEAR:
            step = MOTION_FLOOR_UP
        else:
            step = MOTION_FLOOR_UP / MOTION_FLOOR_FAR_DIVISOR
        self._floor = max(MOTION_NOISE_FLOOR, self._floor + step * (diff - self._floor))
        instant = float(
            np.exp(-MOTION_GAIN * max(diff / self._floor - MOTION_TOLERANCE, 0.0))
        )
        weight = MOTION_ATTACK if instant < self._stability else MOTION_RELEASE
        self._stability = float(
            np.clip((1.0 - weight) * self._stability + weight * instant, 0.0, 1.0)
        )
        return self._stability


#: Header flag marking a stability the meter recomputed rather than the camera
#: stamped. Written into the replay's ``frames.ndjson``, because a benchmark whose
#: index silently claims a number no camera ever sent is a trap for whoever reads
#: it next.
RESTAMPED_KEY = "stability_restamped"


def restamped(entry: FrameEntry, meter: MotionMeter, image: np.ndarray) -> FrameEntry:
    """``entry`` with the stability today's meter scores for ``image``.

    **Why this exists.** Policies read the header's ``stability``, and on every
    take recorded before the 2026-08-27 evening meter fix that field is a *broken
    sensor's* output: the old meter scored an absolute pixel difference, the
    deployment phone's resting difference measured about six times the constant it
    assumed, and the field sat under 0.35 from the first frame of a session to the
    last. Scheduled against that, every policy would be measuring the bug rather
    than itself. Re-metering the recorded JPEGs in seq order produces the number a
    phone streaming that same scene *today* would send.

    The caller owns the meter and must feed it **every** frame of the session, in
    order, whether or not the policy goes on to read that one: the floor is
    learned from the run of frames, and a floor learned from a subsample is a
    different meter. It must also use a fresh meter per streaming session — a
    reconnect is a new camera.

    The header is marked (:data:`RESTAMPED_KEY`) rather than quietly rewritten: a
    benchmark index that claims a number no camera ever sent is a trap for
    whoever reads it next.

    **Expect a burst at the top of every restamped session.** The floor is seeded
    at :data:`MOTION_NOISE_FLOOR` — sensor noise, deliberately below any real rig's
    resting difference, so a camera opened onto motion cannot learn motion as its
    rest — and chases upward slowly. A recorded stream's first second or two of
    frames therefore difference far above the floor the meter has learned so far
    and score as movement. It costs one spurious burst-end read at the start of a
    take and nothing after it. Live this does not arise the same way: the browser
    edge meters from the moment the camera opens, long before anyone presses
    start.
    """
    return FrameEntry(
        header={**entry.header, "stability": meter.update(image), RESTAMPED_KEY: True},
        jpeg_rel=entry.jpeg_rel,
        received_ts=entry.received_ts,
    )
