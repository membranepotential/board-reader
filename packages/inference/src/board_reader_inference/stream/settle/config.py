"""The settle debounce's knobs: window, quorum, deadline and the obscured-energy gate (DESIGN §5, D16)."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

#: How long a square must read as the same new class before the settled board
#: adopts it. With the shipped schedule's burst-end run (three reads 0.3 s apart,
#: after a 0.3 s calm hold) two reads span it, so a move settles 0.3 s after its
#: burst ends. Swept three times: over 0-2.5 s (docs/reports/2026-11-24-settle-tuning.md;
#: 2.5 s, sized for the old crop stack's noisy reads, mostly bought lag), then
#: 0.3-1.0 s against the 0.2-0.6 s spacings the forwarder's tail makes possible
#: (round 2 chose 0.5 s: with the energy gate not yet built, a window at or below
#: the spacing let hands settle), then 0-0.5 s with the energy gate on
#: (docs/reports/2026-11-25-latency-round3.md): 0.3 s settled no hand in any cell
#: and cut live take 2's median move-to-settle from 1.66 s to 0.94 s together with
#: the schedule's 0.3 s hold.
#:
#: **This window relies on the energy gate** (:data:`OBSCURED_ENERGY`,
#: ``BR_OBSCURED_ENERGY``, DESIGN D16), not on its own length, to keep hands out:
#: with the gate off, the round-3 settings let hands settle (6 on the gate takes,
#: 12 wrong boards in take 2's resting-hand stretch). Nothing falls back by
#: itself: with ``BR_OBSCURED_ENERGY=off``, return to round 2's settings with
#: ``BR_SETTLE_SECONDS=0.5`` and the schedule's ``exit_hold_s=0.7``.
#:
#: Shorter than one hand crossing (~1 s) on purpose: a hand the motion meter sees
#: is never read (the schedule skips bursts), and a hand resting still on the
#: board is what the energy gate catches.
SETTLE_SECONDS = 0.3

#: And never on fewer than this many consistent reads, however slow the stream —
#: at one read per window the debounce would otherwise be a pass-through. It is
#: also the quorum a square must have before it can be *dragged* into another
#: square's displacement (see :meth:`~board_reader_inference.stream.settle.stabilizer.BoardStabilizer.observe`).
SETTLE_MIN_READS = 2

#: How many settle windows a ripe-but-blocked square may wait before the
#: debounce stops protecting the board it is blocked by and falls through to the
#: current read across the whole contested set. Without a deadline the coherence
#: rule can starve a square forever: a square that reads its challenger on
#: *every* read still never lands if committing it alone would raise the excess
#: and its partner flickers below quorum. 2 windows puts the worst-case
#: staleness at ~3x SETTLE_SECONDS (one window to ripen, two to time out).
STALE_WINDOWS = 2

#: A read whose frame energy is above this is obscured and commits only balanced
#: piece groups (:mod:`board_reader_inference.stream.settle.balance`). The median of the probe's leave-one-take-out thresholds (each the
#: 99th percentile over the other gate takes' clear frames, -12.13 to -12.93) for
#: ChessQ V6. Energy lives on the model's logit scale, like a temperature: **re-fit
#: it for every model release** (docs/reports/2026-11-24-hand-signal-probe.md).
OBSCURED_ENERGY = -12.4

#: The environment ``serve`` reads the knobs from (:meth:`SettleConfig.from_env`).
SETTLE_ENV = (
    "BR_SETTLE_SECONDS",
    "BR_SETTLE_MIN_READS",
    "BR_SETTLE_STALE_WINDOWS",
    "BR_OBSCURED_ENERGY",
)

#: How ``BR_OBSCURED_ENERGY`` and ``replay --obscured-energy`` spell "no gate".
GATE_OFF = "off"


@dataclass(frozen=True)
class SettleConfig:
    """The debounce's knobs, as one value a node or a replay is built with.

    ``seconds`` and ``min_reads`` are the window a challenger must span
    (:data:`SETTLE_SECONDS`, :data:`SETTLE_MIN_READS`); ``stale_windows`` is how
    many windows a ripe-but-blocked square waits before the coherence rule gives
    way (:data:`STALE_WINDOWS`); ``obscured_energy`` is the frame energy above
    which a read commits only balanced piece groups (:data:`OBSCURED_ENERGY`,
    ``None`` switches that gate off). ``serve`` reads them from :data:`SETTLE_ENV`, ``replay`` from
    ``--settle-*`` and ``--obscured-energy`` (defaulting to the same environment),
    and a replay records them in its manifest, name and config digest.
    """

    seconds: float = SETTLE_SECONDS
    min_reads: int = SETTLE_MIN_READS
    stale_windows: int = STALE_WINDOWS
    obscured_energy: float | None = OBSCURED_ENERGY

    def __post_init__(self) -> None:
        if not (math.isfinite(self.seconds) and self.seconds >= 0):
            raise ValueError(f"settle seconds must be finite and >= 0, got {self.seconds}")
        if self.min_reads < 1:
            raise ValueError(f"settle min_reads must be >= 1, got {self.min_reads}")
        if self.stale_windows < 1:
            raise ValueError(f"settle stale_windows must be >= 1, got {self.stale_windows}")
        if self.obscured_energy is not None and not math.isfinite(self.obscured_energy):
            raise ValueError(
                f"obscured energy must be finite (or {GATE_OFF!r}), got {self.obscured_energy}"
            )

    @property
    def is_default(self) -> bool:
        return self == DEFAULT_SETTLE

    @property
    def slug(self) -> str:
        """A path-safe name for a non-default config: ``settle1.5s-2r-2w``, with
        ``-e-12.0`` (or ``-eoff``) appended when the energy gate is not the default."""
        slug = f"settle{self.seconds:g}s-{self.min_reads}r-{self.stale_windows}w"
        if self.obscured_energy != OBSCURED_ENERGY:
            slug += f"-e{_energy_text(self.obscured_energy)}"
        return slug

    def obscures(self, frame_energy: float | None) -> bool:
        """Whether a read with this frame energy is obscured (:mod:`board_reader_inference.stream.settle.balance`).

        A read without one (the stub) never is, and neither is any read with the
        gate off.
        """
        return (
            self.obscured_energy is not None
            and frame_energy is not None
            and frame_energy > self.obscured_energy
        )

    def to_dict(self) -> dict[str, float | int | None]:
        """The manifest's form (its first two keys predate this class)."""
        return {
            "settle_seconds": self.seconds,
            "settle_min_reads": self.min_reads,
            "stale_windows": self.stale_windows,
            "obscured_energy": self.obscured_energy,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> SettleConfig:
        """``BR_SETTLE_SECONDS`` / ``_MIN_READS`` / ``_STALE_WINDOWS``; unset keeps
        the default. A malformed value raises, so a typo fails at startup."""
        default = cls()
        seconds, min_reads, stale, energy = SETTLE_ENV
        return cls(
            seconds=_from_env(environ, seconds, float, default.seconds),
            min_reads=_from_env(environ, min_reads, int, default.min_reads),
            stale_windows=_from_env(environ, stale, int, default.stale_windows),
            obscured_energy=_from_env(
                environ,
                energy,
                parse_obscured_energy,
                default.obscured_energy,
                f"float or {GATE_OFF!r}",
            ),
        )


#: The shipped knobs, as the default argument everything that builds a debounce takes.
DEFAULT_SETTLE = SettleConfig()


def parse_obscured_energy(raw: str) -> float | None:
    """``"off"`` -> ``None`` (no gate), anything else a float; raises ``ValueError``.

    Finiteness is :class:`SettleConfig`'s check, so ``"nan"`` fails there.
    """
    return None if raw.strip().lower() == GATE_OFF else float(raw)


def _energy_text(energy: float | None) -> str:
    """``-12.0`` -> ``"-12"``, ``None`` -> ``"off"``: repr, a bare ``.0`` trimmed."""
    if energy is None:
        return GATE_OFF
    text = repr(energy)
    return text.removesuffix(".0")


def _from_env[T](
    environ: Mapping[str, str],
    name: str,
    parse: Callable[[str], T],
    default: T,
    what: str | None = None,
) -> T:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return parse(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} is not a valid {what or parse.__name__}") from e
