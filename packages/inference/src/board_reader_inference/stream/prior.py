"""Hard-constraint **decoder** over a per-square distribution.

The estimator sometimes reads a board that cannot exist: two white kings, a pawn
on rank 8, nineteen black pieces. Those are not chess *judgements* — they are
violations of necessary conditions on the piece multiset, and a reader that emits
them is throwing away information it already has in ``square_dist``.

This module is a pure function from an 8x8x13 distribution to a *placement* that
satisfies those conditions, reached by relabelling the fewest, lowest-margin
squares. It is a **decoder, never part of estimation** (DESIGN D1): it does not
mutate or rewrite ``square_dist`` or ``placement``. The raw distribution stays on
the wire exactly as the estimator produced it, so a downstream legality smoother
cannot double-count this prior. The repaired board travels *beside* the raw read
— HTTP responses carry it as ``result["prior"]``, and the live stream carries
:meth:`PriorDecode.to_repair` as :attr:`PositionEvent.prior` — always as an
auxiliary block a consumer may ignore.

Deliberately *only* the hard constraints (:class:`Constraint`). Soft priors — two
queens are unlikely, bishops sit on opposite colours, castled kings have a rook
next to them — need calibrated confidences we do not have yet, and are out of
scope until they do.

**Orientation matters, for exactly one constraint, and it must not be guessed.**
Three of the four read only piece *counts* and are rotation-invariant.
``BACK_RANK_PAWNS`` is about where a pawn sits, so it needs a known frame — and a
*guessed* frame is worse than none, because the orientation guesser the old
corner-frame stack had picked the rotation that **minimises pawns on ranks 1/8**,
the very quantity this constraint measures. Feed it that guess and
the constraint is circular: where the guess is right there is nothing left to
fire on, and where it does fire the residual is evidence the guess was *wrong* —
so the decoder deletes a good square. Measured on
``20260806-235526-0005``: back-rank pawns per rotation ``[2, 1, 2, 1]``, the
guesser picks ``k=1``, and a correct pawn on ``a3`` dies at 3.70 nats.

So ``quarter_turns`` defaults to ``None`` — *frame unknown*, constraint skipped.
Pass an ``int`` only where the orientation is **established rather than inferred**
(the snapshot operator confirms it). Placement and
flips always come back in the frame that was passed in.

    decode(event.square_dist).placement                    # counts only
    decode(dist, quarter_turns=confirmed_k).placement       # + back-rank pawns
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    GRID,
    NUM_CLASSES,
    PriorFlip,
    PriorRepair,
    encode_placement,
    square_name_at,
)

#: Floor under a probability before taking its log — an estimator may emit an
#: exact 0.0 for a class, and ``log 0`` would poison every comparison with -inf.
#: 1e-12 keeps such a flip expensive (~27.6 nats) without being infinite.
MIN_PROB = 1e-12

#: Hard cap on greedy iterations. Each accepted flip *strictly* reduces the
#: integer violation, so this can only bind on adversarial input; it exists so
#: the loop is provably finite by construction rather than by argument.
MAX_FLIPS = 64

_WHITE: frozenset[int] = frozenset(CLASS_INDEX[c] for c in "PNBRQK")
_BLACK: frozenset[int] = frozenset(CLASS_INDEX[c] for c in "pnbrqk")
_PAWNS: frozenset[int] = frozenset((CLASS_INDEX["P"], CLASS_INDEX["p"]))

#: Per colour: (pawn, knight, bishop, rook, queen, king), plus how many of each
#: non-pawn a legal army may hold without a promotion having happened.
_ARMY: tuple[tuple[str, ...], ...] = (("P", "N", "B", "R", "Q", "K"),
                                      ("p", "n", "b", "r", "q", "k"))
_FREE: tuple[int, int, int, int] = (2, 2, 2, 1)  # N, B, R, Q

#: A side starts with 8 pawns, so at most 8 promotions can ever have happened.
PROMOTION_BUDGET = 8
#: A side starts with 16 pieces and can never gain one.
MAX_PIECES = 16


class Constraint(StrEnum):
    """The four necessary conditions. Values match :class:`Violation` field names."""

    #: At most one king per colour. Deliberately ``<= 1`` and not ``== 1``:
    #: back-rank occlusion legitimately hides a king (DESIGN D8), and ``== 1``
    #: would invent one by stealing another square's label exactly when the image
    #: is at its worst. A zero-king board is left alone.
    KINGS = "kings"
    #: No pawn on rank 1 or rank 8 — it would have had to promote. The only
    #: rank-dependent constraint, so the only one that needs a known frame; it is
    #: skipped entirely when the caller cannot name one.
    BACK_RANK_PAWNS = "back_rank_pawns"
    #: At most 16 pieces per colour.
    PIECE_COUNT = "piece_count"
    #: ``max(0,Q-1) + max(0,R-2) + max(0,B-2) + max(0,N-2) + P <= 8`` per colour:
    #: every surplus officer consumed a pawn, and surviving pawns did not promote.
    PROMOTION_BUDGET = "promotion_budget"


@dataclass(frozen=True)
class Violation:
    """How far a board is from possible, one non-negative excess per constraint."""

    kings: int
    back_rank_pawns: int
    piece_count: int
    promotion_budget: int

    @property
    def total(self) -> int:
        return self.kings + self.back_rank_pawns + self.piece_count + self.promotion_budget

    @property
    def satisfied(self) -> bool:
        return self.total == 0

    def __getitem__(self, constraint: Constraint) -> int:
        return int(getattr(self, constraint.value))

    def to_dict(self) -> dict[str, int]:
        return {c.value: self[c] for c in Constraint if self[c]}

    def __str__(self) -> str:
        parts = [f"{c.value} {self[c]}" for c in Constraint if self[c]]
        return " · ".join(parts) if parts else "possible"


@dataclass(frozen=True)
class Tally:
    """The only board facts the constraints depend on: class counts + back-rank pawns.

    ``back_rank_pawns is None`` means the board's orientation is unknown, so where
    a pawn stands is not a fact we have — :attr:`Constraint.BACK_RANK_PAWNS` then
    contributes nothing rather than guessing. The counts are rotation-invariant and
    always available.
    """

    per_class: tuple[int, ...]  # 13 counts, indexed like CLASSES
    back_rank_pawns: int | None


@dataclass(frozen=True)
class Flip:
    """One relabelling the decoder applied, in the order it applied it.

    ``cost`` is ``log p(square, from_class) - log p(square, to_class)`` in nats —
    the margin given up. It is non-negative for the first flip of a square (the
    argmax is being abandoned) and may be negative for a second flip of the same
    square, which is why the greedy step minimises it rather than its absolute value.
    """

    square: str  # algebraic, e.g. "e6"
    from_class: str
    to_class: str
    cost: float
    constraint: Constraint  # which excess this flip relieved most

    def __str__(self) -> str:
        return f"{self.square} {self.from_class}->{self.to_class}"

    def to_dict(self) -> dict:
        return {
            "square": self.square,
            "from": self.from_class,
            "to": self.to_class,
            "cost": round(self.cost, 4),
            "constraint": self.constraint.value,
        }


@dataclass(frozen=True)
class PriorDecode:
    """The decoder's answer: a placement, the audit trail, and what it could not fix."""

    placement: str
    flips: tuple[Flip, ...]
    residual: Violation

    @property
    def repaired(self) -> bool:
        return bool(self.flips)

    def to_dict(self) -> dict:
        return {
            "placement": self.placement,
            "flips": [f.to_dict() for f in self.flips],
            "residual": self.residual.to_dict(),
        }

    def to_repair(self) -> PriorRepair:
        """The wire shape (:attr:`PositionEvent.prior`) of this decode."""
        return PriorRepair(
            placement=self.placement,
            flips=tuple(
                PriorFlip(
                    square=f.square,
                    from_class=f.from_class,
                    to_class=f.to_class,
                    cost=f.cost,
                    constraint=f.constraint.value,
                )
                for f in self.flips
            ),
            residual=self.residual.to_dict(),
        )


#: Flat square index (row 0 = rank 8, col 0 = file a) -> algebraic name. The
#: derivation is the protocol's (it names squares in the grid frame every
#: ``square_dist`` uses); re-exported here because the flip records read best
#: beside the constraints that produce them.
square_name = square_name_at


def tally(labels: np.ndarray, *, oriented: bool = True) -> Tally:
    """Count an 8x8 grid of class indices into the facts the constraints read.

    ``oriented=False`` when row 0 is not known to be rank 8: the back-rank pawns
    are then left unknown rather than counted in an arbitrary frame.
    """
    flat = np.asarray(labels).reshape(-1)
    counts = np.bincount(flat, minlength=NUM_CLASSES)[:NUM_CLASSES]
    back = (
        sum(
            1
            for i in (*range(GRID), *range(GRID * (GRID - 1), GRID * GRID))
            if int(flat[i]) in _PAWNS
        )
        if oriented
        else None
    )
    return Tally(per_class=tuple(int(n) for n in counts), back_rank_pawns=back)


def _tally_with(t: Tally, index: int, current: int, new: int) -> Tally:
    """``t`` as it would be after relabelling one square — O(1), no board needed."""
    counts = list(t.per_class)
    counts[current] -= 1
    counts[new] += 1
    back = t.back_rank_pawns
    if back is not None and index // GRID in (0, GRID - 1):
        back += (new in _PAWNS) - (current in _PAWNS)
    return Tally(per_class=tuple(counts), back_rank_pawns=back)


def _promotions_used(counts: tuple[int, ...], army: tuple[str, ...]) -> int:
    pawn, *officers = army[:5]  # P, N, B, R, Q — the king is not promotable-into
    surplus = sum(
        max(0, counts[CLASS_INDEX[sym]] - free)
        for sym, free in zip(officers, _FREE, strict=True)
    )
    return surplus + counts[CLASS_INDEX[pawn]]


def violation(t: Tally) -> Violation:
    """The single scalar (per constraint) the greedy repair descends."""
    counts = t.per_class
    kings = sum(max(0, counts[CLASS_INDEX[army[5]]] - 1) for army in _ARMY)
    pieces = sum(
        max(0, sum(counts[i] for i in side) - MAX_PIECES) for side in (_WHITE, _BLACK)
    )
    promotions = sum(
        max(0, _promotions_used(counts, army) - PROMOTION_BUDGET) for army in _ARMY
    )
    return Violation(
        kings=kings,
        back_rank_pawns=t.back_rank_pawns or 0,  # unknown frame -> not our claim
        piece_count=pieces,
        promotion_budget=promotions,
    )


def _driver(before: Violation, after: Violation) -> Constraint:
    """Which constraint a flip relieved most; declaration order breaks ties."""
    return min(
        ((-(before[c] - after[c]), i, c) for i, c in enumerate(Constraint)),
    )[2]


def _kings_flip(
    labels: list[int],
    log_p: np.ndarray,
    t: Tally,
    excess: Violation,
) -> tuple[tuple[float, int, int], Tally, Violation] | None:
    """The surplus-kings flip, chosen by king *evidence* rather than flip cost.

    Two argmax kings of one colour ask "which square holds the real king?", and
    the margin the generic greedy minimises answers a different question — "where
    is switching to the runner-up cheapest?". Its ``log p(k) - log p(q)`` is
    dominated by the log of a tiny queen tail, which is noise. Comparing what the
    estimator actually saw, ``p(king)`` per square, and demoting the weakest
    king is both stabler and more often right: on the 08-13 phone-trial replay
    (628 two-king frames, queen d8/king e8 ground truth) the margin rule demoted
    the true king 54.8% of the time and swapped its choice 139 times; lowest
    ``p(king)`` was wrong 26.6% with 61 swaps. Temporal smoothing of the
    distributions fixes neither (the misread is systematic), so this choice of
    *evidence* is the decoder's whole lever.

    **That lever is at its ceiling, and the 08-13 take is not evidence it is
    broken.** Re-measured 2026-08-28 on the shipped cascade+bank
    (``run_e``/``run_f``, ``bank-baseline-20260828``, ``BR_CROP_MASK=gray``),
    the take's two squares' ``p(king)`` genuinely cross: the *false* king d8
    reads ``k`` at a stable 0.33-0.41 with a fat 0.17-0.24 margin, while the
    *true* king e8 ranges 0.20-0.45 with a margin as thin as 0.003 — this rule
    demotes d8 on 17 of 34 two-king events and e8 on the other 17. (The 08-28
    note blamed a session-bank exemplar lock; the 2026-08-31 per-member capture
    corrected that: it is the *trained cascade* that is confidently wrong,
    reading d8 as ``k`` at 0.97-0.99 while true e8 sits at 0.52-0.78.) It is a
    coin flip because the evidence is a coin flip, not because the rule reads
    it wrong; the
    :class:`~board_reader_inference.stream.stabilize.BoardStabilizer` is what recovers
    the take (settled 22/31 perfect from a 15/31 prior, against a 29/31 ceiling
    if every demotion went to d8).

    Three alternatives were scored across all four gate takes (replays of the
    then-shipped stack) and every one of them is worse:
    ``margin`` (the pre-faaa4ed MAP rule, and the true maximum-likelihood
    answer under square independence) collapses 08-13 to 2/31 settled;
    ``max-margin`` and ``neg-entropy`` buy 08-13 (27/31 and 22/31) by demoting
    the *more* confident king, which is anti-MAP — it wins only because this
    take's cascade misread happens to make the wrong king the confident one,
    and it costs 20260827-175539 its settled perfects (5/12 -> 2/12, per-square
    0.9609 -> 0.9440) and 20260828-111421 per-square accuracy (0.9266 ->
    0.9096). Re-opened 2026-08-31 on 13 fresh boards of an unseen style (23 new
    two-king events; scratch/experiments/two_kings_reopen): the best fresh-data
    challenger, ``p(K) - p(Q)``, won the stills (+5 squares, +1 perfect, no
    curated loss) and then collapsed 08-13 from 22/31 to **0/31** settled in
    the replay gate — the queen-mass polarity *inverts* between datasets, and a
    lambda sweep shows no safe amount of queen subtraction. The ceiling is
    confirmed: a rule that fixes one dataset's systematic misread and breaks
    another's is a fitted constant, not a decoder. The remaining lever is
    recognition on d8 (or a temporal identity prior above this pure function),
    never a different per-frame tie-break.

    The demoted square still goes to its most probable class among those that
    strictly reduce the total violation (so relieving KINGS cannot silently
    worsen another constraint), and the recorded cost stays the margin given up,
    comparable with every other flip. Returns ``None`` when no kings are in
    excess or no target class helps — the generic greedy then decides.
    """
    if not excess.kings:
        return None
    for army in _ARMY:
        king = CLASS_INDEX[army[5]]
        if t.per_class[king] <= 1:
            continue
        weakest = min(
            (i for i, current in enumerate(labels) if current == king),
            key=lambda i: (float(log_p[i, king]), i),
        )
        for new in sorted(
            (n for n in range(NUM_CLASSES) if n != king),
            key=lambda n: (-float(log_p[weakest, n]), n),
        ):
            candidate = _tally_with(t, weakest, king, new)
            after = violation(candidate)
            if after.total < excess.total:
                cost = float(log_p[weakest, king] - log_p[weakest, new])
                return (cost, weakest, new), candidate, after
    return None


def decode(
    square_dist: np.ndarray,
    *,
    quarter_turns: int | None = None,
    max_flips: int = MAX_FLIPS,
) -> PriorDecode:
    """Nearest placement satisfying every hard constraint, by greedy relabelling.

    ``square_dist`` is read-only here and is never returned in modified form. A
    board that already satisfies every constraint comes back as its own argmax
    with no flips.

    Each iteration first offers surplus kings their own, evidence-ordered flip
    (:func:`_kings_flip`); otherwise it enumerates all 64x12 single-square
    relabels, keeps those that *strictly* reduce :func:`violation`, and applies
    the cheapest in nats, with ``(cost, square index, class index)`` as a total
    order so the same input always yields the same output. Since the violation is a non-negative integer that
    strictly decreases per accepted flip, the loop terminates; if no single flip
    can help (or ``max_flips`` binds) the best board so far is returned with the
    surviving excess in :attr:`PriorDecode.residual`. It never raises on a
    well-shaped distribution.

    ``quarter_turns`` is the number of ``np.rot90`` turns that bring
    ``square_dist`` into the a1-bottom-left frame, and enables
    :attr:`Constraint.BACK_RANK_PAWNS`. ``None`` (the default) means the frame is
    unknown and that one constraint is skipped — never pass an *inferred*
    rotation, see the module docstring. :attr:`PriorDecode.placement` and every
    flip square are reported in the frame that was passed in.
    """
    arr = np.asarray(square_dist, dtype=np.float64)
    if arr.shape != (GRID, GRID, NUM_CLASSES):
        raise ValueError(
            f"square_dist must be {(GRID, GRID, NUM_CLASSES)}, got {arr.shape}"
        )
    oriented = quarter_turns is not None
    turns = quarter_turns % 4 if quarter_turns is not None else 0
    # Work in the chess frame; ``origin`` maps a working square back to the
    # caller's frame so flips are named in the coordinates they were handed to us in.
    upright = np.rot90(arr, turns, axes=(0, 1)) if turns else arr
    origin = np.rot90(np.arange(GRID * GRID).reshape(GRID, GRID), turns).reshape(-1)
    labels: list[int] = [int(i) for i in upright.reshape(-1, NUM_CLASSES).argmax(axis=-1)]
    log_p = np.log(np.maximum(upright.reshape(-1, NUM_CLASSES), MIN_PROB))

    t = tally(np.asarray(labels).reshape(GRID, GRID), oriented=oriented)
    excess = violation(t)
    flips: list[Flip] = []
    while not excess.satisfied and len(flips) < max_flips:
        # Surplus kings are resolved by evidence first (see _kings_flip): which
        # square keeps the king is not a cost question, and the generic margin
        # below answers it with tail noise.
        best = _kings_flip(labels, log_p, t, excess)
        if best is None:
            for index, current in enumerate(labels):
                for new in range(NUM_CLASSES):
                    if new == current:
                        continue
                    candidate = _tally_with(t, index, current, new)
                    after = violation(candidate)
                    if after.total >= excess.total:
                        continue
                    key = (float(log_p[index, current] - log_p[index, new]), index, new)
                    if best is None or key < best[0]:
                        best = (key, candidate, after)
        if best is None:
            break  # possible in principle; leave the residual visible rather than loop
        (cost, index, new), t, after = best
        flips.append(
            Flip(
                square=square_name(int(origin[index])),
                from_class=CLASSES[labels[index]],
                to_class=CLASSES[new],
                cost=cost,
                constraint=_driver(excess, after),
            )
        )
        labels[index] = new
        excess = after

    grid = np.array([CLASSES[i] for i in labels], dtype="<U5").reshape(GRID, GRID)
    if turns:
        grid = np.rot90(grid, -turns)
    return PriorDecode(
        placement=encode_placement(grid.tolist()), flips=tuple(flips), residual=excess
    )
