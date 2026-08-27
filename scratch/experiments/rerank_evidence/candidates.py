"""Named candidate rules and a tau sweep over each, per take."""
from __future__ import annotations
from dataclasses import dataclass, field
from transforms import Rule


@dataclass
class PairSet(Rule):
    """Swap when (top1, top2) is in ``pairs`` and the margin is below ``tau``."""
    pairs: frozenset[tuple[str, str]]
    tau: float
    name: str = "pair-set"

    def swap(self, c1, p1, c2, p2) -> bool:
        return (c1, c2) in self.pairs and (p1 - p2) < self.tau


@dataclass
class DemoteTop1(Rule):
    """Swap whenever top-1 is ``cls`` and the margin is below ``tau``."""
    cls: str
    tau: float
    name: str = "demote-top1"

    def swap(self, c1, p1, c2, p2) -> bool:
        return c1 == self.cls and (p1 - p2) < self.tau


BP = frozenset({("b", "p")})
BPQ = frozenset({("b", "p"), ("b", "q")})
BPQ_PLUS = frozenset({("b", "p"), ("b", "q"), ("K", "Q"), ("r", "n")})

FAMILIES = {
    "R1 b<-p": lambda t: PairSet(BP, t),
    "R2 b<-{p,q}": lambda t: PairSet(BPQ, t),
    "R3 b<-{p,q} + K<-Q + r<-n": lambda t: PairSet(BPQ_PLUS, t),
    "R4 demote any b": lambda t: DemoteTop1("b", t),
}
