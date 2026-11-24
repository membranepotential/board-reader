"""Derive the candidate hand signals from the probe caches, one take at a time.

    from hand_signals import load_take
    take = load_take("20260827-181802", feat_dir, cv_dir, prototypes)

Every signal is oriented so that larger means "more likely obscured". Per-square
signals are (n, 64) in FEN order (a8..h1); per-frame signals are (n,).

Model signals (from hand_extract.py's cache):

* ``nll_top`` — -log of the top softmax probability (the shipped confidence);
* ``margin`` — minus the top-1 / top-2 logit gap;
* ``energy`` — minus logsumexp of the logits (free energy; low logit mass = high);
* ``maha`` — Mahalanobis distance of the square's query embedding to the class
  mean of its own argmax class (means and shared covariance fitted on other takes:
  ``prototypes``);
* ``q_ring`` / ``q_oracle`` — cosine distance of the square's query embedding to
  the same square's embedding in a reference: ``ring`` is the median of a ring of
  15 frames sampled >= 1 s apart (deployable); ``oracle`` is the last earlier frame
  whose read equalled truth (an upper bound: live, "the last read that agreed with
  the settled board" plays that role);
* ``tok_ring`` / ``tok_oracle`` — the encoder's patch tokens: per-patch distance
  to the same reference, averaged under the square's attention footprint
  (layer 2 head 0, the head that localises squares);
* ``att_shift`` — how far that head's centroid moved from its 9-frame median, in
  patches; ``att_spread`` — its RMS radius; ``att_ent`` — its entropy.

Frame-level versions are the mean of the four largest per-square values
(``<name>_f``), plus ``tok_ring_board`` / ``tok_oracle_board``: the patch distance
averaged over the whole board footprint.

CV signals (hand_cv.py's cache) are passed through under their own names.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from board_reader_core.protocol import CLASSES, parse_placement

from board_reader_inference.measurement.trial import SessionTruth, TruthSegment, load_truth

ROOT = Path(__file__).resolve().parents[2]
RING, SAMPLE_S = 15, 1.0

#: 20261124-221950 has no truth.json. Rows 452-596 hold one position (the king's
#: return to e8 at rows 440-452, then a hand hovering and resting flat on the board
#: until ~row 572, checked by eye on every 4th frame), so it gets a one-segment
#: truth there; everything before row 452 is unscored.
MINI_TRUTH = {
    "20261124-221950": SessionTruth(
        board_id="phone",
        segments=(
            TruthSegment(0, "8/8/8/8/8/8/8/8"),  # placeholder, rows < 452 are not scored
            TruthSegment(452, "rnbqkb1r/ppp2ppp/8/3pP3/4n3/2N5/PPPP2PP/R1BQKBNR"),
        ),
        guard_seqs=0,
    )
}
MINI_FIRST_ROW = {"20261124-221950": 452}

#: Dense hand labels for the same stretch, by eye from every 4th frame: a hand
#: hovers from row 456, rests flat 512-572 and is gone by 576. Rows 504-511 and
#: 573-575 are ambiguous (fingers at the far edge / leaving) and stay unlabelled.
DENSE_221950 = {
    **dict.fromkeys(range(452, 456), "N"),
    **dict.fromkeys(range(456, 504), "H"),
    **dict.fromkeys(range(512, 573), "H"),
    **dict.fromkeys(range(576, 597), "N"),
}


def placement_classes(placement: str) -> np.ndarray:
    return np.array([CLASSES.index(c) for row in parse_placement(placement) for c in row])


@dataclass
class Take:
    name: str
    seq: np.ndarray
    ts: np.ndarray
    argmax: np.ndarray  # (n, 64)
    #: (n, 64) truth class, -1 where unscored (no truth, guard, other session)
    truth: np.ndarray
    sq: dict[str, np.ndarray] = field(default_factory=dict)
    fr: dict[str, np.ndarray] = field(default_factory=dict)
    #: row -> hand label
    labels: dict[int, str] = field(default_factory=dict)

    @property
    def scored(self) -> np.ndarray:
        return self.truth[:, 0] >= 0

    @property
    def sq_wrong(self) -> np.ndarray:
        return (self.argmax != self.truth) & self.scored[:, None]


def _truth(name: str) -> SessionTruth | None:
    if name in MINI_TRUTH:
        return MINI_TRUTH[name]
    path = ROOT / "recordings" / name / "truth.json"
    return load_truth(path) if path.exists() else None


def last_session_start(seq: np.ndarray) -> int:
    """Row where the last seq restart begins (truth keys the last session)."""
    restarts = np.nonzero(np.diff(seq) <= 0)[0]
    return int(restarts[-1] + 1) if len(restarts) else 0


def truth_grid(name: str, seq: np.ndarray) -> np.ndarray:
    out = np.full((len(seq), 64), -1)
    truth = _truth(name)
    if truth is None:
        return out
    first = max(last_session_start(seq), MINI_FIRST_ROW.get(name, 0))
    cache: dict[int, np.ndarray] = {}
    for row in range(first, len(seq)):
        s = int(seq[row])
        i = truth.index_for(s)
        if i < 0 or truth.in_guard(s):
            continue
        out[row] = cache.setdefault(i, placement_classes(truth.segments[i].placement))
    return out


def ring_reference(ts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each row, the median of a ring of RING earlier rows sampled >= SAMPLE_S apart."""
    ring: deque[np.ndarray] = deque(maxlen=RING)
    last = -1e18
    ref = np.empty_like(values, dtype=np.float32)
    for i in range(len(values)):
        ref[i] = np.median(np.stack(ring), axis=0) if ring else values[i]
        if ts[i] - last >= SAMPLE_S:
            ring.append(values[i].astype(np.float32))
            last = ts[i]
    return ref


def oracle_reference(truth: np.ndarray, argmax: np.ndarray) -> np.ndarray:
    """Row index of the last earlier row whose read equalled its truth (-1 if none)."""
    ref = np.full(len(truth), -1)
    last = -1
    for i in range(len(truth)):
        ref[i] = last
        if truth[i, 0] >= 0 and (argmax[i] == truth[i]).all():
            last = i
    return ref


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 1 - (a * b).sum(-1)


def top4(x: np.ndarray) -> np.ndarray:
    return np.sort(x, axis=1)[:, -4:].mean(axis=1)


@dataclass(frozen=True)
class Prototypes:
    """Per-class means and the shared precision of the query embedding."""

    means: np.ndarray  # (13, 384)
    precision: np.ndarray  # (384, 384)

    @classmethod
    def fit(cls, queries: np.ndarray, classes: np.ndarray) -> Prototypes:
        q = queries.astype(np.float64)
        means = np.stack([q[classes == k].mean(axis=0) for k in range(len(CLASSES))])
        resid = q - means[classes]
        cov = resid.T @ resid / len(q) + 1e-3 * np.eye(q.shape[1])
        return cls(means.astype(np.float32), np.linalg.inv(cov).astype(np.float32))

    def distance(self, queries: np.ndarray, classes: np.ndarray) -> np.ndarray:
        d = queries.astype(np.float32) - self.means[classes]
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.precision, d))


def clean_squares(
    feat_dir: Path, name: str, every: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Query embeddings of correctly read squares on scored rows (every 3rd row)."""
    f = np.load(feat_dir / f"{name}.npz")
    truth = truth_grid(name, f["seq"])
    rows = np.nonzero(truth[:, 0] >= 0)[0][::every]
    argmax = f["logits"][rows].argmax(-1)
    ok = argmax == truth[rows]
    return f["query"][rows][ok], argmax[ok]


def load_take(
    name: str, feat_dir: Path, cv_dir: Path | None, prototypes: Prototypes | None
) -> Take:
    f = np.load(feat_dir / f"{name}.npz")
    seq, ts = f["seq"], f["ts"]
    logits = f["logits"].astype(np.float64)
    argmax = logits.argmax(-1)
    take = Take(name=name, seq=seq, ts=ts, argmax=argmax, truth=truth_grid(name, seq))
    lse = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
    top2 = np.sort(logits, axis=-1)[..., -2:]
    sq = take.sq
    sq["nll_top"] = lse - top2[..., 1]
    sq["margin"] = -(top2[..., 1] - top2[..., 0])
    sq["energy"] = -lse
    query = f["query"].astype(np.float32)
    if prototypes is not None:
        sq["maha"] = prototypes.distance(query, argmax)
    sq["q_ring"] = _cos(query, ring_reference(ts, query))
    oracle = oracle_reference(take.truth, argmax)
    q_or = np.full(argmax.shape, np.nan, np.float32)
    has = oracle >= 0
    q_or[has] = _cos(query[has], query[oracle[has]])
    sq["q_oracle"] = q_or

    # Patch tokens: per-patch distance to a reference, pooled under each square's footprint.
    mem = f["mem"].astype(np.float32)  # (n, 46, 46, 32)
    amap = f["amap"].astype(np.float32)  # (n, 64, 23, 23)
    mem2 = mem.reshape(len(mem), 23, 2, 23, 2, -1).mean(axis=(2, 4))  # match amap's pooling
    foot = ring_reference(
        ts, amap
    )  # footprint from the ring, not the current (maybe covered) frame
    foot /= foot.sum(axis=(2, 3), keepdims=True)
    board = foot.sum(axis=1)  # (n, 23, 23)
    board /= board.sum(axis=(1, 2), keepdims=True)
    for kind, ref in (("ring", ring_reference(ts, mem2)), ("oracle", None)):
        if ref is None:
            ref = np.full_like(mem2, np.nan)
            ref[has] = mem2[oracle[has]]
        dist = np.linalg.norm(mem2 - ref, axis=-1)  # (n, 23, 23)
        sq[f"tok_{kind}"] = np.einsum("nshw,nhw->ns", foot, dist)
        take.fr[f"tok_{kind}_board"] = np.einsum("nhw,nhw->n", board, dist)

    cx, cy = f["cx"][:, 2, 0], f["cy"][:, 2, 0]
    med_x = np.stack([np.median(cx[max(0, i - 8) : i + 1], axis=0) for i in range(len(cx))])
    med_y = np.stack([np.median(cy[max(0, i - 8) : i + 1], axis=0) for i in range(len(cy))])
    sq["att_shift"] = np.hypot(cx - med_x, cy - med_y)
    sq["att_spread"] = f["spread"][:, 2, 0]
    sq["att_ent"] = f["ent"][:, 2, 0]

    if cv_dir is not None and (cv_dir / f"{name}.npz").exists():
        cv = np.load(cv_dir / f"{name}.npz")
        for k in cv.files:
            (sq if k.startswith("sq_") else take.fr)[k] = cv[k]

    for k, v in list(sq.items()):
        missing = np.isnan(v).any(axis=1)
        take.fr[f"{k}_f"] = np.where(missing, np.nan, top4(np.nan_to_num(v)))
    take.labels = _labels(name)
    return take


def _labels(name: str) -> dict[int, str]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hand_labels import LABELS

    out = dict(LABELS.get(name, {}))
    if name == "20261124-221950":
        out |= DENSE_221950
    return out


def frame_lines(name: str) -> list[dict]:
    text = (ROOT / "recordings" / name / "frames.ndjson").read_text()
    return [json.loads(s) for s in text.splitlines() if s.strip()]
