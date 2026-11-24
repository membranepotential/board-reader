"""Score the candidate hand signals (hand_signals.py) and simulate gating the
stabilizer with them, for docs/reports/2026-11-24-hand-signal-probe.md.

    uv run python tools/probes/hand_eval.py FEAT_DIR CV_DIR [--sim SIGNAL ...]

Prints three tables:

1. **squares** — per-square ROC-AUC of each per-square signal for "ChessQ's argmax
   on this square is wrong", over every scored frame (truth present, outside the
   guard window), pooled over the seven gate takes, plus the worst single take;
2. **frames** — frame-level ROC-AUC for the hand labels (H vs N, hand_labels.py and
   the dense 221950 stretch), pooled and per take, and for "some square is wrong";
3. **gating** — the shipped burst-end schedule run over each take's frame index
   (gate takes: the base replay guard-only's restamped index; 221950: its own), every
   chosen frame read from the per-frame cache and settled by a fresh
   BoardStabilizer: ungated (must reproduce the replay's reads and settled boards),
   with a frame gate (a read whose frame signal exceeds the threshold is not fed to
   the stabilizer) or a square gate (each square whose signal exceeds it is fed as
   the currently settled class), each with and without a re-read (a gated read
   owes the schedule one more read at burst spacing). The
   threshold is set leave-one-take-out: the 99th percentile of the signal over the
   other takes' clean rows (read equal to truth, >= 3 s from any truth boundary).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from board_reader_core.protocol import CLASSES, encode_placement, parse_placement
from board_reader_core.recording import Recording

from board_reader_inference.measurement.trial import read_events
from board_reader_inference.stream.schedule import DEFAULT_POLICY_SPEC, parse_policy
from board_reader_inference.stream.stabilize import BoardStabilizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_signals import (
    MINI_TRUTH,
    ROOT,
    Prototypes,
    Take,
    _truth,
    clean_squares,
    last_session_start,
    load_take,
)

GATE = (
    "20260813-193308",
    "20260827-175539",
    "20260828-111421",
    "20260827-181802",
    "20260827-163352",
    "20260827-170333",
    "20260827-172736",
)
LIVE = ("20261124-221950", "20261124-204025")
BASE_REPLAY = "guard-only"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC, ties counted half; NaN when a side is empty."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def clean_rows(take: Take) -> np.ndarray:
    """Rows read perfectly and >= 3 s from any truth boundary."""
    truth = _truth(take.name)
    ok = take.scored & (take.argmax == np.where(take.truth < 0, -9, take.truth)).all(axis=1)
    if truth is None:
        return ok
    seq_ts = dict(zip(take.seq.tolist(), take.ts.tolist()))
    bts = np.array([seq_ts[s] for s in truth.starts[1:] if s in seq_ts])
    if len(bts):
        far = np.min(np.abs(take.ts[:, None] - bts[None, :]), axis=1) >= 3.0
        ok &= far
    return ok


# ------------------------------------------------------------------ tables


def square_table(takes: dict[str, Take], names: list[str]) -> None:
    print("\n## per-square AUC: argmax wrong vs right (scored rows, gate takes)")
    gate = [takes[t] for t in GATE]
    print(f"{'signal':<12} {'pooled':>7} {'worst take':>11}   per take (n wrong squares)")
    for name in names:
        pos = np.concatenate([t.sq[name][t.sq_wrong] for t in gate])
        neg = np.concatenate(
            [t.sq[name][t.scored[:, None] & ~t.sq_wrong & (t.truth >= 0)] for t in gate]
        )
        per = [
            (
                t.name[-6:],
                auc(t.sq[name][t.sq_wrong], t.sq[name][t.scored[:, None] & ~t.sq_wrong]),
                int(t.sq_wrong.sum()),
            )
            for t in gate
        ]
        worst = np.nanmin([a for _, a, _ in per])
        print(
            f"{name:<12} {auc(pos, neg):7.3f} {worst:11.3f}   "
            + " ".join(f"{a:.2f}({n})" for _, a, n in per)
        )


def frame_table(takes: dict[str, Take], names: list[str]) -> None:
    print(
        "\n## frame AUC: hand label H vs N (all labelled takes) | some square wrong (gate takes)"
    )
    order = list(GATE) + list(LIVE)
    print(f"{'signal':<16} {'H/N pool':>8} {'worst':>6} {'wrong pool':>10}   per take H/N")
    for name in names:
        hp, hn, per = [], [], []
        for t in order:
            take = takes[t]
            if name not in take.fr:
                continue
            v = take.fr[name]
            h = np.array([r for r, lab in take.labels.items() if lab == "H"], int)
            n = np.array([r for r, lab in take.labels.items() if lab == "N"], int)
            hp.append(v[h]), hn.append(v[n])
            per.append(auc(v[h], v[n]))
        wp = np.concatenate(
            [takes[t].fr[name][takes[t].scored & takes[t].sq_wrong.any(1)] for t in GATE]
        )
        wn = np.concatenate(
            [takes[t].fr[name][takes[t].scored & ~takes[t].sq_wrong.any(1)] for t in GATE]
        )
        print(
            f"{name:<16} {auc(np.concatenate(hp), np.concatenate(hn)):8.3f} {np.nanmin(per):6.3f} "
            f"{auc(wp, wn):10.3f}   " + " ".join(f"{a:.2f}" for a in per)
        )


def recall_table(takes: dict[str, Take], names: list[str]) -> None:
    """At the leave-one-take-out 99th-percentile threshold: H recall, N false-block rate."""
    print(
        "\n## frame gate at LOTO p99-of-clean threshold: share of H / N / wrong frames blocked"
    )
    print(f"{'signal':<16} {'H blocked':>9} {'N blocked':>9} {'wrong blocked':>13}")
    order = list(GATE) + list(LIVE)
    for name in names:
        hb = nb = wb = hc = nc = wc = 0
        for t in order:
            thr = loto_threshold(takes, name, t, frame=True)
            take = takes[t]
            v = take.fr[name]
            for r, lab in take.labels.items():
                if lab == "H":
                    hc += 1
                    hb += v[r] > thr
                elif lab == "N":
                    nc += 1
                    nb += v[r] > thr
            if t in GATE:
                w = take.scored & take.sq_wrong.any(1)
                wc += int(w.sum())
                wb += int((v[w] > thr).sum())
        print(f"{name:<16} {hb / hc:9.2f} {nb / nc:9.2f} {wb / max(wc, 1):13.2f}")


def loto_threshold(
    takes: dict[str, Take], name: str, held: str, *, frame: bool, q: float = 99.0
) -> float:
    vals = []
    for t in GATE:
        if t == held:
            continue
        take = takes[t]
        rows = clean_rows(take)
        v = take.fr[name][rows] if frame else take.sq[name][rows].ravel()
        vals.append(v[np.isfinite(v)])
    return float(np.percentile(np.concatenate(vals), q))


# ------------------------------------------------------------------ gating


@dataclass
class SimResult:
    perfect: int
    scored: int
    other_wrong: int  # settled wrong and not equal to the previous truth (not lag)
    lag_s: float  # mean seconds from a boundary's first frame to settled == truth
    never: int  # boundaries never settled within their segment
    blocked: int  # reads gated (frame) or squares held (square)
    reads: int  # reads the schedule asked for (GPU cost)


@dataclass(frozen=True)
class SimEvent:
    frame_seq: int
    capture_ts: float
    settled: str


def events_for(take: str):
    """The recorded events: a gate take's base replay, a live take's own."""
    rec = ROOT / "recordings" / take
    src = rec if take in LIVE else rec / "replays" / BASE_REPLAY
    return read_events(Recording(src), board_id="phone")


def frame_entries(take: str):
    """The frames the policy sees: a gate take's replay index (its stability
    already restamped where the gate restamps), a live take's own index."""
    rec = ROOT / "recordings" / take
    src = rec if take in LIVE else rec / "replays" / BASE_REPLAY
    return list(Recording(src).frame_index(board_id="phone"))


def placement_of(argmax_row: np.ndarray) -> str:
    return encode_placement(
        [[CLASSES[int(c)] for c in argmax_row[r * 8 : r * 8 + 8]] for r in range(8)]
    )


def simulate(
    take: Take, mode: str, signal: str | None, thr: float, *, reread: bool
) -> tuple[SimResult, list[SimEvent]]:
    """Run the shipped burst-end schedule over the take's frames, read each chosen
    frame from the per-frame cache, and settle with a fresh BoardStabilizer.

    ``mode``: ``none``; ``frame`` (a read whose frame signal > thr is not fed to the
    stabilizer); ``nocommit`` (such a read is fed as evidence, so it can start or
    extend a square's challenger, but may not commit anything: the probe sets the
    stabilizer's window to infinity for that one read); ``square`` (each square
    whose signal > thr is fed as the settled class). With ``reread`` a gated read owes the policy one more read at burst
    spacing (unless a burst is in progress), so a hand that lingers after the
    burst end is read again 0.3 s later instead of at the 2 s keepalive.
    """
    first = last_session_start(take.seq)
    row_of = {int(s): first + i for i, s in enumerate(take.seq[first:])}
    policy = parse_policy(DEFAULT_POLICY_SPEC)
    stab = BoardStabilizer()
    settled_now: str | None = None
    out: list[SimEvent] = []
    blocked = reads = 0
    for entry in frame_entries(take.name):
        if not policy.should_estimate(entry):
            continue
        reads += 1
        row = row_of[entry.seq]
        placement = placement_of(take.argmax[row])
        gated = False
        if settled_now is not None and signal:
            if mode in ("frame", "nocommit") and take.fr[signal][row] > thr:
                gated = True
            elif mode == "square":
                held = take.sq[signal][row] > thr
                if held.any():
                    gated = True
                    grid, sgrid = parse_placement(placement), parse_placement(settled_now)
                    for s in np.nonzero(held)[0]:
                        grid[s // 8][s % 8] = sgrid[s // 8][s % 8]
                    blocked += int(held.sum())
                    placement = encode_placement(grid)
        if gated and reread and not policy.bursting:
            policy._owed = max(policy._owed, 1)
        if gated and mode == "frame":
            blocked += 1
        elif gated and mode == "nocommit":
            blocked += 1
            window, stab._window = stab._window, float("inf")
            try:
                settled_now = stab.observe(placement, entry.capture_ts)
            finally:
                stab._window = window
        else:
            settled_now = stab.observe(placement, entry.capture_ts)
        out.append(SimEvent(entry.seq, entry.capture_ts, settled_now))
    return score(take, out, _truth(take.name), blocked, reads), out


def score(take: Take, events: list[SimEvent], truth, blocked: int, reads: int) -> SimResult:
    perfect = scored = other = 0
    first_row = {"20261124-221950": 452}.get(take.name, 0)
    first = last_session_start(take.seq)
    row_of = {int(s): first + i for i, s in enumerate(take.seq[first:])}
    for e in events:
        i = truth.index_for(e.frame_seq)
        if i < 0 or truth.in_guard(e.frame_seq) or row_of[e.frame_seq] < first_row:
            continue
        scored += 1
        if e.settled == truth.segments[i].placement:
            perfect += 1
        elif i == 0 or e.settled != truth.segments[i - 1].placement:
            other += 1
    lags, never = [], 0
    if take.name not in MINI_TRUTH:
        seq_ts = dict(zip(take.seq[first:].tolist(), take.ts[first:].tolist()))
        for b in range(1, len(truth.segments)):
            start = truth.starts[b]
            end = truth.starts[b + 1] if b + 1 < len(truth.segments) else 1 << 30
            hit = next(
                (
                    e.capture_ts
                    for e in events
                    if start <= e.frame_seq < end
                    and e.settled == truth.segments[b].placement
                ),
                None,
            )
            if hit is None:
                never += 1
            elif start in seq_ts:
                lags.append(hit - seq_ts[start])
    return SimResult(
        perfect,
        scored,
        other,
        float(np.mean(lags)) if lags else float("nan"),
        never,
        blocked,
        reads,
    )


def sim_table(takes: dict[str, Take], signals: list[str]) -> None:
    print(
        "\n## gating the shipped schedule (settled perfect/scored, oN = not-lag wrong, mean lag s, nN = boundaries never settled, reads)"
    )
    order = [*GATE, "20261124-221950"]
    for t in order:
        _, sim = simulate(takes[t], "none", None, 0.0, reread=False)
        rec = {e.frame_seq: e.settled for e in events_for(t)}
        same_frames = sum(e.frame_seq in rec for e in sim)
        same_boards = sum(rec.get(e.frame_seq) == e.settled for e in sim)
        print(
            f"  ungated {t}: {len(sim)} reads, {same_frames} on recorded read frames, {same_boards} with the recorded settled board (recorded {len(rec)})"
        )
    rows = [("ungated", "none", None, False)]
    for s in signals:
        if s in takes[GATE[0]].fr:
            rows += [
                (f"frame:{s}", "frame", s, False),
                (f"nocommit:{s}", "nocommit", s, False),
            ]
        else:
            rows += [(f"square:{s}", "square", s, False)]
    print(
        f"{'variant':<30} "
        + " ".join(f"{t[-6:]:>19}" for t in order)
        + "  total gate | 221950"
    )
    for label, mode, sig, reread in rows:
        cells, tp, ts_, to, tn, tr, lags = [], 0, 0, 0, 0, 0, []
        for t in order:
            thr = loto_threshold(takes, sig, t, frame=(mode != "square")) if sig else 0.0
            r, _ = simulate(takes[t], mode, sig, thr, reread=reread)
            if t in GATE:
                tp, ts_, to, tn, tr = (
                    tp + r.perfect,
                    ts_ + r.scored,
                    to + r.other_wrong,
                    tn + r.never,
                    tr + r.reads,
                )
                lags.append(r.lag_s)
            cells.append(
                f"{r.perfect}/{r.scored} o{r.other_wrong} {r.lag_s:.1f}s"
                + (f" n{r.never}" if r.never else "")
            )
        print(
            f"{label:<30} "
            + " ".join(f"{c:>19}" for c in cells)
            + f"  {tp}/{ts_} o{to} n{tn} lag {np.nanmean(lags):.2f}s reads {tr}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir", type=Path)
    ap.add_argument("cv_dir", type=Path)
    ap.add_argument("--sim", nargs="*", default=[])
    args = ap.parse_args()
    clean = {t: clean_squares(args.feat_dir, t) for t in GATE}
    takes: dict[str, Take] = {}
    for t in GATE + LIVE:
        others = [clean[o] for o in GATE if o != t]
        proto = Prototypes.fit(
            np.concatenate([q for q, _ in others]), np.concatenate([c for _, c in others])
        )
        takes[t] = load_take(t, args.feat_dir, args.cv_dir, proto)
        print(
            f"loaded {t}: {len(takes[t].seq)} rows, {int(takes[t].scored.sum())} scored, "
            f"{int(takes[t].sq_wrong.any(1).sum())} with a wrong square, {len(takes[t].labels)} labelled",
            flush=True,
        )
    sq_names = list(takes[GATE[0]].sq)
    fr_names = list(takes[GATE[0]].fr)
    square_table(takes, sq_names)
    frame_table(takes, fr_names)
    recall_table(takes, fr_names)
    if args.sim:
        sim_table(takes, args.sim)


if __name__ == "__main__":
    main()
