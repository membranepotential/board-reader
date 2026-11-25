"""Latency round 3: a shorter calm hold and a shorter settle, simulated on cached reads.

    uv run python tools/probes/latency3_sim.py CACHE_DIR validate
    uv run python tools/probes/latency3_sim.py CACHE_DIR sweep  > grid.tsv
    uv run python tools/probes/latency3_sim.py CACHE_DIR show HOLD READS SPACING SECS MIN OBSC

``CACHE_DIR`` holds ``latency3_cache.py``'s per-frame ChessQ reads. Nothing here
re-implements the node: the schedule is ``schedule.policies.BurstTriggered``, the debounce
``stabilize.BoardStabilizer`` (and its guard, ``rules.excess``), the orientation
check ``ingest.OrientationGuard``, and the phone's sending is the edge's
``AdaptiveForwarder``, the Python mirror of ``lib/edge/motion.ts``. Only their
parameters change, through their constructors.

**Two streams the node can be fed.**

* ``recorded`` -- the frames as recorded, stability restamped where the gate
  restamps. With the shipped parameters this must reproduce the gate replays
  ``guard-only`` (energy gate off) and ``energy`` (on) read for read (``validate``).
* ``phone`` -- what today's phone would have SENT under a given hold: the recorded
  frames run through ``AdaptiveForwarder(exit_hold_s=hold, tail_s=1)`` (a fake
  meter hands it each frame's stability). The node's ``BurstTriggered`` gets the
  same ``exit_hold_s``, because both sides must segment the same bursts.

**The old takes' missing frames.** Six of the seven gate takes were recorded
before the forwarder's 1 s burst tail existed: after a burst they dropped to the
1 fps keepalive, so a tail frame today's phone would send may not exist. Gaps over
:data:`GAP_S` are filled with synthetic frames every :data:`FILL_DT` (the phone's
12 fps burst rate on its 20 fps capture), stability 1.0 (a gap only happens at
rest: a dip under ``ENTER_BURST`` would have opened a burst and dense sending),
each borrowing the read, energy and direction of the nearest recorded frame. A
synthetic frame is offered to the forwarder **only while it sends at burst rate**
(in a burst or its tail), so it stands in for a tail frame and never displaces a
recorded keepalive frame. ``fill=False`` drops them: the pessimistic bound, where
a read that falls into such a gap waits for the next recorded frame. Numbers that
rest on synthetic reads are projected, not measured (``.claude/rules/stream.md``).

**The obscured rule** is the shipped energy gate (DESIGN D16): the stabilizer is
built with ``SettleConfig(obscured_energy=OBSCURED_ENERGY)`` or ``None`` (off) and
each read goes through ``observe(..., obscured=config.obscures(energy))``, as
``ingest._settle_read`` does. The cached energy is ``chessq.frame_energy`` of
the read's logits.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median

import numpy as np

from board_reader_core.protocol import CLASSES, encode_placement, parse_placement
from board_reader_core.recording import Recording
from board_reader_core.wire import Frame

from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig
from board_reader_inference.measurement.trial import SessionTruth, load_truth, read_events
from board_reader_inference.stream.ingest import OrientationGuard
from board_reader_inference.stream.schedule.constants import EXIT_HOLD_S
from board_reader_inference.stream.schedule.policies import BurstTriggered
from board_reader_inference.stream.stabilize import (
    OBSCURED_ENERGY,
    BoardStabilizer,
    SettleConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate_table import GATE_TAKES

ROOT = Path(__file__).resolve().parents[2]
LIVE = "20261124-221950"
RESTAMP = {t.take: t.restamp_stability for t in GATE_TAKES} | {LIVE: False}
GATE = tuple(t.take for t in GATE_TAKES)
#: The gate replays a variant must reproduce: energy gate off, and on (shipped).
BASE_REPLAYS = {False: "guard-only", True: "energy"}
#: The resting-hand stretch of live take 2 (hand probe): seq 452-596, one position
#: apart from the e4 knight laid on its side (truth segment 489-501).
REST_SEQS = range(452, 597)
#: Gaps in the recorded stream longer than this are the old 1 fps keepalive.
GAP_S = 0.25
FILL_DT = 0.1
KEEPALIVE_S = 2.0


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Take:
    name: str
    seq: np.ndarray
    ts: np.ndarray
    stab: np.ndarray
    placement: list[str]
    energy: np.ndarray
    direction: np.ndarray
    truth: SessionTruth

    def nearest(self, t: float) -> int:
        i = int(np.searchsorted(self.ts, t))
        if i == 0:
            return 0
        if i == len(self.ts):
            return i - 1
        return i if self.ts[i] - t < t - self.ts[i - 1] else i - 1


def load_take(cache: Path, name: str) -> Take:
    z = np.load(cache / f"{name}.npz")
    placement = [
        encode_placement([[CLASSES[int(c)] for c in row[r * 8 : r * 8 + 8]] for r in range(8)])
        for row in z["cls"]
    ]
    return Take(
        name=name,
        seq=z["seq"],
        ts=z["ts"],
        stab=z["stab_re"] if RESTAMP[name] else z["stab_hdr"],
        placement=placement,
        energy=z["energy"],
        direction=z["direction"],
        truth=load_truth(ROOT / "recordings" / name / "truth.json"),
    )


@dataclass(frozen=True)
class Sent:
    """One frame the node receives: the recorded row whose read it carries."""

    row: int
    ts: float
    stability: float
    synthetic: bool
    #: Set on a scheduled read that a burst end owed (not a keepalive read).
    burst_end: bool = False


# ------------------------------------------------------------------- the edge


def recorded(take: Take) -> list[Sent]:
    return [Sent(i, float(take.ts[i]), float(take.stab[i]), False) for i in range(len(take.ts))]


class _FixedMeter:
    """Stands in for the forwarder's MotionMeter: the frame's known stability."""

    value = 1.0

    def update(self, image: np.ndarray) -> float:
        return self.value


_PIXEL = np.zeros((1, 1, 3), np.uint8)


def _camera(take: Take, fill: bool) -> list[Sent]:
    out: list[Sent] = []
    for i in range(len(take.ts)):
        a = float(take.ts[i])
        out.append(Sent(i, a, float(take.stab[i]), False))
        if not fill or i + 1 == len(take.ts):
            continue
        b = float(take.ts[i + 1])
        if b - a <= GAP_S:
            continue
        t = a + FILL_DT
        while t < b - FILL_DT / 2 and t < a + 3.0:  # a tail lasts 1 s; 3 s is plenty
            out.append(Sent(take.nearest(t), t, 1.0, True))
            t += FILL_DT
    return out


def phone(take: Take, hold: float, *, fill: bool) -> list[Sent]:
    """What a phone running ``exit_hold_s=hold`` and the 1 s tail would have sent."""
    meter = _FixedMeter()
    fwd = AdaptiveForwarder(ForwarderConfig(exit_hold_s=hold), meter=meter)  # type: ignore[arg-type]
    sent: list[Sent] = []
    for n, f in enumerate(_camera(take, fill)):
        if f.synthetic:
            tail = fwd._tail_until is not None and f.ts < fwd._tail_until
            if not (fwd.bursting or tail):
                continue  # not a frame the phone would send densely
        meter.value = f.stability
        if fwd.decide(Frame(board_id="phone", seq=n, capture_ts=f.ts, image=_PIXEL)) is not None:
            sent.append(f)
    return sent


# ------------------------------------------------------------------- the node


@dataclass(frozen=True)
class Schedule:
    hold: float = EXIT_HOLD_S
    reads: int = 3
    spacing: float = 0.3

    def policy(self) -> BurstTriggered:
        return BurstTriggered(
            keepalive_s=KEEPALIVE_S,
            burst_end_reads=self.reads,
            spacing_s=self.spacing,
            exit_hold_s=self.hold,
        )


@dataclass(frozen=True)
class Settle:
    seconds: float = 0.5
    min_reads: int = 2
    obscured: bool = False
    #: The energy threshold when ``obscured`` (the shipped one unless swept).
    energy: float = OBSCURED_ENERGY


@dataclass(frozen=True)
class Event:
    row: int
    seq: int
    ts: float
    raw: str
    settled: str
    changed_ts: float
    obscured: bool
    synthetic: bool
    burst_end: bool


def scheduled(sent: list[Sent], schedule: Schedule) -> list[Sent]:
    policy = schedule.policy()
    out = []
    for f in sent:
        policy.observe(f.ts, f.stability)
        owed = policy._owed > 0
        if policy.due(f.ts, f.stability):
            out.append(replace(f, burst_end=owed))
    return out


def settle(take: Take, reads: list[Sent], cfg: Settle) -> list[Event]:
    config = SettleConfig(
        seconds=cfg.seconds,
        min_reads=cfg.min_reads,
        obscured_energy=cfg.energy if cfg.obscured else None,
    )
    stab = BoardStabilizer(config)
    guard = OrientationGuard()
    events: list[Event] = []
    for n, f in enumerate(reads):
        d = float(take.direction[f.row])
        # The guard resets on a seq that does not advance; synthetic frames reuse a
        # neighbour's seq, so it is handed the read's position in the stream.
        if guard.check(n, None if math.isnan(d) else d) is not None:
            continue
        raw = take.placement[f.row]
        obscured = config.obscures(float(take.energy[f.row]))
        settled = stab.observe(raw, f.ts, obscured=obscured)
        assert stab.changed_ts is not None
        events.append(
            Event(f.row, int(take.seq[f.row]), f.ts, raw, settled, stab.changed_ts,
                  obscured, f.synthetic, f.burst_end)
        )
    return events


# ------------------------------------------------------------------- scoring


@dataclass
class Score:
    scored: int = 0
    perfect: int = 0
    not_lag: int = 0
    leaks: list[tuple[str, int, tuple[str, ...]]] = field(default_factory=list)
    #: (take, boundary seq, seconds from the boundary frame to the settling read)
    latency: list[tuple[str, int, float]] = field(default_factory=list)
    boundaries: int = 0
    reads: int = 0
    synthetic_reads: int = 0
    minutes: float = 0.0
    frames_scored: int = 0
    frames_correct: int = 0
    rest_wrong: int = 0
    rest_events: int = 0
    #: Settled changes to a board that is not the current truth board: what a
    #: viewer sees flicker to (partial moves, misreads, hands), guard window included.
    wrong_changes: int = 0

    def add(self, other: Score) -> Score:
        for k, v in vars(other).items():
            setattr(self, k, getattr(self, k) + v)
        return self


def score(take: Take, events: list[Event]) -> Score:
    truth = take.truth
    s = Score(reads=len(events), synthetic_reads=sum(e.synthetic for e in events),
              minutes=_active_minutes(take))
    grids = [np.array(parse_placement(t.placement)) for t in truth.segments]
    for prev, e in itertools.pairwise(events):
        i = truth.index_for(e.seq)
        if i >= 0 and e.settled != prev.settled and e.settled != truth.segments[i].placement:
            s.wrong_changes += 1
    for e in events:
        i = truth.index_for(e.seq)
        if i < 0:
            continue
        if not truth.in_guard(e.seq):
            s.scored += 1
            if e.settled == truth.segments[i].placement:
                s.perfect += 1
            elif i == 0 or e.settled != truth.segments[i - 1].placement:
                s.not_lag += 1
        # A hand leak: a settled square holding a class that no earlier, current
        # or next truth board has there (docs/reports/2026-11-24-latency-round2.md).
        g = np.array(parse_placement(e.settled))
        bad = ~np.any(np.stack([g == a for a in grids[: i + 2]]), axis=0)
        if bad.any():
            names = tuple(f"{'abcdefgh'[c]}{8 - r}" for r, c in zip(*np.nonzero(bad), strict=True))
            s.leaks.append((take.name, e.seq, names))
    if take.name == LIVE:
        ok = {truth.segments[truth.index_for(q)].placement for q in REST_SEQS}
        rest = [e for e in events if e.seq in REST_SEQS]
        s.rest_events = len(rest)
        s.rest_wrong = sum(e.settled not in ok for e in rest)
    # Move -> settle, matched as docs/reports/2026-11-24-changed-utc.md matches.
    cap = dict(zip(take.seq.tolist(), take.ts.tolist(), strict=True))
    starts = truth.starts
    for k in range(1, len(starts)):
        lo, b = starts[k - 1], starts[k]
        hi = starts[k + 1] if k + 1 < len(starts) else 1 << 60
        if b not in cap:
            continue
        s.boundaries += 1
        want = truth.segments[k].placement
        hit = next(
            (e for e in events
             if lo <= e.seq < hi and e.settled == want and e.changed_ts > cap.get(lo, -1e18)),
            None,
        )
        if hit is not None:
            s.latency.append((take.name, b, hit.ts - cap[b]))
    # The settled board in force at every recorded frame (outside the guard).
    ev_ts = np.array([e.ts for e in events])
    for q, t in zip(take.seq.tolist(), take.ts.tolist(), strict=True):
        i = truth.index_for(q)
        j = int(np.searchsorted(ev_ts, t, side="right")) - 1
        if i < 0 or truth.in_guard(q) or j < 0:
            continue
        s.frames_scored += 1
        s.frames_correct += events[j].settled == truth.segments[i].placement
    return s


def _active_minutes(take: Take) -> float:
    """Session time, without gaps over 60 s (live take 2 idles 1.8 h at its end)."""
    d = np.diff(take.ts)
    return float(d[d < 60].sum() / 60)


# ------------------------------------------------------------------- runs


_CACHE: dict = {}


def run(take: Take, stream: str, schedule: Schedule, cfg: Settle) -> list[Event]:
    key = (take.name, stream, schedule)
    if key not in _CACHE:
        if stream == "recorded":
            sent = recorded(take)
        else:
            sent = phone(take, schedule.hold, fill=stream == "phone")
        _CACHE[key] = scheduled(sent, schedule)
    return settle(take, _CACHE[key], cfg)


def pooled(takes: list[Take], stream: str, schedule: Schedule, cfg: Settle) -> Score:
    total = Score()
    for t in takes:
        total.add(score(t, run(t, stream, schedule, cfg)))
    return total


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else math.nan


# ------------------------------------------------------------------- commands


def validate(cache: Path) -> None:
    """The recorded stream at shipped settings must equal the gate replays read for
    read: ``guard-only`` with the energy gate off, ``energy`` with it on."""
    takes = [load_take(cache, t) for t in GATE]
    base = Schedule()
    for obscured, replay_name in BASE_REPLAYS.items():
        total = Score()
        for t in takes:
            evs = run(t, "recorded", base, Settle(obscured=obscured))
            replay_dir = ROOT / "recordings" / t.name / "replays" / replay_name
            real = read_events(Recording(replay_dir), board_id=t.truth.board_id)
            lines = (replay_dir / "events.ndjson").read_text().splitlines()
            flags = [bool(json.loads(x).get("obscured")) for x in lines]
            want = [(r.frame_seq, r.settled, f) for r, f in zip(real, flags, strict=True)]
            got = [(e.seq, e.settled, e.obscured) for e in evs]
            same = sum(a == b for a, b in zip(want, got, strict=False))
            sc = score(t, evs)
            total.add(sc)
            print(f"{replay_name} {t.name}: sim {len(got)} reads, replay {len(want)}, "
                  f"(seq, settled, obscured) equal on {same}; settled perfect "
                  f"{sc.perfect}/{sc.scored}, hand leaks {len(sc.leaks)}")
        obsc = sum(e.obscured for t in takes for e in run(t, "recorded", base,
                                                           Settle(obscured=obscured)))
        print(f"{replay_name} total: settled perfect {total.perfect}/{total.scored}, hand "
              f"leaks {len(total.leaks)}, not-lag wrong {total.not_lag}, obscured reads {obsc}")
    live = load_take(cache, LIVE)
    for obscured in (False, True):
        evs = run(live, "recorded", base, Settle(obscured=obscured))
        sc = score(live, evs)
        print(f"{LIVE} recorded, obscured={obscured}: {len(evs)} reads, resting-hand "
              f"stretch {sc.rest_wrong}/{sc.rest_events} wrong, settled perfect "
              f"{sc.perfect}/{sc.scored}")
    path = ROOT / "recordings" / LIVE / "events.ndjson"
    live_seqs = {json.loads(x)["frame_seq"] for x in path.read_text().splitlines()}
    sim_seqs = {e.seq for e in run(live, "recorded", base, Settle())}
    print(f"{LIVE}: the live node read {len(live_seqs)} frames, the sim {len(sim_seqs)}, "
          f"{len(live_seqs & sim_seqs)} in common")
    for t in [*takes, live]:
        sent = phone(t, EXIT_HOLD_S, fill=True)
        print(f"  phone(hold 0.7) on {t.name}: sends {len(sent)} frames of "
              f"{len(t.ts)} recorded, {sum(f.synthetic for f in sent)} of them synthetic")


HOLDS = (0.7, 0.5, 0.3, 0.1, 0.0)
READS = (3, 2)
SPACINGS = (0.3, 0.15)
SECONDS = (0.5, 0.3, 0.15, 0.0)
MINS = (2, 1)
def row(takes: list[Take], live: Take, stream: str, sch: Schedule, cfg: Settle) -> dict:
    """One variant's line of the sweep: the gate takes pooled, then live take 2."""
    g = pooled(takes, stream, sch, cfg)
    lv = score(live, run(live, stream, sch, cfg))
    gl = [x[2] for x in g.latency]
    ll = [x[2] for x in lv.latency]
    return {
        "stream": stream,
        "hold": sch.hold,
        "reads": sch.reads,
        "spacing": sch.spacing,
        "settle_s": cfg.seconds,
        "min_reads": cfg.min_reads,
        "obscured": int(cfg.obscured),
        "energy": cfg.energy if cfg.obscured else "off",
        "gate_perfect": g.perfect,
        "gate_scored": g.scored,
        "gate_rate": round(g.perfect / g.scored, 4),
        "gate_frames_ok": round(g.frames_correct / g.frames_scored, 4),
        "gate_leaks": len(g.leaks),
        "gate_not_lag": g.not_lag,
        "gate_wrong_changes": g.wrong_changes,
        "gate_matched": len(gl),
        "gate_boundaries": g.boundaries,
        "gate_med": round(median(gl), 2),
        "gate_p90": round(pct(gl, 90), 2),
        "gate_reads_per_min": round(g.reads / g.minutes, 1),
        "gate_synthetic_reads": g.synthetic_reads,
        "live_perfect": lv.perfect,
        "live_scored": lv.scored,
        "live_leaks": len(lv.leaks),
        "live_wrong_changes": lv.wrong_changes,
        "live_rest_wrong": lv.rest_wrong,
        "live_rest_events": lv.rest_events,
        "live_matched": len(ll),
        "live_med": round(median(ll), 2) if ll else math.nan,
        "live_p90": round(pct(ll, 90), 2),
        "live_reads_per_min": round(lv.reads / lv.minutes, 1),
        "all_med": round(median(gl + ll), 2),
        "all_p90": round(pct(gl + ll, 90), 2),
    }


def sweep(cache: Path, streams: tuple[str, ...] = ("phone", "phone-nofill")) -> None:
    takes = [load_take(cache, t) for t in GATE]
    live = load_take(cache, LIVE)
    first = row(takes, live, "recorded", Schedule(), Settle())
    print("\t".join(first))
    print("\t".join(str(v) for v in first.values()))
    print("\t".join(str(v) for v in
                    row(takes, live, "recorded", Schedule(), Settle(obscured=True)).values()))
    for stream in streams:
        for hold, reads, spacing, secs, mins, obsc in itertools.product(
            HOLDS, READS, SPACINGS, SECONDS, MINS, (False, True)
        ):
            if secs > 0 and mins == 1:
                continue  # one read never spans a window > 0: same as min_reads=2
            r = row(takes, live, stream, Schedule(hold, reads, spacing),
                    Settle(secs, mins, obsc))
            print("\t".join(str(v) for v in r.values()), flush=True)


#: The variants the report shortlists: (hold, reads, spacing, settle s, min reads).
SHORTLIST = (
    (0.7, 3, 0.3, 0.5, 2),  # shipped
    (0.7, 3, 0.3, 0.3, 2),
    (0.5, 3, 0.3, 0.3, 2),
    (0.3, 3, 0.3, 0.5, 2),
    (0.3, 3, 0.3, 0.3, 2),
    (0.3, 3, 0.3, 0.0, 2),
    (0.3, 2, 0.3, 0.3, 2),
    (0.7, 3, 0.3, 0.0, 1),
    (0.3, 3, 0.3, 0.0, 1),
)
#: The energy thresholds the robustness check spans: the hand probe's
#: leave-one-take-out range (-12.13 to -12.93) around the shipped -12.4.
THRESHOLDS = (-12.1, OBSCURED_ENERGY, -12.9)


def robust(cache: Path) -> None:
    """The shortlist under both edge streams and the threshold range."""
    takes = [load_take(cache, t) for t in GATE]
    live = load_take(cache, LIVE)
    header = False
    for stream in ("phone", "phone-nofill"):
        for energy in THRESHOLDS:
            for hold, reads, spacing, secs, mins in SHORTLIST:
                r = row(takes, live, stream, Schedule(hold, reads, spacing),
                        Settle(secs, mins, obscured=True, energy=energy))
                if not header:
                    print("\t".join(r))
                    header = True
                print("\t".join(str(v) for v in r.values()), flush=True)


def _variant(spec: str) -> tuple[Schedule, Settle]:
    """``hold,reads,spacing,seconds,min_reads[,energy|off]`` -> the variant."""
    hold, reads, spacing, secs, mins, *rest = spec.split(",")
    energy = rest[0] if rest else str(OBSCURED_ENERGY)
    cfg = (Settle(float(secs), int(mins)) if energy == "off"
           else Settle(float(secs), int(mins), obscured=True, energy=float(energy)))
    return Schedule(float(hold), int(reads), float(spacing)), cfg


def compare(cache: Path, base: str, cand: str, stream: str = "phone") -> None:
    """Two variants per take, and the move -> settle time paired per boundary."""
    (bs, bc), (cs, cc) = _variant(base), _variant(cand)
    print("| take | settled perfect | hand leaks | wrong changes | median move→settle (s) |")
    print("|---|---|---|---|---|")
    deltas = []
    for name in [*GATE, LIVE]:
        t = load_take(cache, name)
        a, b = score(t, run(t, stream, bs, bc)), score(t, run(t, stream, cs, cc))
        la = {x[1]: x[2] for x in a.latency}
        lb = {x[1]: x[2] for x in b.latency}
        deltas += [(name, k, lb[k] - la[k]) for k in la.keys() & lb.keys()]
        def med(xs: dict[int, float]) -> str:
            return f"{median(xs.values()):.2f}" if xs else "—"
        print(f"| {name} | {a.perfect}/{a.scored} → {b.perfect}/{b.scored} | "
              f"{len(a.leaks)} → {len(b.leaks)} | {a.wrong_changes} → {b.wrong_changes} | "
              f"{med(la)} → {med(lb)} |")
    for label, sel in (("gate", lambda n: n != LIVE), ("take 2", lambda n: n == LIVE)):
        d = [x[2] for x in deltas if sel(x[0])]
        print(f"{label}: {len(d)} boundaries settled under both, paired change median "
              f"{median(d):+.2f} s, faster on {sum(x < -0.05 for x in d)}, slower on "
              f"{sum(x > 0.05 for x in d)}")


def show(cache: Path, args: list[str]) -> None:
    """Per-boundary latency and every leak / resting-hand wrong board for one variant."""
    hold, reads, spacing, secs, mins, obsc = args[:6]
    stream = args[6] if len(args) > 6 else "phone"
    sch = Schedule(float(hold), int(reads), float(spacing))
    cfg = Settle(float(secs), int(mins), obsc in ("1", "true", "True"))
    for name in [*GATE, LIVE]:
        t = load_take(cache, name)
        evs = run(t, stream, sch, cfg)
        sc = score(t, evs)
        lat = " ".join(f"{b}:{d:+.2f}" for _, b, d in sc.latency)
        print(f"{name}: reads {sc.reads} (synthetic {sc.synthetic_reads}), perfect "
              f"{sc.perfect}/{sc.scored}, matched {len(sc.latency)}/{sc.boundaries}  {lat}")
        for leak in sc.leaks:
            print("   leak", leak)
        if name == LIVE:
            ok = {t.truth.segments[t.truth.index_for(q)].placement for q in REST_SEQS}
            for e in evs:
                if e.seq in REST_SEQS and e.settled not in ok:
                    print(f"   rest-wrong seq {e.seq} obscured={e.obscured} {e.settled}")


def explain(cache: Path, args: list[str]) -> None:
    """Where a variant's move -> settle time goes, per matched boundary.

    ``first`` is the first read at or after the boundary frame whose raw board is
    the new truth board, ``settle`` the read that settled it (the latency). The
    remainder, ``settle - first``, is the debounce. ``kind`` says whether that
    first read was owed by a burst end or was a keepalive read.
    """
    hold, reads, spacing, secs, mins, obsc = args[:6]
    stream = args[6] if len(args) > 6 else "phone"
    sch = Schedule(float(hold), int(reads), float(spacing))
    cfg = Settle(float(secs), int(mins), obsc in ("1", "true", "True"))
    parts: dict[str, list[tuple[float, float]]] = {"burst-end": [], "keepalive": [], "none": []}
    for name in [*GATE, LIVE]:
        t = load_take(cache, name)
        evs = run(t, stream, sch, cfg)
        cap = dict(zip(t.seq.tolist(), t.ts.tolist(), strict=True))
        for _, b, lat in score(t, evs).latency:
            k = t.truth.starts.index(b)
            want = t.truth.segments[k].placement
            first = next((e for e in evs if e.ts >= cap[b] and e.raw == want), None)
            kind = "none" if first is None else ("burst-end" if first.burst_end else "keepalive")
            to_first = math.nan if first is None else first.ts - cap[b]
            parts[kind].append((to_first, lat))
            print(f"{name} boundary {b}: settle {lat:+.2f} s, first correct read "
                  f"{to_first:+.2f} s ({kind})")
    for kind, xs in parts.items():
        if xs:
            print(f"{kind}: {len(xs)} boundaries, median to first correct read "
                  f"{median(x[0] for x in xs):.2f} s, median to settle "
                  f"{median(x[1] for x in xs):.2f} s")


if __name__ == "__main__":
    cache_dir, cmd, *rest = sys.argv[1:]
    if cmd == "validate":
        validate(Path(cache_dir))
    elif cmd == "sweep":
        sweep(Path(cache_dir))
    elif cmd == "compare":
        compare(Path(cache_dir), *rest)
    elif cmd == "robust":
        robust(Path(cache_dir))
    elif cmd == "explain":
        explain(Path(cache_dir), rest)
    elif cmd == "show":
        show(Path(cache_dir), rest)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
