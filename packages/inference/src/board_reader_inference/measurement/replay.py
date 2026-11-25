"""Re-run a recorded take through the live stack — iteration without the board.

A trial is a person sitting at a real board with a phone, and it is the only
honest test this project has. It is also the slowest: every code change used to
need another take. But every served run already keeps its own input
(:mod:`board_reader_core.recording`: ``frames.ndjson`` + the JPEGs) and its own
output (``events.ndjson``, scored by :mod:`.trial`), so the take is repeatable —
what was missing is the middle, the thing that pushes the recorded *frames*
through today's recognizer and writes fresh events. That is this module.

    board-reader-inference replay recordings/20260813-193308 --score

**It replays the live path, not a copy of it.** The frames go through the same
:func:`~board_reader_inference.recognition.recognizer.frame_reader` read the ``/ingest``
worker runs, the same
:func:`~board_reader_inference.stream.ingest._settle_read` (guard ->
:class:`~board_reader_inference.stream.stabilize.BoardStabilizer` -> change clock), and the same
:func:`~board_reader_inference.stream.ingest.event_line` writer. A replayed event is
what the gateway *would* have published; a second implementation of any of those
three would make the comparison meaningless.

**The clock is the recording's, never the wall's.** Everything time-based here is
driven by each frame's ``capture_ts``: the estimator already stamps ``utc`` from
it, the stabilizer takes its ``ts`` as an argument, and the one remaining wall
clock — the ``published_ts`` the gateway stamps at publish — is *omitted* rather
than faked (see :func:`~board_reader_inference.stream.ingest.event_line`). So a settle
window is 2.5 recorded seconds whether the replay takes ten minutes or ten
milliseconds, and the same recording through the same code and the same
environment produces byte-identical events. That determinism is the point: it is
what lets two replays be diffed as a claim about the *code*.

**What replay does NOT reproduce, 1: the live pre-filter is a QUALITY filter.**
The gateway's latest-wins inbox drops frames whenever the estimator falls behind
(the 08-13 trial published 85 events from 748 frames), and what survives is not
merely the freshest frame — :func:`~board_reader_inference.stream.ingest.prefer_frame`
keeps the *sharper* of two contenders within a one-second window, because a
handheld phone smears half of what it captures. A live run's recorded score is
therefore measured on a blur-filtered subset of its own frames, while replay
reads every frame the policy asks for, blurry ones included.

That has one consequence worth stating in capitals, because it is the easiest
way to misread this harness: **a replay-vs-RECORDED raw delta is not a code
result.** The two numbers are computed over different frame populations and the
replayed one is systematically pessimistic — it pays for the blurred frames the
live pre-filter spent nothing on. Quote it as directional at most. The
comparison that means something is **replay vs replay**: two replays of the same
recording see the same frames in the same order, so a difference between them is
a difference in the code or the models, which is the entire point.

The same split applies to the derived rows: the timing block of a replayed report
is not the live one (it reads a denser stream), and the settled row settles off a
different read rate. The debounce is time-windowed rather than count-based
precisely so that stays as comparable as it can be (see :mod:`.stabilize`) — but
"as comparable as it can be" is not "the same measurement".

**Output layout.** A replay is written as a *recording-shaped* directory so
``eval-session`` scores it unchanged::

    recordings/<source>/replays/<name>/manifest.json   # provenance: source, env, git
                                    /events.ndjson    # the fresh reads
                                    /frames.ndjson    # thin copy: the replayed lines
                                    /jpeg -> ../../jpeg      (symlink, never copied)
                                    /truth.json -> ../../truth.json  (if the source has one)

``<name>`` defaults to the code state plus a digest of the model environment
(``<git-describe>-<env-digest>``), so re-running the same code against the same models
overwrites its own directory while a different model lands beside it. The default
is only offered when the working tree is **clean** and the models it names have
not changed on disk: an uncommitted tree has no code identity, and two different
experiments would replay into one directory and destroy the baseline they were
meant to be diffed against. A dirty tree is asked for ``--name``. The source
recording is never written to, apart from that ``replays/`` subdirectory.

**Multi-session recordings.** A recording directory can hold several streaming
sessions (a reconnecting edge restarts ``seq`` at 0 — a known recorder wart), and
truth is a step function over ``seq``, so replaying two of them into one
``events.ndjson`` would score one session's reads against the other's truth.
Sessions are therefore detected (``seq`` or ``capture_ts`` going backwards within
one board) and only the **longest** is replayed, with a warning naming the rest;
``--all-sessions`` replays them all into one file for the caller who knows better.

**The frame-selection seam.** The node does not have to read every frame that
reaches it, and should not: :mod:`.schedule` holds the policies that decide
(estimate at burst end, slow keepalive, skip mid-burst), and ``--policy`` picks
one. The loop asks it per frame, in order; :class:`~.schedule.EveryFrame` is the
default and the control. A policy sees the recorded header
(:class:`~board_reader_core.recording.FrameEntry`: ``seq``, ``capture_ts``,
``stability``, ``sharpness``, ``burst_id``) and not the pixels, which is both what
such a scheduler needs and what lets a declined frame cost no recognizer read. Its
JPEG is still decoded, because every frame's grey copy goes to the board's
:class:`~board_reader_inference.stream.backdate.FrameRing`, as it does live.

A policy decides from the header alone, so "estimate this one" is a *request*,
not a guarantee: the frame it picked may still turn out to have no JPEG behind it
(a half-copied recording). Those land in their own counter rather than being
charged to the policy — a scheduler is judged on the frames it declined, not on
the ones the disk could not honour.

**``--restamp-stability``, and why a scheduling sweep needs it.** Those policies
read the header's ``stability``, and on every take recorded before 2026-08-27
evening that field is a *broken sensor's* output: the old motion meter scored an
absolute pixel difference, the deployment phone's resting difference measured
about six times the constant it assumed, and the field sat under 0.35 from the
first frame to the last. Scheduling against it would measure the bug.
``--restamp-stability`` runs today's meter (:class:`~.schedule.MotionMeter`, the
edge's algorithm ported — see there for why it is not imported) over the
recording's own JPEGs in seq order and overwrites the field before any policy
sees it. Deterministic, and it is exactly what a phone streaming that scene today
would have sent. It costs one JPEG decode per frame *including the declined ones*
— the meter must see every frame or its floor is learned from a subsample — so a
sparse policy still pays for the pixels it does not read. Restamp uniformly
across a sweep or not at all: recorded and recomputed values are different
scales, and a table mixing them compares nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, get_args

import numpy as np

from board_reader_core.control import DEFAULT_RULES, Rules
from board_reader_core.recording import (
    EVENTS_NAME,
    FORMAT,
    INDEX_NAME,
    JPEG_DIR,
    MANIFEST_NAME,
    FrameEntry,
    RecordedFrame,
    Recording,
)
from board_reader_core.wire import decode_jpeg

from board_reader_inference.measurement.trial import TRUTH_NAME
from board_reader_inference.recognition.recognizer import (
    FrameRead,
    build_from_env,
    frame_reader,
)
from board_reader_inference.stream.backdate import ChangeClock, FrameRing
from board_reader_inference.stream.ingest import (
    Misoriented,
    OrientationGuard,
    _settle_read,
    event_line,
)
from board_reader_inference.stream.schedule import (
    EveryFrame,
    FramePolicy,
    MotionMeter,
    parse_policy,
    policy_slug,
    restamped,
)
from board_reader_inference.stream.stabilize import (
    DEFAULT_SETTLE,
    BoardStabilizer,
    SettleConfig,
)

#: Where replays of a recording live, inside the recording they came from — a
#: benchmark that cannot be traced back to its input is not a benchmark.
REPLAYS_DIR = "replays"

#: The environment that decides which estimator gets built (``server``'s builder
#: reads exactly these). Recorded in the manifest and digested into the default
#: replay name, so two model configurations never overwrite each other's scores.
CONFIG_ENV = ("BR_CHESSQ",)

#: Frames between progress lines on stderr. A real replay is minutes of GPU work
#: and silence for minutes is indistinguishable from a hang.
PROGRESS_EVERY = 25


# --------------------------------------------------------------------------- #
# Sessions inside one recording
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StreamSession:
    """One contiguous streaming session inside a recording, as an index range.

    Positions are into the recording's index *for one board*, not into the file,
    and no header is retained: a long session is O(100k) frames and this has to
    stay cheap enough to run before deciding what to replay.
    """

    start: int  # first position (inclusive) in the board's index
    stop: int  # last position (exclusive)
    first_seq: int
    last_seq: int
    first_ts: float
    last_ts: float

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def seconds(self) -> float:
        return self.last_ts - self.first_ts

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "seconds": round(self.seconds, 3),
        }


def find_sessions(entries: Iterator[FrameEntry] | list[FrameEntry]) -> list[StreamSession]:
    """Split one board's index into contiguous streaming sessions.

    A reconnecting edge restarts its ``seq`` at 0 and its ``capture_ts`` can jump
    backwards with it, so a recording directory is not necessarily one stream.
    Either going backwards starts a new session — the recorder itself keeps no
    session marker, and this is the only signal there is.
    """
    sessions: list[StreamSession] = []
    start = 0
    first_seq = last_seq = 0
    first_ts = last_ts = 0.0
    count = 0
    for i, entry in enumerate(entries):
        if count and (entry.seq <= last_seq or entry.capture_ts < last_ts):
            sessions.append(
                StreamSession(start, i, first_seq, last_seq, first_ts, last_ts)
            )
            start, count = i, 0
        if not count:
            first_seq, first_ts = entry.seq, entry.capture_ts
        last_seq, last_ts = entry.seq, entry.capture_ts
        count += 1
    if count:
        sessions.append(
            StreamSession(start, start + count, first_seq, last_seq, first_ts, last_ts)
        )
    return sessions


def longest_session(sessions: list[StreamSession]) -> StreamSession:
    """The session with the most frames; the later one on a tie.

    "Longest" rather than "last" because the wart it works around is a stream
    that dropped and reconnected: the fragment is the accident, the long run is
    the take. A tie goes to the later one — the operator's last attempt.
    """
    return max(sessions, key=lambda s: (s.frames, s.start))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def git_describe() -> str:
    """The working tree's code state, or ``"unversioned"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return "unversioned"
    described = out.stdout.strip()
    return described.replace("/", "-") if out.returncode == 0 and described else "unversioned"


def config_env() -> dict[str, dict]:
    """The estimator-shaping environment, with each named file's size and mtime.

    The values decide *which* estimator gets built; the stat says which bytes it
    was built from, which is the question asked six weeks later when a replay
    does not reproduce ("was the model the same file?"). Both feed
    :func:`config_digest`, deliberately: models are retrained **in place** here
    (a model file is a path, not a version), and a digest over the
    paths alone would let a retrain silently overwrite its predecessor's score
    with a different model's. The cost is a fresh directory after a no-op copy
    that only moved an mtime, which is the cheap half of the trade.
    """
    out: dict[str, dict] = {}
    for name in CONFIG_ENV:
        value = os.environ.get(name)
        if value is None:
            continue
        entry: dict = {"value": value}
        try:
            stat = Path(value).stat()
            entry["bytes"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
        except OSError:
            pass  # not a path, or gone — the value alone is still provenance
        out[name] = entry
    return out


def config_digest(env: dict[str, dict], settle: SettleConfig = DEFAULT_SETTLE) -> str:
    """A short stable digest of the estimator environment (8 hex chars).

    Over the values *and* the stat of each file they name — see
    :func:`config_env` for why the bytes on disk are part of the identity — and
    over the debounce's knobs when they are not the defaults, so a digest of a
    default-settle replay is the same as before the knobs existed.
    """
    identity: dict[str, object] = dict(sorted(env.items()))
    if not settle.is_default:
        identity["settle"] = settle.to_dict()
    payload = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def default_name(
    env: dict[str, dict] | None = None,
    *,
    policy: FramePolicy | None = None,
    restamp: bool = False,
    settle: SettleConfig = DEFAULT_SETTLE,
    rules: Rules = DEFAULT_RULES,
) -> str:
    """The replay directory name for the current code and model environment.

    Reproducible on purpose: the same code against the same models replays into
    the same directory (a re-run supersedes its own previous answer), while a
    different model lands beside it instead of silently overwriting a score.

    Which is exactly why there is **no default for an uncommitted tree**. Two
    different sets of uncommitted changes describe identically, so they would
    replay into one directory and the second run would destroy the baseline the
    first one was meant to be diffed against — the one thing this harness exists
    to protect. A dirty (or unversioned) tree raises, naming ``--name``; the
    caller then says what they are measuring, which is better provenance than a
    hash could have been anyway.

    ``policy``, ``restamp``, ``settle`` and ``rules`` join the name when they are
    not the defaults, for the same reason the model stat does: they change what the run
    measures, and a sweep whose rows all land in one directory has measured its
    last row only.
    """
    described = git_describe()
    if described == "unversioned" or described.endswith("-dirty"):
        raise ValueError(
            f"the working tree has no reproducible code identity ({described}), so "
            "there is no safe default replay name: two different sets of "
            "uncommitted changes would replay into the same directory and "
            "overwrite each other's score. Pass --name <what you are measuring> "
            "(or commit first)"
        )
    name = f"{described}-{config_digest(env if env is not None else config_env(), settle)}"
    # How the frames were CHOSEN is part of what a replay measures, so it is part
    # of the identity: without this a policy sweep on a clean tree would replay
    # every row into one directory and keep only the last.
    if policy is not None and not isinstance(policy, EveryFrame):
        name += f"-{policy_slug(policy)}"
    if restamp:
        name += "-restamped"
    if not settle.is_default:
        name += f"-{settle.slug}"
    if rules != DEFAULT_RULES:
        name += f"-rules-{rules}"
    return name


def check_name(name: str) -> str:
    """A replay directory name that is one path component. Raises on anything else.

    The replay's ``jpeg`` and ``truth.json`` are symlinks with a **fixed** ``../..``
    hop back to the recording (:func:`_link_source`), so a name carrying a
    separator or a ``..`` would not just escape the tree — it would produce
    dangling links and a replay that scores nothing, discovered ten minutes into
    a GPU run.
    """
    if not name or name in {".", ".."} or name != Path(name).name:
        raise ValueError(
            f"--name {name!r} must be a single directory name: no '/', no '..' "
            "(a replay's jpeg/truth symlinks hop a fixed two levels back to the "
            "recording)"
        )
    return name


# --------------------------------------------------------------------------- #
# Running one
# --------------------------------------------------------------------------- #


@dataclass
class ReplayStats:
    """What the replay did with the frames it was given."""

    frames_total: int = 0  # in the replayed session(s), before the policy
    frames_estimated: int = 0
    frames_skipped_policy: int = 0
    #: ``--limit`` stopped the replay early: the frames after it were never
    #: looked at, so this is a shorter TAKE, not a sparser reading of a long one.
    truncated: bool = False
    frames_failed: int = 0  # the estimator raised; the stream goes on, as it does live
    #: Read, then dropped by the orientation guard (``ingest.OrientationGuard``),
    #: exactly as the live worker drops it.
    frames_misoriented: int = 0
    frames_missing: int = 0  # index line points at a JPEG that is gone or undecodable
    #: Frames the motion meter re-scored under ``--restamp-stability``. Counted
    #: over the whole session, not over the reads: the meter has to see every
    #: frame or the floor it learns is a floor of a subsample.
    frames_restamped: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_estimated": self.frames_estimated,
            "frames_skipped_policy": self.frames_skipped_policy,
            "truncated": self.truncated,
            "frames_failed": self.frames_failed,
            "frames_misoriented": self.frames_misoriented,
            "frames_missing": self.frames_missing,
            "frames_restamped": self.frames_restamped,
            "wall_seconds": round(self.wall_seconds, 2),
        }


@dataclass
class ReplayResult:
    """One finished replay: where it landed, and what it did."""

    path: Path
    source: Path
    board_id: str
    stats: ReplayStats
    sessions: list[StreamSession]
    replayed: list[int] = field(default_factory=list)  # indices into ``sessions``
    #: The policy's spec string — what chose these frames, in ``--policy`` form.
    policy: str = EveryFrame().name
    #: Whether ``--restamp-stability`` was ASKED FOR. Deliberately the flag and
    #: not ``stats.frames_restamped > 0``: a run that restamped nothing (a
    #: ``--limit 0``, a take whose frames all vanished) is still a run whose
    #: directory is named ``-restamped``, and a manifest that infers "no" there
    #: contradicts its own path.
    restamp: bool = False
    #: The debounce the settled stream ran with.
    settle: SettleConfig = DEFAULT_SETTLE
    #: Which chess the settle guard assumed (``--rules``).
    rules: Rules = DEFAULT_RULES


def replay(
    recording: Recording,
    out_dir: str | Path,
    *,
    read: FrameRead,
    reader: str,
    board_id: str | None = None,
    policy: FramePolicy | None = None,
    restamp: bool = False,
    settle: SettleConfig = DEFAULT_SETTLE,
    rules: Rules = DEFAULT_RULES,
    all_sessions: bool = False,
    limit: int | None = None,
    log: TextIO | None = sys.stderr,
) -> ReplayResult:
    """Push ``recording``'s frames through ``read`` and write a replay directory.

    ``reader`` names what reads (the recognizer's class), for the manifest;
    ``settle`` configures the debounce (``serve`` reads the same knobs from
    ``BR_SETTLE_*``); ``rules`` is the board's setting for every read, where the
    live node reads it from ``/control`` per read. The source's own
    ``events.ndjson`` says which setting the live run used; a replay under
    another one warns, since its settled row then measures a different guard.

    Each read goes through :func:`~board_reader_inference.stream.ingest._settle_read`
    — the ingest worker's own body — and the stabilizer is fed the frame's
    ``capture_ts``, so the settle window is recorded time. One stabilizer for the
    whole replay: it is one board's stream, in order, which is exactly the stream
    a debounce needs. Every frame, read or not, is decoded and pushed to the
    board's :class:`~.backdate.FrameRing` first, as ``/ingest`` does live, so
    ``changed_utc`` is back-dated over the same frames (a live node behind on its
    reads holds a few newer frames too, which the back-dating never looks at).

    With ``restamp``, each replayed session's headers get their ``stability``
    rewritten by a fresh :class:`~.schedule.MotionMeter` before the policy is
    asked — one meter per session, because a reconnect is a new camera and a
    learned resting floor does not carry across one. Nothing but the policy reads
    the field (the estimators only copy it onto the published event), so a
    restamped every-frame replay reads the same pixels in the same order as an
    unrestamped one; it is the control a scheduling sweep is diffed against.

    **A fresh policy per session too, for the same reason the meter is fresh.** A
    :class:`~.schedule.FramePolicy` is stateful over one stream — mid-burst,
    reads owed by the last burst end, the keepalive anchor — and a seq restart is
    a different stream. Carried across, session 2 opens *inside* session 1's
    burst and manufactures a move-complete read at its very first calm frame, on
    a board nobody moved. Each replayed session therefore gets its own instance,
    rebuilt from the policy's spec string (:func:`~.schedule.parse_policy` round
    trips :attr:`~.schedule.FramePolicy.name` by construction). A caller's own
    policy object cannot be rebuilt that way, so replaying several sessions with
    one is refused rather than silently leaked across the seam.
    """
    policy = policy or EveryFrame()
    board = board_id or _default_board(recording)
    index = list(recording.frame_index(board_id=board))
    if not index:
        raise ValueError(f"no frames for board {board!r} in {recording.path}")
    sessions = find_sessions(index)
    if all_sessions:
        chosen = list(range(len(sessions)))
    else:
        chosen = [sessions.index(longest_session(sessions))]
        if len(sessions) > 1:
            _say(
                log,
                f"warning: {recording.path} holds {len(sessions)} streaming sessions "
                f"(seq restarts); replaying the longest "
                f"({sessions[chosen[0]].frames} frames, seq "
                f"{sessions[chosen[0]].first_seq}-{sessions[chosen[0]].last_seq}). "
                "Pass --all-sessions to replay them all into one file.",
            )

    # Eagerly, before a directory is made: a run that cannot give every session
    # its own policy must fail with nothing written, not with a half-open replay.
    policies = _per_session_policies(policy, len(chosen))
    out = _checked_out_dir(out_dir, recording)
    out.mkdir(parents=True, exist_ok=True)
    stabilizer = BoardStabilizer(settle)
    # One for the replay, like the stabilizer: it starts afresh by itself on the
    # seq restart that separates two sessions.
    guard = OrientationGuard()
    # The board's held frames and its change clock, one each like the guard: both
    # start afresh by themselves on a seq restart.
    ring = FrameRing()
    clock = ChangeClock()
    stats = ReplayStats()
    recorded = recorded_rules(recording)
    if recorded - {rules}:
        _say(
            log,
            f"warning: the live run settled under {sorted(recorded)} and this replay "
            f"under {rules!r}; pass --rules to reproduce it",
        )
    decode = _decoder(recording)
    total = sum(sessions[i].frames for i in chosen)
    if limit is not None:
        total = min(total, limit)
    started = time.monotonic()

    events_path = out / EVENTS_NAME
    frames_path = out / INDEX_NAME
    with (
        events_path.open("w", encoding="utf-8") as events,
        frames_path.open("w", encoding="utf-8") as frames,
    ):
        for nth, i in enumerate(chosen):
            # One meter AND one policy per streaming session: a reconnect is a new
            # camera, and neither a learned resting floor nor a half-finished burst
            # carries across one.
            meter = MotionMeter()
            session_policy = policies[nth]
            for entry in index[sessions[i].start : sessions[i].stop]:
                if limit is not None and stats.frames_estimated >= limit:
                    # Truncation, not selection: the replay stops looking at the
                    # recording here. Indexing the rest would tell `eval-session`
                    # this take ran for 748 frames at one read per 249 of them,
                    # which is a statement about a run that never happened — and
                    # the policy must not be consulted for frames nobody saw.
                    stats.truncated = True
                    break
                stats.frames_total += 1
                if restamp:
                    # Before the policy, and for EVERY frame including the ones it
                    # will decline: the meter's floor is learned from the run of
                    # frames, and a floor learned from a subsample is a different
                    # meter. This is the whole cost of --restamp-stability, and
                    # `decode` RAISES on a gap rather than leaving one frame at its
                    # recorded value (see _decoder).
                    entry = restamped(entry, meter, decode(entry))
                    stats.frames_restamped += 1
                # Indexed before the policy is asked, and whatever it answers: a
                # frame the policy declined was still input the camera delivered,
                # and hiding it would make a sparse policy look like a slow camera.
                frames.write(_index_line(entry) + "\n")
                # Every frame's pixels reach the ring before the policy is asked, as
                # every arriving frame does live (FrameIngest.submit), so a read
                # back-dates its change over the same frames the node would hold.
                try:
                    jpeg = (recording.path / entry.jpeg_rel).read_bytes()
                    frame = RecordedFrame(
                        header=entry.header, jpeg=jpeg, received_ts=entry.received_ts
                    ).frame()
                except (OSError, ValueError):  # gone, or not a JPEG: frames_missing
                    frame = None
                else:
                    ring.push(frame.seq, frame.capture_ts, frame.image)
                if not session_policy.should_estimate(entry):
                    stats.frames_skipped_policy += 1
                    continue
                if frame is None:
                    stats.frames_missing += 1
                    continue
                try:
                    outcome = _settle_read(
                        read(frame), frame, stabilizer, guard, rules, clock, ring.snapshot()
                    )
                except Exception as e:  # one bad frame is not the end of the stream
                    stats.frames_failed += 1
                    _say(log, f"  seq {entry.seq}: estimation failed ({e})")
                    continue
                if isinstance(outcome, Misoriented):
                    stats.frames_misoriented += 1
                    _say(
                        log,
                        f"  seq {entry.seq}: dropped, board pointing "
                        f"{outcome.direction:.0f} deg vs running {outcome.running:.0f}",
                    )
                    continue
                event = outcome.event
                events.write(
                    json.dumps(event_line(event, frame, published_ts=None, rules=rules))
                    + "\n"
                )
                stats.frames_estimated += 1
                if stats.frames_estimated % PROGRESS_EVERY == 0:
                    rate = stats.frames_estimated / max(time.monotonic() - started, 1e-9)
                    _say(
                        log,
                        f"  {stats.frames_estimated} read(s) of {total} frames"
                        f" · {rate:.2f} reads/s",
                    )
            if stats.truncated:
                break
    stats.wall_seconds = time.monotonic() - started

    result = ReplayResult(
        path=out,
        source=recording.path,
        board_id=board,
        stats=stats,
        sessions=sessions,
        replayed=chosen,
        policy=policy.name,
        restamp=restamp,
        settle=settle,
        rules=rules,
    )
    _link_source(out, recording)
    _write_manifest(result, recording, policy, reader)
    return result


def recorded_rules(recording: Recording) -> set[Rules]:
    """The settings a recording's live reads settled under, from its ``events.ndjson``.

    A line without ``rules`` predates the setting and settled under ``standard``;
    a recording with no events says nothing (an empty set).
    """
    path = recording.path / EVENTS_NAME
    if not path.exists():
        return set()
    seen: set[Rules] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                value = json.loads(line).get("rules", DEFAULT_RULES)
            except (json.JSONDecodeError, AttributeError):
                continue
            if value in get_args(Rules):
                seen.add(value)
    return seen


def _per_session_policies(policy: FramePolicy, sessions: int) -> list[FramePolicy]:
    """One policy instance per replayed streaming session, never one shared.

    A policy is stateful over a *stream*: whether a burst is in progress, how many
    reads that burst end still owes, when the keepalive last fired. A seq restart
    is not a pause in one stream, it is a different one — so carrying the object
    across the seam opens the next session inside the previous session's burst,
    and its first calm frame reads as a move completing on a board nobody touched.
    The motion meter is already reset there for the same reason.

    Rebuilding goes through the spec string rather than a ``reset()``, because
    :attr:`~.schedule.FramePolicy.name` round-trips through
    :func:`~.schedule.parse_policy` by construction (that is the invariant which
    makes a manifest's ``policy`` field the argument that reproduces the run), so
    the rebuilt policy is the same configuration with none of the history. A
    caller's own policy object has no such spec; replaying one session with it is
    fine and unchanged, and replaying several is refused rather than quietly
    leaking state across the seam.
    """
    if sessions <= 1:
        return [policy]
    try:
        return [parse_policy(policy.name) for _ in range(sessions)]
    except ValueError as e:
        raise ValueError(
            f"cannot replay {sessions} streaming sessions with policy "
            f"{policy.name!r}: it is not one of the --policy specs ({e}), so there "
            "is no way to give each session its own instance. A policy carries "
            "burst state across every frame it sees, and one shared across a seq "
            "restart reports a move completing at the top of the next session. "
            "Use a --policy spec, or replay one session at a time."
        ) from e


def _decoder(recording: Recording) -> Callable[[FrameEntry], np.ndarray]:
    """Pixels for one indexed frame. Raises when its JPEG is not there.

    Loud on purpose, and only ever called on the ``--restamp-stability`` path.
    The recorder writes the bytes *before* the index line that names them
    ("a reader that crashes into the end of this file never sees a line pointing
    at nothing" — :meth:`~board_reader_core.recording.FrameRecorder.record`), so
    within its design an indexed frame always has its JPEG and a gap means the
    tree is damaged or half-copied.

    That matters more under a restamp than anywhere else. Skipping a gap would
    leave exactly that frame carrying its **recorded** ``stability`` — an old
    absolute-scale 0.29 sitting in a stream of re-metered 0.99s, which is a
    phantom burst and, one frame later, a fake move-complete read — and its index
    line would silently lack :data:`~.schedule.RESTAMPED_KEY` while the directory
    name says ``-restamped``. A benchmark that is subtly wrong about which frames
    were moving is worse than one that refuses to run, so it refuses.

    The read path keeps its own tolerance for a vanished JPEG (``frames_missing``):
    there, one unread frame costs one estimate and is counted, which is the
    documented "not charged to the policy" case rather than a corrupted signal.
    """

    def decode(entry: FrameEntry) -> np.ndarray:
        try:
            return decode_jpeg((recording.path / entry.jpeg_rel).read_bytes())
        except (OSError, ValueError) as e:
            raise ValueError(
                f"--restamp-stability needs every frame's pixels and seq "
                f"{entry.seq} has none ({entry.jpeg_rel}: {e}). The motion meter "
                "learns its resting floor from the run of frames, so a gap is not "
                "one missing reading — it would leave that frame stamped with the "
                "OLD meter's value in an otherwise re-metered stream. Repair or "
                "re-copy the recording, or replay it without --restamp-stability."
            ) from e

    return decode


def _default_board(recording: Recording) -> str:
    """The one board in the recording, or the first one it recorded."""
    boards = recording.board_ids()
    if not boards:
        raise ValueError(f"{recording.path} recorded no frames")
    return boards[0]


def _index_line(entry: FrameEntry) -> str:
    """One ``frames.ndjson`` line for the thin copy beside the replay.

    ``eval-session`` reads the index for its timing block, so a replay needs one —
    but only for the frames it replayed, and never a copy of the pixels: the paths stay
    relative and resolve through the ``jpeg`` symlink into the source. The
    recorded ``bytes`` field is the one thing not carried over (the reader parses
    it away); the bytes themselves are one symlink hop away.
    """
    return json.dumps(
        {**entry.header, "received_ts": entry.received_ts, "jpeg": entry.jpeg_rel}
    )


def _checked_out_dir(out_dir: str | Path, recording: Recording) -> Path:
    """The replay directory, verified to sit exactly two levels under the source.

    :func:`_link_source` hops a fixed ``../..`` back to the recording for the
    pixels and the truth, so the depth is part of the contract rather than a
    detail of where the CLI happens to put things. Writing anywhere else would
    produce dangling symlinks — a replay that runs for ten GPU minutes and then
    scores nothing — so it is refused up front instead.
    """
    out = Path(out_dir)
    if out.parent.parent.resolve() != recording.path.resolve():
        raise ValueError(
            f"a replay directory must sit two levels under its recording "
            f"(<recording>/{REPLAYS_DIR}/<name>); got {out} for {recording.path}. "
            "Its jpeg/truth symlinks hop a fixed two levels back."
        )
    return out


def _link_source(out: Path, recording: Recording) -> None:
    """Point the replay at the source's pixels and truth, without copying either.

    Relative symlinks, so the whole ``recordings/`` tree stays movable, and
    replaced on every run so a stale link can never outlive its target.
    """
    up = Path("..") / ".."
    links = [(JPEG_DIR, up / JPEG_DIR)]
    if (recording.path / TRUTH_NAME).exists():
        # So `eval-session <replay dir>` just works: truth is a property of the
        # board that was in front of the camera, not of the run that read it.
        links.append((TRUTH_NAME, up / TRUTH_NAME))
    for name, target in links:
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def _write_manifest(
    result: ReplayResult,
    recording: Recording,
    policy: FramePolicy,
    estimator: str,
) -> None:
    """The provenance half of a replay: what it replayed, with which code and models.

    Recording-shaped at the top (``format``/``session_id``/``writer``) so a
    ``Recording`` opens it like any other, with everything replay-specific under
    ``replay``. ``created_ts`` is wall clock and is the only non-reproducible
    field here — the determinism claim is about ``events.ndjson``, not about the
    manifest that says when someone ran it.
    """
    env = config_env()
    sessions = result.sessions
    replayed = [sessions[i] for i in result.replayed]
    (result.path / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": FORMAT,
                "session_id": result.path.name,
                "started_ts": replayed[0].first_ts if replayed else 0.0,
                "writer": "replay",
                "replay": {
                    "source": str(recording.path),
                    "source_manifest": recording.manifest,
                    "board_id": result.board_id,
                    "created_ts": time.time(),
                    "git": git_describe(),
                    "env": env,
                    "config_digest": config_digest(env, result.settle),
                    "estimator": estimator,
                    # The spec string, so this field is literally the --policy
                    # argument that reproduces the run.
                    "policy": policy.name,
                    # The flag as given, not "did any frame get restamped": how
                    # the run was CONFIGURED is the provenance question, and the
                    # count of frames it managed to re-meter is a stat, in
                    # ``frames.frames_restamped`` with the other counters.
                    "restamp_stability": result.restamp,
                    **result.settle.to_dict(),
                    "rules": result.rules,
                    "sessions": [s.to_dict() for s in sessions],
                    "sessions_replayed": result.replayed,
                    "frames": result.stats.to_dict(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _say(log: TextIO | None, message: str) -> None:
    """Progress to ``log``, or nowhere at all when it is ``None``."""
    if log is not None:
        print(message, file=log, flush=True)


def render(result: ReplayResult) -> str:
    """The one-screen summary the CLI prints when a replay finishes."""
    s = result.stats
    lines = [
        f"replayed:  {result.source} · board {result.board_id!r} -> {result.path}",
        f"frames:    {s.frames_total} in session(s) {result.replayed} · "
        f"{s.frames_estimated} estimated in {s.wall_seconds:.1f}s"
        + (f" ({s.frames_estimated / s.wall_seconds:.2f} reads/s)" if s.wall_seconds else ""),
    ]
    skipped = [
        (s.frames_skipped_policy, "by the frame policy"),
        (s.frames_failed, "the estimator raised on"),
        (s.frames_misoriented, "dropped by the orientation guard"),
        (s.frames_missing, "whose JPEG is gone"),
    ]
    detail = ", ".join(f"{n} {what}" for n, what in skipped if n)
    if detail:
        lines.append(f"skipped:   {detail}")
    if result.rules != DEFAULT_RULES:
        lines.append(f"rules:     {result.rules} (the settle guard is off)")
    if not result.settle.is_default:
        lines.append(
            f"settle:    {result.settle.seconds:g} s, >= {result.settle.min_reads} reads, "
            f"stale after {result.settle.stale_windows} windows, "
            f"obscured above energy {result.settle.obscured_energy}"
        )
    if result.policy != EveryFrame().name or result.restamp:
        note = result.policy
        if result.restamp:
            note += (
                f" · stability re-metered on {s.frames_restamped} frame(s) "
                "(recorded values overridden)"
            )
        lines.append(f"policy:    {note}")
    if s.truncated:
        lines.append("stopped:   --limit reached — a truncated take, not a full one")
    if len(result.sessions) > 1:
        lines.append(
            "sessions:  "
            + " · ".join(
                f"[{i}] {sess.frames}f seq {sess.first_seq}-{sess.last_seq}"
                + ("" if i in result.replayed else " (not replayed)")
                for i, sess in enumerate(result.sessions)
            )
        )
    return "\n".join(lines)


def run(
    source: str | Path,
    *,
    name: str | None = None,
    board_id: str | None = None,
    all_sessions: bool = False,
    limit: int | None = None,
    policy: FramePolicy | None = None,
    restamp: bool = False,
    settle: SettleConfig = DEFAULT_SETTLE,
    rules: Rules = DEFAULT_RULES,
    log: TextIO | None = sys.stderr,
) -> ReplayResult:
    """Open ``source``, build the environment's recognizer, and replay through it.

    The recognizer comes from the gateway's own builder
    (:func:`.recognizer.build_from_env`): a replay read differently from the served
    node benchmarks something nobody runs.

    The entry point behind ``board-reader-inference replay``; ``--score`` is the
    caller's business (:func:`board_reader_inference.measurement.trial.run` on the returned
    :attr:`ReplayResult.path`, which is a recording directory like any other).
    """
    recording = Recording.open(source)
    chosen_name = (
        check_name(name)
        if name
        else default_name(
            policy=policy or EveryFrame(), restamp=restamp, settle=settle, rules=rules
        )
    )
    recognizer = build_from_env()
    return replay(
        recording,
        recording.path / REPLAYS_DIR / chosen_name,
        read=frame_reader(recognizer),
        reader=type(recognizer).__name__,
        board_id=board_id,
        policy=policy,
        restamp=restamp,
        settle=settle,
        rules=rules,
        all_sessions=all_sessions,
        limit=limit,
        log=log,
    )
