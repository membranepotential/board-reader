"""Re-run a recorded take through the live stack — iteration without the board.

A trial is a person sitting at a real board with a phone, and it is the only
honest test this project has. It is also the slowest: every code change used to
need another take. But every served run already keeps its own input
(:mod:`board_reader_core.recording`: ``frames.ndjson`` + the JPEGs) and its own
output (``events.ndjson``, scored by :mod:`.trial`), so the take is repeatable —
what was missing is the middle, the thing that pushes the recorded *frames*
through today's estimator and writes fresh events. That is this module.

    board-reader-inference replay recordings/20260813-193308 --score

**It replays the live path, not a copy of it.** The frames go through the same
:class:`~board_reader_inference.session.EstimatorSession` route decision the
``/ingest`` worker takes, the same
:func:`~board_reader_inference.ingest._read_with_prior` (estimate -> prior repair
-> :class:`~board_reader_inference.stabilize.BoardStabilizer`), and the same
:func:`~board_reader_inference.server.event_line` writer. A replayed event is
what the gateway *would* have published; a second implementation of any of those
three would make the comparison meaningless.

**The clock is the recording's, never the wall's.** Everything time-based here is
driven by each frame's ``capture_ts``: the estimator already stamps ``utc`` from
it, the stabilizer takes its ``ts`` as an argument, and the one remaining wall
clock — the ``published_ts`` the gateway stamps at publish — is *omitted* rather
than faked (see :func:`~board_reader_inference.server.event_line`). So a settle
window is 2.5 recorded seconds whether the replay takes ten minutes or ten
milliseconds, and the same recording through the same code and the same
environment produces byte-identical events. That determinism is the point: it is
what lets two replays be diffed as a claim about the *code*.

**What replay does NOT reproduce, 1: the live pre-filter is a QUALITY filter.**
The gateway's latest-wins inbox drops frames whenever the estimator falls behind
(the 08-13 trial published 85 events from 748 frames), and what survives is not
merely the freshest frame — :func:`~board_reader_inference.ingest.prefer_frame`
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

**What replay does NOT reproduce, 2: the live estimator is not frozen.** During a
served run ``POST /calibrate`` (and the §4.6 auto-calibration on ``POST /estimate``)
hot-swaps the session's active estimator, and the stream picks the new one up on
its very next frame — a take where the operator calibrated ten seconds in was
read by *two different estimators*, and its early frames by one that had never
seen this chess set. Replay resolves the environment once and freezes that
estimator and that bank for the whole take, because nothing offline calibrates
(there is no HTTP here). So early-take frames are replayed against knowledge the
live run did not have yet, which flatters the replay exactly where the live run
was still learning the board. Another reason the honest comparison is replay vs
replay.

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
such a scheduler needs and what lets a declined frame cost nothing at all — its
JPEG is never even read off disk.

A policy decides from the header alone, so "estimate this one" is a *request*,
not a guarantee: the frame it picked may still turn out to have no JPEG behind it
(a half-copied recording) or route to
:class:`~board_reader_inference.routing.Uncalibrated` (§4.6, nothing can read
piece types yet). Those land in their own counters rather than being charged to
the policy — a scheduler is judged on the frames it declined, not on the ones the
disk or the estimator could not honour.

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
from typing import TextIO

import numpy as np

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

from board_reader_inference.ingest import _read_with_prior
from board_reader_inference.routing import Uncalibrated
from board_reader_inference.schedule import (
    EveryFrame,
    FramePolicy,
    MotionMeter,
    policy_slug,
    restamped,
)
from board_reader_inference.server import event_line
from board_reader_inference.session import EstimatorSession
from board_reader_inference.stabilize import (
    SETTLE_MIN_READS,
    SETTLE_SECONDS,
    BoardStabilizer,
)
from board_reader_inference.trial import TRUTH_NAME

#: Where replays of a recording live, inside the recording they came from — a
#: benchmark that cannot be traced back to its input is not a benchmark.
REPLAYS_DIR = "replays"

#: The environment that decides which estimator gets built (``server``'s builder
#: reads exactly these). Recorded in the manifest and digested into the default
#: replay name, so two model configurations never overwrite each other's scores.
CONFIG_ENV = (
    "BR_MODEL",
    "BR_MODEL_B",
    "BR_MODEL_TUNED",
    "BR_CENTROIDS",
    "BR_SESSION_BANK",
    "BR_OCCUPANCY_BANK",
    "BR_CASCADE_TAU",
)

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
    does not reproduce ("was the session bank the same file?"). Both feed
    :func:`config_digest`, deliberately: models are retrained **in place** here
    (``models/run_e/model.onnx`` is a path, not a version), and a digest over the
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


def config_digest(env: dict[str, dict]) -> str:
    """A short stable digest of the estimator environment (8 hex chars).

    Over the values *and* the stat of each file they name — see
    :func:`config_env` for why the bytes on disk are part of the identity.
    """
    payload = json.dumps(dict(sorted(env.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def default_name(
    env: dict[str, dict] | None = None,
    *,
    policy: FramePolicy | None = None,
    restamp: bool = False,
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

    ``policy`` and ``restamp`` join the name when they are not the defaults, for
    the same reason the model stat does: they change what the run measures, and a
    sweep whose rows all land in one directory has measured its last row only.
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
    name = f"{described}-{config_digest(env if env is not None else config_env())}"
    # How the frames were CHOSEN is part of what a replay measures, so it is part
    # of the identity: without this a policy sweep on a clean tree would replay
    # every row into one directory and keep only the last.
    if policy is not None and not isinstance(policy, EveryFrame):
        name += f"-{policy_slug(policy)}"
    if restamp:
        name += "-restamped"
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
    frames_skipped_unready: int = 0  # routed Uncalibrated (§4.6): nothing to publish
    frames_failed: int = 0  # the estimator raised; the stream goes on, as it does live
    frames_missing: int = 0  # index line points at a JPEG that is gone
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
            "frames_skipped_unready": self.frames_skipped_unready,
            "frames_failed": self.frames_failed,
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


def build_session() -> EstimatorSession:
    """The estimator stack the environment asks for — the gateway's own builder.

    Deliberately calls into :mod:`.server` rather than growing a second way to
    read ``BR_MODEL``/``BR_SESSION_BANK``/…: a replay whose estimator is assembled
    differently from the served one benchmarks something nobody runs. No
    ``on_swap`` and no hot-swap path, because nothing offline calibrates —
    ``/calibrate`` is an HTTP endpoint and there is no HTTP here.
    """
    from board_reader_inference import server

    occupancy = None
    if os.environ.get("BR_OCCUPANCY_BANK"):
        occupancy = server._estimator_from_bank(os.environ["BR_OCCUPANCY_BANK"])
    return EstimatorSession(server._default_estimator(occupancy), occupancy=occupancy)


def replay(
    recording: Recording,
    out_dir: str | Path,
    *,
    session: EstimatorSession,
    board_id: str | None = None,
    policy: FramePolicy | None = None,
    restamp: bool = False,
    all_sessions: bool = False,
    limit: int | None = None,
    log: TextIO | None = sys.stderr,
) -> ReplayResult:
    """Push ``recording``'s frames through ``session`` and write a replay directory.

    The estimate/repair/debounce is :func:`~board_reader_inference.ingest._read_with_prior`
    — the ingest worker's own body — and the stabilizer is fed the frame's
    ``capture_ts``, so the settle window is recorded time. One stabilizer for the
    whole replay: it is one board's stream, in order, which is exactly the stream
    a debounce needs.

    With ``restamp``, each replayed session's headers get their ``stability``
    rewritten by a fresh :class:`~.schedule.MotionMeter` before the policy is
    asked — one meter per session, because a reconnect is a new camera and a
    learned resting floor does not carry across one. Nothing but the policy reads
    the field (the estimators only copy it onto the published event), so a
    restamped every-frame replay reads the same pixels in the same order as an
    unrestamped one; it is the control a scheduling sweep is diffed against.
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

    out = _checked_out_dir(out_dir, recording)
    out.mkdir(parents=True, exist_ok=True)
    stabilizer = BoardStabilizer()
    stats = ReplayStats()
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
        for i in chosen:
            # One meter per streaming session: a reconnect is a new camera, and a
            # resting floor learned before it says nothing about the one after.
            meter = MotionMeter()
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
                    # meter. This is the whole cost of --restamp-stability.
                    image = decode(entry)
                    if image is not None:
                        entry = restamped(entry, meter, image)
                        stats.frames_restamped += 1
                # Indexed before the policy is asked, and whatever it answers: a
                # frame the policy declined was still input the camera delivered,
                # and hiding it would make a sparse policy look like a slow camera.
                frames.write(_index_line(entry) + "\n")
                if not policy.should_estimate(entry):
                    stats.frames_skipped_policy += 1
                    continue
                try:
                    jpeg = (recording.path / entry.jpeg_rel).read_bytes()
                except OSError:
                    stats.frames_missing += 1
                    continue
                frame = RecordedFrame(
                    header=entry.header, jpeg=jpeg, received_ts=entry.received_ts
                ).frame()
                route = session.route(frame)
                if isinstance(route, Uncalibrated):
                    # §4.6: an occupancy bank is loaded but nothing can read piece
                    # types. The live worker counts and skips; so does this.
                    stats.frames_skipped_unready += 1
                    continue
                try:
                    event = _read_with_prior(route, stabilizer)
                except Exception as e:  # one bad frame is not the end of the stream
                    stats.frames_failed += 1
                    _say(log, f"  seq {entry.seq}: estimation failed ({e})")
                    continue
                events.write(
                    json.dumps(event_line(event, frame, published_ts=None)) + "\n"
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
    )
    _link_source(out, recording)
    _write_manifest(result, recording, policy, type(session.estimator).__name__)
    return result


def _decoder(recording: Recording) -> Callable[[FrameEntry], np.ndarray | None]:
    """Pixels for one indexed frame, or ``None`` when its JPEG is gone.

    The gap case is not an error here: the recorder writes the JPEG before the
    line that names it, but a half-copied tree happens, and one missing frame
    should cost the meter a slightly noisy reading rather than the whole replay.
    """

    def decode(entry: FrameEntry) -> np.ndarray | None:
        try:
            return decode_jpeg((recording.path / entry.jpeg_rel).read_bytes())
        except (OSError, ValueError):
            return None

    return decode


def _default_board(recording: Recording) -> str:
    """The one board in the recording, or the first one it recorded."""
    boards = recording.board_ids()
    if not boards:
        raise ValueError(f"{recording.path} recorded no frames")
    return boards[0]


def _index_line(entry: FrameEntry) -> str:
    """One ``frames.ndjson`` line for the thin copy beside the replay.

    ``eval-session`` reads the index for its timing block and (with
    ``--export-crops``) for the JPEG paths, so a replay needs one — but only for
    the frames it replayed, and never a copy of the pixels: the paths stay
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
                    "config_digest": config_digest(env),
                    "estimator": estimator,
                    # The spec string, so this field is literally the --policy
                    # argument that reproduces the run.
                    "policy": policy.name,
                    "restamp_stability": result.stats.frames_restamped > 0,
                    "settle_seconds": SETTLE_SECONDS,
                    "settle_min_reads": SETTLE_MIN_READS,
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
        (s.frames_skipped_unready, "with no estimator ready"),
        (s.frames_failed, "the estimator raised on"),
        (s.frames_missing, "whose JPEG is gone"),
    ]
    detail = ", ".join(f"{n} {what}" for n, what in skipped if n)
    if detail:
        lines.append(f"skipped:   {detail}")
    if result.policy != EveryFrame().name or s.frames_restamped:
        note = result.policy
        if s.frames_restamped:
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
    log: TextIO | None = sys.stderr,
) -> ReplayResult:
    """Open ``source``, build the environment's estimator, and replay into it.

    The entry point behind ``board-reader-inference replay``; ``--score`` is the
    caller's business (:func:`board_reader_inference.trial.run` on the returned
    :attr:`ReplayResult.path`, which is a recording directory like any other).
    """
    recording = Recording.open(source)
    chosen_name = (
        check_name(name)
        if name
        else default_name(policy=policy or EveryFrame(), restamp=restamp)
    )
    return replay(
        recording,
        recording.path / REPLAYS_DIR / chosen_name,
        session=build_session(),
        board_id=board_id,
        policy=policy,
        restamp=restamp,
        all_sessions=all_sessions,
        limit=limit,
        log=log,
    )
