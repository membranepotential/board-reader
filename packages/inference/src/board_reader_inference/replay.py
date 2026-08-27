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

**What replay does NOT reproduce** is the live pre-filter. The gateway's
latest-wins inbox drops frames when the estimator falls behind (the 08-13 trial
published 85 events from 748 frames); replay estimates every frame the policy
asks for, so it reads a *denser* stream than the run did. That is deliberate —
an offline benchmark should spend all the frames it has — but it means the
timing block of a replayed report is not the live one, and a settled row settles
off a different read rate. The debounce is time-windowed precisely so that stays
comparable (see :mod:`.stabilize`).

**Output layout.** A replay is written as a *recording-shaped* directory so
``eval-session`` scores it unchanged::

    recordings/<source>/replays/<name>/manifest.json   # provenance: source, env, git
                                    /events.ndjson    # the fresh reads
                                    /frames.ndjson    # thin copy: the replayed lines
                                    /jpeg -> ../../jpeg      (symlink, never copied)
                                    /truth.json -> ../../truth.json  (if the source has one)

``<name>`` defaults to the code state plus a digest of the model environment
(``g1a2b3c4-dirty-5e6f7a8b``), so re-running the same code against the same
models overwrites its own directory while a different model lands beside it. The
source recording is never written to, apart from that ``replays/`` subdirectory.

**Multi-session recordings.** A recording directory can hold several streaming
sessions (a reconnecting edge restarts ``seq`` at 0 — a known recorder wart), and
truth is a step function over ``seq``, so replaying two of them into one
``events.ndjson`` would score one session's reads against the other's truth.
Sessions are therefore detected (``seq`` or ``capture_ts`` going backwards within
one board) and only the **longest** is replayed, with a warning naming the rest;
``--all-sessions`` replays them all into one file for the caller who knows better.

**The frame-selection seam.** The next milestone estimates on move events rather
than on every frame (estimate at burst end, slow keepalive, skip mid-burst).
Replay is where that will be measured, so the loop asks a :class:`FramePolicy`
which frames to spend an estimate on. The only policy here is
:class:`EveryFrame` — the scheduler is not this change's job. The policy sees the
recorded header (:class:`~board_reader_core.recording.FrameEntry`: ``seq``,
``capture_ts``, ``stability``, ``sharpness``, ``burst_id``) and not the pixels,
which is both what such a scheduler needs and what lets a skipped frame cost
nothing at all — its JPEG is never even read off disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

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

from board_reader_inference.ingest import _read_with_prior
from board_reader_inference.routing import Uncalibrated
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
# The frame-selection seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class FramePolicy(Protocol):
    """Which recorded frames are worth an estimate.

    Stateful by design: the policy that is coming (burst end + keepalive) decides
    from the *run* of frames — a burst's last frame is only recognisable once the
    next one belongs to another burst — so it is asked frame by frame, in order,
    exactly once each.
    """

    #: Short name for the manifest, so a replay says how its frames were chosen.
    name: str

    def should_estimate(self, entry: FrameEntry) -> bool:
        """Whether ``entry`` gets estimated. Called once per frame, in seq order."""


@dataclass(frozen=True, slots=True)
class EveryFrame:
    """Estimate every recorded frame — the default, and the densest benchmark.

    Not what the live gateway does (its latest-wins inbox drops frames whenever
    the estimator falls behind), and deliberately so: offline there is no reason
    to leave a recorded frame unread.
    """

    name: str = "every-frame"

    def should_estimate(self, entry: FrameEntry) -> bool:
        return True


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
    does not reproduce ("was the session bank the same file?"). Only the values
    feed :func:`config_digest` — an mtime changes on a copy that changed nothing,
    and a replay should not fork its output directory over that.
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
            entry["mtime"] = stat.st_mtime
        except OSError:
            pass  # not a path, or gone — the value alone is still provenance
        out[name] = entry
    return out


def config_digest(env: dict[str, dict]) -> str:
    """A short stable digest of the estimator environment (8 hex chars)."""
    payload = json.dumps(
        {k: v["value"] for k, v in sorted(env.items())}, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def default_name(env: dict[str, dict] | None = None) -> str:
    """The replay directory name for the current code and model environment.

    Reproducible on purpose: the same code against the same models replays into
    the same directory (a re-run supersedes its own previous answer), while a
    different model lands beside it instead of silently overwriting a score.
    """
    return f"{git_describe()}-{config_digest(env if env is not None else config_env())}"


# --------------------------------------------------------------------------- #
# Running one
# --------------------------------------------------------------------------- #


@dataclass
class ReplayStats:
    """What the replay did with the frames it was given."""

    frames_total: int = 0  # in the replayed session(s), before the policy
    frames_estimated: int = 0
    frames_skipped_policy: int = 0
    frames_skipped_limit: int = 0  # past --limit: not read, but still real input
    frames_skipped_unready: int = 0  # routed Uncalibrated (§4.6): nothing to publish
    frames_failed: int = 0  # the estimator raised; the stream goes on, as it does live
    frames_missing: int = 0  # index line points at a JPEG that is gone
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_estimated": self.frames_estimated,
            "frames_skipped_policy": self.frames_skipped_policy,
            "frames_skipped_limit": self.frames_skipped_limit,
            "frames_skipped_unready": self.frames_skipped_unready,
            "frames_failed": self.frames_failed,
            "frames_missing": self.frames_missing,
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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stabilizer = BoardStabilizer()
    stats = ReplayStats()
    total = sum(sessions[i].frames for i in chosen)
    started = time.monotonic()

    events_path = out / EVENTS_NAME
    frames_path = out / INDEX_NAME
    with (
        events_path.open("w", encoding="utf-8") as events,
        frames_path.open("w", encoding="utf-8") as frames,
    ):
        for i in chosen:
            for entry in index[sessions[i].start : sessions[i].stop]:
                stats.frames_total += 1
                frames.write(_index_line(entry) + "\n")
                if limit is not None and stats.frames_estimated >= limit:
                    stats.frames_skipped_limit += 1
                    continue
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
    stats.wall_seconds = time.monotonic() - started

    result = ReplayResult(
        path=out,
        source=recording.path,
        board_id=board,
        stats=stats,
        sessions=sessions,
        replayed=chosen,
    )
    _link_source(out, recording)
    _write_manifest(result, recording, policy, type(session.estimator).__name__)
    return result


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
                    "policy": policy.name,
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
        (s.frames_skipped_limit, "past --limit"),
        (s.frames_skipped_unready, "with no estimator ready"),
        (s.frames_failed, "the estimator raised on"),
        (s.frames_missing, "whose JPEG is gone"),
    ]
    detail = ", ".join(f"{n} {what}" for n, what in skipped if n)
    if detail:
        lines.append(f"skipped:   {detail}")
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
    log: TextIO | None = sys.stderr,
) -> ReplayResult:
    """Open ``source``, build the environment's estimator, and replay into it.

    The entry point behind ``board-reader-inference replay``; ``--score`` is the
    caller's business (:func:`board_reader_inference.trial.run` on the returned
    :attr:`ReplayResult.path`, which is a recording directory like any other).
    """
    recording = Recording.open(source)
    out = recording.path / REPLAYS_DIR / (name or default_name())
    return replay(
        recording,
        out,
        session=build_session(),
        board_id=board_id,
        policy=policy,
        all_sessions=all_sessions,
        limit=limit,
        log=log,
    )
