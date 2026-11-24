"""``board-reader-inference`` CLI — run the gateway and measure what it reads.

    board-reader-inference serve --host 0.0.0.0 --port 8000
    board-reader-inference serve --record-dir runs/             # ingress recorded here
    board-reader-inference positions --changes-only              # tail the position stream
    board-reader-inference positions --out run.ndjson --quiet    # capture it to a file
    board-reader-inference eval                                  # score the deployed stack
    board-reader-inference eval-session recordings/              # score a recorded trial
    board-reader-inference replay recordings/<s> --score         # re-read a take offline

Imports only :mod:`board_reader_core` and this package; it never imports the edge.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_RECORD_DIR = "recordings"


def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from board_reader_inference.schedule import DEFAULT_POLICY_SPEC

    # Every served run records its ingress by default (WP-H2), so a session can
    # always be replayed afterwards — including one driven by the phone /camera
    # page, whose frames reach the same /ingest socket through the viewer's proxy.
    # The default lives here rather than in ``create_app`` on purpose: a real
    # server always records, a test app never writes unless it asks to.
    if args.record_dir:
        os.environ["BR_RECORD_DIR"] = args.record_dir

    # Every served run schedules its reads (D3/D4): the node estimates at the end
    # of each move plus a 3 s keepalive floor, rather than every frame that
    # survives the ingest inbox. ~12x fewer estimates on the swept takes for
    # -0.9 pp prior per-square — see docs/reports/2026-08-27-move-trigger-sweep.md.
    # `BR_SCHEDULE=every-frame` is the way back; any policy spec
    # `replay --policy` accepts works here, so a sweep's finding and the node's
    # behaviour are the same string.
    #
    # Set here rather than in `create_app` for the same reason as BR_RECORD_DIR
    # above: a real server always schedules, while an injected test app reads
    # whatever it is handed unless it asks otherwise.
    if args.schedule:
        os.environ["BR_SCHEDULE"] = args.schedule
    os.environ.setdefault("BR_SCHEDULE", DEFAULT_POLICY_SPEC)

    # The app's own loggers (per-read trial lines, refused frames), which
    # uvicorn's config leaves unconfigured — without this they fall to
    # ``lastResort`` and everything below WARNING is silently dropped.
    logging.basicConfig(
        level=os.environ.get("BR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ``--factory`` matters: the app must be BUILT here, after the env above is
    # set. The server module deliberately has no import-time app — see the note
    # at the bottom of ``server.py``.
    uvicorn.run(
        "board_reader_inference.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    from board_reader_inference.positions import run

    try:
        stats = run(
            args.url,
            board_id=args.board,
            limit=args.limit,
            changes_only=args.changes_only,
            out_path=args.out,
            tail=None if args.quiet else sys.stdout,
        )
    except OSError as e:
        # No gateway on the other end is the overwhelmingly common case; a
        # connection refused deserves one line naming the URL, not a traceback.
        print(f"cannot follow {args.url}: {e}", file=sys.stderr)
        return 2
    if args.out:
        # Report what was WRITTEN, and name the gap when --changes-only made one:
        # "wrote N" must be a fact about the file, or `wc -l` disagrees with it.
        held = stats.consumed - stats.reported
        suffix = f" ({held} unchanged, not written)" if held else ""
        print(
            f"wrote {stats.reported} position events to {args.out}{suffix}",
            file=sys.stderr,
        )
    # 1 when the stream produced nothing: an empty stream is not a working one,
    # and a scripted demo should fail rather than report success over silence
    # (the same rule `eval` applies to an empty eval). Judged on events consumed,
    # not written — a board that never moved was still a live stream.
    return 0 if stats.consumed else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from board_reader_inference.evaluate import run

    try:
        report = run(
            args.paths or None,
            prior=not args.no_prior,
            temperature=args.temperature,
            json_out=args.json,
        )
    except ValueError as e:
        # An unreadable photo or a misconfigured recognizer — say so
        # in one line rather than a traceback; the message names the fix.
        print(f"cannot evaluate: {e}", file=sys.stderr)
        return 2
    print(report.render())
    # 1 when there was nothing to score: an empty eval is not a passing eval.
    return 0 if report.boards else 1


def cmd_eval_session(args: argparse.Namespace) -> int:
    from board_reader_core.recording import Recording

    from board_reader_inference.trial import TRUTH_NAME, TruthError, run, truth_skeleton

    try:
        recording = Recording.open(args.recording)
    except FileNotFoundError as e:
        print(f"cannot score: {e}", file=sys.stderr)
        return 2
    truth_path = recording.path / TRUTH_NAME
    if not truth_path.exists():
        # The operator has to author this by hand — nothing in the recording knows
        # what was on the board — so say exactly where it goes and hand over a
        # skeleton with the board id and first seq already filled in.
        print(
            f"no ground truth for {recording.path}: write {truth_path} first. Skeleton:",
            file=sys.stderr,
        )
        print(truth_skeleton(recording))
        return 2
    try:
        report = run(recording, json_out=args.json)
    except (TruthError, ValueError) as e:
        print(f"cannot score {recording.path}: {e}", file=sys.stderr)
        return 2
    print(report.render())
    # 1 when the guards (or a board-id mismatch) left nothing: an empty session
    # is not a passing one, same rule `eval` applies to an empty eval.
    return 0 if report.events_scored else 1


def cmd_replay(args: argparse.Namespace) -> int:
    from board_reader_core.recording import Recording

    from board_reader_inference.replay import render, run
    from board_reader_inference.schedule import parse_policy
    from board_reader_inference.trial import TRUTH_NAME, TruthError

    try:
        # Parsed before anything expensive: a typo in --policy must cost a line of
        # stderr, not the twenty GPU minutes it takes to notice the sweep measured
        # the defaults under another policy's name.
        policy = parse_policy(args.policy)
    except ValueError as e:
        print(f"cannot replay: {e}", file=sys.stderr)
        return 2
    try:
        result = run(
            args.recording,
            name=args.name,
            board_id=args.board,
            policy=policy,
            restamp=args.restamp_stability,
            all_sessions=args.all_sessions,
            limit=args.limit,
        )
    except FileNotFoundError as e:
        print(f"cannot replay: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        # An empty board or an unreadable model file — one line naming the fix,
        # not a traceback.
        print(f"cannot replay {args.recording}: {e}", file=sys.stderr)
        return 2
    print(render(result))
    if not args.score:
        # Say what to run next: the replay is a recording directory like any
        # other, and that is the whole point of the layout.
        print(f"score it:  uv run board-reader-inference eval-session {result.path}")
        return 0 if result.stats.frames_estimated else 1
    if not (result.path / TRUTH_NAME).exists():
        print(
            f"--score: {result.source} has no {TRUTH_NAME} to score against "
            f"(write one, then: eval-session {result.path})",
            file=sys.stderr,
        )
        return 2
    from board_reader_inference.trial import run as score

    try:
        report = score(Recording(result.path))
    except (TruthError, ValueError) as e:
        print(f"cannot score {result.path}: {e}", file=sys.stderr)
        return 2
    print()
    print(report.render())
    return 0 if report.events_scored else 1


def build_parser() -> argparse.ArgumentParser:

    from board_reader_inference.evaluate import DEFAULT_PATHS

    p = argparse.ArgumentParser(prog="board-reader-inference", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the inference gateway")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--record-dir", dest="record_dir", default=DEFAULT_RECORD_DIR,
        help=f"record every ingested frame under here (default {DEFAULT_RECORD_DIR}/); "
             "replay with `board-reader-edge feed --recording <dir>`",
    )
    s.add_argument(
        "--no-record", dest="record_dir", action="store_const", const=None,
        help="do not record ingested frames (runs are then not reproducible)",
    )
    s.add_argument(
        "--schedule", default=None, metavar="POLICY",
        help="which frames get an estimate, as a `replay --policy` spec "
             "(env BR_SCHEDULE). Default: read at the end of each move plus a 3 s "
             "keepalive floor. `every-frame` reads everything, as the node did "
             "before scheduling — ~12x the GPU",
    )
    s.set_defaults(func=cmd_serve)

    from board_reader_inference.positions import DEFAULT_URL

    o = sub.add_parser(
        "positions",
        help="follow the L3 position stream headlessly (tail it, or capture NDJSON)",
    )
    o.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"the gateway's position-event socket (default {DEFAULT_URL})",
    )
    o.add_argument("--board", help="only this board_id (default: every board)")
    o.add_argument(
        "--out",
        help="append each event as one line of §5 JSON here — the captured stream "
             "itself, so two runs can be diffed",
    )
    o.add_argument(
        "--changes-only", dest="changes_only", action="store_true",
        help="report a board only when its placement differs from its previous "
             "one; an at-rest board republishes at the keepalive rate",
    )
    o.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many events (default: follow until interrupted)",
    )
    o.add_argument(
        "--quiet", action="store_true",
        help="suppress the readable tail on stdout (pair with --out)",
    )
    o.set_defaults(func=cmd_positions)

    e = sub.add_parser(
        "eval",
        help="score the deployed recognizer on labeled boards (perfect-board rate)",
    )
    e.add_argument(
        "paths",
        nargs="*",
        help="files/dirs holding labels (*.label.json, ground_truth.json); "
        f"default: {' '.join(DEFAULT_PATHS)}",
    )
    e.add_argument(
        "--no-prior",
        action="store_true",
        help="isolate the raw estimator by skipping the hard-constraint decode "
        "(<=1 king per colour, no pawn on rank 1/8, <=16 pieces, promotion "
        "budget). The decode is ON by default because it is what /estimate now "
        "serves, and this harness scores what serves",
    )
    e.add_argument(
        "--temperature",
        action="store_true",
        help="also fit and report temperature scaling of the confidences "
        "(leave-one-board-out, so no board is graded by its own labels). It "
        "cannot change a single answer — only how honest the numbers are.",
    )
    e.add_argument("--json", help="also write the full report as JSON here")
    e.set_defaults(func=cmd_eval)

    from board_reader_inference.replay import REPLAYS_DIR
    from board_reader_inference.schedule import POLICIES

    POLICY_DEFAULT = "every-frame"
    POLICY_HELP = "Known: " + "; ".join(
        name + (f" ({', '.join(cls.PARAMS)})" if cls.PARAMS else " (no parameters)")
        for name, cls in POLICIES.items()
    ) + "."

    r = sub.add_parser(
        "replay",
        help="re-read a recorded take through today's recognizer, writing a "
        "recording-shaped directory eval-session can score",
    )
    r.add_argument(
        "recording",
        help="a recording session directory (or a root — the newest is taken)",
    )
    r.add_argument(
        "--name",
        help=f"name the replay directory (<recording>/{REPLAYS_DIR}/<name>, one "
        "path component); default: the git description plus a digest of the "
        "BR_CHESSQ environment, so the same code and models replay into the same "
        "place — offered only on a CLEAN tree, since two sets of uncommitted "
        "changes would replay into one directory and overwrite each other",
    )
    r.add_argument("--board", help="replay this board_id (default: the first recorded)")
    r.add_argument(
        "--policy",
        default=POLICY_DEFAULT,
        metavar="SPEC",
        help="which frames get an estimate: <name>[:key=value,...]. "
        f"{POLICY_HELP} Default: {POLICY_DEFAULT} (every recorded frame — the "
        "control every sparse policy is measured against)",
    )
    r.add_argument(
        "--restamp-stability",
        dest="restamp_stability",
        action="store_true",
        help="re-meter each frame's `stability` from its JPEG with today's motion "
        "meter, overriding the recorded value before the policy sees it. Required "
        "for any take recorded before 2026-08-27 evening: those carry the old "
        "absolute-scale meter's output, which read <0.35 on every frame of every "
        "session. Restamp uniformly across a sweep or not at all",
    )
    r.add_argument(
        "--score",
        action="store_true",
        help="score the replay against the source's truth.json straight away "
        "(same as eval-session on the replay directory)",
    )
    r.add_argument(
        "--all-sessions",
        dest="all_sessions",
        action="store_true",
        help="replay every streaming session in the recording into one file; by "
        "default only the longest is replayed (seq restarts on reconnect, and "
        "truth is a step function over seq)",
    )
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many estimates and leave the rest of the take "
        "unread — a smoke run on a shared GPU. The result is a shorter take, "
        "not a sparser reading of a long one",
    )
    r.set_defaults(func=cmd_replay)

    from board_reader_inference.trial import REPORT_NAME, TRUTH_NAME

    t = sub.add_parser(
        "eval-session",
        help="score a recorded live session against its truth.json (raw vs prior "
        "vs settled)",
    )
    t.add_argument(
        "recording",
        help=f"a recording session directory (or a root — the newest is taken); "
        f"its ground truth is read from <dir>/{TRUTH_NAME}",
    )
    t.add_argument(
        "--json",
        help=f"write the full report JSON here instead of <recording>/{REPORT_NAME}",
    )
    t.set_defaults(func=cmd_eval_session)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
