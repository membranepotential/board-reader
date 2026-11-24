"""Inference gateway (DESIGN.md §4.5) — FastAPI WebSocket ingress + egress.

Runs on the inference node. Three surfaces:

* ``WS /ingest`` — the edge connects and streams frames. Each frame is two WS
  messages: a JSON header then the binary JPEG. The gateway decodes and acks, and
  the per-board ingest worker (:mod:`board_reader_inference.ingest`) estimates off
  the event loop and publishes the position event to the hub.
* ``WS /viewer`` — viewers connect and receive the position-event JSON stream
  (retained latest per board first, then live updates).
* ``GET /boards/{id}`` + ``GET /boards/{id}/frame`` + ``GET /healthz`` — latest
  position snapshot, latest raw ingested JPEG, and liveness.

The gateway holds the shared estimator and the position hub. It never imports the
edge package; it only speaks the wire (:mod:`board_reader_core.wire`).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import itertools
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from board_reader_core.boarddetect import LOW_ANGLE_ASPECT, quad_aspect
from board_reader_core.corners import parse_corners
from board_reader_core.protocol import (
    GRID,
    START_PLACEMENT,
    PositionEvent,
    square_evidence,
)
from board_reader_core.recording import FrameRecorder
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.chessq import ChessQEstimator
from board_reader_inference.estimator import CentroidEstimator, Estimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import FrameIngest
from board_reader_inference.routing import PhotoRead, ReadView, Uncalibrated
from board_reader_inference.schedule import EveryFrame, FramePolicy, parse_policy
from board_reader_inference.session import (
    CalibrationRefused,
    CalibrationRejected,
    EstimatorSession,
    booster_weight,
    session_bank_path,
)
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    square_boxes,
)
from board_reader_inference.trained import (
    DEFAULT_CASCADE_TAU,
    AveragedEstimator,
    CascadeEstimator,
    TrainedEstimator,
    TrainedReader,
)

log = logging.getLogger(__name__)


def _parse_debug(spec: str | None) -> bool:
    """``?debug=1|true`` -> on; absent/anything else -> off."""
    return spec is not None and spec.lower() in ("1", "true")


def _id_epoch(upload_id: str) -> float:
    """Chronological proxy from an upload id's own ``YYYYMMDD-HHMMSS`` prefix.

    ``_save_upload`` always writes a ``ts`` field, so this only matters for a
    hand-edited or half-written sidecar that lost it — ``GET /uploads`` still
    needs *some* sortable value, and the id encodes a timestamp too.
    """
    try:
        date, clock = upload_id.split("-")[:2]
        # Naive on purpose: the id was stamped by `time.strftime` in _save_upload,
        # i.e. in LOCAL time, so reading it back as local is what recovers the
        # instant it named. Forcing UTC here would shift every fallback timestamp
        # by the machine's offset and mis-sort /uploads against the ids' own text.
        return datetime.datetime.strptime(  # noqa: DTZ007
            date + clock, "%Y%m%d%H%M%S"
        ).timestamp()
    except (ValueError, IndexError):
        return 0.0


# Ingress recording is opt-in the same way session-bank persistence is
# (``BR_RECORD_DIR=<dir>``; see :mod:`board_reader_inference.session`), but the
# ``serve`` command opts in on the operator's behalf — see create_app.
def _make_recorder() -> FrameRecorder | None:
    """The ingress recorder for this app, or ``None`` when recording is off."""
    root = os.environ.get("BR_RECORD_DIR")
    if not root:
        return None
    return FrameRecorder(root, writer="inference-gateway", max_bytes=_record_max_bytes())


def _record_max_bytes() -> int | None:
    """``BR_RECORD_MAX_MB`` as bytes — a disk-fill stop, off (unlimited) by default.

    A three-hour session at burst rate is a couple of GB; a recorder left running
    on a small Jetson disk is not something to discover by filling it.
    """
    raw = os.environ.get("BR_RECORD_MAX_MB")
    if not raw:
        return None
    try:
        return max(int(float(raw) * 1_000_000), 0)
    except (ValueError, OverflowError):
        print(f"BR_RECORD_MAX_MB={raw!r} is not a number; recording without a cap")
        return None


# Below this mean pixel value a photo is dim enough that recognition measurably
# degrades (evening-room shots measured ~37/255) — responses carry a warning.
LOW_LIGHT_MEAN = 55.0

# Projected board aspect below which the camera is too close to board level for
# recognition to survive; defined in core beside the detector. Warn, don't block.
LOW_ANGLE_HW = LOW_ANGLE_ASPECT

# The metric and its threshold live in core beside the detector that produces the
# quads, so the serving warning and the capture-side stream QC (tools/video) can
# never drift apart.
_quad_aspect = quad_aspect


def _prior_decode(event):
    """Hard-constraint repair of one read (:mod:`.prior`), for serving beside it.

    ``square_dist`` and ``event.placement`` are left exactly as the estimator
    produced them — the §5 invariant is that ``placement`` is the argmax of
    ``square_dist``, and the hub publishes the raw event. The repair rides along
    in the HTTP response as ``result["prior"]``, so applying it stays the
    consumer's choice and a downstream legality smoother cannot double-count it
    (DESIGN D1).

    **``quarter_turns`` is deliberately not passed.** The only orientation this
    path has is ``autocalib.resolve_rotation``'s guess, which picks the frame that
    *minimises* pawns on ranks 1/8 — the very quantity ``BACK_RANK_PAWNS``
    measures. Feeding it that guess is circular and was measured destroying a
    correct, confident pawn; see :mod:`.prior`'s module docstring for the full
    account. Do not "fix" this by wiring the guess in: that is the bug.
    """
    from board_reader_inference.prior import decode

    return decode(np.asarray(event.square_dist))


def _estimator_from_bank(
    path: str, *, share_from: Estimator | None = None
) -> CentroidEstimator:
    """Build a :class:`CentroidEstimator` from a saved calibration file.

    Rebuilds the same backbone and box geometry the bank was calibrated with
    (dims must match; photo boxes must be cut identically). If ``share_from`` is
    an estimator on the same backbone, its embedder is reused — one ONNX session
    instead of two, which matters on a 4GB GPU.
    """
    from board_reader_inference.centroid import load_classifier
    from board_reader_inference.embed import make_embedder

    classifier = load_classifier(path)
    shared = getattr(share_from, "embedder", None)
    shared_backbone = getattr(getattr(share_from, "classifier", None), "backbone", None)
    embedder = shared if shared is not None and shared_backbone == classifier.backbone else None
    return CentroidEstimator(
        embedder if embedder is not None else make_embedder(classifier.backbone),
        classifier,
        geometry=BboxGeometry(
            headroom=classifier.box_headroom, side_pad=classifier.box_side_pad
        ),
    )


def _frame_policy() -> Callable[[], FramePolicy]:
    """``BR_SCHEDULE`` -> a factory for each board's read scheduler.

    The value is a policy spec string (``burst-end:burst_end_reads=3``,
    ``every-frame``, ``when-still:theta=0.9``) — the same grammar
    ``board-reader-inference replay --policy`` takes and the same string a replay
    manifest records, so what a sweep measured and what the node runs cannot drift
    into being two different things.

    **Unset means every frame here**, which is what the gateway did before
    scheduling existed and what an injected test app should keep doing. The
    shipped default is set by ``serve`` (:data:`~.schedule.DEFAULT_POLICY_SPEC`) —
    the same asymmetry ``BR_RECORD_DIR`` has, and for the same reason: a real
    server always schedules, a ``create_app()`` in a test does exactly what it was
    asked to.

    A factory rather than an instance: a policy is stateful, and each board's
    burst tracking is its own.
    """
    spec = os.environ.get("BR_SCHEDULE")
    if not spec:
        return EveryFrame
    policy = parse_policy(spec)  # a typo fails at startup, not on the first frame
    print(f"scheduling reads with {policy.name}")
    return lambda: parse_policy(spec)


def _cascade_tau() -> float:
    """``BR_CASCADE_TAU`` -> the cascade's confidence gate (default 0.95)."""
    raw = os.environ.get("BR_CASCADE_TAU")
    if not raw:
        return DEFAULT_CASCADE_TAU
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"BR_CASCADE_TAU={raw!r} is not a number") from e


def _trained_component(model: str) -> TrainedReader | None:
    """The zero-shot trained member named by the environment, or ``None``.

    ``BR_MODEL`` alone is one :class:`TrainedEstimator`, unchanged. Exactly one
    companion may join it, and the choice is between *measured* configurations:

    * ``BR_MODEL_B`` — a second, differently-cropped classifier; the pair becomes
      a :class:`CascadeEstimator` (``BR_CASCADE_TAU`` moves the gate). Measured
      0.956 on user_photos, zero-shot on any board.
    * ``BR_MODEL_TUNED`` — a calibration fine-tune of ``BR_MODEL`` produced by
      ``tools/train/calibrate_finetune.py``; the pair becomes an
      :class:`AveragedEstimator`. Measured 0.969 — but only on the board it was
      calibrated for.

    Either way the result is *the trained component*: a session bank still wraps
    it in an ``EnsembleEstimator`` on top, with the same arithmetic.

    A broken ``BR_MODEL`` degrades to the next tier so ``docker compose up`` works
    on a fresh clone without ``models/`` — and with the cascade in compose.yaml
    that courtesy has one more shape: **both** model files absent is the fresh
    clone again, so it degrades too. Any *partial* pair (one file present, or a
    file that exists but will not load) is a misconfiguration and fails hard:
    quietly serving one model instead of the requested two would show up only as
    "the ensemble didn't help" months later. ``BR_MODEL_TUNED`` gets no
    fresh-clone courtesy at all — a tuned model only exists because somebody ran
    the tool, so setting it is always deliberate, and the tuned model *alone*
    hallucinates pieces onto empty squares (it is the base's vote in the average
    that suppresses them).
    """
    model_b = os.environ.get("BR_MODEL_B")
    tuned = os.environ.get("BR_MODEL_TUNED")
    if model_b and tuned:
        # Neither the cascade-of-three nor "which pair wins" has been measured,
        # and the two configurations answer different questions (zero-shot
        # breadth vs one calibrated board). Guessing here would silently ship an
        # unmeasured model.
        raise RuntimeError(
            "BR_MODEL_B and BR_MODEL_TUNED are both set; stacking them is an "
            "UNMEASURED configuration. Pick one: BR_MODEL_B is the zero-shot "
            "cascade (measured 0.956 on user_photos, any board), BR_MODEL_TUNED "
            "is the calibration average (measured 0.969, but only on the board "
            "its frames came from)."
        )
    try:
        trained = TrainedEstimator(model)
    except Exception as e:
        if tuned:
            raise RuntimeError(
                f"BR_MODEL_TUNED={tuned!r} requests the calibration average, but "
                f"its base BR_MODEL={model!r} is not usable ({e}). The average "
                "needs both members — the fine-tuned model alone was measured to "
                "hallucinate pieces onto empty squares."
            ) from e
        if model_b:
            if not Path(model).exists() and not Path(model_b).exists():
                print(
                    f"BR_MODEL={model!r} and BR_MODEL_B={model_b!r} both absent "
                    "(fresh clone?); serving without the trained component"
                )
                return None
            raise RuntimeError(
                f"BR_MODEL_B={model_b!r} requests the cascade, but its first member "
                f"BR_MODEL={model!r} is not usable ({e})"
            ) from e
        print(f"BR_MODEL={model!r} not usable ({e}); serving without it")
        return None
    if tuned:
        return AveragedEstimator(trained, TrainedEstimator(tuned))
    if not model_b:
        return trained
    return CascadeEstimator(trained, TrainedEstimator(model_b), tau=_cascade_tau())


#: The estimator settings ``BR_CHESSQ`` outranks, named at startup when also set.
_CHESSQ_OUTRANKS = ("BR_MODEL", "BR_MODEL_B", "BR_MODEL_TUNED", "BR_CENTROIDS", "BR_SESSION_BANK")


def _chessq_estimator(path: str) -> ChessQEstimator:
    """``BR_CHESSQ=<path>`` -> ChessQ Lite, or a hard error when the file is missing.

    No fresh-clone courtesy (unlike ``BR_MODEL``): the weights are never in the
    repo, so setting the variable is always deliberate, and quietly serving the
    cascade instead would be read as ChessQ's accuracy.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"BR_CHESSQ={path!r} does not name a model file")
    outranked = [n for n in _CHESSQ_OUTRANKS if os.environ.get(n)]
    note = f"; ignoring {', '.join(outranked)}" if outranked else ""
    print(f"serving ChessQ from BR_CHESSQ={path!r} (calibration disabled{note})")
    return ChessQEstimator(path)


def _default_estimator(occupancy_est: CentroidEstimator | None = None) -> Estimator:
    """Pick the style (piece-type) estimator from the environment.

    ``BR_CHESSQ=<path>`` -> ChessQ Lite (opt-in, :mod:`.chessq`), ahead of every
    other setting: it reads the whole frame on its own, so a model, bank or
    calibration beside it would not be part of the read (and calibration is
    refused while it serves, :class:`~.session.CalibrationRefused`).
    ``BR_CENTROIDS=<path>`` -> that calibration. Unset but auto-calibration
    enabled (an occupancy estimator exists) -> restore the persisted session bank
    if a previous run calibrated one. Otherwise the M0 :class:`StubEstimator`.
    Resolved at import time so the uvicorn string-imported ``app`` honours it
    without a code change.
    """
    chessq = os.environ.get("BR_CHESSQ")
    if chessq:
        return _chessq_estimator(chessq)
    path = os.environ.get("BR_CENTROIDS")
    if path:
        return _estimator_from_bank(path, share_from=occupancy_est)
    # The multi-style-trained crop classifier: the zero-shot default
    # (2026-07-18 campaign). Explicit BR_CENTROIDS still outranks it. A persisted
    # session bank does NOT replace it — it boosts it: the 50/50 distribution
    # blend beats either member alone (439 vs 415/430 of 448 on the labeled
    # eval), because the bank is near-perfect in conditions it has seen and the
    # trained model degrades gracefully in ones it hasn't.
    model = os.environ.get("BR_MODEL")
    if not model and os.environ.get("BR_MODEL_TUNED"):
        # The average needs its base. Falling through to the stub here would
        # serve a fixed placement while the operator believes a calibrated model
        # is running (see _trained_component on why the tuned model has no solo
        # mode at all).
        raise RuntimeError(
            f"BR_MODEL_TUNED={os.environ['BR_MODEL_TUNED']!r} is set but BR_MODEL "
            "is not. The calibration average is the tuned model 50/50 with the "
            "base it was fine-tuned from; set BR_MODEL to that base."
        )
    if model:
        from board_reader_inference.estimator import EnsembleEstimator

        trained = _trained_component(model)
        if trained is not None:
            bank = session_bank_path()
            if bank and Path(bank).exists():
                try:
                    bank_est = _estimator_from_bank(bank, share_from=occupancy_est)
                    return EnsembleEstimator(
                        trained,
                        bank_est,
                        weight=booster_weight(bank_est.classifier.exemplars.shape[0]),
                    )
                except (OSError, ValueError, KeyError):
                    pass  # unreadable/stale bank: zero-shot still works without it
            return trained
    bank = session_bank_path()
    if occupancy_est is not None and bank and Path(bank).exists():
        return _estimator_from_bank(bank, share_from=occupancy_est)
    return StubEstimator()


def event_line(
    event: PositionEvent, frame: Frame, *, published_ts: float | None
) -> dict:
    """One published event as its ``events.ndjson`` line (:data:`EVENTS_NAME`).

    ``square_dist`` is stripped — replaying ``frames.ndjson`` through the same
    estimator reproduces it, and at burst rate it is ~25 kB per line. What replay
    cannot cheaply give back is *why* a square read the way it did, so the top-2
    digest (:func:`square_evidence`, ~1.8 kB) rides along instead: it is what
    turns "d8 read k" into "d8 read k at 0.51 over q at 0.49", and it is the
    whole evidence base for the trial report's margin section. The digest is of
    the estimator's own distribution — the ``prior`` repair never rewrites it —
    so it stays comparable across the three streams.

    The frame's own ``seq`` and ``capture_ts`` ride along too, so an event can be
    joined back to the exact recorded frame it was estimated from.

    ``published_ts`` is this node's wall clock at publish, and ``None`` omits it:
    an offline replay (:mod:`board_reader_inference.replay`) never published
    anything, and stamping it with either the replay's wall clock or the capture
    time would put a fabricated capture->publish latency in the trial report. A
    line without the key scores identically and reports "not stamped".

    Shared with the replay harness on purpose: a replayed events.ndjson is only
    worth comparing to a recorded one if the same code wrote both lines.
    """
    line = event.to_dict()
    line.pop("square_dist", None)
    line["evidence"] = square_evidence(event.square_dist)
    line["frame_seq"] = frame.seq
    line["capture_ts"] = frame.capture_ts
    if published_ts is not None:
        line["published_ts"] = published_ts
    return line


def create_app(
    estimator: Estimator | None = None,
    hub: PositionHub | None = None,
    occupancy_estimator: CentroidEstimator | None = None,
) -> FastAPI:
    """Build the gateway app. Estimator/hub/occupancy are injectable for tests.

    ``occupancy_estimator`` (or ``BR_OCCUPANCY_BANK=<path>``, e.g. the ChessReD
    exemplar bank) enables the zero-setup path: while no piece-type calibration is
    active, ``/estimate`` answers with style-robust occupancy only — and the
    moment a photo's occupancy matches the chess starting position, the server
    auto-calibrates piece types from that photo (DESIGN §4.6).
    """
    occ_est: CentroidEstimator | None = occupancy_estimator
    if occ_est is None and os.environ.get("BR_OCCUPANCY_BANK"):
        occ_est = _estimator_from_bank(os.environ["BR_OCCUPANCY_BANK"])
    est: Estimator = estimator if estimator is not None else _default_estimator(occ_est)
    hub_: PositionHub = hub if hub is not None else PositionHub()
    # Every swap of the active estimator goes through the session, which mirrors
    # the new one onto ``app.state`` (bound just below — a swap can only happen
    # inside a request, long after this function returns).
    session = EstimatorSession(
        est,
        occupancy=occ_est,
        on_swap=lambda swapped: setattr(app.state, "estimator", swapped),
    )
    # Streaming ingress (WP-G1/G2): frames are estimated off the event loop by
    # per-board workers, through the same routing decision ``POST /estimate``
    # takes. Resolved per frame so /calibrate's hot-swap reaches the stream —
    # including out of the §4.6 uncalibrated state, where the route is
    # ``Uncalibrated`` and the worker skips the frame instead of publishing the
    # stub's fabricated placement.
    def _record_published(event, frame) -> None:
        """One compact line per published event into the session's events.ndjson."""
        if recorder is None or not recorder.active:
            return
        if not recorder.record_event(event_line(event, frame, published_ts=time.time())):
            print(f"frame recording stopped: {recorder.stopped_reason}")

    ingest_ = FrameIngest(
        session.route, hub_, on_event=_record_published, policy=_frame_policy()
    )
    # Latest-wins raw-frame retention, one entry per board, in-memory only (no
    # disk write, unbounded board count — mirrors the hub's own retention). This
    # is what ``GET /boards/{id}/frame`` serves for the live video-like preview;
    # it is independent of the hub's estimated PositionEvent.
    latest_frames: dict[str, tuple[dict, bytes]] = {}

    # Ingress recording (WP-H2): every frame that crosses the wire is appended to
    # a session directory, so any run can be replayed later with
    # ``board-reader-edge feed --recording <dir>``. ``/ingest`` is the one
    # chokepoint every edge passes through — the CLI feed, the Pi, and the phone
    # ``/camera`` page (which reaches it through the viewer's proxy) — so
    # recording here covers all of them and cannot be bypassed by a new edge.
    #
    # Enabled by ``BR_RECORD_DIR=<dir>``, which ``serve`` sets for you: a real run
    # always records, while ``create_app()`` in a test writes nothing unless it
    # asks to. That asymmetry is deliberate — cwd-relative writes on by default
    # have twice cost us real user state (see ``BR_SESSION_BANK``).
    recorder = _make_recorder()
    if recorder is not None:
        print(f"recording ingress frames to {recorder.path}")

    @contextlib.asynccontextmanager
    async def lifespan(app_: FastAPI):
        try:
            async with ingest_.lifespan(app_):
                yield
        finally:
            if recorder is not None:
                recorder.close()

    app = FastAPI(title="board-reader inference node", lifespan=lifespan)
    # The viewer is served from a different origin (its own container), and the
    # browser connects here over WS; allow it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("BR_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Seq", "X-Capture-Ts", "X-Corners"],
    )
    app.state.estimator = session.estimator
    app.state.session = session
    app.state.hub = hub_
    app.state.ingest = ingest_
    app.state.recorder = recorder

    def _record(header: dict, jpeg: bytes) -> None:
        """Best-effort ingress recording — a failed write stops the recorder, never
        the stream it observes. The transition is announced once, because a run
        that silently stopped recording is worse than one that never started."""
        if recorder is None or not recorder.active:
            return
        if not recorder.record(header, jpeg):
            print(f"frame recording stopped: {recorder.stopped_reason}")

    @app.get("/healthz")
    async def healthz() -> dict:
        # The recording stats ride on liveness so an operator can confirm frames
        # are landing on disk without shelling into the container. The ingest
        # counters ride along for the same reason: "frames are arriving but none
        # are being estimated" is the §4.6 uncalibrated state, and it is otherwise
        # invisible from outside the process — the stream simply looks quiet.
        health = {
            "status": "ok",
            "subscribers": hub_.subscriber_count,
            "ingest": {
                "received": ingest_.frames_received,
                "estimated": ingest_.frames_estimated,
                "dropped": ingest_.frames_dropped,
                "failed": ingest_.frames_failed,
                "skipped": ingest_.frames_skipped,
                "refused": ingest_.frames_refused,
                # The scheduler's two counters. "unscheduled" is the saving —
                # at the shipped burst-end policy it is most of the stream, and
                # "estimated barely moves while received climbs" is the expected
                # picture, not a stall. "backstopped" should be 0: anything else
                # says an edge's capture_ts stopped advancing and the node is
                # reading on its own clock to stay alive.
                "unscheduled": ingest_.frames_unscheduled,
                "backstopped": ingest_.frames_backstopped,
            },
        }
        if recorder is not None:
            health["recording"] = recorder.stats()
        # The cascade's whole economic claim is "the second model runs on under a
        # third of squares". Ride it on liveness so an operator can check the
        # claim against their own boards instead of trusting the experiment.
        trained = session.base_trained()
        if isinstance(trained, CascadeEstimator):
            health["cascade"] = {
                "tau": trained.tau,
                "model": trained.primary.model_path,
                "model_b": trained.secondary.model_path,
                **trained.metrics.as_dict(),
            }
        # Which board a tuned model was calibrated for is not knowable from here,
        # but *that* one is in the average — and which base it is averaged with —
        # is exactly what an operator needs to see before trusting a reading.
        if isinstance(trained, AveragedEstimator):
            health["tuned"] = {
                "model": trained.model_path,
                "model_tuned": trained.tuned_path,
                "weight": 0.5,
            }
        # ChessQ is opt-in and outranks the whole default stack, so an operator
        # needs to see it is the one reading before trusting a number.
        if isinstance(session.estimator, ChessQEstimator):
            health["chessq"] = {"model": session.estimator.model_path}
        return health

    @app.get("/boards/{board_id}")
    async def board_latest(board_id: str) -> JSONResponse:
        event = hub_.latest(board_id)
        if event is None:
            return JSONResponse({"error": "no such board"}, status_code=404)
        return JSONResponse(event.to_dict())

    @app.get("/boards/{board_id}/frame")
    async def board_frame(board_id: str) -> Response:
        """Most recent raw JPEG ingested for ``board_id`` — a video-like preview.

        Served as-received (no re-encode) so the viewer can show what the camera
        actually sent, independent of whatever the estimator made of it. When the
        stream carried the playing-surface quad, ``X-Corners`` reports it in the
        same ``x0,y0,...,x3,y3`` spelling ``?corners=`` takes — so the viewer can
        calibrate straight from a live frame without asking the edge anything.
        """
        entry = latest_frames.get(board_id)
        if entry is None:
            return JSONResponse({"error": "no frame for board"}, status_code=404)
        header, jpeg = entry
        headers = {
            "X-Seq": str(header["seq"]),
            "X-Capture-Ts": str(header["capture_ts"]),
            "Cache-Control": "no-store",
        }
        corners = header.get("corners")
        if corners:
            headers["X-Corners"] = ",".join(str(v) for v in corners)
        return Response(content=jpeg, media_type="image/jpeg", headers=headers)

    # One-shot HTTP ingress for the phone snapshot page (viewer /snapshot).
    #
    # Without ``corners`` the body is an already-rectified top-down crop (the
    # original M1 contract). With ``corners=x0,y0,...,x3,y3`` (a8,h8,h1,a1, pixel
    # coords of the POSTED image) the body is the ORIGINAL photo and recognition
    # prefers tall full-piece boxes cut from it (Chesscog-style) — a warped square
    # decapitates pieces, which is exactly what hurt real-photo accuracy. The
    # estimate path is matched to the centroid file's ``crop_mode`` so inference
    # crops the same way calibration did.
    snapshot_seq = itertools.count()

    # Opt-in raw-photo retention (``BR_SAVE_UPLOADS=<dir>``): every decodable
    # photo POSTed to /estimate or /calibrate is kept with its request metadata,
    # so real-world shots become labeled eval/calibration data instead of being
    # discarded after the response. Off by default (tests, containers).
    save_dir = os.environ.get("BR_SAVE_UPLOADS")
    upload_seq = itertools.count()

    def _save_upload(
        kind: str,
        body: bytes,
        *,
        corners: str | None = None,
        placement: str | None = None,
    ) -> Path | None:
        """Returns the sidecar path so callers can append response metadata."""
        if not save_dir:
            return None
        try:
            d = Path(save_dir)
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{next(upload_seq):04d}-{kind}"
            (d / f"{stem}.jpg").write_bytes(body)
            meta: dict[str, object] = {"kind": kind, "ts": time.time()}
            if corners is not None:
                meta["corners"] = corners
            if placement is not None:
                meta["placement"] = placement
            sidecar = d / f"{stem}.json"
            sidecar.write_text(json.dumps(meta))
            return sidecar
        except OSError:
            return None  # retention is best-effort; never fail the request over it

    def _annotate_upload(sidecar: Path | None, **extra: object) -> None:
        """Best-effort append of response facts (predicted placement, rotation)
        to a retained upload's sidecar — turns every shot into eval data."""
        if sidecar is None:
            return
        try:
            meta = json.loads(sidecar.read_text())
            meta.update({k: v for k, v in extra.items() if v is not None})
            sidecar.write_text(json.dumps(meta))
        except (OSError, ValueError):
            pass

    def _debug_payload(view: ReadView, square_dist, *, rotate_k: int = 0) -> dict:
        """The opt-in ``?debug=1`` visualisation pair, from what the branch saw.

        ``rotate_k`` is the same ``np.rot90`` turn count ``autorotate`` applied to
        ``square_dist``. Boxes/crops are sampled from ``view`` exactly as
        recognition saw them (the photo pixels a piece stands in don't move just
        because a label was wrong) and only THEN reindexed by the identical
        rot90 permutation — not re-derived from a corner-shifted homography.
        The two are not equivalent: both ``headroom`` (upward) and ``side_pad``
        (sideways) extend a box along fixed IMAGE axes using a board-relative
        width, independent of which corner is labeled a8 — so re-deriving boxes
        from shifted corners silently rotates that extension too and samples
        the wrong pixels for a 90°/270° turn (measured up to ~2px at
        headroom=0.2, ~6px at 0.6, on a small test board — see
        test_squares.py). Exact agreement only holds with *both* padding
        params at 0 (no extension to misdirect) or at a 180° turn (central
        symmetry doesn't swap width/height). Reindexing the already-correct
        boxes is an exact array-index permutation, unaffected by either.
        """
        from board_reader_inference.diagnostics import (
            crop_mosaic,
            draw_boxes,
            to_b64_jpeg,
        )

        boxes = square_boxes(view.corners, view.geometry, view.image.shape)
        crops = crop_squares_photo(
            view.image, view.corners, geometry=view.geometry, size=view.size
        )
        if rotate_k:
            boxes = np.rot90(
                boxes.reshape(GRID, GRID, 4, 2), rotate_k, axes=(0, 1)
            ).reshape(-1, 4, 2)
            crop_grid = np.rot90(
                np.stack(crops).reshape(GRID, GRID, *crops[0].shape),
                rotate_k,
                axes=(0, 1),
            )
            crops = list(crop_grid.reshape(-1, *crops[0].shape))
        return {
            "boxes": to_b64_jpeg(draw_boxes(view.image, boxes, square_dist)),
            "mosaic": to_b64_jpeg(crop_mosaic(crops, square_dist)),
        }

    async def _uncalibrated_estimate(
        route: Uncalibrated,
        debug_on: bool,
        warns: list[str],
        sidecar: Path | None = None,
    ) -> JSONResponse:
        """Zero-setup path: occupancy-only readout + start-position auto-calibration.

        Piece type does not transfer across piece styles on frozen features
        (DESIGN §4.6), so with no style calibration active we answer with
        occupancy only (style-robust, HTTP-response-only — never published to the
        hub, which carries real position events exclusively). But if this photo's
        occupancy matches the chess STARTING position, we calibrate piece types
        from the photo itself via the exemplar-bank machinery and answer with a
        full, published position event — the user never pressed anything.
        """
        from board_reader_inference.autocalib import occupancy_grid

        # Both do real model/embedding work (occupancy read, then a fresh
        # exemplar bank built off it) — off the event loop for the same reason
        # /detect's search is (see below): /ingest acks and /viewer must not
        # stall behind a snapshot upload.
        reading = await asyncio.to_thread(route.read)
        auto = await asyncio.to_thread(session.adopt_start, route, reading)
        if auto is None:
            result = {
                "board_id": route.frame.board_id,
                "mode": "occupancy",
                "occupied": occupancy_grid(reading.dist).tolist(),
                "hint": (
                    "uncalibrated for this piece set: showing occupancy only — "
                    "photograph the starting position once to enable full recognition"
                ),
            }
            if warns:
                result["warnings"] = warns
            if debug_on:
                result["debug"] = _debug_payload(route.view(), reading.dist)
            return JSONResponse(result)

        event = await asyncio.to_thread(auto.reread, route.frame, route.corners)
        # This branch answers with real piece types, so it gets the same repair as
        # the calibrated path. (The occupancy-only branch above does not: with no
        # piece identity and no colour there is nothing the constraints can read.)
        # ``auto.rotation`` *would* be an established frame here — it comes from
        # the start-position occupancy pattern, not from the circular pawn
        # heuristic — but a start position has no pawn on rank 1/8 by
        # construction, so the one frame-dependent constraint could never fire.
        # Passing it would add an untestable convention for zero gain.
        #
        # Attached BEFORE the publish: the hub's subscribers (the live page, the
        # /camera panel) draw ``prior.placement``, and an event published bare
        # would put an impossible board on exactly the surface the repair exists
        # for. ``result`` then inherits it through ``to_dict``.
        repair = await asyncio.to_thread(_prior_decode, event)
        event.prior = repair.to_repair()
        await hub_.publish(event)
        result = event.to_dict()
        result["autocalibrated"] = {
            "matches": auto.matches,
            "rotation": auto.rotation,
            "exemplars": auto.exemplars,
            "saved": auto.saved,
        }
        # This branch answers with a real position too, so it is labelable like any
        # other read (see /labels).
        _annotate_upload(
            sidecar,
            placement=event.placement,
            prior_flips=[f.square for f in repair.flips],
        )
        if sidecar is not None:
            result["upload"] = sidecar.stem
        if warns:
            result["warnings"] = warns
        if debug_on:
            result["debug"] = _debug_payload(route.view(), event.square_dist)
        return JSONResponse(result)

    @app.post("/estimate")
    async def estimate(
        request: Request,
        board_id: str = "snapshot",
        corners: str | None = None,
        debug: str | None = None,
        autorotate: str | None = None,
    ) -> JSONResponse:
        body = await request.body()
        try:
            image = decode_jpeg(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        sidecar = _save_upload("estimate", body, corners=corners)
        quad = None
        if corners is not None:
            try:
                quad = parse_corners(corners)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

        debug_on = _parse_debug(debug)
        warns = ["low_light"] if float(image.mean()) < LOW_LIGHT_MEAN else []
        if quad is not None and _quad_aspect(quad) < LOW_ANGLE_HW:
            warns.append("very_low_angle")

        frame = Frame(
            board_id=board_id,
            seq=next(snapshot_seq),
            capture_ts=time.time(),
            image=image,
            corners=quad,
        )
        # The one routing decision — the same one the live stream takes (§4.5,
        # :mod:`.routing`), so a photo posted here and the same photo streamed can
        # never be read through different branches.
        route = session.route(frame)
        if isinstance(route, Uncalibrated):
            # No piece-type calibration active but auto-calibration is enabled:
            # the zero-setup path handles this photo entirely.
            return await _uncalibrated_estimate(route, debug_on, warns, sidecar)

        # The read itself, plus any bank bootstrap/refresh it triggers, are real
        # model/embedding work — off the event loop for the same reason /detect's
        # search is (see below): /ingest acks and /viewer must not stall behind a
        # snapshot upload.
        event = await asyncio.to_thread(route.read)
        autocal = refresh = None
        if isinstance(route, PhotoRead):
            autocal = await asyncio.to_thread(
                session.bootstrap_bank, frame, route.corners, event
            )
            if autocal is not None:
                # Answer through the just-upgraded ensemble, not the bare model.
                event = await asyncio.to_thread(autocal.reread, frame, route.corners)
            else:
                refresh = await asyncio.to_thread(
                    session.refresh_bank, frame, route.corners, event
                )
        rotated_deg = None
        if isinstance(route, PhotoRead) and _parse_debug(autorotate):
            from board_reader_core.protocol import PositionEvent

            from board_reader_inference.autocalib import resolve_rotation

            k, rotated = resolve_rotation(np.asarray(event.square_dist))
            if k:
                event = PositionEvent.from_square_dist(
                    board_id=event.board_id,
                    seq=event.seq,
                    utc=event.utc,
                    square_dist=rotated,
                    stability=event.stability,
                    clock=event.clock,
                    status=event.status,
                    orientation=event.orientation,
                )
                rotated_deg = k * 90
        # Repair attached before the publish — the hub's subscribers draw
        # ``prior.placement``, so publishing bare would show an impossible board
        # on the one surface the repair exists for. ``result`` inherits it.
        repair = await asyncio.to_thread(_prior_decode, event)
        event.prior = repair.to_repair()
        await hub_.publish(event)
        result = event.to_dict()
        if autocal is not None:
            result["autocalibrated"] = {
                "matches": autocal.matches,
                "rotation": autocal.rotation,
                "exemplars": autocal.exemplars,
                "saved": autocal.saved,
            }
        if refresh is not None:
            result["bank_refreshed"] = {
                "added": refresh.added,
                "total": refresh.total,
                "agree": refresh.agree,
                "rotation": refresh.rotation,
                "saved": refresh.saved,
            }
        if rotated_deg:
            result["autorotated"] = rotated_deg
        _annotate_upload(
            sidecar,
            placement=event.placement,
            autorotated=rotated_deg,
            prior_flips=[f.square for f in repair.flips],
        )
        if sidecar is not None:
            # Names the retained photo so the client can confirm this read's
            # position as ground truth (POST /labels/{upload}). Absent when
            # retention is off — then there is no photo to attach a label to.
            result["upload"] = sidecar.stem
        if warns:
            result["warnings"] = warns
        if debug_on:
            # Recompute crops/boxes purely for visualisation — the estimator above
            # already cropped internally; debug is opt-in so this extra pass is
            # acceptable. The view comes from the route, so it is always the
            # geometry recognition used. rotate_k keeps it in step with
            # square_dist above when autorotate fired (see _debug_payload).
            result["debug"] = _debug_payload(
                route.view(), event.square_dist, rotate_k=(rotated_deg or 0) // 90
            )
        return JSONResponse(result)

    @app.post("/calibrate")
    async def calibrate(
        request: Request, corners: str, placement: str = START_PLACEMENT
    ) -> JSONResponse:
        """Few-shot per-set calibration from one labeled photo (DESIGN §4.6, E1/E2).

        Body = ORIGINAL photo; ``corners`` = a8,h8,h1,a1 pixel coords; ``placement``
        = the position actually on the board (default: the start position, which
        contains all 13 classes so one photo calibrates the whole vocabulary).
        Builds a kNN exemplar bank from the photo's own lighting (the
        accuracy-validated replacement for centroids, DESIGN §4.6) and hot-swaps
        the running estimator. If the running estimator is already an exemplar
        bank on the same backbone, the new photo's exemplars are ADDED to it
        rather than replacing it — every calibration photo only grows the bank.
        Persistence is opt-in: the bank is written only when ``BR_SESSION_BANK``
        names a path (the curated ``BR_CENTROIDS`` file is never overwritten).
        While ChessQ serves (``BR_CHESSQ``) calibration is refused with 409.
        """
        body = await request.body()
        try:
            image = decode_jpeg(body)
            quad = parse_corners(corners)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        _save_upload("calibrate", body, corners=corners, placement=placement)
        try:
            # Builds/extends an exemplar bank off real embeddings — the same class
            # of blocking work /detect already moved off the loop (see below); run
            # it in a worker thread so /ingest acks and /viewer keep flowing while
            # a calibration photo is processed.
            outcome = await asyncio.to_thread(
                session.calibrate_from, image, quad, placement
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if isinstance(outcome, CalibrationRefused):
            return JSONResponse(
                {
                    "error": (
                        "calibration is off while ChessQ serves "
                        f"(BR_CHESSQ={outcome.model!r}): it reads every piece set "
                        "as shipped, and a calibration bank would replace it rather "
                        "than add to it. Unset BR_CHESSQ to calibrate."
                    )
                },
                status_code=409,
            )
        if isinstance(outcome, CalibrationRejected):
            fit = outcome.rotated_fit
            if fit is not None:
                deg = 90 * fit.quarter_turns
                return JSONResponse(
                    {
                        "error": (
                            "calibration rejected: the board appears ROTATED in "
                            f"the frame — the same photo passes once turned {deg}° "
                            f"({fit.piece_agree}/{outcome.piece_squares} piece "
                            "squares agree after the turn, "
                            f"{outcome.piece_agree}/{outcome.piece_squares} as "
                            "sent). Piece crops extend toward the top of the "
                            "image, so a sideways board cannot be read: hold the "
                            "phone so the board sits upright in the frame "
                            "(usually landscape), then recalibrate."
                        ),
                        "rotated_deg": deg,
                    },
                    status_code=422,
                )
            return JSONResponse(
                {
                    "error": (
                        "calibration rejected: the photo does not match the "
                        f"given position (only {outcome.piece_agree}/"
                        f"{outcome.piece_squares} claimed piece squares agree, even "
                        "at the best rotation). Check that the board shows the "
                        "position being calibrated, the corners sit on the playing "
                        "surface, and the camera is not too flat."
                    )
                },
                status_code=422,
            )
        return JSONResponse(
            {
                "classes_calibrated": outcome.classes_calibrated,
                "backbone": outcome.backbone,
                "saved": outcome.saved,
                "exemplars": outcome.exemplars,
                "label_rotation": outcome.label_rotation,
            }
        )

    @app.post("/labels/{upload_id}")
    async def label(upload_id: str, request: Request) -> JSONResponse:
        """Record the operator's confirmed position for a retained photo.

        The snapshot page's correction panel already has a human fixing wrong
        squares; this is where that answer is kept instead of being discarded on
        reload. Body: ``{"placement": <FEN field 1>, "corrections": <int>}`` —
        the position **as displayed**, whose ``autorotate`` turn is undone by
        :func:`labels.save_label` so the stored truth is in the frame the photo's
        corners define and never depends on a model output.

        Labels are **eval data only**: nothing reads them back into the
        calibration bank. Fitting on them would recreate exactly the
        contamination they exist to measure (see :mod:`.labels`).
        """
        from board_reader_core.protocol import parse_placement

        from board_reader_inference.labels import UPLOAD_ID, save_label

        if not save_dir:
            return JSONResponse(
                {
                    "error": (
                        "photo retention is off (BR_SAVE_UPLOADS unset), so there is "
                        "no photo to label — a label without its photo is not eval data"
                    )
                },
                status_code=422,
            )
        if not UPLOAD_ID.match(upload_id):
            return JSONResponse({"error": "not an upload id"}, status_code=400)
        sidecar = Path(save_dir) / f"{upload_id}.json"
        if not sidecar.exists():
            return JSONResponse({"error": "no such upload"}, status_code=404)
        try:
            payload = json.loads(await request.body())
            placement = str(payload["placement"])
            parse_placement(placement)
            corrections = int(payload.get("corrections", 0))
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse({"error": f"bad label: {e}"}, status_code=400)
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            meta = {}
        if not meta.get("corners"):
            # A placement only means something in a stated frame. Without the
            # corners the photo cannot be re-cropped, so an eval could never
            # replay this board — refuse rather than write a label nothing loads.
            return JSONResponse(
                {
                    "error": (
                        "this read was made without corners, so its position has no "
                        "frame to be true in and no eval could replay it — read the "
                        "photo again with the board corners set"
                    )
                },
                status_code=422,
            )
        try:
            path = save_label(
                Path(save_dir),
                upload_id,
                placement=placement,
                corners=meta.get("corners"),
                predicted=meta.get("placement"),
                corrections=corrections,
                autorotated=int(meta.get("autorotated") or 0),
                # What the decoder repaired in the read the operator just
                # confirmed — so a later eval can tell truth that came from the
                # board apart from truth that came from the decoder.
                prior_flips=[str(s) for s in (meta.get("prior_flips") or [])],
            )
        except OSError as e:
            return JSONResponse({"error": f"could not write label: {e}"}, status_code=500)
        return JSONResponse(
            {
                "saved": str(path),
                "placement": json.loads(path.read_text())["placement"],
                "corrections": corrections,
            }
        )

    @app.get("/uploads")
    async def list_uploads(limit: int = 100) -> JSONResponse:
        """List retained uploads (``BR_SAVE_UPLOADS``), newest first.

        The snapshot page's "Recent positions" list was session-only (lost on
        reload); this makes it server-backed off the same retention the photo
        endpoint below reads from. Retention off -> ``{"retention": false,
        "uploads": []}`` rather than an error status, so the page can render its
        "turn retention on" explainer instead of treating this as a failure.
        """
        if not save_dir:
            return JSONResponse({"retention": False, "uploads": []})

        from board_reader_inference.labels import LABEL_SUFFIX, UPLOAD_ID

        clamped_limit = max(1, min(500, limit))
        d = Path(save_dir)
        rows: list[dict] = []
        for sidecar in d.glob("*.json"):
            stem = sidecar.name[: -len(".json")]
            if not UPLOAD_ID.match(stem):
                continue  # not an estimate/calibrate sidecar — a label sidecar's
                # stem ends in ".label" and never matches this grammar
            if not (d / f"{stem}.jpg").exists():
                continue  # photo missing: nothing to list or serve
            try:
                meta = json.loads(sidecar.read_text())
                ts = meta.get("ts")
                row = {
                    "id": stem,
                    "ts": float(ts) if isinstance(ts, (int, float)) else _id_epoch(stem),
                    "kind": str(meta["kind"]),
                    "corners": meta.get("corners"),
                    "placement": meta.get("placement"),
                    "autorotated": meta.get("autorotated"),
                    "labeled": False,
                    "label_placement": None,
                    "label_corrections": None,
                }
            except (OSError, ValueError, KeyError, TypeError):
                continue  # half-written sidecar: skip rather than fail the request
            label_meta_path = d / f"{stem}{LABEL_SUFFIX}"
            if label_meta_path.exists():
                try:
                    label_meta = json.loads(label_meta_path.read_text())
                    row["labeled"] = True
                    row["label_placement"] = label_meta["placement"]
                    row["label_corrections"] = int(label_meta.get("corrections", 0))
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # label unreadable: report the upload as unlabeled
            rows.append(row)
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return JSONResponse({"retention": True, "uploads": rows[:clamped_limit]})

    @app.get("/uploads/{upload_id}/photo")
    async def upload_photo(
        upload_id: str, max_side: int | None = Query(default=None, alias="max")
    ) -> Response:
        """Serve a retained upload's photo (``BR_SAVE_UPLOADS``).

        ``?max=<px>`` downscales the long side for a thumbnail; omitted, the
        stored bytes go out verbatim. ``upload_id`` is a request parameter, never
        a path fragment — validated against the same stem grammar as ``/labels``
        before it ever touches the filesystem.
        """
        from board_reader_inference.labels import UPLOAD_ID

        if not save_dir:
            return JSONResponse(
                {
                    "error": (
                        "photo retention is off (BR_SAVE_UPLOADS unset), so "
                        "there is no photo to serve"
                    )
                },
                status_code=422,
            )
        if not UPLOAD_ID.match(upload_id):
            return JSONResponse({"error": "not an upload id"}, status_code=400)
        photo = Path(save_dir) / f"{upload_id}.jpg"
        if not photo.exists():
            return JSONResponse({"error": "no such upload"}, status_code=404)
        body = photo.read_bytes()
        if max_side is not None:
            import cv2

            clamped = min(2048, max(32, int(max_side)))
            image = decode_jpeg(body)
            h, w = image.shape[:2]
            long_side = max(h, w)
            if long_side > clamped:
                scale = clamped / long_side
                small = cv2.resize(
                    image,
                    (round(w * scale), round(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    body = buf.tobytes()
        return Response(
            content=body,
            media_type="image/jpeg",
            # Content is immutable per id (uploads are never overwritten), so a
            # thumbnail or the original can be cached hard.
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.post("/detect")
    async def detect(request: Request) -> JSONResponse:
        """Locate the board in a photo -> proposed corners (geometric TL,TR,BR,BL).

        Orientation (which physical corner is a8) is NOT knowable from geometry
        (DESIGN decision #15) — the snapshot page lets the operator rotate the
        labels. Detection logic lives in core so the edge (live pipe) and this
        snapshot ingress share one implementation.
        """
        body = await request.body()
        try:
            image = decode_jpeg(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        try:
            from board_reader_core.boarddetect import detect_board_quad
        except ImportError:
            return JSONResponse({"error": "detector unavailable"}, status_code=422)
        # A miss (no board in frame) exhausts every coarse candidate and can run
        # multiple seconds; run it off the event loop so /ingest acks and the
        # /viewer hub keep flowing while this photo is searched.
        quad = await asyncio.to_thread(detect_board_quad, image)
        if quad is None:
            return JSONResponse({"error": "no board found"}, status_code=422)
        return JSONResponse({"corners": np.asarray(quad, dtype=float).tolist()})

    @app.websocket("/ingest")
    async def ingest(ws: WebSocket) -> None:
        """Edge -> inference frame ingress. Header (text) then JPEG (binary) per frame.

        Estimation is handed to the per-board ingest worker (WP-G1/G2) so a burst
        never blocks the event loop: we ack the moment a frame is decoded. The ack
        means "received" — the pre-filter may drop this frame for a fresher one
        before it is ever estimated.
        """
        await ws.accept()
        boards: set[str] = set()
        warned_refusals: set[str] = set()
        try:
            while True:
                raw = await ws.receive_text()
                jpeg = await ws.receive_bytes()
                try:
                    header = json.loads(raw)
                    frame = Frame.from_header(header, decode_jpeg(jpeg))
                except (ValueError, TypeError, KeyError) as e:
                    # One bad frame (malformed corners, truncated JPEG, missing
                    # header field) must not kill the whole stream: a browser
                    # edge would reconnect-loop forever on its own bug. Refuse
                    # the frame in the ack slot and keep the socket. Counted, and
                    # logged once per distinct reason per connection — a stream
                    # whose every frame is refused must not be indistinguishable
                    # from one that never sent (the reason only ever reached the
                    # phone's banner), but a 12 fps header bug must not bury the
                    # read lines under 43k identical warnings an hour either.
                    ingest_.frames_refused += 1
                    reason = str(e)
                    if reason not in warned_refusals:
                        warned_refusals.add(reason)
                        log.warning("refused ingest frame: %s", reason)
                    await ws.send_json({"error": f"bad frame: {e}"})
                    continue
                boards.add(frame.board_id)
                # The frame's own header, not the client's: retention and the
                # recording then hold values the wire has already validated and
                # normalised (corners as 8 floats, seq an int), never whatever
                # JSON arrived.
                normalised = frame.header()
                latest_frames[frame.board_id] = (normalised, jpeg)
                # Record what ARRIVED, before the latest-wins pre-filter has had a
                # chance to drop it: a recording is the input to a run, not a log
                # of what the estimator got round to.
                _record(normalised, jpeg)
                ingest_.submit(frame)
                await ws.send_json({"ack": frame.seq, "board_id": frame.board_id})
        except WebSocketDisconnect:
            # Let the stream's last frame land before the connection is forgotten.
            # The workers outlive the socket, so this only waits — it never cancels.
            await ingest_.drain(*boards)

    @app.websocket("/viewer")
    async def viewer(ws: WebSocket) -> None:
        """Inference -> viewer position-event egress."""
        await ws.accept()
        try:
            async for event in hub_.subscribe():
                await ws.send_text(event.to_json())
        except WebSocketDisconnect:
            return

    return app


# Deliberately NO module-level ``app = create_app()``. The package imports this
# module (``__init__`` re-exports create_app), so a module-level app would be
# built at *import* time — before the CLI has set ``BR_RECORD_DIR``/
# ``BR_CENTROIDS`` — and ``serve`` would then run a server whose recording and
# estimator silently ignore its own flags. That is not hypothetical: it shipped,
# and every ``make dev`` trial ran unrecorded. Serve with
# ``uvicorn board_reader_inference.server:create_app --factory`` instead.
