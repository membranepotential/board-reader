"""Inference gateway (DESIGN.md §4.5) — FastAPI WebSocket ingress + egress.

Runs on the inference node. Its surfaces:

* ``WS /ingest`` — the edge connects and streams frames. Each frame is two WS
  messages: a JSON header then the binary JPEG. The gateway decodes and acks, and
  the per-board ingest worker (:mod:`board_reader_inference.stream.ingest`) estimates off
  the event loop and publishes the position event to the hub.
* ``WS /viewer`` — viewers connect and receive the position-event JSON stream
  (retained latest per board first, then live updates), interleaved with
  ``{"type": "motion"}`` updates per ingested frame (display state, never retained).
* ``GET /boards/{id}`` + ``GET /boards/{id}/frame`` + ``GET /healthz`` — latest
  position snapshot, latest raw ingested JPEG, and liveness.
* ``WS /boards/{id}/camera`` + ``WS /boards/{id}/control`` — the camera control
  channel: the control UI's config for each board's camera, pushed to the phone,
  and the phone's state, previews and the node's health pushed to the control UI
  (:mod:`board_reader_inference.control.camera`).

The gateway holds the recognizer and the position hub. It never imports the
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
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from board_reader_core.control import Rules
from board_reader_core.protocol import PositionEvent
from board_reader_core.recording import FrameRecorder
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.control.camera import (
    CameraRegistry,
    NodeStatus,
    StreamedFrame,
)
from board_reader_inference.control.camera import router as camera_router
from board_reader_inference.measurement.labels import READ_FRAME
from board_reader_inference.recognition.recognizer import (
    FrameReading,
    Recognizer,
    StubRecognizer,
    build_from_env,
    frame_reader,
)
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import FrameIngest, drop_line, event_line
from board_reader_inference.stream.schedule import (
    EveryFrame,
    FramePolicy,
    ReadKind,
    parse_policy,
)
from board_reader_inference.stream.stabilize import SettleConfig

log = logging.getLogger(__name__)


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


# Ingress recording is opt-in (``BR_RECORD_DIR=<dir>``), but the ``serve``
# command opts in on the operator's behalf — see create_app.
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


def _frame_policy() -> Callable[[], FramePolicy]:
    """``BR_SCHEDULE`` -> a factory for each board's read scheduler.

    The value is a policy spec string (``burst-end:burst_end_reads=3,spacing_s=0.6``,
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


@dataclass(frozen=True, slots=True)
class _Retained:
    """The latest frame ``/ingest`` accepted for one board, kept for previews."""

    header: dict
    streamed: StreamedFrame


def create_app(
    recognizer: Recognizer | None = None,
    hub: PositionHub | None = None,
) -> FastAPI:
    """Build the gateway app. Recognizer and hub are injectable for tests."""
    recognizer_ = recognizer if recognizer is not None else build_from_env()
    read = frame_reader(recognizer_)
    hub_: PositionHub = hub if hub is not None else PositionHub()

    # The camera control channel (DESIGN D11, "dumb camera"): the control UI sets
    # each board's camera config here and the node pushes it to the phone.
    cameras = CameraRegistry()

    def _on_published(reading: FrameReading, frame: Frame, rules: Rules) -> None:
        """Observers of every published live read (on the event loop).

        The board's direction goes to the camera registry, which tells /control
        which side White sits on. Only reads the orientation guard let through
        get here, and replay never does: it is a live-camera fact.
        """
        _record_published(reading.event, frame, rules)
        if reading.direction is not None:
            cameras.observe_direction(frame.board_id, reading.direction, frame.capture_ts)

    def _record_published(event: PositionEvent, frame: Frame, rules: Rules) -> None:
        """One compact line per published event into the session's events.ndjson."""
        if recorder is None or not recorder.active:
            return
        line = event_line(event, frame, published_ts=time.time(), rules=rules)
        if not recorder.record_event(line):
            print(f"frame recording stopped: {recorder.stopped_reason}")

    # Streaming ingress (WP-G1/G2): frames are read off the event loop by
    # per-board workers, through the same read ``POST /estimate`` runs.
    # The debounce's knobs (BR_SETTLE_*; unset keeps stabilize.py's defaults). A
    # malformed value fails here, at startup, like a malformed BR_SCHEDULE.
    settle = SettleConfig.from_env()
    if not settle.is_default:
        print(f"settling with {settle.to_dict()}")

    def _record_dropped(frame: Frame, kind: ReadKind, rules: Rules, waited_s: float) -> None:
        """A scheduled read the overload rule dropped, into events.ndjson too."""
        if recorder is None or not recorder.active:
            return
        line = drop_line(frame, kind, rules=rules, waited_s=waited_s, published_ts=time.time())
        if not recorder.record_event(line):
            print(f"frame recording stopped: {recorder.stopped_reason}")

    ingest_ = FrameIngest(
        read,
        hub_,
        on_event=_on_published,
        on_drop=_record_dropped,
        policy=_frame_policy(),
        settle=settle,
        # Each board settles under its own /control setting, read per read.
        rules=cameras.rules,
    )
    # Latest-wins raw-frame retention, one entry per board, in-memory only (no
    # disk write, unbounded board count — mirrors the hub's own retention). This
    # is what ``GET /boards/{id}/frame`` serves for the live video-like preview;
    # it is independent of the hub's estimated PositionEvent.
    latest_frames: dict[str, _Retained] = {}

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
    # have cost us real user state before.
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
        expose_headers=["X-Seq", "X-Capture-Ts"],
    )
    app.state.recognizer = recognizer_
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

    # Why the stub serves instead of the model, when it does (D2: degrade loudly).
    fallback = recognizer_.fallback if isinstance(recognizer_, StubRecognizer) else None

    @app.get("/healthz")
    async def healthz() -> dict:
        # The recording stats ride on liveness so an operator can confirm frames
        # are landing on disk without shelling into the container. The ingest
        # counters ride along for the same reason: "frames are arriving but none
        # are being estimated" is otherwise invisible from outside the process —
        # the stream simply looks quiet.
        health = {
            "status": "ok",
            "subscribers": hub_.subscriber_count,
            "ingest": {
                "received": ingest_.frames_received,
                "estimated": ingest_.frames_estimated,
                "dropped": ingest_.frames_dropped,
                "failed": ingest_.frames_failed,
                "refused": ingest_.frames_refused,
                # The scheduler's two counters. "unscheduled" is the saving —
                # at the shipped burst-end policy it is most of the stream, and
                # "estimated barely moves while received climbs" is the expected
                # picture, not a stall. "backstopped" should be 0: anything else
                # says an edge's capture_ts stopped advancing and the node is
                # reading on its own clock to stay alive.
                "unscheduled": ingest_.frames_unscheduled,
                "backstopped": ingest_.frames_backstopped,
                # Reads the orientation guard dropped: the board pointed more
                # than 45 degrees from its recent reads. Should be 0.
                "misoriented": ingest_.frames_misoriented,
                # Of "dropped", burst-end reads: the overload rule drops
                # keepalive reads first, so anything but 0 means a board had a
                # whole backlog of move evidence waiting.
                "dropped_burst_end": ingest_.frames_dropped_burst_end,
                # The read queue every board shares, oldest first: its waits
                # grow with boards x read time, its results do not.
                "queue": ingest_.queue_stats().to_dict(),
            },
            "settle": settle.to_dict(),
        }
        if recorder is not None:
            health["recording"] = recorder.stats()
        # Which recognizer reads, from which file — and, when the weights were
        # missing, that the stub is serving instead and why (D2: degrade loudly).
        health["recognizer"] = {
            "name": recognizer_.name,
            "model": recognizer_.model_path,
            **({"fallback": fallback} if fallback else {}),
        }
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
        actually sent, independent of whatever the recognizer made of it.
        """
        entry = latest_frames.get(board_id)
        if entry is None:
            return JSONResponse({"error": "no frame for board"}, status_code=404)
        headers = {
            "X-Seq": str(entry.header["seq"]),
            "X-Capture-Ts": str(entry.header["capture_ts"]),
            "Cache-Control": "no-store",
        }
        return Response(
            content=entry.streamed.jpeg, media_type="image/jpeg", headers=headers
        )

    def _node_status(board_id: str) -> NodeStatus:
        entry = latest_frames.get(board_id)
        return NodeStatus(
            recognizer=recognizer_.name,
            fallback=fallback,
            received=ingest_.frames_received,
            refused=ingest_.frames_refused,
            frame=None if entry is None else entry.streamed,
        )

    app.state.cameras = cameras
    app.include_router(camera_router(cameras, _node_status))

    # One-shot HTTP ingress for the phone snapshot page (viewer /snapshot): the
    # body is the photo as shot, and the recognizer finds the board in it.
    snapshot_seq = itertools.count()

    # Opt-in raw-photo retention (``BR_SAVE_UPLOADS=<dir>``): every decodable
    # photo POSTed to /estimate is kept with its request metadata, so real-world
    # shots become labeled eval data instead of being
    # discarded after the response. Off by default (tests, containers).
    save_dir = os.environ.get("BR_SAVE_UPLOADS")
    upload_seq = itertools.count()

    def _save_upload(
        kind: str,
        body: bytes,
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

    @app.post("/estimate")
    async def estimate(request: Request, board_id: str = "snapshot") -> JSONResponse:
        body = await request.body()
        try:
            image = decode_jpeg(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        sidecar = _save_upload("estimate", body)
        warns = ["low_light"] if float(image.mean()) < LOW_LIGHT_MEAN else []
        frame = Frame(
            board_id=board_id, seq=next(snapshot_seq), capture_ts=time.time(), image=image
        )
        # The same read the live stream runs. It is real model work — off the event
        # loop, so /ingest acks and /viewer never stall behind a snapshot upload.
        event = (await asyncio.to_thread(read, frame)).event
        await hub_.publish(event)
        result = event.to_dict()
        _annotate_upload(sidecar, frame=READ_FRAME, placement=event.placement)
        if sidecar is not None:
            # Names the retained photo so the client can confirm this read's
            # position as ground truth (POST /labels/{upload}). Absent when
            # retention is off — then there is no photo to attach a label to.
            result["upload"] = sidecar.stem
        if warns:
            result["warnings"] = warns
        return JSONResponse(result)

    @app.post("/labels/{upload_id}")
    async def label(upload_id: str, request: Request) -> JSONResponse:
        """Record the operator's confirmed position for a retained photo.

        The snapshot page's correction panel already has a human fixing wrong
        squares; this is where that answer is kept instead of being discarded on
        reload. Body: ``{"placement": <FEN field 1>, "corrections": <int>}`` —
        the position **as displayed**. The label is written in the frame the read
        was made in (the upload sidecar's ``frame``): a recognizer's read is in
        the chess frame and needs no corners; an older corner-frame read needs its
        corners, and its ``autorotate`` turn is undone by :func:`labels.save_label`
        so the stored truth never depends on a model output.

        Labels are **eval data only**: nothing reads them back into the
        calibration bank. Fitting on them would recreate exactly the
        contamination they exist to measure (see :mod:`.labels`).
        """
        from board_reader_core.protocol import parse_placement

        from board_reader_inference.measurement.labels import (
            UPLOAD_ID,
            parse_frame,
            save_label,
        )

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
            frame = parse_frame(meta.get("frame"))
        except (OSError, ValueError):
            meta, frame = {}, "corners"
        if frame == "corners" and not meta.get("corners"):
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
                frame=frame,
                placement=placement,
                corners=meta.get("corners"),
                predicted=meta.get("placement"),
                corrections=corrections,
                autorotated=int(meta.get("autorotated") or 0),
            )
        except OSError as e:
            return JSONResponse({"error": f"could not write label: {e}"}, status_code=500)
        return JSONResponse(
            {
                "saved": str(path),
                "placement": json.loads(path.read_text())["placement"],
                "frame": frame,
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

        from board_reader_inference.measurement.labels import LABEL_SUFFIX, UPLOAD_ID

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
                    "placement": meta.get("placement"),
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
        from board_reader_inference.measurement.labels import UPLOAD_ID

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
                height, width = frame.image.shape[:2]
                latest_frames[frame.board_id] = _Retained(
                    header=normalised,
                    streamed=StreamedFrame(
                        seq=frame.seq,
                        received_at=time.time(),
                        width=width,
                        height=height,
                        jpeg=jpeg,
                    ),
                )
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
        """Inference -> viewer egress: position events and motion updates."""
        await ws.accept()
        try:
            async for message in hub_.subscribe():
                await ws.send_text(message.to_json())
        except WebSocketDisconnect:
            return

    return app


# Deliberately NO module-level ``app = create_app()``. The package imports this
# module (``__init__`` re-exports create_app), so a module-level app would be
# built at *import* time — before the CLI has set ``BR_RECORD_DIR``/
# ``BR_SCHEDULE`` — and ``serve`` would then run a server that silently ignores
# its own flags. That is not hypothetical: it shipped,
# and every ``make dev`` trial ran unrecorded. Serve with
# ``uvicorn board_reader_inference.server:create_app --factory`` instead.
