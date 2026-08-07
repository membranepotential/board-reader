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

import contextlib
import datetime
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from board_reader_core.boarddetect import LOW_ANGLE_ASPECT, quad_aspect
from board_reader_core.protocol import GRID, START_PLACEMENT
from board_reader_core.recording import FrameRecorder
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.estimator import CentroidEstimator, Estimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import FrameIngest
from board_reader_inference.routing import PhotoRead, ReadView, Uncalibrated
from board_reader_inference.session import (
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


def _parse_corners(spec: str) -> np.ndarray:
    """``x0,y0,...,x3,y3`` -> a 4x2 float32 quad; raises ValueError on bad input."""
    nums = [float(x) for x in spec.split(",") if x.strip()]
    if len(nums) != 8:
        raise ValueError("corners needs 8 comma-separated numbers")
    return np.array(nums, dtype=np.float32).reshape(4, 2)


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


def _default_estimator(occupancy_est: CentroidEstimator | None = None) -> Estimator:
    """Pick the style (piece-type) estimator from the environment.

    ``BR_CENTROIDS=<path>`` -> that calibration. Unset but auto-calibration
    enabled (an occupancy estimator exists) -> restore the persisted session bank
    if a previous run calibrated one. Otherwise the M0 :class:`StubEstimator`.
    Resolved at import time so the uvicorn string-imported ``app`` honours it
    without a code change.
    """
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
    if model:
        from board_reader_inference.estimator import EnsembleEstimator
        from board_reader_inference.trained import TrainedEstimator

        try:
            trained = TrainedEstimator(model)
        except Exception as e:
            # file, ORT version, truncated ONNX) degrades to the next tier so
            # `docker compose up` works on a fresh clone without models/.
            print(f"BR_MODEL={model!r} not usable ({e}); serving without it")
            trained = None
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
    ingest_ = FrameIngest(session.route, hub_)
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
            },
        }
        if recorder is not None:
            health["recording"] = recorder.stats()
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

        reading = route.read()
        auto = session.adopt_start(route, reading)
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

        event = auto.reread(route.frame, route.corners)
        await hub_.publish(event)
        result = event.to_dict()
        # This branch answers with real piece types, so it gets the same repair as
        # the calibrated path. (The occupancy-only branch above does not: with no
        # piece identity and no colour there is nothing the constraints can read.)
        # ``auto.rotation`` *would* be an established frame here — it comes from
        # the start-position occupancy pattern, not from the circular pawn
        # heuristic — but a start position has no pawn on rank 1/8 by
        # construction, so the one frame-dependent constraint could never fire.
        # Passing it would add an untestable convention for zero gain.
        repair = _prior_decode(event)
        result["prior"] = repair.to_dict()
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
                quad = _parse_corners(corners)
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

        event = route.read()
        autocal = refresh = None
        if isinstance(route, PhotoRead):
            autocal = session.bootstrap_bank(frame, route.corners, event)
            if autocal is not None:
                # Answer through the just-upgraded ensemble, not the bare model.
                event = autocal.reread(frame, route.corners)
            else:
                refresh = session.refresh_bank(frame, route.corners, event)
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
        await hub_.publish(event)
        result = event.to_dict()
        repair = _prior_decode(event)
        result["prior"] = repair.to_dict()
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
        """
        body = await request.body()
        try:
            image = decode_jpeg(body)
            quad = _parse_corners(corners)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        _save_upload("calibrate", body, corners=corners, placement=placement)
        try:
            outcome = session.calibrate_from(image, quad, placement)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if isinstance(outcome, CalibrationRejected):
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
        quad = detect_board_quad(image)
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
                    # the frame in the ack slot and keep the socket.
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


# Module-level app for ``uvicorn board_reader_inference.server:app``.
app = create_app()
