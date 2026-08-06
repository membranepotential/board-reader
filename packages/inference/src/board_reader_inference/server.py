"""Inference gateway (DESIGN.md §4.5) — FastAPI WebSocket ingress + egress.

Runs on the inference node. Three surfaces:

* ``WS /ingest`` — the edge connects and streams frames. Each frame is two WS
  messages: a JSON header then the binary JPEG. The gateway decodes, runs the
  estimator, and publishes the resulting position event to the hub.
* ``WS /viewer`` — viewers connect and receive the position-event JSON stream
  (retained latest per board first, then live updates).
* ``GET /boards/{id}`` + ``GET /healthz`` — latest snapshot + liveness.

The gateway holds the shared estimator and the position hub. It never imports the
edge package; it only speaks the wire (:mod:`board_reader_core.wire`).
"""

from __future__ import annotations

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

from board_reader_core.protocol import GRID, START_PLACEMENT
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.estimator import Estimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.squares import (
    crop_squares_photo,
    full_frame_corners,
    photo_square_boxes,
    rectify_photo,
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
        return datetime.datetime.strptime(date + clock, "%Y%m%d%H%M%S").timestamp()
    except (ValueError, IndexError):
        return 0.0


# Session-bank persistence is OPT-IN via ``BR_SESSION_BANK=<path>``: /calibrate,
# auto-calibration, and the calibrated-path refresh persist there, and an
# autocalib-enabled server restores it on startup. Unset -> calibration lives in
# memory only and NOTHING is ever written. Opt-in is deliberate: a cwd-relative
# default made ordinary test runs silently overwrite the real bank — twice.
def _session_bank_path() -> str | None:
    return os.environ.get("BR_SESSION_BANK") or None

# Below this mean pixel value a photo is dim enough that recognition measurably
# degrades (evening-room shots measured ~37/255) — responses carry a warning.
LOW_LIGHT_MEAN = 55.0

# Below this projected board aspect (short/long side of the corner quad) the
# camera is so close to board level that pieces occlude whole ranks and crops
# mostly contain the *next* rank's piece — recognition collapses (measured
# 2026-07-18: near-random output at 0.56 vs usable at 0.73+). Warn, don't block.
LOW_ANGLE_HW = 0.62

# /calibrate sanity floor: of the squares the claimed placement says hold a
# PIECE, at least this fraction must agree with a stateless probe of the photo
# (best rotation). Empty squares are excluded — a mid-game board and the start
# position share a mostly-empty middle, which inflates whole-board agreement
# (~40/64) and hides the mismatch. Below the floor the photo is not showing the
# claimed position (mid-game sent as "start", corners off the surface, unusably
# flat angle) and calibrating would poison the bank — reject (incidents #3, #4).
CALIBRATE_MIN_PIECE_AGREE = 0.6


def _booster_weight(bank_exemplars: int) -> float:
    """Trained-model share of the ensemble blend, by bank maturity.

    A mature bank (many calibrates/appends) is the stronger member under its
    own conditions — lean on it (measured 2026-07-18 on the 10-board eval:
    30/70 = 633/640 vs 50/50 = 625). A thin bootstrap bank (one photo) hasn't
    earned that trust — blend evenly.
    """
    return 0.3 if bank_exemplars >= 1024 else 0.5


def _quad_aspect(quad: np.ndarray) -> float:
    """Projected board aspect: short axis / long axis of the corner quad, in
    (0, 1]. Rotation-of-labeling agnostic (both axis ratios tried); ~1 means
    top-down, small values mean a nearly board-level camera."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if w <= 0 or h <= 0:
        return 1.0
    return float(min(h / w, w / h))


def _estimator_from_bank(path: str, *, share_from: "Estimator | None" = None):
    """Build a :class:`CentroidEstimator` from a saved calibration file.

    Rebuilds the same backbone and box geometry the bank was calibrated with
    (dims must match; photo boxes must be cut identically). If ``share_from`` is
    an estimator on the same backbone, its embedder is reused — one ONNX session
    instead of two, which matters on a 4GB GPU.
    """
    from board_reader_inference.centroid import load_classifier
    from board_reader_inference.embed import make_embedder
    from board_reader_inference.estimator import CentroidEstimator

    classifier = load_classifier(path)
    shared = getattr(share_from, "embedder", None)
    shared_backbone = getattr(getattr(share_from, "classifier", None), "backbone", None)
    embedder = shared if shared is not None and shared_backbone == classifier.backbone else None
    return CentroidEstimator(
        embedder if embedder is not None else make_embedder(classifier.backbone),
        classifier,
        headroom=classifier.box_headroom,
        side_pad=classifier.box_side_pad,
    )


def _default_estimator(occupancy_est: "Estimator | None" = None) -> Estimator:
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
        except Exception as e:  # noqa: BLE001 — boundary: any load failure (missing
            # file, ORT version, truncated ONNX) degrades to the next tier so
            # `docker compose up` works on a fresh clone without models/.
            print(f"BR_MODEL={model!r} not usable ({e}); serving without it")
            trained = None
        if trained is not None:
            session = _session_bank_path()
            if session and Path(session).exists():
                try:
                    bank_est = _estimator_from_bank(session, share_from=occupancy_est)
                    return EnsembleEstimator(
                        trained,
                        bank_est,
                        weight=_booster_weight(bank_est.classifier.exemplars.shape[0]),
                    )
                except (OSError, ValueError, KeyError):
                    pass  # unreadable/stale bank: zero-shot still works without it
            return trained
    session = _session_bank_path()
    if occupancy_est is not None and session and Path(session).exists():
        return _estimator_from_bank(session, share_from=occupancy_est)
    return StubEstimator()


def create_app(
    estimator: Estimator | None = None,
    hub: PositionHub | None = None,
    occupancy_estimator: Estimator | None = None,
) -> FastAPI:
    """Build the gateway app. Estimator/hub/occupancy are injectable for tests.

    ``occupancy_estimator`` (or ``BR_OCCUPANCY_BANK=<path>``, e.g. the ChessReD
    exemplar bank) enables the zero-setup path: while no piece-type calibration is
    active, ``/estimate`` answers with style-robust occupancy only — and the
    moment a photo's occupancy matches the chess starting position, the server
    auto-calibrates piece types from that photo (DESIGN §4.6).
    """
    occ_est: Estimator | None = occupancy_estimator
    if occ_est is None and os.environ.get("BR_OCCUPANCY_BANK"):
        occ_est = _estimator_from_bank(os.environ["BR_OCCUPANCY_BANK"])
    est: Estimator = estimator if estimator is not None else _default_estimator(occ_est)
    hub_: PositionHub = hub if hub is not None else PositionHub()

    app = FastAPI(title="board-reader inference node")
    # The viewer is served from a different origin (its own container), and the
    # browser connects here over WS; allow it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("BR_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.estimator = est
    app.state.hub = hub_

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "subscribers": hub_.subscriber_count}

    @app.get("/boards/{board_id}")
    async def board_latest(board_id: str) -> JSONResponse:
        event = hub_.latest(board_id)
        if event is None:
            return JSONResponse({"error": "no such board"}, status_code=404)
        return JSONResponse(event.to_dict())

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

    def _photo_estimator():
        """The active estimator's photo path, if calibrated/trained for it.

        Photo-mode is declared either by the classifier (kNN/centroid banks) or
        by the estimator itself (the trained crop classifier).
        """
        photo_mode = (
            getattr(getattr(est, "classifier", None), "crop_mode", None) == "photo"
            or getattr(est, "crop_mode", None) == "photo"
        )
        if photo_mode:
            fn = getattr(est, "estimate_photo", None)
            if callable(fn):
                return fn
        return None

    def _base_trained():
        """The zero-shot trained member of the active estimator, if any."""
        from board_reader_inference.estimator import EnsembleEstimator
        from board_reader_inference.trained import TrainedEstimator

        if isinstance(est, EnsembleEstimator):
            return est.primary
        return est if isinstance(est, TrainedEstimator) else None

    def _centroid_part():
        """The per-set exemplar/centroid member of the active estimator, if any."""
        from board_reader_inference.estimator import CentroidEstimator, EnsembleEstimator

        if isinstance(est, EnsembleEstimator):
            return est.booster if isinstance(est.booster, CentroidEstimator) else None
        return est if isinstance(est, CentroidEstimator) else None

    def _save_session(classifier) -> str | None:
        """Persist the session bank iff ``BR_SESSION_BANK`` opts in (best-effort)."""
        path = _session_bank_path()
        if path is None:
            return None
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            classifier.save(path)
        except OSError:
            return None
        return path

    def _debug_payload(image, corners, square_dist, *, headroom, side_pad, size, rotate_k=0):
        """The opt-in ``?debug=1`` visualisation pair, from whatever a branch saw.

        ``rotate_k`` is the same ``np.rot90`` turn count ``autorotate`` applied to
        ``square_dist``. Boxes/crops are sampled from ``corners`` exactly as
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

        boxes = photo_square_boxes(corners, headroom=headroom, side_pad=side_pad)
        crops = crop_squares_photo(
            image, corners, headroom=headroom, side_pad=side_pad, size=size
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
            "boxes": to_b64_jpeg(draw_boxes(image, boxes, square_dist)),
            "mosaic": to_b64_jpeg(crop_mosaic(crops, square_dist)),
        }

    def _maybe_refresh_bank(frame: Frame, corners: np.ndarray, event) -> dict | None:
        """Calibrated-path self-refresh (DESIGN §4.6): seeing the START position
        read at drifted (low) confidence appends label-confirmed exemplars from
        this photo, so the bank tracks new lighting without the user calibrating.
        Gates: exemplar-bank estimator only; mean confidence below
        ``autocalib.APPEND_MAX_CONF``; prediction agrees with the (rotated) start
        placement on ≥ ``APPEND_MIN_AGREE`` squares (only those squares are
        learned, so mid-game photos and misreads can't pollute the bank); bank
        below ``APPEND_BANK_CAP``.
        """
        nonlocal est
        from board_reader_core.protocol import CLASSES, position_confidence
        from board_reader_inference import autocalib
        from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
        from board_reader_inference.estimator import CentroidEstimator, EnsembleEstimator

        cent = _centroid_part()
        if cent is None or not isinstance(cent.classifier, ExemplarClassifier):
            return None
        if cent.classifier.exemplars.shape[0] >= autocalib.APPEND_BANK_CAP:
            return None
        if position_confidence(event.square_dist) >= autocalib.APPEND_MAX_CONF:
            return None
        dist = np.asarray(event.square_dist).reshape(-1, len(CLASSES))
        pred = [CLASSES[i] for i in dist.argmax(axis=1)]
        rotation, agree, mask = autocalib.match_predicted_start(pred)
        if agree < autocalib.APPEND_MIN_AGREE:
            return None
        crops = crop_squares_photo(
            frame.image,
            corners,
            headroom=cent.headroom,
            side_pad=cent.side_pad,
            size=cent.square_size or 224,
        )
        new = build_exemplars(
            cent.embedder,
            crops,
            autocalib.start_placement_for_rotation(rotation),
            mask=mask,
            backbone=cent.classifier.backbone,
            box_headroom=cent.headroom,
            box_side_pad=cent.side_pad,
        )
        classifier = cent.classifier.extended(new.exemplars, new.labels)
        refreshed = CentroidEstimator(
            cent.embedder,
            classifier,
            headroom=cent.headroom,
            side_pad=cent.side_pad,
            square_size=cent.square_size,
            orientation=cent.orientation,
        )
        base = _base_trained()
        est = (
            EnsembleEstimator(
                base, refreshed, weight=_booster_weight(classifier.exemplars.shape[0])
            )
            if base is not None
            else refreshed
        )
        app.state.estimator = est
        return {
            "added": int(new.exemplars.shape[0]),
            "total": int(classifier.exemplars.shape[0]),
            "agree": agree,
            "rotation": rotation,
            "saved": _save_session(classifier),
        }

    def _maybe_bootstrap_bank(frame: Frame, corners: np.ndarray, event) -> dict | None:
        """Zero-tap per-set calibration ON TOP of the zero-shot model (§4.6).

        When the trained model runs WITHOUT a booster bank and its own reading
        matches the start position, build the first exemplar bank from this very
        photo and upgrade to the ensemble — same trigger and pollution guard as
        auto-calibration (only label-agreeing squares are learned), but labeled
        by the trained model's full 13-class reading instead of occupancy alone.
        """
        nonlocal est
        from board_reader_core.protocol import CLASSES
        from board_reader_inference import autocalib
        from board_reader_inference.centroid import build_exemplars
        from board_reader_inference.estimator import CentroidEstimator, EnsembleEstimator

        base = _base_trained()
        if base is None or _centroid_part() is not None:
            return None
        dist = np.asarray(event.square_dist).reshape(-1, len(CLASSES))
        pred = [CLASSES[i] for i in dist.argmax(axis=1)]
        rotation, agree, mask = autocalib.match_predicted_start(pred)
        if agree < autocalib.APPEND_MIN_AGREE:
            return None
        if isinstance(occ_est, CentroidEstimator):
            embedder, backbone = occ_est.embedder, occ_est.classifier.backbone
        else:
            from board_reader_inference.embed import make_embedder

            backbone = "dinov2"
            try:
                embedder = make_embedder(backbone)
            except (OSError, FileNotFoundError):
                # No embedder available (e.g. DINOv2 weights not shipped): the
                # zero-shot answer stands on its own, just without the booster.
                return None
        crops = crop_squares_photo(
            frame.image,
            corners,
            headroom=base.headroom,
            side_pad=base.side_pad,
            size=base.square_size or 224,
        )
        classifier = build_exemplars(
            embedder,
            crops,
            autocalib.start_placement_for_rotation(rotation),
            mask=mask,
            backbone=backbone,
            box_headroom=base.headroom,
            box_side_pad=base.side_pad,
        )
        bank_est = CentroidEstimator(
            embedder, classifier, headroom=base.headroom, side_pad=base.side_pad
        )
        est = EnsembleEstimator(
            base, bank_est, weight=_booster_weight(classifier.exemplars.shape[0])
        )
        app.state.estimator = est
        return {
            "matches": agree,
            "rotation": rotation,
            "exemplars": int(classifier.exemplars.shape[0]),
            "saved": _save_session(classifier),
        }

    async def _uncalibrated_estimate(
        frame: Frame,
        corners: np.ndarray,
        debug_on: bool,
        warns: list[str],
        sidecar: Path | None = None,
    ):
        """Zero-setup path: occupancy-only readout + start-position auto-calibration.

        Piece type does not transfer across piece styles on frozen features
        (DESIGN §4.6), so with no style calibration active we answer with
        occupancy only (style-robust, HTTP-response-only — never published to the
        hub, which carries real position events exclusively). But if this photo's
        occupancy matches the chess STARTING position, we calibrate piece types
        from the photo itself via the exemplar-bank machinery and answer with a
        full, published position event — the user never pressed anything.
        """
        nonlocal est
        from board_reader_inference.autocalib import detect_start, occupancy_grid
        from board_reader_inference.centroid import build_exemplars
        from board_reader_inference.estimator import CentroidEstimator

        assert occ_est is not None
        size = occ_est.square_size or 224
        crops = crop_squares_photo(
            frame.image,
            corners,
            headroom=occ_est.headroom,
            side_pad=occ_est.side_pad,
            size=size,
        )
        dist = occ_est.classifier.classify_board(occ_est.embedder.embed(crops))
        match = detect_start(dist, crops)
        if match is None:
            result = {
                "board_id": frame.board_id,
                "mode": "occupancy",
                "occupied": occupancy_grid(dist).tolist(),
                "hint": (
                    "uncalibrated for this piece set: showing occupancy only — "
                    "photograph the starting position once to enable full recognition"
                ),
            }
            if warns:
                result["warnings"] = warns
            if debug_on:
                result["debug"] = _debug_payload(
                    frame.image, corners, dist,
                    headroom=occ_est.headroom, side_pad=occ_est.side_pad, size=size,
                )
            return JSONResponse(result)

        classifier = build_exemplars(
            occ_est.embedder,
            crops,
            match.placement,
            mask=match.calib_mask,
            backbone=occ_est.classifier.backbone,
            box_headroom=occ_est.headroom,
            box_side_pad=occ_est.side_pad,
        )
        est = CentroidEstimator(
            occ_est.embedder,
            classifier,
            headroom=occ_est.headroom,
            side_pad=occ_est.side_pad,
        )
        app.state.estimator = est
        saved = _save_session(classifier)
        event = est.estimate_photo(frame, corners)
        await hub_.publish(event)
        result = event.to_dict()
        result["autocalibrated"] = {
            "matches": match.matches,
            "rotation": match.rotation,
            "exemplars": int(classifier.exemplars.shape[0]),
            "saved": saved,
        }
        # This branch answers with a real position too, so it is labelable like any
        # other read (see /labels).
        _annotate_upload(sidecar, placement=event.placement)
        if sidecar is not None:
            result["upload"] = sidecar.stem
        if warns:
            result["warnings"] = warns
        if debug_on:
            result["debug"] = _debug_payload(
                frame.image, corners, event.square_dist,
                headroom=occ_est.headroom, side_pad=occ_est.side_pad, size=size,
            )
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
        photo_fn = _photo_estimator()
        warns = ["low_light"] if float(image.mean()) < LOW_LIGHT_MEAN else []
        if quad is not None and _quad_aspect(quad) < LOW_ANGLE_HW:
            warns.append("very_low_angle")

        if photo_fn is None and occ_est is not None:
            # No piece-type calibration active but auto-calibration is enabled:
            # the zero-setup path handles this photo entirely.
            frame = Frame(
                board_id=board_id,
                seq=next(snapshot_seq),
                capture_ts=time.time(),
                image=image,
            )
            corners_q = quad if quad is not None else full_frame_corners(image)
            return await _uncalibrated_estimate(frame, corners_q, debug_on, warns, sidecar)

        # What the debug boxes/mosaic should be drawn from — defaults to "whole
        # posted image is the board", overridden below by whichever branch the
        # estimate itself actually took (kept in lockstep with that branch so the
        # visualisation always matches what recognition saw).
        debug_image, debug_corners = image, full_frame_corners(image)
        debug_headroom, debug_side_pad = 0.0, 0.0

        if quad is not None and photo_fn is None:
            # Rectified-mode estimator handed an original photo: warp server-side
            # so the classifier still sees what it was calibrated on (exact for
            # rectified-mode centroids; the M0 stub ignores pixels anyway).
            image = rectify_photo(image, quad)
            quad = None
            debug_image, debug_corners = image, full_frame_corners(image)
        frame = Frame(
            board_id=board_id,
            seq=next(snapshot_seq),
            capture_ts=time.time(),
            image=image,
        )
        autocal = None
        if photo_fn is not None:
            # Photo-mode centroids: use full-piece boxes. Without corners the body
            # is a rectified crop — treat the whole frame as the board (approximate
            # but keeps the old rectified-crop contract working).
            debug_corners = quad if quad is not None else full_frame_corners(image)
            debug_headroom, debug_side_pad = est.headroom, est.side_pad
            event = photo_fn(frame, debug_corners)
            autocal = _maybe_bootstrap_bank(frame, debug_corners, event)
            if autocal is not None:
                # Answer through the just-upgraded ensemble, not the bare model.
                event = _photo_estimator()(frame, debug_corners)
                refresh = None
            else:
                refresh = _maybe_refresh_bank(frame, debug_corners, event)
        else:
            event = est.estimate(frame)
            refresh = None
        rotated_deg = None
        if photo_fn is not None and _parse_debug(autorotate):
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
        if autocal:
            result["autocalibrated"] = autocal
        if refresh:
            result["bank_refreshed"] = refresh
        if rotated_deg:
            result["autorotated"] = rotated_deg
        _annotate_upload(sidecar, placement=event.placement, autorotated=rotated_deg)
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
            # acceptable. rotate_k keeps them in step with square_dist above when
            # autorotate fired (see _debug_payload).
            size = est.square_size or 224 if photo_fn is not None else 224
            result["debug"] = _debug_payload(
                debug_image, debug_corners, event.square_dist,
                headroom=debug_headroom, side_pad=debug_side_pad, size=size,
                rotate_k=(rotated_deg or 0) // 90,
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
        nonlocal est
        body = await request.body()
        try:
            image = decode_jpeg(body)
            quad = _parse_corners(corners)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        _save_upload("calibrate", body, corners=corners, placement=placement)

        from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
        from board_reader_inference.estimator import CentroidEstimator, EnsembleEstimator

        # Orientation guard (2026-07-18 field incident). Auto-detected corners
        # carry arbitrary orientation and `autorotate` hides mislabeling from the
        # user — so trusting `placement` in corner order let a 180°-off calibrate
        # push a color-swapped bank to disk. Resolve the label rotation against a
        # stateless probe of THIS photo (the trained member if present — it can't
        # have been poisoned) and only accept a rotation that clearly beats the
        # as-given labels.
        label_rotation = 0
        probe_est = _base_trained()
        probe_fn = (
            probe_est.estimate_photo if probe_est is not None else _photo_estimator()
        )
        if probe_fn is not None:
            from board_reader_core.protocol import (
                CLASSES,
                encode_placement,
                parse_placement,
            )

            try:
                probe = probe_fn(
                    Frame(
                        board_id="calibrate-probe",
                        seq=next(snapshot_seq),
                        capture_ts=time.time(),
                        image=image,
                    ),
                    quad,
                )
                pred = [
                    CLASSES[i]
                    for i in np.asarray(probe.square_dist).reshape(64, len(CLASSES)).argmax(-1)
                ]
                grid = np.array(parse_placement(placement), dtype="<U5")
                pred_arr = np.array(pred)
                agrees = [
                    int((np.rot90(grid, k).reshape(-1) == pred_arr).sum())
                    for k in range(4)
                ]
                best = int(np.argmax(agrees))
                claimed = np.rot90(grid, best).reshape(-1)
                pieces = claimed != "empty"
                piece_agree = int((claimed[pieces] == pred_arr[pieces]).sum())
                if piece_agree < CALIBRATE_MIN_PIECE_AGREE * int(pieces.sum()):
                    # The photo is not showing the claimed position (mid-game
                    # board sent as "start", corners off the surface, unusably
                    # flat camera): calibrating would poison the bank.
                    return JSONResponse(
                        {
                            "error": (
                                "calibration rejected: the photo does not match the "
                                f"given position (only {piece_agree}/{int(pieces.sum())} "
                                "claimed piece squares agree, even at the best rotation). "
                                "Check that the board shows the position being calibrated, "
                                "the corners sit on the playing surface, and the camera "
                                "is not too flat."
                            )
                        },
                        status_code=422,
                    )
                if best != 0 and agrees[best] >= agrees[0] + 8:
                    label_rotation = best
                    placement = encode_placement(np.rot90(grid, best).tolist())
            except ValueError:
                pass  # unparseable placement is rejected below as before

        # Reuse the running estimator's bank member's embedder + box geometry
        # when it has one (dims and crops must match); otherwise start a fresh
        # DINOv2 setup with the shipped photo-box defaults.
        cent = _centroid_part()
        if cent is not None:
            embedder, backbone = cent.embedder, cent.classifier.backbone
            headroom, side_pad = cent.headroom, cent.side_pad
        elif isinstance(occ_est, CentroidEstimator):
            # Uncalibrated but autocalib-enabled: share the occupancy bank's
            # embedder/geometry (one ONNX session; crops cut identically).
            embedder, backbone = occ_est.embedder, occ_est.classifier.backbone
            headroom, side_pad = occ_est.headroom, occ_est.side_pad
        else:
            from board_reader_inference.embed import make_embedder

            backbone = "dinov2"
            embedder = make_embedder(backbone)
            headroom, side_pad = 0.3, 0.15
        try:
            crops = crop_squares_photo(
                image, quad, headroom=headroom, side_pad=side_pad
            )
            new = build_exemplars(
                embedder,
                crops,
                placement,
                backbone=backbone,
                box_headroom=headroom,
                box_side_pad=side_pad,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if (
            cent is not None
            and isinstance(cent.classifier, ExemplarClassifier)
            and cent.classifier.backbone == backbone
        ):
            classifier = cent.classifier.extended(new.exemplars, new.labels)
        else:
            classifier = new
        bank_est = CentroidEstimator(
            embedder, classifier, headroom=headroom, side_pad=side_pad
        )
        # Calibration BOOSTS a zero-shot trained model rather than replacing it
        # (the blend beats either member alone — see EnsembleEstimator).
        base = _base_trained()
        est = (
            EnsembleEstimator(
                base, bank_est, weight=_booster_weight(classifier.exemplars.shape[0])
            )
            if base is not None
            else bank_est
        )
        app.state.estimator = est
        classes_calibrated = len(set(new.labels.tolist()))
        path = _save_session(classifier)
        return JSONResponse(
            {
                "classes_calibrated": classes_calibrated,
                "backbone": backbone,
                "saved": path,
                "exemplars": int(classifier.exemplars.shape[0]),
                "label_rotation": label_rotation,
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
        """Edge -> inference frame ingress. Header (text) then JPEG (binary) per frame."""
        await ws.accept()
        try:
            while True:
                header = json.loads(await ws.receive_text())
                jpeg = await ws.receive_bytes()
                frame = Frame.from_header(header, decode_jpeg(jpeg))
                event = est.estimate(frame)
                await hub_.publish(event)
                await ws.send_json({"ack": frame.seq, "board_id": frame.board_id})
        except WebSocketDisconnect:
            return

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
