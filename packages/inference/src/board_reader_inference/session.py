"""The calibration session (DESIGN §4.6): the active estimator and its swap rules.

One board-reading session has exactly one *active estimator*, and four things may
replace it while the server runs:

* :meth:`EstimatorSession.calibrate_from` — the operator's labeled photo
  (``POST /calibrate``): builds a kNN exemplar bank and **boosts** the zero-shot
  trained model with it rather than replacing it;
* :meth:`EstimatorSession.bootstrap_bank` — the trained model reads a start
  position, so the first bank is built from that very photo, zero taps;
* :meth:`EstimatorSession.refresh_bank` — a calibrated bank reads the start
  position at drifted (low) confidence, so label-confirmed exemplars are appended;
* :meth:`EstimatorSession.adopt_start` — the §4.6 zero-setup path: an uncalibrated
  server sees a start position through its occupancy bank and calibrates piece
  types from it.

They all end in the same place (:meth:`_swap`), which is the point of this class:
the estimator used to be a ``nonlocal`` in the gateway's request closure, mutated
from four handlers and read from six, so every rule about *when* an estimator may
be replaced could only be exercised by driving HTTP. Here they are ordinary
methods over ordinary state, and the gateway is left with routing and JSON.

Session-bank persistence is OPT-IN via ``BR_SESSION_BANK=<path>``: every swap above
persists there, and an autocalib-enabled server restores it on startup. Unset ->
calibration lives in memory only and NOTHING is ever written. Opt-in is deliberate:
a cwd-relative default made ordinary test runs silently overwrite the real bank —
twice.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    PositionEvent,
    encode_placement,
    parse_placement,
    position_confidence,
)
from board_reader_core.wire import Frame

from board_reader_inference import autocalib
from board_reader_inference.centroid import (
    CentroidClassifier,
    ExemplarClassifier,
    build_exemplars,
)
from board_reader_inference.estimator import (
    CentroidEstimator,
    EnsembleEstimator,
    Estimator,
)
from board_reader_inference.routing import (
    OccupancyRead,
    PhotoEstimate,
    PhotoReader,
    Route,
    Uncalibrated,
    board_corners,
    photo_estimator,
    route_for,
)
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    rot90_points,
)
from board_reader_inference.trained import TrainedReader

# /calibrate sanity floor: of the squares the claimed placement says hold a
# PIECE, at least this fraction must agree with a stateless probe of the photo
# (best rotation). Empty squares are excluded — a mid-game board and the start
# position share a mostly-empty middle, which inflates whole-board agreement
# (~40/64) and hides the mismatch. Below the floor the photo is not showing the
# claimed position (mid-game sent as "start", corners off the surface, unusably
# flat angle) and calibrating would poison the bank — reject (incidents #3, #4).
CALIBRATE_MIN_PIECE_AGREE = 0.6

# Box geometry for a bank started from scratch (no bank, no occupancy bank to
# borrow from): the shipped photo-box defaults.
FRESH_BANK_GEOMETRY = BboxGeometry(headroom=0.3, side_pad=0.15)


def session_bank_path() -> str | None:
    """Where the session bank persists, or ``None`` when persistence is off."""
    return os.environ.get("BR_SESSION_BANK") or None


def booster_weight(bank_exemplars: int) -> float:
    """Trained-model share of the ensemble blend, by bank maturity.

    A mature bank (many calibrates/appends) is the stronger member under its
    own conditions — lean on it (measured 2026-07-18 on the 10-board eval:
    30/70 = 633/640 vs 50/50 = 625). A thin bootstrap bank (one photo) hasn't
    earned that trust — blend evenly.
    """
    return 0.3 if bank_exemplars >= 1024 else 0.5


@dataclass(frozen=True, slots=True)
class BankBootstrap:
    """The first exemplar bank, built from the trained model's own start-position read."""

    matches: int
    rotation: int
    exemplars: int
    saved: str | None
    #: The upgraded ensemble's photo path. The answer must be re-read through it —
    #: the caller asked the bare model, and the session is no longer that model.
    reread: PhotoEstimate


@dataclass(frozen=True, slots=True)
class BankRefresh:
    """Exemplars appended to a calibrated bank that had drifted out of its lighting."""

    added: int
    total: int
    agree: int
    rotation: int
    saved: str | None


@dataclass(frozen=True, slots=True)
class Autocalibration:
    """Piece types calibrated from an occupancy-detected start position (§4.6)."""

    matches: int
    rotation: int
    exemplars: int
    saved: str | None
    #: The freshly calibrated estimator's photo path — see :attr:`BankBootstrap.reread`.
    reread: PhotoEstimate


@dataclass(frozen=True, slots=True)
class Calibration:
    """A completed ``/calibrate``: the bank the session now reads through."""

    classes_calibrated: int
    backbone: str
    exemplars: int
    saved: str | None
    label_rotation: int


@dataclass(frozen=True, slots=True)
class RotatedFit:
    """The same photo clears the calibrate gate once turned ``quarter_turns``×90°.

    Evidence that the *board is rotated in the frame* (a portrait-held phone, a
    sideways mount): per-square crops extend toward the top of the image
    (``squares.photo_square_boxes``), so a sideways board puts every piece body in
    a neighbouring square's crop — piece types collapse while occupancy holds.
    No label rotation or corner nudge can rescue that; re-aiming the camera can.
    """

    quarter_turns: int  # np.rot90 turns that made the photo pass
    piece_agree: int


@dataclass(frozen=True, slots=True)
class CalibrationRejected:
    """The photo does not show the claimed position — calibrating would poison the bank."""

    piece_agree: int
    piece_squares: int
    rotated_fit: RotatedFit | None = None


CalibrateOutcome = Calibration | CalibrationRejected


class EstimatorSession:
    """The active estimator, its members, and the rules for swapping it.

    ``occupancy`` (the style-robust occupancy bank, e.g. ChessReD) enables the §4.6
    zero-setup path: while no piece-type calibration is active, reads route to
    :class:`~board_reader_inference.routing.Uncalibrated`. ``on_swap`` is called
    with every new active estimator, which is how the gateway keeps
    ``app.state.estimator`` in step without any handler remembering to.
    """

    def __init__(
        self,
        estimator: Estimator,
        *,
        occupancy: CentroidEstimator | None = None,
        on_swap: Callable[[Estimator], None] | None = None,
    ) -> None:
        self._estimator = estimator
        self._occupancy = occupancy
        self._on_swap = on_swap

    @property
    def estimator(self) -> Estimator:
        """The estimator serving reads right now (swapped by calibration)."""
        return self._estimator

    # --- routing ---------------------------------------------------------------

    def route(self, frame: Frame) -> Route:
        """How this frame should be read — the one decision both ingresses take.

        The §4.6 uncalibrated state (an occupancy bank loaded, no piece-type
        estimator) outranks the rest: piece type does not transfer across piece
        styles on frozen features, so occupancy is all that can honestly be read
        until something calibrates.
        """
        occ = self._occupancy
        if occ is not None and self.photo_estimator() is None:
            return Uncalibrated(occ, frame, board_corners(frame))
        return route_for(self._estimator, frame)

    # --- members of the active estimator ---------------------------------------

    def photo_estimator(self) -> PhotoReader | None:
        """The active estimator's photo path, if it has one."""
        return photo_estimator(self._estimator)

    def base_trained(self) -> TrainedReader | None:
        """The zero-shot trained member of the active estimator, if any.

        One model, the two-model cascade, or a fine-tune averaged with its base —
        the bank arithmetic on top does not care which, and neither does
        auto-calibration.
        """
        est = self._estimator
        if isinstance(est, EnsembleEstimator):
            est = est.primary
        return est if isinstance(est, TrainedReader) else None

    def centroid_part(self) -> CentroidEstimator | None:
        """The per-set exemplar/centroid member of the active estimator, if any."""
        if isinstance(self._estimator, EnsembleEstimator):
            booster = self._estimator.booster
            return booster if isinstance(booster, CentroidEstimator) else None
        return self._estimator if isinstance(self._estimator, CentroidEstimator) else None

    # --- swaps -----------------------------------------------------------------

    def bootstrap_bank(
        self, frame: Frame, corners: np.ndarray, event: PositionEvent
    ) -> BankBootstrap | None:
        """Zero-tap per-set calibration ON TOP of the zero-shot model (§4.6).

        When the trained model runs WITHOUT a booster bank and its own reading
        matches the start position, build the first exemplar bank from this very
        photo and upgrade to the ensemble — same trigger and pollution guard as
        auto-calibration (only label-agreeing squares are learned), but labeled
        by the trained model's full 13-class reading instead of occupancy alone.
        """
        base = self.base_trained()
        if base is None or self.centroid_part() is not None:
            return None
        rotation, agree, mask = _match_start(event)
        if agree < autocalib.APPEND_MIN_AGREE:
            return None
        occ = self._occupancy
        if occ is not None:
            embedder, backbone = occ.embedder, occ.classifier.backbone
        else:
            from board_reader_inference.embed import make_embedder

            backbone = "dinov2"
            try:
                embedder = make_embedder(backbone)
            except (OSError, FileNotFoundError):
                # No embedder available (e.g. DINOv2 weights not shipped): the
                # zero-shot answer stands on its own, just without the booster.
                return None
        # The bank is bbox-only; a pose-geometry model reports the deployed bank
        # default here rather than inventing scalars (see TrainedEstimator).
        geometry = BboxGeometry(headroom=base.headroom, side_pad=base.side_pad)
        crops = crop_squares_photo(
            frame.image, corners, geometry=geometry, size=base.square_size or 224
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
        bank = CentroidEstimator(embedder, classifier, geometry=geometry)
        upgraded = EnsembleEstimator(
            base, bank, weight=booster_weight(classifier.exemplars.shape[0])
        )
        self._swap(upgraded)
        return BankBootstrap(
            matches=agree,
            rotation=rotation,
            exemplars=int(classifier.exemplars.shape[0]),
            saved=self._save(classifier),
            reread=upgraded.estimate_photo,
        )

    def refresh_bank(
        self, frame: Frame, corners: np.ndarray, event: PositionEvent
    ) -> BankRefresh | None:
        """Calibrated-path self-refresh (DESIGN §4.6): seeing the START position
        read at drifted (low) confidence appends label-confirmed exemplars from
        this photo, so the bank tracks new lighting without the user calibrating.
        Gates: exemplar-bank estimator only; mean confidence below
        ``autocalib.APPEND_MAX_CONF``; prediction agrees with the (rotated) start
        placement on ≥ ``APPEND_MIN_AGREE`` squares (only those squares are
        learned, so mid-game photos and misreads can't pollute the bank); bank
        below ``APPEND_BANK_CAP``.
        """
        cent = self.centroid_part()
        if cent is None or not isinstance(cent.classifier, ExemplarClassifier):
            return None
        if cent.classifier.exemplars.shape[0] >= autocalib.APPEND_BANK_CAP:
            return None
        if position_confidence(event.square_dist) >= autocalib.APPEND_MAX_CONF:
            return None
        rotation, agree, mask = _match_start(event)
        if agree < autocalib.APPEND_MIN_AGREE:
            return None
        crops = crop_squares_photo(
            frame.image,
            corners,
            geometry=cent.geometry,
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
        self._swap_booster(
            CentroidEstimator(
                cent.embedder,
                classifier,
                geometry=cent.geometry,
                square_size=cent.square_size,
                orientation=cent.orientation,
            ),
            classifier,
        )
        return BankRefresh(
            added=int(new.exemplars.shape[0]),
            total=int(classifier.exemplars.shape[0]),
            agree=agree,
            rotation=rotation,
            saved=self._save(classifier),
        )

    def adopt_start(
        self, route: Uncalibrated, reading: OccupancyRead
    ) -> Autocalibration | None:
        """Calibrate piece types from an occupancy-detected start position (§4.6).

        The zero-setup path's payoff: piece type does not transfer across piece
        styles, but the *occupancy pattern* of the starting position is
        unmistakable and style-robust — so the moment an uncalibrated server sees
        one, it calibrates from that very photo and starts answering with full
        positions. ``None`` = this photo is not a start position; the caller stays
        on occupancy.
        """
        match = autocalib.detect_start(reading.dist, reading.crops)
        if match is None:
            return None
        occ = route.occupancy
        classifier = build_exemplars(
            occ.embedder,
            reading.crops,
            match.placement,
            mask=match.calib_mask,
            backbone=occ.classifier.backbone,
            box_headroom=occ.headroom,
            box_side_pad=occ.side_pad,
        )
        calibrated = CentroidEstimator(occ.embedder, classifier, geometry=occ.geometry)
        self._swap(calibrated)
        return Autocalibration(
            matches=match.matches,
            rotation=match.rotation,
            exemplars=int(classifier.exemplars.shape[0]),
            saved=self._save(classifier),
            reread=calibrated.estimate_photo,
        )

    def calibrate_from(
        self, image: np.ndarray, corners: np.ndarray, placement: str
    ) -> CalibrateOutcome:
        """Few-shot per-set calibration from one labeled photo (DESIGN §4.6, E1/E2).

        ``placement`` is the position actually on the board (the start position
        contains all 13 classes, so one photo calibrates the whole vocabulary).
        If the running estimator is already an exemplar bank on the same backbone,
        the new photo's exemplars are ADDED to it rather than replacing it — every
        calibration photo only grows the bank. Raises ``ValueError`` on an
        unusable placement/photo pair; returns :class:`CalibrationRejected` when
        the photo does not show the claimed position at all.
        """
        guard = self._resolve_label_rotation(image, corners, placement)
        if isinstance(guard, CalibrationRejected):
            return guard
        label_rotation, placement = guard

        # Reuse the running estimator's bank member's embedder + box geometry
        # when it has one (dims and crops must match); otherwise borrow the
        # occupancy bank's (one ONNX session; crops cut identically), and failing
        # that start a fresh DINOv2 setup with the shipped photo-box defaults.
        source = self.centroid_part() or self._occupancy
        if source is not None:
            embedder, backbone = source.embedder, source.classifier.backbone
            geometry = BboxGeometry(headroom=source.headroom, side_pad=source.side_pad)
        else:
            from board_reader_inference.embed import make_embedder

            backbone = "dinov2"
            embedder = make_embedder(backbone)
            geometry = FRESH_BANK_GEOMETRY
        crops = crop_squares_photo(image, corners, geometry=geometry)
        new = build_exemplars(
            embedder,
            crops,
            placement,
            backbone=backbone,
            box_headroom=geometry.headroom,
            box_side_pad=geometry.side_pad,
        )
        cent = self.centroid_part()
        if (
            cent is not None
            and isinstance(cent.classifier, ExemplarClassifier)
            and cent.classifier.backbone == backbone
        ):
            classifier = cent.classifier.extended(new.exemplars, new.labels)
        else:
            classifier = new
        self._swap_booster(
            CentroidEstimator(embedder, classifier, geometry=geometry), classifier
        )
        return Calibration(
            classes_calibrated=len(set(new.labels.tolist())),
            backbone=backbone,
            exemplars=int(classifier.exemplars.shape[0]),
            saved=self._save(classifier),
            label_rotation=label_rotation,
        )

    # --- internals -------------------------------------------------------------

    def _resolve_label_rotation(
        self, image: np.ndarray, corners: np.ndarray, placement: str
    ) -> tuple[int, str] | CalibrationRejected:
        """Check the operator's labels against a stateless probe of THIS photo.

        Orientation guard (2026-07-18 field incident). Auto-detected corners carry
        arbitrary orientation and `autorotate` hides mislabeling from the user — so
        trusting `placement` in corner order let a 180°-off calibrate push a
        color-swapped bank to disk. Resolve the label rotation against the trained
        member if present (it cannot have been poisoned) and only accept a rotation
        that clearly beats the as-given labels.

        On rejection, the four *image* rotations are probed too (2026-08-27 field
        incident: a portrait-held phone failed every calibrate all afternoon with
        the generic mismatch message). A photo that passes once turned means the
        board is rotated in the frame — a cause the operator can actually fix —
        and the rejection names it as :class:`RotatedFit`.
        """
        probe_est = self.base_trained()
        probe: PhotoReader | None = (
            probe_est if probe_est is not None else self.photo_estimator()
        )
        if probe is None:
            return 0, placement

        def read_pred(img: np.ndarray, crn: np.ndarray) -> np.ndarray:
            read = probe.estimate_photo(
                Frame(
                    board_id="calibrate-probe",
                    seq=0,
                    capture_ts=0.0,
                    image=img,
                ),
                crn,
            )
            return np.array(
                [
                    CLASSES[i]
                    for i in np.asarray(read.square_dist)
                    .reshape(64, len(CLASSES))
                    .argmax(-1)
                ]
            )

        try:
            pred = read_pred(image, corners)
            grid = np.array(parse_placement(placement), dtype="<U5")
        except ValueError:
            return 0, placement  # unparseable placement is rejected downstream as before

        def best_label_fit(pred: np.ndarray) -> tuple[int, list[int], int]:
            """(best k, all 4 totals, piece agreement at best k) against ``pred``."""
            agrees = [
                int((np.rot90(grid, k).reshape(-1) == pred).sum()) for k in range(4)
            ]
            k = int(np.argmax(agrees))
            claimed = np.rot90(grid, k).reshape(-1)
            mask = claimed != "empty"
            return k, agrees, int((claimed[mask] == pred[mask]).sum())

        n_pieces = int((grid.reshape(-1) != "empty").sum())
        best, agrees, piece_agree = best_label_fit(pred)
        if piece_agree < CALIBRATE_MIN_PIECE_AGREE * n_pieces:
            # The photo is not showing the claimed position (mid-game board sent
            # as "start", corners off the surface, unusably flat camera):
            # calibrating would poison the bank. Before refusing generically, ask
            # whether the PHOTO would read once turned — a board rotated in the
            # frame garbles every piece crop while occupancy holds, and only
            # re-aiming the camera fixes it.
            fit: RotatedFit | None = None
            for k in (1, 2, 3):
                try:
                    turned = read_pred(
                        np.rot90(image, k).copy(),
                        rot90_points(corners, k, image.shape),
                    )
                except ValueError:
                    continue
                _, _, turned_agree = best_label_fit(turned)
                if turned_agree >= CALIBRATE_MIN_PIECE_AGREE * n_pieces and (
                    fit is None or turned_agree > fit.piece_agree
                ):
                    fit = RotatedFit(quarter_turns=k, piece_agree=turned_agree)
            return CalibrationRejected(
                piece_agree=piece_agree, piece_squares=n_pieces, rotated_fit=fit
            )
        if best != 0 and agrees[best] >= agrees[0] + 8:
            return best, encode_placement(np.rot90(grid, best).tolist())
        return 0, placement

    def _swap_booster(
        self, bank: CentroidEstimator, classifier: ExemplarClassifier
    ) -> None:
        """Install ``bank`` as the per-set member, boosting the trained model if any.

        Calibration BOOSTS a zero-shot trained model rather than replacing it (the
        blend beats either member alone — see EnsembleEstimator).
        """
        base = self.base_trained()
        weight = booster_weight(classifier.exemplars.shape[0])
        self._swap(
            EnsembleEstimator(base, bank, weight=weight) if base is not None else bank
        )

    def _swap(self, estimator: Estimator) -> None:
        """Make ``estimator`` the active one, announcing it. The only mutation here."""
        self._estimator = estimator
        if self._on_swap is not None:
            self._on_swap(estimator)

    def _save(self, classifier: CentroidClassifier | ExemplarClassifier) -> str | None:
        """Persist the session bank iff ``BR_SESSION_BANK`` opts in (best-effort)."""
        path = session_bank_path()
        if path is None:
            return None
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            classifier.save(path)
        except OSError:
            return None
        return path


def _match_start(event: PositionEvent) -> tuple[int, int, np.ndarray]:
    """``(rotation, agree, mask)`` of this read against the start position."""
    dist = np.asarray(event.square_dist).reshape(-1, len(CLASSES))
    return autocalib.match_predicted_start([CLASSES[i] for i in dist.argmax(axis=1)])
