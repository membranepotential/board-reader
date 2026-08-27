"""Tests for the calibration session (:mod:`board_reader_inference.session`).

These are the rules for *when the active estimator may be replaced* — the ones
that used to be reachable only by driving HTTP, and that decide whether a bank
grows, is rebuilt, or is refused. Here they are exercised directly: no FastAPI, no
TestClient, one session object per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASSES,
    NUM_CLASSES,
    START_PLACEMENT,
    encode_placement,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference import autocalib
from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.estimator import (
    CentroidEstimator,
    EnsembleEstimator,
    StubEstimator,
)
from board_reader_inference.session import (
    Calibration,
    CalibrationRejected,
    EstimatorSession,
    booster_weight,
)
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    full_frame_corners,
    rot90_points,
)
from board_reader_inference.trained import TrainedEstimator

MIDGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"

# Flat renders + full-frame corners: photo boxes with zero headroom are then exact
# squares, which the Lite embedder can classify (it cannot read tall perspective
# boxes — real-photo quality is the DINOv2 eval's job).
FLAT = BboxGeometry(0.0, 0.0)
SQUARE_PX = 64


def _photo(placement: str = START_PLACEMENT) -> tuple[np.ndarray, np.ndarray]:
    image = render_board(placement, square_px=24)
    return image, full_frame_corners(image)


def _frame(image: np.ndarray, corners: np.ndarray | None = None) -> Frame:
    return Frame(
        board_id="T1", seq=0, capture_ts=0.0, image=image, corners=corners
    )


def _lite_bank(placement: str = START_PLACEMENT) -> CentroidEstimator:
    """A Lite-embedder exemplar-bank estimator calibrated on one flat render."""
    image, quad = _photo(placement)
    emb = LiteEmbedder()
    bank = build_exemplars(
        emb,
        crop_squares_photo(image, quad, geometry=FLAT, size=SQUARE_PX),
        placement,
        backbone="lite",
        box_headroom=0.0,
        box_side_pad=0.0,
    )
    return CentroidEstimator(emb, bank, square_size=SQUARE_PX, geometry=FLAT)


def _cycling_oracle(placement: str):
    """Logits fn recognizing crops by position in the 64-crop sequence.

    It cycles, so any number of reads (and ensemble members re-cropping) keep
    getting consistent answers.
    """
    labels = [c for row in parse_placement(placement) for c in row]
    state = {"i": 0}

    def fn(batch: np.ndarray) -> np.ndarray:
        n = batch.shape[0]
        out = np.full((n, NUM_CLASSES), -5.0, dtype=np.float32)
        for j in range(n):
            out[j, CLASSES.index(labels[(state["i"] + j) % 64])] = 5.0
        state["i"] = (state["i"] + n) % 64
        return out

    return fn


def _trained(placement: str) -> TrainedEstimator:
    return TrainedEstimator(
        _cycling_oracle(placement), geometry=FLAT, square_size=SQUARE_PX
    )


def test_booster_weight_by_bank_maturity():
    """A mature bank earns the heavier blend share; a thin bootstrap doesn't."""
    assert booster_weight(256) == 0.5
    assert booster_weight(4096) == 0.3


def test_bootstrap_upgrades_a_trained_model_that_reads_a_start_position():
    """§4.6 zero-tap: the model's own start-position reading becomes the first bank,
    and the answer is re-read through the upgraded ensemble, not the bare model."""
    image, quad = _photo()
    trained = _trained(START_PLACEMENT)
    swaps: list[object] = []
    session = EstimatorSession(
        trained, occupancy=_lite_bank(), on_swap=swaps.append
    )
    frame = _frame(image, quad)

    boot = session.bootstrap_bank(frame, quad, trained.estimate_photo(frame, quad))
    assert boot is not None
    assert boot.matches == 64 and boot.rotation == 0
    assert boot.exemplars == 64 * 4  # photometric variants included
    assert isinstance(session.estimator, EnsembleEstimator)
    assert session.estimator.primary is trained
    assert swaps == [session.estimator]  # the swap announced itself, exactly once
    assert boot.reread(frame, quad).placement == START_PLACEMENT

    # A second sighting finds the booster already there and does nothing.
    again = session.bootstrap_bank(frame, quad, trained.estimate_photo(frame, quad))
    assert again is None


def test_bootstrap_declines_when_the_read_is_not_the_start_position():
    """The pollution guard: a mid-game board must never seed a start-labeled bank."""
    image, quad = _photo(MIDGAME)
    trained = _trained(MIDGAME)
    session = EstimatorSession(trained, occupancy=_lite_bank())
    frame = _frame(image, quad)

    assert session.bootstrap_bank(frame, quad, trained.estimate_photo(frame, quad)) is None
    assert session.estimator is trained


def test_refresh_appends_only_under_the_confidence_agreement_and_cap_gates(monkeypatch):
    """A calibrated bank re-learns a drifted start position — but only when its own
    confidence says it has drifted, its reading agrees with the start placement,
    and the bank is still below the cap."""
    bank = _lite_bank()
    before = bank.classifier.exemplars.shape[0]
    session = EstimatorSession(bank)
    image, quad = _photo()
    frame = _frame(image, quad)
    event = bank.estimate_photo(frame, quad)

    # Confidence gate: with the threshold at 0 every read counts as "confident
    # enough already". (The lite stand-in's absolute confidence is not meaningful,
    # so the gate is exercised via explicit thresholds.)
    monkeypatch.setattr(autocalib, "APPEND_MAX_CONF", 0.0)
    assert session.refresh_bank(frame, quad, event) is None
    assert session.estimator is bank

    # Agreement gate: this photo is a mid-game board, not the start position.
    monkeypatch.setattr(autocalib, "APPEND_MAX_CONF", 2.0)
    mid_image, mid_quad = _photo(MIDGAME)
    mid_frame = _frame(mid_image, mid_quad)
    mid_event = bank.estimate_photo(mid_frame, mid_quad)
    assert session.refresh_bank(mid_frame, mid_quad, mid_event) is None
    assert session.estimator is bank

    # Drifted lighting on a start position: the bank grows.
    refresh = session.refresh_bank(frame, quad, event)
    assert refresh is not None
    assert refresh.agree >= autocalib.APPEND_MIN_AGREE
    assert refresh.total == before + refresh.added
    grown = session.estimator
    assert isinstance(grown, CentroidEstimator)
    assert grown.classifier.exemplars.shape[0] == refresh.total

    # The cap stops unbounded growth.
    monkeypatch.setattr(autocalib, "APPEND_BANK_CAP", refresh.total)
    assert session.refresh_bank(frame, quad, grown.estimate_photo(frame, quad)) is None
    assert session.estimator is grown


def test_calibrate_extends_a_same_backbone_bank_rather_than_replacing_it():
    """Every calibration photo only grows the bank (DESIGN §4.6 E1/E2)."""
    bank = _lite_bank()
    before = bank.classifier.exemplars.shape[0]
    session = EstimatorSession(bank)
    image, quad = _photo(MIDGAME)

    outcome = session.calibrate_from(image, quad, MIDGAME)
    assert isinstance(outcome, Calibration)
    assert outcome.backbone == "lite"
    assert 0 < outcome.classes_calibrated <= 13
    assert outcome.exemplars > before  # extended, not rebuilt
    grown = session.estimator
    assert isinstance(grown, CentroidEstimator)
    assert isinstance(grown.classifier, ExemplarClassifier)
    assert grown.classifier.exemplars.shape[0] == outcome.exemplars
    assert grown.estimate_photo(_frame(image), quad).placement == MIDGAME


def test_calibrate_boosts_a_trained_model_instead_of_replacing_it():
    trained = _trained(START_PLACEMENT)
    session = EstimatorSession(trained, occupancy=_lite_bank())
    image, quad = _photo()

    assert isinstance(session.calibrate_from(image, quad, START_PLACEMENT), Calibration)
    assert isinstance(session.estimator, EnsembleEstimator)
    assert session.estimator.primary is trained


def test_calibrate_rotates_labels_that_disagree_with_the_photo_by_a_turn():
    """The 2026-07-18 field incident: a 180°-off calibrate pushed a color-swapped
    bank to disk. The labels are resolved against a probe of THIS photo."""
    trained = _trained(START_PLACEMENT)
    session = EstimatorSession(trained, occupancy=_lite_bank())
    image, quad = _photo()
    flipped = encode_placement(
        np.rot90(np.array(parse_placement(START_PLACEMENT), dtype="<U5"), 2).tolist()
    )

    outcome = session.calibrate_from(image, quad, flipped)
    assert isinstance(outcome, Calibration)
    assert outcome.label_rotation == 2


def test_calibrate_is_refused_when_the_photo_does_not_show_the_claimed_position():
    """Field incident #4: a mid-game photo calibrated as "start" would poison the
    bank, and no rotation of the claimed labels rescues it."""
    # Far enough from the start position that no rotation of the claimed labels
    # clears the piece-agreement floor — an opening position still shares most of
    # its pieces with the start and is deliberately NOT refused.
    late = "r2qr2k/ppp2Qpp/8/2b1N3/3npn2/2N5/PPP2PPP/R3R1K1"
    trained = _trained(late)
    session = EstimatorSession(trained, occupancy=_lite_bank())
    image, quad = _photo(late)

    outcome = session.calibrate_from(image, quad, START_PLACEMENT)
    assert isinstance(outcome, CalibrationRejected)
    assert outcome.piece_agree < outcome.piece_squares
    assert outcome.rotated_fit is None  # a true mismatch, not a turned board
    assert session.estimator is trained  # nothing swapped


def test_calibrate_names_the_rotation_when_the_board_is_turned_in_frame():
    """The 2026-08-27 field incident: a portrait-held phone failed every calibrate
    all afternoon with the generic mismatch message, because a board rotated 90°
    in the frame garbles every piece crop while occupancy holds. The rejection
    must name that cause — it is the one the operator can fix at the tripod."""
    session = EstimatorSession(_lite_bank(), occupancy=_lite_bank())
    image, quad = _photo()
    turned = np.rot90(image).copy()  # the board a quarter-turn over in the frame
    turned_quad = rot90_points(quad, 1, image.shape)

    outcome = session.calibrate_from(turned, turned_quad, START_PLACEMENT)
    assert isinstance(outcome, CalibrationRejected)
    assert outcome.rotated_fit is not None
    assert outcome.rotated_fit.quarter_turns == 3  # three more turns undo the one
    assert outcome.rotated_fit.piece_agree == 32  # turned back, the photo is exact


def test_adopt_start_calibrates_piece_types_off_the_occupancy_bank():
    """The zero-setup path: an uncalibrated server that sees a start position
    calibrates from that very photo and starts answering with full positions."""
    occ = _lite_bank()
    session = EstimatorSession(StubEstimator(START_PLACEMENT), occupancy=occ)
    image, quad = _photo()
    route = session.route(_frame(image, quad))

    from board_reader_inference.routing import PhotoRead, Uncalibrated

    assert isinstance(route, Uncalibrated)
    auto = session.adopt_start(route, route.read())
    assert auto is not None
    assert auto.matches == 64 and auto.rotation == 0
    assert auto.reread(route.frame, route.corners).placement == START_PLACEMENT
    # ...and the session now routes through the calibrated bank.
    assert isinstance(session.route(_frame(image, quad)), PhotoRead)


def test_adopt_start_declines_a_position_that_is_not_the_start():
    occ = _lite_bank()
    stub = StubEstimator(START_PLACEMENT)
    session = EstimatorSession(stub, occupancy=occ)
    image, quad = _photo(MIDGAME)
    route = session.route(_frame(image, quad))

    assert session.adopt_start(route, route.read()) is None
    assert session.estimator is stub


@pytest.mark.parametrize("opt_in", [True, False])
def test_session_bank_is_persisted_only_when_the_env_opts_in(
    opt_in, tmp_path, monkeypatch
):
    """A cwd-relative default silently clobbered the real bank from test runs —
    twice. Persistence is opt-in via BR_SESSION_BANK, and off means nothing is
    ever written."""
    monkeypatch.chdir(tmp_path)  # any accidental relative write would land here
    path = tmp_path / "banks" / "session.npz"
    if opt_in:
        monkeypatch.setenv("BR_SESSION_BANK", str(path))
    else:
        monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    session = EstimatorSession(_lite_bank())
    image, quad = _photo(MIDGAME)

    outcome = session.calibrate_from(image, quad, MIDGAME)
    assert isinstance(outcome, Calibration)
    if opt_in:
        assert outcome.saved == str(path) and path.exists()
    else:
        assert outcome.saved is None
        assert list(tmp_path.rglob("*.npz")) == []
