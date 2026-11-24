"""ChessQ Lite recognizer (``BR_CHESSQ``): preprocessing, output mapping, serving.

Everything except the last test drives the recognizer with a fake logits callable,
so the suite needs no weights. The last one runs the real release on one held-out
still and is ``gpu``-marked (a real ONNX session), like the other real-model tests.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    START_PLACEMENT,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference import chessq as chessq_mod
from board_reader_inference import replay
from board_reader_inference.chessq import (
    CHESSQ_CLASSES,
    INPUT_PX,
    ChessQRecognizer,
    preprocess_frame,
    square_dist_from_logits,
)
from board_reader_inference.hub import PositionHub
from board_reader_inference.recognizer import (
    RecognizerEstimator,
    StubRecognizer,
    build_from_env,
)
from board_reader_inference.routing import PhotoRead, route_for
from board_reader_inference.server import create_app
from board_reader_inference.session import CalibrationRefused, EstimatorSession
from board_reader_inference.squares import full_frame_corners

# Not left-right or top-bottom symmetric, so a transposed or flipped square order
# cannot read back as itself.
RUY_LOPEZ = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R"


def _logits_for(placement: str, peak: float = 8.0) -> np.ndarray:
    """What the model would emit reading ``placement``: (1, 64, 13), FEN order,
    ChessQ's own class order."""
    logits = np.zeros((1, 64, 13), dtype=np.float32)
    for i, name in enumerate(np.array(parse_placement(placement)).reshape(-1)):
        logits[0, i, CHESSQ_CLASSES.index("." if name == "empty" else name)] = peak
    return logits


def test_class_order_is_the_published_contract() -> None:
    """Pinned to the release descriptor's ``output.class_symbols`` (V6 r1), written
    out here rather than derived, so a wrong ``CHESSQ_CLASSES`` cannot test itself."""
    assert "".join([".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]) == CHESSQ_CLASSES


def _served(model) -> RecognizerEstimator:
    """ChessQ over ``model`` as the gateway serves it."""
    return RecognizerEstimator(ChessQRecognizer(model))


def _fake_model(placement: str):
    seen: list[np.ndarray] = []

    def run(batch: np.ndarray) -> np.ndarray:
        seen.append(batch)
        return _logits_for(placement)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _frame(placement: str = START_PLACEMENT, *, corners: np.ndarray | None = None) -> Frame:
    return Frame(
        board_id="A1", seq=3, capture_ts=12.5, image=render_board(placement), corners=corners
    )


# --------------------------------------------------------------------------- #
# Unit: the model contract
# --------------------------------------------------------------------------- #


def test_preprocess_squashes_whole_frame_to_rgb_imagenet_nchw():
    # A non-square BGR frame of one colour: B=0, G=128, R=255.
    image = np.empty((300, 500, 3), dtype=np.uint8)
    image[...] = (0, 128, 255)
    x = preprocess_frame(image)
    assert x.shape == (1, 3, INPUT_PX, INPUT_PX)
    assert x.dtype == np.float32
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    for channel, value in enumerate((255, 128, 0)):  # RGB, not BGR
        expected = (value / 255.0 - mean[channel]) / std[channel]
        np.testing.assert_allclose(x[0, channel], expected, atol=1e-5)


def test_logits_map_onto_protocol_squares_and_classes():
    dist = square_dist_from_logits(_logits_for(RUY_LOPEZ))
    assert dist.shape == (8, 8, len(CLASSES))
    np.testing.assert_allclose(dist.sum(-1), 1.0, atol=1e-5)
    grid = np.array(parse_placement(RUY_LOPEZ))
    np.testing.assert_array_equal(np.array(CLASSES)[dist.argmax(-1)], grid)


def test_logits_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="ChessQ returned"):
        square_dist_from_logits(np.zeros((64, 13), dtype=np.float32))


def test_recognizer_reads_in_the_chess_frame():
    recognizer = ChessQRecognizer(_fake_model(RUY_LOPEZ))
    assert (recognizer.name, recognizer.frame, recognizer.model_path) == ("chessq", "chess", None)
    dist = recognizer.read(render_board(RUY_LOPEZ))
    np.testing.assert_array_equal(
        np.array(CLASSES)[dist.argmax(-1)], np.array(parse_placement(RUY_LOPEZ))
    )


def test_module_is_self_contained():
    """The core recognizer borrows nothing from the crop-classifier stack, so
    Phase B can delete that stack without touching it."""
    tree = ast.parse(Path(chessq_mod.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    ours = {m for m in imported if m.startswith("board_reader_inference")}
    assert ours == {"board_reader_inference.onnxrt"}


def test_estimate_reads_the_whole_frame_and_ignores_corners():
    model = _fake_model(RUY_LOPEZ)
    est = _served(model)
    frame = _frame(RUY_LOPEZ)
    event = est.estimate(frame)
    assert event.placement == RUY_LOPEZ
    assert (event.board_id, event.seq, event.status) == ("A1", 3, "ok")
    # A quad that would crop most of the board away changes nothing: the model
    # was handed the same whole-frame tensor.
    corners = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32)
    assert est.estimate_photo(frame, corners).placement == RUY_LOPEZ
    np.testing.assert_array_equal(model.seen[0], model.seen[1])
    np.testing.assert_array_equal(model.seen[0], preprocess_frame(frame.image))


def test_black_bishop_demotion_is_not_applied():
    # A narrow b-over-p win (margin 0.1 < ATTRACTOR_MARGIN) — read_event would
    # swap it to p; ChessQ's read must keep the model's own argmax.
    logits = _logits_for(START_PLACEMENT)
    c8 = 2  # FEN index of c8, a black bishop in the start position
    logits[0, c8] = -10.0
    logits[0, c8, CHESSQ_CLASSES.index("b")] = np.log(0.55)
    logits[0, c8, CHESSQ_CLASSES.index("p")] = np.log(0.45)
    event = _served(lambda batch: logits).estimate(_frame())
    assert event.placement == START_PLACEMENT
    assert event.square_dist[0, 2].argmax() == CLASS_INDEX["b"]


def test_routes_through_photo_path_with_original_pixels():
    # With corners on the frame, a non-photo estimator would be handed a server-side
    # warp (RectifiedRead). ChessQ must get the frame as shot.
    est = _served(_fake_model(START_PLACEMENT))
    frame = _frame(corners=full_frame_corners(render_board()) * 0.5)
    route = route_for(est, frame)
    assert isinstance(route, PhotoRead)
    assert route.frame.image is frame.image


# --------------------------------------------------------------------------- #
# Integration: selection, calibration, serving
# --------------------------------------------------------------------------- #


@pytest.fixture
def chessq_env(tmp_path, monkeypatch):
    """``BR_CHESSQ`` naming a (fake) model file, beside the whole default stack."""
    model = tmp_path / "cq.onnx"
    model.write_bytes(b"not loaded: _onnx_logits_fn is patched")
    monkeypatch.setattr(chessq_mod, "_onnx_logits_fn", lambda path: _fake_model(START_PLACEMENT))
    monkeypatch.setenv("BR_CHESSQ", str(model))
    monkeypatch.setenv("BR_MODEL", str(tmp_path / "run_e.onnx"))
    monkeypatch.setenv("BR_CENTROIDS", str(tmp_path / "centroids.npz"))
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "bank.npz"))
    monkeypatch.chdir(tmp_path)  # nothing a calibration might write escapes
    return model


def test_br_chessq_outranks_every_other_estimator_setting(chessq_env):
    est = build_from_env(None)
    assert isinstance(est, RecognizerEstimator)
    assert isinstance(est.recognizer, ChessQRecognizer)
    assert est.recognizer.model_path == str(chessq_env)


def test_missing_weights_degrade_to_the_stub_loudly(tmp_path, monkeypatch, capsys):
    """A fresh clone has no weights (plan decision D2): the node still starts, on
    the stub, says so at startup, and ``/healthz`` carries the reason."""
    monkeypatch.setenv("BR_CHESSQ", str(tmp_path / "absent.onnx"))
    est = build_from_env(None)
    assert isinstance(est, RecognizerEstimator)
    assert isinstance(est.recognizer, StubRecognizer)
    assert "make models" in (est.fallback or "")
    assert "STUB recognizer" in capsys.readouterr().err
    health = TestClient(create_app(est, PositionHub())).get("/healthz").json()
    assert health["recognizer"]["name"] == "stub"
    assert "absent.onnx" in health["recognizer"]["fallback"]


def test_calibrate_is_refused_and_keeps_chessq(chessq_env):
    client = TestClient(create_app(hub=PositionHub()))
    photo = render_board(START_PLACEMENT)
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo))
    r = client.post(f"/calibrate?corners={corners}", content=encode_jpeg(photo))
    assert r.status_code == 409
    assert "BR_CHESSQ" in r.json()["error"]
    assert isinstance(client.app.state.estimator, RecognizerEstimator)
    assert not Path("bank.npz").exists()
    assert client.get("/healthz").json()["recognizer"] == {
        "name": "chessq",
        "model": str(chessq_env),
    }


def test_start_position_reads_do_not_bootstrap_a_bank(chessq_env):
    # The fake model reads the start position — exactly what makes the trained
    # stack build its first bank and swap to an ensemble on POST /estimate.
    client = TestClient(create_app(hub=PositionHub()))
    photo = render_board(START_PLACEMENT)
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo))
    body = client.post(f"/estimate?corners={corners}", content=encode_jpeg(photo)).json()
    assert body["placement"] == START_PLACEMENT
    assert "autocalibrated" not in body and "bank_refreshed" not in body
    assert isinstance(client.app.state.estimator, RecognizerEstimator)


def test_session_never_swaps_chessq_out():
    session = EstimatorSession(_served(_fake_model(START_PLACEMENT)))
    photo = render_board(START_PLACEMENT)
    outcome = session.calibrate_from(photo, full_frame_corners(photo), START_PLACEMENT)
    assert outcome == CalibrationRefused(recognizer="chessq")
    with pytest.raises(RuntimeError, match="never swapped"):
        session._swap(_served(_fake_model(START_PLACEMENT)))


def test_ingest_publishes_chessq_read_with_prior(gateway, drain_ingest):
    client = gateway(create_app(_served(_fake_model(RUY_LOPEZ)), PositionHub()))
    frame = _frame(RUY_LOPEZ, corners=full_frame_corners(render_board()))
    header, jpeg = frame.encode()
    with client.websocket_connect("/ingest") as ws:
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()
    drain_ingest(client)
    snap = client.get("/boards/A1").json()
    assert snap["placement"] == RUY_LOPEZ
    assert snap["prior"]["placement"] == RUY_LOPEZ


def test_replay_config_digest_distinguishes_br_chessq(chessq_env):
    env = replay.config_env()
    assert env["BR_CHESSQ"]["value"] == str(chessq_env)


# --------------------------------------------------------------------------- #
# The real release on one held-out still (`make test-gpu`; skips without files)
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).resolve().parents[3]
_V6 = _ROOT / "models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx"
_STILL = _ROOT / "data/eval/heldout-20260831/img_0763.label.json"


@pytest.mark.gpu
@pytest.mark.skipif(
    not (_V6.exists() and _STILL.exists()),
    reason=f"needs {_V6.relative_to(_ROOT)} and {_STILL.relative_to(_ROOT)} (gitignored)",
)
def test_real_v6_reads_a_held_out_midgame_still():
    import cv2

    label = json.loads(_STILL.read_text())
    image = cv2.imread(str(_STILL.parent / label["photo"]))
    event = _served(_V6).estimate(Frame(board_id="still", seq=0, capture_ts=0.0, image=image))
    assert event.placement == label["placement"]
