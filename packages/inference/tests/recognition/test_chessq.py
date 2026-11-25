"""ChessQ Lite recognizer (``BR_CHESSQ``): preprocessing, output mapping, serving.

Everything except the last test drives the recognizer with a fake logits callable,
so the suite needs no weights. The last one runs the real release on one held-out
still and is ``gpu``-marked (a real ONNX session), like the other real-model tests.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    START_PLACEMENT,
    PositionEvent,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.measurement import replay
from board_reader_inference.recognition import chessq as chessq_mod
from board_reader_inference.recognition.chessq import (
    CHESSQ_CLASSES,
    INPUT_PX,
    ChessQRecognizer,
    ModelOutputs,
    direction_degrees,
    frame_energy,
    preprocess_frame,
    square_dist_from_logits,
)
from board_reader_inference.recognition.recognizer import (
    FrameRead,
    StubRecognizer,
    build_from_env,
    frame_reader,
)
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

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


def _served(model) -> Callable[[Frame], PositionEvent]:
    """ChessQ over ``model`` as the gateway reads frames with it: the event."""
    read: FrameRead = frame_reader(ChessQRecognizer(model))
    return lambda frame: read(frame).event


#: The direction output of a board seen from White's left (e8 to the right),
#: what every recorded take reads: (cos, sin) = (0, 1), 90 degrees.
FROM_THE_SIDE = np.array([[0.0, 1.0]], dtype=np.float32)


def _fake_model(
    placement: str,
    direction: np.ndarray = FROM_THE_SIDE,
    attention: np.ndarray | None = None,
):
    seen: list[np.ndarray] = []

    def run(batch: np.ndarray) -> ModelOutputs:
        seen.append(batch)
        return ModelOutputs(_logits_for(placement), direction, attention)

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


@pytest.mark.parametrize(
    ("cos_sin", "degrees"),
    [
        ((1.0, 0.0), 0.0),  # e8 straight up: White at the bottom
        ((0.0, 1.0), 90.0),  # e8 to the right: White on the left
        ((0.0, -1.0), -90.0),  # a quarter turn the other way
        ((-1.0, 0.0), 180.0),  # upside down
        ((0.05, 0.98), 87.08),  # an unnormalised read off a recorded take
    ],
)
def test_direction_is_degrees_clockwise_from_up(cos_sin, degrees):
    out = direction_degrees(np.array([cos_sin], dtype=np.float32))
    assert out == pytest.approx(degrees, abs=0.01)


def test_a_direction_with_no_length_or_a_nan_reports_none():
    assert direction_degrees(np.zeros((1, 2), dtype=np.float32)) is None
    assert direction_degrees(np.array([[np.nan, 1.0]], dtype=np.float32)) is None


def test_logits_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="ChessQ returned"):
        square_dist_from_logits(np.zeros((64, 13), dtype=np.float32))


def test_frame_energy_is_the_mean_of_the_four_highest_square_energies():
    """Per square minus logsumexp of its 13 logits, per frame the mean of the 4
    highest (the hand-signal probe's ``energy_f``): one weak square in 64 moves the
    frame, a board of confident squares stays low."""
    logits = _logits_for(START_PLACEMENT, peak=14.0).astype(np.float64)
    confident = -(14.0 + np.log1p(12 * np.exp(-14.0)))  # every square alike
    assert frame_energy(logits) == pytest.approx(confident)
    logits[0, 27] = 0.0  # d5: no class stands out, energy -log(13)
    logits[0, 28, 0] = 2.0  # e5: a weak "empty"
    expected = (-np.log(13.0) - (2.0 + np.log1p(12 * np.exp(-2.0))) + 2 * confident) / 4
    assert frame_energy(logits) == pytest.approx(expected)


def test_frame_energy_refuses_logits_of_the_wrong_shape():
    with pytest.raises(ValueError, match="ChessQ returned"):
        frame_energy(np.zeros((64, 13), dtype=np.float32))


def test_the_recognizer_reports_the_frame_energy_of_its_logits():
    recognition = ChessQRecognizer(_fake_model(RUY_LOPEZ)).read(render_board(RUY_LOPEZ))
    assert recognition.frame_energy == pytest.approx(frame_energy(_logits_for(RUY_LOPEZ)))


def test_recognizer_reads_in_the_chess_frame():
    recognizer = ChessQRecognizer(_fake_model(RUY_LOPEZ))
    assert (recognizer.name, recognizer.model_path) == ("chessq", None)
    dist = recognizer.read(render_board(RUY_LOPEZ)).square_dist
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
    # The session helper, the seam it implements and the grid fit: nothing else of ours.
    assert ours == {
        "board_reader_inference.recognition.locate",
        "board_reader_inference.recognition.onnxrt",
        "board_reader_inference.recognition.recognizer",
    }


#: Square query ``i``'s attention peak, in patches of the 46x46 grid: file ``c``
#: and rank row ``r`` at patch ``(5c + 3, 5r + 3)``, a board filling most of the input.
def _one_hot_attention() -> np.ndarray:
    side = INPUT_PX // 14
    maps = np.zeros((1, 3, 64, side, side), dtype=np.float32)
    for i in range(64):
        maps[0, :, i, 5 * (i // 8) + 3, 5 * (i % 8) + 3] = 1.0
    return maps


def test_a_read_carries_the_board_grid_from_the_same_run():
    """The attention of the one model call becomes the read's grid, in fractions of
    the frame: square centre c + 0.5 sits at patch 5c + 3, so the board's outer
    edge (c = 0, 8) is at patch 0.5 and 40.5, i.e. (p + 0.5) * 14 / 644 of the frame."""
    model = _fake_model(RUY_LOPEZ, attention=_one_hot_attention())
    read = frame_reader(ChessQRecognizer(model))(_frame(RUY_LOPEZ))
    assert len(model.seen) == 1
    grid = read.grid
    assert grid is not None and grid.trusted and grid.r2 == pytest.approx(1.0)
    lo, hi = 1.0 * 14 / INPUT_PX, 41.0 * 14 / INPUT_PX
    np.testing.assert_allclose(grid.corners, [[lo, lo], [hi, lo], [hi, hi], [lo, hi]], atol=1e-5)
    assert read.event.placement == RUY_LOPEZ  # the grid rides beside the event, not in it


def test_a_model_file_without_attention_reads_with_no_grid():
    assert ChessQRecognizer(_fake_model(RUY_LOPEZ)).read(render_board(RUY_LOPEZ)).grid is None


def test_a_read_takes_the_whole_frame_and_ignores_corners():
    model = _fake_model(RUY_LOPEZ)
    read = _served(model)
    frame = _frame(RUY_LOPEZ)
    event = read(frame)
    assert event.placement == RUY_LOPEZ
    assert (event.board_id, event.seq, event.status) == ("A1", 3, "ok")
    # A legacy quad that would crop most of the board away changes nothing: the
    # model was handed the same whole-frame tensor.
    corners = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32)
    assert read(_frame(RUY_LOPEZ, corners=corners)).placement == RUY_LOPEZ
    np.testing.assert_array_equal(model.seen[0], model.seen[1])
    np.testing.assert_array_equal(model.seen[0], preprocess_frame(frame.image))


def test_black_bishop_demotion_is_not_applied():
    # A narrow b-over-p win (margin 0.1) — the old crop-classifier stack would have
    # swapped it to p; ChessQ's read must keep the model's own argmax.
    logits = _logits_for(START_PLACEMENT)
    c8 = 2  # FEN index of c8, a black bishop in the start position
    logits[0, c8] = -10.0
    logits[0, c8, CHESSQ_CLASSES.index("b")] = np.log(0.55)
    logits[0, c8, CHESSQ_CLASSES.index("p")] = np.log(0.45)
    event = _served(lambda batch: ModelOutputs(logits, FROM_THE_SIDE))(_frame())
    assert event.placement == START_PLACEMENT
    assert event.square_dist[0, 2].argmax() == CLASS_INDEX["b"]


@pytest.fixture
def chessq_env(tmp_path, monkeypatch):
    """``BR_CHESSQ`` naming a (fake) model file."""
    model = tmp_path / "cq.onnx"
    model.write_bytes(b"not loaded: _onnx_model_fn is patched")
    monkeypatch.setattr(chessq_mod, "_onnx_model_fn", lambda path: _fake_model(START_PLACEMENT))
    monkeypatch.setenv("BR_CHESSQ", str(model))
    return model


def test_br_chessq_serves_chessq_from_its_file(chessq_env):
    recognizer = build_from_env()
    assert isinstance(recognizer, ChessQRecognizer)
    assert recognizer.model_path == str(chessq_env)


def test_missing_weights_degrade_to_the_stub_loudly(tmp_path, monkeypatch, capsys):
    """A fresh clone has no weights (plan decision D2): the node still starts, on
    the stub, says so at startup, and ``/healthz`` carries the reason."""
    monkeypatch.setenv("BR_CHESSQ", str(tmp_path / "absent.onnx"))
    recognizer = build_from_env()
    assert isinstance(recognizer, StubRecognizer)
    assert "make models" in (recognizer.fallback or "")
    assert "STUB recognizer" in capsys.readouterr().err
    health = TestClient(create_app(recognizer, PositionHub())).get("/healthz").json()
    assert health["recognizer"]["name"] == "stub"
    assert "absent.onnx" in health["recognizer"]["fallback"]


def test_ingest_publishes_and_settles_a_chessq_read(gateway, drain_ingest):
    client = gateway(create_app(ChessQRecognizer(_fake_model(RUY_LOPEZ)), PositionHub()))
    frame = _frame(RUY_LOPEZ, corners=np.array([[0, 0], [384, 0], [384, 384], [0, 384]], dtype=np.float32))
    header, jpeg = frame.encode()
    with client.websocket_connect("/ingest") as ws:
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()
    drain_ingest(client)
    snap = client.get("/boards/A1").json()
    assert snap["placement"] == RUY_LOPEZ
    assert snap["pending"] == []  # the settled board is the read


def test_the_session_prefers_cuda_and_keeps_the_cpu_fallback(tmp_path, monkeypatch):
    """The model file opens on :func:`onnxrt.cuda_providers` — CUDA when the build
    and the environment allow it, CPU last — so one ``BR_CHESSQ`` serves either."""
    import onnxruntime as ort

    seen: dict[str, object] = {}

    class Output:
        def __init__(self, name: str) -> None:
            self.name = name

    class Session:
        def __init__(self, path: str, providers: list) -> None:
            seen["path"], seen["providers"] = path, providers

        def get_outputs(self) -> list[Output]:
            # A plain release file: no square attention.
            return [Output("direction"), Output("logits")]

        def run(self, names: list[str], feeds: dict) -> list[np.ndarray]:
            seen["outputs"] = names
            # The order the names asked for, whatever order the file declares.
            return [{"logits": _logits_for(RUY_LOPEZ), "direction": FROM_THE_SIDE}[n] for n in names]

    chosen = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(chessq_mod, "cuda_providers", lambda: chosen)
    model = tmp_path / "cq.onnx"
    recognition = ChessQRecognizer(model).read(render_board(RUY_LOPEZ))
    assert seen == {
        "path": str(model),
        "providers": chosen,
        "outputs": ["logits", "direction"],
    }
    assert recognition.square_dist.shape == (8, 8, len(CLASSES))
    assert recognition.direction == pytest.approx(90.0)


def test_replay_config_digest_distinguishes_br_chessq(chessq_env):
    env = replay.config_env()
    assert env["BR_CHESSQ"]["value"] == str(chessq_env)


# --------------------------------------------------------------------------- #
# The real release on one held-out still (`make test-gpu`; skips without files)
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).resolve().parents[4]
_STILL = _ROOT / "data/eval/heldout-20260831/img_0763.label.json"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model",
    [
        _ROOT / "models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1-attn.onnx",  # the CPU file
        _ROOT / "models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32-attn.onnx",  # the GPU file
    ],
    ids=["int8", "fp32"],
)
def test_real_v6_reads_a_held_out_midgame_still(model: Path):
    import cv2

    if not (model.exists() and _STILL.exists()):
        pytest.skip(f"needs {model.relative_to(_ROOT)} and {_STILL.relative_to(_ROOT)} (gitignored)")
    label = json.loads(_STILL.read_text())
    image = cv2.imread(str(_STILL.parent / label["photo"]))
    read = frame_reader(ChessQRecognizer(model))
    reading = read(Frame(board_id="still", seq=0, capture_ts=0.0, image=image))
    assert reading.event.placement == label["placement"]
    # The served -attn files locate the board on every read: within 3 % of the
    # board's width of the hand-checked corners (the probe measured ~1.2 %).
    assert reading.grid is not None and reading.grid.trusted
    h, w = image.shape[:2]
    truth = np.array([float(v) for v in label["corners"].split(",")]).reshape(4, 2)
    error = np.linalg.norm(reading.grid.corners * [w, h] - truth, axis=1).mean()
    assert error < 0.03 * (truth[:, 0].max() - truth[:, 0].min())
