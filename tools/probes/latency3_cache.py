"""Read every frame of the latency-round-3 takes with ChessQ once, and cache it.

    flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/latency3_cache.py OUT_DIR

One ``<take>.npz`` per take in ``OUT_DIR``, over the take's longest streaming
session (the session ``replay`` picks), one row per recorded frame:

* ``seq``, ``ts`` (capture time), ``stab_hdr`` (the header's stability through
  ``schedule.seam.stability_of``) and ``stab_re`` (today's ``MotionMeter`` over every
  frame of the session, what ``replay --restamp-stability`` stamps);
* ``cls`` (64 protocol class indices, a8..h1: the argmax of ``square_dist``, so the
  raw board), ``energy`` (``chessq.frame_energy``, what the energy gate compares),
  ``direction`` (degrees, NaN for none);
* ``logits`` in float16, for anything a later question needs.

The ONNX session is the product's (``ChessQRecognizer`` -> ``onnxrt.cuda_providers``),
fp32 V6, called through its model function so the logits can be kept.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from board_reader_core.protocol import GRID
from board_reader_core.recording import Recording
from board_reader_core.wire import decode_jpeg

from board_reader_inference.measurement import replay as rp
from board_reader_inference.recognition.chessq import (
    ChessQRecognizer,
    direction_degrees,
    frame_energy,
    preprocess_frame,
    square_dist_from_logits,
)
from board_reader_inference.stream.schedule.meter import MotionMeter
from board_reader_inference.stream.schedule.seam import stability_of

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "chessq" / "cq-vit-s-644-v6-seed1-onnx-fp32.onnx"

#: The seven gate takes (tools/gate_table.py) and live take 2 of 2026-11-24.
TAKES = (
    "20260813-193308",
    "20260827-175539",
    "20260828-111421",
    "20260827-181802",
    "20260827-163352",
    "20260827-170333",
    "20260827-172736",
    "20261124-221950",
)
def cache_take(recognizer: ChessQRecognizer, take: str, out: Path) -> None:
    rec = Recording.open(ROOT / "recordings" / take)
    index = list(rec.frame_index(board_id=rp._default_board(rec)))
    session = rp.longest_session(rp.find_sessions(index))
    entries = index[session.start : session.stop]
    meter = MotionMeter()
    rows: dict[str, list] = {k: [] for k in ("seq", "ts", "stab_hdr", "stab_re", "cls",
                                             "energy", "direction", "logits")}
    started = time.monotonic()
    for entry in entries:
        image = decode_jpeg((rec.path / entry.jpeg_rel).read_bytes())
        logits, direction = recognizer.model_fn(preprocess_frame(image))
        dist = square_dist_from_logits(logits)
        bearing = direction_degrees(direction)
        rows["seq"].append(entry.seq)
        rows["ts"].append(entry.capture_ts)
        rows["stab_hdr"].append(stability_of(entry.header.get("stability")))
        rows["stab_re"].append(meter.update(image))
        rows["cls"].append(dist.reshape(GRID * GRID, -1).argmax(axis=-1).astype(np.uint8))
        rows["energy"].append(frame_energy(logits))
        rows["direction"].append(np.nan if bearing is None else bearing)
        rows["logits"].append(np.asarray(logits[0], dtype=np.float16))
    np.savez_compressed(
        out / f"{take}.npz",
        seq=np.array(rows["seq"], np.int32),
        ts=np.array(rows["ts"], np.float64),
        stab_hdr=np.array(rows["stab_hdr"], np.float32),
        stab_re=np.array(rows["stab_re"], np.float32),
        cls=np.stack(rows["cls"]),
        energy=np.array(rows["energy"], np.float64),
        direction=np.array(rows["direction"], np.float64),
        logits=np.stack(rows["logits"]),
    )
    rate = len(entries) / (time.monotonic() - started)
    print(f"{take}: {len(entries)} frames, {rate:.1f} frames/s", flush=True)


def main(out: Path, takes: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    recognizer = ChessQRecognizer(MODEL)
    for take in takes or TAKES:
        if (out / f"{take}.npz").exists():
            print(f"{take}: cached", flush=True)
            continue
        cache_take(recognizer, take, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
