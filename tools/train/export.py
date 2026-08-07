"""Export a trained checkpoint to ONNX and verify onnxruntime parity.

The exported graph is the proposed inference seam: input ``crop`` (N,3,224,224,
ImageNet-normalised RGB) -> output ``logits`` (N,13). Softmax of ``logits`` is the
per-square distribution the estimator assembles into an 8x8x13 ``square_dist``.
Batch is a dynamic axis (a board is 64 crops; calibration hands hundreds).

Round-trips a random batch through torch and onnxruntime and asserts the logits
match within tolerance, so a green export means the deployed CPU/GPU path will
reproduce training-time predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from config import Config

from board_reader_core.protocol import NUM_CLASSES
from models import build_model

SIZE = 224


def load_model(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"]) if isinstance(ckpt.get("config"), dict) else Config()
    model = build_model(
        cfg.model, pretrained=False, unfreeze_blocks=cfg.unfreeze_blocks,
        freeze_backbone=cfg.freeze_backbone,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def export(ckpt_path: Path, out_path: Path, *, opset: int = 17) -> None:
    model = load_model(ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["crop"], output_names=["logits"],
        dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"exported -> {out_path}")
    # Geometry sidecar the deployed TrainedEstimator reads (<model>.onnx.meta.json).
    # This IS the train/serve contract: whatever cut the training crops must cut
    # the inference crops, and the only thing carrying that across is this file.
    import json
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config", {})
    cfg = Config(**raw) if isinstance(raw, dict) else Config()
    meta = {
        "crop_geometry": cfg.geometry.to_dict(),
        # Kept so a sidecar stays readable by pre-ADT code paths; bbox-only.
        "headroom": float(getattr(cfg.geometry, "headroom", 0.0)),
        "side_pad": float(getattr(cfg.geometry, "side_pad", 0.0)),
        "square_size": SIZE,
    }
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta))
    print(f"wrote {meta_path}: {meta}")
    roundtrip(model, out_path)


def roundtrip(model: torch.nn.Module, onnx_path: Path, *, n: int = 8, atol: float = 1e-3) -> None:
    import onnxruntime as ort

    x = torch.randn(n, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    if got.shape != (n, NUM_CLASSES):
        raise SystemExit(f"onnx output shape {got.shape} != {(n, NUM_CLASSES)}")
    max_abs = float(np.abs(ref - got).max())
    agree = int((ref.argmax(1) == got.argmax(1)).sum())
    print(f"round-trip: max|torch-onnx| = {max_abs:.2e}, argmax agree {agree}/{n}")
    if max_abs > atol:
        raise SystemExit(f"FAIL: logits differ by {max_abs:.2e} > {atol}")
    print("round-trip OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <ckpt dir>/model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = args.out or args.ckpt.parent / "model.onnx"
    export(args.ckpt, out, opset=args.opset)


if __name__ == "__main__":
    main()
