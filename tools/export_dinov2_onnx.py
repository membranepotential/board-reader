"""Export the official Meta DINOv2 ViT-S/14 to ONNX (DESIGN.md §4.6, D9).

We deliberately download **official first-party weights** (`facebook/dinov2-small`,
Meta's own HF org — Apache-2.0) and convert to ONNX **ourselves**, rather than pulling
a third-party pre-converted artifact. The exported model emits the CLS embedding
(384-d) used by the recognizer's centroid classifier.

This is one-time export tooling, kept out of the lean inference runtime. Run it in an
isolated environment with torch + transformers + onnx (see tools/README.md):

    uv run python tools/export_dinov2_onnx.py --out models/dinov2_small.onnx

The runtime only needs onnxruntime to load the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EMBED_DIM = 384  # DINOv2 ViT-S/14 CLS embedding width
MODEL_ID = "facebook/dinov2-small"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/dinov2_small.onnx", help="output .onnx path")
    ap.add_argument("--model", default=MODEL_ID, help="official HF model id (Meta)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading official weights: {args.model}")
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    class ClsEmbedding(torch.nn.Module):
        """Expose just the pooled CLS embedding (B, 384) — what the classifier wants."""

        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            return self.m(pixel_values=pixel_values).pooler_output

    wrapper = ClsEmbedding(model).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        ref = wrapper(dummy)
    assert ref.shape[-1] == EMBED_DIM, f"expected {EMBED_DIM}-d, got {tuple(ref.shape)}"

    print(f"exporting -> {out} (opset {args.opset}, dynamic batch)")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # The exporter may spill weights to a sidecar (.onnx.data). Consolidate into a
    # single self-contained file so the artifact can't be split from its weights.
    import onnx

    model_proto = onnx.load(str(out))  # pulls in any external data
    onnx.save_model(model_proto, str(out), save_as_external_data=False)
    sidecar = out.with_name(out.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"done: {out} ({out.stat().st_size / 1e6:.1f} MB), embedding dim {EMBED_DIM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
