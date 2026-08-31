"""Export an official Meta DINO backbone (v2 or v3) to ONNX (DESIGN.md §4.6, D9).

We deliberately download **official first-party weights** (`facebook/dinov2-small` /
`facebook/dinov3-vits16-pretrain-lvd1689m`, Meta's own HF org) and convert to ONNX
**ourselves**, rather than pulling a third-party pre-converted artifact. The exported
model emits the pooled embedding used by the recognizer's centroid classifier.

Geometry (patch size, embedding width, register-token count) is read from the model
config, so the same script handles DINOv2 (patch 14, no registers) and DINOv3
(patch 16, 4 register tokens — DINOv3 weights are license-gated on HF; accept it on
the model page first).

This is one-time export tooling, kept out of the lean inference runtime. Run it in an
isolated environment with torch + transformers + onnx (see tools/README.md):

    uv run python tools/export_dinov2_onnx.py --out models/dinov2_small.onnx

The runtime only needs onnxruntime to load the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "facebook/dinov2-small"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/dinov2_small.onnx", help="output .onnx path")
    ap.add_argument("--model", default=MODEL_ID, help="official HF model id (Meta)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument(
        "--pooling", choices=["cls", "central"], default="cls",
        help="cls: the pooled CLS token (the shipped default). central: mean of "
             "the middle --keep-columns patch-token columns, measured +9/320 "
             "cross-style on the same weights.",
    )
    ap.add_argument(
        "--keep-columns", type=int, default=None,
        help="--pooling central: how many token columns to keep (default: half the "
             "grid — 8/16 on DINOv2, the measured configuration)",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading official weights: {args.model}")
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # Geometry comes from the checkpoint, not from constants: DINOv2 ViT-S/14 has
    # patch 14 and no register tokens; DINOv3 ViT-S/16 has patch 16 and 4 register
    # tokens sitting between CLS and the patch tokens in last_hidden_state.
    patch: int = model.config.patch_size
    embed_dim: int = model.config.hidden_size
    n_prefix: int = 1 + getattr(model.config, "num_register_tokens", 0)
    print(f"geometry: patch {patch}, embed {embed_dim}-d, {n_prefix} prefix token(s)")

    class ClsEmbedding(torch.nn.Module):
        """Expose just the pooled CLS embedding (B, D) — what the classifier wants."""

        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.m(pixel_values=pixel_values).pooler_output

    class CentralColumnEmbedding(torch.nn.Module):
        """Mean of the middle ``keep`` patch-token **columns** (B, 384).

        A square crop is mostly board, and a standing piece occupies a narrow
        vertical strip through the middle of it. Averaging only the central
        columns of the 16x16 patch grid drops most of the neighbouring-square
        context that CLS pooling mixes in. Measured on the frozen-backbone
        bakeoff (ChessReD bank -> user photos): **256/320 cross-style vs 247 for
        CLS on the same weights** — the single transferable win of that campaign,
        and the reason this export mode exists.
        """

        def __init__(self, m: torch.nn.Module, *, grid: int, keep: int, prefix: int) -> None:
            super().__init__()
            self.m = m
            self.grid = grid
            self.keep = keep
            self.prefix = prefix

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Drop ALL prefix tokens (CLS + any register tokens), not just CLS —
            # DINOv3 puts 4 register tokens before the patch grid.
            tokens = self.m(pixel_values=pixel_values).last_hidden_state[:, self.prefix :, :]
            # Reshape with a **-1 batch**: tracing a concrete `tokens.shape[0]`
            # bakes the export-time batch into the graph, and the model then
            # only ever accepts batches of that one size (it exported cleanly
            # and failed at the first 64-crop board).
            grid = tokens.reshape(-1, self.grid, self.grid, tokens.shape[-1])
            lo = (self.grid - self.keep) // 2
            return grid[:, :, lo : lo + self.keep, :].mean(dim=(1, 2))

    if args.pooling == "cls":
        wrapper = ClsEmbedding(model).eval()
    else:
        if args.image_size % patch:
            raise SystemExit(f"--image-size {args.image_size} not divisible by patch {patch}")
        grid = args.image_size // patch
        keep = args.keep_columns if args.keep_columns is not None else grid // 2
        if keep > grid:
            raise SystemExit(f"--keep-columns {keep} > patch grid {grid}")
        print(
            f"pooling: central {keep}/{grid} token columns "
            f"({args.image_size}px / patch {patch})"
        )
        wrapper = CentralColumnEmbedding(
            model, grid=grid, keep=keep, prefix=n_prefix
        ).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        ref = wrapper(dummy)
    assert ref.shape[-1] == embed_dim, f"expected {embed_dim}-d, got {tuple(ref.shape)}"

    print(f"exporting -> {out} (opset {args.opset}, dynamic batch)")
    # `dynamo=False` pins the legacy TorchScript exporter, which honours
    # `dynamic_axes`. torch 2.12's dynamo path silently traces the batch as a
    # CONSTANT here: the graph exports and validates at batch 1, then dies on the
    # first real 64-crop board with a reshape error from deep inside the
    # backbone. A model that only accepts the batch size it was exported at is
    # useless to us — a board is 64 crops and calibration hands over hundreds.
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # The exporter may spill weights to a sidecar (.onnx.data). Consolidate into a
    # single self-contained file so the artifact can't be split from its weights.
    import onnx

    model_proto = onnx.load(str(out))  # pulls in any external data
    onnx.save_model(model_proto, str(out), save_as_external_data=False)
    sidecar = out.with_name(out.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(
        f"done: {out} ({out.stat().st_size / 1e6:.1f} MB), embedding dim {embed_dim}, "
        f"pooling {args.pooling}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
