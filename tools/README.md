# tools/

One-off developer tooling that is **not** part of any deployable package.

## `export_dinov2_onnx.py` — DINOv2 ViT-S/14 → ONNX

Produces the recognition backbone (`models/dinov2_small.onnx`, ~90 MB, git-ignored)
from **official Meta weights** (`facebook/dinov2-small`, Apache-2.0). We download the
first-party weights and convert ourselves — we never pull a third-party pre-converted
ONNX (supply-chain trust; see DESIGN §4.6).

The heavy export deps (torch, transformers, onnx) are kept **out of the board-reader
workspace** in an isolated uv project at `../dinov2-export/` so they never touch the
lean inference runtime. The runtime only needs `onnxruntime` (a dependency of
`board-reader-inference`) to load the result.

```bash
# from the isolated tooling project (CPU torch):
cd ../dinov2-export
uv run python ../board-reader/tools/export_dinov2_onnx.py \
  --out ../board-reader/models/dinov2_small.onnx
```

The exported model: input `pixel_values [batch,3,224,224]` (ImageNet-normalised RGB),
output `embedding [batch,384]` (the pooled CLS token), dynamic batch. `DinoV2Embedder`
(in `board_reader_inference.embed`) loads it via ONNX Runtime; point elsewhere with
`BR_DINOV2_ONNX`.
