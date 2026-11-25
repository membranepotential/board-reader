---
paths:
  - 'models/chessq/**'
  - 'tools/chessq_export/**'
  - 'tools/fetch_models.py'
  - 'tests/test_chessq_export.py'
  - 'tests/test_fetch_models.py'
---

# The recognizer weights: fetch and export

- Track only the release descriptors (`models/chessq/*.json`) and keep `.gitignore`'s un-ignore chain (`models/*`, `!models/chessq/`, `models/chessq/*`, `!models/chessq/*.json`), because the weights are fetched by `make models`, never committed.
- Keep `tools/fetch_models.py` refusing a non-https URL or a URL without a pinned revision, and verifying size and SHA-256 before it replaces a file, because the descriptor is the only thing that says which bytes we serve.
- Keep the export wrapper's `forward` exactly as exported, no-op dtype casts included, because the export is byte-reproducible under the locked env and the committed descriptor pins its SHA-256; dropping a cast such as `logits.float()` removes an ONNX `Cast` node and changes the bytes, though not the math.
- Check an export by feeding the official package's prepared tensor (`Predictor.prepare_image`) to both the package on CPU fp32 and onnxruntime (max |Δlogit| is ~3e-5), because that isolates the export from preprocessing.
- Let `tools/chessq_export/attention.py` only add nodes and outputs to a release graph, never change or remove one, and keep its parity check bit-exact (`np.array_equal` on `logits` and `direction`), because the served `-attn` files must read exactly like the release files the replay gate measured; their descriptors pin the copies' SHA-256, so a rebuild that changes bytes shows up in `git diff`.
- Run the export only in its own uv project (`uv run --project tools/chessq_export --frozen`), never add torch to the workspace, because the node and the workspace stay torch-free by design.
