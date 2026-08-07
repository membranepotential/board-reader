# board-reader

Camera-based chess **board reader**. A computer-vision system that watches a live
over-the-board game and streams **timestamped piece placements with per-square
confidences** — a low-cost, no-special-equipment alternative to DGT electronic
boards. It reads *positions*, not moves: legality, move reconstruction, and PGN are
the downstream application layer's job (and out of scope here).

> **Docs:** [docs/index.md](docs/index.md) · **Design:** [docs/DESIGN.md](docs/DESIGN.md)
> · **Testing:** [docs/TESTING.md](docs/TESTING.md) · **Status:** [STATUS.md](STATUS.md)
> · **Contributor guide:** [CLAUDE.md](CLAUDE.md)

## Status

**M0 + M1 done.** The end-to-end pipe is live and demoable, and per-frame recognition
reads real phone photos: a multi-style-trained crop classifier blended with a per-set
nearest-exemplar bank, at **633/640** squares on the labeled evaluation set. Point a
phone at a board via `/snapshot` and you get a position back with no taps — corners are
auto-detected, orientation auto-resolved, calibration bootstraps itself.

**Next up: M2** — a live position stream over a full recorded game. See
[STATUS.md](STATUS.md) for the current state and [DESIGN §7](docs/DESIGN.md) for the
milestone ladder.

## Quick start

```bash
docker compose up --build       # then open http://localhost:8080
```

You'll see a live board fed by a synthetic edge: `edge → inference → viewer`.

Or run locally without Docker:

```bash
uv sync
uv run board-reader-inference serve            # gateway  :8000
uv run board-reader-viewer --port 8080         # UI       :8080
uv run board-reader-edge feed                  # synthetic feed
uv run board-reader-inference positions        # tail the position stream (headless)
uv run pytest                                  # the test suite
```

## Layout

```
packages/
  core/        board_reader_core      — shared L3 protocol + L1→L2 wire + sim
  edge/        board_reader_edge      — the "Pi": capture & forward frames
  inference/   board_reader_inference — stateless per-frame estimator + WS gateway
  viewer/      board_reader_viewer    — serves the live-position web UI
docker/        one parametrised image; compose.yaml wires the three services
tools/         offline tooling: bank builders, ONNX export, training harness
docs/          DESIGN.md (decisions) + an Obsidian-linked doc set (index.md)
aspects.yaml   which parts must stay in sync with which, and why
```

The **edge** and **inference** nodes never share code — they meet only on the wire.
This is enforced by `tests/test_layer_separation.py`.
