# board-reader

Camera-based chess **board reader**. A computer-vision system that watches a live
over-the-board game and streams **timestamped piece placements with per-square
confidences** — a low-cost, no-special-equipment alternative to DGT electronic
boards. It reads *positions*, not moves: legality, move reconstruction, and PGN are
the downstream application layer's job (and out of scope here).

> **Design:** [docs/DESIGN.md](docs/DESIGN.md) · **Testing:** [docs/TESTING.md](docs/TESTING.md)
> · **Contributor guide:** [CLAUDE.md](CLAUDE.md)

## Status

**M0 — walking skeleton** (the end-to-end pipe is live and demoable). The estimator
is currently a stub that reports the start position; real per-frame recognition
(DINOv2 + classifier head) lands in **M1**. See milestones in DESIGN §7.

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
docs/          DESIGN.md, TESTING.md, youtube-streams.md
```

The **edge** and **inference** nodes never share code — they meet only on the wire.
This is enforced by `tests/test_layer_separation.py`.
