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

**M0 + M1 done; M2 in flight.** The end-to-end pipe is live and demoable. Since
2026-11-24 the core recognizer is **ChessQ Lite V6** (Lichess's end-to-end board
recognizer, DESIGN D10): it reads the whole frame, finds the board and its
orientation itself. On the seven-take replay gate the shipped stack reads 341/361 boards
perfectly per read and 315/361 in the settled stream (on the first four-take gate the old
cascade + bank managed 5 and 35 of 101 where ChessQ first managed 92 and 84). Point a phone at a
board via `/snapshot`, or stream one from `/camera` (the phone only films; you draw the
board's rectangle, pick the lens and start the stream from `/control` on the laptop),
and you get a position back with no taps on the phone and no calibration.

The pipeline was then simplified around ChessQ: the calibration round, the exemplar
bank, the in-house crop classifier and its training harness are gone, and every frame
takes one read path ([the plan](docs/plans/2026-11-24-simplify-around-chessq.md),
[recognition](docs/recognition-pipeline.md)).

**Next up:** a live take with the dumb-camera phone (Phase C), then more varied
recordings to check the settle defaults. See
[STATUS.md](STATUS.md) for the current state and [DESIGN §7](docs/DESIGN.md) for the
milestone ladder.

## Quick start

```bash
make models                     # the ChessQ weights: pinned, SHA-256 checked, fp32 exported, -attn copies
docker compose up --build       # then open http://localhost:8080
```

You'll see a live board fed by a synthetic edge: `edge → inference → viewer`. The
synthetic renders are not real pieces, so ChessQ reads them only partly: the demo
shows the pipe working, a real board on `/camera` shows recognition. Without
`make models` the node serves a stub (a fixed start position) and says so.

Or run locally without Docker:

```bash
uv sync
make models
make dev                                       # inference + viewer, ChessQ serving
# or by hand:
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32-attn.onnx \
  uv run board-reader-inference serve          # gateway  :8000 (GPU file; int8 for CPU)
node packages/viewer/server/main.ts --port 8080   # UI :8080 (after `make ui`)
uv run board-reader-edge feed                  # synthetic feed
uv run board-reader-inference positions        # tail the position stream (headless)
make test                                      # the test suite
```

## Layout

```
packages/
  core/        board_reader_core      — shared L3 protocol + L1→L2 wire + sim
  edge/        board_reader_edge      — the "Pi": capture & forward frames
  inference/   board_reader_inference — stateless per-frame recognizer + WS gateway
  viewer/      the viewer (Node)      — the SvelteKit pages + server/main.ts, the relay to inference
docker/        the Python image (edge, inference) and viewer.Dockerfile; compose.yaml wires all three
tools/         offline tooling: model fetch, replay gate table, label migration, video capture
docs/          DESIGN.md (decisions) + an Obsidian-linked doc set (index.md)
```

The **edge** and **inference** nodes never share code — they meet only on the wire.
This is enforced by `tests/test_layer_separation.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).

Two carve-outs, both spelled out in **[THIRD-PARTY.md](THIRD-PARTY.md)**:

- **The built viewer bundle is GPL-3.0-or-later.** `packages/viewer` links
  [chessground](https://github.com/lichess-org/chessground) (GPL-3.0-or-later) and its
  cburnett sprites (GPLv2+), so whatever `make ui` produces is a combined GPL work.
  Our source stays Apache-2.0, and `core`/`edge`/`inference` carry no GPL dependency
  at all.
- **Model weights are not redistributable.** The core recognizer, ChessQ Lite V6, is
  **PolyForm Noncommercial 1.0.0**: personal and research use only, **not** commercial
  use. `make models` fetches it from its pinned upstream revision (the int8 ONNX and
  the FP16 safetensors), exports the fp32 ONNX and writes the `-attn` copies the
  node serves; only the
  descriptors (URL, size, SHA-256, export provenance) are committed. `models/`, `data/`,
  `scratch/` and `recordings/` are git-ignored for exactly this reason — do not attach
  weights to a release or bake them into a published image.
