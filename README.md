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
orientation itself, and on the four-take replay gate reads 92/101 raw and 84/101
settled boards perfectly (the old cascade + bank: 5 and 35). Point a phone at a
board via `/snapshot` or `/camera` and you get a position back with no taps and no
calibration.

The most recent live phone trial (2026-08-13) streamed **748/748 frames** without loss.
Scored against hand-labelled truth for that recording (`recordings/<session>/report.json`,
regenerate with `board-reader-inference eval-session`), the debounced `settled` board is
**97.7 % per-square** and **25.8 % perfect boards** — that row is replayed offline, since
the recording predates the settled block shipping. The residual is *entirely* one
confusion — king vs queen on d8/e8, where a phone-height camera decapitates the rank-8
crops — so a perfect board costs getting both of those squares right at once.

**Next up:** simplifying the pipeline around ChessQ, then retuning the settle window
([the plan](docs/plans/2026-11-24-simplify-around-chessq.md)). See
[STATUS.md](STATUS.md) for the current state and [DESIGN §7](docs/DESIGN.md) for the
milestone ladder.

## Quick start

```bash
make models                     # the ChessQ weights: pinned revision, SHA-256 checked
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
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx \
  uv run board-reader-inference serve          # gateway  :8000
uv run board-reader-viewer --port 8080         # UI       :8080
uv run board-reader-edge feed                  # synthetic feed
uv run board-reader-inference positions        # tail the position stream (headless)
make test                                      # the test suite
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

## License

Apache-2.0 — see [LICENSE](LICENSE).

Two carve-outs, both spelled out in **[THIRD-PARTY.md](THIRD-PARTY.md)**:

- **The built viewer bundle is GPL-3.0-or-later.** `packages/viewer/ui` links
  [chessground](https://github.com/lichess-org/chessground) (GPL-3.0-or-later) and its
  cburnett sprites (GPLv2+), so whatever `make ui` produces is a combined GPL work.
  Our source stays Apache-2.0, and `core`/`edge`/`inference` carry no GPL dependency
  at all.
- **Model weights are not redistributable.** The core recognizer, ChessQ Lite V6, is
  **PolyForm Noncommercial 1.0.0**: personal and research use only, **not** commercial
  use. `make models` fetches it from its pinned upstream revision; only its release
  descriptor (URL, size, SHA-256) is committed. The older in-house models derive from
  ChessReD (CC BY-NC-SA 4.0) and geometry of unclear provenance. `models/`, `data/`,
  `scratch/` and `recordings/` are git-ignored for exactly this reason — do not attach
  weights to a release or bake them into a published image.
