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

**M0 + M1 done; M2 in flight.** The end-to-end pipe is live and demoable, and per-frame
recognition reads real phone photos: a multi-style-trained crop classifier blended with
a per-set nearest-exemplar bank. Point a phone at a board via `/snapshot` and you get a
position back with no taps — corners are auto-detected, orientation auto-resolved,
calibration bootstraps itself.

The most recent live phone trial (2026-08-13) streamed **748/748 frames** without loss.
Scored against hand-labelled truth for that recording, the debounced `settled` board is
**98.96 % per-square** and **66.6 % perfect boards**; the residual is almost entirely one
confusion — king vs queen on d8/e8, where a phone-height camera decapitates the rank-8
crops.

**Next up:** the tall-crop retrain that attacks that residual, gated on real recordings
rather than a synthetic holdout. See [STATUS.md](STATUS.md) for the current state and
[DESIGN §7](docs/DESIGN.md) for the milestone ladder.

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

## License

Apache-2.0 — see [LICENSE](LICENSE).

Two carve-outs, both spelled out in **[THIRD-PARTY.md](THIRD-PARTY.md)**:

- **The built viewer bundle is GPL-3.0-or-later.** `packages/viewer/ui` links
  [chessground](https://github.com/lichess-org/chessground) (GPL-3.0-or-later) and its
  cburnett sprites (GPLv2+), so whatever `make ui` produces is a combined GPL work.
  Our source stays Apache-2.0, and `core`/`edge`/`inference` carry no GPL dependency
  at all.
- **Trained model artifacts are not redistributable.** The recognition model derives
  from ChessReD (CC BY-NC-SA 4.0) and geometry of unclear provenance: fine for personal
  and research use, **not** for commercial use, and not ours to hand on. `models/`,
  `data/`, `scratch/` and `recordings/` are git-ignored for exactly this reason — do not
  attach weights to a release or bake them into a published image. Run
  `tools/export_dinov2_onnx.py` and calibrate against your own board instead.
