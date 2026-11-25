# STATUS — board-reader

_reconciled: 2026-11-24 @ `480a5ea`_ · decisions: [docs/DESIGN.md](docs/DESIGN.md) · history: `git log`, [docs/reports/](docs/reports/)

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those belong to the downstream application layer.

## Now (2 agents, files partitioned)

- **latency round 3, build** — node hold 0.7 → 0.3 s, settle 0.5 → 0.3 s (report `0ada93d`: take 2 median
  1.66 → 0.94 s, 0 leaks; needs the energy gate); gate vs `energy` — running (agent).
- **auto-crop probe** — ChessQ attention centroids → board grid → proposed crop; board 83–15 % of the frame,
  single pass vs coarse-to-fine — running (measure only; tools/probes/autocrop_*).

## Done (2026-11-24 — the ChessQ rebuild; plan: [docs/plans/2026-11-24-simplify-around-chessq.md](docs/plans/2026-11-24-simplify-around-chessq.md))

- **ChessQ is the core recognizer** (Lite V6, fp32 ONNX on CUDA ~0.2 s/read; int8 on CPU); old stack
  deleted (≈ −26k lines: DINOv2/bank/crops/corners/training/RunPod). Labels + truth in the chess frame.
- **Gate: 7 takes**, base `energy`: 341/361 raw, 312/361 settled perfect, 0 hand leaks. Take 2
  (`20261124-221950`) has truth (72/86 settled), kept out of the gate (staged non-moves).
- **Latency:** phone motion thumbnail area-averaged; forwarder tail 1 s, burst-end 3 reads × 0.3 s,
  settle 0.5 s, keepalive 2 s. Live take median move→settle ~2 s.
- **Dumb-camera phone** (D11): `/camera` only films; `/control` sets crop, lens, resolution,
  start/stop over two WebSockets; config persisted on the node; "White sits on …" + camera-moved warning.
  Live take 2 (`recordings/20261124-221950`, 597 frames): user verdict good.
- **Viewer** (D12): SvelteKit 3.3 on Node in `packages/viewer` (`/api` routes; `server/` only for the
  WS relay + TLS); live card: motion chip + pending rings, no confidence bars, no rotate button.
- **One wire contract** (D13): pydantic models in core → JSON Schema → `lib/protocol/generated.ts`
  (zod 4), rebuilt by every `make ui`/`dev`/`test-front`; drift + shared-sample tests; strict headers.
- **Inference regrouped** into recognition/ stream/ control/ measurement/; tests mirror src.
- **Priors → guard** (`5d0cdb2`..`61e1822`): decoder + `prior` field gone; count guard in `stream/rules.py`
  (python-chess); per-board `rules: standard|free`; one placement per event + `pending`.
- **`changed_utc`** (`acfcb8b`, `9fb7671`): back-dated by pixels over unread frames; lag behind the true
  change, median: gate 1.46 → 1.35 s, take 2 0.94 → 0.57 s (rest = hand still on the piece).
- **Energy gate** (`138d941`, D16): obscured reads (ChessQ frame energy > −12.4) never commit; take 2
  resting hand 9 wrong boards → 0, whole take 80 → 88/94; gate 314 → 312/361; "hand?" chip.
- **PGN panel** (`81408ac`): beam Viterbi over legal positions → live PGN, copy, Lichess link; 7/7 games exact.
- **Small fixes:** `serve` honours `BR_RECORD_DIR` (`3be7f47`); `MOVING_BELOW` removed.
- **Docs + rules:** DESIGN.md rewritten; learnings → path-scoped `.claude/rules/` (store frozen);
  big data archived to the HDD (`/run/media/felix/archive/board-reader/`, symlinked).

## Next

1. **Resting-hand re-test** (you): rest a hand on the board mid-game; no wrong board, "hand?" chip shows.
2. **Motion meter over the board only** — 45/60 gate moves wait for the 2 s idle read (the meter sees
   the hand at the clock); crop-aware meter or the hand signal (report `0ada93d`).
3. **Record more varied games** — second set, second camera, other light. All gate takes are one
   set; the settle defaults are fitted to it.
4. **Python edge on the control channel** (today: static `--roi`) — needed for a Pi/Jetson camera.
5. **Model the remaining HTTP replies in core** (`/estimate`, `/uploads`, `/labels`, the `/ingest`
   ack) so they are generated too.
6. **Idea, parked:** fully on-phone portable app — [docs/on-phone-app.md](docs/on-phone-app.md).

## Open questions

- **Where do positions go?** No downstream consumer yet; the on-phone idea makes the "sink"
  question concrete (Lichess broadcast/PGN, a WebSocket endpoint, local file).
- **Target hardware latency** — ChessQ fp32 needs a GPU for ~0.2 s; Jetson/Pi unmeasured.
- **Licence** — ChessQ weights are PolyForm Noncommercial (accepted 2026-11-24); a commercial use
  would need another model.
