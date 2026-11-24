# STATUS — board-reader

_reconciled: 2026-11-24 @ `480a5ea`_ · decisions: [docs/DESIGN.md](docs/DESIGN.md) · history: `git log`, [docs/reports/](docs/reports/)

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those belong to the downstream application layer.

## Now (2 agents, files partitioned)

- **hand-signal probe DONE** (`8645523`): ChessQ frame energy (−logsumexp, mean of top-4 squares) flags hand
  frames at AUC 0.97; as a no-commit gate: resting-hand take 9 wrong boards → 0, gate 314 → 311/361.
- **priors DONE** (`5d0cdb2`..`61e1822`): decoder + `prior` field gone; count guard in `stream/rules.py`
  (python-chess, inference only); per-board `rules: standard|free`; one placement per event + `pending` +
  `changed_utc` (median 1.45 s after the true change vs 3.12 s); gate base `guard-only` 314/361, 0 leaks.
- **PGN builder DONE** (`81408ac`, `2bd0f32`): beam Viterbi over legal positions (chessops) → live game panel
  with copy + Lichess link; gate takes 7/7 games exact (80/80 plies), commit lag median 3.0 s.
- **changed_utc refine** — after a settle, pixel-compare the changed squares in the held frames (no extra
  ChessQ reads) to find the first frame of the new board; take 2 truth — running (agent; refine module,
  ingest.py, replay).
- **energy gate** — obscured reads never commit (chessq.py score, stabilize.py, `obscured` on the event,
  "hand?" chip); wires ingest.py after the refine commit — running (agent).

## Done (2026-11-24 — the ChessQ rebuild; plan: [docs/plans/2026-11-24-simplify-around-chessq.md](docs/plans/2026-11-24-simplify-around-chessq.md))

- **ChessQ is the core recognizer** (Lite V6, fp32 ONNX on CUDA ~0.2 s/read; int8 on CPU); old stack
  deleted (≈ −26k lines: DINOv2/bank/crops/corners/training/RunPod). Labels + truth in the chess frame.
- **Gate: 7 takes**, base `sp03-settle05-7`: 341/361 raw, 315/361 settled perfect, 0 hand leaks.
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
- **Small fixes:** `serve` honours `BR_RECORD_DIR` (`3be7f47`); `MOVING_BELOW` removed.
- **Docs + rules:** DESIGN.md rewritten; learnings → path-scoped `.claude/rules/` (store frozen);
  big data archived to the HDD (`/run/media/felix/archive/board-reader/`, symlinked).

## Next

1. **Latency round 3** (after the energy gate): read the first calm frame instead of holding 0.7 s; shorter
   settle once obscured reads cannot commit; gate sweep, keep only no-loss settings (~2 s → ~0.5–0.8 s?).
2. **Resting-hand re-test** (you) — after the energy gate lands: rest a hand on the board mid-game.
3. **Record more varied games** — second set, second camera, other light. All gate takes are one
   set; the settle defaults are fitted to it.
4. **Python edge on the control channel** (today: static `--roi`) — needed for a Pi/Jetson camera.
5. **Model the remaining HTTP replies in core** (`/estimate`, `/uploads`, `/labels`, the `/ingest`
   ack) so they are generated too.
6. **Idea, parked:** fully on-phone portable app — [docs/on-phone-app.md](docs/on-phone-app.md).
   (ChessCam v2 dropped 2026-11-24 by decision.)

## Open questions

- **Where do positions go?** No downstream consumer yet; the on-phone idea makes the "sink"
  question concrete (Lichess broadcast/PGN, a WebSocket endpoint, local file).
- **Target hardware latency** — ChessQ fp32 needs a GPU for ~0.2 s; Jetson/Pi unmeasured.
- **Licence** — ChessQ weights are PolyForm Noncommercial (accepted 2026-11-24); a commercial use
  would need another model.
