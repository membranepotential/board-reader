# STATUS — board-reader

_reconciled: 2026-11-24 @ `46e5ff7`_ · decisions: [docs/DESIGN.md](docs/DESIGN.md) · history: `git log`, [docs/reports/](docs/reports/)

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those belong to the downstream application layer.

## Now (5 agents in parallel, files partitioned; live re-test after all land)

- **regroup + 7-take gate DONE** (`a758020`..`f90efc0`): inference in recognition/ stream/ control/ measurement/,
  tests mirror it, import-direction test; gate events identical. 7-take base `sp03-settle05-7`: 341/361 raw,
  315/361 settled perfect.
- **protocol DONE** (`1b52a3b`..`df5c93f`, strict header `d2793ea`): pydantic models in core → JSON Schema →
  `tools/gen_protocol.py` → `lib/protocol/generated.ts` (zod 4); drift + shared-sample tests; gate identical.
- **viewer move DONE**: `packages/viewer/ui` → `packages/viewer`, `$lib` → `#lib` (package.json `imports`).
- **latency2+liveview DONE** (`5c1b9a0`, `8ea0057`; report `8664336`): tail 1 s, 3 reads × 0.3 s, settle
  0.5 s; gate settled 138/154 → 140/156, 0 hand leaks; live view: motion chip + pending rings, no conf bars.
  OPEN: a hand resting still settles (meter sees rest) — needs a direct hand signal.
- **viewer-svelte DONE** (`ba87f7e`, `8d7091f`, `aac0484`): SvelteKit 3.3 on Node — `/api` as routes, short
  `server/{main,ws,tls}.ts` for the socket relay + TLS; Python viewer deleted. Open: `serve` ignored `BR_RECORD_DIR` in the agent's smoke (check).
- **read-path DONE** (`57b145c`, `7be3a41`): motion updates on /viewer, direction guard (0 drops, gate identical).
- **control DONE** (`d916df6`, `cc0233d`, `5b54d74`): camera config persisted (`BR_CAMERA_CONFIG`, default
  `data/camera-config.json`); /control shows "White sits on …" + camera-moved warning.
- **held-out truth DONE** (`1bfea35`): 163352/170333/172736 complete + boundaries fixed → can join the gate.
- WS control channel DONE (`3d2b58c`); vestigial `Recognizer.frame` removed (`02157ea`).

## Done (2026-11-24 — the ChessQ rebuild; plan: [docs/plans/2026-11-24-simplify-around-chessq.md](docs/plans/2026-11-24-simplify-around-chessq.md))

- **ChessQ is the core recognizer** (ChessQ Lite V6, fp32 ONNX on CUDA ~0.2 s/read; int8 on CPU).
  Gate vs the old stack: settled perfect 35 → 104/119; reports in `docs/reports/2026-11-24-*`.
- **Old stack deleted** (≈ −26k lines): DINOv2/bank/calibration/crop classifiers/routing, corner
  detection + rectification, tools/train, RunPod, aspects.yaml. Labels + truth in the chess frame.
- **Settle tuned** (1.0 s / 2 reads, burst-end 3 reads × 0.6 s): median move→settle 4.3 → 3.0 s on
  replays, 0 hand leaks. Framing: board must fill ≥ ~50 % of the sent width.
- **Dumb-camera phone** (DESIGN D11): `/camera` only films; laptop `/control` sets crop, lens,
  resolution, start/stop. Live take 1 (`recordings/20261124-200631`): reads good, latency bad (see Now).
- **Live latency fixed** (`bf23c2a`, `3cf765b`): phone motion thumbnail point-sampled sensor noise →
  no bursts; now area-averaged. Keepalive 3 → 2 s. Gate settled 104/119 → 138/154, 0 hand leaks;
  live-take median move→settle 3.1 → 2.1 s (replayed). Gate base is now `ka2-tuned`.
- **Docs rewritten:** DESIGN.md 1,103 → 618 lines (superseded decisions in one table), all notes
  audited, mockups deleted, `docs/on-phone-app.md`.
- **Learnings → 14 path-scoped rules** in `.claude/rules/`; store frozen (13 verdicts left).
- **Data hygiene:** truth fixed for 181802 + 111421; big datasets/videos/old models archived to the HDD
  (`/run/media/felix/archive/board-reader/`, symlinked; mount it before using them).

## Next

1. **User re-test** (TESTING Option J): verify bursts open on the iPhone (expect ~2 s from the hand
   leaving), then score the take (build its truth).
2. **Truth check for the 3 non-gate takes** (163352, 170333, 172736): moves read but missing from truth
   (same pattern as the 111421 fix) — fix by sweep + eye check, then add them to the gate.
3. **Record more varied games** — second physical set, second camera, mixed openings. The gate is still
   4 takes on one set; settle defaults are fitted to it.
4. **Python edge on the control channel** (today: static `--roi`) — needed for a Pi/Jetson camera.
5. **ChessCam v2 as the fast/portable recognizer** (45 ms CPU, emits square centres) — candidate for the
   Pi/Jetson and the on-phone idea; one more file behind the recognizer seam.
6. **Turn on the prior's back-rank constraint** — `prior.decode` is never given the frame, though the
   chess frame is now always known; pass `quarter_turns=0` and gate it.
7. **Idea, parked:** fully on-phone portable app — [docs/on-phone-app.md](docs/on-phone-app.md).

## Open questions

- **Where do positions go?** No downstream consumer yet (deferred 2026-08-28); the on-phone idea makes
  the "sink" question concrete (Lichess broadcast/PGN, a WebSocket endpoint, local file).
- **Target hardware latency** — ChessQ fp32 needs a GPU for ~0.2 s; Jetson/Pi unmeasured.
- **Confidence is not a hand signal** — ChessQ reads squares under a hand as empty at p ≥ 0.96; only
  burst-end scheduling + the debounce absorb it. Enough on more varied takes?
- **Licence** — ChessQ weights are PolyForm Noncommercial (accepted 2026-11-24); a commercial use
  would need another model.
