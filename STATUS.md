# STATUS — board-reader

_reconciled: 2026-11-24 @ `46e5ff7`_ · decisions: [docs/DESIGN.md](docs/DESIGN.md) · history: `git log`, [docs/reports/](docs/reports/)

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those belong to the downstream application layer.

## Now

- **latency-fix** — phone motion meter stuck at stability 1.000 in live take 1 (no bursts → 3–6 s per
  move); fix meter + evaluate keepalive ~1 s / settle 0.5 s on the gate — implementer-high running — next:
  gate + held-out tables, then user re-test.

## Done (2026-11-24 — the ChessQ rebuild; plan: [docs/plans/2026-11-24-simplify-around-chessq.md](docs/plans/2026-11-24-simplify-around-chessq.md))

- **ChessQ is the core recognizer** (ChessQ Lite V6, fp32 ONNX on CUDA ~0.2 s/read; int8 on CPU).
  Gate vs the old stack: settled perfect 35 → 104/119; reports in `docs/reports/2026-11-24-*`.
- **Old stack deleted** (≈ −26k lines): DINOv2/bank/calibration/crop classifiers/routing, corner
  detection + rectification, tools/train, RunPod, aspects.yaml. Labels + truth in the chess frame.
- **Settle tuned** (1.0 s / 2 reads, burst-end 3 reads × 0.6 s): median move→settle 4.3 → 3.0 s on
  replays, 0 hand leaks. Framing: board must fill ≥ ~50 % of the sent width.
- **Dumb-camera phone** (DESIGN D11): `/camera` only films; laptop `/control` sets crop, lens,
  resolution, start/stop. Live take 1 (`recordings/20261124-200631`): reads good, latency bad (see Now).
- **Docs rewritten:** DESIGN.md 1,103 → 618 lines (superseded decisions in one table), all notes
  audited, mockups deleted, `docs/on-phone-app.md`.
- **Learnings → 14 path-scoped rules** in `.claude/rules/`; store frozen (13 verdicts left).
- **Data hygiene:** truth fixed for 181802 + 111421; big datasets/videos/old models archived to the HDD
  (`/run/media/felix/archive/board-reader/`, symlinked; mount it before using them).

## Next

1. **User re-test** after latency-fix lands (TESTING Option J), then score the take (build its truth).
2. **Camera control over WebSockets, no polling** (user): config pushed down the phone's `/ingest`
   socket, report on change, preview only while `/control` watches; `/control` gets one push socket.
   Starts after latency-fix (same files).
3. **Truth check for the 3 non-gate takes** (163352, 170333, 172736): moves read but missing from truth
   (same pattern as the 111421 fix) — fix by sweep + eye check, then add them to the gate.
4. **Record more varied games** — second physical set, second camera, mixed openings. The gate is still
   4 takes on one set; settle defaults are fitted to it.
5. **Python edge on the control channel** (today: static `--roi`) — needed for a Pi/Jetson camera.
6. **ChessCam v2 as the fast/portable recognizer** (45 ms CPU, emits square centres) — candidate for the
   Pi/Jetson and the on-phone idea; one more file behind the recognizer seam.
7. **Turn on the prior's back-rank constraint** — `prior.decode` is never given the frame, though the
   chess frame is now always known; pass `quarter_turns=0` and gate it.
8. **Idea, parked:** fully on-phone portable app — [docs/on-phone-app.md](docs/on-phone-app.md).

## Open questions

- **Where do positions go?** No downstream consumer yet (deferred 2026-08-28); the on-phone idea makes
  the "sink" question concrete (Lichess broadcast/PGN, a WebSocket endpoint, local file).
- **Target hardware latency** — ChessQ fp32 needs a GPU for ~0.2 s; Jetson/Pi unmeasured.
- **Confidence is not a hand signal** — ChessQ reads squares under a hand as empty at p ≥ 0.96; only
  burst-end scheduling + the debounce absorb it. Enough on more varied takes?
- **Licence** — ChessQ weights are PolyForm Noncommercial (accepted 2026-11-24); a commercial use
  would need another model.
