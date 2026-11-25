# STATUS — board-reader

_reconciled: 2026-11-24 @ `480a5ea`_ · decisions: [docs/DESIGN.md](docs/DESIGN.md) · history: `git log`, [docs/reports/](docs/reports/)

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those belong to the downstream application layer.

## Now (4 agents, files partitioned)

- **camera identity** — device id on the phone → auto board id (no `phone` default), /control overview +
  persistent rename, duplicate guard — running (agent A; control.py, camera.py, /camera, /control).
- **QR code** — viewer route with external URLs (never localhost) + `CameraQr` component — running (agent B).
- **multi-board + spikes** — load test: results and timestamps independent of load; fix what differs —
  running (agent C; stream/, /ingest handler).
- **auto-crop probe** — ChessQ attention grid vs classical line detection (+ both) → proposed crop; board
  83–15 % of the frame; where each runs (phone/Pi/node) — running (measure only; tools/probes/autocrop_*).

## Done (2026-11-24/25 — the ChessQ rebuild; plan: [docs/plans/2026-11-24-simplify-around-chessq.md](docs/plans/2026-11-24-simplify-around-chessq.md))

- **ChessQ is the core recognizer** (Lite V6, fp32 CUDA ~0.2 s/read; int8 CPU); old stack deleted (≈ −26k
  lines). Gate: 7 takes, base `latency3` 374/405 raw, 350/405 settled, 0 hand leaks; take 2 has truth.
- **Latency:** area-averaged phone meter, 1 s tail; node hold 0.3 s, settle 0.3 s (`8eac6f6`, needs the
  energy gate); take 2 median move→settle 1.66 → 0.94 s.
- **Energy gate** (D16, `138d941`): obscured reads never commit; resting hand 9 wrong boards → 0.
- **Event** (D15): one settled placement + `pending` + `changed_utc` (pixel back-dated, `9fb7671`: take 2
  0.57 s after the true change); priors → python-chess count guard + `rules: standard|free`.
- **Dumb-camera phone** (D11) + `/control` over WebSockets; **viewer** (D12) SvelteKit on Node; **one wire
  contract** (D13) pydantic → generated zod TS; **PGN drawer** (`81408ac`, `b58968a`): 7/7 games exact.
- **Docs + rules:** DESIGN rewritten; learnings → `.claude/rules/`; inference regrouped; big data on the HDD
  (`/run/media/felix/archive/board-reader/`, symlinked).

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
