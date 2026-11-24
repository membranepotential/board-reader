---
aliases: [on-phone app, portable app, on-device recognition]
tags: [board-reader/ideas, board-reader/recognition]
---

# Idea — a fully on-phone, portable app

_Parked, not planned (user idea, 2026-11-24). When it is picked up it gets a plan in
`docs/plans/` and a measurement like any other change._

## The idea

Run recognition in the phone's browser, so **the phone is the whole system**: camera,
edge, recognizer and debounce in one page, with no laptop and no inference node. Today
the phone is a dumb camera (DESIGN D11) and the laptop does every read (D10).

## Why it looks feasible

- **ChessQ Lite int8 targets the browser.** Its model card names "ONNX Runtime,
  including browser WASM", and Lichess's "From image" runs ChessQ Lite client-side with
  `onnxruntime-web` ([[2026-10-09-chessqueries-handoff]]). Our V6 int8 file is 36 MB.
- **ChessCam already does it.** Its own ChessQueries-architecture model (MobileNetV4-M
  encoder, 320 px, a 24 MB `.tflite` for LiteRT) runs in the browser at ~100 ms per
  frame on a Pixel 7a CPU ([[2026-11-24-chesscam-camerachess]]). In our cold probe it
  read at 45 ms per frame on the laptop CPU, against ~1 s for Lite V6, at 98.3 % vs
  99.2 % per square on gate-take frames (probe results in [[2026-10-09-chessqueries-handoff]]).
- **What we would port is small.** The recognizer contract is one function (an image
  in, a 64×13 distribution in the chess frame out), with a squash resize and ImageNet
  normalisation in front of it.

## What carries over

| Part | Today | On the phone |
|---|---|---|
| Recognizer contract | `Recognizer.read` → `(8, 8, 13)`, chess frame (`recognizer.py`) | the same contract in TypeScript, over `onnxruntime-web` or LiteRT |
| Capture, crop, motion, forwarding | already TypeScript in `/camera` (`lib/edge/`) | unchanged; "forwarding" becomes "hand to the local reader" |
| Crop and lens setup | `/control` on the laptop, over the camera control channel | the same config; it could be set on the phone itself, or still from a laptop |
| Read scheduling (`schedule.py`), prior repair (`prior.py`), settle debounce (`stabilize.py`) | Python on the node | need a TypeScript port, tested against the Python on the same recordings |
| Position events | `board_reader_core.protocol` | the same JSON shape (§5), so any consumer keeps working |

## Open questions

- **The sink: where do positions go?** Candidates: a Lichess broadcast or a PGN upload
  (needs move reconstruction, which is application-layer work, DESIGN D1); a WebSocket
  endpoint such as our node's hub; a local file or the phone's share sheet.
- **Latency and heat** on a mid-range phone over a whole game, not one frame.
- **Which model.** Lite V6 int8 is the more accurate reader; ChessCam's MobileNetV4 model
  is much faster. Neither has been run on a phone by us.
- **Licence.** Unchanged: the weights are PolyForm Noncommercial, so no commercial use.

## First experiment

Benchmark ChessQ Lite V6 int8 (`onnxruntime-web`) and ChessCam (LiteRT) in mobile
Safari and Chrome on the user's phone, on our probe stills (the 21 in `data/eval/`):
milliseconds per read (cold and warm), per-square and perfect-board accuracy against
the laptop's reads of the same stills, and whether the phone throttles over ~10 minutes
of reads.

## Related

- [[ideas]] · [[architecture]] · [[recognition-pipeline]] · [[DESIGN]]
