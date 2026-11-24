# ChessCam and CameraChess: two systems built on ChessQ (2026-11-24)

The ChessQueries author (Joël Seytre) pointed us to these two projects; both are listed as
integrations in the [ChessQueries README](https://github.com/JSeytre/chessqueries).
Desk research only: nothing was run. Companion to
[2026-10-09-chessqueries-handoff.md](2026-10-09-chessqueries-handoff.md).

## Summary

| | ChessCam | CameraChess |
|---|---|---|
| Repo | [Pbatch/ChessCam](https://github.com/Pbatch/ChessCam) (app), [Pbatch/ChessCamRecorder](https://github.com/Pbatch/ChessCamRecorder) (engine, npm `@chesscam/recorder`) | [AmethystineAlpaca/camera_chess](https://github.com/AmethystineAlpaca/camera_chess) |
| Activity | 270 stars, 111 commits since 2026-08, last push 2026-11-22 | 9 stars, 20 commits 2026-10-09…11, quiet since |
| Purpose | Record a game to PGN in the browser, play Lichess on a real board | Play Stockfish on a physical board, macOS only |
| Recognizer | Own ChessQueries-architecture model on a MobileNetV4-M encoder, 320 px, 24 MB, CPU/WASM, ~100 ms/frame on a Pixel 7a ([model card](https://huggingface.co/PBatch23888/chessq-chesscam)) | ChessQ Lite V4 (ViT-S, 644 px), frozen, PyTorch on MPS, ~387 ms/frame |
| Board finding | The model also outputs each square's image position; a robust homography fitted to those gives the corners, then a 15 % padded crop | OpenCV line/contour proposals + checkerboard score, LK optical-flow tracking, rectify to 768 px |
| Temporal layer | Bayesian belief over "no move / legal move / move + reply", plus a beam of 4 histories that can take back a played move | Previous-state "CRF" scorer + 3-of-4 vote gate |
| Real-video accuracy | 39/43 held-out blitz games perfect (91 %), 2.0 corrections/game; ChessQ Lite V5 with the same tracker: 30/43, 9.5 corrections/game | Not measured on video |
| License | App AGPL-3.0, engine MIT, weights PolyForm Noncommercial | No license file; ChessQ weights PolyForm NC |

## The CRF (CameraChess)

It is not a spatial CRF over squares. The repo's own log calls it "a single-step
conditional log-linear model, described here as a CRF postprocessor with its previous
state fixed" (`docs/crf-experiment-log.md`).

- Labels are whole boards: the unchanged previous board, every legal move from it
  (python-chess), and simple "supported mistakes" (a piece moved to an empty square).
- Each candidate's score is a weighted sum over the changed squares of features of
  ChessQ's 64×13 probabilities (log-prob, grouped empty/white/black, argmax match),
  plus weights on move flags (legal, capture, castling, promotion…). v1 has 25 weights;
  v2 has 250.
- Inference enumerates the candidates and takes a softmax. Training uses conditional NLL on
  200k *synthetic* probability arrays from 20k Lichess-PGN transitions; no real images were used.
- Result on 176 still photos, each paired with a synthetic previous position: raw argmax 1/176
  exact boards, CRF v1 125/176, default changed-square MLP 128/176. The whole gain comes
  from knowing the previous position.

## What this means for board-reader

- **Neither temporal layer fits the CV layer as-is.** Both reason about moves and
  legality, which DESIGN D1 assigns to the downstream application layer. They are
  reference designs for that layer, not for the estimator.
- **ChessCam's recognizer is the relevant new data point.** It is an end-to-end
  ChessQueries-style model that also outputs square positions (corner finding for
  free), runs on a phone CPU, and beats ChessQ Lite V5 on real video. That makes it a
  second candidate next to ChessQ Lite for the still-unrun `chessq-lite-probe` benchmark.
  Its test set is the author's own blitz-video domain; our oblique angles are untested.
- **"Square position + sigma" output** could replace our corner detector and
  rectification stage, if it holds on our footage.
- **Licensing**: all usable weights are noncommercial (same caveat as the handoff).

## Next

1. Run `chessq-lite-probe` (handoff step 1), adding the ChessCam `.tflite` model as a
   second candidate (needs a LiteRT/TFLite runtime, or a conversion to ONNX).
