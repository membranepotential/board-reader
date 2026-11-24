# tools/

One-off developer tooling that is **not** part of any deployable package.

## `tools/video/` — YouTube broadcast footage as test data

Downloads a chess broadcast, lets an operator drag a board crop, and extracts cropped
still frames for training/eval. Absorbed from a scratch repo
(`../otb_video_recognition`) that predates the workspace. Four scripts plus a committed
registry:

- `crop_selector.py` — opens a still frame in an OpenCV window; the operator drags a
  rectangle, records it to `crops.csv`. Never re-encodes the video — every other tool
  crops on the fly from that row.
- `check_angle.py` — runs `board_reader_core.boarddetect.detect_board_quad` +
  `quad_aspect` over a video or a frame directory and reports the median projected
  board aspect, the same camera-elevation gate the server warns on.
- `extract_frames.py` — reads the crop from `crops.csv`, writes cropped PNGs at
  explicit or random timestamps.
- `watch_cropped.py` — plays the video with the crop applied live; saves a frame on
  keypress. The interactive alternative to `extract_frames.py`.
- `crops.csv` — the committed crop registry, keyed on the 11-character YouTube video
  id (not the downloaded file's name, which changes with resolution or a re-download).
  Also carries each stream's `check_angle.py` verdict (`median_aspect`, `status`). This
  is what makes any extracted frame set reproducible from nothing but a video id — see
  `tools/video/README.md` for the full CLI and the crop table format, and
  [docs/youtube-streams.md](../docs/youtube-streams.md) for the angle audit and a
  worked recovery of a lost crop row.

**Two interpreters, deliberately.** `crop_selector.py` and `watch_cropped.py` open an
OpenCV/mpv window and need the **system** `python3` — the uv workspace ships
`opencv-python-headless`, and `cv2.imshow` is a no-op there. `check_angle.py` needs
`uv run` instead, because it imports `board_reader_core`. Reaching for the wrong
interpreter fails loudly (`ModuleNotFoundError` either way), not silently.

**The angle gate rejects most broadcast footage.** Running `check_angle.py` over the six
broadcasts collected so far, gating on the same threshold the server warns at
(`quad_aspect >= 0.62`, [docs/operating-envelope.md](../docs/operating-envelope.md)),
rejected 4 of 6 — table-level chess broadcasts are shot too shallow for a per-square CV
pipeline, and the two survivors (`hexamind` 0.650, `ccbf-r5` 0.635) are both merely
*degraded*, never the `>= 0.73` good band. Searching YouTube specifically for a steeper
camera did not find one: long real OTB games are never shot from overhead. See
[docs/youtube-streams.md](../docs/youtube-streams.md) for the measured aspect per
broadcast and the false-negative caveat on the detection rates.

These are third-party broadcast recordings. Downloaded videos and extracted frames
live under `data/streams/` (git-ignored), kept for local/personal research use only —
the same footing as `data/external/chessred`. The end-to-end operator workflow (when to
ask the user for a crop, which frame-extraction mode to use) is documented in the
`get-frames-from-yt` Claude Code skill (`.claude/skills/get-frames-from-yt/`).

## `fetch_models.py` + `chessq_export/` — `make models`

`fetch_models.py` downloads every release descriptor's file under `models/chessq/`
(pinned Hugging Face revision, size, SHA-256): the int8 ONNX served on a CPU and the
FP16 safetensors. `chessq_export/` is its own uv project (PyTorch + the official
`chessqueries` package at a pinned commit; **not** a workspace member, so nothing
that serves depends on torch). Its `export.py` turns the safetensors into the fp32
ONNX served on a GPU, checks per-square argmax against the package's own prediction
(drawn boards by default, `--samples` adds photos), and writes the export's
descriptor through `descriptor.py` (stdlib only, tested by
`tests/test_chessq_export.py`). See the [runbook](../docs/runbook.md#running-it).

## `gate_table.py` — the four-take replay gate as one table

Reads `recordings/<take>/replays/<name>/report.json` for two replay names on the
four gate takes and prints `base→candidate` per take plus a pooled total. `make gate
GATE_NAME=<new>` replays the takes through `BR_CHESSQ` and prints it against
`cq-v6`. See [measuring accuracy](../docs/measuring-accuracy.md).

## `migrate_labels_to_chess_frame.py` — corner-frame labels → chess frame (D10)

One-off. `migrate <dir>` turns each corner-frame `*.label.json` by the quarter-turn
ChessQ agrees with, only when a margin, a back-rank-pawn check, the truth's own
plausibility and the old autorotate all agree; anything else is listed and left.
`mark <files|dirs>` adds `frame: "chess"` to truth that already is. Dry run unless
`--apply`; originals go to `.pre-chess-frame/` beside each file.
