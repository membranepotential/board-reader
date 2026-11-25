---
paths:
  - 'packages/core/src/board_reader_core/recording.py'
  - 'packages/core/src/board_reader_core/wire.py'
  - 'packages/core/src/board_reader_core/protocol.py'
  - 'packages/core/tests/test_recording.py'
  - 'packages/core/tests/test_wire.py'
  - 'packages/inference/src/board_reader_inference/server.py'
  - 'packages/inference/src/board_reader_inference/measurement/replay.py'
  - 'packages/inference/tests/test_recording_*.py'
  - 'packages/edge/tests/test_recording_replay.py'
---

# Recordings: what the node writes and replay reads

- Record ingress at the gateway's WS `/ingest` handler, before the read scheduler and its queue, because every edge (CLI feed, Pi, the phone via the viewer's relay) crosses that one socket, and a recording is the run's input: recording after the filter reproduces an easier run than happened.
- Replay the stored JPEG bytes, never decode and re-encode them, because a q90 round trip changes the pixels and the run is then no reproduction. Keep `--throttle` meaningful: it sets arrival timing, which the wall-clock backstop and the overload rule see.
- Stop a size cap (`BR_RECORD_MAX_MB`) after writing a frame, not before, and keep `max_bytes` a whole-run cap, never per session directory, because a check before the write records nothing when the cap is smaller than one JPEG, and a per-directory cap resets on every reconnect.
- Let `FrameRecorder` roll to a new session directory when a board's seq stops advancing (`<=`), and route an event to the previous directory when its `capture_ts` is older than the new session's first frame, because a read started before the roll publishes after it and seq cannot tell the sessions apart (both start at 0). Keep `replay.find_sessions` for recordings made before the roll existed.
- Write `protocol.square_evidence()` (the top-2 digest per square, ~1.8 kB per line) into `events.ndjson`, never the full `square_dist` (~25 kB per line at burst rate), and return `None` for recordings that predate the digest instead of fabricating one, because the margin between the top two is what says whether a misread is fixable.
- Keep `wire.Frame` parsing `corners` although no edge sends it any more, because old recordings carry it and replay through `Frame`.
- Read a recorded event line through `trial._boards`, never `line["placement"]` alone, because three generations of lines exist: since 2026-11-24 `placement` is the settled board (a `pending` key marks the line) and the read is the top row of `evidence`; before, `placement` was the read and the settled board sat in a `settled` block; before the node settled at all, there is neither. The gate bases (`sp03-settle05-7`, `guard-only`) and every older recording are in the older shapes.
