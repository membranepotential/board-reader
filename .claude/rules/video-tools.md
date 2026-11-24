---
paths:
  - 'tools/video/**'
  - '.claude/skills/get-frames-from-yt/**'
---

# Broadcast capture tools (tools/video)

- Crop a broadcast frame by hand first and run `boarddetect.detect_board_quad` second, never on the full frame, because on full 720p/1080p frames it failed outright on 3 of 6 and once returned the broadcast's 2D overlay board; a quad with aspect ~1.000 and no perspective on a broadcast frame is that false positive.
- Angle-gate a stream (median `quad_aspect` over ~10 frames, `check_angle.py`) before extracting a batch of frames, because most table-level broadcast cameras sit outside the operating envelope (4 of 5 streams rejected), and one fixed camera gives near-identical detections, so the median is a reliable per-stream number.
- Keep `streams.py` validating every crop row against the frame size at load time, because ffmpeg silently clamps an out-of-range crop, shifts every extracted frame, and still exits 0.
- Run `detect_board_quad` several times when you evaluate detection, because `cv2.findChessboardCornersSBWithMeta` is nondeterministic even single-threaded.
- Import `boarddetect` in `tools/video/tests` as a sibling module (`sys.path.insert(0, parents[1])`), because it is a standalone tool now, not part of `board_reader_core`.
