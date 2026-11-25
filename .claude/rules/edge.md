---
paths:
  - 'packages/edge/**'
  - 'packages/viewer/src/lib/edge/motion.ts'
  - 'packages/viewer/src/lib/edge/motion.test.ts'
---

# The edge: sources, crop, motion, forwarding

- Make every frame source lazy (O(1) memory), like `RepeatSource`, never a pre-built list of frames, because `feed --synthetic N` once built N full images up front and the container hung without a log at large N.
- Crop to the static `--roi` (`core.control.CropRect`) in `FileEdgeAdapter` before `AdaptiveForwarder.decide`, because motion and stability must be measured inside the rectangle, so a hand outside it opens no burst. Keep `--roi` in `_REPROCESSING_FLAGS`, so `--recording` refuses it.
- Change the motion meter in both places at once, `packages/edge/src/board_reader_edge/motion.py` and the phone's `packages/viewer/src/lib/edge/motion.ts`, because they mirror each other and the node treats their stability the same.
- Tune a stability constant by replaying recorded frames through the metric offline, never by feel, and when a stability-gated feature "never happens", dump the recorded frames' stability distribution first, because a threshold tuned on one camera silently parked the deployment phone below every gate for weeks; the meter now learns each rig's noise floor, and gains are ratios to that floor.
- Build the phone's motion thumbnail by drawing the crop oversampled (`MOTION_OVERSAMPLE`) and block-averaging it in `sampleThumbnail`, never by `drawImage` straight to the 32×32 size, because a canvas downscale point-samples and keeps sensor noise (~8× an area average): the learned floor then sits at the noise and a hand never opens a burst (live take 2026-11-24: stability pinned at 1.0, moves took 3–6 s). Diagnose a blind meter from a take's startup ramp by fitting the floor recursion to get the resting difference. Rebase, never reset, the meter on a crop or camera change.
- Keep the forwarder tail (`tail_s`, both forwarders) at least as long as the node's burst-end read run (`(burst_end_reads − 1) × spacing_s`), because those reads need fresh frames: without a tail, each read after a burst waited for the next 1 fps rest frame (~1 s each on the 2026-11-24 live take).
- Put the camera's device id on every frame `board-reader-edge camera` sends (`Frame.device_id`, set by `CameraPipeline`), because the node's `/ingest` owner guard takes a camera's board id only from that camera: the first live run without it had all 177 frames refused (`not_owned`).
- Sample a stream at the phone's `CAPTURE_FPS` (20, `camera/stream.py`) before the motion meter, never at the source's 30 or 60 fps, because the meter's floor and gains and the forwarder's hysteresis were tuned on phone takes at that frame spacing.
- Keep everything that blocks (OpenCV opens and reads, with 15 s timeouts, and the VOD's real-time pacing sleeps) on the camera thread in `camera/run.py`, and hand frames to the event loop with `call_soon_threadsafe`, because on the loop they would stall the control socket and `/ingest` for as long as a read hangs. The loop replaces `Settings` whole; the thread reads it once per frame.
- Choose a stream's yt-dlp format by lines, not frame rate (`format_selector`), because preferring 30 fps picked 480p on a 1080p60 live stream, and the 60 fps decode keeps up anyway (about 95 fps measured).
- Keep yt-dlp on its own seasoning date (`exclude-newer-package` in the root `pyproject.toml`) and depend on `yt-dlp` plus `yt-dlp-ejs`, never `yt-dlp[default]`, because YouTube breaks old releases within weeks (2026.3.17, the newest under the workspace date, got 403 from every media URL it resolved), and `[default]` pulls a certifi newer than the workspace date. yt-dlp also needs deno on the system for YouTube's JS challenges.
