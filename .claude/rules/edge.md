---
paths:
  - 'packages/edge/**'
  - 'packages/viewer/ui/src/lib/edge/motion.ts'
  - 'packages/viewer/ui/src/lib/edge/motion.test.ts'
---

# The edge: sources, crop, motion, forwarding

- Make every frame source lazy (O(1) memory), like `RepeatSource`, never a pre-built list of frames, because `feed --synthetic N` once built N full images up front and the container hung without a log at large N.
- Crop to the static `--roi` (`core.control.CropRect`) in `FileEdgeAdapter` before `AdaptiveForwarder.decide`, because motion and stability must be measured inside the rectangle, so a hand outside it opens no burst. Keep `--roi` in `_REPROCESSING_FLAGS`, so `--recording` refuses it.
- Change the motion meter in both places at once, `packages/edge/src/board_reader_edge/motion.py` and the phone's `packages/viewer/ui/src/lib/edge/motion.ts`, because they mirror each other and the node treats their stability the same.
- Tune a stability constant by replaying recorded frames through the metric offline, never by feel, and when a stability-gated feature "never happens", dump the recorded frames' stability distribution first, because a threshold tuned on one camera silently parked the deployment phone below every gate for weeks; the meter now learns each rig's noise floor, and gains are ratios to that floor.
- Build the phone's motion thumbnail by drawing the crop oversampled (`MOTION_OVERSAMPLE`) and block-averaging it in `sampleThumbnail`, never by `drawImage` straight to the 32×32 size, because a canvas downscale point-samples and keeps sensor noise (~8× an area average): the learned floor then sits at the noise and a hand never opens a burst (live take 2026-11-24: stability pinned at 1.0, moves took 3–6 s). Diagnose a blind meter from a take's startup ramp by fitting the floor recursion to get the resting difference. Rebase, never reset, the meter on a crop or camera change.
