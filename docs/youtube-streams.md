---
aliases: [youtube streams, stream registry, broadcast footage]
tags: [board-reader/data, board-reader/ops]
---

# Youtube Streams

Real OTB broadcast footage used as test data — the harder, uncontrolled counterpart to
studio photos. Five broadcasts were absorbed from a scratch repo
(`../otb_video_recognition`, now deleted; the tooling lives on as `tools/video/`, see
`tools/README.md`) and run through the camera-elevation gate in [[operating-envelope]].
Four failed it; a sixth was then sourced deliberately to beat them. This note is the
registry of what was tried, what survived, and why.

## Broadcasts and the angle audit

Every extracted frame of each broadcast was run through
`board_reader_core.boarddetect.detect_board_quad` + `quad_aspect`, and the median
projected board aspect compared against the [[operating-envelope#Camera angle|camera
angle]] thresholds: usable at `>= 0.73`, `very_low_angle` fires below `0.62`.

| slug | video id | detections | median aspect | verdict |
|---|---|---|---|---|
| ccbf-r5 | `2sRJ4JuL_m8` | 11/16 | 0.635 | degraded — **kept, the M2 target** |
| hexamind | `JTmovXvZnPM` | 9/13 | 0.650 | degraded — kept |
| wtr-day2 | `Fed1aBzUMcs` | 9/14 | 0.573 | too shallow — video deleted |
| coffeechess | `1O2ahFZh6G8` | 3/6 | 0.508 | too shallow — video deleted |
| grandswiss | `62h7O9Ns_eA` | 2/5 | 0.450 | too shallow — video deleted |
| divya-r14 | `ZoYhhoG6dIY` | 1/8 | 0.445 | too shallow — video deleted |

**4 of 6 broadcasts failed the gate**, and neither survivor reaches the `>= 0.73` band
every good recognition result in [[operating-envelope]] was measured in — both sit in the
*degraded* 0.62–0.73 band. Table-level chess broadcast cameras are, by and large, outside
the reader's operating envelope: producers shoot from a low, wide angle for the room, not
from overhead for a per-square CV pipeline. A steeper/overhead camera remains a **sourcing
requirement** for future broadcast test data, not a nice-to-have.

> [!warning] Standing M2 risk
> Neither survivor is in the good band. M2's recorded-game demo rests on *degraded*
> footage until a steeper-angle source (or own footage) is found — so treat its
> recognition quality as a floor, not a measurement of the pipeline.

### ccbf-r5 — the sourced-on-purpose stream

The first five were whatever the scratch repo happened to hold. `2sRJ4JuL_m8` was found
by searching explicitly for a *higher* camera: ~40 candidates screened with
`yt-dlp ytsearch`, one sampled frame each, run through
`detect_board_quad`/`quad_aspect`. **Chess League | CCBF | Round 5 Board 1** (Chess Club
Bobby Fischer, 3:04:36) won — one board, big in frame, and a camera that never moves
across the full three hours: aspect **0.633–0.639 over 5 probe frames spanning the whole
video**. It does not beat hexamind on angle (0.635 vs 0.650); it beats it on everything
else — length, steadiness, resolution, and being a single continuous real game.

Seven operator-picked frames spread over the 3 h live in
`data/streams/frames/ccbf-r5_2sRJ4JuL_m8/` and re-audit at **0.637 (6/7 detected)**,
matching the whole-video gate.

> [!important] This one was downloaded at 1080p, not the skill's default 720p
> The board occupies ~1150 px inside the crop at 1080p — roughly **144 px/square**. The
> same crop at 720p yields ~430 px of board, *below* the ~500 px marginal line the
> `get-frames-from-yt` skill itself sets. When a crop is a small fraction of the frame,
> the 720p default is the wrong call; check the crop's projected pixel width before
> choosing a resolution. Cost of the exception: 3.9 GB on disk.
>
> The crop also keeps headroom above the far rank, so tall pieces are not decapitated by
> the crop itself — a separate failure mode from the pose-crop decapitation in
> [[training-and-data]].

> [!quote] The honest finding from the search
> **Nothing on YouTube is both a long real OTB game and `>= 0.73`.** Broadcast cameras
> frame players, not boards. The footage that *does* clear 0.73 is short CV-demo or
> overhead-rig clips — `kg_whG0Ijh4` at 0.790, `B_3v5sqqU0Y` at 0.743 — not games. The
> good band will have to come from our own camera.

## What survives a culled stream

Nothing is lost when a video is deleted: the video id, its crop, and any frames already
extracted are enough to reproduce it exactly with `yt-dlp` + `tools/video/crops.csv`.
Frames live at `data/streams/frames/<slug>_<video_id>/` and are kept deliberately — they
are documented out-of-envelope hard cases, useful for testing the low-angle regime
[[operating-envelope#Camera angle|synthetic training data now samples]] rather than for
production recognition.

All of `data/` (including `data/streams/videos/` and `data/streams/frames/`) is
git-ignored — multi-GB, copyrighted broadcast footage never belongs in the repo.
`tools/video/crops.csv` **is** committed, and it is what makes the whole set
reproducible from nothing but a video id.

## The crop registry

`tools/video/crops.csv` keys rows on the **11-character YouTube video id**, not the
downloaded file's name — a file gets renamed, re-downloaded at another resolution, or
deleted, while the id is stable. Crop coordinates are pixel coordinates measured against
a download *of one specific resolution* — the table also records `frame_width` /
`frame_height` because three of the six broadcasts were downloaded at 1080p and three at
720p, and a 720p crop is meaningless against a 1080p re-download of the same video. Each
row also carries the verdict from the angle audit above directly (`median_aspect`,
`status` of `keep` or `rejected_low_angle`) — the table above is this registry's own
record, not a separate measurement kept in sync by hand.

> [!example] Recovering a lost crop
> `62h7O9Ns_eA` (grandswiss) had no recorded row. It was recovered by template-matching a
> surviving extracted frame back into the source video — correlation 1.0 at two
> independent timestamps — giving `x=606, y=594, 748x396` at 1920x1080. Worth reusing as
> a precedent if another crop ever goes missing: an extracted frame is enough to
> reconstruct the crop that produced it, without redoing the manual selection.

## Detection on broadcast frames

Three findings that shaped the crop-first workflow:

- `detect_board_quad` on **full** 720p/1080p broadcast frames is unreliable: outright
  failure on 3 of 6 tested frames, and on a 4th it locked onto the broadcast's **2D
  overlay board** — a perfectly axis-aligned square quad — instead of the physical
  board. On already-cropped frames it works. The shared detector does handle real
  photos (studio shots, phone snapshots); it does not handle arbitrary, cluttered full
  broadcast frames. Hence the workflow is crop-first (manual, via
  `tools/video/crop_selector.py`), detect-second.
- Detections **within** one broadcast are near-identical, because the camera is fixed
  for the whole stream. That is why a median over a handful of frames is a trustworthy
  per-stream measurement rather than noise.
- **A no-detection frame is usually the detector's fault, not the footage's.** On
  ccbf-r5 all six frames the gate failed to detect (5 from the 16-frame video pass,
  1 from the 7-frame audit) were pulled and inspected by eye: every one is a clean,
  fully-framed board — including a textbook opening position — with no camera cut and
  nothing occluding. So `detect_board_quad` runs a **~30% false-negative rate on a crop
  whose frames are ~100% usable**. Read the detection rate as a measurement *of the
  detector*, not of stream quality: the earlier per-stream rates in the table above
  understate the footage and overstate the detector.

> [!caution] Consequence for M2
> A ~30% miss rate means the edge `Rectifier`'s cadenced re-detection will fail roughly
> one attempt in three on real footage. Run recorded-game demos with `feed --corners`
> (operator-supplied corners) rather than relying on auto-detection.
>
> Suspected cause, **unverified**: this board shows visible barrel distortion, which
> breaks the straight-edge quad model the detector assumes.

## Earlier streams (pre-audit, 2026-06)

Two streams predate the five-broadcast absorption and were validated by hand rather than
through the angle gate above.

- <https://www.youtube.com/watch?v=TDEnuPnvLaA>
- <https://www.youtube.com/watch?v=7u9zRjq3ePw>

### Reference timestamps for test frames

| Stream | Starting position | Complex position(s) | Cropping |
|---|---|---|---|
| TDEnuPnvLaA | 4:42 | 1:06:25 | **Needs cropping** — physical board is a sub-region of the full frame (layout has a 2D overlay board, chat, clocks). Crop region (1080p): `y 430:745, x 1095:1545`. |
| 7u9zRjq3ePw | 0:03 | 25:00 (and many others) | **No crop needed** per request — board fills the view; a 2D overlay board is also present but we don't crop it out. |

Naming: `<id>_<mmss>_full.png` (raw 1920×1080 frame) and `<id>_<mmss>_board.png`
(cropped to the physical board). Pulled with `yt-dlp -f 137 --download-sections` (just
the needed seconds, not the full multi-hour stream) + `ffmpeg -frames:v 1`. These frames
were not retained and would need re-pulling before reuse — the corners below were picked
by hand with `tools/corner_picker.html` (load the frame, click the 4 board corners), back
when the naive Canny/contour detector in use at the time could not localise these
cluttered broadcast frames.

#### Operator corners used for validation (pixel coords, any order)

| Frame | Corners `x0,y0,...` |
|---|---|
| `TDEnuPnvLaA_0442_board.png` | `10,232,351,234,310,112,58,110` |
| `7u9zRjq3ePw_0003_full.png` | `687,943,1249,934,1171,720,751,727` (mild fish-eye) |

### Real-footage validation findings (2026-06-17)

Rectification works on both streams with operator corners. Recognition (rectify → crop
→ DINOv2 centroid) **within-frame self-consistency**: video1 start 92% piece-type /
100% occupancy / 98% colour; video2 start 98% / 100% / 100%. **Cross-frame
generalisation is poor**: these are very low/side broadcast angles, so tall pieces lean
nearly horizontal and smear across neighbouring squares, breaking the per-square
assumption ([[operating-envelope|DESIGN's documented top risk]]). This is the same
failure mode the five-broadcast audit above confirms at scale, with numbers: low
broadcast angles are the norm for table-level chess coverage, not the exception.

## Related

- [[operating-envelope]] — the camera-angle thresholds this audit is measured against
- `tools/README.md` — the `tools/video/` toolset and its two-interpreter split
- `tools/corner_picker.html` — manual corner-tap tool for validating rectification by hand
