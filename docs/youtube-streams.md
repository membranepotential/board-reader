# Youtube Streams

Links to streams of chess games for test data. Whether a frame needs cropping to the
board depends on the stream's layout (see per-stream notes below, and the edge
rectifier, WP-A).

- <https://www.youtube.com/watch?v=TDEnuPnvLaA>
- <https://www.youtube.com/watch?v=7u9zRjq3ePw>

## Reference timestamps for test frames

| Stream | Starting position | Complex position(s) | Cropping |
|---|---|---|---|
| TDEnuPnvLaA | 4:42 | 1:06:25 | **Needs cropping** — physical board is a sub-region of the full frame (layout has a 2D overlay board, chat, clocks). Crop region (1080p): `y 430:745, x 1095:1545`. |
| 7u9zRjq3ePw | 0:03 | 25:00 (and many others) | **No crop needed** per request — board fills the view; a 2D overlay board is also present but we don't crop it out. |

Extracted frames live under `tests/data/` (git-ignored). Naming: `<id>_<mmss>_full.png`
(raw 1920×1080 frame) and `<id>_<mmss>_board.png` (cropped to the physical board).
Pulled with `yt-dlp -f 137 --download-sections` (just the needed seconds, not the full
multi-hour stream) + `ffmpeg -frames:v 1`.

Note: the corners below were picked by hand because the naive Canny/contour detector
in use at the time could not localise these cluttered broadcast frames. That detector
is gone — the edge now shares `board_reader_core.boarddetect.detect_board_quad` with
the snapshot path, which does handle real photos. **Untested on these frames**: they
were not retained, so re-pull a few and check before assuming the corner-tap is still
needed here. Pick corners with `tools/corner_picker.html` (load the frame, click the
4 board corners).

### Operator corners used for validation (pixel coords, any order)

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
assumption (DESIGN's documented top risk). Next: a labelled eval set (WP-B4), a higher
camera angle, and/or multi-frame burst voting.
