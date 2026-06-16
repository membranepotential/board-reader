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

Note: the simple auto board-detector (`find_board_quad`) does **not** localise these
real cluttered boards — they need the operator corner-tap (`rectify --corners …`).
Robust real-photo detection (Hough-line clustering, WP-A1) is later work.
