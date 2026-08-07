# tools/video — broadcast streams → cropped training frames

Pull a YouTube chess broadcast, mark where the board is, check the camera is
high enough to be usable, and extract cropped stills. Videos and frames live
under `data/streams/` (git-ignored); only `crops.csv` is committed.

| Script | What it does |
|---|---|
| `streams.py` | Typed registry over `crops.csv`. Library only, no CLI. |
| `crop_selector.py` | **Manual** crop selection — drag a box on one preview frame. Always the first step. |
| `check_angle.py` | QC gate: measures camera elevation, exits non-zero on a stream we should not use. |
| `extract_frames.py` | Writes cropped PNGs at random or explicit timestamps. |
| `watch_cropped.py` | Plays the video cropped in mpv; `s` saves an already-cropped screenshot. |

```bash
python3    tools/video/crop_selector.py  "…[VIDEOID].mp4"          # c=confirm r=reset q=cancel
uv run python tools/video/check_angle.py "…[VIDEOID].mp4"          # gate; non-zero = reject
uv run python tools/video/extract_frames.py "…[VIDEOID].mp4" -n 10 # -> data/streams/frames/<slug>_<id>/
python3    tools/video/watch_cropped.py  "…[VIDEOID].mp4" --start 01:42:46
```

## Two interpreters, on purpose

`crop_selector.py` and `watch_cropped.py` run on the **system `python3`**;
`check_angle.py` runs under **`uv run python`**. This trips everybody up once:

* the uv workspace has `opencv-python-headless`, where `cv2.imshow` silently
  does nothing — so the interactive selector cannot run there. Arch's
  `python-opencv` (pacman) is Qt-enabled and works;
* the system interpreter has no `board_reader_core` — so the angle check, which
  needs `detect_board_quad`, cannot run there.

Hence: the hand-operated tools must never import anything from the workspace,
and `streams.py` stays **stdlib-only** so both halves can share it.
`extract_frames.py` is stdlib + ffmpeg subprocesses, so it runs under either.

## The crop is resolution-specific

`crops.csv` stores the crop in pixels **of a download at that exact
resolution**, which is why each row also records the `frame_width`/
`frame_height` it was picked at. Two rows were selected on 1080p downloads and
three on 720p ones. Re-download a video at another resolution and you must
re-run `crop_selector.py`; `load_streams` refuses any crop that does not fit
inside its recorded frame.

That check matters because **ffmpeg silently clamps an out-of-range crop**
(`y = min(y, ih - h)`) instead of failing — so a bad row does not blow up, it
quietly shifts every extracted frame. The `1O2ahFZh6G8` row was corrected to the
clamped `y` ffmpeg actually applied, which is what its frames on disk contain.

Rows are keyed on the **11-char YouTube id**, parsed out of the `[<id>]` yt-dlp
puts in the filename — renaming or re-downloading a file must not orphan its
crop. Re-recording a crop appends a row; the last one for an id wins.

## Four of the five streams are rejected

`status=rejected_low_angle` means the camera sits too close to board level: below
`board_reader_core.boarddetect.LOW_ANGLE_ASPECT` (0.62) pieces occlude whole
ranks and recognition collapses (see `docs/operating-envelope.md`). Only
`hexamind` survives, and only just — 0.65, inside the degraded band, not the
good one (≥ 0.73). `extract_frames.py` warns loudly rather than refusing, so a
rejected stream can still be pulled deliberately.

Broadcast camera angle is therefore the binding constraint on this data source,
not availability. Check the angle *before* investing in a download.

`check_angle.py` samples a video through its registered crop rather than the
full frame: measured 2026-08-07, full-frame detection failed on 3 of 6 videos
and latched onto a 2D *overlay* board on a 4th, while the cropped frames
detected fine. That is why the manual crop comes first and the automatic check
second. A partial detection rate is normal — replays, cuts and graphics have no
board in them.

The `62h7O9Ns_eA` crop was not hand-picked: its row went missing, and was
recovered by template-matching a surviving extracted frame back into the source
video (correlation 1.0 at two independent timestamps).

## Tests

`tools/video/tests/` runs in the normal workspace suite (`uv run pytest`). It
covers the registry, the timestamp/sampling helpers and the verdict logic with
synthetic inputs only — never the detector, never anything under `data/`.
