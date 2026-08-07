---
name: get-frames-from-yt
description: Download a YouTube chess broadcast at reduced resolution, have the user pick the board crop region interactively, gate it on camera angle, and extract cropped still frames as training/eval data. Use whenever the user wants frames, a crop, or a new video pulled into this dataset — e.g. "download <url>, query me for a crop, then query me for frames", "add this broadcast to the frames", "grab board crops from this stream".
---

# Get frames from a YouTube broadcast

Four steps, in order: **download → crop → angle check → frames**. Steps 2 and 4
need the user in the loop — do not guess a crop region or a frame count. Step 3
is automated but mandatory: do not skip straight from crop to frames.

The repo's scripts do the work, all under `tools/video/`:

| Script | Role |
|---|---|
| `crop_selector.py` | Opens a still frame in an OpenCV window; user drags a rectangle. Records the region to `crops.csv` — nothing is re-encoded. |
| `check_angle.py` | Audits camera elevation against `docs/operating-envelope.md`'s projected-board-aspect thresholds. The gate between crop and frames. |
| `watch_cropped.py` | Plays the video in mpv with the crop applied live; `s` saves an already-cropped frame. The interactive way to pick frames. |
| `extract_frames.py` | Reads the crop from `crops.csv`, writes cropped PNGs at random or explicit timestamps. The batch way. |

Data lives under `data/streams/`: downloaded `.mp4`s in `data/streams/videos/`,
extracted frames in `data/streams/frames/<slug>_<video_id>/`. The whole `data/`
tree is **git-ignored** — these are multi-GB copyrighted broadcast files and must
never be committed. `tools/video/crops.csv` **is** committed: it is the registry
that makes every frame set reproducible from a video id alone, keyed on the
11-char YouTube video id (not the mp4 basename), with columns `video_id, slug,
title, url, x, y, width, height, frame_width, frame_height, preview_at_s,
median_aspect, status, recorded_at`.

**Crop coordinates are pixel coordinates in a download of that exact
resolution.** `frame_width`/`frame_height` record which, and the loader
enforces it — `extract_frames.py`/`watch_cropped.py` reject a crop that
doesn't fit the video they're pointed at, rather than silently misbehaving.
That check earns its keep: **ffmpeg itself silently clamps** an out-of-range
crop instead of failing, so a stale row shifts every extracted frame with no
error at all — it already happened once in this dataset (a recorded `y` sat
outside a 720-tall frame; the row was corrected to the value ffmpeg had
actually applied). If you re-download at a different resolution, re-run
`crop_selector.py` — don't hand-edit the old row.

## Python interpreter — split by GUI need

The uv workspace's OpenCV is `opencv-python-headless`, so `cv2.imshow` does not
work under `uv run`. System `python3` has Qt-enabled OpenCV (pacman
`python-opencv`) but no `board_reader_core`. Match the tool to the interpreter
that has what it needs — do not default to one for everything:

| Script | Interpreter | Why |
|---|---|---|
| `crop_selector.py` | system `python3` | opens a GUI window (`cv2.imshow`) |
| `watch_cropped.py` | system `python3` | opens an mpv window |
| `check_angle.py` | `uv run python` | needs `board_reader_core.boarddetect` |
| `extract_frames.py` | either | pure stdlib + `ffmpeg`/`ffprobe`; `uv run` for consistency |

```bash
python3 tools/video/crop_selector.py ...            # correct — GUI script
uv run python tools/video/check_angle.py ...         # correct — needs core
.venv/bin/python tools/video/crop_selector.py ...    # fails: no Qt cv2
python3 tools/video/check_angle.py ...                # fails: no board_reader_core
```

## 1. Download

Check the format table first — long broadcasts are big, and the highest offered
resolution is usually 1080p:

```bash
URL="https://www.youtube.com/watch?v=<id>"
yt-dlp -F "$URL"
yt-dlp -O "%(title)s | %(duration>%H:%M:%S)s" "$URL"   # duration sanity-check
```

Download **720p H.264**, into `data/streams/videos/`:

```bash
yt-dlp -t mp4 -S res:720 -o 'data/streams/videos/%(title)s [%(id)s].%(ext)s' "$URL"
```

Keep yt-dlp's default `<title> [<id>].ext` naming — every tool below resolves
the crop row through the `[<id>]` in the basename, so don't rename the file
after downloading it.

Why 720p: the board typically occupies a bit over half the frame width, so a
1080p-relative crop of ~1100px lands near ~740px at 720p — still comfortably
above what the detector needs, at roughly half the bytes. Go lower only if the
user asks; at 480p the same crop drops to ~500px, which is marginal. Prefer
H.264 (`avc1`) over AV1/VP9 — seeking is faster and OpenCV handles it without
surprises.

These files are multi-GB. Run the download with `run_in_background: true`, and
**do not pipe it through `tail`/`head`** — that buffers all output until the
process exits, so you lose progress visibility. Redirect to a log instead, and
watch the growing `.part` file with `du -h data/streams/videos/*<id>*.part` if
you want a size read.

Filenames contain `｜` (U+FF5C fullwidth vertical line, yt-dlp's substitute for
`|`) and other non-ASCII. Always quote them.

## 2. Query the user for a crop

Ask **which timestamp** the selection frame should come from before opening the
window — `t=0` is usually a title card or a wide hall shot, not the board. An
hour in is a good default for a multi-hour broadcast. Present it as a real
choice; if the frame turns out to be a camera cut the user presses `q` and you
reopen at another timestamp, which is cheap.

```bash
python3 tools/video/crop_selector.py --at 1:00:00 "data/streams/videos/<video> [<id>].mp4"
```

- Recording the crop region is the script's only behaviour now — it no longer
  re-encodes a cropped copy of the video and takes no output-file argument, so
  there is no `--select-only` flag (nor `--fps`/`--start`/`--end`, which went
  with the re-encode path). `--at` and `--table` are the only flags.
- `--at` accepts `HH:MM:SS`, `MM:SS`, or plain seconds.
- `--table` points at the crop registry; leave it unset to use the default
  (`tools/video/crops.csv`).
- In the window: drag to select, `c` to confirm, `r` to reset, `q` to cancel.

The window opens on the user's X display. Run it in the background so it does
not block you, or hand the user the command to run themselves with the `!`
prefix. Either way the drag is theirs to make — you cannot do it.

**There is no metadata handoff from the download step** — the script derives
everything from the filename it's given: `video_id` is the last `[...]` group
in the basename (broadcast titles have their own brackets, so it takes the
last one), `title` is the rest with yt-dlp's non-ASCII substitutions reversed,
`url` is built from `video_id`. If the filename has no `[id]`, it errors —
re-download with the default template rather than renaming by hand. `slug` is
auto-derived from the title (lowercased, hyphenated words appended up to ~24
chars) — treat it as a first draft: **shorten it in `crops.csv` before
extracting frames**, since it names the frame output directory (the committed
rows `hexamind`, `wtr-day2` were hand-shortened this way). New rows land with
`status=keep` and an empty `median_aspect` — nothing writes those back
automatically; after the angle check in step 3, set `status=rejected_low_angle`
yourself if the verdict says so.

Confirm afterwards that a row landed in `tools/video/crops.csv`; `q` exits
without recording.

**Crop selection stays manual.** The obvious shortcut — auto-detect the crop
with the repo's own `detect_board_quad` on the full broadcast frame — does not
work: tried against real broadcast frames, it failed outright on half of them
and on another it confidently returned the broadcast's flat 2D overlay board
instead of the physical one. Cropping first and detecting second (step 3) is
reliable; detecting first is not. The operator drags the crop; the detector
only ever audits it afterward.

## 3. Angle-check the crop — mandatory gate before extracting frames

Recognition collapses when the camera is near board level. The metric is the
**projected board aspect** — short axis over long axis of the board's corner
quad, via `board_reader_core.boarddetect.quad_aspect` over quads from
`detect_board_quad` — and `docs/operating-envelope.md` is the authority on the
bands:

| Aspect | Verdict | Exit code |
|---|---|---|
| `>= 0.73` | good — the band every good recognition result was measured in | 0 |
| `0.62 – 0.73` | degraded — usable, but expect errors | 0 |
| `< 0.62` | too_shallow — not worth keeping | 1 |
| no detection anywhere | unknown — not a pass | 2 |

Run it right after the crop is recorded, before extracting a batch of frames —
extracting 50 frames from an unusable stream is the waste this step prevents:

```bash
uv run python tools/video/check_angle.py --frames data/streams/frames/hexamind_JTmovXvZnPM   # audit an existing frame set (every image in the dir)
uv run python tools/video/check_angle.py "data/streams/videos/<video> [<id>].mp4" -n 12       # sample a video directly (12 is the default, evenly spread, first/last 2% skipped)
```

`VIDEO` and `--frames DIR` are mutually exclusive. Real output:

```
$ uv run python tools/video/check_angle.py --frames data/streams/frames/hexamind_JTmovXvZnPM
Source: data/streams/frames/hexamind_JTmovXvZnPM

  hexamind_1_t00h36m48s.png    0.650
  hexamind_2_t01h42m46s.png    0.652
  hexamind_4_t04h04m20s.png    no board detected
  ...
Detected:       9/13 (69%)
Median aspect:  0.650   (good >= 0.73, reject < 0.62)
Verdict:        degraded — usable, but the camera is low; expect errors
$ echo $?
0
```

Add `--json` when chaining this into another command; keys are `source,
frames[{label, aspect}], sampled, detected, detection_rate, median_aspect,
verdict, good_aspect, low_angle_aspect` (`verdict` ∈
`good|degraded|too_shallow|unknown`; the aspect fields are `null` when nothing
was detected). It exits non-zero on `too_shallow` (1) *and* on `unknown` (2) —
unknown is not a pass, treat it the same as a failure and investigate (usually
a bad crop or an off-board timestamp) rather than proceeding. Detection is
mildly nondeterministic across runs — don't quote an exact rate as a
guarantee, go by the verdict.

This gate is not hypothetical: applied across the collection so far it
**rejected 4 of 5 broadcasts** — `wtr-day2` too_shallow 0.573 (10/14, exit 1),
`coffeechess` too_shallow 0.508 (3/6), `grandswiss` too_shallow 0.450 (2/5),
`divya-r14` **unknown** 0/8 (exit 2, no detection at all) — leaving only
`hexamind` (`JTmovXvZnPM`) at 0.650, itself merely "degraded", never the good
band. Most low-angle, table-level chess broadcasts fail this; when picking a
new broadcast, prefer a steeper overhead camera and treat table-level streams
as a likely reject. A rejected video's file can be deleted — only its id, crop
row, and any frames already pulled need to stay.

`extract_frames.py` does **not** enforce this gate — it prints a loud `!!`
warning to stderr and extracts anyway if the crop row is marked
`rejected_low_angle`, so the check is advisory by design and the operator (you
and the user) is the actual gate. Don't extract a batch from a rejected stream
without saying so out loud first.

## 4. Query the user for frames

Two modes. **Ask which one** — do not assume, and note that "query me for frames"
usually means the interactive one.

### Interactive (default) — `watch_cropped.py`

The user scrubs the broadcast in mpv, already cropped, and grabs the moments they
want. Nothing is re-encoded and no frame is wasted on a cutaway.

```bash
python3 tools/video/watch_cropped.py -s 1:00:00 "data/streams/videos/<video> [<id>].mp4"
```

- Keys: `s` save frame, `space` pause, `.`/`,` frame step, `←`/`→` seek 5s,
  `↑`/`↓` seek 60s, `q` quit.
- `-s`/`--start` opens at a position where play is underway instead of the
  title card.
- Same filename→video_id lookup as `extract_frames.py`: `-o`/`--outdir`
  defaults to `data/streams/frames/<slug>_<video_id>/`, resolved from the
  registry. There is no `--prefix`; saved frames are prefixed with the
  stream's slug.
- `--format png|jpg|webp` (default `png`); unrecognised args pass through to mpv.
- Filenames carry milliseconds, so frame-stepped shots within one second stay
  distinct.
- Requires `mpv`.

Launch it in the background — it holds the terminal until the user quits.

### Batch — `extract_frames.py`

For a bulk sample when the user does not want to sit through the video:

```bash
uv run python tools/video/extract_frames.py -n 50 --seed 0 "data/streams/videos/<video> [<id>].mp4"
uv run python tools/video/extract_frames.py --times 01:42:46 02:15:00 "data/streams/videos/<video> [<id>].mp4"
```

- `-n`/`--count` and `--times` are mutually exclusive, and one of them is **required**
  — there is no default frame count, by design: a batch size is the user's call.
- `--outdir` defaults to `data/streams/frames/<slug>_<video_id>`, resolved from the
  registry, so the correct per-video directory is the zero-argument behaviour. Pass
  `-o` only to send frames somewhere else (e.g. a scratch dir for a crop check).
- `--prefix` defaults to the stream's slug.
- `--seed N` makes a random draw reproducible — pass it whenever the frames might
  need regenerating.
- `--margin` (default `0.02`) skips that fraction at each end, keeping intros and
  outros out of the sample.
- Looks its crop up from `tools/video/crops.csv` by the video's id — no
  `--table` override here, unlike `crop_selector.py`.
- Output names: `<prefix>_<i>_t<HH>h<MM>m<SS>s.png`.

Random timestamps across a whole broadcast will land on commentary cutaways,
crowd shots, and replays — frames where the board is absent or the camera has
moved off the recorded crop. Oversample and let the user cull; do not try to
filter automatically.

### Both modes

Both scripts default their output directory to `data/streams/frames/<slug>_<video_id>/`
(e.g. `hexamind_JTmovXvZnPM`), resolved from the registry by the video's id —
so per-video separation, even when one player appears in several videos, is
the zero-argument behaviour. Pass `-o`/`--outdir` only to send frames
somewhere else, as in the crop-verification check below.

## Verify the crop before committing to a batch

Extract one frame at the same timestamp the crop was selected at, and actually
look at it:

```bash
uv run python tools/video/extract_frames.py --times 1:00:00 --outdir <scratchpad> \
  --prefix croptest "data/streams/videos/<video> [<id>].mp4"
```

Read the PNG back. Check the board is whole and, in particular, whether the far
rank's piece tops are clipped — a crop drawn tightly to the board edge captures
the board plane (fine for corner detection) but truncates tall pieces at the far
side (a problem for silhouette-based classification). Cheaper to catch here than
after 50 frames, and cheaper still than finding out from the angle check.

## Several videos at once

Downloads are independent — start them all in the background. Crop selection is
serial: one OpenCV window at a time, and each video needs its own crop row, since
camera framing differs per broadcast. Never assume one video's crop transfers to
another. Run the angle check per video too — a good crop on a bad broadcast is
still a bad broadcast.
