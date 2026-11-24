---
paths:
  - 'packages/viewer/ui/src/routes/camera/**'
  - 'packages/viewer/ui/src/routes/control/**'
  - 'packages/viewer/ui/src/lib/components/camera/**'
  - 'packages/viewer/ui/src/lib/components/control/**'
  - 'packages/viewer/ui/src/lib/control/**'
  - 'packages/viewer/ui/src/lib/edge/**'
  - 'packages/core/src/board_reader_core/control.py'
  - 'packages/inference/src/board_reader_inference/camera_control.py'
---

# The dumb-camera phone (/camera) and its control page (/control)

- Let the phone pull its config: it POSTs its `CameraReport` to `/boards/{id}/camera/report` every 500 ms and applies the reply, with every poll bounded (`AbortSignal.timeout`, `CONTROL_TIMEOUT_MS`) and never overlapped, because no socket then has to stay open while the camera idles, and one hung fetch would otherwise stop the phone hearing any change with nothing on screen. Measure ages in the control state against the node's `now`, never the browser clock.
- Record what the phone asked the camera for (`Capture`: lens label + resolution) before the `getUserMedia` call, and let `nextCapture()` compare the config against that, not against what opened, so a lens that refuses is tried once rather than on every poll.
- Choose a lens by opening any camera first, then `enumerateDevices`, and persist the choice by label; stop the outgoing tracks before requesting the next lens, because iOS shows labels only after a `getUserMedia` call, regenerates deviceIds per session, hands out one camera at a time and has no `zoom` capability.
- Keep the three recovery layers working together: track `ended`/`mute`/`unmute` listeners (guard `stream !== next`, since a replaced lens's tracks end too), `visibilitychange`/`pageshow`, and the `video.currentTime` watchdog (`noteFrameClock`). The visible handler must reset the clock itself (`freshFrameClock`), because iOS suspends `setInterval` in the background; serialize every `openCamera` through one promise queue and keep `recoverCamera`'s in-flight guard, because two interleaved opens stop each other's tracks and the loser keeps the camera claimed.
- Make every WebSocket handler close over its own socket and return when a newer socket replaced it (`if (socket !== this.socket) return`), because a stale socket's close otherwise nulls the live connection and spawns a duplicate. Compute anything derived from mutable UI state before the first `await` in the send path, and name bursts from a never-reset counter, not from seq, because seq advances only on frames actually sent.
- Snap crop edges to integer units of 1/10000 (`lib/control/crop.ts`) and derive `x` and `w` from them, because with float edges `x + w` can land a hair past 1 and the node's `CropRect` (`x + w <= 1`, `w >= MIN_CROP_SIDE`) refuses the save.
- Size a picture wrapper in JS (`k = min(box.w/src.w, box.h/src.h)`) and put the image and its overlay inside it, never `object-fit: contain` plus a coordinate-conversion helper, because `contain` letterboxes inside the element, so every touch needs a second coordinate space and a mistake there sends wrong geometry while the preview still looks right (`CropEditor.svelte` does it this way).
