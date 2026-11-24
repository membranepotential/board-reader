---
paths:
  - 'packages/viewer/tests/**'
  - 'packages/viewer/playwright.config.ts'
---

# The Playwright suite (packages/viewer/tests)

- Fake the phone camera with `tests/helpers/camera.ts`'s `drawnBoardCamera`, which returns a fresh `source.clone()` of the canvas stream per `getUserMedia` call and patches each clone track's `getSettings().deviceId`, because the page stops the outgoing tracks before every reopen, so handing back one shared stream after a lens or resolution switch returns a stopped track and the stall watchdog reopens 6 s later, which looks like a page bug.
- Use the drawn-canvas camera, not Chromium's `--use-fake-device`, for anything clock-driven, because its `requestAnimationFrame` draw loop is what keeps `video.currentTime` advancing for the stall watchdog.
- Prove that a crop reached the wire by decoding the JPEG SOF marker of each binary `/ws/ingest` message (`jpegSize`), never by trusting a counter.
- Poll for a result the page produces asynchronously (`expect.poll`, `toPass`) instead of reading it once, because with six workers driving capture loops, contention alone stretches a 2 s step past 10 s.
