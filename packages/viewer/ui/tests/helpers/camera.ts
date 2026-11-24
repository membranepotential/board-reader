/**
 * The camera the `/camera` specs drive, and the geometry they measure it against.
 *
 * Chromium's `--use-fake-device-for-media-stream` gives one rolling pattern and no
 * way to say what is in it — good enough for a viewfinder that only has to exist,
 * useless for anything the tracker or the detector has to agree with. So the specs
 * that need a BOARD swap `getUserMedia` for a canvas `captureStream` they draw
 * themselves: a known quad, at a known place.
 *
 * It lives here rather than in one spec file because two files need the same
 * board: `camera.spec.ts` measures the rotate warning a moved framing raises, and
 * `camera-auto.spec.ts` measures what the page does with it when nobody is touching
 * the phone.
 */

import { expect, type Page } from '@playwright/test';

/** The two stages a phone presents, and the ones every layout number here is for. */
export const LANDSCAPE = { width: 844, height: 390 };
export const PORTRAIT = { width: 390, height: 844 };

// ---- a camera whose scene the specs decide ------------------------------------

export const FRAME = { width: 1280, height: 720 };
/** Where the board is drawn, and what `/detect` is stubbed to return. */
export const BOARD = { x: 240, y: 150, w: 800, h: 448 };

/** The drawn board's four corners in frame pixels, optionally displaced. */
export function boardQuad(dx = 0, dy = 0): [number, number][] {
  return [
    [BOARD.x + dx, BOARD.y + dy],
    [BOARD.x + BOARD.w + dx, BOARD.y + dy],
    [BOARD.x + BOARD.w + dx, BOARD.y + BOARD.h + dy],
    [BOARD.x + dx, BOARD.y + BOARD.h + dy]
  ];
}

/** `cornerKey('phone')` — where the crop for the default board is remembered. */
export const CORNERS_KEY = 'board-reader.camera.corners.phone';

/**
 * Seed the crop a previous session saved for this board, as `serialiseFraming`
 * spells it: fractions of the frame plus the labelling, so a resolution change is
 * absorbed on the way back in.
 *
 * A page that opens on one of these does not go looking for the board — it VERIFIES
 * what it was given, with one detection, as soon as the scene is still.
 */
export async function seedFraming(
  page: Page,
  quad: readonly (readonly [number, number])[],
  rot: number
): Promise<void> {
  await page.addInitScript(
    ({ key, pts, rot: turns }) => localStorage.setItem(key, JSON.stringify({ pts, rot: turns })),
    {
      key: CORNERS_KEY,
      pts: quad.map(([x, y]) => [x / FRAME.width, y / FRAME.height]),
      rot
    }
  );
}

export async function drawnBoardCamera(page: Page): Promise<void> {
  await page.addInitScript(
    ({ FRAME, BOARD }) => {
      const canvas = document.createElement('canvas');
      canvas.width = FRAME.width;
      canvas.height = FRAME.height;
      const ctx = canvas.getContext('2d');
      if (ctx === null) return;

      // A bare checkerboard is periodic, and a translation of half a square is then
      // indistinguishable from one of one and a half. These give the homography fit
      // something unambiguous. Deterministic, so a failure is reproducible.
      let seed = 7;
      const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
      const marks = Array.from({ length: 24 }, () => ({ x: rnd(), y: rnd(), r: 4 + rnd() * 10 }));

      const draw = () => {
        ctx.fillStyle = 'rgb(128,128,128)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const bx = BOARD.x;
        const cw = BOARD.w / 8;
        const ch = BOARD.h / 8;
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? 'rgb(51,51,51)' : 'rgb(204,204,204)';
            ctx.fillRect(bx + c * cw, BOARD.y + r * ch, cw + 1, ch + 1);
          }
        }
        ctx.fillStyle = 'rgb(240,120,40)';
        for (const mark of marks) {
          ctx.beginPath();
          ctx.arc(bx + mark.x * BOARD.w, BOARD.y + mark.y * BOARD.h, mark.r, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      // Redrawn every animation frame, not once: a canvas that never changes
      // produces no new frames, and a video whose `currentTime` stands still is
      // exactly what the camera-stall watchdog reopens the camera for.
      const loop = () => {
        draw();
        requestAnimationFrame(loop);
      };
      loop();

      const stream = canvas.captureStream(30);
      Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
        configurable: true,
        value: () => Promise.resolve(stream)
      });
      Object.defineProperty(navigator.mediaDevices, 'enumerateDevices', {
        configurable: true,
        value: () => Promise.resolve([])
      });
    },
    { FRAME, BOARD }
  );
}

/** The live quad's top-left x in VIDEO pixels — the polygon's viewBox is the frame. */
export async function liveQuadX(page: Page): Promise<number> {
  const points = (await page.locator('polygon.quad-line').getAttribute('points')) ?? '';
  return Number(points.trim().split(/[\s,]+/)[0]);
}


/**
 * Wait for the fake camera to actually deliver a frame.
 *
 * `.picture` is sized from the video's own dimensions, so it has no box at all until
 * `loadedmetadata` has fired — measuring before that reads 0 and proves nothing.
 */
export async function liveViewfinder(page: Page): Promise<void> {
  await expect
    .poll(async () => (await page.locator('.picture').boundingBox())?.width ?? 0, {
      // Generous on purpose: several specs now drive a camera at once, and a
      // contended machine can take a while to hand out the first frame.
      timeout: 20_000
    })
    .toBeGreaterThan(0);
}
