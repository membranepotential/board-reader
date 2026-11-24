/**
 * A camera whose picture the `/camera` specs decide.
 *
 * Chromium's `--use-fake-device-for-media-stream` gives one rolling pattern at a size
 * the spec cannot choose. The specs that measure the crop, the preview or portrait
 * framing swap `getUserMedia` for a canvas `captureStream` instead: a board drawn at
 * a known place, in a frame of a known size. `crossHand` sends a hand across it.
 */

import { expect, type Page } from '@playwright/test';

/** The stages a phone presents. */
export const LANDSCAPE = { width: 844, height: 390 };
export const PORTRAIT = { width: 390, height: 844 };

export const FRAME = { width: 1280, height: 720 };

/** The lenses the drawn camera claims to have, as `enumerateDevices` lists them. */
export const LENSES = [
  { deviceId: 'lens-wide', label: 'Back Camera' },
  { deviceId: 'lens-ultra', label: 'Back Ultra Wide Camera' }
] as const;

export async function drawnBoardCamera(
  page: Page,
  frame: { readonly width: number; readonly height: number } = FRAME
): Promise<void> {
  await page.addInitScript(
    ({ frame, lenses }) => {
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d');
      if (ctx === null) return;
      const side = Math.min(frame.width, frame.height) * 0.6;
      const bx = (frame.width - side) / 2;
      const by = (frame.height - side) / 2;
      let tick = 0;
      /** When the hand set off (`crossHand`); it takes HAND_MS to cross the board. */
      let handFrom: number | null = null;
      const HAND_MS = 1500;
      Object.defineProperty(window, '__crossHand', {
        value: () => {
          handFrom = performance.now();
        }
      });

      const draw = () => {
        ctx.fillStyle = 'rgb(128,128,128)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const cell = side / 8;
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? 'rgb(51,51,51)' : 'rgb(204,204,204)';
            ctx.fillRect(bx + c * cell, by + r * cell, cell + 1, cell + 1);
          }
        }
        const crossed = handFrom === null ? 1 : (performance.now() - handFrom) / HAND_MS;
        if (crossed < 1) {
          ctx.fillStyle = 'rgb(40,30,25)';
          ctx.beginPath();
          ctx.arc(bx + side * (0.1 + 0.8 * crossed), by + side * 0.55, side * 0.12, 0, 2 * Math.PI);
          ctx.fill();
        }
        // One pixel that changes every frame: a canvas that never changes produces
        // no new frames, and a video whose `currentTime` stands still is exactly
        // what the camera-stall watchdog reopens the camera for.
        ctx.fillStyle = `rgb(${tick++ % 256},0,0)`;
        ctx.fillRect(0, 0, 1, 1);
      };
      const loop = () => {
        draw();
        requestAnimationFrame(loop);
      };
      loop();

      const source = canvas.captureStream(30);
      // A fresh clone per request: the page stops the outgoing tracks before every
      // reopen (iOS hands out one camera at a time), and handing back the stopped
      // stream would leave a lens switch with a dead picture.
      const open = (): MediaStream => {
        const stream = source.clone();
        const track = stream.getVideoTracks()[0];
        if (track !== undefined) {
          // The page names the open lens from the track's deviceId.
          const settings = track.getSettings.bind(track);
          track.getSettings = () => ({ ...settings(), deviceId: lenses[0].deviceId });
        }
        return stream;
      };
      // Every request is kept, so a spec can read what the page asked the camera for.
      const asked: MediaStreamConstraints[] = [];
      Object.defineProperty(window, '__cameraRequests', { value: asked });
      Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
        configurable: true,
        value: (constraints: MediaStreamConstraints) => {
          asked.push(constraints);
          return Promise.resolve(open());
        }
      });
      Object.defineProperty(navigator.mediaDevices, 'enumerateDevices', {
        configurable: true,
        value: () =>
          Promise.resolve(
            lenses.map((lens) => ({ ...lens, kind: 'videoinput', groupId: '', toJSON: () => lens }))
          )
      });
    },
    { frame, lenses: LENSES }
  );
}

/** Send a hand across the drawn board, once, over 1.5 s. */
export async function crossHand(page: Page): Promise<void> {
  await page.evaluate(() => (window as unknown as { __crossHand: () => void }).__crossHand());
}

/**
 * Wait for the fake camera to actually deliver a frame: the status line names the
 * frame size only once `loadedmetadata` has fired.
 */
export async function liveViewfinder(page: Page): Promise<void> {
  await expect(page.getByTestId('camera-status')).toContainText(/\d+×\d+/, { timeout: 20_000 });
}

/** `[width, height]` of a baseline or progressive JPEG, read from its SOF marker. */
export function jpegSize(bytes: Buffer): [number, number] {
  let i = 2;
  while (i + 9 < bytes.length) {
    if (bytes[i] !== 0xff) throw new Error('not a JPEG marker stream');
    const marker = bytes[i + 1]!;
    const length = bytes.readUInt16BE(i + 2);
    if (marker === 0xc0 || marker === 0xc2) {
      return [bytes.readUInt16BE(i + 7), bytes.readUInt16BE(i + 5)];
    }
    i += 2 + length;
  }
  throw new Error('no SOF marker');
}

/** Every `getUserMedia` request the page made of the drawn camera, oldest first. */
export function cameraRequests(page: Page): Promise<MediaStreamConstraints[]> {
  return page.evaluate(
    () => (window as unknown as { __cameraRequests: MediaStreamConstraints[] }).__cameraRequests
  );
}
