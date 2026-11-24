/**
 * The /camera page's state and the pure decisions it makes.
 *
 * The phone is only a camera (DESIGN D10, "dumb camera"). It has no controls: the
 * control UI on the laptop chooses the lens, the resolution, the crop rectangle and
 * whether to stream, and the phone hears that config on its report poll and applies
 * it. What is left here is small on purpose:
 *
 *  - `CameraState` — the camera is idle, being asked for, live (with the frame size
 *    it delivers), or blocked (with why).
 *  - `Capture` — what the open camera was ASKED for (lens label, resolution), so a
 *    config is turned into a reopen exactly once, not on every poll.
 *  - the crop box, the rate meter and the stall watchdog.
 *
 * Everything here is pure and DOM-free so it is tested directly (`state.test.ts`).
 */

import { RESOLUTIONS, type Resolution } from '$lib/protocol';
import type { CameraConfig, CameraDevice, CropRect, FrameSize } from '$lib/protocol/generated';

// ---- constants ---------------------------------------------------------------

/** Downscale before encoding: the recognizer reads 644 px, not pores. */
export const SEND_LONG_SIDE = 960;
export const JPEG_QUALITY = 0.75;
/** The focus score is measured on this small copy of the sent crop. */
export const SHARPNESS_LONG_SIDE = 400;
/** The full-frame preview the control UI draws the crop on: small and rare. */
export const PREVIEW_LONG_SIDE = 480;
export const PREVIEW_QUALITY = 0.6;
/** Sent only while a control page watches (the node's `previews` message). */
export const PREVIEW_MS = 1500;
/** Status repaint period; the capture loop runs far faster and does not wait on it. */
export const TICK_MS = 200;
export const RATE_WINDOW_MS = 3000;

export const DEFAULT_BOARD_ID = 'phone';

/** `?board=<id>` names the board; the phone and the control UI both read it. */
export function boardIdFrom(search: string): string {
  return new URLSearchParams(search).get('board')?.trim() || DEFAULT_BOARD_ID;
}

// ---- the camera --------------------------------------------------------------

/** Why the camera is not available. The page renders one explainer per variant. */
export type BlockedReason =
  | { readonly kind: 'insecure' }
  | { readonly kind: 'unsupported' }
  | { readonly kind: 'refused'; readonly name: string; readonly detail: string };

export type CameraState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'requesting' }
  | { readonly kind: 'live'; readonly frame: FrameSize }
  | { readonly kind: 'blocked'; readonly reason: BlockedReason };

/** What the open (or opening) camera was asked for. */
export interface Capture {
  /** The lens label asked for; `null` is the default rear camera. */
  readonly device: string | null;
  readonly resolution: Resolution;
}

/**
 * The capture a config asks for, or `null` when the current one already is it.
 *
 * A lens the phone does not have is not asked for: the current lens stays, and
 * the report (which names the lens actually open) lets the control UI see that.
 */
export function nextCapture(
  current: Capture,
  config: CameraConfig,
  lenses: readonly CameraDevice[]
): Capture | null {
  const device =
    config.device === null || lenses.some((lens) => lens.label === config.device)
      ? config.device
      : current.device;
  if (device === current.device && config.resolution === current.resolution) return null;
  return { device, resolution: config.resolution };
}

/** The `getUserMedia` video constraints for a capture. */
export function videoConstraints(
  resolution: Resolution,
  deviceId: string | null
): MediaTrackConstraints {
  const { width, height } = RESOLUTIONS[resolution];
  // `ideal`, never `exact`: a camera that cannot deliver the size gets the nearest
  // it can, and a laptop with only a front camera still gets a picture.
  const size = { width: { ideal: width }, height: { ideal: height } };
  return deviceId === null
    ? { facingMode: { ideal: 'environment' }, ...size }
    : { deviceId: { exact: deviceId }, ...size };
}

// ---- the crop ----------------------------------------------------------------

/** A rectangle in whole frame pixels. */
export interface PixelBox {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

/** The configured crop in this frame's pixels; no crop is the whole frame. */
export function cropBox(crop: CropRect | null, frame: FrameSize): PixelBox {
  if (crop === null) return { x: 0, y: 0, w: frame.width, h: frame.height };
  const x = Math.min(Math.round(crop.x * frame.width), frame.width - 1);
  const y = Math.min(Math.round(crop.y * frame.height), frame.height - 1);
  const right = Math.min(Math.round((crop.x + crop.w) * frame.width), frame.width);
  const bottom = Math.min(Math.round((crop.y + crop.h) * frame.height), frame.height);
  return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
}

/** `box` scaled down (never up) so its long side is at most `longSide`. */
export function fitLongSide(
  box: { readonly w: number; readonly h: number },
  longSide: number
): { readonly w: number; readonly h: number } {
  const k = Math.min(1, longSide / Math.max(box.w, box.h));
  return { w: Math.max(1, Math.round(box.w * k)), h: Math.max(1, Math.round(box.h * k)) };
}

// ---- rates -------------------------------------------------------------------

/**
 * Send rate over the trailing window.
 *
 * `(n-1)/span`, not `n/window`: at the 1 fps keepalive the window holds four
 * stamps spanning 3 s, which is 1.0/s — `n/window` would claim 1.3/s.
 */
export function ratePerSecond(stamps: readonly number[], streaming: boolean): number {
  const n = stamps.length;
  const first = stamps[0];
  const last = stamps[n - 1];
  const span = n >= 2 && first !== undefined && last !== undefined ? (last - first) / 1000 : 0;
  if (span > 0) return (n - 1) / span;
  return streaming ? n / (RATE_WINDOW_MS / 1000) : 0;
}

/** Stamps still inside the rate window. */
export function withinRateWindow(stamps: readonly number[], now: number): number[] {
  return stamps.filter((t) => now - t <= RATE_WINDOW_MS);
}

// ---- camera stall watchdog ----------------------------------------------------
// A phone's camera track dies quietly: iOS mutes it when the app is backgrounded
// or another app takes the camera, and sometimes never revives it on return. The
// preview then shows the last frame forever while the loop happily streams it.
// The watchdog reads `video.currentTime` every tick — a live capture advances it;
// a dead one does not — and escalates: first `play()` (fixes the paused-on-return
// case for free), then a full camera reopen, backed off so a genuinely absent
// camera is not hammered.

/** No clock advance for this long (while visible) → try `play()`. */
export const STALL_PLAY_MS = 2000;
/** Still no advance → reopen the camera. Also the floor of the reopen backoff. */
export const STALL_REOPEN_MS = 6000;
export const STALL_REOPEN_MAX_MS = 60000;

export interface FrameClock {
  /** Last observed `video.currentTime` (seconds — its own clock, not ours). */
  readonly time: number;
  /** When it last advanced, `performance.now()` ms. */
  readonly changedAt: number;
  /** Whether `play()` has already been tried for THIS stall. */
  readonly played: boolean;
  /** Consecutive reopens without the clock ever advancing in between. */
  readonly reopens: number;
  readonly nextReopenAt: number;
}

export type StallAction = 'none' | 'play' | 'reopen';

export function freshFrameClock(now: number): FrameClock {
  return { time: -1, changedAt: now, played: false, reopens: 0, nextReopenAt: 0 };
}

/**
 * Fold one tick's reading in. `hidden` resets the grace period rather than
 * counting: a backgrounded page throttles both the video clock and this very
 * loop, and recovery must not fire the instant the operator returns.
 */
export function noteFrameClock(
  clock: FrameClock,
  time: number,
  now: number,
  hidden: boolean
): { clock: FrameClock; action: StallAction } {
  if (hidden || time !== clock.time) {
    return {
      clock: { time, changedAt: now, played: false, reopens: 0, nextReopenAt: 0 },
      action: 'none'
    };
  }
  const stalledMs = now - clock.changedAt;
  if (stalledMs >= STALL_PLAY_MS && !clock.played) {
    return { clock: { ...clock, played: true }, action: 'play' };
  }
  if (stalledMs >= STALL_REOPEN_MS && now >= clock.nextReopenAt) {
    const backoff = Math.min(STALL_REOPEN_MS * 2 ** clock.reopens, STALL_REOPEN_MAX_MS);
    return {
      clock: { ...clock, reopens: clock.reopens + 1, nextReopenAt: now + backoff },
      action: 'reopen'
    };
  }
  return { clock, action: 'none' };
}
