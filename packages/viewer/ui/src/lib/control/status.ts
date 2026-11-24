/**
 * What the control page says about the phone and its stream, derived from one or
 * two `CameraView`s. All ages are on the node's clock (`view.now`): the phone, the
 * laptop and the node need not agree on the time.
 */

import type { CameraView } from './client';

/** How often the control page polls the node. */
export const POLL_MS = 1_000;

/**
 * The phone reports about twice a second; three seconds of silence is several
 * missed polls, not jitter.
 */
export const PHONE_SILENT_S = 3;

/** Seconds since the phone last reported, or `null` if it never has. */
export function reportAge(view: CameraView): number | null {
  return view.seen_at === null ? null : Math.max(0, view.now - view.seen_at);
}

export function phoneConnected(view: CameraView): boolean {
  const age = reportAge(view);
  return age !== null && age < PHONE_SILENT_S;
}

/** Seconds since `/ingest` last received a frame for the board, or `null`. */
export function frameAge(view: CameraView): number | null {
  return view.frame === null ? null : Math.max(0, view.now - view.frame.received_at);
}

/** One reading of the stream's position, for measuring its rate between two polls. */
export interface StreamSample {
  readonly seq: number;
  readonly now: number;
}

export function streamSample(view: CameraView): StreamSample | null {
  return view.frame === null ? null : { seq: view.frame.seq, now: view.now };
}

/**
 * Frames per second received between two samples, from the frame sequence number.
 *
 * `null` when there is nothing to measure: no earlier sample, no time between them,
 * or a sequence that went backwards (the phone reloaded and started again at 0).
 */
export function frameRate(previous: StreamSample | null, next: StreamSample | null): number | null {
  if (previous === null || next === null) return null;
  const dt = next.now - previous.now;
  const frames = next.seq - previous.seq;
  return dt > 0 && frames >= 0 ? frames / dt : null;
}

/** `2.3 s`, `4 min`: an age as an operator reads it. */
export function formatAge(seconds: number | null): string {
  if (seconds === null) return 'never';
  if (seconds < 60) return `${seconds.toFixed(1)} s ago`;
  return `${Math.round(seconds / 60)} min ago`;
}
