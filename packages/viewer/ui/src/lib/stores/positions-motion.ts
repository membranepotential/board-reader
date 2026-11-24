/**
 * Per-frame motion on `/ws/viewer` — whether the camera sees the board moving
 * RIGHT NOW, independent of the reads.
 *
 * A position event's own `stability` is the stability of the frame that was READ,
 * and the node reads after motion ends (the burst-end schedule), so it says "at
 * rest" on nearly every event. Motion is therefore its own small message on the
 * same stream, sent by the node per ingested frame (at most ~5/s per board, and
 * always on a moving <-> rest transition):
 *
 *     {"type": "motion", "board_id": "phone", "moving": true, "stability": 0.42, "ts": 17955.3}
 *
 * `moving` is whether the node sees the board moving at that frame, `stability`
 * the frame's meter reading and `ts` its capture time (edge clock, never compared
 * with ours). The message is optional: a node that does not send it leaves the
 * readout `none`, and the card shows no motion chip at all rather than a guess.
 */

import { isRecord } from '../protocol';

export const MOTION_TYPE = 'motion';

export interface MotionMessage {
  readonly type: typeof MOTION_TYPE;
  readonly board_id: string;
  readonly moving: boolean;
  readonly stability: number;
  readonly ts: number;
}

/**
 * A board's last motion sample, stamped with OUR arrival clock: freshness is
 * judged against the page's clock, never the edge's capture time.
 */
export interface MotionSample {
  readonly moving: boolean;
  readonly stability: number;
  readonly at: number;
}

/**
 * What the card shows. `none`: this board never sent motion (an older node, or a
 * non-motion edge). `stale`: it did, but nothing arrived for `MOTION_STALE_MS`,
 * so the last answer is no longer evidence — a stream that dies mid-move must not
 * claim 'moving' forever.
 */
export type MotionReadout =
  | { readonly kind: 'none' }
  | { readonly kind: 'stale' }
  | { readonly kind: 'moving' | 'still'; readonly stability: number };

/**
 * The edge sends at least one frame a second at rest (its keepalive), and the node
 * a motion message per ingested frame (rate-capped), so three silent seconds mean
 * the signal is gone, not that the board is still. Position events carry no `type`
 * key; only `type: "motion"` is routed here.
 */
export const MOTION_STALE_MS = 3000;

/** `raw` as a motion message, or `null` when it is anything else (a position event). */
export function parseMotionMessage(raw: unknown): MotionMessage | null {
  if (!isRecord(raw) || raw['type'] !== MOTION_TYPE) return null;
  const { board_id, moving, stability, ts } = raw;
  if (typeof board_id !== 'string' || board_id === '') return null;
  if (typeof moving !== 'boolean') return null;
  if (typeof stability !== 'number' || !Number.isFinite(stability)) return null;
  if (typeof ts !== 'number' || !Number.isFinite(ts)) return null;
  const clamped = Math.min(1, Math.max(0, stability));
  return { type: MOTION_TYPE, board_id, moving, stability: clamped, ts };
}

export function motionReadout(sample: MotionSample | undefined, now: number): MotionReadout {
  if (sample === undefined) return { kind: 'none' };
  if (now - sample.at > MOTION_STALE_MS) return { kind: 'stale' };
  return { kind: sample.moving ? 'moving' : 'still', stability: sample.stability };
}
