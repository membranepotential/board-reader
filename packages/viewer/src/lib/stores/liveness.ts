/**
 * How fresh a board's reading is — the pure half of the live page's readout.
 *
 * Kept apart from the store so the arithmetic can be tested without a socket or a
 * clock: `positions.svelte.ts` is then only wiring (functional core, imperative
 * shell). Every function here takes the time explicitly; nothing reads a clock.
 */

import type { SquareKey } from '../protocol';

/** No event for this long and the card reads as stale. */
export const STALE_MS = 5000;
/** Rolling window behind the events/sec figure. */
export const RATE_WINDOW_MS = 3000;
/** How long a changed square keeps its ring before fading out. */
export const FLASH_MS = 900;
/** One ticker drives every card's age, rate and flash decay. */
export const TICK_MS = 200;

/**
 * Events per second over a window of arrival stamps.
 *
 * **`(n-1)/span`, not `n/window`.** A 1 fps keepalive puts 4 stamps 3 s apart in a
 * 3 s window; that is 1.0/s, and `n/window` would claim 1.33/s. Only when fewer
 * than two stamps make a span is there nothing to measure, and the count over the
 * window is the best available answer.
 */
export function eventsPerSecond(stamps: readonly number[], windowMs = RATE_WINDOW_MS): number {
  const n = stamps.length;
  if (n === 0) return 0;
  const first = stamps[0] as number;
  const last = stamps[n - 1] as number;
  const span = n >= 2 ? (last - first) / 1000 : 0;
  return span > 0 ? (n - 1) / span : n / (windowMs / 1000);
}

/** Drop the stamps that have fallen out of the window. Pure: returns a new array. */
export function withinWindow(
  stamps: readonly number[],
  now: number,
  windowMs = RATE_WINDOW_MS
): number[] {
  return stamps.filter((at) => now - at <= windowMs);
}

export function isStale(ageMs: number, staleMs = STALE_MS): boolean {
  return ageMs > staleMs;
}

/** "3.4s ago" / "42s ago" / "2m 5s ago" — the age as the card says it. */
export function formatAge(ms: number): string {
  const s = Math.max(0, ms) / 1000;
  if (s < 10) return `${s.toFixed(1)}s ago`;
  if (s < 60) return `${Math.round(s)}s ago`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s ago`;
}

/**
 * A square that just changed, and how far its ring has faded.
 *
 * The decay is derived from the change's age rather than driven by a timer per
 * event: one shared ticker advancing `now` repaints every ring, and a card that
 * stops receiving events simply stops fading rather than leaking timers.
 */
export type RingPhase = 'fresh' | 'fading';

export interface ChangeRing {
  readonly square: SquareKey;
  readonly phase: RingPhase;
}

export function changeRings(
  changedAt: Readonly<Partial<Record<SquareKey, number>>>,
  now: number,
  flashMs = FLASH_MS
): ChangeRing[] {
  const rings: ChangeRing[] = [];
  for (const [square, at] of Object.entries(changedAt)) {
    if (at === undefined) continue;
    const age = now - at;
    if (age > flashMs) continue;
    rings.push({ square: square as SquareKey, phase: age < flashMs / 2 ? 'fresh' : 'fading' });
  }
  return rings;
}

/** Forget rings that have finished fading, so the record stays the visible ones. */
export function pruneChanges(
  changedAt: Readonly<Partial<Record<SquareKey, number>>>,
  now: number,
  flashMs = FLASH_MS
): Partial<Record<SquareKey, number>> {
  const kept: Partial<Record<SquareKey, number>> = {};
  for (const [square, at] of Object.entries(changedAt)) {
    if (at !== undefined && now - at <= flashMs) kept[square as SquareKey] = at;
  }
  return kept;
}
