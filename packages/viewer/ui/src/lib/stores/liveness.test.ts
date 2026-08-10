import { describe, expect, it } from 'vitest';
import {
  FLASH_MS,
  RATE_WINDOW_MS,
  STALE_MS,
  changeRings,
  eventsPerSecond,
  formatAge,
  isStale,
  pruneChanges,
  withinWindow
} from './liveness';

describe('eventsPerSecond', () => {
  it('reads a 1 fps keepalive as 1.0/s, not as n/window', () => {
    // The window holds 4 stamps spanning 3 s. (n-1)/span = 3/3 = 1.0.
    // n/window would claim 4/3 = 1.33/s, which is the bug this formula exists to avoid.
    expect(eventsPerSecond([0, 1000, 2000, 3000])).toBeCloseTo(1.0, 9);
  });

  it('reads a 12 fps burst as 12/s', () => {
    const stamps = Array.from({ length: 37 }, (_, i) => (i * 1000) / 12);
    expect(eventsPerSecond(stamps)).toBeCloseTo(12, 6);
  });

  it('falls back to the count over the window when there is no span to measure', () => {
    expect(eventsPerSecond([500])).toBeCloseTo(1 / (RATE_WINDOW_MS / 1000), 9);
    expect(eventsPerSecond([500, 500])).toBeCloseTo(2 / (RATE_WINDOW_MS / 1000), 9);
  });

  it('is 0 with no events at all', () => {
    expect(eventsPerSecond([])).toBe(0);
  });
});

describe('withinWindow', () => {
  it('keeps only the stamps still inside the window', () => {
    expect(withinWindow([0, 1000, 5000, 7000], 7000)).toEqual([5000, 7000]);
  });

  it('does not mutate the array it is given', () => {
    const stamps = [0, 1000];
    withinWindow(stamps, 9999);
    expect(stamps).toEqual([0, 1000]);
  });
});

describe('staleness', () => {
  it('turns over exactly at STALE_MS', () => {
    expect(isStale(STALE_MS)).toBe(false);
    expect(isStale(STALE_MS + 1)).toBe(true);
  });
});

describe('formatAge', () => {
  it.each([
    [0, '0.0s ago'],
    [3400, '3.4s ago'],
    [42_000, '42s ago'],
    [125_000, '2m 5s ago'],
    [-100, '0.0s ago']
  ])('formats %i ms as %s', (ms, text) => {
    expect(formatAge(ms)).toBe(text);
  });
});

describe('change rings', () => {
  it('is fresh for the first half of the flash and fading for the second', () => {
    const changedAt = { e4: 1000, e2: 1000 } as const;
    expect(changeRings(changedAt, 1000 + FLASH_MS / 4)).toEqual([
      { square: 'e4', phase: 'fresh' },
      { square: 'e2', phase: 'fresh' }
    ]);
    expect(changeRings(changedAt, 1000 + (FLASH_MS * 3) / 4).map((r) => r.phase)).toEqual([
      'fading',
      'fading'
    ]);
  });

  it('drops a ring that has finished fading', () => {
    expect(changeRings({ e4: 0 }, FLASH_MS + 1)).toEqual([]);
  });

  it('prunes to exactly the rings still visible', () => {
    expect(pruneChanges({ e4: 0, e2: 900 }, 950)).toEqual({ e2: 900 });
  });
});
