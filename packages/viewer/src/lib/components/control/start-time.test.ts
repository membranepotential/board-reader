import { describe, expect, it } from 'vitest';

import { parseStartTime } from './start-time.ts';

describe('parseStartTime', () => {
  it('reads nothing as the beginning', () => {
    expect(parseStartTime('  ')).toEqual({ ok: true, seconds: null });
  });

  it.each([
    ['90', 90],
    ['1:30', 90],
    ['1:02:03', 3723],
    ['0:00:07.5', 7.5]
  ])('reads %s as %d s', (text, seconds) => {
    expect(parseStartTime(text)).toEqual({ ok: true, seconds });
  });

  it.each(['1:75', '1:60:00', 'abc', '-3', '1::2'])('refuses %s', (text) => {
    expect(parseStartTime(text).ok).toBe(false);
  });
});
