import { describe, expect, it } from 'vitest';
import { formatAge } from './status';

describe('formatAge', () => {
  it('formats an age for the operator', () => {
    expect(formatAge(null)).toBe('never');
    expect(formatAge(2.34)).toBe('2.3 s ago');
    expect(formatAge(240)).toBe('4 min ago');
  });
});
