import { describe, expect, it } from 'vitest';
import { whiteSideCopy } from './direction';

describe('the side White sits on', () => {
  it('reads as the operator sees the preview', () => {
    expect(whiteSideCopy('left')).toBe('White sits on the left');
    expect(whiteSideCopy('bottom')).toBe('White sits at the bottom');
    expect(whiteSideCopy('top')).toBe('White sits at the top');
    expect(whiteSideCopy('right')).toBe('White sits on the right');
  });
});
