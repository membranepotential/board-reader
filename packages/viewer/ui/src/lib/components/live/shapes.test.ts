import { describe, expect, it } from 'vitest';
import { boardShapes } from './shapes';
import type { ChangeRing } from '../../stores/liveness';

describe('boardShapes', () => {
  it('ring a low-confidence square yellow', () => {
    expect(boardShapes(['e4'], [])).toEqual([{ orig: 'e4', brush: 'yellow' }]);
  });

  it('rings a fresh change green and a fading one paleGreen', () => {
    const rings: ChangeRing[] = [
      { square: 'a1', phase: 'fresh' },
      { square: 'h8', phase: 'fading' }
    ];
    expect(boardShapes([], rings)).toEqual([
      { orig: 'a1', brush: 'green' },
      { orig: 'h8', brush: 'paleGreen' }
    ]);
  });

  it('combines low-confidence and change rings, low-confidence first', () => {
    const rings: ChangeRing[] = [{ square: 'd4', phase: 'fresh' }];
    expect(boardShapes(['e4'], rings)).toEqual([
      { orig: 'e4', brush: 'yellow' },
      { orig: 'd4', brush: 'green' }
    ]);
  });

  it('draws nothing for a settled, confident board', () => {
    expect(boardShapes([], [])).toEqual([]);
  });
});
