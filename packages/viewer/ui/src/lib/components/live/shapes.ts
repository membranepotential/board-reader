/**
 * Per-square chessground shapes for the live page — ported from index.html's
 * `shapesFor`, minus the decay math: `$lib/stores/liveness` (`changeRings`) now
 * owns the fade timing and hands this a ready-to-draw list, so this is a pure
 * mapping from the store's readout to chessground's `DrawShape` vocabulary.
 *
 * A yellow ring marks a square under the confidence threshold; a red ring one
 * the prior decoder relabelled to make the board possible (the drawn piece is
 * the decoder's, not the estimator's); a green ring on a square that just
 * changed, fading to pale — all outlines, never fills, so they read as
 * annotations on the board rather than as the board itself.
 */

import type { DrawShape } from '@lichess-org/chessground/draw';
import type { SquareKey } from '../../protocol';
import type { ChangeRing } from '../../stores/liveness';

export function boardShapes(
  low: readonly SquareKey[],
  rings: readonly ChangeRing[],
  flipped: readonly SquareKey[] = []
): DrawShape[] {
  const shapes: DrawShape[] = low.map((square) => ({ orig: square, brush: 'yellow' }));
  for (const square of flipped) shapes.push({ orig: square, brush: 'red' });
  for (const ring of rings) {
    shapes.push({ orig: ring.square, brush: ring.phase === 'fresh' ? 'green' : 'paleGreen' });
  }
  return shapes;
}
