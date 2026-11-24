/**
 * Per-square chessground shapes for the live page — ported from index.html's
 * `shapesFor`, minus the decay math: `#lib/stores/liveness` (`changeRings`) now
 * owns the fade timing and hands this a ready-to-draw list, so this is a pure
 * mapping from the store's readout to chessground's `DrawShape` vocabulary.
 *
 * A red ring marks a square the prior decoder relabelled to make the board
 * possible (the drawn piece is the decoder's, not the estimator's); a dashed
 * amber ring one the settled board hasn't adopted yet (a move in flight, or a
 * misread still being debounced — the drawn piece is still whatever was last
 * settled there); a green ring a square the settled board just changed, fading
 * to pale — all outlines, never fills, so they read as annotations on the board
 * rather than as the board itself. There is no confidence ring: the recognizer
 * reports ~1.0 on its wrong squares too, so a threshold on it almost never fired
 * and said nothing when it did.
 */

import type { DrawShape } from '@lichess-org/chessground/draw';
import type { SquareKey } from '../../protocol';
import type { ChangeRing } from '../../stores/liveness';

/**
 * Dashed, unlike every other ring here: chessground's built-in brushes are
 * solid-only, so "not yet settled" is drawn as a `customSvg` instead of a named
 * brush. `cx="0" cy="0"` is deliberate, not a placeholder — chessground centers
 * a `customSvg` box on the square's origin corner, not its middle, so the ring
 * is authored around that same origin to land centered (verified against
 * `svg.ts`'s `renderShape`/`pos2user`, which is where a lichess-styled ring
 * would otherwise land half a square off).
 */
const PENDING_RING_SVG =
  '<circle cx="0" cy="0" r="42" fill="none" stroke="#e2a03f" stroke-width="7" ' +
  'stroke-dasharray="14 9" stroke-linecap="round"/>';

/** One dashed amber ring per square the settled board hasn't adopted yet. */
export function pendingShapes(pending: readonly SquareKey[]): DrawShape[] {
  return pending.map((square) => ({
    orig: square,
    customSvg: { html: PENDING_RING_SVG, center: 'orig' }
  }));
}

export function boardShapes(
  rings: readonly ChangeRing[],
  flipped: readonly SquareKey[] = [],
  pending: readonly SquareKey[] = []
): DrawShape[] {
  const shapes: DrawShape[] = flipped.map((square) => ({ orig: square, brush: 'red' }));
  shapes.push(...pendingShapes(pending));
  for (const ring of rings) {
    shapes.push({ orig: ring.square, brush: ring.phase === 'fresh' ? 'green' : 'paleGreen' });
  }
  return shapes;
}
