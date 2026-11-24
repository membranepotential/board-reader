/**
 * A valid position event for tests, as the node sends it: every key present.
 *
 * The generated validators accept an event whole or not at all, so a test that
 * builds a bare `{board_id, seq, placement}` gets a refusal, not a board. Start
 * from `positionEvent()` and override what the test is about.
 */

import { CLASSES, GRID, START_PLACEMENT } from '#lib/protocol/index.ts';

/** Every square certain of `empty`: a test reads `placement`, never the argmax. */
export const QUIET_DIST = Array.from({ length: GRID }, () =>
  Array.from({ length: GRID }, () => CLASSES.map((_, i) => (i === 0 ? 1 : 0)))
);

export function positionEvent(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    board_id: 'phone',
    seq: 1,
    utc: '2026-08-07T10:00:00.000Z',
    changed_utc: '2026-08-07T10:00:00.000Z',
    orientation: 'a1_bottom_left',
    classes: [...CLASSES],
    placement: START_PLACEMENT,
    pending: [],
    square_dist: QUIET_DIST,
    confidence: 0.9,
    stability: 0.95,
    clock: { running: 'none' },
    status: 'ok',
    obscured: false,
    ...over
  };
}
