import { describe, expect, it } from 'vitest';

import { NUM_CLASSES } from '#lib/protocol/index.ts';
import { QUIET_DIST as QUIET_ROWS } from '#lib/protocol/fixtures.ts';
import { PositionEvent } from '#lib/protocol/generated.ts';
import {
  argmaxLabels,
  largestGains,
  logLikelihood,
  mismatches,
  squareCeilings,
  squareLogs,
  SQUARES
} from './emission.ts';

const asDist = (rows: unknown) => PositionEvent.shape.square_dist.parse(rows);
/** Every square certain of `empty`. */
const QUIET_DIST = asDist(QUIET_ROWS);
/** Every square split evenly between empty and a white pawn. */
const SPLIT = asDist(QUIET_ROWS.map((row) => row.map(() => [0.5, 0.5, ...Array<number>(NUM_CLASSES - 2).fill(0)])));

describe('squareLogs', () => {
  it('is a distribution per square after the temperature and the floor', () => {
    const logs = squareLogs(SPLIT, { temperature: 2, floor: 0.05 });
    for (let square = 0; square < SQUARES; square++) {
      const total = logs.slice(square * NUM_CLASSES, (square + 1) * NUM_CLASSES).reduce((sum, l) => sum + Math.exp(l), 0);
      expect(total).toBeCloseTo(1, 9);
    }
  });

  it('keeps the read as it is with no temperature and no floor', () => {
    const logs = squareLogs(SPLIT, { temperature: 1, floor: 0 });
    expect(logs[0]).toBeCloseTo(Math.log(0.5), 9);
    expect(logs[1]).toBeCloseTo(Math.log(0.5), 9);
  });

  it('bounds what a square certain of the wrong class costs', () => {
    const logs = squareLogs(QUIET_DIST, { temperature: 2, floor: 0.05 });
    // Square 0 is certain of `empty`; a pawn there costs log(floor / 13), not -Infinity.
    expect(logs[1]).toBeCloseTo(Math.log(0.05 / NUM_CLASSES), 6);
  });
});

describe('largestGains', () => {
  it('is what fixing the worst squares gains, so no change of that many squares gains more', () => {
    const logs = squareLogs(QUIET_DIST, { temperature: 1, floor: 0.05 });
    const ceilings = squareCeilings(logs);
    const board = argmaxLabels(QUIET_DIST);
    const wrong = board.slice();
    wrong[0] = 1;
    wrong[9] = 7;
    const fixed = logLikelihood(logs, board) - logLikelihood(logs, wrong);
    expect(largestGains(logs, ceilings, wrong, 4)).toBeCloseTo(fixed, 9);
    expect(largestGains(logs, ceilings, wrong, 1)).toBeCloseTo(fixed / 2, 9);
  });
});

describe('logLikelihood, argmaxLabels, mismatches', () => {
  it('scores a board by its squares, and counts the squares two boards disagree on', () => {
    const empty = argmaxLabels(QUIET_DIST);
    expect([...empty].every((label) => label === 0)).toBe(true);
    const logs = squareLogs(QUIET_DIST, { temperature: 1, floor: 0.05 });
    const pawn = empty.slice();
    pawn[12] = 1;
    expect(logLikelihood(logs, empty)).toBeGreaterThan(logLikelihood(logs, pawn));
    expect(mismatches(empty, pawn)).toBe(1);
  });
});
