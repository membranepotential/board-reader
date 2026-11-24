/**
 * How well one read explains a board: the decoder's emission model.
 *
 * A board is a `Labels` array: one class index (into `CLASSES`) per square, in the
 * grid order every `square_dist` uses (row 0 is rank 8, column 0 is file a). The read's
 * per-square distribution is softened by a temperature, because the recognizer is
 * overconfident, and mixed with a uniform floor, so that one occluded or misread square
 * costs a bounded amount instead of vetoing the right board.
 */

import { GRID, NUM_CLASSES, type SquareDist } from '#lib/protocol/index.ts';

/** One class index per square, `GRID * GRID` long, in grid order. */
export type Labels = Uint8Array;

export const SQUARES = GRID * GRID;

export interface EmissionParams {
  /** Each probability becomes `p^(1/temperature)`, renormalized; 1 keeps the read as is. */
  readonly temperature: number;
  /** Share of each square's probability spread evenly over all classes, in [0, 1). */
  readonly floor: number;
}

/** The smallest probability taken to a power, so an exact 0 stays finite. */
const TINY = 1e-30;

/**
 * `log p(class)` for every square of one read, flattened as
 * `[square * NUM_CLASSES + class]`, after the temperature and the floor.
 */
export function squareLogs(dist: SquareDist, { temperature, floor }: EmissionParams): Float64Array {
  const logs = new Float64Array(SQUARES * NUM_CLASSES);
  const scores = new Float64Array(NUM_CLASSES);
  for (let square = 0; square < SQUARES; square++) {
    const probabilities = dist[Math.floor(square / GRID)]![square % GRID]!;
    let sum = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      scores[c] = Math.max(probabilities[c]!, TINY) ** (1 / temperature);
      sum += scores[c]!;
    }
    for (let c = 0; c < NUM_CLASSES; c++) {
      logs[square * NUM_CLASSES + c] = Math.log(((1 - floor) * scores[c]!) / sum + floor / NUM_CLASSES);
    }
  }
  return logs;
}

/** `log p(read | board)`: the squares taken as independent, which they are not quite. */
export function logLikelihood(logs: Float64Array, labels: Labels): number {
  let total = 0;
  for (let square = 0; square < SQUARES; square++) {
    total += logs[square * NUM_CLASSES + labels[square]!]!;
  }
  return total;
}

/** Each square's best `log p` over the classes: what no board can beat there. */
export function squareCeilings(logs: Float64Array): Float64Array {
  const ceilings = new Float64Array(SQUARES).fill(-Infinity);
  for (let i = 0; i < logs.length; i++) {
    const square = Math.floor(i / NUM_CLASSES);
    ceilings[square] = Math.max(ceilings[square]!, logs[i]!);
  }
  return ceilings;
}

/**
 * The most that changing `count` squares of `labels` can raise its log-likelihood:
 * the sum of the `count` largest per-square gains up to the ceilings.
 */
export function largestGains(logs: Float64Array, ceilings: Float64Array, labels: Labels, count: number): number {
  const gains = new Float64Array(SQUARES);
  for (let square = 0; square < SQUARES; square++) {
    gains[square] = ceilings[square]! - logs[square * NUM_CLASSES + labels[square]!]!;
  }
  return gains
    .sort()
    .subarray(SQUARES - count)
    .reduce((sum, gain) => sum + gain, 0);
}

/** The read's own board: the most likely class on each square. */
export function argmaxLabels(dist: SquareDist): Labels {
  const labels = new Uint8Array(SQUARES);
  for (let square = 0; square < SQUARES; square++) {
    const probabilities = dist[Math.floor(square / GRID)]![square % GRID]!;
    let best = 0;
    for (let c = 1; c < NUM_CLASSES; c++) {
      if (probabilities[c]! > probabilities[best]!) best = c;
    }
    labels[square] = best;
  }
  return labels;
}

/** How many squares two boards disagree on. */
export function mismatches(a: Labels, b: Labels): number {
  let count = 0;
  for (let square = 0; square < SQUARES; square++) {
    if (a[square] !== b[square]) count += 1;
  }
  return count;
}
