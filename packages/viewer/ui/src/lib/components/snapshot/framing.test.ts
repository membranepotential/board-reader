import { describe, expect, it } from 'vitest';
import {
  EMPTY_FRAMING,
  absorbedFraming,
  autoRotateNote,
  clearedFraming,
  cornerHint,
  cornerProvenance,
  detectedFraming,
  failedDetection,
  labelAt,
  nextLabel,
  pointCount,
  quadOf,
  rotatedFraming,
  withMovedPoint,
  withPoint,
  withoutLastPoint
} from './framing';
import { orderedCorners } from '$lib/geometry/corners';
import type { CornerQuad, Point } from '$lib/protocol';

const QUAD: CornerQuad = [
  [10, 10],
  [90, 12],
  [92, 88],
  [8, 86]
];

/** The hint as one plain string, for asserting on wording. */
const hintText = (segments: readonly { text: string }[]): string =>
  segments.map((s) => s.text).join('');

describe('placing corners', () => {
  it('is not a placed quad until the fourth point lands', () => {
    let framing = EMPTY_FRAMING;
    for (const point of QUAD.slice(0, 3) as readonly Point[]) {
      framing = withPoint(framing, point);
      expect(quadOf(framing)).toBeNull();
    }
    framing = withPoint(framing, QUAD[3]);
    expect(quadOf(framing)).toEqual(QUAD);
  });

  it('ignores a fifth tap', () => {
    const framing = QUAD.reduce<ReturnType<typeof withPoint>>(
      (f, p) => withPoint(f, p),
      EMPTY_FRAMING
    );
    expect(pointCount(withPoint(framing, [1, 1]))).toBe(4);
  });

  it('makes the labelling the operator’s the moment they touch a corner', () => {
    expect(EMPTY_FRAMING.edited).toBe(false);
    expect(withPoint(EMPTY_FRAMING, [1, 1]).edited).toBe(true);
    expect(withMovedPoint(detectedFraming(QUAD), 0, [5, 5]).edited).toBe(true);
    expect(rotatedFraming(detectedFraming(QUAD)).edited).toBe(true);
  });

  it('undo drops the last point and re-opens the quad', () => {
    const framing = withoutLastPoint(detectedFraming(QUAD));
    expect(quadOf(framing)).toBeNull();
    expect(pointCount(framing)).toBe(3);
  });

  it('reset clears the points, the rotation and the failed-detect state', () => {
    const framing = clearedFraming(rotatedFraming(failedDetection()));
    expect(pointCount(framing)).toBe(0);
    expect(framing.rot).toBe(0);
    expect(framing.detectFailed).toBe(false);
  });
});

describe('labelling', () => {
  it('a rotation relabels and never moves a point', () => {
    const framing = rotatedFraming(detectedFraming(QUAD));
    expect(framing.placement.points).toEqual(QUAD);
    expect(labelAt(framing, 0)).toBe('h8');
    expect(labelAt(framing, 3)).toBe('a8');
  });

  it('four rotations are the identity', () => {
    let framing = detectedFraming(QUAD);
    for (let i = 0; i < 4; i++) framing = rotatedFraming(framing);
    expect(framing.rot).toBe(0);
  });

  it('asks for the next unplaced corner by its rotated name', () => {
    expect(nextLabel(EMPTY_FRAMING)).toBe('a8');
    expect(nextLabel(withPoint(EMPTY_FRAMING, [1, 1]))).toBe('h8');
  });

  it('a node rotation is absorbed as a cyclic shift of the ordered corners', () => {
    const framing = detectedFraming(QUAD);
    const absorbed = absorbedFraming(framing, 90);
    const before = orderedCorners(QUAD, framing.rot);
    const after = orderedCorners(QUAD, absorbed.rot);
    // rot90(grid, k) reindexes the lattice exactly as oc'[i] = oc[(i + k) % 4].
    expect(after).toEqual([before[1], before[2], before[3], before[0]]);
    expect(absorbed.adopted).toBe(1);
    // Still auto-detected: nobody asserted an orientation, so later reads keep asking
    // the node to auto-rotate.
    expect(absorbed.edited).toBe(false);
  });

  it('absorbing nothing changes nothing', () => {
    const framing = detectedFraming(QUAD);
    expect(absorbedFraming(framing, 0)).toBe(framing);
  });

  it('a hand rotation drops the node’s claim on the labelling', () => {
    const framing = rotatedFraming(absorbedFraming(detectedFraming(QUAD), 180));
    expect(framing.adopted).toBe(0);
    expect(autoRotateNote(framing)).toContain('your labelling wins');
  });
});

describe('what the page says about the corners', () => {
  it('names the corner being asked for, and admits a failed detection', () => {
    expect(hintText(cornerHint(EMPTY_FRAMING))).toContain('Tap the a8 corner');
    expect(hintText(cornerHint(EMPTY_FRAMING))).toContain('(0/4)');
    expect(hintText(cornerHint(failedDetection()))).toContain("Auto-detect didn't find the board");
  });

  it('switches to checking the labels once four are placed', () => {
    expect(hintText(cornerHint(detectedFraming(QUAD)))).toContain('Check the labels');
  });

  it('emphasises exactly the corner name', () => {
    const strong = cornerHint(withPoint(EMPTY_FRAMING, [1, 1])).filter((s) => s.strong);
    expect(strong.map((s) => s.text)).toEqual(['h8']);
  });

  it('states where the corners came from', () => {
    expect(cornerProvenance(detectedFraming(QUAD))).toBe('auto-detected');
    expect(cornerProvenance(withMovedPoint(detectedFraming(QUAD), 1, [1, 1]))).toBe(
      'auto-detected, edited'
    );
    expect(cornerProvenance(withPoint(failedDetection(), [1, 1]))).toBe('hand-placed');
  });

  it('distinguishes an unverified guess from an adopted node rotation', () => {
    expect(autoRotateNote(detectedFraming(QUAD))).toContain('unverified guess');
    expect(autoRotateNote(absorbedFraming(detectedFraming(QUAD), 270))).toContain(
      'rotated the labels by 270 degrees'
    );
  });
});
