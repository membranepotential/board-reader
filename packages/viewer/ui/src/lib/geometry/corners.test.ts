import { describe, expect, it } from 'vitest';
import {
  CORNER_ORDER,
  CROP_HEADROOM,
  CROP_MARGIN,
  absorbRotation,
  asRotation,
  checkQuadInFrame,
  checkQuadShape,
  cornersInCrop,
  cropBox,
  labelOf,
  lowAngleWarning,
  nextRotation,
  orderedCorners,
  quarterTurnsOf,
  type LabelRotation
} from './corners';
import type { CornerQuad } from '../protocol';

const SQUARE: CornerQuad = [
  [100, 100],
  [500, 100],
  [500, 500],
  [100, 500]
];

describe('label rotation', () => {
  it('names the storage order a8,h8,h1,a1 at rotation 0', () => {
    expect([0, 1, 2, 3].map((i) => labelOf(i, 0))).toEqual([...CORNER_ORDER]);
  });

  it('slides the labels round the quad', () => {
    expect([0, 1, 2, 3].map((i) => labelOf(i, 1))).toEqual(['h8', 'h1', 'a1', 'a8']);
  });

  it('is the inverse of orderedCorners at every rotation', () => {
    for (const rot of [0, 1, 2, 3] as LabelRotation[]) {
      const ordered = orderedCorners(SQUARE, rot);
      // The corner ordered into slot k is the one whose label is CORNER_ORDER[k].
      for (let k = 0; k < 4; k++) {
        const storageIndex = SQUARE.findIndex((p) => p === ordered[k]);
        expect(labelOf(storageIndex, rot)).toBe(CORNER_ORDER[k]);
      }
    }
  });

  it('never moves a point — a rotation is a relabelling', () => {
    for (const rot of [0, 1, 2, 3] as LabelRotation[]) {
      expect([...orderedCorners(SQUARE, rot)].sort()).toEqual([...SQUARE].sort());
    }
  });

  it('cycles through four rotations and back', () => {
    let rot: LabelRotation = 0;
    for (let i = 0; i < 4; i++) rot = nextRotation(rot);
    expect(rot).toBe(0);
  });

  it('folds any integer into the four legal rotations', () => {
    expect(asRotation(-1)).toBe(3);
    expect(asRotation(7)).toBe(3);
    expect(asRotation(4)).toBe(0);
  });

  it('absorbs a node rotation as rot -= k, which is what undoes rot90', () => {
    // A read that came back turned 90 degrees must leave the ordered corners shifted
    // by exactly one, so the handles, the crop and the board share one orientation.
    const k = quarterTurnsOf(90);
    expect(k).toBe(1);
    const before = orderedCorners(SQUARE, 0);
    const after = orderedCorners(SQUARE, absorbRotation(0, k));
    expect(after[0]).toEqual(before[1]);
    expect(after[1]).toEqual(before[2]);
  });

  it('reads degrees the way the node reports them', () => {
    expect(quarterTurnsOf(0)).toBe(0);
    expect(quarterTurnsOf(180)).toBe(2);
    expect(quarterTurnsOf(270)).toBe(3);
    expect(quarterTurnsOf(-90)).toBe(3);
  });
});

describe('cropBox', () => {
  const frame = { width: 1280, height: 720 };

  it('keeps a margin all round', () => {
    const box = cropBox([[400, 300], [800, 300], [800, 500], [400, 500]], frame);
    expect(box.x).toBe(Math.floor(400 - CROP_MARGIN * 400));
    expect(box.x + box.w).toBe(Math.ceil(800 + CROP_MARGIN * 400));
  });

  it('keeps considerably MORE above the top edge, because pieces are tall', () => {
    const box = cropBox([[400, 300], [800, 300], [800, 500], [400, 500]], frame);
    const above = 300 - box.y;
    const below = box.y + box.h - 500;
    expect(above).toBe(Math.ceil((CROP_MARGIN + CROP_HEADROOM) * 200));
    expect(above).toBeGreaterThan(below);
  });

  it('clamps to the frame rather than cropping outside it', () => {
    const box = cropBox([[0, 0], [1280, 0], [1280, 720], [0, 720]], frame);
    expect(box).toEqual({ x: 0, y: 0, w: 1280, h: 720 });
  });

  it('is never smaller than a pixel, even on a degenerate quad', () => {
    const box = cropBox([[5, 5], [5, 5], [5, 5], [5, 5]], frame);
    expect(box.w).toBeGreaterThanOrEqual(1);
    expect(box.h).toBeGreaterThanOrEqual(1);
  });
});

describe('cornersInCrop', () => {
  it('rebases the quad into the pixels of the image actually sent', () => {
    const box = { x: 100, y: 100, w: 400, h: 400 };
    const rebased = cornersInCrop(SQUARE, box, { w: 200, h: 200 });
    expect(rebased).toEqual([
      [0, 0],
      [200, 0],
      [200, 200],
      [0, 200]
    ]);
  });
});

describe('quad validation (mirror of board_reader_edge.quadcheck)', () => {
  it('accepts a plausible board outline', () => {
    expect(checkQuadShape(SQUARE)).toBeNull();
  });

  it('catches a repeated point before anything else', () => {
    const defect = checkQuadShape([[100, 100], [100, 100], [500, 500], [100, 500]]);
    expect(defect?.kind).toBe('coincident_points');
  });

  it('catches the classic a8,h8,a1,h1 slip as a bowtie', () => {
    const defect = checkQuadShape([[100, 100], [500, 100], [100, 500], [500, 500]]);
    expect(defect?.kind).toBe('self_intersecting');
    expect(defect?.message).toContain('a8,h8,h1,a1');
  });

  it('calls a too-small quad degenerate, not merely odd', () => {
    const defect = checkQuadShape([[0, 0], [10, 0], [10, 10], [0, 10]]);
    expect(defect?.kind).toBe('degenerate');
  });

  it('calls near-collinear points a sliver', () => {
    const defect = checkQuadShape([[0, 0], [2000, 0], [2000, 4], [0, 4]]);
    expect(defect?.kind).toBe('degenerate');
    expect(defect?.message).toContain('sliver');
  });

  it('refuses a non-convex quad', () => {
    const defect = checkQuadShape([[0, 0], [500, 0], [100, 100], [0, 500]]);
    expect(defect?.kind).toBe('not_convex');
  });

  it('refuses a non-finite coordinate', () => {
    expect(checkQuadShape([[NaN, 0], [500, 0], [500, 500], [0, 500]])?.kind).toBe('not_a_quad');
  });

  it('accepts a corner slightly off-frame — a cropped near corner is a real case', () => {
    expect(checkQuadInFrame(SQUARE, { width: 512, height: 512 })).toBeNull();
    expect(checkQuadInFrame([[-20, -20], [500, 100], [500, 500], [100, 500]], {
      width: 512,
      height: 512
    })).toBeNull();
  });

  it('refuses corners measured at another resolution', () => {
    const defect = checkQuadInFrame(SQUARE, { width: 128, height: 128 });
    expect(defect?.kind).toBe('out_of_frame');
    expect(defect?.message).toContain('128x128');
  });
});

describe('lowAngleWarning', () => {
  it('says nothing about a top-down quad', () => {
    expect(lowAngleWarning(SQUARE)).toBeNull();
  });

  it('warns — but does not refuse — a near board-level camera', () => {
    // A quad 400 wide and 100 tall projects to aspect 0.25, well under 0.62.
    const flat: CornerQuad = [
      [100, 100],
      [500, 100],
      [500, 200],
      [100, 200]
    ];
    const warning = lowAngleWarning(flat);
    expect(warning).not.toBeNull();
    expect(warning).toContain('0.25');
    expect(checkQuadShape(flat)).toBeNull();
  });
});
