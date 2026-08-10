import { describe, expect, it } from 'vitest';
import { applyMatrix3, homography, meanLuma, quadGeometry, rectifyQuad } from './homography';
import type { CornerQuad } from '../protocol';

function close(actual: number, expected: number, tolerance: number): void {
  expect(Math.abs(actual - expected) <= tolerance, `got ${actual}, expected ${expected}`).toBe(true);
}

const UNIT: CornerQuad = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1]
];

describe('homography', () => {
  it('recovers a pure scale from the unit square', () => {
    const H = homography(UNIT, [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ]);
    expect(H).not.toBeNull();
    const centre = applyMatrix3(H!, 0.5, 0.5);
    close(centre![0], 5, 1e-9);
    close(centre![1], 5, 1e-9);
  });

  it('maps each named corner onto its counterpart', () => {
    const target: CornerQuad = [
      [536, 918],
      [631, 336],
      [1458, 320],
      [1638, 896]
    ];
    const H = homography(UNIT, target);
    expect(H).not.toBeNull();
    for (const [i, source] of UNIT.entries()) {
      const p = applyMatrix3(H!, source[0], source[1]);
      close(p![0], target[i]![0], 1e-6);
      close(p![1], target[i]![1], 1e-6);
    }
  });

  it('recovers a known perspective map on interior points too', () => {
    // A quad with genuine perspective: the far edge is shorter than the near one.
    const target: CornerQuad = [
      [200, 100],
      [400, 100],
      [500, 400],
      [100, 400]
    ];
    const H = homography(UNIT, target);
    expect(H).not.toBeNull();
    const mid = applyMatrix3(H!, 0.5, 0);
    close(mid![0], 300, 1e-6);
    close(mid![1], 100, 1e-6);
  });

  it('returns null rather than NaN when the SOURCE quad is degenerate', () => {
    // Collinear sources make the DLT system singular. A degenerate *target* does
    // not: mapping a square onto a flattened quad is a perfectly solvable system,
    // and refusing it here would refuse a legitimately foreshortened board.
    expect(
      homography(
        [
          [0, 0],
          [1, 0],
          [2, 0],
          [3, 0]
        ],
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10]
        ]
      )
    ).toBeNull();
  });
});

describe('quadGeometry', () => {
  it('is 1.0 for a square seen from straight above', () => {
    close(
      quadGeometry([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ]).aspect,
      1,
      1e-12
    );
  });

  it('is the short/long ratio, whichever way the quad is labelled', () => {
    const wide = quadGeometry([
      [0, 0],
      [400, 0],
      [400, 100],
      [0, 100]
    ]);
    const tall = quadGeometry([
      [0, 0],
      [100, 0],
      [100, 400],
      [0, 400]
    ]);
    close(wide.aspect, 0.25, 1e-12);
    close(tall.aspect, 0.25, 1e-12);
  });

  it('reports the mean of each pair of opposite edges', () => {
    const g = quadGeometry([
      [0, 0],
      [100, 0],
      [200, 100],
      [0, 100]
    ]);
    close(g.top, 100, 1e-9);
    close(g.bottom, 200, 1e-9);
    close(g.w, 150, 1e-9);
  });

  it('degrades to 1 rather than dividing by zero on a collapsed quad', () => {
    close(quadGeometry([[0, 0], [0, 0], [0, 0], [0, 0]]).aspect, 1, 1e-12);
  });
});

describe('rectifyQuad', () => {
  /** A 4x4 image whose top-left quadrant is red and bottom-right blue. */
  function swatch(): { data: Uint8ClampedArray; width: number; height: number } {
    const width = 4;
    const height = 4;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const red = x < 2 && y < 2;
        data[i] = red ? 255 : 0;
        data[i + 2] = red ? 0 : 255;
        data[i + 3] = 255;
      }
    }
    return { data, width, height };
  }

  it('warps the whole image to the output square', () => {
    const image = swatch();
    const out = rectifyQuad(image, [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4]
    ], 4);
    expect(out).not.toBeNull();
    expect(out!.size).toBe(4);
    // Top-left stays red, bottom-right stays blue.
    expect(out!.data[0]).toBe(255);
    expect(out!.data[(3 * 4 + 3) * 4 + 2]).toBe(255);
  });

  it('leaves out-of-photo pixels black and OPAQUE, so the edge is visible', () => {
    const image = swatch();
    const out = rectifyQuad(image, [
      [100, 100],
      [104, 100],
      [104, 104],
      [100, 104]
    ], 2);
    expect(out).not.toBeNull();
    for (let i = 0; i < out!.data.length; i += 4) {
      expect(out!.data[i]).toBe(0);
      expect(out!.data[i + 3]).toBe(255);
    }
  });

  it('returns null rather than dividing by zero when the output has no size', () => {
    expect(
      rectifyQuad(swatch(), [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ], 0)
    ).toBeNull();
  });
});

describe('meanLuma', () => {
  it('is the mean of the channels, subsampled', () => {
    const width = 8;
    const height = 8;
    const data = new Uint8ClampedArray(width * height * 4).fill(120);
    close(meanLuma({ data, width, height }), 120, 1e-9);
  });

  it('is 0 for an empty image rather than NaN', () => {
    close(meanLuma({ data: new Uint8ClampedArray(0), width: 0, height: 0 }), 0, 0);
  });
});
