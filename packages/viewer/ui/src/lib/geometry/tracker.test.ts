/**
 * The executable checks for the plane tracker — the Vitest port of
 * `packages/viewer/tests/tracker_checks.mjs` (which pytest drove through `node`).
 *
 * Every check that existed there exists here, with the same scenes and the same
 * tolerances. The tracker is the one piece of the camera page that is mathematics
 * rather than markup, so it is tested by RUNNING it: each case is a synthetic scene
 * with a KNOWN answer — a homography we chose, applied to a texture we generated,
 * recovered from pixels alone. String-asserting the source cannot tell a working
 * RANSAC from a broken one; these can.
 */

import { describe, expect, it } from 'vitest';
import {
  PlaneTracker,
  applyHomography,
  buildPyramid,
  homographyFromPairs,
  isSaneQuad,
  ransacHomography,
  toGray,
  trackPoints,
  unitSquareToQuad,
  type Homography3
} from './tracker';

/** `|actual - expected| <= tolerance`, reported the way the original checks did. */
function close(actual: number, expected: number, tolerance: number): void {
  expect(
    Math.abs(actual - expected) <= tolerance,
    `got ${actual}, expected ${expected} +/- ${tolerance}`
  ).toBe(true);
}

function mapped(H: Homography3, x: number, y: number): [number, number] {
  const p = applyHomography(H, x, y);
  if (p === null) throw new Error('point mapped to infinity');
  return p;
}

// ---- a smooth, non-repeating texture -----------------------------------------
// Incommensurate frequencies: locally rich enough for Lucas-Kanade to lock onto, and
// quasi-periodic rather than periodic so a window cannot match the wrong neighbour.
// The contrast is deliberately board-like (the shortest wavelength is ~18px, about a
// square's worth at the tracker's working resolution) — a flatter field would be
// rejected by the tracker's own flat-region gate, and rightly so.
function texture(x: number, y: number): number {
  return (
    0.5 +
    0.22 * Math.sin(0.21 * x + 0.13 * y) +
    0.16 * Math.sin(0.34 * x - 0.19 * y + 1.7) +
    0.1 * Math.sin(0.061 * x + 0.083 * y + 0.6) +
    0.06 * Math.sin(0.017 * x - 0.029 * y + 2.3)
  );
}

function invert3(m: readonly number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m as [
    number, number, number, number, number, number, number, number, number
  ];
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    A, c * h - b * i, b * f - c * e,
    B, a * i - c * g, c * d - a * f,
    C, b * g - a * h, a * e - b * d
  ].map((v) => v / det);
}

/**
 * The image you get by looking at `texture` through `H`: every output pixel samples
 * the texture at its pre-image, so a scene point q lands at H(q).
 */
function render(H: readonly number[], width: number, height: number): Float32Array {
  const inverse = invert3(H);
  const gray = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = mapped(inverse, x, y);
      gray[y * width + x] = texture(p[0], p[1]);
    }
  }
  return gray;
}

function compose(m: readonly number[], n: readonly number[]): number[] {
  const out = new Array<number>(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = m[r * 3]! * n[c]! + m[r * 3 + 1]! * n[3 + c]! + m[r * 3 + 2]! * n[6 + c]!;
    }
  }
  return out.map((v) => v / out[8]!);
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const FRAME_W = 320;
const FRAME_H = 240;
const CORNERS = [
  [0.24, 0.26],
  [0.76, 0.22],
  [0.8, 0.74],
  [0.2, 0.7]
] as const;

function expectedCorners(motion: readonly number[]): [number, number][] {
  return CORNERS.map(([x, y]) => {
    const p = mapped(motion, x * FRAME_W, y * FRAME_H);
    return [p[0] / FRAME_W, p[1] / FRAME_H] as [number, number];
  });
}

// ---- 1. the linear algebra ----------------------------------------------------

describe('homography from pairs', () => {
  const truth = [1.04, -0.06, 7.5, 0.05, 0.98, -3.2, 1.2e-4, -8e-5, 1];
  const src: number[] = [];
  const dst: number[] = [];
  for (const [x, y] of [
    [10, 12], [300, 20], [280, 220], [30, 200], [150, 90], [90, 170]
  ] as const) {
    const p = mapped(truth, x, y);
    src.push(x, y);
    dst.push(p[0], p[1]);
  }
  const fromFour = homographyFromPairs(
    new Float64Array(src.slice(0, 8)),
    new Float64Array(dst.slice(0, 8))
  );
  const fromAll = homographyFromPairs(new Float64Array(src), new Float64Array(dst));

  it('solves from 4 exact pairs', () => {
    expect(fromFour).not.toBeNull();
  });

  it('least-squares over 6 pairs', () => {
    expect(fromAll).not.toBeNull();
  });

  it.each([
    ['4-point', () => fromFour],
    ['6-point', () => fromAll]
  ])('recovers the truth matrix (%s)', (_label, get) => {
    const H = get();
    expect(H).not.toBeNull();
    for (let i = 0; i < 9; i++) close(H![i]!, truth[i]!, 1e-6);
  });

  it('refuses fewer than 4 pairs', () => {
    expect(
      homographyFromPairs(new Float64Array([0, 0, 1, 0, 1, 1]), new Float64Array(6))
    ).toBeNull();
  });
});

// ---- 2. RANSAC survives the scene, not just noise -----------------------------
// A third of the correspondences are gross outliers: the points that landed on a
// tall piece or on a hand crossing the board.

describe('ransac homography', () => {
  const truth = [0.99, 0.03, -4, -0.04, 1.01, 6, -9e-5, 5e-5, 1];
  const n = 60;
  const src = new Float64Array(n * 2);
  const dst = new Float64Array(n * 2);
  let rng = 12345;
  const random = () => ((rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = 0; i < n; i++) {
    const x = 20 + random() * 280;
    const y = 20 + random() * 200;
    const p = mapped(truth, x, y);
    src[2 * i] = x;
    src[2 * i + 1] = y;
    if (i % 3 === 0) {
      // an outlier, moving its own way entirely
      dst[2 * i] = p[0] + 30 + random() * 40;
      dst[2 * i + 1] = p[1] - 25 - random() * 40;
    } else {
      dst[2 * i] = p[0] + (random() - 0.5) * 0.4;
      dst[2 * i + 1] = p[1] + (random() - 0.5) * 0.4;
    }
  }
  const fit = ransacHomography(src, dst);

  it('fits', () => {
    expect(fit).not.toBeNull();
  });

  it('rejects the outlier third', () => {
    expect(fit!.ratio > 0.6 && fit!.ratio < 0.75, `inlier ratio ${fit!.ratio}`).toBe(true);
  });

  it.each([
    [40, 40],
    [260, 60],
    [200, 200]
  ])('maps (%i,%i) where the truth does', (x, y) => {
    const got = mapped(fit!.H, x, y);
    const want = mapped(truth, x, y);
    close(got[0], want[0], 0.5);
    close(got[1], want[1], 0.5);
  });

  // Determinism is a requirement, not an accident: same input, same answer, always.
  it('is deterministic', () => {
    expect(Array.from(ransacHomography(src, dst)!.H)).toEqual(Array.from(fit!.H));
  });
});

// ---- 3. the quad sanity gate --------------------------------------------------

describe('quad sanity gate', () => {
  const square = [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
    [0.2, 0.8]
  ] as const;

  it('passes a plain quad', () => {
    expect(isSaneQuad(square)).toBe(true);
  });

  it('fails a bow-tie', () => {
    expect(
      isSaneQuad([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.8, 0.8]
      ])
    ).toBe(false);
  });

  it('fails a sliver', () => {
    expect(
      isSaneQuad([
        [0.2, 0.5],
        [0.8, 0.5],
        [0.8, 0.505],
        [0.2, 0.505]
      ])
    ).toBe(false);
  });

  it('fails far off-frame', () => {
    expect(
      isSaneQuad([
        [3, 3],
        [4, 3],
        [4, 4],
        [3, 4]
      ])
    ).toBe(false);
  });

  it('fails on NaN', () => {
    expect(
      isSaneQuad([
        [NaN, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8]
      ])
    ).toBe(false);
  });

  it('needs four corners', () => {
    expect(isSaneQuad(square.slice(0, 3))).toBe(false);
  });

  it('maps the unit square onto the quad centre', () => {
    const H = unitSquareToQuad(square);
    expect(H).not.toBeNull();
    const centre = mapped(H!, 0.5, 0.5);
    close(centre[0], 0.5, 1e-9);
    close(centre[1], 0.5, 1e-9);
  });
});

// ---- 4. optical flow on real pixels -------------------------------------------

describe('pyramidal Lucas-Kanade', () => {
  const shift = [1, 0, 3.5, 0, 1, -2.25, 0, 0, 1];
  const a = buildPyramid(render(IDENTITY, FRAME_W, FRAME_H), FRAME_W, FRAME_H, 3);
  const b = buildPyramid(render(shift, FRAME_W, FRAME_H), FRAME_W, FRAME_H, 3);
  const points = new Float64Array([80, 60, 160, 120, 240, 180, 120, 200]);
  const { points: moved, ok } = trackPoints(a, b, points);

  it.each([0, 1, 2, 3])('recovers the shift at point %i', (i) => {
    expect(ok[i]).toBe(1);
    close(moved[2 * i]! - points[2 * i]!, 3.5, 0.25);
    close(moved[2 * i + 1]! - points[2 * i + 1]!, -2.25, 0.25);
  });
});

// ---- 5. the tracker end to end ------------------------------------------------

describe('PlaneTracker over a sustained camera move', () => {
  // A plausible single frame of movement at 20 fps: a few degrees of roll, a little
  // zoom, a few pixels of pan, and genuine perspective change from stepping sideways.
  const angle = 0.045;
  const cos = Math.cos(angle) * 1.015;
  const sin = Math.sin(angle) * 1.015;
  const motion = [cos, -sin, 5.5, sin, cos, -3.0, 1.4e-4, -6e-5, 1];

  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  const first = tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  const locked = tracker.update(render(motion, FRAME_W, FRAME_H));

  let cumulative: readonly number[] = motion;
  let state = locked;
  for (let step = 2; step <= 6; step++) {
    cumulative = compose(motion, cumulative);
    state = tracker.update(render(cumulative, FRAME_W, FRAME_H));
  }

  it('only seeds on the first frame', () => {
    expect(first.state).toBe('seeded');
  });

  it('locks on the second frame', () => {
    expect(locked.state, `state ${locked.state}`).toBe('locked');
  });

  it('has most points agree', () => {
    expect(locked.ratio > 0.8, `ratio ${locked.ratio}`).toBe(true);
  });

  it.each([0, 1, 2, 3])('moves corner %i to where the homography sends it', (i) => {
    const want = expectedCorners(motion);
    close(locked.corners![i]![0], want[i]![0], 0.006);
    close(locked.corners![i]![1], want[i]![1], 0.006);
  });

  // ...and it keeps up over a sustained move, which is what an orbit around the
  // board actually is. Drift is allowed to accumulate; running away is not.
  it('is still locked after six frames', () => {
    expect(state.state, `state ${state.state}`).toBe('locked');
  });

  it.each([0, 1, 2, 3])('has bounded drift at corner %i after six frames', (i) => {
    const far = expectedCorners(cumulative);
    close(state.corners![i]![0], far[i]![0], 0.02);
    close(state.corners![i]![1], far[i]![1], 0.02);
  });
});

// ---- 6. losing the board freezes the quad rather than inventing one -----------

describe('losing the board', () => {
  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  const before = JSON.stringify(tracker.corners);

  // The lens is covered: uniform gray, no texture to track at all.
  const blank = new Float32Array(FRAME_W * FRAME_H).fill(0.5);
  const lost = tracker.update(blank);

  // The hand moves away and the board is back where it was: the tracker recovers by
  // itself, because it keeps tracking FROM the frozen quad.
  tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  const back = tracker.update(render([1, 0, 2, 0, 1, 1, 0, 0, 1], FRAME_W, FRAME_H));

  it('reports the loss', () => {
    expect(lost.state, `state ${lost.state}`).toBe('lost');
  });

  it('freezes the last good quad', () => {
    expect(before).toBe(JSON.stringify(lost.corners));
  });

  it('recovers once the view returns', () => {
    expect(back.state, `state ${back.state}`).toBe('locked');
  });
});

// ---- 7. grayscale matches the edge's weights ----------------------------------

describe('toGray', () => {
  it('uses cv2 BGR2GRAY weights', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    const gray = toGray(rgba, new Float32Array(3));
    close(gray[0]!, 0.299, 1e-6);
    close(gray[1]!, 0.587, 1e-6);
    close(gray[2]!, 0.114, 1e-6);
  });
});

// ---- 8. the seed lattice is the BOARD's lattice -------------------------------
// Not decoration: an arbitrary NxN grid misses the square corners, which are a chess
// board's only high-gradient points, and it fails SILENTLY — measured on a real board
// frame, an 8x8 grid kept 18 of its points where the 9x9 lattice kept 31.

describe('the seed lattice', () => {
  it('is the 9x9 grid of square corners at one subdivision', () => {
    expect(new PlaneTracker({ width: 100, height: 100, subdivisions: 1 }).grid).toBe(9);
  });

  it('is 17x17 at two subdivisions', () => {
    expect(new PlaneTracker({ width: 100, height: 100, subdivisions: 2 }).grid).toBe(17);
  });

  // With an 8x8 frame and the unit square as the quad, every seed lands on an integer
  // exactly when it lands on a square corner.
  const tracker = new PlaneTracker({ width: 8, height: 8, subdivisions: 1 });
  const seeded = tracker.seedPoints([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
  ]);

  it('has 81 seeds', () => {
    expect(seeded!.length).toBe(81 * 2);
  });

  it('puts every seed on a square corner', () => {
    expect(Array.from(seeded!).every((v) => Math.abs(v - Math.round(v)) < 1e-9)).toBe(true);
  });
});

// ---- 9. the caller's capture buffer is reused, and that must not matter ---------
// The regression that motivated this check was found in a browser, not here: a page
// captures video into ONE Float32Array per frame, and an earlier pyramid kept a
// reference to it rather than a copy. One frame later the "previous" image held the
// current pixels, so every point measured itself, converged on zero displacement, and
// the tracker reported a confident lock while the quad never moved at all.

describe('a reused capture buffer', () => {
  const motion = [1, 0, 6, 0, 1, 3.5, 0, 0, 1];
  const scratch = new Float32Array(FRAME_W * FRAME_H); // ONE buffer, as a page has
  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  scratch.set(render(IDENTITY, FRAME_W, FRAME_H));
  tracker.update(scratch);
  scratch.set(render(motion, FRAME_W, FRAME_H)); // same buffer, new pixels
  const moved = tracker.update(scratch);

  it('still locks', () => {
    expect(moved.state, `state ${moved.state}`).toBe('locked');
  });

  it.each([0, 1, 2, 3])('follows corner %i', (i) => {
    const want = expectedCorners(motion);
    close(moved.corners![i]![0], want[i]![0], 0.006);
    close(moved.corners![i]![1], want[i]![1], 0.006);
  });

  // The point of the check: the quad actually MOVED, rather than reporting a healthy
  // fit while standing still.
  it('moved the quad at all', () => {
    expect(Math.abs(moved.corners![0]![0] - CORNERS[0][0]) * FRAME_W > 3).toBe(true);
  });
});
