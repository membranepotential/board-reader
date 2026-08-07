/* Executable checks for static/tracker.js — the one piece of the camera page that is
   mathematics rather than markup, so it is tested by RUNNING it, not by asserting on
   its source. Driven from test_tracker_js.py, which copies tracker.js in beside this
   file as ./tracker.mjs (the repo has no package.json, so a bare .js would be read as
   CommonJS and refuse to export).

   Every check is a synthetic scene with a KNOWN answer: a homography we chose, applied
   to a texture we generated, recovered from pixels alone. */

import {
  PlaneTracker, applyHomography, buildPyramid, homographyFromPairs, isSaneQuad,
  ransacHomography, toGray, trackPoints, unitSquareToQuad,
} from "./tracker.mjs";

const failures = [];
let checks = 0;

function check(name, condition, detail = "") {
  checks++;
  if (!condition) failures.push(`${name}${detail ? " — " + detail : ""}`);
}

function close(name, actual, expected, tolerance) {
  check(name, Math.abs(actual - expected) <= tolerance,
        `got ${actual}, expected ${expected} +/- ${tolerance}`);
}

// ---- a smooth, non-repeating texture -----------------------------------------
// Incommensurate frequencies: locally rich enough for Lucas-Kanade to lock onto, and
// quasi-periodic rather than periodic so a window cannot match the wrong neighbour.
// The contrast is deliberately board-like (the shortest wavelength is ~18px, about a
// square's worth at the tracker's working resolution) — a flatter field would be
// rejected by the tracker's own flat-region gate, and rightly so.
function texture(x, y) {
  return 0.5
    + 0.22 * Math.sin(0.210 * x + 0.130 * y)
    + 0.16 * Math.sin(0.340 * x - 0.190 * y + 1.7)
    + 0.10 * Math.sin(0.061 * x + 0.083 * y + 0.6)
    + 0.06 * Math.sin(0.017 * x - 0.029 * y + 2.3);
}

function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const inv = [
    A, c * h - b * i, b * f - c * e,
    B, a * i - c * g, c * d - a * f,
    C, b * g - a * h, a * e - b * d,
  ].map((v) => v / det);
  return inv;
}

/** The image you get by looking at `texture` through `H`: every output pixel samples
 *  the texture at its pre-image, so a scene point q lands at H(q). */
function render(H, width, height) {
  const inverse = invert3(H);
  const gray = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = applyHomography(inverse, x, y);
      gray[y * width + x] = texture(p[0], p[1]);
    }
  }
  return gray;
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// ---- 1. the linear algebra ----------------------------------------------------
{
  const truth = [1.04, -0.06, 7.5, 0.05, 0.98, -3.2, 1.2e-4, -8e-5, 1];
  const src = [], dst = [];
  for (const [x, y] of [[10, 12], [300, 20], [280, 220], [30, 200], [150, 90], [90, 170]]) {
    const p = applyHomography(truth, x, y);
    src.push(x, y);
    dst.push(p[0], p[1]);
  }
  const fromFour = homographyFromPairs(new Float64Array(src.slice(0, 8)),
                                       new Float64Array(dst.slice(0, 8)));
  const fromAll = homographyFromPairs(new Float64Array(src), new Float64Array(dst));
  check("homography: 4 exact pairs solve", fromFour !== null);
  check("homography: 6 pairs least-squares", fromAll !== null);
  for (const [label, H] of [["4-point", fromFour], ["6-point", fromAll]]) {
    for (let i = 0; i < 9; i++) close(`homography ${label} h${i}`, H[i], truth[i], 1e-6);
  }
  check("homography: refuses fewer than 4 pairs",
        homographyFromPairs(new Float64Array([0, 0, 1, 0, 1, 1]), new Float64Array(6)) === null);
}

// ---- 2. RANSAC survives the scene, not just noise -----------------------------
// A third of the correspondences are gross outliers: the points that landed on a
// tall piece or on a hand crossing the board.
{
  const truth = [0.99, 0.03, -4, -0.04, 1.01, 6, -9e-5, 5e-5, 1];
  const n = 60;
  const src = new Float64Array(n * 2);
  const dst = new Float64Array(n * 2);
  let rng = 12345;
  const random = () => ((rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = 0; i < n; i++) {
    const x = 20 + random() * 280, y = 20 + random() * 200;
    const p = applyHomography(truth, x, y);
    src[2 * i] = x; src[2 * i + 1] = y;
    if (i % 3 === 0) {                       // an outlier, moving its own way entirely
      dst[2 * i] = p[0] + 30 + random() * 40;
      dst[2 * i + 1] = p[1] - 25 - random() * 40;
    } else {
      dst[2 * i] = p[0] + (random() - 0.5) * 0.4;
      dst[2 * i + 1] = p[1] + (random() - 0.5) * 0.4;
    }
  }
  const fit = ransacHomography(src, dst);
  check("ransac: fits", fit !== null);
  check("ransac: rejects the outlier third", fit.ratio > 0.6 && fit.ratio < 0.75,
        `inlier ratio ${fit?.ratio}`);
  for (const [x, y] of [[40, 40], [260, 60], [200, 200]]) {
    const got = applyHomography(fit.H, x, y);
    const want = applyHomography(truth, x, y);
    close(`ransac: maps (${x},${y}).x`, got[0], want[0], 0.5);
    close(`ransac: maps (${x},${y}).y`, got[1], want[1], 0.5);
  }
  // Determinism is a requirement, not an accident: same input, same answer, always.
  check("ransac: deterministic",
        JSON.stringify(Array.from(ransacHomography(src, dst).H)) === JSON.stringify(Array.from(fit.H)));
}

// ---- 3. the quad sanity gate --------------------------------------------------
{
  const square = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]];
  check("sane: a plain quad passes", isSaneQuad(square));
  check("sane: a bow-tie fails", !isSaneQuad([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]));
  check("sane: a sliver fails", !isSaneQuad([[0.2, 0.5], [0.8, 0.5], [0.8, 0.505], [0.2, 0.505]]));
  check("sane: far off-frame fails", !isSaneQuad([[3, 3], [4, 3], [4, 4], [3, 4]]));
  check("sane: NaN fails", !isSaneQuad([[NaN, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]));
  check("sane: needs four corners", !isSaneQuad(square.slice(0, 3)));
  const H = unitSquareToQuad(square);
  const centre = applyHomography(H, 0.5, 0.5);
  close("unit square maps to the quad centre.x", centre[0], 0.5, 1e-9);
  close("unit square maps to the quad centre.y", centre[1], 0.5, 1e-9);
}

// ---- 4. optical flow on real pixels -------------------------------------------
const FRAME_W = 320, FRAME_H = 240;
{
  const shift = [1, 0, 3.5, 0, 1, -2.25, 0, 0, 1];
  const a = buildPyramid(render(IDENTITY, FRAME_W, FRAME_H), FRAME_W, FRAME_H, 3);
  const b = buildPyramid(render(shift, FRAME_W, FRAME_H), FRAME_W, FRAME_H, 3);
  const points = new Float64Array([80, 60, 160, 120, 240, 180, 120, 200]);
  const { points: moved, ok } = trackPoints(a, b, points);
  for (let i = 0; i < 4; i++) {
    check(`flow: point ${i} tracked`, ok[i] === 1);
    close(`flow: point ${i} dx`, moved[2 * i] - points[2 * i], 3.5, 0.25);
    close(`flow: point ${i} dy`, moved[2 * i + 1] - points[2 * i + 1], -2.25, 0.25);
  }
}

// ---- 5. the tracker end to end ------------------------------------------------
const CORNERS = [[0.24, 0.26], [0.76, 0.22], [0.80, 0.74], [0.20, 0.70]];

function expected(motion) {
  return CORNERS.map(([x, y]) => {
    const p = applyHomography(motion, x * FRAME_W, y * FRAME_H);
    return [p[0] / FRAME_W, p[1] / FRAME_H];
  });
}

{
  // A plausible single frame of movement at 20 fps: a few degrees of roll, a little
  // zoom, a few pixels of pan, and genuine perspective change from stepping sideways.
  const angle = 0.045;
  const cos = Math.cos(angle) * 1.015, sin = Math.sin(angle) * 1.015;
  const motion = [cos, -sin, 5.5, sin, cos, -3.0, 1.4e-4, -6e-5, 1];

  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  const first = tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  check("tracker: the first frame only seeds", first.state === "seeded");

  const locked = tracker.update(render(motion, FRAME_W, FRAME_H));
  check("tracker: locks on the second frame", locked.state === "locked", `state ${locked.state}`);
  check("tracker: most points agree", locked.ratio > 0.8, `ratio ${locked.ratio}`);
  const want = expected(motion);
  for (let i = 0; i < 4; i++) {
    close(`tracker: corner ${i}.x`, locked.corners[i][0], want[i][0], 0.006);
    close(`tracker: corner ${i}.y`, locked.corners[i][1], want[i][1], 0.006);
  }

  // ...and it keeps up over a sustained move, which is what an orbit around the
  // board actually is. Drift is allowed to accumulate; running away is not.
  let cumulative = motion;
  let state = locked;
  for (let step = 2; step <= 6; step++) {
    cumulative = compose(motion, cumulative);
    state = tracker.update(render(cumulative, FRAME_W, FRAME_H));
  }
  check("tracker: still locked after six frames", state.state === "locked", `state ${state.state}`);
  const far = expected(cumulative);
  for (let i = 0; i < 4; i++) {
    close(`tracker: drift after six frames, corner ${i}.x`, state.corners[i][0], far[i][0], 0.02);
    close(`tracker: drift after six frames, corner ${i}.y`, state.corners[i][1], far[i][1], 0.02);
  }
}

function compose(m, n) {
  const out = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = m[r * 3] * n[c] + m[r * 3 + 1] * n[3 + c] + m[r * 3 + 2] * n[6 + c];
    }
  }
  return out.map((v) => v / out[8]);
}

// ---- 6. losing the board freezes the quad rather than inventing one -----------
{
  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  const before = JSON.stringify(tracker.corners);

  // The lens is covered: uniform gray, no texture to track at all.
  const blank = new Float32Array(FRAME_W * FRAME_H).fill(0.5);
  const lost = tracker.update(blank);
  check("lost: reports the loss", lost.state === "lost", `state ${lost.state}`);
  check("lost: freezes the last good quad", JSON.stringify(tracker.corners) === before);

  // The hand moves away and the board is back where it was: the tracker recovers by
  // itself, because it keeps tracking FROM the frozen quad.
  tracker.update(render(IDENTITY, FRAME_W, FRAME_H));
  const back = tracker.update(render([1, 0, 2, 0, 1, 1, 0, 0, 1], FRAME_W, FRAME_H));
  check("lost: recovers once the view returns", back.state === "locked", `state ${back.state}`);
}

// ---- 7. grayscale matches the edge's weights ----------------------------------
{
  const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
  const gray = toGray(rgba, new Float32Array(3));
  close("gray: red", gray[0], 0.299, 1e-6);
  close("gray: green", gray[1], 0.587, 1e-6);
  close("gray: blue", gray[2], 0.114, 1e-6);
}

// ---- 8. the seed lattice is the BOARD's lattice -------------------------------
// Not decoration: an arbitrary NxN grid misses the square corners, which are a chess
// board's only high-gradient points, and it fails SILENTLY — measured on a real board
// frame, an 8x8 grid kept 18 of its points where the 9x9 lattice kept 31.
{
  check("lattice: one subdivision is the 9x9 grid of square corners",
        new PlaneTracker({ width: 100, height: 100, subdivisions: 1 }).grid === 9);
  check("lattice: two subdivisions is 17x17",
        new PlaneTracker({ width: 100, height: 100, subdivisions: 2 }).grid === 17);
  // With an 8x8 frame and the unit square as the quad, every seed lands on an integer
  // exactly when it lands on a square corner.
  const tracker = new PlaneTracker({ width: 8, height: 8, subdivisions: 1 });
  const seeded = tracker.seedPoints([[0, 0], [1, 0], [1, 1], [0, 1]]);
  check("lattice: 81 seeds", seeded.length === 81 * 2);
  check("lattice: every seed sits on a square corner",
        Array.from(seeded).every((v) => Math.abs(v - Math.round(v)) < 1e-9));
}

// ---- 9. the caller's capture buffer is reused, and that must not matter ---------
// The regression that motivated this check was found in a browser, not here: a page
// captures video into ONE Float32Array per frame, and an earlier pyramid kept a
// reference to it rather than a copy. One frame later the "previous" image held the
// current pixels, so every point measured itself, converged on zero displacement, and
// the tracker reported a confident lock while the quad never moved at all.
{
  const motion = [1, 0, 6, 0, 1, 3.5, 0, 0, 1];
  const scratch = new Float32Array(FRAME_W * FRAME_H);   // ONE buffer, as a page has
  const tracker = new PlaneTracker({ width: FRAME_W, height: FRAME_H, subdivisions: 1 });
  tracker.reset(CORNERS);
  scratch.set(render(IDENTITY, FRAME_W, FRAME_H));
  tracker.update(scratch);
  scratch.set(render(motion, FRAME_W, FRAME_H));         // same buffer, new pixels
  const moved = tracker.update(scratch);
  check("reuse: still locks", moved.state === "locked", `state ${moved.state}`);
  const want = expected(motion);
  for (let i = 0; i < 4; i++) {
    close(`reuse: corner ${i}.x follows`, moved.corners[i][0], want[i][0], 0.006);
    close(`reuse: corner ${i}.y follows`, moved.corners[i][1], want[i][1], 0.006);
  }
  // The point of the check: the quad actually MOVED, rather than reporting a healthy
  // fit while standing still.
  check("reuse: the quad moved at all",
        Math.abs(moved.corners[0][0] - CORNERS[0][0]) * FRAME_W > 3);
}

if (failures.length) {
  console.error(`${failures.length} of ${checks} checks FAILED:`);
  for (const failure of failures) console.error("  - " + failure);
  process.exit(1);
}
console.log(`tracker.js: ${checks} checks passed`);
