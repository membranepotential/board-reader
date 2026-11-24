import { describe, expect, it } from 'vitest';
import { tenengrad, toGray, type GrayImage } from './sharpness';

/** A synthetic board: `cell`-sized light and dark squares, i.e. nothing but edges. */
function checkerboard(width: number, height: number, cell: number): GrayImage {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dark = (Math.floor(x / cell) + Math.floor(y / cell)) % 2 === 0;
      data[y * width + x] = dark ? 0.2 : 0.8;
    }
  }
  return { data, width, height };
}

function flat(width: number, height: number, value: number): GrayImage {
  return { data: new Float32Array(width * height).fill(value), width, height };
}

/**
 * Bars of `period` px running along ONE axis, so the whole gradient is in the other.
 * `axis: 'x'` varies with x (vertical stripes), `axis: 'y'` with y.
 */
function stripes(width: number, height: number, period: number, axis: 'x' | 'y'): GrayImage {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const along = axis === 'x' ? x : y;
      data[y * width + x] = Math.floor(along / period) % 2 === 0 ? 0.2 : 0.8;
    }
  }
  return { data, width, height };
}

/**
 * Camera shake is a low-pass filter, so the test models it as one: a separable box
 * blur of radius `r`, clamped at the borders. It preserves the mean exactly, which
 * is what makes the comparison below about sharpness and nothing else.
 */
function blur({ data, width, height }: GrayImage, r: number): GrayImage {
  const clamp = (v: number, hi: number) => Math.max(0, Math.min(hi, v));
  const taps = 2 * r + 1;
  const mid = new Float32Array(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += data[y * width + clamp(x + k, width - 1)]!;
      mid[y * width + x] = sum / taps;
    }
  }
  const out = new Float32Array(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += mid[clamp(y + k, height - 1) * width + x]!;
      out[y * width + x] = sum / taps;
    }
  }
  return { data: out, width, height };
}

describe('tenengrad', () => {
  it('scores a flat field at zero — there is nothing to focus on', () => {
    expect(tenengrad(flat(64, 64, 0.5))).toBe(0);
    expect(tenengrad(flat(64, 64, 0))).toBe(0);
  });

  it('ranks a sharp pattern above a blurred copy of itself', () => {
    const sharp = checkerboard(96, 96, 8);
    const soft = blur(sharp, 2);
    const softer = blur(sharp, 4);
    expect(tenengrad(sharp)).toBeGreaterThan(tenengrad(soft));
    expect(tenengrad(soft)).toBeGreaterThan(tenengrad(softer));
    expect(tenengrad(softer)).toBeGreaterThan(0);
  });

  /**
   * The monotonicity tests above cannot see a broken kernel: a checkerboard is
   * symmetric in x and y, so `gy = gx`, a transposed pair, a wrong row in the gy
   * kernel and Prewitt weights all reproduce its ordering exactly. These two do see
   * it. Each image carries its gradient in ONE axis, and the Sobel pair is symmetric
   * under transposition, so the two scores must come out EQUAL — which is false for
   * every one of those mutations.
   */
  it('reads a gradient in either axis, and reads the two identically', () => {
    const vertical = stripes(96, 96, 8, 'x');
    const horizontal = stripes(96, 96, 8, 'y');
    const nothing = tenengrad(flat(96, 96, 0.5));

    expect(tenengrad(vertical)).toBeGreaterThan(nothing + 1);
    expect(tenengrad(horizontal)).toBeGreaterThan(nothing + 1);
    // The transpose of an image is the transpose of the kernel pair, so this is an
    // identity rather than an approximation.
    expect(tenengrad(vertical)).toBeCloseTo(tenengrad(horizontal), 10);
  });

  it('is Sobel, at the weights Sobel actually has', () => {
    // Symmetry catches a broken kernel but not a differently-weighted one: Prewitt
    // (1,1,1 instead of 1,2,1) ranks every image above identically. So one value is
    // pinned outright, on the smallest image with an interior — a single pixel, whose
    // arithmetic a reader can check:
    //
    //   sixteenths 1..9 in reading order, so the centre is 5/16
    //   gx = (3 + 2*6 + 9)/16 - (1 + 2*4 + 7)/16 = 24/16 - 16/16 = 0.5
    //   gy = (7 + 2*8 + 9)/16 - (1 + 2*2 + 3)/16 = 32/16 -  8/16 = 1.5
    //   (gx² + gy²) / 1 px / (5/16)² = 2.5 / 0.09765625 = 25.6
    //
    // Every sixteenth is exactly representable, so this is arithmetic and not a
    // recorded output. It fails for Prewitt weights (14.4), for `gy = gx` (5.12) and
    // for a gy stencil that reads the wrong top row (29.6).
    const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9], (v) => v / 16);
    expect(tenengrad({ data, width: 3, height: 3 })).toBeCloseTo(25.6, 10);
  });

  it('still ranks a one-axis pattern by focus, in both axes', () => {
    // Blur is isotropic here, so an axis the kernel cannot see would show up as a
    // pattern whose score does not move when it is smeared.
    for (const axis of ['x', 'y'] as const) {
      const sharp = stripes(96, 96, 8, axis);
      expect(tenengrad(sharp), `${axis} stripes`).toBeGreaterThan(tenengrad(blur(sharp, 3)));
    }
  });

  it('is insensitive to exposure, so a burst that brightens is still ranked by focus', () => {
    // The scores are a ranking key, not a quantity — but a candidate must not win
    // for having been captured half a stop brighter.
    const sharp = checkerboard(96, 96, 8);
    const brighter: GrayImage = { ...sharp, data: sharp.data.map((v) => v * 1.2) };
    expect(tenengrad(brighter)).toBeCloseTo(tenengrad(sharp), 6);
    // And a blurred bright frame still loses to a sharp dim one.
    expect(tenengrad(blur(brighter, 3))).toBeLessThan(tenengrad(sharp));
  });

  it('scores anything too small to hold a 3x3 neighbourhood at zero, never throwing', () => {
    expect(tenengrad(checkerboard(2, 2, 1))).toBe(0);
    expect(tenengrad({ data: new Float32Array(0), width: 0, height: 0 })).toBe(0);
    // A truncated buffer is a caller bug, but it must not read past its own end.
    expect(tenengrad({ data: new Float32Array(10), width: 8, height: 8 })).toBe(0);
  });
});

describe('toGray', () => {
  it('turns RGBA bytes into luma in [0,1] with the BGR2GRAY weights', () => {
    const rgba = [255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255];
    const gray = toGray(rgba, new Float32Array(3));
    expect(gray[0]).toBeCloseTo(1, 5);
    expect(gray[1]).toBe(0);
    expect(gray[2]).toBeCloseTo(0.299, 5);
  });
});
