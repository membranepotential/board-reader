/**
 * A synthetic photo for the snapshot page's file input.
 *
 * `setInputFiles` wants real image bytes — a dispatched `change` event would be
 * untrusted and would not exercise the browser's own file plumbing — and a checked-in
 * binary fixture is a thing that rots silently. A PNG is a header plus three chunks, so
 * it is cheaper to write one than to depend on an encoder.
 */

import { deflateSync } from 'node:zlib';

const CRC_TABLE = ((): Uint32Array => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) === 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(data: Buffer): number {
  let c = 0xffffffff;
  for (const byte of data) c = (CRC_TABLE[(c ^ byte) & 0xff] as number) ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([length, body, crc]);
}

/**
 * An 8-bit RGB PNG of a light/dark checkerboard — bright enough that the capture
 * stage's mean-luma reading stays out of its low-light warning, and board-like enough
 * that a screenshot of a failing run is legible.
 */
export function checkerboardPng(width: number, height: number, squarePx = 48): Buffer {
  const stride = 1 + width * 3;
  const raw = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const row = y * stride;
    raw[row] = 0; // filter type 0 (none)
    for (let x = 0; x < width; x++) {
      const dark = (Math.floor(x / squarePx) + Math.floor(y / squarePx)) % 2 === 0;
      const value = dark ? 70 : 215;
      const at = row + 1 + x * 3;
      raw[at] = value;
      raw[at + 1] = value;
      raw[at + 2] = value;
    }
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0))
  ]);
}
