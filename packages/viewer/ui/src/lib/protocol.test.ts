import { describe, expect, it } from 'vitest';
import {
  CLASSES,
  NUM_CLASSES,
  START_PLACEMENT,
  parseCameraConfig,
  parseEstimateResponse,
  parseIngestReply,
  parsePositionEvent,
  parseUploadsResponse
} from './protocol';

function distFor(placementPeak = 0.9): number[][][] {
  const other = (1 - placementPeak) / (NUM_CLASSES - 1);
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () =>
      Array.from({ length: NUM_CLASSES }, (_, i) => (i === 0 ? placementPeak : other))
    )
  );
}

const validEvent = {
  board_id: 'phone',
  seq: 12,
  utc: '2026-08-07T10:00:00.000Z',
  orientation: 'a1_bottom_left',
  classes: [...CLASSES],
  placement: START_PLACEMENT,
  square_dist: distFor(),
  confidence: 0.87,
  stability: 0.42,
  clock: { running: 'white' },
  status: 'ok'
};

describe('parsePositionEvent', () => {
  it('accepts a well-formed event', () => {
    const parsed = parsePositionEvent(validEvent);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.board_id).toBe('phone');
    expect(parsed.value.seq).toBe(12);
    expect(parsed.value.clock).toBe('white');
    expect(parsed.value.status).toBe('ok');
    expect(parsed.value.square_dist).not.toBeNull();
  });

  it('parses the JSON string form the socket delivers', () => {
    const parsed = parsePositionEvent(JSON.stringify(validEvent));
    expect(parsed.ok).toBe(true);
  });

  it('carries the prior repair when the event has one, and null when not', () => {
    const bare = parsePositionEvent(validEvent);
    expect(bare.ok && bare.value.prior).toBeNull();

    const parsed = parsePositionEvent({
      ...validEvent,
      prior: {
        placement: '8/8/8/8/8/8/8/8',
        flips: [{ square: 'e8', from: 'k', to: 'q', cost: 0.74, constraint: 'kings' }],
        residual: { kings: 1 }
      }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.prior?.placement).toBe('8/8/8/8/8/8/8/8');
    expect(parsed.value.prior?.flips[0]).toEqual({
      square: 'e8', from: 'k', to: 'q', cost: 0.74, constraint: 'kings'
    });
    expect(parsed.value.prior?.residual).toEqual({ kings: 1 });
  });

  it('carries the settled board when the event has one, and null when not', () => {
    const bare = parsePositionEvent(validEvent);
    expect(bare.ok && bare.value.settled).toBeNull();

    const parsed = parsePositionEvent({
      ...validEvent,
      settled: { placement: '8/8/8/8/8/8/8/8', pending: ['e4', 'e2'] }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.settled?.placement).toBe('8/8/8/8/8/8/8/8');
    expect(parsed.value.settled?.pending).toEqual(['e4', 'e2']);
  });

  it('reads a settled block with no pending field as settled but quiescent', () => {
    const parsed = parsePositionEvent({
      ...validEvent,
      settled: { placement: START_PLACEMENT }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.settled?.pending).toEqual([]);
  });

  it('reads a settled block with no usable placement as absent, not partial', () => {
    const parsed = parsePositionEvent({ ...validEvent, settled: { pending: ['e4'] } });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.settled).toBeNull();
  });

  it.each(['board_id', 'seq', 'utc', 'placement'])('refuses an event missing %s', (field) => {
    const broken: Record<string, unknown> = { ...validEvent };
    delete broken[field];
    expect(parsePositionEvent(broken).ok).toBe(false);
  });

  it.each([
    ['not JSON at all', '{oops'],
    ['a bare array', [1, 2, 3]],
    ['null', null],
    ['a negative seq', { ...validEvent, seq: -1 }],
    ['a fractional seq', { ...validEvent, seq: 1.5 }]
  ])('refuses %s', (_label, raw) => {
    expect(parsePositionEvent(raw).ok).toBe(false);
  });

  it('falls back to the protocol default for an unknown status or clock', () => {
    const parsed = parsePositionEvent({
      ...validEvent,
      status: 'illegal',
      clock: { running: 'sideways' }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    // `status` has no `illegal` member by design (DESIGN §5) — an unrecognised chip
    // must not blank the whole reading.
    expect(parsed.value.status).toBe('ok');
    expect(parsed.value.clock).toBe('none');
  });

  it('keeps a missing stability distinguishable from "at rest"', () => {
    const { stability: _drop, confidence: _drop2, ...bare } = validEvent;
    const parsed = parsePositionEvent(bare);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.stability).toBeNull();
    expect(parsed.value.confidence).toBeNull();
  });

  it.each([
    ['a missing distribution', undefined],
    ['seven ranks', distFor().slice(0, 7)],
    ['twelve classes on a square', distFor().map((r) => r.map((c) => c.slice(0, 12)))],
    ['a non-numeric entry', [['x']]]
  ])('reads %s as no distribution rather than a partial one', (_label, dist) => {
    const parsed = parsePositionEvent({ ...validEvent, square_dist: dist });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.square_dist).toBeNull();
  });
});

describe('parseCameraConfig', () => {
  const config = {
    crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.6 },
    device: 'Back Camera',
    resolution: '2160p',
    streaming: true
  };

  it('reads what board_reader_core.control writes', () => {
    expect(parseCameraConfig(config)).toEqual({ ok: true, value: config });
    expect(parseCameraConfig({ ...config, crop: null, device: null }).ok).toBe(true);
  });

  it.each([
    ['an unknown resolution', { ...config, resolution: '4k' }],
    ['a crop missing a side', { ...config, crop: { x: 0, y: 0, w: 1 } }],
    ['a streaming flag that is not a boolean', { ...config, streaming: 'yes' }],
    ['a device that is not a label', { ...config, device: 3 }]
  ])('refuses %s', (_label, raw) => {
    expect(parseCameraConfig(raw).ok).toBe(false);
  });
});

describe('parseIngestReply', () => {
  it('reads an ack', () => {
    expect(parseIngestReply('{"ack": 7, "board_id": "phone"}')).toEqual({
      kind: 'ack',
      seq: 7,
      board_id: 'phone'
    });
  });

  it('reads a per-frame refusal', () => {
    expect(parseIngestReply('{"error": "bad frame: corners"}')).toEqual({
      kind: 'error',
      error: 'bad frame: corners'
    });
  });

  it.each(['', 'not json', '{}', '{"ack": "seven"}'])('ignores %s', (raw) => {
    expect(parseIngestReply(raw)).toBeNull();
  });
});

describe('parseEstimateResponse', () => {
  it('refuses a reply that is not a position event', () => {
    expect(parseEstimateResponse({ board_id: 'snapshot' }).ok).toBe(false);
  });

  it('reads the position event, with its prior and its extras', () => {
    const parsed = parseEstimateResponse({
      ...validEvent,
      upload: '20260807-100000-0001-estimate',
      warnings: ['low_light'],
      prior: {
        placement: START_PLACEMENT,
        flips: [{ square: 'd8', from: 'q', to: 'b', cost: 0.4, constraint: 'piece_count' }],
        residual: { kings: 1 }
      }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.prior?.flips[0]?.square).toBe('d8');
    expect(parsed.value.prior?.residual['kings']).toBe(1);
    expect(parsed.value.warnings).toEqual(['low_light']);
    expect(parsed.value.upload).toBe('20260807-100000-0001-estimate');
  });
});

describe('parseUploadsResponse', () => {
  it('reads retention off as an answer, not an error', () => {
    const parsed = parseUploadsResponse({ retention: false, uploads: [] });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.retention).toBe(false);
  });

  it('drops a row with no id and keeps the rest', () => {
    const parsed = parseUploadsResponse({
      retention: true,
      uploads: [{ kind: 'estimate' }, { id: 'a', ts: 1_754_000_000, kind: 'estimate', labeled: true }]
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.uploads).toHaveLength(1);
    expect(parsed.value.uploads[0]?.labeled).toBe(true);
  });

  it('takes a timestamp as ISO, epoch seconds or epoch milliseconds', () => {
    const parsed = parseUploadsResponse({
      uploads: [
        { id: 'iso', ts: '2026-08-07T10:00:00Z' },
        { id: 'sec', ts: 1_754_560_800 },
        { id: 'ms', ts: 1_754_560_800_000 },
        { id: 'junk', ts: 'whenever' }
      ]
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const [iso, sec, ms, junk] = parsed.value.uploads;
    expect(iso?.ts).toBe(Date.parse('2026-08-07T10:00:00Z') / 1000);
    expect(sec?.ts).toBe(1_754_560_800);
    expect(ms?.ts).toBe(1_754_560_800);
    expect(junk?.ts).toBeNull();
  });
});
