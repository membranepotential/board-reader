import { describe, expect, it } from 'vitest';
import {
  CLASSES,
  NUM_CLASSES,
  START_PLACEMENT,
  parseEstimateResponse,
  parseIngestReply,
  parseUploadsResponse,
  parseWire
} from '#lib/protocol/index.ts';
import { CameraConfig, PositionEvent } from './generated';

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

const parseEvent = (raw: unknown) => parseWire(PositionEvent, raw);

describe('a position event off the wire', () => {
  it('accepts a well-formed event', () => {
    const parsed = parseEvent(validEvent);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.board_id).toBe('phone');
    expect(parsed.value.seq).toBe(12);
    expect(parsed.value.clock.running).toBe('white');
    expect(parsed.value.square_dist).toHaveLength(8);
  });

  it('parses the JSON string form the socket delivers', () => {
    expect(parseEvent(JSON.stringify(validEvent)).ok).toBe(true);
  });

  it('carries the settled board when the event has one', () => {
    const bare = parseEvent(validEvent);
    expect(bare.ok && bare.value.settled).toBeUndefined();

    const parsed = parseEvent({
      ...validEvent,
      settled: { placement: '8/8/8/8/8/8/8/8', pending: ['e4', 'e2'] }
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.settled?.pending).toEqual(['e4', 'e2']);
  });

  it.each(['board_id', 'seq', 'utc', 'placement', 'stability', 'status', 'square_dist'])(
    'refuses an event missing %s, as the schema states every key the node sends',
    (field) => {
      const broken: Record<string, unknown> = { ...validEvent };
      delete broken[field];
      expect(parseEvent(broken).ok).toBe(false);
    }
  );

  // Strict exactly where pydantic is (`board_reader_core.wiremodel`): an event the
  // node would refuse is refused here too, whole, rather than shown with holes.
  it.each([
    ['not JSON at all', '{oops'],
    ['a bare array', [1, 2, 3]],
    ['a negative seq', { ...validEvent, seq: -1 }],
    ['a fractional seq', { ...validEvent, seq: 1.5 }],
    ['an unknown status', { ...validEvent, status: 'illegal' }],
    ['an unknown clock', { ...validEvent, clock: { running: 'sideways' } }],
    ['a stability above 1', { ...validEvent, stability: 1.2 }],
    ['seven ranks of distribution', { ...validEvent, square_dist: distFor().slice(0, 7) }],
    ['a placement of seven ranks', { ...validEvent, placement: '8/8/8/8/8/8/8' }],
    ['a settled board with no placement', { ...validEvent, settled: { pending: ['e4'] } }],
    [
      'a pending square that is not one',
      { ...validEvent, settled: { placement: START_PLACEMENT, pending: ['z9'] } }
    ],
    [
      'the prior block the node no longer sends',
      { ...validEvent, prior: { placement: START_PLACEMENT, flips: [], residual: {} } }
    ],
    ['a key the protocol does not define', { ...validEvent, streming: true }]
  ])('refuses %s', (_label, raw) => {
    expect(parseEvent(raw).ok).toBe(false);
  });

  it('says why in one line, each problem prefixed with where it is', () => {
    const parsed = parseEvent({ ...validEvent, seq: -1, clock: { running: 'sideways' } });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.error).toMatch(/^clock\.running: .+; seq: .+$|^seq: .+; clock\.running: .+$/);
  });
});

describe('a camera config off the wire', () => {
  const config = {
    crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.6 },
    device: 'Back Camera',
    resolution: '2160p',
    streaming: true
  };

  it('reads what board_reader_core.control writes', () => {
    expect(parseWire(CameraConfig, config)).toEqual({ ok: true, value: config });
    expect(parseWire(CameraConfig, { ...config, crop: null, device: null }).ok).toBe(true);
  });

  it.each([
    ['an unknown resolution', { ...config, resolution: '4k' }],
    ['a crop missing a side', { ...config, crop: { x: 0, y: 0, w: 1 } }],
    ['a crop that ends past the frame (cropInside)', { ...config, crop: { x: 0.6, y: 0, w: 0.5, h: 1 } }],
    ['a streaming flag that is not a boolean', { ...config, streaming: 'yes' }],
    ['a device that is not a label', { ...config, device: 3 }]
  ])('refuses %s', (_label, raw) => {
    expect(parseWire(CameraConfig, raw).ok).toBe(false);
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

  it('reads the position event and its extras', () => {
    const parsed = parseEstimateResponse({
      ...validEvent,
      upload: '20260807-100000-0001-estimate',
      warnings: ['low_light']
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.event.placement).toBe(validEvent.placement);
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
