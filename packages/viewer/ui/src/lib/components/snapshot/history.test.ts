import { describe, expect, it } from 'vitest';
import {
  historyNote,
  historyRows,
  readFlags,
  uploadFlags,
  type SessionRead
} from './history';
import { EMPTY_FRAMING } from './framing';
import { labelState, NO_CORRECTIONS, type Corrections } from './reading';
import type { CaptureFacts, SourceImage } from './photo';
import {
  START_PLACEMENT,
  parseEstimateResponse,
  type EstimateResponse,
  type UploadRecord
} from '$lib/protocol';

const IMAGE: SourceImage = {
  width: 800,
  height: 600,
  dataUrl: 'data:image/jpeg;base64,AAA',
  bytes: 1024,
  meanLuma: 120
};

const FACTS: CaptureFacts = {
  origin: 'IMG.jpg · 1 kB',
  retainedAt: null,
  originalWidth: 1600,
  originalHeight: 1200,
  scale: 0.5
};

function response(raw: Record<string, unknown>): EstimateResponse {
  const parsed = parseEstimateResponse(raw);
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.value;
}

function read(
  id: number,
  seconds: number,
  raw: Record<string, unknown>,
  corrections: Corrections = NO_CORRECTIONS
): SessionRead {
  return {
    id,
    at: new Date(seconds * 1000),
    response: response(raw),
    corrections,
    image: IMAGE,
    facts: FACTS,
    framing: EMPTY_FRAMING,
    detect: { state: 'ok', summary: 'auto-detected' },
    detectMs: 40,
    estimateMs: 180
  };
}

const position = {
  board_id: 'snapshot',
  seq: 1,
  utc: '2026-08-07T10:00:00Z',
  placement: START_PLACEMENT,
  confidence: 0.9
};

function upload(id: string, ts: number | null, extra: Partial<UploadRecord> = {}): UploadRecord {
  return {
    id,
    ts,
    kind: 'estimate',
    corners: null,
    placement: null,
    autorotated: null,
    labeled: false,
    label_placement: null,
    label_corrections: null,
    ...extra
  };
}

describe('the merged history', () => {
  it('is newest first across both sources', () => {
    const rows = historyRows(
      [read(1, 100, position), read(2, 300, position)],
      [upload('a', 200), upload('b', 400)]
    );
    expect(rows.map((row) => row.key)).toEqual(['ub', 's2', 'ua', 's1']);
  });

  it('shows a photo this session read as the SESSION entry, not twice', () => {
    const rows = historyRows(
      [read(1, 100, { ...position, upload: 'u1' })],
      [upload('u1', 100, { labeled: true })]
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe('session');
    // …and the node's record still contributes the one thing only it knows.
    expect(rows[0]?.kind === 'session' && rows[0].upload?.labeled).toBe(true);
  });

  it('sorts a row with an unreadable timestamp last rather than dropping it', () => {
    const rows = historyRows([], [upload('a', null), upload('b', 500)]);
    expect(rows.map((row) => row.key)).toEqual(['ub', 'ua']);
  });

  it('says why the list is short when the node is not retaining or not answering', () => {
    expect(historyNote(false)).toContain('retention is off');
    expect(historyNote(null)).toContain('did not answer /uploads');
    expect(historyNote(true)).toContain('every photo the inference node still holds');
  });
});

describe('how a row describes itself', () => {
  it('flags corrections, rotation, labelling and node warnings', () => {
    const entry = read(
      1,
      1,
      {
        ...position,
        autorotated: 90,
        warnings: ['low_light']
      },
      new Map([['e4', 'q']]) as Corrections
    );
    expect(readFlags(entry, upload('u1', 1, { labeled: true }))).toEqual([
      '1 corrected',
      'rotated 90deg',
      'labelled',
      'low_light'
    ]);
  });

  it('names a node row’s own kind only when it is not a plain estimate', () => {
    expect(uploadFlags(upload('u1', 1))).toEqual([]);
    expect(uploadFlags(upload('u1', 1, { kind: 'calibrate' }))).toEqual(['calibrate']);
  });
});

describe('the ground-truth button', () => {
  it('explains itself when the node retains nothing', () => {
    expect(labelState(null, START_PLACEMENT, null)).toBe('no-retention');
  });

  it('offers a save for a retained photo, and an update once the board is edited', () => {
    expect(labelState('u1', START_PLACEMENT, null)).toBe('unsaved');
    expect(labelState('u1', START_PLACEMENT, START_PLACEMENT)).toBe('saved');
    expect(labelState('u1', '8/8/8/8/8/8/8/8', START_PLACEMENT)).toBe('stale');
  });
});
