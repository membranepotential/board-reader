import { describe, expect, it } from 'vitest';
import {
  classTable,
  cornerTable,
  geometryTable,
  materialTable,
  orientTable,
  repairTable
} from './tables';
import {
  EMPTY_FRAMING,
  absorbedFraming,
  detectedFraming,
  failedDetection,
  withMovedPoint,
  withPoint
} from './framing';
import type { KvRow } from './kv';
import { fenToGrid } from '$lib/board/fen';
import { quadGeometry } from '$lib/geometry/homography';
import {
  START_PLACEMENT,
  parseEstimateResponse,
  type CornerQuad,
  type EstimateResponse
} from '$lib/protocol';

const QUAD: CornerQuad = [
  [100, 100],
  [500, 100],
  [500, 400],
  [100, 400]
];

const row = (rows: readonly KvRow[], key: string): KvRow | undefined =>
  rows.find((r) => r.key === key);

function parsed(raw: Record<string, unknown>): EstimateResponse {
  const result = parseEstimateResponse(raw);
  if (!result.ok) throw new Error(result.error);
  return result.value;
}

const POSITION = {
  board_id: 'snapshot',
  seq: 4,
  utc: '2026-08-07T10:00:00Z',
  placement: START_PLACEMENT,
  confidence: 0.9
};

describe('the corner table', () => {
  it('counts the corners while they are still being placed', () => {
    expect(cornerTable(EMPTY_FRAMING, null, null)).toEqual([
      { key: 'corners', value: '0/4 placed', tone: 'plain' }
    ]);
    expect(row(cornerTable(withPoint(EMPTY_FRAMING, [1, 1]), null, null), 'corners')?.tone).toBe(
      'warn'
    );
  });

  it('lists the quad in canonical a8,h8,h1,a1 order — the order that goes on the wire', () => {
    const rows = cornerTable(detectedFraming(QUAD), QUAD, 42);
    expect(rows.map((r) => r.key)).toEqual([
      'source',
      'detect latency',
      'label rotation',
      'a8',
      'h8',
      'h1',
      'a1',
      'server auto-rotate'
    ]);
    expect(row(rows, 'a8')?.value).toBe('100, 100');
    expect(row(rows, 'detect latency')?.value).toBe('42 ms');
  });

  it('warns while the labelling is still an unverified guess, and stops once it is owned', () => {
    expect(row(cornerTable(detectedFraming(QUAD), QUAD, null), 'server auto-rotate')?.tone).toBe(
      'warn'
    );
    const edited = withMovedPoint(detectedFraming(QUAD), 0, [90, 90]);
    expect(row(cornerTable(edited, QUAD, null), 'server auto-rotate')?.tone).toBe('ok');
    expect(
      row(cornerTable(absorbedFraming(detectedFraming(QUAD), 90), QUAD, null), 'server auto-rotate')
        ?.tone
    ).toBe('ok');
  });

  it('calls hand-placed corners hand-placed', () => {
    const byHand = [...QUAD].reduce((f, p) => withPoint(f, p), failedDetection());
    expect(row(cornerTable(byHand, QUAD, null), 'source')?.value).toBe('hand-placed');
  });
});

describe('the geometry table', () => {
  it('warns on a flat camera and a dim photo, and only then', () => {
    const square = geometryTable(quadGeometry(QUAD), 120);
    expect(row(square, 'projected aspect')?.tone).toBe('ok');
    expect(row(square, 'mean luma')?.tone).toBe('ok');
    expect(row(square, 'square pitch')?.value).toBe('~37.5 px on the short axis');

    const flat = geometryTable(
      quadGeometry([
        [100, 100],
        [500, 100],
        [500, 180],
        [100, 180]
      ]),
      20
    );
    expect(row(flat, 'projected aspect')?.tone).toBe('warn');
    expect(row(flat, 'mean luma')?.tone).toBe('warn');
  });
});

describe('the classification table', () => {
  it('reports the read and its confidence', () => {
    const rows = classTable(parsed(POSITION), 180);
    expect(row(rows, 'position confidence')?.value).toBe('0.900');
    expect(row(rows, 'estimate latency')?.value).toBe('180 ms');
  });

  it('is empty before anything has been read, and reports a missing latency as a dash', () => {
    expect(classTable(null, null)).toEqual([]);
    expect(row(classTable(parsed(POSITION), null), 'estimate latency')?.value).toBe('—');
  });
});

describe('the orientation table', () => {
  it('states who owns the labelling and what the node did with it', () => {
    const auto = orientTable(parsed({ ...POSITION, autorotated: 90 }), detectedFraming(QUAD));
    expect(row(auto, 'corner labels')?.value).toBe('auto-detected guess');
    expect(row(auto, 'autorotate requested')?.value).toContain('yes');
    expect(row(auto, 'rotation applied')?.tone).toBe('warn');
    expect(row(auto, 'adopted by this page')?.value).toContain('shifted by 90 degrees');

    const owned = orientTable(parsed(POSITION), withMovedPoint(detectedFraming(QUAD), 0, [1, 1]));
    expect(row(owned, 'autorotate requested')?.value).toBe('no');
    expect(row(owned, 'adopted by this page')?.value).toBe('nothing to adopt');
  });
});

describe('the repair table', () => {
  it('reports a board that needed no repair as the good case', () => {
    expect(repairTable({ placement: START_PLACEMENT, flips: [], residual: {} })).toEqual([
      { key: 'flips', value: 'none — no constraint was violated', tone: 'plain' }
    ]);
  });

  it('lists every relabelling, cheapest first, and what is still impossible', () => {
    const rows = repairTable({
      placement: START_PLACEMENT,
      flips: [
        { square: 'd8', from: 'q', to: 'b', cost: 0.5, constraint: 'one_black_queen' },
        { square: 'a1', from: 'K', to: 'R', cost: 1.25, constraint: 'one_white_king' }
      ],
      residual: { one_white_king: 1 }
    });
    expect(rows.map((r) => r.key)).toEqual(['1. d8', '2. a1', 'unresolved']);
    expect(rows[0]?.value).toBe('q → b · one_black_queen · cost 0.500');
    expect(row(rows, 'unresolved')?.tone).toBe('warn');
  });

  it('says so when the node does not decode a repair at all', () => {
    expect(repairTable(null)[0]?.value).toContain('does not decode a repair');
  });
});

describe('the material census table', () => {
  it('passes the start position and flags an impossible one', () => {
    expect(row(materialTable(fenToGrid(START_PLACEMENT)), 'implausible')?.tone).toBe('ok');
    const rows = materialTable(fenToGrid('8/8/8/8/8/8/8/8'));
    expect(row(rows, 'implausible')?.tone).toBe('warn');
    expect(row(rows, 'occupied squares')?.value).toBe('0');
  });
});
