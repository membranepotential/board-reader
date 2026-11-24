import { describe, expect, it } from 'vitest';
import { positionEvent } from '#lib/protocol/fixtures.ts';
import { readTable, materialTable, repairTable } from './tables';
import type { KvRow } from './kv';
import { fenToGrid } from '#lib/board/fen.ts';
import { START_PLACEMENT, parseEstimateResponse, type EstimateResponse } from '#lib/protocol/index.ts';

const row = (rows: readonly KvRow[], key: string): KvRow | undefined =>
  rows.find((r) => r.key === key);

function parsed(raw: Record<string, unknown>): EstimateResponse {
  const result = parseEstimateResponse(raw);
  if (!result.ok) throw new Error(result.error);
  return result.value;
}

const POSITION = positionEvent({ board_id: 'snapshot', seq: 4, confidence: 0.9 });

describe('the read table', () => {
  it('reports the read and its confidence', () => {
    const rows = readTable(parsed(POSITION), 180);
    expect(row(rows, 'position confidence')?.value).toBe('0.900');
    expect(row(rows, 'estimate latency')?.value).toBe('180 ms');
  });

  it('is empty before anything has been read, and reports a missing latency as a dash', () => {
    expect(readTable(null, null)).toEqual([]);
    expect(row(readTable(parsed(POSITION), null), 'estimate latency')?.value).toBe('—');
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
