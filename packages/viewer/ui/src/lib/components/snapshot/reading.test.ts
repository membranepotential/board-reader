import { describe, expect, it } from 'vitest';
import {
  assignmentFor,
  argmaxClass,
  baselinePlacement,
  castlingRights,
  classOfKey,
  confidenceSummary,
  correctedGrid,
  cycleTarget,
  fullFen,
  heatColor,
  lichessUrls,
  materialCensus,
  placementOfGrid,
  rankedClasses,
  rawEventJson,
  scoresAt,
  shownClassAt,
  weakestSquares,
  withCorrection,
  type Corrections
} from './reading';
import { fenToGrid } from '$lib/board/fen';
import {
  CLASSES,
  NUM_CLASSES,
  START_PLACEMENT,
  parseEstimateResponse,
  parseSquareDist,
  type ClassName,
  type ClassScores,
  type EstimateResponse,
  type SquareDist
} from '$lib/protocol';

function scores(peaks: Partial<Record<ClassName, number>>): number[] {
  const cell = CLASSES.map((cls) => peaks[cls] ?? 0);
  const total = cell.reduce((a, b) => a + b, 0);
  return cell.map((p) => p / total);
}

/** An 8×8 of the same distribution, with `overrides` denting individual squares. */
function distOf(
  base: Partial<Record<ClassName, number>>,
  overrides: ReadonlyMap<string, Partial<Record<ClassName, number>>> = new Map()
): SquareDist {
  const raw = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => scores(overrides.get(`${r},${c}`) ?? base))
  );
  const parsed = parseSquareDist(raw);
  if (parsed === null) throw new Error('test distribution is malformed');
  return parsed;
}

const flatCell = (): ClassScores => {
  const dist = distOf({ empty: 9, P: 1 });
  return dist[0][0];
};

describe('the board as shown', () => {
  const base = fenToGrid(START_PLACEMENT);

  it('is the node’s read when there are no corrections', () => {
    expect(correctedGrid(base, new Map())).toBe(base);
  });

  it('layers a correction without touching the baseline', () => {
    const corrections: Corrections = new Map([['e1', 'q' as ClassName]]);
    expect(placementOfGrid(correctedGrid(base, corrections))).toBe(
      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQqBNR'
    );
    expect(placementOfGrid(base)).toBe(START_PLACEMENT);
  });

  it('spells an emptied square as an empty cell, not the word', () => {
    const corrections: Corrections = new Map([['a1', 'empty' as ClassName]]);
    expect(placementOfGrid(correctedGrid(base, corrections))).toBe(
      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR'
    );
  });

  it('reports the correction where there is one and the read everywhere else', () => {
    const corrections: Corrections = new Map([['a8', 'K' as ClassName]]);
    expect(shownClassAt(base, corrections, 0, 0)).toBe('K');
    expect(shownClassAt(base, corrections, 0, 1)).toBe('n');
    expect(shownClassAt(base, corrections, 4, 4)).toBe('empty');
  });
});

describe('one square’s distribution', () => {
  it('ranks the classes most likely first', () => {
    const cell = distOf({ empty: 1, P: 6, N: 3 })[0][0];
    expect(rankedClasses(cell).slice(0, 3).map((r) => r.cls)).toEqual(['P', 'N', 'empty']);
    expect(argmaxClass(cell)).toBe('P');
  });

  it('reads an absent distribution as fully confident rather than as doubt', () => {
    expect(scoresAt(null, 3, 3)).toBeNull();
    expect(confidenceSummary(null).mean).toBe(1);
    expect(confidenceSummary(null).low).toBe(0);
  });

  it('summarises the weakest squares and the low count', () => {
    const dist = distOf({ empty: 9, P: 1 }, new Map([['2,3', { empty: 4, P: 4, N: 2 }]]));
    const summary = confidenceSummary(dist);
    expect(summary.low).toBe(1);
    expect(summary.min).toBeCloseTo(0.4, 6);
    expect(weakestSquares(dist)[0]?.square).toBe('d6');
    expect(weakestSquares(dist)).toHaveLength(8);
  });
});

describe('assigning and cycling', () => {
  const cell = distOf({ P: 6, N: 3, B: 1 })[0][0];

  it('walks argmax → 2nd → 3rd → back to no correction', () => {
    expect(cycleTarget(cell, undefined)).toBe('N');
    expect(cycleTarget(cell, 'N')).toBe('B');
    expect(cycleTarget(cell, 'B')).toBeNull();
  });

  it('resets to the read from a class outside the top three', () => {
    // A grid pick that is not in the top 3 is still a valid state to toggle from, and
    // "reset to the argmax" is spelled as clearing the correction.
    expect(cycleTarget(cell, 'k')).toBeNull();
  });

  it('treats picking the model’s own read as no correction', () => {
    expect(assignmentFor(cell, 'P')).toBeNull();
    expect(assignmentFor(cell, 'q')).toBe('q');
  });

  it('adds and clears corrections without mutating the previous set', () => {
    const first = withCorrection(new Map(), 'e4', 'q');
    const second = withCorrection(first, 'e4', null);
    expect(first.get('e4')).toBe('q');
    expect(second.size).toBe(0);
  });

  it('maps keyboard keys to classes, case carrying colour as in a FEN', () => {
    expect(classOfKey('K')).toBe('K');
    expect(classOfKey('k')).toBe('k');
    expect(classOfKey('.')).toBe('empty');
    expect(classOfKey('Backspace')).toBe('empty');
    expect(classOfKey('z')).toBeNull();
    expect(classOfKey('ArrowLeft')).toBeNull();
  });
});

describe('the census and the FEN', () => {
  it('accepts the start position', () => {
    const census = materialCensus(fenToGrid(START_PLACEMENT));
    expect(census.white).toBe('K1 Q1 R2 B2 N2 P8');
    expect(census.occupied).toBe(32);
    expect(census.implausible).toEqual([]);
  });

  it('names counts no position can have — without refusing to show them', () => {
    const census = materialCensus(fenToGrid('KKKKKKKK/8/8/8/8/8/8/8'));
    expect(census.implausible).toEqual(['white kings: 8', 'black kings: 0']);
  });

  it('grants castling only to a king and rook still on their start squares', () => {
    expect(castlingRights(fenToGrid(START_PLACEMENT))).toBe('KQkq');
    expect(castlingRights(fenToGrid('r3k2r/8/8/8/8/8/8/R3K1R1'))).toBe('Qkq');
    expect(castlingRights(fenToGrid('8/8/8/8/8/8/8/8'))).toBe('-');
  });

  it('builds a full FEN and the Lichess links from it', () => {
    expect(fullFen(START_PLACEMENT, 'b')).toBe(`${START_PLACEMENT} b KQkq - 0 1`);
    expect(lichessUrls(START_PLACEMENT, 'w').editor).toBe(
      `https://lichess.org/editor/${START_PLACEMENT}_w_KQkq_-_0_1`
    );
  });
});

describe('the raw event block', () => {
  const positionRaw = {
    board_id: 'snapshot',
    seq: 3,
    utc: '2026-08-07T10:00:00Z',
    placement: START_PLACEMENT,
    confidence: 0.91,
    square_dist: Array.from({ length: 8 }, () =>
      Array.from({ length: 8 }, () => Array.from({ length: NUM_CLASSES }, () => 1 / NUM_CLASSES))
    ),
    autorotated: 90,
    upload: 'u1',
    prior: { placement: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1', flips: [], residual: {} },
    debug: { boxes: 'AAA', mosaic: 'BBB' },
    warnings: ['low_light']
  };

  function parsed(raw: unknown): EstimateResponse {
    const result = parseEstimateResponse(raw);
    if (!result.ok) throw new Error(result.error);
    return result.value;
  }

  it('prefers the repaired board as the baseline, leaving the raw argmax in the event', () => {
    const response = parsed(positionRaw);
    expect(baselinePlacement(response)).toBe('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1');
    expect(response.event.placement).toBe(START_PLACEMENT);
  });

  it('strips the debug JPEGs and the 8×8×13 floats', () => {
    const json = rawEventJson(parsed(positionRaw));
    expect(json).not.toContain('AAA');
    expect(json).toContain('see the confidence map');
    expect(JSON.parse(json)).toMatchObject({ upload: 'u1', autorotated: 90 });
  });
});

describe('the heat ramp', () => {
  it('runs red → yellow → green and never leaves the ramp', () => {
    expect(heatColor(0)).toBe('rgb(150,45,45)');
    expect(heatColor(0.5)).toBe('rgb(226,85,85)');
    expect(heatColor(0.8)).toBe('rgb(255,200,60)');
    expect(heatColor(1)).toBe('rgb(130,210,130)');
  });

  it('is defined for the flat distribution a missing read produces', () => {
    expect(heatColor(Math.max(...flatCell()))).toMatch(/^rgb\(\d+,\d+,\d+\)$/);
  });
});
