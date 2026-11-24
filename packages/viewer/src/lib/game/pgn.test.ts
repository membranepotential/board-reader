import { describe, expect, it } from 'vitest';

import { parsePgn, parseComment } from 'chessops/pgn';

import { START_PLACEMENT } from '#lib/protocol/index.ts';
import type { DecodedGame, DecodedMove } from './decoder.ts';
import { gamePgn, LICHESS_ANALYSIS, lichessAnalysisUrl } from './pgn.ts';

const T0 = Date.parse('2026-11-24T19:41:05.200Z');

function move(san: string, seconds: number): DecodedMove {
  return { san, uci: '', at: T0 + seconds * 1000, committed: true };
}

function game(moves: readonly DecodedMove[], over: Partial<DecodedGame> = {}): DecodedGame {
  return { startedAt: T0, moves, placement: START_PLACEMENT, outcome: undefined, mismatch: 0, ...over };
}

describe('gamePgn', () => {
  it('writes the tags, the moves and the time each move took', () => {
    const pgn = gamePgn(game([move('e4', 3.4), move('e5', 12), move('Qh5', 12)]));
    const [parsed] = parsePgn(pgn);
    expect(Object.fromEntries(parsed!.headers)).toMatchObject({
      Date: '2026.11.24',
      UTCDate: '2026.11.24',
      UTCTime: '19:41:05',
      Result: '*'
    });
    const nodes = [...parsed!.moves.mainline()];
    expect(nodes.map((n) => n.san)).toEqual(['e4', 'e5', 'Qh5']);
    expect(nodes.map((n) => parseComment(n.comments![0]!).emt)).toEqual([3, 9, 0]);
    expect(pgn).toContain('1. e4 { [%emt 0:00:03] } e5 { [%emt 0:00:09] }');
  });

  it('writes the result of a finished game', () => {
    const mate = game([move('f3', 1), move('e5', 2), move('g4', 3), move('Qh4#', 4)], { outcome: { winner: 'black' } });
    expect(gamePgn(mate)).toMatch(/\[Result "0-1"\][\s\S]*Qh4# \{ \[%emt 0:00:01\] \} 0-1/);
  });

  it('writes an unknown date before the first read', () => {
    expect(gamePgn(game([], { startedAt: null }))).toContain('[Date "????.??.??"]');
  });
});

describe('lichessAnalysisUrl', () => {
  it('carries the movetext alone, spaces as underscores and a mate sign escaped', () => {
    expect(lichessAnalysisUrl(game([move('f3', 1), move('e5', 2), move('g4', 3), move('Qh4#', 4)]))).toBe(
      `${LICHESS_ANALYSIS}1._f3_e5_2._g4_Qh4%23_*`
    );
  });

  it('opens an empty board for a game with no moves', () => {
    expect(lichessAnalysisUrl(game([]))).toBe(`${LICHESS_ANALYSIS}*`);
  });
});
