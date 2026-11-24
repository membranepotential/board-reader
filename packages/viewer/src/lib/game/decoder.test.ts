import { describe, expect, it } from 'vitest';

import { CLASSES, GRID, START_PLACEMENT } from '#lib/protocol/index.ts';
import { positionEvent } from '#lib/protocol/fixtures.ts';
import { PositionEvent } from '#lib/protocol/generated.ts';
import type { SquareDist } from '#lib/protocol/index.ts';
import type { Labels } from './emission.ts';
import { DEFAULT_PARAMS, GameDecoder, observationOf, type Observation } from './decoder.ts';
import { PositionGraph, type PositionNode } from './positions.ts';

const graph = new PositionGraph();

/** The position after `sans` from the initial position. */
function after(...sans: string[]): PositionNode {
  return sans.reduce((node, san) => {
    const edge = graph.edges(node).find((e) => e.san === san);
    if (edge === undefined) throw new Error(`${san} is not legal here`);
    return edge.to;
  }, graph.start());
}

/** A read certain (to `p`) of `labels` on every square. */
function distOf(labels: Labels, p = 0.98): SquareDist {
  const rest = (1 - p) / (CLASSES.length - 1);
  return asDist(Array.from({ length: GRID }, (_, row) =>
    Array.from({ length: GRID }, (_, col) => CLASSES.map((_, c) => (c === labels[row * GRID + col] ? p : rest)))
  ));
}

const asDist = (rows: unknown): SquareDist => PositionEvent.shape.square_dist.parse(rows);

const T0 = Date.parse('2026-11-24T12:00:00.000Z');

/** A read of `node` at `T0 + seconds`, the node's settled board taken to be the same. */
function read(node: PositionNode, seconds: number, labels: Labels = node.labels): Observation {
  return {
    dist: distOf(labels),
    at: T0 + seconds * 1000,
    settled: { placement: node.placement, since: T0 + seconds * 1000 }
  };
}

function decode(reads: readonly Observation[], params = DEFAULT_PARAMS): GameDecoder {
  const decoder = new GameDecoder(params);
  for (const r of reads) decoder.push(r);
  return decoder;
}

const sans = (decoder: GameDecoder) => decoder.game.moves.map((m) => m.san);

describe('GameDecoder', () => {
  it('starts at the initial position with no moves', () => {
    const game = new GameDecoder().game;
    expect(game).toMatchObject({ startedAt: null, moves: [], placement: START_PLACEMENT, outcome: undefined });
  });

  it('decodes clean reads into the moves, each timed and committed', () => {
    const start = graph.start();
    const e5 = after('e4', 'e5');
    const decoder = decode([read(start, 0), read(after('e4'), 2), ...[4, 6, 8, 10].map((s) => read(e5, s))]);
    expect(decoder.game.moves).toEqual([
      { san: 'e4', uci: 'e2e4', at: T0 + 2000, committed: true },
      { san: 'e5', uci: 'e7e5', at: T0 + 4000, committed: true }
    ]);
    expect(decoder.game.startedAt).toBe(T0);
    expect(decoder.game.mismatch).toBe(0);
  });

  it('finds two plies that fell between two reads', () => {
    expect(sans(decode([read(graph.start(), 0), read(after('d4', 'd5'), 2)]))).toEqual(['d4', 'd5']);
  });

  it('plays no move for one misread square, and counts it as a mismatch', () => {
    const start = graph.start();
    const misread = start.labels.slice();
    misread[52] = 0; // e2 read empty
    const decoder = decode([read(start, 0), read(start, 2, misread)]);
    expect(decoder.game.moves).toEqual([]);
    expect(decoder.game.mismatch).toBe(1);
  });

  it('never plays an illegal move: a board no legal move reaches keeps the game', () => {
    const start = graph.start();
    const teleported = start.labels.slice();
    teleported[57] = 0; // b1 empty
    teleported[36] = start.labels[57]!; // and the knight on e4
    const decoder = decode([read(start, 0), read(start, 2, teleported), read(start, 4, teleported)]);
    expect(decoder.game.moves).toEqual([]);
  });

  it('takes a move back when the board returns to the position before it', () => {
    const start = graph.start();
    const e4 = after('e4');
    const decoder = decode([read(start, 0), read(e4, 2), read(e4, 4), ...[6, 8, 10, 12].map((s) => read(start, s))]);
    expect(decoder.game.moves).toEqual([]);
  });

  it('promotes to the piece the read shows, and to a queen when it cannot tell', () => {
    // A pawn on e7, one capture from promoting on d8.
    const line = ['h4', 'g5', 'hxg5', 'Nf6', 'gxf6', 'Rg8', 'fxe7', 'a6'];
    const lead = [read(graph.start(), 0), ...line.map((_, i) => read(after(...line.slice(0, i + 1)), 2 * (i + 1)))];
    const knight = after(...line, 'exd8=N');
    expect(sans(decode([...lead, read(knight, 20), read(knight, 22)])).at(-1)).toBe('exd8=N');

    // d8 split evenly between a white queen and a white knight.
    const d8 = 3;
    const split = (seconds: number): Observation => {
      const shown = read(knight, seconds);
      const dist = shown.dist.map((row, r) =>
        row.map((square, c) => (r * GRID + c === d8 ? CLASSES.map((name) => (name === 'Q' || name === 'N' ? 0.5 : 0)) : square))
      );
      return { ...shown, dist: asDist(dist) };
    };
    expect(sans(decode([...lead, split(20), split(22)])).at(-1)).toBe('exd8=Q+');
  });

  it('keeps a move provisional while an explanation without it stays within the margin', () => {
    const decoder = decode([read(graph.start(), 0), read(after('e4'), 2)], { ...DEFAULT_PARAMS, margin: 1000 });
    expect(decoder.game.moves).toEqual([expect.objectContaining({ san: 'e4', committed: false })]);
  });

  it("times a move by the settled board's change when that is earlier than the read", () => {
    const e4 = after('e4');
    const late = { ...read(e4, 4), settled: { placement: e4.placement, since: T0 + 3000 } };
    expect(decode([read(graph.start(), 0), late]).game.moves[0]!.at).toBe(T0 + 3000);
  });

  it('ignores a read that is not newer than the last one', () => {
    const decoder = decode([read(graph.start(), 2)]);
    const before = decoder.game;
    expect(decoder.push(read(after('e4'), 2))).toBe(before);
  });
});

describe('observationOf', () => {
  it('reads an ok event, timed by its utc, with the settled board and its changed_utc', () => {
    const event = PositionEvent.parse(
      positionEvent({ utc: '2026-11-24T12:00:05.000Z', changed_utc: '2026-11-24T12:00:03.500Z' })
    );
    expect(observationOf(event)).toEqual({
      dist: event.square_dist,
      at: Date.parse('2026-11-24T12:00:05.000Z'),
      settled: { placement: START_PLACEMENT, since: Date.parse('2026-11-24T12:00:03.500Z') }
    });
  });

  it('refuses an event that is not a read of the board', () => {
    expect(observationOf(PositionEvent.parse(positionEvent({ status: 'needs_setup' })))).toBeNull();
  });
});
