import { describe, expect, it } from 'vitest';

import { CLASSES, START_PLACEMENT } from '#lib/protocol/index.ts';
import { PositionGraph, type PositionNode } from './positions.ts';

/** Follow SAN moves through the graph from the initial position. */
function walk(graph: PositionGraph, sans: readonly string[]): PositionNode {
  return sans.reduce((node, san) => {
    const edge = graph.edges(node).find((e) => e.san === san);
    if (edge === undefined) throw new Error(`${san} is not legal here`);
    return edge.to;
  }, graph.start());
}

describe('PositionGraph', () => {
  it('starts at the initial position, labelled in grid order (a8 first)', () => {
    const start = new PositionGraph().start();
    expect(start.placement).toBe(START_PLACEMENT);
    expect(CLASSES[start.labels[0]!]).toBe('r');
    expect(CLASSES[start.labels[4]!]).toBe('k');
    expect(CLASSES[start.labels[60]!]).toBe('K');
    expect(CLASSES[start.labels[32]!]).toBe('empty');
  });

  it('generates every legal move once, castling included', () => {
    const graph = new PositionGraph();
    expect(graph.edges(graph.start())).toHaveLength(20);
    const castles = graph.edges(walk(graph, ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5'])).filter((e) => e.san === 'O-O');
    expect(castles).toHaveLength(1);
    expect(castles[0]!.to.placement).toBe('r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1');
  });

  it('shares one node between two move orders that reach the same position', () => {
    const graph = new PositionGraph();
    expect(walk(graph, ['Nf3', 'Nf6', 'Nc3'])).toBe(walk(graph, ['Nc3', 'Nf6', 'Nf3']));
  });

  it('offers all four promotions and marks the three that are not a queen', () => {
    const graph = new PositionGraph();
    const node = walk(graph, ['h4', 'g5', 'hxg5', 'Nf6', 'gxf6', 'Rg8', 'fxe7', 'a6']);
    const promotions = graph.edges(node).filter((e) => e.san.startsWith('exd8'));
    expect(promotions.map((e) => [e.san, e.underpromotion])).toEqual([
      ['exd8=Q+', false],
      ['exd8=N', true],
      ['exd8=R+', true],
      ['exd8=B', true]
    ]);
  });
});
