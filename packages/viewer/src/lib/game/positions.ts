/**
 * The legal positions the decoder walks, built lazily with chessops.
 *
 * A `PositionNode` is one legal position with what the emission model needs (its
 * `Labels`) and what the page needs (its FEN placement field). Nodes are interned by
 * key, so two move orders that reach the same position share one node, and each node's
 * legal moves are generated once and kept, because the same few positions are scored
 * against every read until a move is played.
 */

import { Chess } from 'chessops/chess';
import { makeBoardFen, makeFen } from 'chessops/fen';
import { makeSan } from 'chessops/san';
import { SquareSet } from 'chessops/squareSet';
import type { Move, Role } from 'chessops/types';
import { makeUci, roleToChar } from 'chessops/util';

import { CLASSES, EMPTY_CLASS_INDEX, GRID } from '#lib/protocol/index.ts';
import { SQUARES, type Labels } from './emission.ts';

export interface PositionNode {
  /** The FEN without its move counters, which differ between move orders. */
  readonly key: string;
  /** Never played on: clone it first. */
  readonly position: Chess;
  readonly labels: Labels;
  /** The FEN placement field, as `PositionEvent.placement` spells it. */
  readonly placement: string;
}

/** One legal move out of a node. */
export interface Edge {
  readonly san: string;
  readonly uci: string;
  /** A promotion to anything but a queen: rarer, and penalised by the decoder. */
  readonly underpromotion: boolean;
  readonly to: PositionNode;
}

const CLASS_INDEX: ReadonlyMap<string, number> = new Map(CLASSES.map((name, index) => [name, index]));

/** Queen first, so a tie between promotions resolves to the queen. */
const PROMOTIONS: readonly Role[] = ['queen', 'knight', 'rook', 'bishop'];

/** The board as class indices, in grid order (row 0 = rank 8; chessops' square 0 is a1). */
export function labelsOf(position: Chess): Labels {
  const labels = new Uint8Array(SQUARES);
  for (let square = 0; square < SQUARES; square++) {
    const row = Math.floor(square / GRID);
    const piece = position.board.get((GRID - 1 - row) * GRID + (square % GRID));
    if (piece === undefined) {
      labels[square] = EMPTY_CLASS_INDEX;
      continue;
    }
    const char = roleToChar(piece.role);
    labels[square] = CLASS_INDEX.get(piece.color === 'white' ? char.toUpperCase() : char)!;
  }
  return labels;
}

function positionKey(position: Chess): string {
  return makeFen(position.toSetup()).split(' ').slice(0, 4).join(' ');
}

/** Above this many interned nodes the table starts afresh; live hypotheses keep theirs. */
const MAX_NODES = 50_000;

export class PositionGraph {
  readonly #nodes = new Map<string, PositionNode>();
  readonly #edges = new WeakMap<PositionNode, readonly Edge[]>();

  /** The standard initial position. */
  start(): PositionNode {
    return this.#intern(Chess.default());
  }

  /** Every legal move from `node`, generated on first use. */
  edges(node: PositionNode): readonly Edge[] {
    const known = this.#edges.get(node);
    if (known !== undefined) return known;
    const edges: Edge[] = [];
    const backranks = SquareSet.backranks();
    for (const [from, dests] of node.position.allDests()) {
      const pawn = node.position.board.get(from)?.role === 'pawn';
      for (const to of dests) {
        const roles = pawn && backranks.has(to) ? PROMOTIONS : [undefined];
        for (const promotion of roles) {
          const move: Move = promotion === undefined ? { from, to } : { from, to, promotion };
          const next = node.position.clone();
          const san = makeSan(next, move);
          next.play(move);
          edges.push({
            san,
            uci: makeUci(move),
            underpromotion: promotion !== undefined && promotion !== 'queen',
            to: this.#intern(next)
          });
        }
      }
    }
    this.#edges.set(node, edges);
    return edges;
  }

  #intern(position: Chess): PositionNode {
    const key = positionKey(position);
    const known = this.#nodes.get(key);
    if (known !== undefined) return known;
    if (this.#nodes.size >= MAX_NODES) this.#nodes.clear();
    const node: PositionNode = {
      key,
      position,
      labels: labelsOf(position),
      placement: makeBoardFen(position.board)
    };
    this.#nodes.set(key, node);
    return node;
  }
}
