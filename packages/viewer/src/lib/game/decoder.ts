/**
 * The PGN decoder: turns a board's stream of reads into the legal game that best
 * explains them (docs/plans/2026-11-24-pgn-decoder.md).
 *
 * This is the application layer DESIGN D1 keeps out of the CV layer: the node streams
 * positions, and this client reconstructs moves from them.
 *
 * It is a beam Viterbi decoder over legal positions, one step per read. Each
 * hypothesis is a position, the move path that reached it from the initial position,
 * and a log score. On each read every hypothesis may stay, play one legal ply, play
 * two (a move no read showed), or take back its last ply. Each candidate is scored by
 * how well its position explains the read (`./emission.ts`). Candidates that reach the
 * same position keep only the best path. The best `width` within `margin` of the best
 * survive. A move is **committed** once every survivor's path contains it: then no
 * explanation the beam still considers leaves it out.
 */

import type { Outcome } from 'chessops/types';

import type { SquareDist } from '#lib/protocol/index.ts';
import type { PositionEvent } from '#lib/protocol/generated.ts';
import {
  argmaxLabels,
  largestGains,
  logLikelihood,
  mismatches,
  squareCeilings,
  squareLogs,
  type EmissionParams
} from './emission.ts';
import { PositionGraph, type Edge, type PositionNode } from './positions.ts';

/** One read, as the decoder needs it. Times are ms since the epoch. */
export interface Observation {
  readonly dist: SquareDist;
  /** When this read was captured. */
  readonly at: number;
  /** The node's settled board, and when that board last changed. */
  readonly settled: { readonly placement: string; readonly since: number };
}

/** A read the decoder can use: only an `ok` event is a read of the board. */
export function observationOf(event: PositionEvent): Observation | null {
  if (event.status !== 'ok') return null;
  return {
    dist: event.square_dist,
    at: Date.parse(event.utc),
    settled: { placement: event.placement, since: Date.parse(event.changed_utc) }
  };
}

export interface DecodedMove {
  readonly san: string;
  readonly uci: string;
  /** When the move was made, ms since the epoch: the earliest read that showed it. */
  readonly at: number;
  /** Every hypothesis the beam keeps agrees on this move (and on every move before it). */
  readonly committed: boolean;
}

export interface DecodedGame {
  /** The first read's capture time, or null before any read. */
  readonly startedAt: number | null;
  /** The best path from the initial position. */
  readonly moves: readonly DecodedMove[];
  /** The FEN placement field after `moves`. */
  readonly placement: string;
  /** Checkmate, stalemate or insufficient material after `moves`; else undefined. */
  readonly outcome: Outcome | undefined;
  /** Squares where the latest read's own board disagrees with `placement`. */
  readonly mismatch: number;
}

export interface DecoderParams extends EmissionParams {
  /** Prior per read of one ply, of two plies, and of taking back the last ply. The rest stays. */
  readonly onePly: number;
  readonly twoPlies: number;
  readonly takeback: number;
  /** Extra factor on a promotion to anything but a queen. */
  readonly underpromotion: number;
  /** How many hypotheses survive a read, and how far behind the best (in nats) they may be. */
  readonly width: number;
  readonly margin: number;
}

/**
 * Chosen on the seven gate takes with `tools/pgn_eval.ts` (the plan's results table),
 * where every take decodes exactly for any margin from 8 to 25 and any temperature
 * from 1 to 4. With these, about two clean reads that show a move commit it.
 */
export const DEFAULT_PARAMS: DecoderParams = {
  temperature: 2,
  floor: 0.05,
  onePly: 0.1,
  twoPlies: 0.02,
  takeback: 0.001,
  underpromotion: 0.05,
  width: 8,
  margin: 12
};

/** Castling changes four squares; every other move two or three. */
const SQUARES_A_MOVE_CHANGES = 4;

/** One played ply: a node of a path, shared between the hypotheses that agree on it. */
interface Ply {
  readonly san: string;
  readonly uci: string;
  /** Moved earlier when a later settled board shows it began sooner (`refineTimes`). */
  at: number;
  readonly from: PositionNode;
  readonly parent: Ply | null;
}

interface Hypothesis {
  readonly node: PositionNode;
  readonly path: Ply | null;
  readonly score: number;
}

/** A candidate before pruning: its path is built only if it survives. */
interface Candidate {
  readonly node: PositionNode;
  readonly score: number;
  readonly parent: Ply | null;
  readonly steps: readonly (readonly [PositionNode, Edge])[];
}

function pathOf(ply: Ply | null): Ply[] {
  const plies: Ply[] = [];
  for (let at = ply; at !== null; at = at.parent) plies.push(at);
  return plies.reverse();
}

/** How many leading plies every path has in common. */
function sharedPrefix(paths: readonly (readonly Ply[])[]): number {
  const [first, ...rest] = paths;
  if (first === undefined) return 0;
  let shared = 0;
  while (
    shared < first.length &&
    rest.every((path) => shared < path.length && path[shared]!.uci === first[shared]!.uci)
  ) {
    shared += 1;
  }
  return shared;
}

export class GameDecoder {
  readonly #params: DecoderParams;
  readonly #graph = new PositionGraph();
  #beam: Hypothesis[];
  #startedAt: number | null = null;
  #lastAt = -Infinity;
  #mismatch = 0;
  #game: DecodedGame;

  constructor(params: DecoderParams = DEFAULT_PARAMS) {
    this.#params = params;
    this.#beam = [{ node: this.#graph.start(), path: null, score: 0 }];
    this.#game = this.#view();
  }

  get game(): DecodedGame {
    return this.#game;
  }

  /** Fold one read in. A read no newer than the last one changes nothing. */
  push(observation: Observation): DecodedGame {
    if (observation.at <= this.#lastAt) return this.#game;
    this.#lastAt = observation.at;
    this.#startedAt ??= observation.at;

    const logs = squareLogs(observation.dist, this.#params);
    this.#beam = this.#prune(this.#expand(logs), observation.at);
    this.#refineTimes(observation.settled);
    this.#mismatch = mismatches(argmaxLabels(observation.dist), this.#beam[0]!.node.labels);
    this.#game = this.#view();
    return this.#game;
  }

  /**
   * Every way each hypothesis can explain the read, the best per position.
   *
   * Two-ply candidates are branch-and-bound: a move changes at most four squares
   * (castling), so a reply can raise its position's likelihood by at most the four
   * largest per-square gains. A first ply whose bound is below the best candidate
   * minus the margin cannot survive `#prune`, so its replies are never generated; the
   * result is the same as expanding them all, at a fraction of the cost.
   */
  #expand(logs: Float64Array): Map<string, Candidate> {
    const { onePly, twoPlies, takeback, underpromotion, margin } = this.#params;
    const stay = Math.log(1 - onePly - twoPlies - takeback);
    const undo = Math.log(takeback);
    const under = Math.log(underpromotion);
    const ceilings = squareCeilings(logs);
    const candidates = new Map<string, Candidate>();
    let best = -Infinity;
    const offer = (node: PositionNode, prior: number, parent: Ply | null, steps: Candidate['steps']): number => {
      const likelihood = logLikelihood(logs, node.labels);
      const score = prior + likelihood;
      const known = candidates.get(node.key);
      if (known === undefined || score > known.score) candidates.set(node.key, { node, score, parent, steps });
      best = Math.max(best, score);
      return likelihood;
    };

    const replies: { bound: number; prior: number; first: Edge; from: PositionNode; path: Ply | null }[] = [];
    for (const { node, path, score } of this.#beam) {
      offer(node, score + stay, path, []);
      if (path !== null) offer(path.from, score + undo, path.parent, []);
      const firsts = this.#graph.edges(node);
      const one = score + Math.log(onePly / firsts.length);
      const two = score + Math.log(twoPlies / firsts.length);
      for (const first of firsts) {
        const prior = two + (first.underpromotion ? under : 0);
        const likelihood = offer(first.to, one + (first.underpromotion ? under : 0), path, [[node, first]]);
        const gain = largestGains(logs, ceilings, first.to.labels, SQUARES_A_MOVE_CHANGES);
        replies.push({ bound: prior + likelihood + gain, prior, first, from: node, path });
      }
    }

    replies.sort((a, b) => b.bound - a.bound);
    for (const { bound, prior, first, from, path } of replies) {
      if (bound < best - margin) break;
      const seconds = this.#graph.edges(first.to);
      const second = prior - Math.log(seconds.length);
      for (const reply of seconds) {
        offer(reply.to, second + (reply.underpromotion ? under : 0), path, [
          [from, first],
          [first.to, reply]
        ]);
      }
    }
    return candidates;
  }

  /** The best `width` within `margin` of the best, scores relative to it, paths built. */
  #prune(candidates: Map<string, Candidate>, at: number): Hypothesis[] {
    const { width, margin } = this.#params;
    const ranked = [...candidates.values()].sort((a, b) => b.score - a.score);
    const best = ranked[0]!.score;
    return ranked
      .filter((candidate) => candidate.score >= best - margin)
      .slice(0, width)
      .map(({ node, score, parent, steps }) => ({
        node,
        score: score - best,
        path: steps.reduce<Ply | null>(
          (previous, [from, edge]) => ({ san: edge.san, uci: edge.uci, at, from, parent: previous }),
          parent
        )
      }));
  }

  /**
   * `changed_utc` counts reads the node never published, so when a settled board shows
   * a hypothesis's position, its last move began at that board's `since` if that is
   * earlier than the read that took it (and not before the move ahead of it).
   */
  #refineTimes(settled: Observation['settled']): void {
    for (const { node, path } of this.#beam) {
      if (path === null || node.placement !== settled.placement) continue;
      const floor = path.parent?.at ?? this.#startedAt ?? -Infinity;
      if (settled.since < path.at && settled.since >= floor) path.at = settled.since;
    }
  }

  #view(): DecodedGame {
    const [best, ...others] = this.#beam;
    const path = pathOf(best!.path);
    const committed = sharedPrefix([path, ...others.map((h) => pathOf(h.path))]);
    return {
      startedAt: this.#startedAt,
      moves: path.map(({ san, uci, at }, index) => ({ san, uci, at, committed: index < committed })),
      placement: best!.node.placement,
      outcome: best!.node.position.outcome(),
      mismatch: this.#mismatch
    };
  }
}
