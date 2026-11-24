/**
 * The evidence tables, as functions of the read.
 *
 * Each stage body's key/value table is a *view* of something the page already holds —
 * the distribution, the node's repair, the board as shown — so each one is
 * a pure function here rather than a block of markup or a `kv(...)` call buried in a
 * render path. That is what makes "what does the read stage say about this reply?"
 * a question with a test instead of a screenshot.
 */

import type { BoardGrid } from '$lib/board/fen';
import { LOW_CONFIDENCE, type EstimateResponse, type PriorDecode } from '$lib/protocol';
import { kv, kvIf, type KvRow } from './kv';
import { confidenceSummary, fmt, materialCensus } from './reading';

const ms = (value: number | null): string => (value === null ? '—' : `${value.toFixed(0)} ms`);

export function readTable(
  response: EstimateResponse | null,
  estimateMs: number | null
): readonly KvRow[] {
  if (response === null) return [];
  const summary = confidenceSummary(response.event.square_dist);
  return [
    kv('position confidence', fmt(response.event.confidence ?? 0, 3)),
    kv('mean square confidence', fmt(summary.mean, 3)),
    kvIf('weakest square', fmt(summary.min, 3), summary.min < LOW_CONFIDENCE),
    kvIf(`squares below ${LOW_CONFIDENCE}`, String(summary.low), summary.low > 0),
    kv('estimate latency', ms(estimateMs)),
    kv('status', response.event.status),
    kv('board / seq', `${response.event.board_id} / ${response.event.seq}`),
    kv('captured', response.event.utc)
  ];
}

/**
 * The repair, cheapest flip first. An empty `flips` list is the GOOD case — the read was
 * already a possible board — and is reported as such rather than hidden.
 */
export function repairTable(prior: PriorDecode | null): readonly KvRow[] {
  if (prior === null) return [kv('served', 'this build of the server does not decode a repair')];
  const rows: KvRow[] = prior.flips.map((flip, i) =>
    kv(
      `${i + 1}. ${flip.square}`,
      `${flip.from} → ${flip.to} · ${flip.constraint} · cost ${fmt(flip.cost, 3)}`
    )
  );
  const residual = Object.entries(prior.residual);
  if (residual.length) {
    rows.push(kv('unresolved', residual.map(([key, count]) => `${key} ${count}`).join(' · '), 'warn'));
  }
  return rows.length ? rows : [kv('flips', 'none — no constraint was violated')];
}

export function materialTable(grid: BoardGrid): readonly KvRow[] {
  const census = materialCensus(grid);
  return [
    kv('white', census.white),
    kv('black', census.black),
    kv('occupied squares', String(census.occupied)),
    kvIf(
      'implausible',
      census.implausible.length ? census.implausible.join(', ') : 'no — counts are legal',
      census.implausible.length > 0
    )
  ];
}
