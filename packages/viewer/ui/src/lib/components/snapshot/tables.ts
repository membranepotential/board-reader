/**
 * The evidence tables, as functions of the read.
 *
 * Each stage body's key/value table is a *view* of something the page already holds —
 * the corners, the quad's geometry, the distribution, the node's repair — so each one is
 * a pure function here rather than a block of markup or a `kv(...)` call buried in a
 * render path. That is what makes "what does stage 5 say about an occupancy-only reply?"
 * a question with a test instead of a screenshot.
 */

import type { BoardGrid } from '$lib/board/fen';
import type { QuadGeometry } from '$lib/geometry/homography';
import {
  LOW_ANGLE_ASPECT,
  LOW_CONFIDENCE,
  LOW_LIGHT_MEAN,
  type CornerQuad,
  type EstimateResponse,
  type PriorDecode
} from '$lib/protocol';
import { CORNER_ORDER, autoRotateNote, cornerProvenance, rotationNote, type Framing } from './framing';
import { kv, kvIf, type KvRow } from './kv';
import { confidenceSummary, fmt, materialCensus } from './reading';

const ms = (value: number | null): string => (value === null ? '—' : `${value.toFixed(0)} ms`);

/**
 * Where the four corners are, who put them there, and who owns the labelling. The
 * corners are listed in canonical a8,h8,h1,a1 order — the order that goes on the wire,
 * not storage order, so this table can be compared against `?corners=` directly.
 */
export function cornerTable(
  framing: Framing,
  ordered: CornerQuad | null,
  detectMs: number | null
): readonly KvRow[] {
  if (ordered === null) {
    const placed = framing.placement.points.length;
    return [kv('corners', `${placed}/4 placed`, placed ? 'warn' : 'plain')];
  }
  return [
    kv('source', cornerProvenance(framing), framing.edited || framing.detectFailed ? 'warn' : 'ok'),
    kv('detect latency', ms(detectMs)),
    kv('label rotation', rotationNote(framing)),
    ...CORNER_ORDER.map((label, i) => kv(label, `${ordered[i]?.[0]}, ${ordered[i]?.[1]}`)),
    kv(
      'server auto-rotate',
      autoRotateNote(framing),
      framing.edited || framing.adopted !== 0 ? 'ok' : 'warn'
    )
  ];
}

/** What the quad says about the camera, next to the two thresholds the node warns at. */
export function geometryTable(geometry: QuadGeometry, meanLuma: number): readonly KvRow[] {
  return [
    kvIf(
      'projected aspect',
      `${fmt(geometry.aspect, 3)} (warn below ${LOW_ANGLE_ASPECT})`,
      geometry.aspect < LOW_ANGLE_ASPECT
    ),
    kv(
      'mean edge width',
      `${fmt(geometry.w, 1)} px (a8-h8 ${fmt(geometry.top, 1)}, a1-h1 ${fmt(geometry.bottom, 1)})`
    ),
    kv(
      'mean edge height',
      `${fmt(geometry.h, 1)} px (a8-a1 ${fmt(geometry.left, 1)}, h8-h1 ${fmt(geometry.right, 1)})`
    ),
    kv('square pitch', `~${fmt(Math.min(geometry.w, geometry.h) / 8, 1)} px on the short axis`),
    kvIf('mean luma', `${fmt(meanLuma, 1)} (warn below ${LOW_LIGHT_MEAN})`, meanLuma < LOW_LIGHT_MEAN)
  ];
}

export function classTable(
  response: EstimateResponse | null,
  estimateMs: number | null
): readonly KvRow[] {
  if (response === null) return [];
  if (response.kind === 'occupancy') {
    return [
      kv('mode', 'occupancy (uncalibrated for this piece set)', 'warn'),
      kv('occupied squares', `${response.occupied.flat().filter(Boolean).length} / 64`),
      kv('estimate latency', ms(estimateMs)),
      kv('how to unlock', 'photograph the starting position once — the server self-calibrates')
    ];
  }
  const summary = confidenceSummary(response.event.square_dist);
  return [
    kv('mode', 'position (13-class per square)', 'ok'),
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

/** Who decided which corner is a8 for this read, and what the node did about it. */
export function orientTable(
  response: EstimateResponse | null,
  framing: Framing
): readonly KvRow[] {
  if (response === null) return [];
  if (response.kind === 'occupancy') {
    return [kv('orientation', 'unknown — occupancy carries no colour cue')];
  }
  const rotated = response.autorotated;
  return [
    kv('corner labels', framing.edited ? 'set by you' : 'auto-detected guess'),
    kv('autorotate requested', framing.edited ? 'no' : 'yes (&autorotate=1)'),
    kvIf('rotation applied', `${rotated} degrees`, rotated !== 0),
    kv(
      'adopted by this page',
      rotated
        ? `yes — the corner labels were shifted by ${rotated} degrees, so a8 on the photo is a8 on the board`
        : 'nothing to adopt',
      rotated ? 'ok' : 'plain'
    ),
    kv('page label rotation', rotationNote(framing)),
    kv('reported orientation', response.event.orientation)
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

/**
 * Whatever the node volunteered about the estimator's own state this read: a bank
 * bootstrapped, refreshed, or auto-calibrated from this very photo.
 */
export function estimatorNotes(response: EstimateResponse | null): readonly string[] {
  if (response === null || response.kind !== 'position') return [];
  const notes: string[] = [];
  const auto = response.autocalibrated;
  if (auto !== null) {
    notes.push(
      `Auto-calibrated from this photo: ${auto.matches}/64 squares agreed with the start ` +
        `position at rotation ${auto.rotation * 90} degrees; bank now holds ${auto.exemplars} ` +
        `exemplars${auto.saved === null ? ' (in memory only)' : ` (saved to ${auto.saved})`}.`
    );
  }
  const refresh = response.bank_refreshed;
  if (refresh !== null) {
    notes.push(
      `Bank self-refreshed: +${refresh.added} exemplars (total ${refresh.total}) from ` +
        `${refresh.agree} agreeing squares at rotation ${refresh.rotation * 90} degrees` +
        `${refresh.saved === null ? '' : `, saved to ${refresh.saved}`}.`
    );
  }
  return notes;
}
