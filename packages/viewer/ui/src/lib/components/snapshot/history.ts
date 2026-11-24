/**
 * Recent positions: this session's reads merged with the photos the node still holds.
 *
 * Every completed read is kept in memory — never persisted, the entries hold whole
 * photos — with everything needed to put the *whole* pipeline back: the photo, the
 * event, and any hand corrections made against it.
 * Restoring is how two reads get compared without re-shooting either, which is the core
 * diagnostic move on this page.
 *
 * The session list alone made every earlier photo the node is holding invisible, so
 * `GET /uploads` is merged into the same list. A photo this session read is shown as the
 * *session* entry — it owns the full event and its corrections, so it
 * restores instead of re-reading; the node's record of it only contributes the
 * `labelled` flag. A node that predates the endpoint (or is unreachable) simply leaves
 * the session list standing.
 */

import type { EstimateResponse, UploadRecord } from '$lib/protocol';
import type { CaptureFacts, SourceImage } from './photo';
import { fmt, type Corrections } from './reading';

/** How many reads are kept. Each holds a photo, so this is a memory bound. */
export const READS_CAP = 8;

/**
 * One read, with the pipeline context it ran in — enough to rebuild every stage, so the
 * evidence on screen always belongs to one and the same read and never a mix of two.
 */
export interface SessionRead {
  readonly id: number;
  readonly at: Date;
  readonly response: EstimateResponse;
  readonly corrections: Corrections;
  readonly image: SourceImage;
  readonly facts: CaptureFacts;
  readonly estimateMs: number | null;
}

export type HistoryRow =
  | {
      readonly kind: 'session';
      readonly key: string;
      readonly at: Date;
      readonly read: SessionRead;
      /** The node's own record of the same photo, when it has one. */
      readonly upload: UploadRecord | null;
    }
  | {
      readonly kind: 'node';
      readonly key: string;
      readonly at: Date | null;
      readonly upload: UploadRecord;
    };

export function uploadTime(upload: UploadRecord): Date | null {
  return upload.ts === null ? null : new Date(upload.ts * 1000);
}

/** One merged, newest-first list. A row's origin is a permanent property of the row. */
export function historyRows(
  reads: readonly SessionRead[],
  uploads: readonly UploadRecord[]
): readonly HistoryRow[] {
  const owned = new Set(reads.map((read) => read.response.upload).filter((id) => id !== null));
  const rows: HistoryRow[] = reads.map((read) => {
    const id = read.response.upload;
    return {
      kind: 'session',
      key: `s${read.id}`,
      at: read.at,
      read,
      upload: uploads.find((u) => u.id === id) ?? null
    };
  });
  for (const upload of uploads) {
    if (owned.has(upload.id)) continue;
    rows.push({ kind: 'node', key: `u${upload.id}`, at: uploadTime(upload), upload });
  }
  // A row whose timestamp is unreadable still belongs in the list — it just sorts last.
  return rows.sort((a, b) => (b.at?.getTime() ?? 0) - (a.at?.getTime() ?? 0));
}

/** `retention === null` means the node never answered `/uploads` at all. */
export function historyNote(retention: boolean | null): string {
  if (retention === false) {
    return 'Photo retention is off on the inference node — only this session’s reads are listed.';
  }
  if (retention === null) {
    return 'The inference node did not answer /uploads — only this session’s reads are listed.';
  }
  return (
    'This session’s reads plus every photo the inference node still holds. ' +
    'Clear removes only this session’s entries; the node’s own are listed again on the next refresh.'
  );
}

/** The short, comma-joined "what was unusual about this read" line. */
export function readFlags(read: SessionRead, upload: UploadRecord | null): readonly string[] {
  const flags: string[] = [];
  if (read.corrections.size) flags.push(`${read.corrections.size} corrected`);
  if (upload?.labeled) flags.push('labelled');
  flags.push(...read.response.warnings);
  return flags;
}

/** The metadata line under a session read's placement. */
export function readMeta(read: SessionRead): string {
  const bits = [read.at.toLocaleTimeString()];
  bits.push(`conf ${fmt(read.response.event.confidence, 3)}`);
  if (read.estimateMs !== null) bits.push(`${read.estimateMs.toFixed(0)} ms`);
  return bits.join(' · ');
}

export function uploadFlags(upload: UploadRecord): readonly string[] {
  const flags: string[] = [];
  if (upload.labeled) flags.push('labelled');
  if (upload.kind !== 'estimate') flags.push(upload.kind);
  return flags;
}
