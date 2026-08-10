/**
 * Pure calibration-gating logic for the live page's "calibrate" button — ported
 * from index.html's `calibBlockReason` / `updateCalibState`.
 *
 * Calibration rebuilds the piece bank from one frame, so it is only meaningful on
 * a live, settled, `ok` reading — anything else would poison the bank with motion
 * blur or a half-set-up board. The reason a calibration is unavailable is the
 * button's `title`, never a silent disable.
 *
 * index.html additionally gated on `lastSeq === null` ("no frame has arrived
 * yet"), guarding a card that exists before its first position event. That state
 * is unreachable here: a `BoardCard` only ever renders for a `BoardState`, and
 * `BoardState.event` is a non-nullable `PositionEvent` by construction — so the
 * check is dropped rather than carried across as a branch that can never run.
 */

import { MOVING_BELOW, type CalibrateResponse, type PositionStatus } from '../../protocol';

export interface CalibGate {
  /** The stream has gone quiet — no live frame left to calibrate from. */
  readonly stale: boolean;
  readonly status: PositionStatus;
  /** `null` reads as "motion unknown", which blocks nothing on its own. */
  readonly stability: number | null;
}

/** Why calibration is blocked right now, or `null` when it is available. */
export function calibBlockReason(gate: CalibGate): string | null {
  if (gate.stale) return 'the stream is stale — no live frame to calibrate from';
  if (gate.status !== 'ok') return `the reading is "${gate.status}" — calibrate from an ok reading`;
  if (gate.stability !== null && gate.stability < MOVING_BELOW) {
    return 'the board is moving — calibrate with the pieces at rest';
  }
  return null;
}

/** The calibrate button's `title` — always something, never a silent disable. */
export function calibTitle(reason: string | null, busy: boolean): string {
  if (busy) return 'calibrating…';
  return (
    reason ?? 'rebuild the piece bank from the live frame (the board must show the starting position)'
  );
}

/** The success banner's text: classes calibrated, any label rotation, exemplar count. */
export function calibSuccessMessage(response: CalibrateResponse): string {
  const rotation = response.label_rotation
    ? ` (corner labels auto-corrected by ${response.label_rotation * 90} degrees)`
    : '';
  return (
    `Calibrated ${response.classes_calibrated}/13 classes${rotation} — ` +
    `bank now holds ${response.exemplars} exemplars`
  );
}
