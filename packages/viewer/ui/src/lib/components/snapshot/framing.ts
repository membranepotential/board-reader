/**
 * Where the board is in the photo, and which corner of it is a8.
 *
 * Two separate facts, and the page has always had to carry both: `/detect` returns an
 * *unlabelled* quad in geometric TL,TR,BR,BL order, because which physical corner is
 * a8 is not knowable from pixels (DESIGN decision #15). So the labelling is a rotation
 * layered over four points in storage order — a rotation relabels, it never moves a
 * point — plus two flags that decide who owns the labelling:
 *
 *  - `edited`: the operator has placed, dragged or rotated a corner, so their
 *    labelling is authoritative and reads stop asking the node to auto-rotate.
 *  - `adopted`: quarter-turns this page took over from a node auto-rotation. It is
 *    purely explanatory (the turn itself already lives in `rot`), but without it the
 *    corner table cannot tell "the labels are still an unverified guess" apart from
 *    "the node rotated them and we took its word for it".
 *
 * "Four corners are placed" is in the type rather than in a `points.length === 4`
 * check at every call site: `quadOf` is the only way to reach a `CornerQuad`, and a
 * `CornerQuad` is the only thing `/estimate`, `/calibrate` and the rectifier accept.
 */

import {
  CORNER_ORDER,
  absorbRotation,
  asRotation,
  labelOf,
  nextRotation,
  quarterTurnsOf,
  type CornerLabel,
  type LabelRotation
} from '$lib/geometry/corners';
import type { CornerQuad, Point } from '$lib/protocol';

/**
 * Corners as they are being entered. `placing` holds fewer than four points (the
 * guided tap-by-tap path); `placed` holds exactly four and is the only variant
 * anything downstream accepts.
 */
export type CornerPlacement =
  | { readonly kind: 'placing'; readonly points: readonly Point[] }
  | { readonly kind: 'placed'; readonly points: CornerQuad };

export interface Framing {
  readonly placement: CornerPlacement;
  readonly rot: LabelRotation;
  readonly edited: boolean;
  readonly adopted: LabelRotation;
  /** `/detect` found no board on this photo — the hint says so and manual entry opens. */
  readonly detectFailed: boolean;
}

/**
 * The one place a list of points becomes a placement. A length other than 0-4 is a bug
 * in this module, never wire data — the detector's reply is length-checked and refused
 * by `parseDetectResponse` long before it gets here.
 */
export function placementOf(points: readonly Point[]): CornerPlacement {
  if (points.length > 4) throw new RangeError(`a board has four corners, got ${points.length}`);
  return points.length === 4
    ? { kind: 'placed', points: points as unknown as CornerQuad }
    : { kind: 'placing', points };
}

export const EMPTY_FRAMING: Framing = {
  placement: { kind: 'placing', points: [] },
  rot: 0,
  edited: false,
  adopted: 0,
  detectFailed: false
};

export function quadOf(framing: Framing): CornerQuad | null {
  return framing.placement.kind === 'placed' ? framing.placement.points : null;
}

export function pointCount(framing: Framing): number {
  return framing.placement.points.length;
}

/**
 * One more corner, from a tap on the photo. Hand-placed corners are the operator's
 * assertion of where a8 is, so this marks the labelling as theirs.
 */
export function withPoint(framing: Framing, point: Point): Framing {
  if (framing.placement.kind === 'placed') return framing;
  return {
    ...framing,
    placement: placementOf([...framing.placement.points, point]),
    edited: true
  };
}

/** A handle dragged. Same ownership rule: touching a corner is asserting the labelling. */
export function withMovedPoint(framing: Framing, index: number, point: Point): Framing {
  const points = framing.placement.points.map((p, i) => (i === index ? point : p));
  return { ...framing, placement: placementOf(points), edited: true };
}

export function withoutLastPoint(framing: Framing): Framing {
  return { ...framing, placement: placementOf(framing.placement.points.slice(0, -1)) };
}

/**
 * Back to no corners at all. `detectFailed` is cleared with them: the hint it drives
 * describes the corners on screen, and there are none.
 */
export function clearedFraming(framing: Framing): Framing {
  return { ...framing, placement: { kind: 'placing', points: [] }, rot: 0, adopted: 0, detectFailed: false };
}

/**
 * The detector's quad, adopted whole. `edited` goes back to false — the orientation is
 * a guess, so the node may still auto-rotate the reading.
 */
export function detectedFraming(quad: CornerQuad): Framing {
  return {
    placement: { kind: 'placed', points: quad },
    rot: 0,
    edited: false,
    adopted: 0,
    detectFailed: false
  };
}

export function failedDetection(): Framing {
  return { ...EMPTY_FRAMING, detectFailed: true };
}

/**
 * "Rotate labels": the next quarter-turn, points untouched. A hand rotation makes the
 * labelling the operator's own, so whatever the node once contributed to `rot` is no
 * longer the explanation for where a8 sits.
 */
export function rotatedFraming(framing: Framing): Framing {
  return { ...framing, rot: nextRotation(framing.rot), edited: true, adopted: 0 };
}

/**
 * Absorb a rotation the node applied to a reading (`autorotated`, in degrees).
 *
 * Left alone the page would hold TWO orientations at once — the board showing the
 * rotated position while the handles, the corner table, the rectified preview and the
 * per-square crop still describe the pre-rotation frame. That was the reported symptom:
 * "the position is correct, but the corners are not aligned and the crop view shows the
 * wrong square."
 *
 * `edited` stays FALSE on purpose: the corners are still auto-detected, nobody has
 * asserted an orientation by hand, so later reads keep asking for auto-rotation.
 */
export function absorbedFraming(framing: Framing, degrees: number): Framing {
  const turns = quarterTurnsOf(degrees);
  if (turns === 0) return framing;
  return { ...framing, rot: absorbRotation(framing.rot, turns), adopted: turns };
}

/** What the corner at storage index `i` is called under this framing. */
export function labelAt(framing: Framing, index: number): CornerLabel {
  return labelOf(index, framing.rot);
}

/** The corner the operator is being asked for next, while placing. */
export function nextLabel(framing: Framing): CornerLabel {
  return labelOf(pointCount(framing), framing.rot);
}

// ---- the hint under the photo ------------------------------------------------

/**
 * The hint is a sentence with one emphasised corner name in it, so it is returned as
 * segments rather than as markup. The hand-written page built this string with
 * `innerHTML` and `<b>`; nothing on this page does that any more.
 */
export interface HintSegment {
  readonly text: string;
  readonly strong: boolean;
}

export function cornerHint(framing: Framing): readonly HintSegment[] {
  const plain = (text: string): HintSegment => ({ text, strong: false });
  const strong = (text: string): HintSegment => ({ text, strong: true });
  if (framing.placement.kind === 'placed') {
    return [
      plain('Check the labels: '),
      strong('a8'),
      plain(" must sit on the board's a8 corner — use "),
      strong('rotate labels'),
      plain(' if not. Drag any handle to fine-tune.')
    ];
  }
  const placed = pointCount(framing);
  const lead: HintSegment[] = framing.detectFailed
    ? [plain("Auto-detect didn't find the board — tap the corners yourself. ")]
    : [];
  return [
    ...lead,
    plain('Tap the '),
    strong(nextLabel(framing)),
    plain(
      ` corner of the playing surface (${placed}/4). ` +
        'A handle appears there — drag it to fine-tune.'
    )
  ];
}

/** How the corner table describes where these corners came from. */
export function cornerProvenance(framing: Framing): string {
  if (framing.detectFailed) return 'hand-placed';
  return framing.edited ? 'auto-detected, edited' : 'auto-detected';
}

/** How the corner table describes who owns the labelling for the next read. */
export function autoRotateNote(framing: Framing): string {
  if (framing.edited) return 'off (your labelling wins)';
  if (framing.adopted !== 0) {
    return (
      `on — the server rotated the labels by ${framing.adopted * 90} degrees ` +
      'and this page adopted it'
    );
  }
  return 'on — the labels are an unverified guess the server may still rotate';
}

/** The rotation line: total turn, and how much of it the node contributed. */
export function rotationNote(framing: Framing): string {
  const total = `${framing.rot * 90}deg`;
  return framing.adopted === 0
    ? `${total} set on this page`
    : `${total} (${framing.adopted * 90}deg of it adopted from the server)`;
}

export { CORNER_ORDER, asRotation };
export type { CornerLabel, LabelRotation };
