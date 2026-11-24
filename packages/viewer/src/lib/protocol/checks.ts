/**
 * The wire rules no JSON Schema keyword can state, in TypeScript.
 *
 * Each is the twin of a pydantic validator in `board_reader_core` that the schema
 * names with `x-check`; `protocol/generated.ts` applies it as a refinement. The
 * Python validator is the definition; `protocol/samples.json` holds the cases both
 * sides must judge alike. Imports nothing, so the generated module can depend on
 * it without a cycle.
 */

const GRID = 8;
const RANK = /^[1-8PNBRQKpnbrqk]+$/;

/** FEN field 1: eight ranks, each expanding to eight squares (`protocol.parse_placement`). */
export function isPlacement(placement: string): boolean {
  const ranks = placement.split('/');
  return (
    ranks.length === GRID &&
    ranks.every(
      (rank) =>
        RANK.test(rank) &&
        [...rank].reduce((squares, ch) => squares + (ch >= '1' && ch <= '8' ? Number(ch) : 1), 0) ===
          GRID
    )
  );
}

/** The crop ends inside the frame: `x + w <= 1` and `y + h <= 1` (`control.CropRect`). */
export function cropInside(crop: {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}): boolean {
  return crop.x + crop.w <= 1 && crop.y + crop.h <= 1;
}
