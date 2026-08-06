"""Start-position auto-calibration (DESIGN §4.6, the zero-setup path).

Cross-piece-style transfer of piece *type* does not work on frozen DINOv2
features (measured: ChessReD exemplar bank 255/320 and a ChessReD-trained linear
probe 254/320 zero-shot, vs 294/320 same-style calibrated) — but *occupancy* is
style-robust (94–96% zero-shot). This module exploits that split: the occupancy
pattern of a chess STARTING position (back two ranks full on both sides, middle
empty) is unmistakable in occupancy alone, so the moment an uncalibrated server
sees a start position it can silently calibrate piece types from that very photo
via the existing exemplar-bank machinery.

Pure functions over the classifier's ``square_dist`` and the 64 photo crops —
no I/O, no estimator state. The gateway (server.py) does the wiring.

Orientation: the start pattern is 180°-symmetric, so occupancy alone only
resolves the board's rotation up to 180°. The remaining flip (which side is
White) is decided by piece brightness — white pieces are the brighter band in
essentially every physical set. If that heuristic ever guesses wrong, the bank
comes out color-swapped; the UI's rotate/recalibrate affordances remain the
manual override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_core.protocol import (
    EMPTY,
    GRID,
    START_PLACEMENT,
    encode_placement,
    parse_placement,
)

# Occupancy of the start position in canonical orientation (row 0 = rank 8):
# ranks 8,7,2,1 occupied, middle four ranks empty.
START_OCC: np.ndarray = np.zeros((GRID, GRID), dtype=bool)
START_OCC[:2] = True
START_OCC[-2:] = True

# Minimum occupancy squares (of 64) that must match the start pattern to trigger
# auto-calibration. 60 admits a couple of opening moves already played (1.e4 = 62
# matching squares, 1.e4 c5 = 60) — safe, because mismatching squares are excluded
# from calibration anyway (calib_mask) — while any real middlegame is far below.
START_MIN_MATCHES = 60


# Calibrated-path bank refresh (server): a start-position sighting re-calibrates
# only when mean confidence is below APPEND_MAX_CONF (i.e. lighting has drifted
# from what the bank covers), only from squares whose PREDICTION already agrees
# with the start placement (label-confirmed, so a wrong-orientation read or a
# mid-game photo can never pollute the bank), and never past APPEND_BANK_CAP.
APPEND_MIN_AGREE = 55
APPEND_MAX_CONF = 0.8
APPEND_BANK_CAP = 20_000


def match_predicted_start(pred_labels: list[str]) -> tuple[int, int, np.ndarray]:
    """Best rotation aligning a PREDICTED placement to the start position.

    Returns ``(rotation, agree, mask)``: rotation as in :class:`StartMatch`,
    ``agree`` = squares whose prediction equals the rotated start label, and
    ``mask`` (64 bools, cropped order) marking exactly those squares — the only
    ones safe to learn from.
    """
    grid = np.asarray(pred_labels, dtype="<U5").reshape(GRID, GRID)
    best: tuple[int, int, np.ndarray] | None = None
    for k in range(4):
        m = grid == np.rot90(_start_grid(), -k)
        c = int(m.sum())
        if best is None or c > best[1]:
            best = (k, c, m.reshape(GRID * GRID))
    assert best is not None
    return best


@dataclass(frozen=True)
class StartMatch:
    """A detected start position, in the AS-CROPPED square order.

    ``rotation`` is how many ``np.rot90`` turns take the cropped grid to canonical
    orientation (a8 top-left, White at the bottom rows) — it exists for reporting;
    ``placement``/``calib_mask`` are already expressed in cropped order, ready to
    feed :func:`board_reader_inference.centroid.build_exemplars` directly.
    """

    rotation: int  # 0..3
    matches: int  # occupancy squares matching the start pattern, of 64
    placement: str  # START_PLACEMENT as seen through the cropped orientation
    calib_mask: np.ndarray  # (64,) bool — squares whose occupancy agrees; safe to learn


def occupancy_grid(square_dist: np.ndarray) -> np.ndarray:
    """(8,8,13) distribution -> (8,8) bool occupancy (argmax is not ``empty``)."""
    dist = np.asarray(square_dist)
    if dist.shape != (GRID, GRID, dist.shape[-1]):
        raise ValueError(f"expected ({GRID},{GRID},C) square_dist, got {dist.shape}")
    return dist.argmax(axis=-1) != EMPTY


def piece_brightness(crops: list[np.ndarray]) -> np.ndarray:
    """(8,8) mean brightness of each crop's piece region (lower-central patch).

    Coarse by design: it is only ever *compared between the two occupied bands* of
    a start position, where the equal light/dark square backgrounds cancel and the
    piece pixels dominate the difference.
    """
    if len(crops) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} crops, got {len(crops)}")
    vals = np.empty(GRID * GRID, dtype=np.float64)
    for i, crop in enumerate(crops):
        gray = crop.mean(axis=2) if crop.ndim == 3 else crop
        h, w = gray.shape
        vals[i] = float(gray[int(h * 0.40) : int(h * 0.95), int(w * 0.25) : int(w * 0.75)].mean())
    return vals.reshape(GRID, GRID)


def _start_grid() -> np.ndarray:
    return np.array(parse_placement(START_PLACEMENT), dtype="<U5")


def start_placement_for_rotation(rotation: int) -> str:
    """:data:`START_PLACEMENT` as seen through a cropped orientation that is
    ``rotation`` quarter-turns away from canonical (cf. :class:`StartMatch`)."""
    return encode_placement(np.rot90(_start_grid(), -rotation).tolist())


def detect_start(
    square_dist: np.ndarray,
    crops: list[np.ndarray],
    *,
    min_matches: int = START_MIN_MATCHES,
) -> StartMatch | None:
    """Detect a start position in an (any-orientation) board reading, or ``None``.

    Rotation is chosen so the occupancy pattern lines up with :data:`START_OCC`
    (this also self-corrects 90°-off corner detection); the 180° ambiguity left by
    the pattern's symmetry is resolved by :func:`piece_brightness` (White = the
    brighter occupied band). The returned ``calib_mask`` keeps only squares whose
    occupancy agrees with the pattern, so a photo taken one or two moves *after*
    the start (or with a couple of occupancy misreads) never contributes wrongly
    labeled exemplars.
    """
    occ = occupancy_grid(square_dist)
    scores = {k: int((np.rot90(occ, k) == START_OCC).sum()) for k in (0, 1)}
    axis = max(scores, key=lambda k: scores[k])  # 180° partner ties by symmetry
    matches = scores[axis]
    if matches < min_matches:
        return None

    bright = np.rot90(piece_brightness(crops), axis)
    occ_axis = np.rot90(occ, axis)
    top = bright[:2][occ_axis[:2] & START_OCC[:2]]
    bottom = bright[-2:][occ_axis[-2:] & START_OCC[-2:]]
    if top.size == 0 or bottom.size == 0:  # degenerate; can't resolve White's side
        return None
    rotation = axis if bottom.mean() >= top.mean() else (axis + 2) % 4

    # Express the start placement and the agreement mask in cropped order:
    # canonical = rot90(cropped, rotation)  =>  cropped = rot90(canonical, -rotation).
    placement = start_placement_for_rotation(rotation)
    expected_occ = np.rot90(START_OCC, -rotation)
    calib_mask = (occ == expected_occ).reshape(GRID * GRID)
    return StartMatch(
        rotation=rotation,
        matches=matches,
        placement=placement,
        calib_mask=calib_mask,
    )


def resolve_rotation(square_dist: np.ndarray) -> tuple[int, np.ndarray]:
    """Pick the grid rotation under which the prediction looks like a chess game.

    Which physical corner is a8 is unknowable from geometry (DESIGN decision
    #15), and auto-detected corners come back in arbitrary orientation — so a
    mislabeled quad yields a perfectly classified but rotated position (seen in
    the field 2026-07-18: a 180°-flipped board read as White playing down).
    Among the four rotations of the predicted grid, prefer

    1. fewest pawns standing on ranks 1/8 — pawns can never stand there;
    2. as tie-break, White's men nearer rank 1 and Black's nearer rank 8
       (maximized piece-mass score), which settles pawn-less ties and 180° flips.

    Returns ``(k, rotated_dist)`` where ``k`` is the number of ``np.rot90``
    quarter-turns applied (0 = unchanged; ties prefer smaller ``k``, so a
    plausible as-given orientation is never second-guessed).
    """
    from board_reader_core.protocol import CLASSES

    dist = np.asarray(square_dist, dtype=np.float32)
    if dist.shape != (GRID, GRID, len(CLASSES)):
        raise ValueError(f"expected ({GRID},{GRID},{len(CLASSES)}) square_dist, got {dist.shape}")
    best: tuple[tuple[int, float], int] | None = None
    for k in range(4):
        grid = np.rot90(dist, k, axes=(0, 1)).argmax(axis=-1)
        pawns_on_back = 0
        mass = 0.0
        for r in range(GRID):
            for c in range(GRID):
                cls = CLASSES[grid[r, c]]
                if cls == "empty":
                    continue
                if cls in ("P", "p") and r in (0, GRID - 1):
                    pawns_on_back += 1
                # Row 0 = rank 8. White wants high r (near rank 1), Black low r.
                mass += r if cls.isupper() else (GRID - 1 - r)
        key = (pawns_on_back, -mass)
        if best is None or key < best[0]:
            best = (key, k)
    assert best is not None
    k = best[1]
    return k, np.rot90(dist, k, axes=(0, 1))
