"""Tests for start-position auto-calibration logic (autocalib.py)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import (
    START_PLACEMENT,
    dist_from_placement,
    encode_placement,
    parse_placement,
)

from board_reader_inference.autocalib import (
    START_OCC,
    detect_start,
    occupancy_grid,
    piece_brightness,
)

# 180°-rotated start position: colors change sides and the K/Q files swap.
START_ROT180 = "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr"


def _grid(placement: str) -> np.ndarray:
    return np.array(parse_placement(placement), dtype="<U5")


def _crops_for(placement: str, *, white: int = 240, black: int = 20) -> list[np.ndarray]:
    """64 synthetic crops whose piece-region brightness encodes piece color."""
    crops = []
    for row in parse_placement(placement):
        for cell in row:
            level = 128 if cell == "empty" else (white if cell.isupper() else black)
            crops.append(np.full((64, 64, 3), level, dtype=np.uint8))
    return crops


def test_start_occ_pattern():
    assert START_OCC.sum() == 32
    assert START_OCC[:2].all() and START_OCC[-2:].all()
    assert not START_OCC[2:6].any()


def test_occupancy_grid_from_dist():
    occ = occupancy_grid(dist_from_placement(START_PLACEMENT))
    assert (occ == START_OCC).all()


def test_piece_brightness_orders_bands():
    b = piece_brightness(_crops_for(START_PLACEMENT))
    assert b[-2:].mean() > b[:2].mean()  # White band (bottom) is brighter


def test_detect_start_canonical():
    m = detect_start(dist_from_placement(START_PLACEMENT), _crops_for(START_PLACEMENT))
    assert m is not None
    assert m.rotation == 0
    assert m.matches == 64
    assert m.placement == START_PLACEMENT
    assert m.calib_mask.all()


def test_detect_start_labels_match_crops_in_every_rotation():
    """The acid test: whatever orientation the corners were labeled in, the
    placement handed to calibration must describe exactly what is physically on
    each crop (else the bank learns swapped/rotated piece labels)."""
    for k in range(4):
        cropped_grid = np.rot90(_grid(START_PLACEMENT), -k)
        placement_seen = encode_placement(cropped_grid.tolist())
        m = detect_start(
            dist_from_placement(placement_seen), _crops_for(placement_seen)
        )
        assert m is not None, k
        assert m.rotation == k
        assert m.placement == placement_seen  # labels == pixel truth, per crop
        assert m.calib_mask.all()
        # And rotating the cropped view by `rotation` recovers the canonical board.
        assert (np.rot90(_grid(m.placement), m.rotation) == _grid(START_PLACEMENT)).all()


def test_detect_start_flipped_colors_resolved_by_brightness():
    """Occupancy is 180°-symmetric; brightness must pick White's side. A board
    photographed 'upside down' (White at the top of the cropped grid) yields the
    rotated placement, never a color-swapped canonical one."""
    m = detect_start(dist_from_placement(START_ROT180), _crops_for(START_ROT180))
    assert m is not None
    assert m.rotation == 2
    assert m.placement == START_ROT180


def test_detect_start_after_one_move_masks_moved_squares():
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    m = detect_start(dist_from_placement(after_e4), _crops_for(after_e4))
    assert m is not None
    assert m.rotation == 0
    assert m.matches == 62
    assert m.placement == START_PLACEMENT  # labels are the START, mask drops the delta
    mask = m.calib_mask.reshape(8, 8)
    assert not mask[6, 4]  # e2: expected occupied, actually empty
    assert not mask[4, 4]  # e4: expected empty, actually occupied
    assert mask.sum() == 62


def test_detect_start_rejects_midgame():
    mid = "r2qr2k/ppp2Qpp/8/2b1N3/3npn2/2N5/PPP2PPP/R3R1K1"
    assert detect_start(dist_from_placement(mid), _crops_for(mid)) is None


def test_detect_start_rejects_empty_board():
    empty = "8/8/8/8/8/8/8/8"
    assert detect_start(dist_from_placement(empty), _crops_for(empty)) is None


def test_build_exemplars_mask_drops_untrusted_squares():
    from board_reader_inference.centroid import build_exemplars
    from board_reader_inference.embed import LiteEmbedder

    crops = _crops_for(START_PLACEMENT)
    mask = np.ones(64, dtype=bool)
    mask[[3, 36, 52]] = False  # d8 (the only black queen), e4, e2
    clf = build_exemplars(
        LiteEmbedder(), crops, START_PLACEMENT, mask=mask, augment=False
    )
    assert clf.exemplars.shape[0] == 61
    assert "q" not in clf.labels.tolist()  # d8 was the sole q — masked out entirely


def test_match_predicted_start_exact_and_rotated():
    from board_reader_inference.autocalib import (
        match_predicted_start,
        start_placement_for_rotation,
    )

    for k in range(4):
        pred = [c for row in parse_placement(start_placement_for_rotation(k)) for c in row]
        rotation, agree, mask = match_predicted_start(pred)
        assert rotation == k and agree == 64 and mask.all()


def test_match_predicted_start_partial_agreement():
    from board_reader_inference.autocalib import match_predicted_start

    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    pred = [c for row in parse_placement(after_e4) for c in row]
    rotation, agree, mask = match_predicted_start(pred)
    assert rotation == 0 and agree == 62
    assert not mask[52] and not mask[36]  # e2 and e4 disagree -> excluded


def test_match_predicted_start_midgame_scores_low():
    from board_reader_inference.autocalib import APPEND_MIN_AGREE, match_predicted_start

    mid = "r2qr2k/ppp2Qpp/8/2b1N3/3npn2/2N5/PPP2PPP/R3R1K1"
    pred = [c for row in parse_placement(mid) for c in row]
    _, agree, _ = match_predicted_start(pred)
    assert agree < APPEND_MIN_AGREE


def test_resolve_rotation_keeps_plausible_orientation():
    from board_reader_inference.autocalib import resolve_rotation

    dist = dist_from_placement(START_PLACEMENT)
    k, out = resolve_rotation(dist)
    assert k == 0
    np.testing.assert_array_equal(out, dist)


def test_resolve_rotation_recovers_all_rotations():
    from board_reader_inference.autocalib import resolve_rotation

    mid = "3r2k1/ppp2pp1/7p/2Q5/5qbP/3B4/PPPRrP2/2K5"  # field photo 2026-07-18
    dist = np.asarray(dist_from_placement(mid))
    for applied in range(4):
        seen = np.rot90(dist, -applied, axes=(0, 1))  # what mislabeled corners yield
        k, out = resolve_rotation(seen)
        assert (k + (4 - applied)) % 4 == 0 or (applied == 0 and k == 0)
        np.testing.assert_array_equal(out, dist)


def test_resolve_rotation_flips_white_playing_down():
    """180°-mislabeled corners: pawns stay off the back ranks in both flips, so
    the piece-mass tie-break must catch it (the 20260718-0002 field case)."""
    from board_reader_inference.autocalib import resolve_rotation

    grid = np.array(parse_placement(START_PLACEMENT), dtype="<U5")
    flipped = encode_placement(np.rot90(grid, 2).tolist())
    k, out = resolve_rotation(dist_from_placement(flipped))
    assert k == 2
    assert encode_placement(
        [list(row) for row in np.array(parse_placement(flipped))[::-1, ::-1]]
    ) == START_PLACEMENT
    np.testing.assert_array_equal(out, np.asarray(dist_from_placement(START_PLACEMENT)))
