"""Tests for 64-square cropping (WP-B1), rectified and full-piece photo variants."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import GRID, START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view

from board_reader_inference.squares import (
    BboxGeometry,
    PoseGeometry,
    _board_to_image,
    _focal_from_homography,
    board_pose,
    crop_squares,
    crop_squares_photo,
    full_frame_corners,
    geometry_from_dict,
    geometry_tag,
    photo_square_boxes,
    rectify_photo,
    rot90_points,
    square_boxes,
)


def _near_white(crop: np.ndarray) -> bool:
    """Does the crop contain the white glyph colour (245,245,245) of sim pieces?"""
    return bool((crop > 235).all(axis=-1).any())


def test_crop_squares_photo_geometry_and_content():
    board = render_board(START_PLACEMENT, square_px=48)
    photo, quad = simulate_angled_view(board)  # quad = a8,h8,h1,a1 ground truth
    crops = crop_squares_photo(photo, quad, size=224)
    assert len(crops) == 64
    assert all(c.shape == (224, 224, 3) for c in crops)
    # e2 holds a white pawn glyph; e4's tall box looks over empty e4..e6 only
    # (headroom extends upward = toward rank 8 in the angled sim view).
    e2, e4 = crops[6 * 8 + 4], crops[4 * 8 + 4]
    assert _near_white(e2), "the piece glyph must land in its own square's crop"
    assert not _near_white(e4), "an empty square's crop must not contain a glyph"


def test_crop_squares_photo_clips_at_image_edges():
    board = render_board(square_px=32)
    h, w = board.shape[:2]
    # Board fills the whole frame: rank-8 boxes' headroom pokes above the image.
    crops = crop_squares_photo(board, full_frame_corners(board), size=64)
    assert len(crops) == 64 and all(c.shape == (64, 64, 3) for c in crops)


def test_rectify_photo_roundtrips_the_sim_warp():
    board = render_board(square_px=48)
    photo, quad = simulate_angled_view(board)
    rect = rectify_photo(photo, quad, out_size=board.shape[0])
    # The un-warped board should closely match the original rendering.
    diff = np.abs(rect.astype(int) - board.astype(int)).mean()
    assert diff < 12, f"rectified board deviates too much (mean abs diff {diff:.1f})"


def test_crops_64_squares_in_protocol_order():
    board = render_board(START_PLACEMENT, square_px=48)
    crops = crop_squares(board)
    assert len(crops) == 64
    # Index r*8+c is rank 8-r, file a+c. Top-left (index 0) is a8 = a black rook square,
    # which differs from the empty middle (index 4*8 = a4).
    a8, a4 = crops[0], crops[32]
    assert a8.shape[0] > 0 and a4.shape[0] > 0
    assert not np.array_equal(a8, a4)


def test_resize_and_inset_change_crop_size():
    board = render_board(square_px=48)
    full = crop_squares(board)[0]
    inset = crop_squares(board, inset=0.2)[0]
    sized = crop_squares(board, size=32)[0]
    assert inset.shape[0] < full.shape[0]  # inset trims edges
    assert sized.shape[:2] == (32, 32)


def test_inset_out_of_range_rejected():
    board = render_board(square_px=24)
    for bad in (-0.1, 0.5, 0.9):
        try:
            crop_squares(board, inset=bad)
        except ValueError:
            continue
        raise AssertionError(f"inset={bad} should have raised")


def test_photo_square_boxes_shape():
    board = render_board(square_px=32)
    boxes = photo_square_boxes(full_frame_corners(board))
    assert boxes.shape == (64, 4, 2)
    assert boxes.dtype == np.float32


def test_photo_square_boxes_are_the_flat_squares_at_zero_headroom_and_pad():
    """On a flat full-frame image, headroom=0/side_pad=0 boxes ARE the 64 squares."""
    board = render_board(square_px=32)
    h, w = board.shape[:2]
    sh, sw = h / GRID, w / GRID
    boxes = photo_square_boxes(
        full_frame_corners(board), headroom=0.0, side_pad=0.0
    )
    # index 0 = a8 = top-left square; index 32 = a4 (row 4, col 0).
    expected0 = np.array(
        [[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32
    )
    expected32 = np.array(
        [[0, 4 * sh], [sw, 4 * sh], [sw, 5 * sh], [0, 5 * sh]], dtype=np.float32
    )
    assert np.allclose(boxes[0], expected0, atol=1e-3)
    assert np.allclose(boxes[32], expected32, atol=1e-3)


def test_photo_square_boxes_from_shifted_corners_diverges_from_rot90_with_padding():
    """A pitfall the /estimate autorotate debug fix avoids: both ``headroom``
    (upward) and ``side_pad`` (sideways) extend a box along fixed IMAGE axes,
    scaled by a board-relative width — independent of which corner is labeled
    a8. So re-deriving boxes from a corner order shifted by k is NOT the same
    as rotating the already-computed boxes by k: the extension direction
    rotates with the (wrong) re-derivation too, so it samples the wrong pixels
    for a 90/270-degree turn. Confirmed here for headroom alone, side_pad
    alone, and both together (each ~1.5-2px on a 32px-square board — the
    divergence grows with the padding, e.g. ~6px at headroom=0.6). Only with
    *both* paddings at zero (no extension to misdirect) do the two agree —
    the special case that made this easy to miss."""
    board = render_board(square_px=32)
    photo, quad = simulate_angled_view(board)  # a8,h8,h1,a1, perspective-warped
    for headroom, side_pad in [(0.2, 0.0), (0.0, 0.15), (0.2, 0.15)]:
        boxes = photo_square_boxes(quad, headroom=headroom, side_pad=side_pad)
        grid = boxes.reshape(8, 8, 4, 2)
        for k in (1, 3):
            shifted = quad[[(i + k) % 4 for i in range(4)]]
            recomputed = photo_square_boxes(shifted, headroom=headroom, side_pad=side_pad)
            rotated = np.rot90(grid, k, axes=(0, 1)).reshape(-1, 4, 2)
            assert not np.allclose(recomputed, rotated, atol=1e-2)

    boxes0 = photo_square_boxes(quad, headroom=0.0, side_pad=0.0).reshape(8, 8, 4, 2)
    shifted1 = quad[[(i + 1) % 4 for i in range(4)]]
    recomputed0 = photo_square_boxes(shifted1, headroom=0.0, side_pad=0.0)
    assert np.allclose(
        recomputed0, np.rot90(boxes0, 1, axes=(0, 1)).reshape(-1, 4, 2), atol=1e-2
    )


def test_crop_squares_photo_behavior_unchanged():
    """crop_squares_photo's output is unaffected by exposing photo_square_boxes."""
    board = render_board(START_PLACEMENT, square_px=48)
    photo, quad = simulate_angled_view(board)
    crops = crop_squares_photo(photo, quad, size=224)
    assert len(crops) == 64
    assert all(c.shape == (224, 224, 3) for c in crops)


# --------------------------------------------------------------------------- #
# Pose geometry (PoseGeometry): boxes from the recovered camera pose
# --------------------------------------------------------------------------- #


def test_pose_boxes_agree_with_the_homography_on_the_board_plane():
    """The ``Z = 0`` face of each prism must be the square's own base quad.

    ``r1``, ``r2`` and ``t`` come straight from the homography precisely so that
    nothing about the existing, well-tested plane geometry changes — only the
    height is new. If this drifts, the pose path has stopped agreeing with the
    board it is standing on.
    """
    board = render_board(square_px=32)
    photo, quad = simulate_angled_view(board)
    flat = square_boxes(quad, BboxGeometry(0.0, 0.0), photo.shape)
    geometry = PoseGeometry(piece_height=1.9, piece_radius=0.5)
    K, r1, r2, r3t = board_pose(quad, photo.shape, geometry)
    r3, t = r3t[0], r3t[1]

    for r in range(GRID):
        for c in range(GRID):
            base = np.array(
                [[c, r, 0.0], [c + 1, r, 0.0], [c + 1, r + 1, 0.0], [c, r + 1, 0.0]]
            )
            cam = base @ np.stack([r1, r2, r3]) + t
            img = (K @ cam.T).T
            uv = img[:, :2] / img[:, 2:3]
            # `flat` is the bbox of that quad; compare bboxes, not vertex order.
            got = np.array(
                [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
            )
            want_quad = flat[r * GRID + c]
            want = np.array(
                [
                    want_quad[:, 0].min(),
                    want_quad[:, 1].min(),
                    want_quad[:, 0].max(),
                    want_quad[:, 1].max(),
                ]
            )
            assert np.allclose(got, want, atol=0.5), f"square r{r}c{c}: {got} != {want}"


def test_pose_box_height_tracks_piece_height_not_foreshortened_square_width():
    """The whole point: on an angled view, a real piece's projected height varies
    far less across the board than a square's projected width does.

    ``BboxGeometry`` ties box height to that width, so its near-rank boxes are
    much taller than its far-rank ones even though the pieces are identical —
    the documented steep-angle decapitation. ``PoseGeometry`` carries the height
    through the camera pose instead, so the spread collapses.
    """
    board = render_board(square_px=32)
    photo, quad = simulate_angled_view(board)

    def rank_heights(boxes):
        h = (boxes[:, 2, 1] - boxes[:, 0, 1]).reshape(GRID, GRID)
        return h.mean(axis=1)

    bbox_h = rank_heights(square_boxes(quad, BboxGeometry(1.6, 0.15), photo.shape))
    pose_h = rank_heights(
        square_boxes(quad, PoseGeometry(piece_height=1.9, piece_radius=0.5), photo.shape)
    )
    bbox_spread = float(bbox_h.max() / bbox_h.min())
    pose_spread = float(pose_h.max() / pose_h.min())
    assert pose_spread < bbox_spread, (
        f"pose spread {pose_spread:.2f} should be tighter than bbox {bbox_spread:.2f}"
    )
    # Near-rank boxes are still taller than far-rank ones (they are closer), just
    # by perspective alone rather than by foreshortening.
    assert 1.0 < pose_spread < 1.5


def test_pose_boxes_from_shifted_corners_match_rot90():
    """Pose boxes re-derived from a rotated corner labelling equal the rotated
    boxes — the property ``BboxGeometry`` provably lacks (see the divergence
    test above).

    The reason is physical: a bbox is extended along fixed *image* axes, so
    relabelling a8 silently rotates that extension. A prism stands along the
    *board normal*, which no relabelling can move. Worth pinning down: it means
    the autorotate debug path is exact under pose geometry rather than merely
    close.
    """
    board = render_board(square_px=32)
    photo, quad = simulate_angled_view(board)
    geometry = PoseGeometry(piece_height=1.9, piece_radius=0.5)
    grid = square_boxes(quad, geometry, photo.shape).reshape(GRID, GRID, 4, 2)
    for k in (1, 2, 3):
        shifted = quad[[(i + k) % 4 for i in range(4)]]
        recomputed = square_boxes(shifted, geometry, photo.shape)
        rotated = np.rot90(grid, k, axes=(0, 1)).reshape(-1, 4, 2)
        assert np.allclose(recomputed, rotated, atol=0.5), f"k={k}"


def test_focal_is_recovered_from_a_synthetic_projection():
    """``_focal_from_homography`` must recover the focal length that generated a
    view, since every pose box's height scales with it."""
    f_true = 900.0
    w_img, h_img = 1200, 900
    cx, cy = w_img / 2, h_img / 2
    K = np.array([[f_true, 0, cx], [0, f_true, cy], [0, 0, 1.0]])
    # A board tilted 55 degrees away from the camera and pushed back.
    a = np.deg2rad(55.0)
    R = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    t = np.array([-4.0, -2.0, 14.0])
    H = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    got = _focal_from_homography(H, cx, cy)
    assert got is not None
    assert abs(got - f_true) / f_true < 0.01, f"recovered {got}, want {f_true}"


def test_focal_falls_back_when_the_view_is_top_down():
    """A fronto-parallel board carries no information about focal length, so the
    solver must say so rather than return a fabricated number."""
    board = render_board(square_px=32)
    corners = full_frame_corners(board)
    H = _board_to_image(corners).astype(np.float64)
    assert _focal_from_homography(H, board.shape[1] / 2, board.shape[0] / 2) is None
    # ...and the caller still produces usable boxes via PoseGeometry.focal_ratio.
    boxes = square_boxes(corners, PoseGeometry(1.9, 0.5), board.shape)
    assert boxes.shape == (64, 4, 2) and np.isfinite(boxes).all()


def test_pose_geometry_needs_an_image_shape():
    board = render_board(square_px=32)
    with pytest.raises(ValueError, match="image_shape"):
        square_boxes(full_frame_corners(board), PoseGeometry(1.9, 0.5), None)


@pytest.mark.parametrize(
    "geometry",
    [BboxGeometry(1.6, 0.15), BboxGeometry(0.0, 0.0), PoseGeometry(1.9, 0.5, 1.0)],
)
def test_geometry_round_trips_through_its_dict_form(geometry):
    """The sidecar and the crop cache both persist geometry as a dict; a lossy
    round-trip here is a silent train/serve mismatch there."""
    assert geometry_from_dict(geometry.to_dict()) == geometry
    assert geometry_tag(geometry)


def test_legacy_sidecar_without_a_kind_is_bbox():
    """Sidecars written before pose geometry existed carry bare scalars."""
    assert geometry_from_dict({"headroom": 1.6, "side_pad": 0.25}) == BboxGeometry(1.6, 0.25)


def test_geometry_rejects_impossible_values():
    for bad in (lambda: BboxGeometry(-1.0, 0.15), lambda: BboxGeometry(1.6, -0.1),
                lambda: PoseGeometry(0.0, 0.5), lambda: PoseGeometry(1.9, -0.5),
                lambda: PoseGeometry(1.9, 0.5, 0.0)):
        with pytest.raises(ValueError):
            bad()


def test_rot90_points_tracks_pixels_through_np_rot90():
    """A marked pixel and its mapped coordinates land in the same place, every k."""
    image = np.zeros((4, 6), dtype=np.uint8)
    image[1, 2] = 255  # centre at continuous (2.5, 1.5)
    for k in range(1, 5):
        turned = np.rot90(image, k)
        (x, y) = rot90_points(np.array([[2.5, 1.5]]), k, image.shape)[0]
        r, c = int(np.argwhere(turned == 255)[0][0]), int(np.argwhere(turned == 255)[0][1])
        assert (x, y) == (c + 0.5, r + 0.5)


def test_rot90_points_full_turn_is_the_identity():
    image = np.zeros((4, 6), dtype=np.uint8)
    quad = full_frame_corners(image)
    assert np.allclose(rot90_points(quad, 4, image.shape), quad)


def test_rot90_points_keeps_labels_on_physical_corners():
    """Full-frame corners follow the frame; a8 (old top-left) lands where the
    counter-clockwise turn puts it — the bottom-left of the new frame."""
    image = np.zeros((4, 6), dtype=np.uint8)
    quad = full_frame_corners(image)
    moved = rot90_points(quad, 1, image.shape)
    assert np.allclose(moved[0], [0.0, 6.0])  # a8: old (0,0) -> new bottom-left
    assert np.allclose(sorted(map(tuple, moved)), sorted(map(tuple, full_frame_corners(np.rot90(image)))))
