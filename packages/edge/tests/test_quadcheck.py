"""Tests for the operator corner-quad sanity checks (DESIGN §4.5).

Pure geometry: nothing here builds a CLI, opens a socket or decodes a frame. Each
case is a quad an operator could plausibly type by mistake, and the assertion is
that it is named — the defect *kind*, not just "invalid" — because the whole point
of the check is to tell them which of the four numbers to look at.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_edge.quadcheck import (
    check_quad_in_frame,
    check_quad_shape,
    low_angle_warning,
)

# A plausible angled-broadcast board: the far edge foreshortened, in the operator's
# a8,h8,h1,a1 walk around the surface.
GOOD = np.array(
    [[320.0, 120.0], [960.0, 120.0], [1120.0, 700.0], [160.0, 700.0]], dtype=np.float32
)


def _kind(pts) -> str | None:
    defect = check_quad_shape(pts)
    return None if defect is None else defect.kind


def test_a_plausible_board_quad_passes():
    assert check_quad_shape(GOOD) is None


def test_winding_direction_does_not_matter():
    # a8,h8,h1,a1 walks clockwise in image space for a board seen from White's
    # side and anticlockwise from Black's; both are legitimate declarations.
    assert check_quad_shape(GOOD[::-1]) is None


@pytest.mark.parametrize("count", [6, 7, 10])
def test_wrong_number_of_coordinates_is_named(count):
    assert _kind(np.arange(count, dtype=np.float32)) == "not_a_quad"


def test_a_non_finite_coordinate_is_named():
    bad = GOOD.copy()
    bad[2, 1] = np.nan
    assert _kind(bad) == "not_a_quad"


def test_a_repeated_corner_is_named():
    bad = GOOD.copy()
    bad[3] = bad[0]  # tapped a8 twice
    assert _kind(bad) == "coincident_points"


def test_swapping_the_last_two_corners_reads_as_a_bowtie():
    """The classic slip: entering a8,h8,a1,h1 instead of a8,h8,h1,a1.

    Geometrically it is a self-intersecting quad, and its shoelace area nearly
    cancels — so this also pins the check order, since "degenerate" would send the
    operator looking for a mistyped coordinate that isn't there.
    """
    bad = GOOD[[0, 1, 3, 2]]
    defect = check_quad_shape(bad)
    assert defect is not None and defect.kind == "self_intersecting"
    assert "swapped" in defect.message


def test_a_concave_quad_is_named():
    bad = np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 400.0], [0.0, 1000.0]])
    assert _kind(bad) == "not_convex"


def test_a_board_too_small_to_hold_64_squares_is_named():
    assert _kind(np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]])) == (
        "degenerate"
    )


def test_a_sliver_is_named():
    # Big enough in area to clear the absolute floor, but 5 px tall.
    assert _kind(np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 5.0], [0.0, 5.0]])) == (
        "degenerate"
    )


def test_collinear_points_are_named():
    assert _kind(np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])) == (
        "degenerate"
    )


def test_corners_inside_the_frame_pass():
    assert check_quad_in_frame(GOOD, width=1280, height=720) is None


def test_corners_from_another_resolution_are_refused():
    """4K corners on a 720p stream — the failure the in-frame check exists for."""
    defect = check_quad_in_frame(GOOD * 3.0, width=1280, height=720)
    assert defect is not None and defect.kind == "out_of_frame"
    assert "1280x720" in defect.message
    # Names the offending corners so the operator can fix the right numbers.
    assert "2=" in defect.message and "3=" in defect.message


def test_a_corner_just_off_the_frame_edge_is_allowed():
    # A board whose near corner is cropped by the frame edge is a real setup; the
    # operator extrapolating that point slightly outside must not be refused.
    quad = GOOD.copy()
    quad[3] = [-40.0, 740.0]
    assert check_quad_in_frame(quad, width=1280, height=720) is None


def test_a_usable_camera_angle_gets_no_warning():
    assert low_angle_warning(GOOD) is None


def test_a_near_board_level_camera_warns():
    flat = np.array([[300.0, 200.0], [900.0, 200.0], [1000.0, 380.0], [200.0, 380.0]])
    warning = low_angle_warning(flat)
    assert warning is not None and "aspect" in warning
