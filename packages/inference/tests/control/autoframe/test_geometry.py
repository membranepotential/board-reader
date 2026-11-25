"""Auto-frame's board displacement (control/autoframe/geometry.py)."""

from __future__ import annotations

import pytest

from board_reader_inference.control.autoframe.geometry import Located, displacement

ASPECT = 16 / 9
#: A board 0.3 of the frame wide.
HOME = Located(((0.30, 0.30), (0.60, 0.30), (0.60, 0.80), (0.30, 0.80)), 0.97)


def moved(board: Located, dx: float, dy: float = 0.0) -> Located:
    (a, b, c, d) = ((x + dx, y + dy) for x, y in board.corners)
    return Located((a, b, c, d), board.r2)


def test_displacement_ignores_corner_labels_and_counts_pixels():
    a8, h8, h1, a1 = HOME.corners
    turned = Located((h8, h1, a1, a8), 0.95)  # the same board read a quarter turn off
    assert displacement(HOME, turned, ASPECT) == 0
    # 0.03 of the frame's width is 10 % of a board 0.3 wide, whatever the aspect.
    assert displacement(HOME, moved(HOME, 0.03), ASPECT) == pytest.approx(0.1)
    # 0.03 of its height is 0.03 / (0.3 * 16/9) of the board's width.
    assert displacement(HOME, moved(HOME, 0, 0.03), ASPECT) == pytest.approx(0.05625)
