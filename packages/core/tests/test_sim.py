"""Tests for the synthetic board renderer (sim harness)."""

from __future__ import annotations

import inspect
import itertools

import cv2
import numpy as np
import pytest

from board_reader_core.protocol import EMPTY_PLACEMENT, START_PLACEMENT, parse_placement
from board_reader_core.sim import (
    SAMPLE_GAME_PLACEMENTS,
    board_corners,
    render_board,
    render_frames,
    render_game_frames,
    simulate_angled_view,
    warp_to_quad,
    write_game_clip,
)


def test_render_board_shape_and_dtype():
    img = render_board(START_PLACEMENT, square_px=32)
    assert img.shape == (8 * 32, 8 * 32, 3)
    assert img.dtype == np.uint8


def test_render_board_draws_pieces_only_where_present():
    """The start position has more non-background ink than the empty board."""
    start = render_board(START_PLACEMENT)
    empty = render_board(EMPTY_PLACEMENT)
    # Pieces add light/dark glyph pixels; the populated board differs from empty.
    assert not np.array_equal(start, empty)
    assert int(np.abs(start.astype(int) - empty.astype(int)).sum()) > 0


def test_render_frames_returns_one_image_per_placement():
    frames = render_frames([START_PLACEMENT, EMPTY_PLACEMENT])
    assert len(frames) == 2
    assert all(f.shape == frames[0].shape for f in frames)


def test_warp_to_quad_places_the_board_at_the_given_corners():
    board = render_board(square_px=32)
    side = board.shape[0]
    # Identity quad on a same-size canvas reproduces the board (corner-to-corner).
    img = warp_to_quad(board, board_corners(side), canvas=(side, side))
    assert img.shape == board.shape
    assert int(np.abs(img.astype(int) - board.astype(int)).mean()) < 5


def test_simulate_angled_view_returns_image_and_ground_truth_quad():
    board = render_board(square_px=32)
    img, quad = simulate_angled_view(board)
    assert img.ndim == 3 and img.shape[0] == img.shape[1]
    assert quad.shape == (4, 2)
    # Foreshortening: the far (top) edge is shorter than the near (bottom) edge.
    top_w = np.linalg.norm(quad[1] - quad[0])
    bottom_w = np.linalg.norm(quad[2] - quad[3])
    assert top_w < bottom_w


# --- The synthetic recorded game (DESIGN §6.1, M2) -----------------------------

SQ = 16  # small board: these tests are about frame *streams*, not glyph quality


def _changed_squares(before: str, after: str) -> list[tuple[int, int]]:
    """The (row, col) squares whose contents differ — the test's own cell diff."""
    a, b = parse_placement(before), parse_placement(after)
    return [(r, c) for r in range(8) for c in range(8) if a[r][c] != b[r][c]]


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _changed_region(before: str, after: str) -> tuple[slice, slice]:
    """Bounding box of the changed squares, as (rows, cols) pixel slices."""
    changed = _changed_squares(before, after)
    rows = [r for r, _ in changed]
    cols = [c for _, c in changed]
    return (
        slice(min(rows) * SQ, (max(rows) + 1) * SQ),
        slice(min(cols) * SQ, (max(cols) + 1) * SQ),
    )


def test_sample_game_starts_from_the_start_position():
    assert SAMPLE_GAME_PLACEMENTS[0] == START_PLACEMENT
    assert len(SAMPLE_GAME_PLACEMENTS) >= 12


def test_sample_game_advances_by_one_move_effect_per_entry():
    """Every consecutive pair differs by exactly one piece leaving and arriving."""
    for before, after in itertools.pairwise(SAMPLE_GAME_PLACEMENTS):
        assert len(_changed_squares(before, after)) == 2, (before, after)


def test_sample_game_contains_a_capture_and_a_pawn_move():
    counts = [
        sum(cell != "empty" for row in parse_placement(p) for cell in row)
        for p in SAMPLE_GAME_PLACEMENTS
    ]
    assert min(counts) < counts[0], "no capture: the piece count never drops"

    pawn_moved = False
    for before, after in itertools.pairwise(SAMPLE_GAME_PLACEMENTS):
        grid = parse_placement(before)
        vacated = [
            grid[r][c]
            for r, c in _changed_squares(before, after)
            if parse_placement(after)[r][c] == "empty"
        ]
        pawn_moved = pawn_moved or any(cell in ("P", "p") for cell in vacated)
    assert pawn_moved, "no pawn move in the sample game"


def test_render_game_frames_is_a_lazy_generator():
    """O(1) memory: frames are produced on demand, never accumulated."""
    stream = render_game_frames(SAMPLE_GAME_PLACEMENTS, square_px=SQ)
    assert inspect.isgenerator(stream)
    first = next(stream)
    assert first.shape == (8 * SQ, 8 * SQ, 3)
    assert first.dtype == np.uint8
    stream.close()


def test_render_game_frames_yields_settled_and_transition_frames_for_every_ply():
    placements = SAMPLE_GAME_PLACEMENTS[:4]
    frames = list(
        render_game_frames(
            placements, frames_per_position=3, transition_frames=2, square_px=SQ
        )
    )
    assert len(frames) == len(placements) * 3 + (len(placements) - 1) * 2


def test_render_game_frames_is_deterministic_for_a_seed():
    def run(seed: int) -> list[np.ndarray]:
        return list(
            render_game_frames(
                SAMPLE_GAME_PLACEMENTS[:3],
                frames_per_position=2,
                transition_frames=2,
                seed=seed,
                square_px=SQ,
            )
        )

    a, b, other = run(0), run(0), run(7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, other))


def test_transition_frames_carry_far_more_motion_than_settled_frames():
    """What the edge's motion meter keys on: a settled board barely moves."""
    fpp, tf = 3, 4
    frames = list(
        render_game_frames(
            SAMPLE_GAME_PLACEMENTS[:2],
            frames_per_position=fpp,
            transition_frames=tf,
            square_px=SQ,
        )
    )
    steps = [_mean_abs_diff(a, b) for a, b in itertools.pairwise(frames)]
    settled = steps[: fpp - 1] + steps[fpp + tf :]
    transition = steps[fpp - 1 : fpp + tf]
    assert max(settled) > 0.0, "settled frames should still carry sensor noise"
    assert min(transition) > 3 * max(settled)


def test_the_hand_occludes_the_changing_squares_during_a_transition():
    before, after = SAMPLE_GAME_PLACEMENTS[0], SAMPLE_GAME_PLACEMENTS[1]
    fpp, tf = 2, 4
    rows, cols = _changed_region(before, after)

    def stream(hand: bool) -> list[np.ndarray]:
        return list(
            render_game_frames(
                (before, after),
                frames_per_position=fpp,
                transition_frames=tf,
                hand=hand,
                square_px=SQ,
            )
        )

    occluded = stream(hand=True)[fpp + tf - 1]  # last transition frame: fully reached
    clean = stream(hand=False)[fpp + tf - 1]
    settled_after = stream(hand=False)[fpp + tf]  # first settled frame of `after`

    region = lambda f: f[rows, cols]  # noqa: E731
    assert _mean_abs_diff(region(occluded), region(settled_after)) > 10
    # Without the hand the same frame is just the new placement plus noise.
    assert _mean_abs_diff(region(clean), region(settled_after)) < 3


def test_render_game_frames_angled_uses_one_camera_for_the_whole_stream():
    board = render_board(SAMPLE_GAME_PLACEMENTS[0], square_px=SQ)
    canvas = simulate_angled_view(board)[0].shape
    frames = list(
        render_game_frames(
            SAMPLE_GAME_PLACEMENTS[:3],
            frames_per_position=2,
            transition_frames=2,
            angled=True,
            square_px=SQ,
        )
    )
    assert all(f.shape == canvas for f in frames)
    # The far top-left corner is outside the board quad in every frame, so the
    # warp — one board, one camera — never moved.
    assert all(float(f[:4, :4].mean()) < 5 for f in frames)


@pytest.mark.parametrize(
    "kwargs", [{"frames_per_position": 0}, {"transition_frames": -1}]
)
def test_render_game_frames_rejects_impossible_stream_shapes(kwargs):
    with pytest.raises(ValueError):
        next(render_game_frames(SAMPLE_GAME_PLACEMENTS, **kwargs))


def test_render_game_frames_rejects_an_empty_game():
    with pytest.raises(ValueError):
        next(render_game_frames([]))


def test_write_game_clip_writes_a_readable_clip(tmp_path):
    path = tmp_path / "game.mp4"
    written = write_game_clip(
        path,
        SAMPLE_GAME_PLACEMENTS[:3],
        fps=8.0,
        frames_per_position=2,
        transition_frames=2,
        square_px=SQ,
    )
    assert written == 3 * 2 + 2 * 2
    assert path.exists() and path.stat().st_size > 0

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    try:
        read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (8 * SQ, 8 * SQ, 3)
            read += 1
    finally:
        cap.release()
    assert read == written
