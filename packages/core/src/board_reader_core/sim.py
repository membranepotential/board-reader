"""Synthetic board rendering for the sim harness (DESIGN.md §6).

A dependency-free way to produce board frames without a camera or a downloaded
clip: render a top-down board image from a placement string, warp it to an angled
view, and stream a whole synthetic game (settled positions, hand-occluded
transitions) as frames or an mp4. Used by the edge synthetic feed and by tests.
This is sim/test scaffolding, intentionally simple — not the real capture path.
Lives in core so both the edge and tests can use it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict, Unpack

import cv2
import numpy as np

from board_reader_core.protocol import START_PLACEMENT, parse_placement

_LIGHT_SQ = (181, 217, 240)  # BGR, a "light" board square
_DARK_SQ = (85, 138, 181)


def render_board(
    placement: str = START_PLACEMENT,
    *,
    square_px: int = 48,
) -> np.ndarray:
    """Render a top-down BGR board image from a placement (rank 8 top, file a left)."""
    grid = parse_placement(placement)
    n = 8 * square_px
    img = np.zeros((n, n, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(8):
        for c in range(8):
            y0, x0 = r * square_px, c * square_px
            color = _LIGHT_SQ if (r + c) % 2 == 0 else _DARK_SQ
            img[y0 : y0 + square_px, x0 : x0 + square_px] = color
            cell = grid[r][c]
            if cell == "empty":
                continue
            glyph = cell.upper()
            glyph_color = (245, 245, 245) if cell.isupper() else (20, 20, 20)
            scale = square_px / 48.0
            (tw, th), _ = cv2.getTextSize(glyph, font, scale, 2)
            cv2.putText(
                img,
                glyph,
                (x0 + (square_px - tw) // 2, y0 + (square_px + th) // 2),
                font,
                scale,
                glyph_color,
                2,
                cv2.LINE_AA,
            )
    return img


def render_frames(placements: list[str], *, square_px: int = 48) -> list[np.ndarray]:
    """Render a sequence of placements into board images."""
    return [render_board(p, square_px=square_px) for p in placements]


# --- Perspective simulation (the "angled broadcast view", DESIGN D8) -----------
#
# A real camera sees the board at an angle. To develop and test against that
# offline, we take a canonical board image and warp it onto a larger canvas as a
# foreshortened quadrilateral. Corner order is **TL, TR, BR, BL** everywhere — for a canonical
# board (rank 8 top, file a left) that is a8, h8, h1, a1.

# Corner order of the warp helpers.
CORNER_ORDER: tuple[str, ...] = ("a8", "h8", "h1", "a1")


def board_corners(size: int) -> np.ndarray:
    """The 4 corners (TL,TR,BR,BL) of a canonical ``size``x``size`` board image."""
    s = float(size)
    return np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)


def warp_to_quad(
    board: np.ndarray,
    dst_quad: np.ndarray,
    *,
    canvas: tuple[int, int] | None = None,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Warp a canonical square ``board`` onto ``dst_quad`` (TL,TR,BR,BL) of a canvas.

    Returns a BGR image of size ``canvas`` (h, w); defaults to a square canvas just
    big enough to hold the quad. The area outside the quad is filled with
    ``background`` (black by default) so the board reads as a clear foreground
    quadrilateral — exactly what the detector keys on.
    """
    h, w = board.shape[:2]
    if h != w:
        raise ValueError("board image must be square")
    src = board_corners(w)
    dst = np.asarray(dst_quad, dtype=np.float32)
    if dst.shape != (4, 2):
        raise ValueError("dst_quad must be 4 (x, y) corners")
    if canvas is None:
        side = int(np.ceil(dst.max())) + 1
        canvas = (side, side)
    out_h, out_w = canvas
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        board,
        matrix,
        (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def simulate_angled_view(
    board: np.ndarray,
    *,
    tilt: float = 0.25,
    yaw: float = 0.06,
    margin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``board`` as a plausible angled view; return ``(image, dst_quad)``.

    Deterministic (no RNG). ``tilt`` foreshortens the far (top) edge, ``yaw`` skews
    left/right, and ``margin`` insets the board within the canvas so it sits fully
    inside the frame. ``dst_quad`` is the ground-truth board outline in image
    coordinates (TL,TR,BR,BL) — handy for tests that exercise manual corners.
    """
    side = board.shape[0]
    canvas = round(side * (1.0 + 2 * margin))
    m = side * margin
    top = side * tilt / 2.0  # how much the far edge shrinks inward on each side
    sk = side * yaw  # horizontal skew
    dst = np.array(
        [
            [m + top + sk, m],                 # TL (a8, far)
            [m + side - top + sk, m],          # TR (h8, far)
            [m + side, m + side],              # BR (h1, near)
            [m, m + side],                     # BL (a1, near)
        ],
        dtype=np.float32,
    )
    img = warp_to_quad(board, dst, canvas=(canvas, canvas))
    return img, dst


# --- Synthetic recorded game (DESIGN §6.1, milestone M2) ----------------------
#
# M2 streams positions over a *full recorded game* seen at an angle, staying
# sensible while hands reach in and pieces move. Real footage is curated
# separately (§6.1 step 1); this section renders a deterministic stand-in so the
# whole pipe — motion meter, adaptive forward, estimator, viewer — can be driven
# end to end without a camera, and so tests can assert on frames we control.
#
# Placements only. Consecutive entries of SAMPLE_GAME_PLACEMENTS happen to differ
# by one move's effect, but nothing here reasons about moves or legality (D1): the
# renderer only ever asks *which squares changed*, to know where to put the hand.

# Légal's Mate (Légall de Kermeur, Paris c. 1750): 1.e4 e5 2.Bc4 d6 3.Nf3 Bg4
# 4.Nc3 g6 5.Nxe5 Bxd1 6.Bxf7+ Ke7 7.Nd5#. Short, has captures and pawn moves,
# and every ply changes exactly two squares (no castling, no en passant).
SAMPLE_GAME_PLACEMENTS: tuple[str, ...] = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
    "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR",
    "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR",
    "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
    "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P1b1/5N2/PPPP1PPP/RNBQK2R",
    "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R",
    "rn1qkbnr/ppp2p1p/3p2p1/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R",
    "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P1b1/2N5/PPPP1PPP/R1BQK2R",
    "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R",
    "rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R",
    "rn1q1bnr/ppp1kB1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R",
    "rn1q1bnr/ppp1kB1p/3p2p1/3NN3/4P3/8/PPPP1PPP/R1BbK2R",
)

_SKIN = (120, 160, 200)  # BGR, the synthetic hand — darker than either square
_SKIN_EDGE = (85, 115, 150)

# Per-frame sensor noise, so "settled" frames are near-identical but not bitwise
# equal — a motion meter tuned against a perfectly static feed would be lying to
# itself. Small enough that transition motion dwarfs it.
_NOISE_SIGMA = 0.7

# Where the hand's sweep begins, as a fraction of the way from the off-frame wrist
# to the changed squares (1.0 = palm on them). Starting part-way in means the hand
# is already over the board on the first transition frame, so every transition step
# carries motion rather than the first one being a no-op.
_REACH_START = 0.25


def _changed_centre(before: str, after: str, square_px: int) -> tuple[float, float]:
    """Pixel centre (x, y) of the squares whose contents differ between two placements.

    A cell-wise diff of two grids — deliberately *not* move inference (D1). All we
    want is a place to reach for.
    """
    a, b = parse_placement(before), parse_placement(after)
    changed = [(r, c) for r in range(8) for c in range(8) if a[r][c] != b[r][c]]
    if not changed:  # identical placements: sweep across the middle anyway
        changed = [(3, 3), (4, 4)]
    x = sum((c + 0.5) * square_px for _, c in changed) / len(changed)
    y = sum((r + 0.5) * square_px for r, _ in changed) / len(changed)
    return x, y


def _px(point: tuple[float, float]) -> tuple[int, int]:
    """Round a float (x, y) to the integer pixel coordinates OpenCV draws with."""
    return round(point[0]), round(point[1])


def _draw_hand(
    img: np.ndarray,
    palm: tuple[float, float],
    wrist: tuple[float, float],
    *,
    square_px: int,
) -> None:
    """Draw a hand reaching in from ``wrist`` to ``palm``, in place.

    A thick forearm bar plus a palm ellipse — crude on purpose. Its job is to
    occlude squares and to move a lot between frames, not to look like a hand.
    """
    w, p = _px(wrist), _px(palm)
    axes = (int(1.2 * square_px), int(0.9 * square_px))
    cv2.line(img, w, p, _SKIN, int(1.3 * square_px), cv2.LINE_AA)
    cv2.ellipse(img, p, axes, 0.0, 0, 360, _SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(
        img, p, axes, 0.0, 0, 360, _SKIN_EDGE, max(1, square_px // 16), cv2.LINE_AA
    )


def _capture(
    board: np.ndarray, *, angled: bool, rng: np.random.Generator
) -> np.ndarray:
    """One camera frame of ``board``: optional perspective, then sensor noise.

    Always returns a fresh array, so the caller can keep reusing its board image.
    """
    frame = simulate_angled_view(board)[0] if angled else board
    noisy = frame.astype(np.float32) + rng.normal(0.0, _NOISE_SIGMA, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def render_game_frames(
    placements: Sequence[str],
    *,
    frames_per_position: int = 6,
    transition_frames: int = 4,
    hand: bool = True,
    angled: bool = False,
    seed: int = 0,
    square_px: int = 48,
) -> Iterator[np.ndarray]:
    """Lazily yield BGR frames of a synthetic game: settled positions and transitions.

    Each placement gets ``frames_per_position`` settled frames; between consecutive
    placements come ``transition_frames`` in which the board flips to the new
    placement at the midpoint and (when ``hand``) a synthetic hand sweeps in from
    the near edge over the changed squares — partial occlusion plus the strong
    inter-frame motion the edge's motion meter keys on. Total frame count is
    ``len(placements) * frames_per_position + (len(placements) - 1) *
    transition_frames``.

    ``angled`` runs every frame through :func:`simulate_angled_view` — one board,
    one camera, so the warp is identical for the whole stream. Deterministic given
    ``seed``.

    This is a **generator**: frames are rendered as they are consumed and never
    accumulated, so a long game costs O(1) memory. Do not wrap it in a list.
    """
    if not placements:
        raise ValueError("need at least one placement")
    if frames_per_position < 1:
        raise ValueError("frames_per_position must be >= 1")
    if transition_frames < 0:
        raise ValueError("transition_frames must be >= 0")

    rng = np.random.default_rng(seed)
    side = 8 * square_px
    board = render_board(placements[0], square_px=square_px)
    for i, placement in enumerate(placements):
        for _ in range(frames_per_position):
            yield _capture(board, angled=angled, rng=rng)
        if i + 1 == len(placements):
            return
        nxt = render_board(placements[i + 1], square_px=square_px)
        cx, cy = _changed_centre(placement, placements[i + 1], square_px)
        # The arm comes in over the near edge, off to one side, and straightens
        # as it reaches — so both the palm and the forearm move every frame.
        wrist = (cx - 3.5 * square_px, side + 0.8 * square_px)
        for t in range(transition_frames):
            base = board if 2 * t < transition_frames else nxt
            reach = _REACH_START + (1.0 - _REACH_START) * (t + 1) / transition_frames
            frame = base
            if hand:
                frame = base.copy()
                palm = (
                    wrist[0] + reach * (cx - wrist[0]),
                    wrist[1] + reach * (cy - wrist[1]),
                )
                _draw_hand(frame, palm, wrist, square_px=square_px)
            yield _capture(frame, angled=angled, rng=rng)
        board = nxt


class _GameFrameOptions(TypedDict, total=False):
    """The :func:`render_game_frames` knobs :func:`write_game_clip` forwards."""

    frames_per_position: int
    transition_frames: int
    hand: bool
    angled: bool
    seed: int
    square_px: int


def write_game_clip(
    path: str | Path,
    placements: Sequence[str] = SAMPLE_GAME_PLACEMENTS,
    *,
    fps: float = 12.0,
    **kwargs: Unpack[_GameFrameOptions],
) -> int:
    """Write a synthetic game to an mp4 at ``path``; return the number of frames.

    The manual-demo artifact for M2 (docs/TESTING.md): ``board-reader-edge feed
    --video <clip>`` replays it through the real pipe. Frames go to the encoder
    straight off the generator, so a long game never sits in memory.
    """
    frames = render_game_frames(placements, **kwargs)
    first = next(frames)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {path}")
    try:
        writer.write(first)
        count = 1
        for frame in frames:
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count
