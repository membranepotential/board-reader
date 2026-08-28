"""64-square cropping from a rectified board or an original photo (DESIGN.md WP-B1).

Two cropping regimes feed the recognizer, both in the **L3 protocol order** (row 0 =
rank 8, column 0 = file a):

* :func:`crop_squares` — split a canonical top-down crop into 64 flat squares. Fine
  for occupancy, but a warped square *decapitates* pieces: the top-down projection
  smears a piece across neighbouring squares and the square cell keeps only its base.
* :func:`crop_squares_photo` — tall boxes taken from the ORIGINAL photo, so the crop
  contains the whole standing piece. This is what an embedding backbone should
  actually see; per-square accuracy on real photos is markedly better.

How tall those boxes are is the :class:`CropGeometry` decision, and there are two
answers:

* :class:`BboxGeometry` — the Chesscog-style original: extend the projected base
  quad's bounding box *upward in image space* by ``headroom`` × the square's
  projected **width**. Cheap, needs nothing but the corners. Its flaw is that a
  square's projected width shrinks with camera elevation while a real piece's
  projected height does not, so one scalar cannot cover both a steep view and a
  shallow one — the documented steep-angle decapitation.
* :class:`PoseGeometry` — recover the camera pose from the board homography and
  project the **bounding prism of a piece standing on the square**. The box height
  then follows the piece's real height in board units, as the camera actually sees
  it, instead of a fixed multiple of a foreshortened width.

A center ``inset`` (rectified path) keeps only the middle of each square,
suppressing grid lines and neighbouring-square spill.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from board_reader_core.protocol import GRID

# Canonical frame side used for square geometry; matches the snapshot crop size.
CANONICAL = 512.0


# --------------------------------------------------------------------------- #
# Crop geometry — the (train, calibrate, serve) contract
# --------------------------------------------------------------------------- #
# Whatever cut a model's training crops must cut its inference crops: a silent
# mismatch decapitates pieces and reads as "the model got worse" (it has, once).
# So geometry is a value that travels — in the `<model>.onnx.meta.json` sidecar,
# in the exemplar bank, and in the crop-cache key — never a loose pair of floats
# that a caller might forget to pass on.


@dataclass(frozen=True)
class BboxGeometry:
    """Image-axis box: the base quad's bbox, grown by multiples of square width.

    ``headroom`` extends it **upward** (piece tops point up in a normal photo),
    ``side_pad`` sideways, both in units of the square's projected width.
    """

    headroom: float = 1.6
    side_pad: float = 0.15

    def __post_init__(self) -> None:
        if self.headroom < 0 or self.side_pad < 0:
            raise ValueError("headroom and side_pad must be >= 0")

    def to_dict(self) -> dict:
        return {"kind": "bbox", "headroom": self.headroom, "side_pad": self.side_pad}


@dataclass(frozen=True)
class PoseGeometry:
    """Projected bounding prism of a piece standing on the square.

    ``piece_height`` and ``piece_radius`` are in **board units** (1 = one square
    width), so they describe the physical set, not the photo: a tournament
    Staunton king is ~95 mm on a ~57 mm square, i.e. ~1.67 squares tall, and a
    piece base is ~0.5 squares across.

    ``focal_ratio`` only matters when the focal length cannot be recovered from
    the homography itself (a near-top-down view constrains it barely or not at
    all); it is then the assumed focal as a multiple of the image's long side.
    See :func:`board_pose`.
    """

    piece_height: float = 1.9
    piece_radius: float = 0.5
    focal_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.piece_height <= 0 or self.piece_radius <= 0:
            raise ValueError("piece_height and piece_radius must be > 0")
        if self.focal_ratio <= 0:
            raise ValueError("focal_ratio must be > 0")

    def to_dict(self) -> dict:
        return {
            "kind": "pose",
            "piece_height": self.piece_height,
            "piece_radius": self.piece_radius,
            "focal_ratio": self.focal_ratio,
        }


CropGeometry = BboxGeometry | PoseGeometry

#: What the deployed models were trained with before pose geometry existed.
DEFAULT_GEOMETRY: CropGeometry = BboxGeometry(headroom=1.6, side_pad=0.15)

#: The geometry of the pre-sidecar models (v1). Only ever a fallback.
LEGACY_GEOMETRY: CropGeometry = BboxGeometry(headroom=0.15, side_pad=0.15)


def geometry_from_dict(data: dict) -> CropGeometry:
    """Rebuild a :data:`CropGeometry` from a sidecar/bank record.

    Records written before pose geometry existed carry bare ``headroom`` /
    ``side_pad`` keys and no ``kind``; they are bbox geometry by definition.
    """
    kind = data.get("kind", "bbox")
    if kind == "bbox":
        return BboxGeometry(
            headroom=float(data.get("headroom", DEFAULT_GEOMETRY.headroom)),
            side_pad=float(data.get("side_pad", DEFAULT_GEOMETRY.side_pad)),
        )
    if kind == "pose":
        return PoseGeometry(
            piece_height=float(data["piece_height"]),
            piece_radius=float(data["piece_radius"]),
            focal_ratio=float(data.get("focal_ratio", 1.0)),
        )
    raise ValueError(f"unknown crop geometry kind {kind!r}")


def geometry_tag(geometry: CropGeometry) -> str:
    """Short filesystem-safe identity for a geometry (crop-cache dir names)."""
    if isinstance(geometry, BboxGeometry):
        return f"hr{geometry.headroom:g}_sp{geometry.side_pad:g}"
    return (
        f"pose_h{geometry.piece_height:g}_r{geometry.piece_radius:g}"
        f"_f{geometry.focal_ratio:g}"
    )


def crop_squares(
    board: np.ndarray, *, inset: float = 0.0, size: int | None = None
) -> list[np.ndarray]:
    """Split a rectified ``board`` into 64 square crops (rank 8→1, file a→h).

    Parameters
    ----------
    board:
        Canonical top-down BGR crop; should be (near) square.
    inset:
        Fraction of each square trimmed off every edge (0 = whole square, 0.1 = keep
        the central 80%). Useful to drop grid lines / spill before embedding.
    size:
        If given, each crop is resized to ``size`` x ``size`` (square embedders want a
        fixed input). If ``None``, crops keep their native pixel size.

    Returns a list of 64 images in row-major order: index ``r*8 + c`` is rank ``8-r``,
    file ``chr(ord('a')+c)``.
    """
    if not 0.0 <= inset < 0.5:
        raise ValueError("inset must be in [0, 0.5)")
    h, w = board.shape[:2]
    sh, sw = h / GRID, w / GRID
    crops: list[np.ndarray] = []
    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = r * sh, (r + 1) * sh
            x0, x1 = c * sw, (c + 1) * sw
            dy, dx = (y1 - y0) * inset, (x1 - x0) * inset
            sq = board[
                round(y0 + dy) : round(y1 - dy),
                round(x0 + dx) : round(x1 - dx),
            ]
            if size is not None:
                sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
            crops.append(sq)
    return crops


def _canonical_to_image(corners: np.ndarray) -> np.ndarray:
    """Homography mapping the canonical ``CANONICAL``² frame onto image ``corners``.

    ``corners`` is the playing surface in image pixels, order **a8, h8, h1, a1**
    (= TL,TR,BR,BL of the canonical frame, :data:`board_reader_core.sim.CORNER_ORDER`).
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    src = np.array(
        [[0, 0], [CANONICAL, 0], [CANONICAL, CANONICAL], [0, CANONICAL]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, c)


def rectify_photo(image: np.ndarray, corners: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Warp an original photo to the canonical top-down crop (server-side fallback).

    Deliberately mirrors the edge's rectify; duplicated here because the inference
    node must never import the edge package (layer separation) but the snapshot
    HTTP ingress accepts original photos + corners.
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32
    )
    return cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(c, dst), (out_size, out_size)
    )


def full_frame_corners(image: np.ndarray) -> np.ndarray:
    """Corners of a whole (already-rectified) image, a8,h8,h1,a1 order."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def rot90_points(points: np.ndarray, k: int, image_shape: tuple[int, ...]) -> np.ndarray:
    """Where ``points`` land after ``np.rot90(image, k)`` — labels ride along.

    Continuous image coordinates ((0, 0) the top-left corner, ``(w, h)`` the
    bottom-right), matching :func:`full_frame_corners`. Turning a photo and its
    corner quad through this pair keeps every label attached to the same physical
    point, which is what lets the calibrate gate ask "would this photo read once
    turned?" without touching which corner is a8.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    h, w = float(image_shape[0]), float(image_shape[1])
    for _ in range(k % 4):
        x = pts[:, 0].copy()
        pts[:, 0] = pts[:, 1]
        pts[:, 1] = w - x
        h, w = w, h  # np.rot90 swaps the frame's sides
    return pts


def photo_square_boxes(
    corners: np.ndarray, *, headroom: float = 1.6, side_pad: float = 0.15
) -> np.ndarray:
    """The 64 axis-aligned sampling boxes :func:`crop_squares_photo` cuts (rank 8→1, file a→h).

    For each square, its 4 base corners are projected into the image via the
    corner homography; the box is the projected quad's bounding box extended
    **upward** by ``headroom``× the square's projected width (piece tops point up
    in a normal photo) and sideways by ``side_pad``×. Exposed on its own (from the
    geometry :func:`crop_squares_photo` itself uses) so diagnostics can draw the
    exact boxes recognition sampled, without re-deriving the homography.

    ``corners``: playing-surface corners in image pixels, order **a8, h8, h1, a1**.
    Returns ``(64, 4, 2)`` float32 quads (TL, TR, BR, BL), **not** clipped to any
    image size — callers that slice pixels (like :func:`crop_squares_photo`) clip
    themselves; callers that only draw (like diagnostics) don't need to.
    """
    if headroom < 0 or side_pad < 0:
        raise ValueError("headroom and side_pad must be >= 0")
    H = _canonical_to_image(corners)
    s = CANONICAL / GRID
    # Project all 81 lattice points once; index [r, c] = canonical (c*s, r*s).
    lat = np.array(
        [[[c * s, r * s]] for r in range(GRID + 1) for c in range(GRID + 1)],
        dtype=np.float32,
    )
    pts = cv2.perspectiveTransform(lat, H).reshape(GRID + 1, GRID + 1, 2)
    boxes = np.empty((GRID * GRID, 4, 2), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            quad = np.array(
                [pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]],
                dtype=np.float32,
            )
            w = 0.5 * (
                float(np.linalg.norm(quad[1] - quad[0]))
                + float(np.linalg.norm(quad[2] - quad[3]))
            )
            x0 = float(quad[:, 0].min()) - side_pad * w
            x1 = float(quad[:, 0].max()) + side_pad * w
            y0 = float(quad[:, 1].min()) - headroom * w
            y1 = float(quad[:, 1].max())
            boxes[r * GRID + c] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return boxes


def _board_to_image(corners: np.ndarray) -> np.ndarray:
    """Homography from **board units** (1 = one square, origin at a8) to image px."""
    s = CANONICAL / GRID
    return _canonical_to_image(corners) @ np.diag([s, s, 1.0])


def _focal_from_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Solve the camera focal length from a plane homography (Zhang's constraints).

    ``H``'s first two columns are ``lambda*K*r1`` and ``lambda*K*r2`` for an
    orthonormal ``r1, r2``, which gives two independent equations in ``f``:
    ``b1 . b2 = 0`` and ``|b1| = |b2|``. Both degenerate as the view approaches
    top-down (the third row of ``H`` goes to zero and the image stops carrying
    any perspective information about ``f``) — returns ``None`` there, and the
    caller falls back to :attr:`PoseGeometry.focal_ratio`.
    """
    u, v = H[:, 0], H[:, 1]
    a = np.array([u[0] - cx * u[2], u[1] - cy * u[2]])
    c = np.array([v[0] - cx * v[2], v[1] - cy * v[2]])
    candidates: list[tuple[float, float]] = []  # (conditioning, f^2)
    den_orth = u[2] * v[2]
    if den_orth != 0:
        candidates.append((abs(den_orth), -float(a @ c) / float(den_orth)))
    den_norm = u[2] ** 2 - v[2] ** 2
    if den_norm != 0:
        candidates.append(
            (abs(den_norm), float(c @ c - a @ a) / float(den_norm))
        )
    for _, f_sq in sorted(candidates, key=lambda t: -t[0]):
        if f_sq > 0:
            return math.sqrt(f_sq)
    return None


def board_pose(
    corners: np.ndarray, image_shape: tuple[int, ...], geometry: PoseGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera pose of the board plane: ``(K, r1, r2, r3_plus_t)`` — see below.

    Returns the intrinsics ``K`` and the three board-basis vectors plus the
    translation, packed so that a board point ``(X, Y, Z)`` — ``X`` along files,
    ``Y`` along ranks, ``Z`` **up off the table** — projects as
    ``K @ (X*r1 + Y*r2 + Z*r3 + t)``.

    ``r1``, ``r2`` and ``t`` come straight from the homography, so the ``Z = 0``
    plane reproduces :func:`photo_square_boxes`' base quads exactly; only ``Z``
    is new information, and it is exactly what a plane homography cannot supply
    on its own. ``r3`` is the unit board normal, signed so the camera lies on the
    positive-``Z`` side (pieces stand *toward* the camera, never into the table).
    """
    h_img, w_img = float(image_shape[0]), float(image_shape[1])
    cx, cy = w_img / 2.0, h_img / 2.0
    H = _board_to_image(corners).astype(np.float64)
    f = _focal_from_homography(H, cx, cy) or geometry.focal_ratio * max(h_img, w_img)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    B = np.linalg.inv(K) @ H
    norms = np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1])
    if norms == 0:
        raise ValueError("degenerate board corners: cannot recover camera pose")
    scale = 2.0 / norms
    r1, r2, t = B[:, 0] * scale, B[:, 1] * scale, B[:, 2] * scale
    if t[2] < 0:  # board behind the camera — the homography's sign is free
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    n = np.linalg.norm(r3)
    if n == 0:
        raise ValueError("degenerate board corners: cannot recover camera pose")
    r3 = r3 / n
    # The camera centre is the origin in camera coords; its board coords solve
    # M w = -t. Up is whichever normal puts the camera above the table.
    M = np.stack([r1, r2, r3], axis=1)
    try:
        w = np.linalg.solve(M, -t)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerically rare
        raise ValueError("degenerate board corners: cannot recover camera pose") from exc
    if w[2] < 0:
        r3 = -r3
    return K, r1, r2, np.stack([r3, t], axis=0)


def pose_square_boxes(
    corners: np.ndarray, image_shape: tuple[int, ...], geometry: PoseGeometry
) -> np.ndarray:
    """The 64 boxes bounding a piece-sized prism on each square (rank 8→1, file a→h).

    Each square gets an upright box of ``piece_radius`` half-width and
    ``piece_height`` height in board units, standing on the square centre; the
    sampling box is that prism's image-space bounding box. Because the height is
    a *world* quantity carried through the recovered pose, a steeply-viewed board
    gets tall boxes and a shallow one gets short boxes automatically — the thing
    a scalar ``headroom`` × projected-width cannot do.

    Returns ``(64, 4, 2)`` float32 quads (TL, TR, BR, BL), unclipped, matching
    :func:`photo_square_boxes`' contract.
    """
    K, r1, r2, r3t = board_pose(corners, image_shape, geometry)
    r3, t = r3t[0], r3t[1]
    rad, height = geometry.piece_radius, geometry.piece_height

    # The 8 prism corners of a unit cell, as offsets to be scaled per square.
    offsets = np.array(
        [
            [dx, dy, dz]
            for dx in (-rad, rad)
            for dy in (-rad, rad)
            for dz in (0.0, height)
        ]
    )
    boxes = np.empty((GRID * GRID, 4, 2), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            pts = offsets + np.array([c + 0.5, r + 0.5, 0.0])
            cam = pts @ np.stack([r1, r2, r3]) + t
            img = (K @ cam.T).T
            uv = img[:, :2] / img[:, 2:3]
            x0, y0 = float(uv[:, 0].min()), float(uv[:, 1].min())
            x1, y1 = float(uv[:, 0].max()), float(uv[:, 1].max())
            boxes[r * GRID + c] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return boxes


def square_boxes(
    corners: np.ndarray,
    geometry: CropGeometry,
    image_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """The 64 sampling boxes for ``geometry`` — the one entry point that dispatches.

    ``image_shape`` is required by :class:`PoseGeometry` (the intrinsics depend on
    the image's size) and ignored by :class:`BboxGeometry`.
    """
    if isinstance(geometry, BboxGeometry):
        return photo_square_boxes(
            corners, headroom=geometry.headroom, side_pad=geometry.side_pad
        )
    if image_shape is None:
        raise ValueError("PoseGeometry needs image_shape to recover the camera pose")
    return pose_square_boxes(corners, image_shape, geometry)


# --------------------------------------------------------------------------- #
# Occlusion-aware crop masking (E6, env-flagged experiment) — pure layer
# --------------------------------------------------------------------------- #
# At inference we do not know which neighbouring squares are OCCUPIED (truth is
# unavailable), so masking is keyed on geometry alone: any square whose piece
# prism would stand physically in front of and inside square S's crop box gets
# its pixels replaced there, regardless of whether that square actually holds a
# piece. An empty closer square's prism region is mostly bare board surface, so
# masking it loses little; a genuinely occupied one is exactly the neighbor-bleed
# source E4/E5 measured (c5 bishop -> b5 hallucinations etc).

#: Fill strategy for a masked pixel region. Selected by ``BR_CROP_MASK``.
OcclusionFill = Literal["gray", "blur"]

#: Neighbor-prism geometry used to derive occlusion, independent of whatever
#: geometry the calling model was trained on: occlusion is a fact about the
#: physical scene, not about a particular model's crop convention.
_MASK_GEOMETRY = PoseGeometry()


_WARNED_MASK_VALUES: set[str] = set()


def _occlusion_fill_from_env() -> OcclusionFill | None:
    """``BR_CROP_MASK={gray,blur}`` picks the fill; unset/empty = off.

    Off is the default and must stay byte-identical to pre-E6 behavior: no
    masking code runs at all when this returns ``None``.

    An unknown value WARNS once and runs unmasked rather than raising: this is
    read on the per-frame path, where a raise would be swallowed per frame by
    the ingest loop and silently kill the board's stream forever — a typo must
    degrade to "no masking", visibly, not to a dead board.
    """
    raw = os.environ.get("BR_CROP_MASK", "").strip().lower()
    if not raw:
        return None
    if raw not in ("gray", "blur"):
        if raw not in _WARNED_MASK_VALUES:
            _WARNED_MASK_VALUES.add(raw)
            logging.getLogger(__name__).warning(
                "BR_CROP_MASK=%r is not 'gray' or 'blur' — masking is OFF", raw
            )
        return None
    return raw  # type: ignore[return-value]


def _prism_rects(corners: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """``(64, 4)`` neighbor-prism boxes as ``(x0, y0, x1, y1)``, image pixels."""
    boxes = pose_square_boxes(corners, image_shape, _MASK_GEOMETRY)
    return np.stack([boxes[:, 0, 0], boxes[:, 0, 1], boxes[:, 2, 0], boxes[:, 2, 1]], axis=1)


#: Two squares whose camera depths differ by less than this are treated as
#: equidistant — neither occludes the other. Depths are in board units (square
#: edges); real perspective separates adjacent ranks by orders of magnitude
#: more, while the degenerate poses (rectified boards / full-frame corners,
#: where every base center recovers the same depth) collapse to a spread of
#: exactly zero. The epsilon exists for float noise in that second case.
_DEPTH_EPS = 1e-6


def _prism_depth(corners: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """``(64,)`` camera-space depth of each square's base center (``Z=0``).

    Nearness to the camera is a DEPTH comparison, not an image-row one: the
    projected-``y`` heuristic this replaces inverted under the rot90 orientation
    probe (session.py feeds rotated images through this path) and ordered by
    row index on rectified boards where nothing occludes anything. Depth is
    orientation-invariant and equal-by-construction in the degenerate poses,
    which the caller turns into "no occluders" via :data:`_DEPTH_EPS`.
    Uses the same recovered pose as :func:`pose_square_boxes`.
    """
    K, r1, r2, r3t = board_pose(corners, image_shape, _MASK_GEOMETRY)
    r3, t = r3t[0], r3t[1]
    centers = np.array(
        [[c + 0.5, r + 0.5, 0.0] for r in range(GRID) for c in range(GRID)]
    )
    cam = centers @ np.stack([r1, r2, r3]) + t
    return cam[:, 2]


def _closer_neighbor_prisms(
    square: int,
    prism_rects: np.ndarray,
    depth: np.ndarray,
    box: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Prism rects of squares CLOSER to the camera than ``square`` that overlap ``box``.

    Excludes ``square`` itself. "Closer" = smaller camera depth by more than
    :data:`_DEPTH_EPS` (see :func:`_prism_depth`) — equidistant squares never
    occlude each other, which makes the degenerate rectified/full-frame poses
    a no-op by construction. Overlap is a plain axis-aligned rectangle test
    against ``box`` (the target square's own crop box, ``(x0, y0, x1, y1)``).
    """
    x0, y0, x1, y1 = box
    out: list[tuple[float, float, float, float]] = []
    for n in range(prism_rects.shape[0]):
        if n == square or depth[n] >= depth[square] - _DEPTH_EPS:
            continue
        nx0, ny0, nx1, ny1 = prism_rects[n]
        if nx1 <= x0 or nx0 >= x1 or ny1 <= y0 or ny0 >= y1:
            continue
        out.append((float(nx0), float(ny0), float(nx1), float(ny1)))
    return out


def _occlusion_mask(
    box_px: tuple[int, int, int, int],
    own_prism: tuple[float, float, float, float],
    neighbor_prisms: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    """Boolean ``(H, W)`` mask over the pixel rect ``box_px`` — True = replace.

    True wherever a pixel falls inside some ``neighbor_prisms`` rect, EXCLUDING
    anywhere inside ``own_prism`` (square S's own prism region is never masked,
    even where a closer neighbor's prism overlaps it too).
    """
    xi0, yi0, xi1, yi1 = box_px
    h, w = yi1 - yi0, xi1 - xi0
    if h <= 0 or w <= 0 or not neighbor_prisms:
        return np.zeros((max(h, 0), max(w, 0)), dtype=bool)
    ys = np.arange(yi0, yi1)
    xs = np.arange(xi0, xi1)
    occluded = np.zeros((h, w), dtype=bool)
    for nx0, ny0, nx1, ny1 in neighbor_prisms:
        row = (ys >= ny0) & (ys < ny1)
        col = (xs >= nx0) & (xs < nx1)
        occluded |= row[:, None] & col[None, :]
    ox0, oy0, ox1, oy1 = own_prism
    own_row = (ys >= oy0) & (ys < oy1)
    own_col = (xs >= ox0) & (xs < ox1)
    occluded &= ~(own_row[:, None] & own_col[None, :])
    return occluded


def _apply_occlusion_fill(
    patch: np.ndarray, occluded: np.ndarray, fill: OcclusionFill
) -> np.ndarray:
    """Replace ``patch`` pixels where ``occluded`` is True, per ``fill``."""
    if fill == "gray":
        # Mean over the pixels that survive: averaging the occluder into its
        # own replacement would leak some of what the mask is there to remove.
        # (All-occluded fallback: whole-patch mean is the only color there is.)
        kept = patch[~occluded]
        src = kept if kept.size else patch.reshape(-1, *patch.shape[2:])
        mean_color = np.round(src.reshape(-1, *patch.shape[2:]).mean(axis=0))
        out = patch.copy()
        out[occluded] = mean_color.astype(patch.dtype)
        return out
    if fill == "blur":
        # Heavy blur: sigma proportional to the crop's own (pre-resize) extent,
        # so it scales with the source photo's resolution rather than a fixed
        # pixel count. cv2.GaussianBlur's cost scales with kernel size (~6x
        # sigma per side), so at a full-resolution photo's crop sizes a literal
        # call blows up (measured: a 300x180 patch at sigma~37 costs ~100+ms,
        # 100x+ this function's other branch, enough to turn a 30-frame replay
        # into a multi-minute one). Downsample, blur small, upsample: a
        # standard equivalent-strength approximation at a fraction of the cost.
        sigma = max(patch.shape[0], patch.shape[1]) / 8.0
        scale = max(1, round(sigma / 4.0))
        if scale > 1:
            h, w = patch.shape[:2]
            sw, sh = max(1, w // scale), max(1, h // scale)
            small = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / scale)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
        out = patch.copy()
        out[occluded] = blurred[occluded]
        return out
    raise ValueError(f"unknown occlusion fill {fill!r}")  # pragma: no cover - guarded upstream


def crop_squares_photo(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    geometry: CropGeometry = DEFAULT_GEOMETRY,
    size: int = 224,
    squares: Sequence[int] | None = None,
    mask: OcclusionFill | None = None,
) -> list[np.ndarray]:
    """Cut *tall* per-square boxes out of the ORIGINAL photo (rank 8→1, file a→h).

    Each box (:func:`square_boxes`, per ``geometry``) is clipped to the image,
    padded to a square (edge replication), and resized to ``size``².

    Compared to slicing the top-down warp (:func:`crop_squares`), the classifier
    sees the whole standing piece instead of a decapitated, smeared base — the
    trick that recovers real-photo accuracy.

    ``corners``: playing-surface corners in image pixels, order **a8, h8, h1, a1**.
    ``squares``: which of the 64 board indices to cut, in the order given (default
    all 64). The cascade's second model only ever looks at the squares its first
    model was unsure about, so it pays for those crops and no others.

    ``mask``: occlusion-aware crop masking (E6). ``None`` (default) reads
    ``BR_CROP_MASK`` — this is the single choke point every crop-classifier and
    the exemplar-bank embedder cut crops through, so setting the env var affects
    all of them uniformly with no other call site touching this flag. Left
    unset, no masking code runs and output is byte-identical to before E6.
    """
    if mask is None:
        mask = _occlusion_fill_from_env()
    h_img, w_img = image.shape[:2]
    boxes = square_boxes(corners, geometry, image.shape)
    if squares is not None:
        sel = np.asarray(squares, dtype=int)
        boxes = boxes[sel]
        square_ids = sel.tolist()
    else:
        square_ids = list(range(GRID * GRID))
    prism_rects: np.ndarray | None = None
    depth: np.ndarray | None = None
    if mask is not None:
        prism_rects = _prism_rects(corners, image.shape)
        depth = _prism_depth(corners, image.shape)
    crops: list[np.ndarray] = []
    for box, sq in zip(boxes, square_ids, strict=True):
        x0, y0 = box[0]
        x1, y1 = box[2]
        xi0, yi0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        xi1, yi1 = min(w_img, math.ceil(x1)), min(h_img, math.ceil(y1))
        patch = image[yi0:yi1, xi0:xi1]
        if patch.size == 0:  # corner off-frame; keep the slot with a blank
            patch = np.zeros(
                (8, 8, 3) if image.ndim == 3 else (8, 8), dtype=image.dtype
            )
        elif mask is not None:
            assert prism_rects is not None and depth is not None
            neighbors = _closer_neighbor_prisms(sq, prism_rects, depth, (x0, y0, x1, y1))
            if neighbors:
                occluded = _occlusion_mask(
                    (xi0, yi0, xi1, yi1), tuple(prism_rects[sq]), neighbors
                )
                if occluded.any():
                    patch = _apply_occlusion_fill(patch, occluded, mask)
        ph, pw = patch.shape[:2]
        if ph != pw:  # pad to square so the resize doesn't change aspect
            d = abs(ph - pw)
            top, bot = (0, 0) if ph > pw else (d // 2, d - d // 2)
            left, right = (d // 2, d - d // 2) if ph > pw else (0, 0)
            patch = cv2.copyMakeBorder(
                patch, top, bot, left, right, cv2.BORDER_REPLICATE
            )
        crops.append(cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA))
    return crops
