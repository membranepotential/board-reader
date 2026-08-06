"""Operator-confirmed board labels — the ground truth the eval harness reads.

The snapshot page already asks a human to fix wrong squares (the correction
panel); until now it threw those answers away when the page reloaded. This
module is the format they are kept in, and the reader the evaluation harness
uses to find them.

**A label is eval data only.** Nothing here is ever fed back into the
calibration bank. The bank's base was built *from* ``data/eval/ground_truth.json``
— which is why the bank scores 64/64 on exactly the boards it was built from and
43/64 on the one it never saw. Fitting on confirmed labels would recreate that
contamination in the one dataset that exists to measure it.

Two on-disk shapes are understood:

``<stem>.label.json``
    Written by ``POST /labels/{upload}`` next to the retained photo
    (``BR_SAVE_UPLOADS``). Self-contained: photo, corners, truth, and what the
    server predicted at label time.

``ground_truth.json``
    The curated hand-labeled set (``data/eval/``): a mapping of board id ->
    ``{placement, source, corners_a8_h8_h1_a1}``. Kept readable so the harness
    runs on the boards we already have.

**Frame convention.** ``placement`` is always in the frame the ``corners``
define — corner 0 is a8, corner 1 is h8, and so on. It is therefore independent
of any model output: a read's ``autorotate`` quarter-turn is undone before the
label is written, so re-running a different estimator over the same photo
compares against the same truth.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from board_reader_core.protocol import encode_placement, parse_placement

LABEL_SUFFIX = ".label.json"

# The stem ``server._save_upload`` writes: ``<date>-<time>-<seq>-<kind>``. Label
# requests name an upload by this id, so it is matched strictly rather than
# joined onto a directory — an id is a request parameter, not a path.
UPLOAD_ID = re.compile(r"^\d{8}-\d{6}-\d{4}-(?:estimate|calibrate)$")

#: Provenance tags, kept on every board so a report can say where its truth came
#: from without the reader having to know about file layouts.
SOURCE_CONFIRMED = "confirmed"  # a human confirmed it in the snapshot page
SOURCE_CURATED = "curated"  # hand-authored ground_truth.json


def rotate_placement(placement: str, k: int) -> str:
    """Rotate a placement by ``k`` quarter-turns, matching ``np.rot90``.

    ``autocalib.resolve_rotation`` reports the turn it applied to the *grid*, so
    undoing it on a placement is ``rotate_placement(shown, -k)``.
    """
    grid = np.array(parse_placement(placement), dtype="<U5")
    return encode_placement(np.rot90(grid, k).tolist())


def rotate_square(square: str, k: int) -> str:
    """Rotate one algebraic square by ``k`` quarter-turns, matching :func:`rotate_placement`.

    Derived from ``np.rot90`` on an index grid rather than from arithmetic, so the
    two can never drift apart: the piece on ``square`` in ``placement`` is always
    the piece on ``rotate_square(square, k)`` in ``rotate_placement(placement, k)``
    (locked by a test). Used to carry square references — like the hard-constraint
    decoder's repaired squares — from the frame the operator *saw* into the corner
    frame a label is stored in.
    """
    file_, rank = square[0], square[1:]
    if file_ not in "abcdefgh" or rank not in [str(n) for n in range(1, 9)]:
        raise ValueError(f"not an algebraic square: {square!r}")
    r, c = 8 - int(rank), "abcdefgh".index(file_)
    moved = np.rot90(np.arange(64).reshape(8, 8), k)
    rows, cols = np.where(moved == r * 8 + c)
    return f"{'abcdefgh'[int(cols[0])]}{8 - int(rows[0])}"


def parse_corners(spec: str) -> np.ndarray:
    """``x0,y0,...,x3,y3`` -> a 4x2 float32 quad (a8, h8, h1, a1).

    Anything that is not such a string is a ValueError, including ``None`` — a
    label written before the corner requirement existed carries a null there, and
    a loader must skip it rather than die on it.
    """
    if not isinstance(spec, str):
        raise ValueError("corners must be a comma-separated string")
    nums = [float(x) for x in spec.split(",") if x.strip()]
    if len(nums) != 8:
        raise ValueError("corners needs 8 comma-separated numbers")
    return np.array(nums, dtype=np.float32).reshape(4, 2)


@dataclass(frozen=True)
class LabeledBoard:
    """One photo whose true position is known, ready to be read and scored."""

    board_id: str
    image: Path
    corners: np.ndarray  # (4, 2) float32, a8/h8/h1/a1 in ``image`` pixels
    placement: str  # truth, in the frame ``corners`` defines
    source: str  # SOURCE_CONFIRMED | SOURCE_CURATED
    note: str | None = None


def save_label(
    directory: Path,
    upload_id: str,
    *,
    placement: str,
    corners: str | None,
    predicted: str | None,
    corrections: int,
    autorotated: int,
    prior_flips: list[str] | None = None,
) -> Path:
    """Write ``<upload_id>.label.json``; returns the path.

    ``placement`` arrives as the operator *saw* it, i.e. after the read's
    ``autorotated`` quarter-turns; those are undone here so what lands on disk is
    in the corner frame. Re-labelling the same upload overwrites — the last
    confirmation is the truth.

    ``prior_flips`` are the squares the hard-constraint decoder repaired in this
    read (:mod:`.prior`), in the frame the operator saw, un-rotated to the corner
    frame like everything else here. They are recorded for one reason: since the
    decoder's repair is what gets *displayed*, an operator confirming with no
    corrections can hand back truth that came from the decoder rather than from
    the board. That is the same bias the bank incidents taught us to distrust, so
    a later eval must be able to find these labels — it cannot if we do not
    write them down. Nothing reads them back into the bank; see the module note.
    """
    k = (autorotated // 90) % 4
    truth = rotate_placement(placement, -k) if k else placement
    parse_placement(truth)  # reject a malformed label before it reaches disk
    flips = [rotate_square(s, -k) if k else s for s in (prior_flips or [])]
    path = directory / f"{upload_id}{LABEL_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "kind": "label",
                "ts": time.time(),
                "photo": f"{upload_id}.jpg",
                "corners": corners,
                "placement": truth,
                "placement_shown": placement,
                "predicted": predicted,
                "corrections": corrections,
                "autorotated": autorotated,
                "prior_flips": flips,
            }
        )
    )
    return path


def _board_from_label(path: Path) -> LabeledBoard | None:
    try:
        meta = json.loads(path.read_text())
        corners = parse_corners(meta["corners"])
        placement = meta["placement"]
        parse_placement(placement)
    except (OSError, ValueError, KeyError, TypeError):
        return None  # a half-written or hand-mangled label is skipped, not fatal
    image = path.parent / str(meta.get("photo") or "")
    if not image.exists():
        return None
    return LabeledBoard(
        board_id=path.name[: -len(LABEL_SUFFIX)],
        image=image,
        corners=corners,
        placement=placement,
        source=SOURCE_CONFIRMED,
        note=(
            f"{meta.get('corrections', 0)} correction(s)"
            if meta.get("corrections")
            else "confirmed as read"
        ),
    )


def _boards_from_ground_truth(path: Path) -> list[LabeledBoard]:
    """Read the curated ``ground_truth.json`` mapping.

    Its ``source`` paths are relative to the repo root (or ``~``-prefixed), which
    is where this file was authored from; missing images are skipped so a partial
    checkout still evaluates what it has.
    """
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(entries, dict):
        return []
    boards: list[LabeledBoard] = []
    for board_id, meta in entries.items():
        if not isinstance(meta, dict) or "corners_a8_h8_h1_a1" not in meta:
            continue
        image = Path(str(meta.get("source", ""))).expanduser()
        if not image.exists():
            continue
        try:
            corners = np.asarray(meta["corners_a8_h8_h1_a1"], dtype=np.float32).reshape(4, 2)
            parse_placement(meta["placement"])
        except (ValueError, KeyError, TypeError):
            continue
        boards.append(
            LabeledBoard(
                board_id=board_id,
                image=image,
                corners=corners,
                placement=meta["placement"],
                source=SOURCE_CURATED,
                note=meta.get("note"),
            )
        )
    return boards


def load_labels(paths: list[str] | list[Path]) -> list[LabeledBoard]:
    """Collect labeled boards from files and directories, de-duplicated by id.

    A directory is searched recursively for ``*.label.json`` and for any
    ``ground_truth.json``. A file is read as whichever of the two shapes its name
    says. Order is stable (sorted by board id) so two runs are comparable.
    """
    found: dict[str, LabeledBoard] = {}

    def take(board: LabeledBoard | None) -> None:
        if board is not None and board.board_id not in found:
            found[board.board_id] = board

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for label in sorted(p.rglob(f"*{LABEL_SUFFIX}")):
                take(_board_from_label(label))
            for gt in sorted(p.rglob("ground_truth.json")):
                for board in _boards_from_ground_truth(gt):
                    take(board)
        elif p.name.endswith(LABEL_SUFFIX):
            take(_board_from_label(p))
        elif p.is_file():
            for board in _boards_from_ground_truth(p):
                take(board)
    return [found[k] for k in sorted(found)]
