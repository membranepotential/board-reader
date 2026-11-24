"""``tools/migrate_labels_to_chess_frame.py``: rotation only, checked, backed up."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_core.protocol import dist_from_placement, parse_placement

from board_reader_inference.measurement.labels import load_labels, rotate_placement

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "migrate_labels_to_chess_frame.py"


def _load():
    spec = importlib.util.spec_from_file_location("migrate_labels", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mig = _load()

# Chess frame: White at the bottom; asymmetric, no pawn on rank 1/8 in any frame.
CHESS = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R"
CORNERS = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64)


def _read_as(placement: str):
    """A recognizer that reads ``placement`` from any photo."""
    return lambda image: dist_from_placement(placement, 0.9)


def _label(directory: Path, stem: str, *, k: int, autorotated: int) -> Path:
    """A corner-frame label whose corners sit ``k`` turns off the chess frame."""
    cv2.imwrite(str(directory / f"{stem}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    path = directory / f"{stem}.label.json"
    path.write_text(json.dumps({
        "kind": "label",
        "photo": f"{stem}.jpg",
        "corners": ",".join(f"{v:g}" for v in mig.rotate_corners(CORNERS, -k).reshape(-1)),
        "placement": rotate_placement(CHESS, -k),
        "autorotated": autorotated,
        "prior_flips": ["e4"],
    }))
    return path


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_corners_follow_the_placement_turn(k):
    """The image corner named a8 must hold the same piece before and after."""
    corner_cells = [(0, 0), (0, 7), (7, 7), (7, 0)]  # a8, h8, h1, a1 in a grid
    old = np.array(parse_placement(rotate_placement(CHESS, -k)))
    new = np.array(parse_placement(CHESS))
    old_corners = mig.rotate_corners(CORNERS, -k)
    for i, point in enumerate(CORNERS):
        j = next(j for j, p in enumerate(old_corners) if np.array_equal(p, point))
        assert new[corner_cells[i]] == old[corner_cells[j]]


def test_a_turned_label_is_migrated_and_backed_up(tmp_path):
    path = _label(tmp_path, "20260806-231719-0000-estimate", k=1, autorotated=90)
    original = path.read_text()
    assert not mig.migrate_dir(tmp_path, _read_as(CHESS), apply=False)[0].problems
    assert path.read_text() == original  # a dry run writes nothing

    [verdict] = mig.migrate_dir(tmp_path, _read_as(CHESS), apply=True)
    assert (verdict.rotation, verdict.problems) == (1, ())
    meta = json.loads(path.read_text())
    assert (meta["frame"], meta["placement"]) == ("chess", CHESS)
    assert meta["corners"] == "10,10,90,10,90,90,10,90"
    assert meta["migrated"]["rotation_deg"] == 90
    assert (tmp_path / ".pre-chess-frame" / path.name).read_text() == original
    [board] = load_labels([tmp_path])
    assert (board.frame, board.placement) == ("chess", CHESS)
    # A second run leaves a chess-frame label alone.
    assert mig.migrate_dir(tmp_path, _read_as(CHESS), apply=True) == []


def test_a_disagreement_is_reported_and_never_resolved(tmp_path):
    path = _label(tmp_path, "20260806-231719-0001-estimate", k=1, autorotated=180)
    original = path.read_text()
    [verdict] = mig.migrate_dir(tmp_path, _read_as(CHESS), apply=True)
    assert verdict.problems == ("old autorotate showed it turned 180°",)
    assert path.read_text() == original
    assert not (tmp_path / ".pre-chess-frame").exists()


def test_a_read_that_cannot_tell_the_turns_apart_is_left(tmp_path):
    _label(tmp_path, "20260806-231719-0002-estimate", k=2, autorotated=0)
    [verdict] = mig.migrate_dir(tmp_path, _read_as("8/8/8/8/8/8/8/8"), apply=True)
    assert verdict.margin == 0 and "margin 0 < 8" in verdict.problems


def test_a_back_rank_pawn_under_the_chosen_turn_is_a_problem(tmp_path):
    bad = "P7/8/8/8/8/8/8/K6k"  # a pawn on a8: no frame makes this a real board
    verdict = mig.judge("x", dist_from_placement(bad, 0.9), bad, autorotated=0)
    assert verdict.rotation == 0 and any("rank 1/8" in p for p in verdict.problems)


def test_mark_adds_the_frame_only_where_the_read_agrees_unturned(tmp_path):
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({
        "a": {"placement": CHESS, "source": str(tmp_path / "a.jpg")},
        "b": {"placement": rotate_placement(CHESS, 2), "source": str(tmp_path / "b.jpg")},
    }))
    verdicts = mig.mark([gt], _read_as(CHESS), apply=True)
    assert [v.rotation for v in verdicts] == [0, 2]
    data = json.loads(gt.read_text())
    assert data["a"]["frame"] == "chess" and "frame" not in data["b"]
    assert (tmp_path / ".pre-chess-frame" / "ground_truth.json").exists()
