"""Tests for operator-confirmed ground-truth labels (:mod:`board_reader_inference.labels`)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.protocol import START_PLACEMENT, parse_placement

from board_reader_inference.labels import (
    LABEL_SUFFIX,
    SOURCE_CONFIRMED,
    SOURCE_CURATED,
    UPLOAD_ID,
    load_labels,
    rotate_placement,
    save_label,
)

CORNERS = "10,10,90,10,90,90,10,90"
MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"


def _photo(tmp_path, stem: str) -> None:
    """A retained upload needs its .jpg to exist for the label to be loadable."""
    (tmp_path / f"{stem}.jpg").write_bytes(b"not really a jpeg, never decoded here")


def test_rotate_placement_matches_rot90_and_round_trips():
    for k in range(4):
        expected = np.rot90(np.array(parse_placement(MIDGAME), dtype="<U5"), k)
        rotated = rotate_placement(MIDGAME, k)
        assert np.array_equal(np.array(parse_placement(rotated), dtype="<U5"), expected)
        # Undoing the turn is the operation save_label relies on.
        assert rotate_placement(rotated, -k) == MIDGAME


def test_rotation_axis_convention_is_pinned_by_a_single_piece():
    """A lone king walks the corners counter-clockwise, as ``np.rot90`` turns.

    An axis swap here would be invisible on symmetric positions and would quietly
    transpose every stored label, so it is pinned on a case that can be checked by
    eye: a1 -> h1 -> h8 -> a8 -> a1.
    """
    a1, h1, h8, a8 = "8/8/8/8/8/8/8/K7", "8/8/8/8/8/8/8/7K", "7K/8/8/8/8/8/8/8", "K7/8/8/8/8/8/8/8"
    assert rotate_placement(a1, 1) == h1
    assert rotate_placement(h1, 1) == h8
    assert rotate_placement(h8, 1) == a8
    assert rotate_placement(a8, 1) == a1
    # Four quarter-turns is the identity, on an asymmetric position too.
    assert rotate_placement(START_PLACEMENT, 4) == START_PLACEMENT
    assert rotate_placement(MIDGAME, 4) == MIDGAME


def test_save_label_stores_truth_in_the_corner_frame(tmp_path):
    """The label must not inherit the model's orientation guess.

    The operator confirms what they SEE, which is the autorotated board; truth on
    disk is that undone, so a later run whose autorotate differs is still scored
    against the same position.
    """
    shown = rotate_placement(MIDGAME, 2)  # what a 180-autorotated read displayed
    path = save_label(
        tmp_path,
        "20260806-210118-0000-estimate",
        placement=shown,
        corners=CORNERS,
        predicted=shown,
        corrections=2,
        autorotated=180,
    )
    meta = json.loads(path.read_text())
    assert meta["placement"] == MIDGAME
    assert meta["placement_shown"] == shown
    assert meta["corrections"] == 2 and meta["autorotated"] == 180
    assert path.name.endswith(LABEL_SUFFIX)


def test_save_label_without_rotation_stores_the_placement_as_given(tmp_path):
    path = save_label(
        tmp_path, "20260806-210118-0001-estimate",
        placement=MIDGAME, corners=CORNERS, predicted=MIDGAME,
        corrections=0, autorotated=0,
    )
    assert json.loads(path.read_text())["placement"] == MIDGAME


def test_save_label_rejects_a_malformed_placement(tmp_path):
    with pytest.raises(ValueError):
        save_label(
            tmp_path, "20260806-210118-0002-estimate",
            placement="not/a/position", corners=CORNERS, predicted=None,
            corrections=0, autorotated=0,
        )
    assert not list(tmp_path.glob(f"*{LABEL_SUFFIX}"))


def test_load_labels_reads_a_directory_of_confirmed_labels(tmp_path):
    stem = "20260806-210118-0000-estimate"
    _photo(tmp_path, stem)
    save_label(
        tmp_path, stem, placement=MIDGAME, corners=CORNERS,
        predicted=MIDGAME, corrections=1, autorotated=0,
    )
    boards = load_labels([tmp_path])
    assert len(boards) == 1
    board = boards[0]
    assert board.board_id == stem
    assert board.placement == MIDGAME
    assert board.source == SOURCE_CONFIRMED
    assert board.image.name == f"{stem}.jpg"
    np.testing.assert_allclose(board.corners, [[10, 10], [90, 10], [90, 90], [10, 90]])


def test_load_labels_skips_a_label_whose_photo_is_gone(tmp_path):
    """A label without its image cannot be replayed, so it is not eval data."""
    save_label(
        tmp_path, "20260806-210118-0000-estimate", placement=MIDGAME,
        corners=CORNERS, predicted=None, corrections=0, autorotated=0,
    )
    assert load_labels([tmp_path]) == []


def test_load_labels_skips_a_label_with_no_corners(tmp_path):
    """A frameless label cannot be replayed. Older labels (written before the
    server required corners) carry a null there and must be skipped, not crash
    the whole run — a scan is over a directory of other people's files."""
    stem = "20260806-210118-0000-estimate"
    _photo(tmp_path, stem)
    save_label(
        tmp_path, stem, placement=MIDGAME, corners=None,
        predicted=None, corrections=0, autorotated=0,
    )
    assert load_labels([tmp_path]) == []


def test_load_labels_survives_a_corrupt_label(tmp_path):
    stem = "20260806-210118-0000-estimate"
    _photo(tmp_path, stem)
    save_label(
        tmp_path, stem, placement=MIDGAME, corners=CORNERS,
        predicted=None, corrections=0, autorotated=0,
    )
    (tmp_path / f"20260806-210118-0009-estimate{LABEL_SUFFIX}").write_text("{ truncated")
    assert [b.board_id for b in load_labels([tmp_path])] == [stem]


def test_load_labels_reads_the_curated_ground_truth_shape(tmp_path):
    photo = tmp_path / "board.jpg"
    photo.write_bytes(b"x")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(
        json.dumps(
            {
                "img_0565": {
                    "placement": MIDGAME,
                    "source": str(photo),
                    "corners_a8_h8_h1_a1": [[10, 10], [90, 10], [90, 90], [10, 90]],
                },
                "missing_image": {
                    "placement": MIDGAME,
                    "source": str(tmp_path / "nope.jpg"),
                    "corners_a8_h8_h1_a1": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
            }
        )
    )
    boards = load_labels([gt])
    assert [b.board_id for b in boards] == ["img_0565"]
    assert boards[0].source == SOURCE_CURATED


def test_load_labels_deduplicates_and_sorts(tmp_path):
    for seq in ("0002", "0000", "0001"):
        stem = f"20260806-210118-{seq}-estimate"
        _photo(tmp_path, stem)
        save_label(
            tmp_path, stem, placement=MIDGAME, corners=CORNERS,
            predicted=None, corrections=0, autorotated=0,
        )
    ids = [b.board_id for b in load_labels([tmp_path, tmp_path])]
    assert ids == sorted(ids) and len(ids) == 3


@pytest.mark.parametrize(
    "candidate",
    ["20260806-210118-0000-estimate", "20260101-000000-9999-calibrate"],
)
def test_upload_id_accepts_real_stems(candidate):
    assert UPLOAD_ID.match(candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        "../../etc/passwd",
        "20260806-210118-0000-estimate/../x",
        "20260806-210118-0000-other",
        "20260806-210118-0000-estimate.json",
        "",
    ],
)
def test_upload_id_rejects_anything_that_is_not_a_stem(candidate):
    """An upload id is a request parameter — it must never reach a path join loose."""
    assert not UPLOAD_ID.match(candidate)
