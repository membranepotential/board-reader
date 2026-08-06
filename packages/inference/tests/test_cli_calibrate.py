"""Tests for ``board-reader-inference calibrate`` (rectified and --photo modes)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_inference.centroid import CentroidClassifier
from board_reader_inference.cli import main


def test_calibrate_photo_mode_records_crop_mode(tmp_path):
    photo, quad = simulate_angled_view(render_board())
    photo_path = tmp_path / "photo.png"
    out = tmp_path / "c.npz"
    cv2.imwrite(str(photo_path), photo)
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in quad)
    rc = main(
        [
            "calibrate",
            "--photo", str(photo_path),
            "--corners", corners,
            "--square-size", "64",
            "-o", str(out),
        ]
    )
    assert rc == 0
    clf = CentroidClassifier.load(out)
    assert clf.crop_mode == "photo"
    assert clf.backbone == "lite"
    present = (~np.isnan(clf.centroids).any(axis=1)).sum()
    assert present == 13  # the start position calibrates every class


def test_calibrate_rectified_mode_still_works(tmp_path):
    board_path, out = tmp_path / "board.png", tmp_path / "c.npz"
    cv2.imwrite(str(board_path), render_board())
    assert main(["calibrate", "--image", str(board_path), "-o", str(out)]) == 0
    assert CentroidClassifier.load(out).crop_mode == "rectified"


def test_calibrate_rejects_ambiguous_or_missing_sources(tmp_path):
    board_path = tmp_path / "board.png"
    cv2.imwrite(str(board_path), render_board())
    with pytest.raises(SystemExit):
        main(["calibrate"])  # neither source
    with pytest.raises(SystemExit):
        main(["calibrate", "--photo", str(board_path)])  # --photo without --corners
    with pytest.raises(SystemExit):
        main(  # both sources
            [
                "calibrate",
                "--image", str(board_path),
                "--photo", str(board_path),
                "--corners", "0,0,1,0,1,1,0,1",
            ]
        )


def test_calibrate_photo_file_drives_estimator_deterministically(tmp_path):
    """The saved --photo centroids + estimate_photo reproduce direct classification.

    Proves the plumbing (CLI -> npz -> estimator photo path, identical box
    geometry) end to end. It deliberately does NOT assert accuracy — the Lite
    embedder cannot read tall perspective boxes; real-photo quality is the
    DINOv2 evaluation's concern.
    """
    from board_reader_core.protocol import placement_from_dist
    from board_reader_core.wire import Frame
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo

    placement = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    photo, quad = simulate_angled_view(render_board(placement))
    photo_path, out = tmp_path / "p.png", tmp_path / "c.npz"
    cv2.imwrite(str(photo_path), photo)
    corners = ",".join(f"{x:.6f},{y:.6f}" for x, y in quad)
    main(
        [
            "calibrate",
            "--photo", str(photo_path),
            "--corners", corners,
            "--placement", placement,
            "--square-size", "64",
            "--no-augment",
            "-o", str(out),
        ]
    )
    clf = CentroidClassifier.load(out)
    assert clf.crop_mode == "photo"
    emb = LiteEmbedder()
    est = CentroidEstimator(
        emb, clf, square_size=64,
        headroom=clf.box_headroom, side_pad=clf.box_side_pad,
    )
    event = est.estimate_photo(
        Frame(board_id="t", seq=0, capture_ts=0.0, image=photo),
        np.asarray(quad, dtype=np.float32),
    )
    crops = crop_squares_photo(
        photo, np.asarray(quad, np.float32),
        headroom=clf.box_headroom, side_pad=clf.box_side_pad, size=64,
    )
    direct = placement_from_dist(clf.classify_board(emb.embed(crops)))
    assert event.placement == direct
