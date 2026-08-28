"""E7: occlusion-masked crop-cache plumbing (cropcache.py + train.py's cache
keying). The occlusion geometry itself (which pixels get replaced, gray vs blur,
env-var default) is squares.py's own contract and is fully covered by
packages/inference/tests/test_squares.py; these tests only check that this
harness threads ``mask`` through correctly and keys the cache so a masked build
never collides with (or gets silently reused as) an unmasked one.

Runs under the torch venv (see README "Tests").
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from cropcache import build_cache, geo_cache_root, load_meta
from styles import CROP, CropSample, Style, StyleSample

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view

from board_reader_inference.squares import BboxGeometry, PoseGeometry


@pytest.fixture()
def board_style(tmp_path: Path) -> Style:
    """One real, angled synthetic board photo -- masking has something to bite
    on (occupied squares occlude squares behind them at a steep angle)."""
    board = render_board(START_PLACEMENT, square_px=48)
    photo, quad = simulate_angled_view(board)
    img_path = tmp_path / "board.png"
    cv2.imwrite(str(img_path), photo)
    sample = StyleSample(image_path=img_path, placement=START_PLACEMENT, corners=quad, group="g")
    return Style(name="synthtest", kind="synthetic", _samples=[sample])


@pytest.fixture()
def crop_style(tmp_path: Path) -> Style:
    """A CROP-mode style: already-cut single-square images, no corners."""
    img_path = tmp_path / "sq.png"
    cv2.imwrite(str(img_path), np.zeros((32, 32, 3), dtype=np.uint8))
    sample = CropSample(image_path=img_path, label="empty", group="g")
    return Style(name="croptest", kind="real", _samples=[sample], mode=CROP)


def test_build_cache_records_mask_in_meta(tmp_path, board_style):
    out_root = tmp_path / "cache"
    meta = build_cache(board_style, out_root=out_root, geometry=BboxGeometry(1.6, 0.15), mask="gray")
    assert meta.mask == "gray"
    reloaded = load_meta(board_style.name, out_root=out_root)
    assert reloaded.mask == "gray"


def test_build_cache_unmasked_records_none(tmp_path, board_style):
    out_root = tmp_path / "cache"
    meta = build_cache(board_style, out_root=out_root, geometry=BboxGeometry(1.6, 0.15))
    assert meta.mask is None


def test_masked_and_unmasked_caches_differ(tmp_path, board_style):
    """The whole point: masking a board that actually has occluding pieces must
    change at least one crop's bytes relative to the unmasked cache."""
    unmasked_root, masked_root = tmp_path / "u", tmp_path / "m"
    build_cache(board_style, out_root=unmasked_root, geometry=BboxGeometry(1.6, 0.15))
    build_cache(board_style, out_root=masked_root, geometry=BboxGeometry(1.6, 0.15), mask="gray")
    u_shard = np.load(next((unmasked_root / board_style.name).glob("shard_*.npz")))
    m_shard = np.load(next((masked_root / board_style.name).glob("shard_*.npz")))
    assert not np.array_equal(u_shard["blob"], m_shard["blob"]), (
        "masked cache must not be byte-identical to the unmasked one on an angled board"
    )


def test_crop_mode_style_ignores_mask_and_says_so(tmp_path, crop_style, capsys):
    out_root = tmp_path / "cache"
    meta = build_cache(crop_style, out_root=out_root, mask="gray")
    assert meta.mask is None
    assert "unmasked despite mask" in capsys.readouterr().out


def test_stale_cache_rebuilds_when_mask_changes(tmp_path, board_style):
    """A cache built unmasked must not be silently served up for a masked
    request against the same out_root (defensive: normally avoided entirely by
    geo_cache_root's mask-suffixed directory, but build_cache must not trust that)."""
    out_root = tmp_path / "cache"
    build_cache(board_style, out_root=out_root, geometry=BboxGeometry(1.6, 0.15))
    remasked = build_cache(board_style, out_root=out_root, geometry=BboxGeometry(1.6, 0.15), mask="gray")
    assert remasked.mask == "gray"


def test_geo_cache_root_suffixes_mask_for_bbox_geometry():
    unmasked = geo_cache_root(Config(headroom=1.6, side_pad=0.15))
    masked = geo_cache_root(Config(headroom=1.6, side_pad=0.15, mask="gray"))
    assert masked != unmasked
    assert masked.name == f"{unmasked.name}_maskgray"


def test_geo_cache_root_suffixes_mask_for_pose_geometry():
    cfg = Config(crop_geometry={"kind": "pose", "piece_height": 1.9, "piece_radius": 0.5, "focal_ratio": 1.0})
    unmasked = geo_cache_root(cfg)
    cfg.mask = "gray"
    masked = geo_cache_root(cfg)
    assert masked != unmasked
    assert masked.name.endswith("_maskgray")
    assert isinstance(cfg.geometry, PoseGeometry)


def test_geo_cache_root_default_geometry_still_maps_to_cache_root_when_unmasked():
    from cropcache import CACHE_ROOT

    assert geo_cache_root(Config()) == CACHE_ROOT


def test_geo_cache_root_default_geometry_gets_masked_subdir():
    from cropcache import CACHE_ROOT

    root = geo_cache_root(Config(mask="gray"))
    assert root != CACHE_ROOT
    assert root.parent == CACHE_ROOT


def test_config_rejects_unknown_mask_value(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mask: sepia\n")
    with pytest.raises(ValueError, match="mask"):
        Config.load(bad)
