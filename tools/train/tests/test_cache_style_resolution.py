"""Cache-backed style resolution (the cloud-pod case).

A pod holds staged crop caches and (almost) no raw datasets, so training must
resolve styles from complete caches when the raw registry misses them — the
live failure that motivated this: run_g on RunPod died with
``styles not found: [...]; available: []`` while a perfectly good 16 GB cache
sat on the volume.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cropcache as cropcache_mod
import pytest
from config import Config
from cropcache import cached_style_modes, ensure_caches, resolve_styles


def _mk_cache(root: Path, name: str, mode: str = "board", *, legacy: bool = False) -> None:
    d = root / name
    d.mkdir(parents=True)
    meta = {
        "style": name,
        "kind": "unknown",
        "n_boards": 2,
        "n_crops": 128,
        "size": 224,
        "headroom": 1.6,
        "side_pad": 0.15,
        "shards": ["shard_000.npz"],
        "group_names": ["g0"],
        "class_counts": {},
    }
    if not legacy:
        meta["mode"] = mode
    (d / "meta.json").write_text(json.dumps(meta))


def _cfg(tmp_path: Path, **over) -> Config:
    base = {
        "styles": ["all"],
        "headroom": 1.6,
        "side_pad": 0.15,
        "mask": "gray",
        "out_dir": str(tmp_path / "out"),
    }
    base.update(over)
    return Config(**base)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "_cropcache" / "hr1.6_sp0.15_maskgray"
    root.mkdir(parents=True)
    monkeypatch.setattr(cropcache_mod, "geo_cache_root", lambda cfg: root)
    return root


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(cropcache_mod, "discover_styles", dict)


def test_explicit_styles_resolve_from_cache_alone(cache_root, empty_registry, tmp_path):
    _mk_cache(cache_root, "chessred")
    _mk_cache(cache_root, "w2_bellanger")
    cfg = _cfg(tmp_path, styles=["chessred", "w2_bellanger"], require_mode="board")
    assert resolve_styles(cfg) == ["chessred", "w2_bellanger"]


def test_all_appends_cache_only_styles(cache_root, empty_registry, tmp_path):
    _mk_cache(cache_root, "b_style")
    _mk_cache(cache_root, "a_style")
    cfg = _cfg(tmp_path, styles=["all"], require_mode="board")
    assert resolve_styles(cfg) == ["a_style", "b_style"]  # sorted cache scan


def test_require_mode_filters_cached_styles(cache_root, empty_registry, tmp_path):
    _mk_cache(cache_root, "boardy", mode="board")
    _mk_cache(cache_root, "croppy", mode="crop")
    cfg = _cfg(tmp_path, styles=["all"], require_mode="board")
    assert resolve_styles(cfg) == ["boardy"]
    with pytest.raises(SystemExit, match="styles not found"):
        resolve_styles(_cfg(tmp_path, styles=["croppy"], require_mode="board"))


def test_legacy_meta_without_mode_reads_as_board(cache_root, empty_registry, tmp_path):
    _mk_cache(cache_root, "old_cache", legacy=True)
    assert cached_style_modes(_cfg(tmp_path)) == {"old_cache": "board"}


def test_missing_style_still_dies(cache_root, empty_registry, tmp_path):
    _mk_cache(cache_root, "present")
    with pytest.raises(SystemExit, match="styles not found"):
        resolve_styles(_cfg(tmp_path, styles=["absent"]))


def test_registry_wins_over_cache_for_all(cache_root, tmp_path, monkeypatch):
    class FakeStyle:
        mode = "board"

    monkeypatch.setattr(cropcache_mod, "discover_styles", lambda: {"reg_style": FakeStyle()})
    _mk_cache(cache_root, "cache_style")
    _mk_cache(cache_root, "reg_style")  # cached AND registered -> registry order first
    cfg = _cfg(tmp_path, styles=["all"], require_mode="board")
    assert resolve_styles(cfg) == ["reg_style", "cache_style"]


def test_ensure_caches_skips_complete_caches_without_registry(
    cache_root, empty_registry, tmp_path
):
    _mk_cache(cache_root, "chessred")
    # Would raise KeyError/SystemExit if it consulted the (empty) registry.
    ensure_caches(["chessred"], _cfg(tmp_path))


def test_ensure_caches_still_needs_registry_for_missing(cache_root, empty_registry, tmp_path):
    with pytest.raises(KeyError):
        ensure_caches(["never_built"], _cfg(tmp_path))


def test_discover_styles_follows_symlinked_dirs(tmp_path):
    """Pod layout: every style under data/external is a symlink onto the volume.

    pathlib's ``**`` glob does not descend into symlinked dirs, which made the
    pod's registry empty; discovery must walk with followlinks instead.
    """
    import styles as styles_mod

    real = tmp_path / "volume" / "somestyle"
    real.mkdir(parents=True)
    (real / styles_mod.NORMALIZED_MANIFEST).write_text(json.dumps([]))
    external = tmp_path / "external"
    external.mkdir()
    (external / "somestyle").symlink_to(real)

    seen: list[Path] = []

    def spy(manifest):
        seen.append(manifest)
        return {}

    real_loader = styles_mod.load_normalized_styles
    styles_mod.load_normalized_styles = spy
    try:
        styles_mod.discover_styles(external)
    finally:
        styles_mod.load_normalized_styles = real_loader

    assert seen == [external / "somestyle" / styles_mod.NORMALIZED_MANIFEST], (
        "discovery did not reach the manifest behind the symlinked style dir"
    )
