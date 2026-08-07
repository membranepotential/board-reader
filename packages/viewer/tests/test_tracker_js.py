"""Run ``static/tracker.js`` under node and check its mathematics.

The rest of the camera page is markup and DOM wiring, tested structurally against the
served HTML. The tracker is not: it is optical flow and a homography fit, and a string
assertion cannot tell a working RANSAC from a broken one. So this test *executes* it,
against synthetic scenes whose answer we chose in advance (``tracker_checks.mjs``).

The repo has no ``package.json``, so node reads a bare ``.js`` as CommonJS and the
module's ``export``s fail to parse — the module is therefore copied into the tmp dir
as ``tracker.mjs``, which node always treats as ESM.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_TRACKER = _TESTS.parent / "src" / "board_reader_viewer" / "static" / "tracker.js"
_CHECKS = _TESTS / "tracker_checks.mjs"


def test_tracker_module_exists_and_is_served_to_the_page():
    """It lives under ``static/``, so the viewer's StaticFiles mount serves it as-is —
    no build step, no bundler, nothing to keep in sync with the page."""
    assert _TRACKER.is_file()
    source = _TRACKER.read_text(encoding="utf-8")
    assert "export class PlaneTracker" in source
    # Self-contained and DOM-free — which is what lets the test below run it under plain
    # node, and what keeps the page's one dependency a file the viewer already serves.
    assert "\nimport " not in source
    for forbidden in ("document.", "navigator.", ".getContext(", "requestAnimationFrame"):
        assert forbidden not in source, forbidden


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tracker_math_under_node(tmp_path):
    shutil.copy(_TRACKER, tmp_path / "tracker.mjs")
    shutil.copy(_CHECKS, tmp_path / "checks.mjs")
    result = subprocess.run(
        ["node", "checks.mjs"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout
