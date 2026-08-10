"""The one viewer test that survives a checkout with no frontend build.

It imports nothing from ``board_reader_viewer`` on purpose (see ``conftest.py``): its
whole job is to turn "the bundle is missing" into one readable skip in the report,
instead of a collection error that aborts the entire repo suite.
"""

from __future__ import annotations

import pytest
from _bundle import MISSING_UI, PAGES, UI_IS_BUILT, WEB


def test_the_ui_bundle_is_built():
    """`make ui` prerenders exactly the pages `server.py` serves, plus their assets."""
    if not UI_IS_BUILT:
        pytest.skip(MISSING_UI)
    for route, filename in PAGES.items():
        assert (WEB / filename).is_file(), route
    assert (WEB / "_app").is_dir()
