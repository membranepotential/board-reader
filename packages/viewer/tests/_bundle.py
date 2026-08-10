"""Where the prerendered UI bundle lives, and whether it is there yet.

Shared by ``conftest.py`` (which withholds the tests that need it) and the tests
themselves (which read the built files). Kept out of ``conftest.py`` so both can import
it by an unambiguous name — there is more than one ``conftest`` module in this repo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _web_dir() -> Path:
    """The bundle directory, found without *importing* the package.

    ``find_spec`` reads the installed package's location off its loader, which answers
    "where would the bundle be?" without importing anything — the question this module
    has to answer before ``conftest`` decides what to collect.
    """
    spec = importlib.util.find_spec("board_reader_viewer")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))) / "web"
    # Not installed at all: fall back to the source tree, so a missing bundle still
    # explains itself instead of raising out of a conftest.
    return Path(__file__).resolve().parents[1] / "src" / "board_reader_viewer" / "web"


WEB = _web_dir()

#: The three prerendered pages ``server.py`` serves, by the route that serves each.
PAGES = {"/": "index.html", "/snapshot": "snapshot.html", "/camera": "camera.html"}

#: ``server._require_web`` treats the bundle as present exactly when ``index.html`` is,
#: so this suite uses that same definition rather than a second one that could drift.
UI_IS_BUILT = (WEB / "index.html").is_file()

MISSING_UI = (
    f"no UI build at {WEB} — run `make ui` (or, from packages/viewer/ui, "
    "`pnpm install --frozen-lockfile && pnpm build`). Only the tests that assert on "
    "page content are skipped; the relay and TLS tests do not need the frontend and "
    "have run."
)
