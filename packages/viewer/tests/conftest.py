"""Viewer tests need the prerendered UI bundle to exist. Decide that once, here.

The pages are a SvelteKit build (``packages/viewer/ui`` → ``src/board_reader_viewer/web``)
that is **generated and gitignored**, so a fresh checkout has none until ``make ui`` has
run. ``create_app`` refuses to build without it (``_require_web``), so the modules that
construct an app cannot run there.

The choice is **skip, not build**: pytest must not shell out to ``pnpm`` behind
someone's back, and a run that quietly spent a minute installing node_modules would be
worse than one that tells you what to type. ``test_ui_bundle.py`` reports a single skip
carrying the instruction.

Only the modules that genuinely assert on *page content* are withheld. ``test_proxy``
and ``test_tls`` are not among them, by design: the relay and the certificate have
nothing to do with the frontend, ``server.py`` builds no module-level ``app``, and
``create_app`` no longer requires the bundle (missing pages 503; ``main`` is what
refuses to start). So a fresh clone still gets the relay and TLS coverage instead of a
directory of skips — no Node toolchain needed to test a Python byte-forwarder.
"""

from __future__ import annotations

from _bundle import UI_IS_BUILT

#: Modules that construct an app (and so need the prerendered pages on disk). Everything
#: else here is collected unconditionally.
_NEEDS_BUNDLE = frozenset({"test_viewer_server.py"})


def pytest_ignore_collect(collection_path, config):
    if UI_IS_BUILT:
        return None
    return collection_path.name in _NEEDS_BUNDLE
