"""Every commit SHA cited in a tracked file must still resolve.

This repo cites commits in prose — ``STATUS.md`` credits work to them, docstrings
point at the change that explains a workaround, the ``Makefile`` names the commit
whose drift a comment describes. Prose is exactly what no other check reads, so a
history rewrite or a dropped commit silently turns those into dangling pointers.

That is not hypothetical: the 2026-08-20 author-email rewrite invalidated all 97
SHAs and left 11 dangling citations behind. This is the guard that would have
caught it, and the tool it drives (``tools/check_commit_refs.py``) can also repair
them given the pre-rewrite history.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TOOL = _ROOT / "tools" / "check_commit_refs.py"


def _load_tool():
    """Import the tool by path — ``tools/`` is scripts, not an installed package.

    The module must be in ``sys.modules`` *before* it executes: ``dataclasses``
    resolves the field types of a ``slots=True`` class through
    ``sys.modules[cls.__module__]``, which is None for a module loaded by spec
    alone, and the class bodies then fail at import time.
    """
    if (cached := sys.modules.get("check_commit_refs")) is not None:
        return cached
    spec = importlib.util.spec_from_file_location("check_commit_refs", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_git_checkout() -> bool:
    return subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "--git-dir"],
        capture_output=True,
        check=False,
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _is_git_checkout(),
    reason="needs a git checkout (source tarballs carry no history to resolve against)",
)


def test_no_dangling_commit_citations() -> None:
    tool = _load_tool()
    ignore = tool.KNOWN_NON_COMMITS | tool.learning_ids(_ROOT)
    citations = tool.scan(_ROOT, tool.tracked_files(_ROOT), ignore)
    resolved = tool.resolve(_ROOT, sorted({c.sha for c in citations}))

    dangling = sorted(
        {f"{c.path}:{c.line}: {c.sha}" for c in citations if resolved.get(c.sha) is None}
    )
    assert not dangling, (
        "commit SHAs cited in tracked files no longer resolve:\n  "
        + "\n  ".join(dangling)
        + "\n\nRepair them with:\n"
        "  uv run tools/check_commit_refs.py --baseline <pre-rewrite repo> --apply"
    )


def test_scanner_rejects_the_hex_that_is_not_a_sha() -> None:
    """The filters that keep the signal usable — each one earns its place.

    Without them a bare hex scan over this repo returns ~400 hits for 8 real
    citations, and a guard nobody can read is a guard nobody keeps.
    """
    tool = _load_tool()
    ignore = tool.KNOWN_NON_COMMITS | tool.learning_ids(_ROOT)
    citations = tool.scan(_ROOT, tool.tracked_files(_ROOT), ignore)
    seen = {c.sha for c in citations}

    assert "20260813" not in seen, "date stamps are all-digits and must be filtered"
    assert "4294967296" not in seen, "magic numbers are all-digits and must be filtered"
    assert not any(c.path.name == "uv.lock" for c in citations), "lockfiles are skipped"
    # A learning id cited in another learning's prose is shape-identical to a
    # short SHA; the store itself is what tells them apart.
    assert "85aab09c98c4" not in seen, "learning ids come from the store, not a list"
