"""Enforce the edge/inference layer split (DESIGN.md §4 — they meet only on the wire).

These tests are guardrails: the architecture says the edge (Pi) and the inference
node never share code, only the wire contract in ``board_reader_core``. We check it
two ways — statically (no forbidden imports in source) and structurally (the
package manifests declare no cross-dependency).
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PKGS = _ROOT / "packages"


def _py_files(pkg: str) -> list[Path]:
    return list((_PKGS / pkg / "src").rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names actually imported by a module (AST, not prose)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _all_imports(pkg: str) -> set[str]:
    roots: set[str] = set()
    for f in _py_files(pkg):
        roots |= _imported_roots(f)
    return roots


def test_edge_source_never_imports_inference():
    assert "board_reader_inference" not in _all_imports("edge")


def test_inference_source_never_imports_edge():
    assert "board_reader_edge" not in _all_imports("inference")


def test_core_never_imports_edge_or_inference():
    """Core is the shared contract — it must not depend upward on either node."""
    roots = _all_imports("core")
    assert "board_reader_edge" not in roots
    assert "board_reader_inference" not in roots


def _deps(pkg: str) -> list[str]:
    data = tomllib.loads((_PKGS / pkg / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"].get("dependencies", [])


def test_manifests_declare_no_cross_dependency():
    edge_deps = " ".join(_deps("edge"))
    inf_deps = " ".join(_deps("inference"))
    assert "board-reader-inference" not in edge_deps
    assert "board-reader-edge" not in inf_deps
    # both legitimately depend on core
    assert "board-reader-core" in edge_deps
    assert "board-reader-core" in inf_deps


def test_core_does_not_depend_on_any_sibling():
    core_deps = " ".join(_deps("core"))
    assert "board-reader-edge" not in core_deps
    assert "board-reader-inference" not in core_deps
    assert "board-reader-viewer" not in core_deps
