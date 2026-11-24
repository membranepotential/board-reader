"""The inference node's subpackages import one way only (.claude/rules/code-organization.md).

``recognition`` imports nothing from the other groups, ``stream`` may import
``recognition``, ``measurement`` may import both, ``control`` stands alone, and
only the package root (the app factory, the CLI, the headless viewer client)
wires them together. That is what keeps the replay harness on the same read path
as the live node: if ``measurement`` could reach into the root, the gate could
measure something the node does not run.

Imports are read with :mod:`ast` (absolute and relative, function-level
included), never by substring, because docstrings legitimately name the other
groups.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "board_reader_inference"
SRC = Path(__file__).resolve().parents[1] / "packages" / "inference" / "src" / PACKAGE

ROOT = "<root>"
ROOT_MODULES = {"__init__", "server", "cli", "positions"}
GROUPS = ("recognition", "stream", "control", "measurement")
#: group -> the other groups it may import. The root may import anything.
ALLOWED: dict[str, frozenset[str]] = {
    "recognition": frozenset(),
    "stream": frozenset({"recognition"}),
    "control": frozenset(),
    "measurement": frozenset({"recognition", "stream"}),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _group_of(module: str) -> str:
    """``board_reader_inference.stream.ingest`` -> ``stream``; root modules -> ROOT."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in GROUPS else ROOT


def _imported_modules(path: Path) -> set[str]:
    """Every in-package module ``path`` imports, relative imports resolved."""
    is_init = path.name == "__init__.py"
    package = (
        _module_name(path).split(".") if is_init else _module_name(path).split(".")[:-1]
    )
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            found.add(target)
            # `from board_reader_inference import stream` names a group too.
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return {m for m in found if m == PACKAGE or m.startswith(PACKAGE + ".")}


def _edges() -> list[tuple[str, str, str]]:
    """(importing module, its group, imported group) for every cross-group import."""
    edges = []
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        group = _group_of(module)
        for target in _imported_modules(path):
            target_group = _group_of(target)
            if target_group != group and target != PACKAGE:
                edges.append((module, group, target_group))
    return edges


def test_the_root_holds_only_the_entry_points():
    root = {p.stem for p in SRC.glob("*.py")}
    assert root == ROOT_MODULES, "put a new module into the group whose concern it serves"
    assert {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__"} == set(
        GROUPS
    )


def test_every_group_has_a_docstring_init():
    for group in GROUPS:
        init = SRC / group / "__init__.py"
        assert ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))), init


def test_groups_import_one_way_only():
    violations = [
        f"{module} ({group}) imports {target}"
        for module, group, target in _edges()
        if group != ROOT and target not in ALLOWED[group]
    ]
    assert violations == []


def test_the_scan_sees_cross_group_imports():
    """Guard the guard: the known edges exist, so an empty violation list means something."""
    edges = {(group, target) for _, group, target in _edges()}
    assert ("stream", "recognition") in edges
    assert ("measurement", "stream") in edges
    assert (ROOT, "control") in edges
