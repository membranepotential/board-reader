"""Pin the read seam: one calibrated place where a distribution becomes an event.

:func:`board_reader_inference.estimator.read_event` is where the black-bishop
demotion is applied (docs/reports/2026-08-27-evidence-rerank.md, §6). Nothing in
the *type* system stops a future estimator from calling
``PositionEvent.from_square_dist`` itself, and if one did it would serve
uncalibrated reads — silently, with every existing test still green, because an
uncalibrated read is a perfectly valid event.

So the call sites are enumerated here instead. Adding one is fine; adding one
*without noticing* is what this prevents. The scan is AST, not text, for the same
reason ``tests/test_layer_separation.py``'s is: a docstring mentioning the name
must not count, and a call must not hide behind formatting.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

#: Every place in the inference source that may build a ``PositionEvent`` from a
#: distribution, as ``module: qualified function``, and why it is allowed to.
ALLOWED_CONSTRUCTORS = {
    # THE seam: applies demote_attractor, then builds the event. Estimators that
    # read pixels go through here — see its docstring.
    ("estimator.py", "read_event"),
    # The walking skeleton emits an AUTHORED placement, not a model output, so
    # there is nothing to calibrate and a stub that served a different board than
    # the one it was configured with would make M0 lie about the pipe.
    ("estimator.py", "StubEstimator.estimate"),
    # A bare exemplar bank is outside the calibration's measured domain: 0.20 is
    # an absolute margin fitted to the trained stack, and a kNN bank's own
    # CORRECT bishops read inside the band (0.105 / 0.116 on the fixture).
    ("estimator.py", "CentroidEstimator._event"),
    # The recognizer seam (DESIGN D10): ChessQ and any later recognizer are
    # different models entirely, and the 0.20 margin was fitted on the trained
    # stack's distribution, never measured on theirs.
    ("recognizer.py", "event_from_dist"),
}


class _CallScan(ast.NodeVisitor):
    """Collect the qualified scope of every call to a given function name."""

    def __init__(self, module: str, wanted: str) -> None:
        self.module = module
        self.wanted = wanted
        self.scope: list[str] = []
        self.sites: set[tuple[str, str]] = set()

    def _scoped(self, node: ast.AST) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_ClassDef = _scoped

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == self.wanted:
            self.sites.add((self.module, ".".join(self.scope)))
        self.generic_visit(node)


def _call_sites(name: str) -> set[tuple[str, str]]:
    """Every call to ``name`` in the inference source, by enclosing scope."""
    sites: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        scan = _CallScan(path.name, name)
        scan.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        sites |= scan.sites
    return sites


def test_only_the_registered_sites_build_a_position_event_from_a_distribution():
    """A new call site must register itself above, with the reason it is not
    going through the seam. If it has no such reason, it should call
    ``read_event`` — that is the fix, not an entry in the set."""
    found = _call_sites("from_square_dist")

    unregistered = found - ALLOWED_CONSTRUCTORS
    assert not unregistered, (
        "new PositionEvent.from_square_dist call site(s) in the inference source: "
        f"{sorted(unregistered)}. Either read through estimator.read_event (so the "
        "black-bishop calibration is applied), or add it to ALLOWED_CONSTRUCTORS "
        "with the reason it must not be."
    )
    stale = ALLOWED_CONSTRUCTORS - found
    assert not stale, f"ALLOWED_CONSTRUCTORS names sites that no longer exist: {sorted(stale)}"


def test_the_seam_is_the_only_caller_of_the_demotion():
    """The mirror image: nothing may apply ``demote_attractor`` on its own either.
    Applied twice it is idempotent, but applied *before* a blend it is not — that
    is exactly the bug PhotoDistReader exists to prevent."""
    assert _call_sites("demote_attractor") == {("estimator.py", "read_event")}
