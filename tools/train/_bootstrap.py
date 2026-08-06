"""Make the deployed project packages importable from this scratch training venv.

The training harness runs in a *separate* torch venv (the project workspace stays
torch-free, CLAUDE.md). It must nonetheless crop squares with the EXACT deployed
geometry, so instead of copying ``crop_squares_photo`` we put the project's
``core`` and ``inference`` source roots on ``sys.path`` and import the real thing.
That guarantees training crops can never silently drift from what the server cuts
at inference time.

Import this module (``import _bootstrap``) before importing any
``board_reader_*`` package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# tools/train/_bootstrap.py -> repo root is three parents up.
ROOT = Path(__file__).resolve().parents[2]

_SRC_ROOTS = (
    ROOT / "packages" / "core" / "src",
    ROOT / "packages" / "inference" / "src",
)

for _p in _SRC_ROOTS:
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _register_light_inference() -> None:
    """Expose ``board_reader_inference.squares`` WITHOUT its heavy package init.

    The real ``board_reader_inference/__init__`` eagerly imports the FastAPI server
    (and its whole dependency tree), which the torch training venv deliberately
    does not install. We only need the pure-numpy/cv2 geometry (``squares``), so we
    pre-seed ``sys.modules`` with a stub package whose ``__path__`` points at the
    real source dir. Submodule imports (``board_reader_inference.squares``) then
    resolve normally and never trigger the real ``__init__``.
    """
    name = "board_reader_inference"
    if name in sys.modules:
        return
    src = ROOT / "packages" / "inference" / "src" / "board_reader_inference"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(src)]
    pkg.__package__ = name
    sys.modules[name] = pkg


_register_light_inference()

__all__ = ["ROOT"]
