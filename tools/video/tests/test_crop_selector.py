"""RectSelector: the mouse-drag state machine, minus the GUI.

`show()` is stubbed out — headless cv2 has the event constants but no window
stack, and the clamp logic is what needs pinning, not the rendering.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_selector import RectSelector  # noqa: E402


def _selector(w: int = 640, h: int = 360) -> RectSelector:
    sel = RectSelector(frame=np.zeros((h, w, 3), dtype=np.uint8))
    # object.__setattr__-free: slots dataclass, but show is a method — stub via
    # subclass would be heavier; monkeypatch the bound lookup instead.
    RectSelector.show = lambda self: None  # type: ignore[method-assign]
    return sel


def test_drag_released_outside_frame_is_clamped() -> None:
    """The observed crash: a drag ending at y=-6 must clamp, not raise."""
    sel = _selector()
    sel.on_mouse(cv2.EVENT_LBUTTONDOWN, 187, 300, 0, None)
    sel.on_mouse(cv2.EVENT_MOUSEMOVE, 500, 10, 0, None)
    sel.on_mouse(cv2.EVENT_LBUTTONUP, 700, -6, 0, None)  # past right + top edges
    box = sel.box
    assert box is not None
    assert box.x == 187 and box.y == 0
    assert box.x + box.width <= 640
    assert box.y + box.height <= 360


def test_in_frame_drag_unchanged() -> None:
    sel = _selector()
    sel.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
    sel.on_mouse(cv2.EVENT_LBUTTONUP, 110, 220, 0, None)
    box = sel.box
    assert box is not None
    assert (box.x, box.y, box.width, box.height) == (10, 20, 100, 200)
