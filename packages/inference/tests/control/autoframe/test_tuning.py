"""Auto-frame's thresholds (control/autoframe/tuning.py)."""

from __future__ import annotations

import pytest

from board_reader_inference.control.autoframe.tuning import Tuning


@pytest.mark.parametrize(
    "bad",
    [
        {"preview_min_r2": 1.5},
        {"min_preview_share": -0.1},
        {"unseen_previews": 0},
        {"drift": 0.0},
        {"drift_previews": 0},
        {"full_frame_timeout": 0.0},
        {"backoff": ()},
        {"backoff": (10.0, 3.0)},
    ],
)
def test_tuning_refuses_nonsense(bad):
    with pytest.raises(ValueError):
        Tuning(**bad)
