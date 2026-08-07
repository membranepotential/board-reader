"""Tests for the aspect aggregation + verdict logic in ``check_angle.py``.

Quads are synthetic on purpose: ``detect_board_quad`` is mildly
nondeterministic and the recorded frame sets live under git-ignored ``data/``,
so neither may decide whether this gate behaves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_reader_core.boarddetect import LOW_ANGLE_ASPECT  # noqa: E402
from check_angle import GOOD_ASPECT, AngleReport, Verdict, render  # noqa: E402


def quad(aspect: float) -> np.ndarray:
    """A TL,TR,BR,BL rectangle whose ``quad_aspect`` is exactly ``aspect``."""
    return np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0 * aspect], [0.0, 100.0 * aspect]],
        dtype=np.float32,
    )


def report(*aspects: float | None) -> AngleReport:
    return AngleReport.from_quads(
        "synthetic",
        ((f"f{i}", None if a is None else quad(a)) for i, a in enumerate(aspects)),
    )


def test_synthetic_quad_measures_the_aspect_it_was_built_for() -> None:
    assert report(0.65).frames[0].aspect == pytest.approx(0.65, abs=1e-4)


# --- aggregation ------------------------------------------------------------


def test_median_ignores_undetected_frames() -> None:
    assert report(0.90, None, 0.50, None, 0.70).median_aspect == pytest.approx(0.70)


def test_median_of_an_even_count_averages_the_middle_pair() -> None:
    assert report(0.60, 0.70, 0.80, 0.90).median_aspect == pytest.approx(0.75)


def test_detection_rate_counts_undetected_frames_in_the_denominator() -> None:
    r = report(0.80, None, None, 0.80)
    assert r.detection_rate == pytest.approx(0.5)
    assert len(r.aspects) == 2


def test_an_outlier_frame_cannot_swing_the_verdict() -> None:
    # A single replay/graphic that detects as a near-square board must not
    # rescue a shallow stream — which is why the median, not the max, decides.
    assert report(0.45, 0.46, 0.99).verdict is Verdict.TOO_SHALLOW


# --- verdicts ---------------------------------------------------------------


@pytest.mark.parametrize("aspect", [0.99, 0.80, GOOD_ASPECT])
def test_good_band(aspect: float) -> None:
    assert report(aspect).verdict is Verdict.GOOD


@pytest.mark.parametrize("aspect", [GOOD_ASPECT - 0.01, 0.65, LOW_ANGLE_ASPECT])
def test_degraded_band(aspect: float) -> None:
    assert report(aspect).verdict is Verdict.DEGRADED


@pytest.mark.parametrize("aspect", [LOW_ANGLE_ASPECT - 0.01, 0.45, 0.10])
def test_too_shallow_band(aspect: float) -> None:
    assert report(aspect).verdict is Verdict.TOO_SHALLOW


def test_no_detection_anywhere_is_unknown_not_a_pass() -> None:
    r = report(None, None, None)
    assert r.verdict is Verdict.UNKNOWN
    assert r.median_aspect is None
    assert r.verdict.exit_code != 0


def test_an_empty_sample_is_unknown() -> None:
    r = report()
    assert r.verdict is Verdict.UNKNOWN
    assert r.detection_rate == 0.0


# --- exit codes: this tool is a gate ---------------------------------------


@pytest.mark.parametrize(
    ("verdict", "code"),
    [
        (Verdict.GOOD, 0),
        (Verdict.DEGRADED, 0),
        (Verdict.TOO_SHALLOW, 1),
        (Verdict.UNKNOWN, 2),
    ],
)
def test_exit_codes(verdict: Verdict, code: int) -> None:
    assert verdict.exit_code == code


def test_every_verdict_has_an_exit_code_and_a_summary() -> None:
    for verdict in Verdict:
        assert verdict.exit_code >= 0
        assert verdict.summary


# --- output shapes ----------------------------------------------------------


def test_json_payload_is_serialisable_and_carries_the_thresholds() -> None:
    payload = json.loads(json.dumps(report(0.65, None).as_dict()))
    assert payload["verdict"] == "degraded"
    assert payload["sampled"] == 2
    assert payload["detected"] == 1
    assert payload["frames"][1]["aspect"] is None
    assert payload["low_angle_aspect"] == LOW_ANGLE_ASPECT
    assert payload["good_aspect"] == GOOD_ASPECT


def test_render_survives_a_report_with_no_detections() -> None:
    text = render(report(None))
    assert "no board detected" in text
    assert "UNKNOWN" in text
