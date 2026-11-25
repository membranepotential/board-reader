"""Tests for the error taxonomy (:mod:`board_reader_inference.measurement.evaluate.errors`).
"""

from __future__ import annotations

import pytest

from board_reader_inference.measurement.evaluate.errors import error_kind


@pytest.mark.parametrize(
    ("truth", "predicted", "kind"),
    [
        ("empty", "P", "hallucination"),
        ("P", "empty", "missed"),
        ("P", "p", "colour"),
        ("k", "K", "colour"),
        ("Q", "K", "type"),
        ("r", "p", "type"),
    ],
)
def test_error_kind_separates_occupancy_from_recognition(truth, predicted, kind):
    assert error_kind(truth, predicted) == kind
