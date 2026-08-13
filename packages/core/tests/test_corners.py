"""Tests for the shared ``--corners``/``corners`` string parser."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.corners import parse_corners


def test_parses_eight_comma_separated_numbers_in_order():
    quad = parse_corners("0,0,10,0,10,10,0,10")
    assert quad.dtype == np.float32
    assert quad.shape == (4, 2)
    assert quad.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_semicolons_are_accepted_as_a_separator_too():
    assert np.array_equal(
        parse_corners("0,0,10,0,10,10,0,10"),
        parse_corners("0;0;10;0;10;10;0;10"),
    )


def test_wrong_count_is_a_value_error():
    with pytest.raises(ValueError, match="8 numbers"):
        parse_corners("0,0,10,0,10,10")


def test_non_numeric_input_is_a_value_error():
    with pytest.raises(ValueError):
        parse_corners("a,b,c,d,e,f,g,h")


def test_non_string_input_is_a_value_error():
    """A label sidecar predating the corners requirement carries a null there —
    a loader must be able to catch this rather than crash on it (see
    board_reader_inference.labels)."""
    with pytest.raises(ValueError, match="comma-separated string"):
        parse_corners(None)  # type: ignore[arg-type]
