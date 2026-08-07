"""Tests for the timestamp helpers in ``extract_frames.py``.

No ffmpeg, no video: only the pure sampling/formatting logic, which is where a
silent regression would quietly poison a whole frame set.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_frames import (
    even_times,
    format_timestamp,
    parse_time,
    random_times,
)

DURATION = 10_000.0


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0", 0.0),
        ("90", 90.0),
        ("1:30", 90.0),
        ("01:42:46", 6166.0),
        ("2:00:00", 7200.0),
        ("0.5", 0.5),
    ],
)
def test_parse_time(text: str, seconds: float) -> None:
    assert parse_time(text) == seconds


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(0, "00h00m00s"), (6166, "01h42m46s"), (86_399, "23h59m59s"), (6166.9, "01h42m46s")],
)
def test_format_timestamp(seconds: float, formatted: str) -> None:
    assert format_timestamp(seconds) == formatted


def test_timestamps_round_trip_through_the_filename_format() -> None:
    assert parse_time("01:42:46") == 6166.0
    assert format_timestamp(6166.0).replace("h", ":").replace("m", ":").rstrip("s") == (
        "01:42:46"
    )


# --- random sampling --------------------------------------------------------


def test_same_seed_gives_the_same_times() -> None:
    assert random_times(DURATION, 8, 0.02, seed=7) == random_times(
        DURATION, 8, 0.02, seed=7
    )


def test_different_seeds_give_different_times() -> None:
    assert random_times(DURATION, 8, 0.02, seed=7) != random_times(
        DURATION, 8, 0.02, seed=8
    )


def test_seeding_does_not_disturb_the_global_rng() -> None:
    # random_times must use its own Random(), or seeding it here would make an
    # unrelated caller's draws reproducible too.
    import random

    random.seed(1)
    before = random.random()
    random.seed(1)
    random_times(DURATION, 5, 0.02, seed=99)
    assert random.random() == before


def test_unseeded_draws_differ() -> None:
    assert random_times(DURATION, 8, 0.02, seed=None) != random_times(
        DURATION, 8, 0.02, seed=None
    )


@pytest.mark.parametrize("margin", [0.0, 0.02, 0.25])
def test_random_times_honour_the_margin(margin: float) -> None:
    times = random_times(DURATION, 200, margin, seed=3)
    assert min(times) >= DURATION * margin
    assert max(times) <= DURATION * (1 - margin)


def test_random_times_are_sorted_and_counted() -> None:
    times = random_times(DURATION, 12, 0.02, seed=3)
    assert len(times) == 12
    assert times == sorted(times)


@pytest.mark.parametrize("margin", [-0.01, 0.5, 1.0])
def test_out_of_range_margin_is_rejected(margin: float) -> None:
    with pytest.raises(SystemExit, match="margin"):
        random_times(DURATION, 5, margin, seed=1)


# --- even sampling (what check_angle uses) ----------------------------------


def test_even_times_span_the_margined_range_inclusively() -> None:
    times = even_times(DURATION, 5, 0.02)
    assert times[0] == pytest.approx(200.0)
    assert times[-1] == pytest.approx(9800.0)
    assert len(times) == 5


def test_even_times_are_evenly_spaced() -> None:
    times = even_times(DURATION, 6, 0.02)
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert gaps == pytest.approx([gaps[0]] * len(gaps))


def test_a_single_even_sample_lands_mid_video() -> None:
    assert even_times(DURATION, 1, 0.02) == pytest.approx([5000.0])


def test_even_times_needs_at_least_one_sample() -> None:
    with pytest.raises(SystemExit, match="at least one"):
        even_times(DURATION, 0, 0.02)
