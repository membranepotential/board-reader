"""Tests for the settle debounce's knobs (:mod:`board_reader_inference.stream.settle.config`)."""

from __future__ import annotations

import pytest

from board_reader_inference.stream.settle.config import OBSCURED_ENERGY, SettleConfig


def test_rejects_nonsensical_parameters():
    with pytest.raises(ValueError):
        SettleConfig(seconds=-1.0)
    with pytest.raises(ValueError):
        SettleConfig(seconds=float("nan"))
    with pytest.raises(ValueError):
        SettleConfig(min_reads=0)
    with pytest.raises(ValueError):
        SettleConfig(stale_windows=0)
    for energy in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            SettleConfig(obscured_energy=energy)


def test_settle_config_from_env_keeps_unset_knobs_at_their_defaults():
    assert SettleConfig.from_env({"BR_SETTLE_SECONDS": "1.5", "BR_SETTLE_MIN_READS": ""}) == (
        SettleConfig(seconds=1.5)
    )
    assert SettleConfig.from_env({}) == SettleConfig()
    assert SettleConfig.from_env(
        {"BR_SETTLE_SECONDS": "0.5", "BR_SETTLE_MIN_READS": "1", "BR_SETTLE_STALE_WINDOWS": "3"}
    ) == SettleConfig(seconds=0.5, min_reads=1, stale_windows=3)
    assert SettleConfig.from_env({"BR_OBSCURED_ENERGY": "-11.5"}) == (
        SettleConfig(obscured_energy=-11.5)
    )
    assert SettleConfig.from_env({"BR_OBSCURED_ENERGY": "off"}) == (
        SettleConfig(obscured_energy=None)
    )


@pytest.mark.parametrize(
    "env",
    [
        {"BR_SETTLE_SECONDS": "fast"},
        {"BR_SETTLE_MIN_READS": "1.5"},
        {"BR_SETTLE_STALE_WINDOWS": "0"},
        {"BR_OBSCURED_ENERGY": "high"},
        {"BR_OBSCURED_ENERGY": "nan"},
    ],
)
def test_a_malformed_settle_env_fails_loudly(env):
    with pytest.raises(ValueError):
        SettleConfig.from_env(env)


def test_settle_config_names_itself_for_a_replay_directory():
    assert SettleConfig().is_default
    config = SettleConfig(seconds=1.5, min_reads=1, stale_windows=3)
    assert not config.is_default
    assert config.slug == "settle1.5s-1r-3w"
    assert config.to_dict() == {
        "settle_seconds": 1.5,
        "settle_min_reads": 1,
        "stale_windows": 3,
        "obscured_energy": OBSCURED_ENERGY,
    }
    assert SettleConfig(obscured_energy=-12.0).slug == "settle0.3s-2r-2w-e-12"
    assert SettleConfig(obscured_energy=-12.25).slug == "settle0.3s-2r-2w-e-12.25"
    assert SettleConfig(obscured_energy=None).slug == "settle0.3s-2r-2w-eoff"


def test_a_read_is_obscured_only_above_the_threshold():
    config = SettleConfig(obscured_energy=-12.4)
    assert config.obscures(-12.0)
    assert not config.obscures(-12.4)  # strictly above, as the probe gated
    assert not config.obscures(-13.0)
    assert not config.obscures(None)  # a recognizer that reports no energy
    assert not SettleConfig(obscured_energy=None).obscures(0.0)  # the gate is off
