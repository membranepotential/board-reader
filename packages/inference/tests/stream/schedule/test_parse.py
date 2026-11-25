"""Tests for the policy spec grammar (:mod:`board_reader_inference.stream.schedule.parse`)."""

from __future__ import annotations

import pytest

from board_reader_inference.stream.schedule.parse import POLICIES, parse_policy
from board_reader_inference.stream.schedule.policies import (
    BurstTriggered,
    EveryFrame,
    WhenStill,
)
from board_reader_inference.stream.schedule.seam import FramePolicy


@pytest.mark.parametrize(
    "policy",
    [
        EveryFrame(),
        BurstTriggered(),
        BurstTriggered(keepalive_s=5.0),
        BurstTriggered(keepalive_s=3.0, burst_end_reads=3, spacing_s=1.3),
        BurstTriggered(enter=0.6, exit=0.9, exit_hold_s=1.0),
        WhenStill(theta=0.9, spacing_s=1.0),
    ],
)
def test_a_policys_name_is_the_argument_that_rebuilds_it(policy):
    """The manifest's ``policy`` field is the ``--policy`` string, not a label —
    which is the difference between a sweep you can rerun and one you cannot."""
    assert parse_policy(policy.name) == policy


def test_an_unknown_policy_or_parameter_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown policy 'burst'"):
        parse_policy("burst")
    with pytest.raises(ValueError, match="no parameter 'keepalive'"):
        parse_policy("burst-end:keepalive=3")
    with pytest.raises(ValueError, match="no parameter"):
        parse_policy("every-frame:keepalive_s=3")
    with pytest.raises(ValueError, match="not a float"):
        parse_policy("burst-end:keepalive_s=soon")
    with pytest.raises(ValueError, match="no parameter 'keepalive_s'"):
        parse_policy("burst-end:keepalive_s")


def test_every_policy_satisfies_the_seam():
    for cls in POLICIES.values():
        assert isinstance(cls(), FramePolicy)
