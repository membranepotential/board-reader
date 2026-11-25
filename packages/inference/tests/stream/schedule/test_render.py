"""Tests for a policy's canonical spec string and slug (:mod:`board_reader_inference.stream.schedule.render`)."""

from __future__ import annotations

from pathlib import Path

from board_reader_inference.stream.schedule.parse import parse_policy
from board_reader_inference.stream.schedule.policies import BurstTriggered, EveryFrame
from board_reader_inference.stream.schedule.render import policy_slug


def test_a_name_states_only_what_was_asked_for():
    assert BurstTriggered().name == "burst-end"
    assert BurstTriggered(keepalive_s=5.0).name == "burst-end:keepalive_s=5"
    assert EveryFrame().name == "every-frame"


def test_a_name_survives_the_values_display_formatting_would_have_mangled():
    """``%g`` is a *display* format and this is an identity.

    It rendered a large int in exponent form — which the int coercion in
    ``parse_policy`` then refused outright, so the policy's own name no longer
    rebuilt it — and it rounded floats to six significant digits, collapsing two
    genuinely different policies onto one name and therefore one replay directory.
    """
    big = BurstTriggered(burst_end_reads=1_000_000)
    assert big.name == "burst-end:burst_end_reads=1000000"
    assert parse_policy(big.name) == big

    fine = BurstTriggered(spacing_s=1.2500001)
    assert parse_policy(fine.name) == fine
    assert fine.name != BurstTriggered(spacing_s=1.25).name
    assert policy_slug(fine) != policy_slug(BurstTriggered(spacing_s=1.25))


def test_a_slug_is_a_readable_path_component():
    slug = policy_slug(BurstTriggered(keepalive_s=5.0, burst_end_reads=3, spacing_s=2.0))
    assert slug == "burst-end-keepalive_s5-burst_end_reads3-spacing_s2"
    assert Path(slug).name == slug
    # Two policies that differ must not share a replay directory.
    assert policy_slug(BurstTriggered(keepalive_s=5.0)) != policy_slug(BurstTriggered())
