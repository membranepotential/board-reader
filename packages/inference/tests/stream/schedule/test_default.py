"""Tests for the shipped schedule (:mod:`board_reader_inference.stream.schedule.default`)."""

from __future__ import annotations

from board_reader_inference.stream.schedule.constants import EXIT_HOLD_S
from board_reader_inference.stream.schedule.default import DEFAULT_POLICY_SPEC
from board_reader_inference.stream.schedule.parse import parse_policy
from board_reader_inference.stream.schedule.policies import BurstTriggered


def test_the_shipped_default_is_the_policy_the_sweep_recommended():
    """``BR_SCHEDULE``'s default is a spec string like any other, so it is checked
    like one: it must name exactly the constants the round-2 latency sweep chose
    (three reads 0.3 s apart, with the forwarder's tail), the 2 s keepalive the
    2026-11-24 live take asked for, and the 0.3 s node hold of latency round 3 —
    shorter than the edge's mirrored ``EXIT_HOLD_S`` on purpose (see
    ``DEFAULT_POLICY_SPEC``)."""
    assert parse_policy(DEFAULT_POLICY_SPEC) == BurstTriggered(
        keepalive_s=2.0, burst_end_reads=3, spacing_s=0.3, exit_hold_s=0.3
    )
    assert EXIT_HOLD_S == 0.7, "the edge's hold, mirrored from motion.ts, is unchanged"
    # ...and it is canonical, so a manifest and the env var read the same.
    assert parse_policy(DEFAULT_POLICY_SPEC).name == DEFAULT_POLICY_SPEC
