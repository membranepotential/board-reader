"""Tests for the estimation scheduler (:mod:`board_reader_inference.schedule`).

The claims, in the order they matter:

* the **keepalive is a floor** — a policy fed a permanently-broken motion meter
  still reads, at the keepalive rate, forever (DESIGN D3, and the dead-meter
  incident these policies were written after);
* burst detection uses the **edge's own hysteresis**, so one move is one burst
  and the pause between two strides does not split it;
* a burst end can be worth **several reads, spaced**, because the debounce needs a
  quorum spanning ``SETTLE_SECONDS`` and one read per move can never move it;
* decisions come from **capture time only** — no wall clock anywhere, or the
  replay harness's determinism dies with it;
* the ported motion meter is the **edge's meter**, value for value.

Every policy here is exercised on a scripted stability sequence rather than on
pixels: a scheduler's whole input is the header, so that is the honest unit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from board_reader_core.recording import FrameEntry

from board_reader_inference import schedule as schedule_mod
from board_reader_inference.schedule import (
    DEFAULT_KEEPALIVE_S,
    ENTER_BURST,
    EXIT_BURST,
    EXIT_HOLD_S,
    POLICIES,
    RESTAMPED_KEY,
    BurstTriggered,
    EveryFrame,
    FramePolicy,
    MotionMeter,
    WhenStill,
    parse_policy,
    policy_slug,
    restamped,
)

#: The old takes' cadence: a flat ~8.3 fps, whatever the board was doing.
FLAT_DT = 0.12
#: Take 5's cadence: the adaptive forwarder's 1 fps at rest.
REST_DT = 1.0


def entries(stabilities, *, dt: float = FLAT_DT, first_ts: float = 1_000.0):
    """One header per stability, ``dt`` recorded seconds apart."""
    return [
        FrameEntry(
            header={
                "board_id": "phone",
                "seq": i,
                "capture_ts": first_ts + i * dt,
                "stability": s,
            },
            jpeg_rel=f"jpeg/0000/{i:06d}.jpg",
            received_ts=first_ts + i * dt,
        )
        for i, s in enumerate(stabilities)
    ]


def read_seqs(policy: FramePolicy, index) -> list[int]:
    """The seqs a policy spends an estimate on, in order."""
    return [e.seq for e in index if policy.should_estimate(e)]


def read_times(policy: FramePolicy, index) -> list[float]:
    return [e.capture_ts for e in index if policy.should_estimate(e)]


# --------------------------------------------------------------------------- #
# The floor (DESIGN D3)
# --------------------------------------------------------------------------- #


def test_a_dead_meter_degrades_to_slow_reads_never_to_none():
    """The cautionary tale, as a test.

    Before the 2026-08-27 fix the meter read under 0.35 all day on the deployment
    phone: permanently "moving", never once at rest. A burst-end scheduler on that
    stream never sees a burst *end* — so if the keepalive were conditional on
    anything the meter says, the node would have read the board exactly zero times
    for a whole session. It is not conditional on anything.
    """
    index = entries([0.2] * 500)  # 60 s of a stuck-low meter at 8.3 fps
    policy = BurstTriggered(keepalive_s=3.0)
    times = read_times(policy, index)

    assert policy.bursting  # it never believes the move ended, and reads anyway
    span = index[-1].capture_ts - index[0].capture_ts
    assert len(times) == pytest.approx(span / 3.0, abs=1.5)
    gaps = np.diff(times)
    assert gaps.max() <= 3.0 + FLAT_DT + 1e-9  # never a gap longer than the floor


def test_a_meter_stuck_at_rest_also_only_costs_read_rate():
    """The other failure direction: a meter that never dips reports no bursts.

    There is then nothing to trigger on and the policy falls back to the floor —
    a slow reader, not a blind one.
    """
    index = entries([1.0] * 500)
    policy = BurstTriggered(keepalive_s=5.0)
    times = read_times(policy, index)

    assert not policy.bursting
    assert np.diff(times).max() <= 5.0 + FLAT_DT + 1e-9
    assert len(times) >= 10


def test_the_keepalive_floor_cannot_be_switched_off():
    """There is no "0 = never" spelling: a policy without a floor is the incident."""
    with pytest.raises(ValueError, match="keepalive_s must be > 0"):
        BurstTriggered(keepalive_s=0.0)
    with pytest.raises(ValueError, match="keepalive_s must be > 0"):
        WhenStill(keepalive_s=-1.0)


def test_the_first_frame_of_a_stream_is_always_read():
    """A reader that waits out a keepalive before its first read shows nothing for
    three seconds after connecting, which is the one moment a viewer is watching."""
    assert read_seqs(BurstTriggered(), entries([1.0] * 3))[0] == 0
    assert read_seqs(WhenStill(), entries([0.1] * 3))[0] == 0


# --------------------------------------------------------------------------- #
# Burst detection
# --------------------------------------------------------------------------- #


def _one_move(rest_frames: int = 40, move_frames: int = 20) -> list[float]:
    """Rest, one move (a dip well under ENTER_BURST), rest again."""
    return [0.97] * rest_frames + [0.3] * move_frames + [0.97] * rest_frames


def test_a_move_is_read_when_it_finishes_not_while_it_happens():
    index = entries(_one_move())
    policy = BurstTriggered(keepalive_s=30.0)  # floor pushed away: burst ends only
    seqs = read_seqs(policy, index)

    move = range(40, 60)
    assert not any(s in move for s in seqs[1:]), "no read inside the burst"
    # The burst ends EXIT_HOLD_S of calm after the last moving frame, and the very
    # next frame is read: that is the move-complete moment.
    end = 60 + round(EXIT_HOLD_S / FLAT_DT)
    assert next(s for s in seqs if s >= 60) == pytest.approx(end, abs=1)


def test_the_pause_between_two_strides_does_not_split_one_move():
    """Hysteresis, mirrored from the forwarder: a brief calm inside a move is not
    a move-complete moment, or the reader estimates a piece in mid-air."""
    stabilities = (
        [0.97] * 20
        + [0.3] * 8
        + [0.95] * 4  # a 0.48 s pause — shorter than EXIT_HOLD_S
        + [0.3] * 8
        + [0.97] * 30
    )
    policy = BurstTriggered(keepalive_s=30.0)
    seqs = read_seqs(policy, entries(stabilities))
    assert len([s for s in seqs if s > 0]) == 1, "one move, one burst, one read"


def test_the_hysteresis_band_is_the_forwarders_own():
    """A dip into the band (between exit and enter) must not start a burst, and a
    recovery into it must not end one — otherwise the policy segments a stream
    into bursts the edge that sent it never had."""
    mid = (ENTER_BURST + EXIT_BURST) / 2
    policy = BurstTriggered(keepalive_s=30.0)
    assert read_seqs(policy, entries([1.0] * 5 + [mid] * 30)) == [0]
    assert not policy.bursting

    resting = BurstTriggered(keepalive_s=30.0)
    index = entries([1.0] * 3 + [0.2] * 5 + [mid] * 40)
    assert read_seqs(resting, index) == [0]
    assert resting.bursting, "recovery only into the band never ends the burst"


# --------------------------------------------------------------------------- #
# Feeding the debounce its quorum
# --------------------------------------------------------------------------- #


def test_a_burst_end_can_buy_several_reads_a_beat_apart():
    """``SETTLE_MIN_READS`` reads spanning ``SETTLE_SECONDS`` is what the
    stabilizer wants; three reads 1.3 s apart span 2.6 s and deliver it."""
    index = entries(_one_move(rest_frames=60), dt=FLAT_DT)
    policy = BurstTriggered(keepalive_s=30.0, burst_end_reads=3, spacing_s=1.3)
    times = [t for t in read_times(policy, index) if t > index[60].capture_ts]

    assert len(times) == 3
    gaps = np.diff(times)
    assert gaps.min() >= 1.3 - FLAT_DT
    assert times[-1] - times[0] >= 2.5, "the run spans a settle window"


def test_while_a_burst_end_is_owed_reads_the_spacing_governs_not_the_floor():
    """The two intervals are one knob at a time: while a burst end still owes
    reads they come at ``spacing_s``, and only afterwards does the floor take
    over. A keepalive that lands inside the run satisfies one of the owed reads
    rather than being spent twice on the same board."""
    index = entries(_one_move(rest_frames=60))
    policy = BurstTriggered(keepalive_s=1.0, burst_end_reads=3, spacing_s=1.3)
    ended = index[80].capture_ts + EXIT_HOLD_S
    owed = [t for t in read_times(policy, index) if t >= ended][:3]

    assert len(owed) == 3
    assert np.diff(owed).min() >= 1.3 - FLAT_DT, "the tighter floor did not win"


def test_one_read_per_move_is_the_default_and_stays_sparse():
    """26 recorded seconds and one move, on a floor pushed out of the way: the
    default spends two estimates, the opening board and the move-complete one."""
    index = entries(_one_move(rest_frames=100))
    assert len(read_seqs(BurstTriggered(keepalive_s=30.0), index)) == 2


# --------------------------------------------------------------------------- #
# The rest-cadence take
# --------------------------------------------------------------------------- #


def test_a_one_fps_rest_cadence_still_gets_its_burst_end_and_its_floor():
    """Take 5 arrives at ~1 fps at rest, not 8.3. A policy schedules whatever
    arrives, so the burst end lands on the first frame that shows it — up to a
    frame period later — and the floor is still honoured to within one period."""
    index = entries([0.97] * 10 + [0.3] * 6 + [0.97] * 20, dt=REST_DT)
    policy = BurstTriggered(keepalive_s=3.0, burst_end_reads=2, spacing_s=1.3)
    times = read_times(policy, index)
    assert np.diff(times).max() <= 3.0 + REST_DT + 1e-9
    after = [t for t in times if t > index[15].capture_ts]
    assert len(after) >= 2


# --------------------------------------------------------------------------- #
# when-still
# --------------------------------------------------------------------------- #


def test_when_still_reads_calm_frames_at_its_spacing_and_moving_ones_at_the_floor():
    index = entries([0.97] * 40 + [0.3] * 40, dt=FLAT_DT)
    policy = WhenStill(theta=0.9, spacing_s=1.0, keepalive_s=3.0)
    calm = [t for t in read_times(policy, index) if t < index[40].capture_ts]
    moving = [t for t in read_times(WhenStill(theta=0.9, spacing_s=1.0), index)
              if t >= index[40].capture_ts]

    assert np.diff(calm).min() >= 1.0 - FLAT_DT
    assert len(calm) == pytest.approx(40 * FLAT_DT / 1.0, abs=1)
    assert len(moving) <= 40 * FLAT_DT / 3.0 + 1


# --------------------------------------------------------------------------- #
# Determinism and the clock
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "policy_factory",
    [EveryFrame, BurstTriggered, lambda: BurstTriggered(burst_end_reads=3), WhenStill],
)
def test_two_runs_of_one_sequence_schedule_the_same_frames(policy_factory):
    index = entries(_one_move() + _one_move() + [0.4, 0.9, 0.2, 1.0] * 10)
    assert read_seqs(policy_factory(), index) == read_seqs(policy_factory(), index)


def test_no_policy_reads_a_wall_clock():
    """Structural, because the failure is invisible: a replay whose scheduler
    consults ``time.time()`` picks different frames on every run and takes the
    harness's byte-identical guarantee with it. The module may not import a clock
    at all.

    (The AST is the check rather than a grep for the same reason
    ``tests/test_layer_separation.py`` uses one: prose in a docstring is not an
    import.)
    """
    tree = ast.parse(Path(schedule_mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"time", "datetime", "random"}


# --------------------------------------------------------------------------- #
# Spec strings
# --------------------------------------------------------------------------- #


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


def test_a_name_states_only_what_was_asked_for():
    assert BurstTriggered().name == "burst-end"
    assert BurstTriggered(keepalive_s=5.0).name == "burst-end:keepalive_s=5"
    assert EveryFrame().name == "every-frame"


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


def test_a_slug_is_a_readable_path_component():
    slug = policy_slug(BurstTriggered(keepalive_s=5.0, burst_end_reads=3, spacing_s=2.0))
    assert slug == "burst-end-keepalive_s5-burst_end_reads3-spacing_s2"
    assert Path(slug).name == slug
    # Two policies that differ must not share a replay directory.
    assert policy_slug(BurstTriggered(keepalive_s=5.0)) != policy_slug(BurstTriggered())


# --------------------------------------------------------------------------- #
# The ported meter
# --------------------------------------------------------------------------- #


def _frames(n: int, *, moving: bool = False, seed: int = 0) -> list[np.ndarray]:
    """A static scene with sensor noise, optionally with something crossing it."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        image = scene.astype(np.int16) + rng.integers(-2, 3, scene.shape)
        if moving:
            image[:, (i * 4) % 40 : (i * 4) % 40 + 8] = 0
        out.append(np.clip(image, 0, 255).astype(np.uint8))
    return out


def test_the_meter_reports_rest_on_the_first_frame_and_finds_the_floor():
    meter = MotionMeter()
    scores = [meter.update(f) for f in _frames(30)]
    assert scores[0] == 1.0
    assert scores[-1] > EXIT_BURST, "a still scene settles above the burst-exit line"


def test_the_meter_drops_on_motion_and_recovers():
    meter = MotionMeter()
    for f in _frames(30):
        meter.update(f)
    for f in _frames(10, moving=True, seed=1):
        meter.update(f)
    assert meter.stability < ENTER_BURST
    for f in _frames(30, seed=2):
        meter.update(f)
    assert meter.stability > EXIT_BURST


def test_the_ported_meter_matches_the_edges_value_for_value():
    """The mirror, pinned.

    :class:`MotionMeter` is a copy of ``board_reader_edge.motion.MotionMeter``
    because DESIGN §4 forbids this package from importing the edge's (see
    ``tests/test_layer_separation.py``, which scans ``src`` only — a test may
    import both, and this is exactly what that is for). A copy with no equality
    check is a fork waiting to happen, so: same frames, same numbers.
    """
    edge_motion = pytest.importorskip("board_reader_edge.motion")
    ours, theirs = MotionMeter(), edge_motion.MotionMeter()
    frames = _frames(20) + _frames(10, moving=True, seed=3) + _frames(20, seed=4)
    assert [ours.update(f) for f in frames] == [theirs.update(f) for f in frames]


# --------------------------------------------------------------------------- #
# Restamping
# --------------------------------------------------------------------------- #


def test_restamping_overrides_the_recorded_value_and_says_so():
    [entry] = entries([0.31])
    meter = MotionMeter()
    out = restamped(entry, meter, _frames(1)[0])

    assert out.header["stability"] == 1.0  # the meter's first-frame invariant
    assert out.header[RESTAMPED_KEY] is True
    assert out.seq == entry.seq and out.capture_ts == entry.capture_ts
    assert entry.header["stability"] == 0.31, "the recorded entry is not mutated"


def test_restamping_a_dead_meters_take_is_what_makes_it_schedulable():
    """The data caveat, end to end: the recorded field says "moving" on every
    frame of a still scene, and a policy reading it can only ever hit its floor.
    Re-metered, the same frames read as rest."""
    frames = _frames(40)
    index = entries([0.29] * len(frames))  # what the old meter actually wrote
    stale = BurstTriggered(keepalive_s=DEFAULT_KEEPALIVE_S)
    read_seqs(stale, index)
    assert stale.bursting, "a still scene, reported as one long move"

    meter = MotionMeter()
    fresh = [restamped(e, meter, f) for e, f in zip(index, frames, strict=True)]
    live = BurstTriggered(keepalive_s=DEFAULT_KEEPALIVE_S)
    read_seqs(live, fresh)

    assert all(e.header["stability"] > EXIT_BURST for e in fresh[5:])
    assert not live.bursting, "re-metered, the same frames read as rest"
