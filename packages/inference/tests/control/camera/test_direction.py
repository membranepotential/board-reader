"""Which way each board lies: the median of the latest reads, the nearest side, a moved camera."""

from __future__ import annotations

from board_reader_core.control import BoardDirection, CameraConfig, CropRect

from board_reader_inference.control.camera.direction import DIRECTION_READS, white_side
from board_reader_inference.control.camera.registry import CameraRegistry


def _read(registry: CameraRegistry, degrees: list[float], start: float = 0.0) -> float:
    """Feed ``degrees`` as live reads 0.5 s apart; the time after the last one."""
    for i, angle in enumerate(degrees):
        registry.observe_direction("phone", angle, start + 0.5 * i)
    return start + 0.5 * len(degrees)


def test_white_sits_on_the_nearest_side():
    assert [white_side(a) for a in (2.0, 89.0, 91.0, 179.0, -179.0, -88.0, 271.0)] == [
        "bottom", "left", "left", "top", "top", "right", "right",
    ]


def test_the_direction_follows_the_reads_and_ignores_an_odd_one():
    registry = CameraRegistry()
    assert registry.direction("phone") is None
    _read(registry, [88.0, 90.0, 89.0, 200.0, 91.0])
    assert registry.direction("phone") == BoardDirection(90.0, "left", camera_moved=False)


def test_a_camera_turned_after_settling_is_called_moved_until_reframed():
    registry = CameraRegistry()
    now = _read(registry, [89.0] * DIRECTION_READS)
    # Turned by a quarter: once most of the latest reads say so, the camera moved.
    _read(registry, [1.0, -1.0, 0.0], start=now)
    assert registry.direction("phone") == BoardDirection(1.0, "bottom", camera_moved=True)

    # A new crop is a new framing: the angle settles again from here.
    registry.configure("phone", CameraConfig(crop=CropRect(0.1, 0.1, 0.8, 0.8)))
    assert registry.direction("phone") is None
    _read(registry, [0.0] * DIRECTION_READS, start=10.0)
    assert registry.direction("phone") == BoardDirection(0.0, "bottom", camera_moved=False)


def test_streaming_on_and_off_keeps_the_settled_angle():
    registry = CameraRegistry()
    now = _read(registry, [89.0] * DIRECTION_READS)
    registry.configure("phone", CameraConfig(streaming=True))
    _read(registry, [179.0] * 3, start=now)
    view = registry.direction("phone")
    assert view is not None and view.camera_moved


def test_a_burst_of_reads_of_one_scene_does_not_settle_the_angle():
    registry = CameraRegistry()
    for i in range(DIRECTION_READS):
        registry.observe_direction("phone", 89.0, 0.1 * i)  # all within half a second
    registry.observe_direction("phone", 0.0, 0.6)
    registry.observe_direction("phone", 0.0, 0.7)
    registry.observe_direction("phone", 0.0, 0.8)
    view = registry.direction("phone")
    assert view is not None and view.white_side == "bottom" and not view.camera_moved
