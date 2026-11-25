"""The health tick's frame rate."""

from __future__ import annotations

from board_reader_inference.control.camera.health import StreamSample, frame_rate


def test_the_frame_rate_is_measured_between_two_ticks():
    assert frame_rate(StreamSample(10, 100.0), StreamSample(25, 101.5)) == 10.0
    assert frame_rate(None, StreamSample(25, 101.5)) is None
    # The phone reloaded and its seq began again: nothing to measure.
    assert frame_rate(StreamSample(10, 100.0), StreamSample(2, 101.0)) is None
