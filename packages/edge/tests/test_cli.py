"""Tests for the ``board-reader-edge`` CLI's argument wiring.

Only the pure argument -> stage mapping is exercised here; nothing in this file
opens a socket or a file. Streaming itself is covered end-to-end elsewhere.
"""

from __future__ import annotations

import pytest

from board_reader_core.control import CropRect

from board_reader_edge import cli
from board_reader_edge.adapter import DEFAULT_JPEG_QUALITY
from board_reader_edge.cli import main


def _capture_adapter(monkeypatch) -> dict:
    """Run ``main`` without a gateway; the adapter it would stream lands in the dict."""
    seen: dict = {}

    async def capture(adapter, url):
        seen["adapter"] = adapter
        return 0

    monkeypatch.setattr(cli, "stream_to_gateway", capture)
    return seen


def test_roi_reaches_the_adapter(monkeypatch):
    seen = _capture_adapter(monkeypatch)
    main(["feed", "--synthetic", "1", "--roi", "0.1,0.2,0.5,0.6"])
    assert seen["adapter"].roi == CropRect(x=0.1, y=0.2, w=0.5, h=0.6)


def test_no_roi_sends_the_whole_frame(monkeypatch):
    seen = _capture_adapter(monkeypatch)
    main(["feed", "--synthetic", "1"])
    assert seen["adapter"].roi is None


@pytest.mark.parametrize(
    "spec",
    [
        "0.1,0.2,0.5",  # one number short
        "a,b,c,d",  # not numbers
        "0.6,0,0.5,1",  # runs off the right edge
        "0,0,0.01,1",  # a slip of the mouse, not a board
    ],
)
def test_a_bad_roi_is_refused_before_anything_is_dialled(spec, monkeypatch, capsys):
    """argparse refuses it, so no source is opened and no socket is dialled."""
    def refuse(*_args, **_kw):
        raise AssertionError("the gateway must never be dialled with a bad --roi")

    monkeypatch.setattr(cli, "stream_to_gateway", refuse)
    with pytest.raises(SystemExit):
        main(["feed", "--synthetic", "1", "--roi", spec])
    assert "--roi" in capsys.readouterr().err


def test_jpeg_quality_reaches_the_adapter(monkeypatch):
    seen = _capture_adapter(monkeypatch)
    cli.main(["feed", "--synthetic", "1", "--jpeg-quality", "60"])
    assert seen["adapter"].jpeg_quality == 60


def test_jpeg_quality_defaults_to_the_adapters_own_default(monkeypatch):
    """``--jpeg-quality`` parses to ``None`` when unset (that is how ``--recording``
    tells "asked for a re-encode" from "took the default"), so the CLI must still
    hand the adapter a real number."""
    seen = _capture_adapter(monkeypatch)
    cli.main(["feed", "--synthetic", "1"])
    assert seen["adapter"].jpeg_quality == DEFAULT_JPEG_QUALITY


@pytest.mark.parametrize("value", ["0", "101", "-5"])
def test_jpeg_quality_outside_1_100_is_refused(value):
    with pytest.raises(SystemExit):
        main(["feed", "--synthetic", "1", "--jpeg-quality", value])


@pytest.mark.parametrize(
    ("spec", "seconds"), [("90", 90.0), ("1:30", 90.0), ("1:02:00", 3720.0), ("7.5", 7.5)]
)
def test_camera_start_takes_seconds_or_clock_time(spec, seconds):
    assert cli._start(spec) == seconds


@pytest.mark.parametrize("spec", ["1:2:3:4", "-5", "ten"])
def test_camera_start_refuses_what_is_no_time(spec):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["camera", "x.mp4", "--start", spec])
