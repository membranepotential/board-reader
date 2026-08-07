"""Tests for the ``board-reader-edge`` CLI's argument wiring.

Only the pure argument -> stage mapping is exercised here; nothing in this file
opens a socket or a file. Streaming itself is covered end-to-end elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_edge.adapter import DEFAULT_JPEG_QUALITY
from board_reader_edge.cli import (
    _build_rectifier,
    _build_wire_corners,
    build_parser,
    main,
)

CORNERS = "300,20,300,300,20,300,20,20"


def _feed_args(*argv: str):
    return build_parser().parse_args(["feed", *argv])


def test_forward_original_sends_the_corners_to_the_wire_not_a_rectifier():
    args = _feed_args("--forward-original", "--corners", CORNERS)
    assert _build_rectifier(args) is None  # nothing is warped at the edge
    quad = _build_wire_corners(args)
    # Parsed verbatim in the operator's semantic a8,h8,h1,a1 order.
    assert np.array_equal(
        quad, np.array([[300, 20], [300, 300], [20, 300], [20, 20]], dtype="float32")
    )


def test_cmd_feed_wires_the_corners_onto_the_adapter(monkeypatch):
    """The one line that enables the mode: the parsed quad must reach the adapter.

    Everything else in this file exercises the helpers in isolation; without this
    test, dropping the ``wire_corners=`` kwarg in ``cmd_feed`` stays green while
    the edge silently forwards corner-less full frames.
    """
    import board_reader_edge.cli as cli

    seen = {}

    async def capture(adapter, url):
        seen["adapter"] = adapter
        return 0

    monkeypatch.setattr(cli, "stream_to_gateway", capture)
    cli.main(["feed", "--forward-original", "--corners", CORNERS, "--synthetic", "1"])
    adapter = seen["adapter"]
    assert adapter.rectifier is None
    assert np.array_equal(
        adapter.wire_corners,
        np.array([[300, 20], [300, 300], [20, 300], [20, 20]], dtype="float32"),
    )


def test_corners_without_forward_original_still_builds_a_manual_rectifier():
    args = _feed_args("--corners", CORNERS)
    assert _build_wire_corners(args) is None
    rect = _build_rectifier(args)
    assert rect is not None and rect.locked


def test_forward_original_requires_corners():
    with pytest.raises(SystemExit, match="--corners"):
        main(["feed", "--forward-original"])


def test_forward_original_rejects_rectify():
    with pytest.raises(SystemExit, match="exclusive"):
        main(["feed", "--forward-original", "--rectify", "--corners", CORNERS])


def _refuse_to_stream(*_args, **_kw):
    raise AssertionError("the gateway must never be dialled with a bad quad")


async def _stream_nothing(*_args, **_kw) -> int:
    return 0


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("300,20,300,300,20,300", "8 numbers"),          # one corner short
        ("300,20,300,300,20,300,300,20", "repeats"),      # same point twice
        ("300,20,300,300,20,20,20,300", "crosses itself"),  # last two swapped
        ("0,0,20,0,20,20,0,20", "too small"),             # a 20 px "board"
        ("0,0,900,0,900,5,0,5", "sliver"),                # near-collinear
        ("0,0,900,0,450,300,0,900", "not convex"),        # one coordinate mistyped
    ],
)
def test_a_bad_quad_is_refused_before_anything_is_dialled(spec, expected, monkeypatch):
    """The whole point of the check: a typo'd quad never becomes a run.

    ``stream_to_gateway`` is replaced with a landmine — the failure this guards
    against is hours of confidently-wrong geometry on the wire, so the refusal has
    to land before any connection, not on the first frame.
    """
    import board_reader_edge.cli as cli

    monkeypatch.setattr(cli, "stream_to_gateway", _refuse_to_stream)
    with pytest.raises(SystemExit, match=expected):
        cli.main(["feed", "--forward-original", "--corners", spec, "--synthetic", "1"])


def test_a_bad_quad_is_refused_on_the_rectify_at_edge_path_too(monkeypatch):
    """--corners means something else without --forward-original — but a quad that
    cannot be a board is just as wrong when the edge is the one warping with it."""
    import board_reader_edge.cli as cli

    monkeypatch.setattr(cli, "stream_to_gateway", _refuse_to_stream)
    with pytest.raises(SystemExit, match="crosses itself"):
        cli.main(["feed", "--corners", "300,20,300,300,20,20,20,300"])


def test_the_quad_is_checked_before_the_source_is_even_opened(monkeypatch):
    """Ordering, pinned: a bad quad costs an error line, not an opened clip.

    ``--video`` points at a file that does not exist, so whichever check runs first
    is the one that speaks — and it must be the corners.
    """
    import board_reader_edge.cli as cli

    monkeypatch.setattr(cli, "stream_to_gateway", _refuse_to_stream)
    with pytest.raises(SystemExit, match="crosses itself"):
        cli.main(["feed", "--video", "/nope/missing.mp4",
                  "--corners", "300,20,300,300,20,20,20,300"])


def test_a_near_board_level_quad_warns_but_still_runs(monkeypatch, capsys):
    """Warn, do not block — the gateway's convention for the same measurement."""
    import board_reader_edge.cli as cli

    monkeypatch.setattr(cli, "stream_to_gateway", _stream_nothing)
    # Aspect ~0.3 on the 384px synthetic board: a camera nearly at board level.
    cli.main(
        ["feed", "--forward-original", "--synthetic", "1",
         "--corners", "100,100,300,100,330,160,70,160"]
    )
    assert "aspect" in capsys.readouterr().err


def test_jpeg_quality_reaches_the_adapter(monkeypatch):
    import board_reader_edge.cli as cli

    seen = {}

    async def capture(adapter, url):
        seen["adapter"] = adapter
        return 0

    monkeypatch.setattr(cli, "stream_to_gateway", capture)
    cli.main(["feed", "--synthetic", "1", "--jpeg-quality", "60"])
    assert seen["adapter"].jpeg_quality == 60


def test_jpeg_quality_defaults_to_the_adapters_own_default(monkeypatch):
    """``--jpeg-quality`` parses to ``None`` when unset (that is how ``--recording``
    tells "asked for a re-encode" from "took the default"), so the CLI must still
    hand the adapter a real number."""
    import board_reader_edge.cli as cli

    seen = {}

    async def capture(adapter, url):
        seen["adapter"] = adapter
        return 0

    monkeypatch.setattr(cli, "stream_to_gateway", capture)
    cli.main(["feed", "--synthetic", "1"])
    assert seen["adapter"].jpeg_quality == DEFAULT_JPEG_QUALITY


@pytest.mark.parametrize("value", ["0", "101", "-5"])
def test_jpeg_quality_outside_1_100_is_refused(value):
    with pytest.raises(SystemExit):
        main(["feed", "--synthetic", "1", "--jpeg-quality", value])
