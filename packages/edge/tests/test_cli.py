"""Tests for the ``board-reader-edge`` CLI's argument wiring.

Only the pure argument -> stage mapping is exercised here; nothing in this file
opens a socket or a file. Streaming itself is covered end-to-end elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

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
