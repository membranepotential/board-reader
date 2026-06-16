"""Tests for the viewer static server."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from board_reader_viewer.server import create_app


def test_healthz():
    assert TestClient(create_app()).get("/healthz").json() == {"status": "ok"}


def test_index_served_and_ws_url_injected(monkeypatch):
    monkeypatch.setenv("BR_INFERENCE_WS", "ws://example.test:9000/viewer")
    # Re-import not needed: the env is read per-request inside the route.
    html = TestClient(create_app()).get("/").text
    assert "ws://example.test:9000/viewer" in html
    assert "__BR_INFERENCE_WS__" not in html  # placeholder fully substituted


def test_index_default_ws_url(monkeypatch):
    monkeypatch.delenv("BR_INFERENCE_WS", raising=False)
    html = TestClient(create_app()).get("/").text
    assert "ws://localhost:8000/viewer" in html


def test_viewer_does_not_depend_on_node_packages():
    """The viewer is a pure static server — it must not import core/edge/inference."""
    import board_reader_viewer.server as srv

    text = Path(srv.__file__).read_text(encoding="utf-8")
    for forbidden in ("board_reader_core", "board_reader_edge", "board_reader_inference"):
        assert forbidden not in text
