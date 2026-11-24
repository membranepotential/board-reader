"""Tests for ``GET /uploads`` and ``GET /uploads/{id}/photo``.

The snapshot page's "Recent positions" list was session-only (lost on reload);
these endpoints make it server-backed off the existing opt-in retention
(``BR_SAVE_UPLOADS``) that ``/estimate``, ``/calibrate``, and ``/labels``
already write to. Follows the style of ``test_server_labels.py``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg

from board_reader_inference.hub import PositionHub
from board_reader_inference.recognizer import StubRecognizer
from board_reader_inference.server import create_app

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"


@pytest.fixture
def retaining(tmp_path, monkeypatch):
    """A gateway with photo retention on, sandboxed to ``tmp_path``."""
    monkeypatch.setenv("BR_SAVE_UPLOADS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def _estimate(client: TestClient) -> dict:
    r = client.post(
        "/estimate",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    return r.json()


def test_uploads_report_retention_off(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    r = client.get("/uploads")
    assert r.status_code == 200
    assert r.json() == {"retention": False, "uploads": []}


def test_uploads_lists_newest_first_with_labeled_and_unlabeled_entries(retaining):
    first = _estimate(retaining)
    second = _estimate(retaining)
    retaining.post(
        f"/labels/{second['upload']}", json={"placement": MIDGAME, "corrections": 2}
    )

    body = retaining.get("/uploads").json()
    assert body["retention"] is True
    ids = [u["id"] for u in body["uploads"]]
    assert ids == [second["upload"], first["upload"]]  # newest first

    labeled = body["uploads"][0]
    assert labeled["labeled"] is True
    assert labeled["label_placement"] == MIDGAME
    assert labeled["label_corrections"] == 2
    assert labeled["kind"] == "estimate"
    assert "corners" not in labeled and "autorotated" not in labeled
    assert labeled["placement"] == START_PLACEMENT  # what the server predicted

    unlabeled = body["uploads"][1]
    assert unlabeled["labeled"] is False
    assert unlabeled["label_placement"] is None
    assert unlabeled["label_corrections"] is None


def test_uploads_limit_is_clamped_and_applied(retaining):
    for _ in range(3):
        _estimate(retaining)
    assert len(retaining.get("/uploads?limit=2").json()["uploads"]) == 2
    # A limit outside [1, 500] is clamped, not rejected.
    assert len(retaining.get("/uploads?limit=0").json()["uploads"]) == 1
    assert len(retaining.get("/uploads?limit=10000").json()["uploads"]) == 3


def test_uploads_skips_a_malformed_sidecar(retaining, tmp_path):
    good = _estimate(retaining)["upload"]
    (tmp_path / "20200101-000000-0000-estimate.json").write_text("not json")
    (tmp_path / "20200101-000000-0000-estimate.jpg").write_bytes(b"x")
    ids = [u["id"] for u in retaining.get("/uploads").json()["uploads"]]
    assert ids == [good]


def test_uploads_skips_an_entry_whose_photo_is_missing(retaining, tmp_path):
    good = _estimate(retaining)["upload"]
    (tmp_path / "20200101-000000-0000-estimate.json").write_text(
        json.dumps({"kind": "estimate", "ts": 1.0})
    )
    ids = [u["id"] for u in retaining.get("/uploads").json()["uploads"]]
    assert ids == [good]


def test_photo_retention_off_is_422(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    r = client.get("/uploads/20260806-213500-0001-estimate/photo")
    assert r.status_code == 422


def test_photo_bad_id_is_400(retaining):
    r = retaining.get("/uploads/not-an-id/photo")
    assert r.status_code == 400


def test_photo_unknown_id_is_404(retaining):
    r = retaining.get("/uploads/20200101-000000-0000-estimate/photo")
    assert r.status_code == 404


def test_photo_serves_the_stored_bytes_verbatim(retaining):
    body = encode_jpeg(render_board())
    r = retaining.post(
        "/estimate",
        content=body,
        headers={"Content-Type": "image/jpeg"},
    )
    upload = r.json()["upload"]
    got = retaining.get(f"/uploads/{upload}/photo")
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/jpeg"
    assert got.headers["cache-control"] == "public, max-age=3600"
    assert got.content == body


def test_photo_max_downscales_to_a_strictly_smaller_image(retaining):
    upload = _estimate(retaining)["upload"]
    original = retaining.get(f"/uploads/{upload}/photo").content
    thumb = retaining.get(f"/uploads/{upload}/photo?max=32").content
    assert len(thumb) < len(original)
    arr = np.frombuffer(thumb, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 32
