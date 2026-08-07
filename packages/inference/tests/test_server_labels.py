"""Tests for ``POST /labels/{upload}`` — keeping the operator's corrections.

The snapshot page's correction panel is the only place a human states what a
board really holds. These tests pin the contract that turns those statements into
eval data: they attach to a retained photo, they are stored in the corner frame
rather than the model's guessed orientation, and a hostile upload id can never
walk out of the retention directory.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.labels import LABEL_SUFFIX, rotate_placement
from board_reader_inference.server import create_app

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"


@pytest.fixture
def retaining(tmp_path, monkeypatch):
    """A gateway with photo retention on, sandboxed to ``tmp_path``."""
    monkeypatch.setenv("BR_SAVE_UPLOADS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))


def _estimate(client: TestClient, corners: str | None = "0,0,80,0,80,80,0,80") -> dict:
    query = f"?corners={corners}" if corners else ""
    r = client.post(
        f"/estimate{query}",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    return r.json()


def test_estimate_names_the_upload_so_a_read_can_be_labeled(retaining):
    body = _estimate(retaining)
    assert body["upload"].endswith("-estimate")


def test_estimate_omits_the_upload_id_when_retention_is_off(tmp_path, monkeypatch):
    """No photo means a label would have nothing to belong to — so the snapshot
    page is never offered the affordance in the first place."""
    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    assert "upload" not in _estimate(client)


def test_confirming_a_read_writes_a_label_beside_the_photo(retaining, tmp_path):
    upload = _estimate(retaining)["upload"]
    r = retaining.post(
        f"/labels/{upload}", json={"placement": MIDGAME, "corrections": 3}
    )
    assert r.status_code == 200
    assert r.json()["placement"] == MIDGAME and r.json()["corrections"] == 3

    label = tmp_path / f"{upload}{LABEL_SUFFIX}"
    assert label.exists()
    meta = json.loads(label.read_text())
    assert meta["kind"] == "label"
    assert meta["placement"] == MIDGAME
    assert meta["photo"] == f"{upload}.jpg"
    assert (tmp_path / meta["photo"]).exists()
    # The server's own read is kept next to the truth, so a label records both
    # what was claimed and what was predicted at the time.
    assert meta["predicted"] == START_PLACEMENT


def test_a_label_is_stored_in_the_corner_frame_not_the_rotated_one(
    retaining, tmp_path
):
    """An autorotated read is confirmed as displayed; truth on disk undoes the
    turn, so it never depends on a model output."""
    upload = _estimate(retaining)["upload"]
    sidecar = tmp_path / f"{upload}.json"
    meta = json.loads(sidecar.read_text())
    meta["autorotated"] = 180
    sidecar.write_text(json.dumps(meta))

    shown = rotate_placement(MIDGAME, 2)
    r = retaining.post(f"/labels/{upload}", json={"placement": shown})
    assert r.status_code == 200
    stored = json.loads((tmp_path / f"{upload}{LABEL_SUFFIX}").read_text())
    assert stored["placement"] == MIDGAME
    assert stored["placement_shown"] == shown


def test_relabelling_overwrites_so_the_last_word_wins(retaining, tmp_path):
    upload = _estimate(retaining)["upload"]
    retaining.post(f"/labels/{upload}", json={"placement": START_PLACEMENT})
    retaining.post(f"/labels/{upload}", json={"placement": MIDGAME, "corrections": 1})
    assert len(list(tmp_path.glob(f"*{LABEL_SUFFIX}"))) == 1
    stored = json.loads((tmp_path / f"{upload}{LABEL_SUFFIX}").read_text())
    assert stored["placement"] == MIDGAME


def test_labeling_is_refused_when_retention_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    r = client.post(
        "/labels/20260806-210118-0000-estimate", json={"placement": MIDGAME}
    )
    assert r.status_code == 422
    assert "BR_SAVE_UPLOADS" in r.json()["error"]


def test_labeling_an_unknown_upload_is_404(retaining):
    r = retaining.post(
        "/labels/20200101-000000-0000-estimate", json={"placement": MIDGAME}
    )
    assert r.status_code == 404


def test_a_read_without_corners_cannot_be_labeled(retaining, tmp_path):
    """A placement is only true in a stated frame. Without corners the photo can
    never be re-cropped, so the label would be data no eval could ever load."""
    upload = _estimate(retaining, corners=None)["upload"]
    r = retaining.post(f"/labels/{upload}", json={"placement": MIDGAME})
    assert r.status_code == 422
    assert "corners" in r.json()["error"]
    assert not list(tmp_path.glob(f"*{LABEL_SUFFIX}"))


def test_a_malformed_placement_is_rejected_and_nothing_is_written(
    retaining, tmp_path
):
    upload = _estimate(retaining)["upload"]
    r = retaining.post(f"/labels/{upload}", json={"placement": "8/8/8"})
    assert r.status_code == 400
    assert not list(tmp_path.glob(f"*{LABEL_SUFFIX}"))


@pytest.mark.parametrize(
    "hostile",
    [
        "..%2F..%2Fetc%2Fpasswd",
        "20260806-210118-0000-estimate.json",
        "not-an-id",
    ],
)
def test_a_hostile_upload_id_never_reaches_the_filesystem(retaining, hostile):
    """The id is a request parameter, so it is matched against the stem grammar
    rather than joined onto the retention directory."""
    r = retaining.post(f"/labels/{hostile}", json={"placement": MIDGAME})
    assert r.status_code in (400, 404)


def test_labels_do_not_change_the_running_estimator(retaining):
    """Ground truth is scoring data only. Feeding it back into the bank would
    recreate exactly the contamination it exists to measure."""
    before = retaining.app.state.estimator
    upload = _estimate(retaining)["upload"]
    retaining.post(f"/labels/{upload}", json={"placement": MIDGAME, "corrections": 4})
    assert retaining.app.state.estimator is before
