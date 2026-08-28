"""Offline rebuild of the session exemplar bank from retained calibrate uploads."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_core.protocol import CLASSES, GRID, START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_core.wire import Frame

from board_reader_inference.centroid import ExemplarClassifier, load_classifier
from board_reader_inference.cli import main
from board_reader_inference.embed import Embedder, LiteEmbedder
from board_reader_inference.estimator import CentroidEstimator
from board_reader_inference.rebuild import (
    CalibrateUpload,
    Rebuild,
    crop_mask,
    load_calibrate_uploads,
    parse_mask,
    rebuild_bank,
)
from board_reader_inference.session import FRESH_BANK_GEOMETRY
from board_reader_inference.squares import BboxGeometry

#: A flat box geometry keeps the sim's ASCII glyphs inside their own square, which
#: is what LiteEmbedder can actually read (see test_centroid._photo_bank).
FLAT = BboxGeometry(headroom=0.0, side_pad=0.0)

#: Photometric augmentation is on in `build_exemplars`, so one board contributes
#: four exemplars per square.
PER_BOARD = GRID * GRID * 4


def _angled_photo() -> tuple[np.ndarray, np.ndarray]:
    return simulate_angled_view(render_board(START_PLACEMENT, square_px=48))


#: ``_write_upload(corners=...)`` default: the simulated board's own quad. A
#: literal ``None`` means "write a sidecar with no corners at all".
AUTO = "auto"


def _write_upload(
    directory: Path,
    upload_id: str,
    *,
    corners: np.ndarray | str | None = AUTO,
    placement: str | None = START_PLACEMENT,
    kind: str = "calibrate",
    write_photo: bool = True,
) -> Path:
    """Write one ``<id>.jpg``/``<id>.json`` pair the way ``server._save_upload`` does."""
    directory.mkdir(parents=True, exist_ok=True)
    photo, quad = _angled_photo()
    if isinstance(corners, str) and corners == AUTO:
        corners = quad
    if write_photo:
        cv2.imwrite(str(directory / f"{upload_id}.jpg"), photo)
    meta: dict[str, object] = {"kind": kind, "ts": 1.0}
    if corners is not None:
        meta["corners"] = (
            corners
            if isinstance(corners, str)
            else ",".join(f"{v:.1f}" for v in np.asarray(corners).reshape(-1))
        )
    if placement is not None:
        meta["placement"] = placement
    sidecar = directory / f"{upload_id}.json"
    sidecar.write_text(json.dumps(meta))
    return sidecar


class RecordingEmbedder:
    """A :class:`Embedder` that keeps the crops it was handed, for mask assertions."""

    def __init__(self) -> None:
        self.inner = LiteEmbedder()
        self.crops: list[np.ndarray] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self.crops.extend(crops)
        return self.inner.embed(crops)


# --- sidecar parsing --------------------------------------------------------


def test_load_calibrate_uploads_parses_corners_and_placement(tmp_path):
    _, quad = _angled_photo()
    _write_upload(tmp_path, "20260828-101010-0001-calibrate", corners=quad)

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert skipped == []
    (upload,) = uploads
    assert upload.upload_id == "20260828-101010-0001-calibrate"
    assert upload.photo == tmp_path / "20260828-101010-0001-calibrate.jpg"
    assert upload.corners.shape == (4, 2)
    assert upload.corners.dtype == np.float32
    np.testing.assert_allclose(upload.corners, quad, atol=0.05)
    assert upload.placement == START_PLACEMENT


def test_load_calibrate_uploads_is_ordered_by_upload_id(tmp_path):
    for stem in ("20260828-101010-0003", "20260827-090000-0001", "20260828-090000-0002"):
        _write_upload(tmp_path, f"{stem}-calibrate")

    uploads, _ = load_calibrate_uploads(tmp_path)

    assert [u.upload_id for u in uploads] == [
        "20260827-090000-0001-calibrate",
        "20260828-090000-0002-calibrate",
        "20260828-101010-0003-calibrate",
    ]


@pytest.mark.parametrize(
    ("name", "kwargs", "reason_fragment"),
    [
        ("no-corners", {"corners": None}, "corners"),
        ("bad-corners", {"corners": "1,2,3"}, "corners"),
        ("no-placement", {"placement": None}, "placement"),
        ("bad-placement", {"placement": "not-a-fen"}, "placement"),
        ("no-photo", {"write_photo": False}, "missing"),
    ],
)
def test_unusable_uploads_are_skipped_with_a_reason(
    tmp_path, name, kwargs, reason_fragment
):
    _write_upload(tmp_path, f"{name}-calibrate", **kwargs)

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert uploads == []
    (skip,) = skipped
    assert skip.sidecar.name == f"{name}-calibrate.json"
    assert reason_fragment in skip.reason


def test_unreadable_sidecar_is_skipped_not_fatal(tmp_path):
    _write_upload(tmp_path, "good-calibrate")
    (tmp_path / "broken-calibrate.json").write_text("{ truncated")

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert [u.upload_id for u in uploads] == ["good-calibrate"]
    assert [s.reason for s in skipped] == ["sidecar is not readable JSON"]


def test_estimate_sidecars_are_not_calibrate_data(tmp_path):
    """`/estimate` uploads carry no truth — only `*-calibrate.json` may seed a bank."""
    _write_upload(tmp_path, "20260828-101010-0001-estimate", placement=None)
    _write_upload(tmp_path, "mislabelled-calibrate", kind="estimate")

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert uploads == []
    assert [s.reason for s in skipped] == ["kind is 'estimate', not 'calibrate'"]


# --- masking ----------------------------------------------------------------


def test_parse_mask_maps_none_to_off():
    assert parse_mask("none") is None
    assert parse_mask("gray") == "gray"
    assert parse_mask("blur") == "blur"
    with pytest.raises(ValueError):
        parse_mask("grey")


def test_crop_mask_forces_the_env_choke_point_and_restores_it(monkeypatch):
    import os

    monkeypatch.setenv("BR_CROP_MASK", "blur")
    with crop_mask("gray"):
        assert os.environ["BR_CROP_MASK"] == "gray"
    assert os.environ["BR_CROP_MASK"] == "blur"
    with crop_mask(None):
        assert os.environ["BR_CROP_MASK"] == ""
    assert os.environ["BR_CROP_MASK"] == "blur"

    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    with crop_mask("gray"):
        assert os.environ["BR_CROP_MASK"] == "gray"
    assert "BR_CROP_MASK" not in os.environ


def _rebuild_crops(tmp_path, mask, geometry=FRESH_BANK_GEOMETRY) -> list[np.ndarray]:
    uploads, _ = load_calibrate_uploads(tmp_path)
    embedder = RecordingEmbedder()
    rebuild_bank(uploads, embedder, mask=mask, backbone="lite", geometry=geometry)
    return embedder.crops


def test_mask_reaches_the_crops_the_embedder_sees(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "20260828-101010-0001-calibrate")

    masked = _rebuild_crops(tmp_path, "gray")
    unmasked = _rebuild_crops(tmp_path, None)

    assert len(masked) == len(unmasked) == PER_BOARD
    assert any(
        not np.array_equal(a, b) for a, b in zip(masked, unmasked, strict=True)
    ), "mask=gray must change at least one crop on an angled board"


def test_mask_none_ignores_an_ambient_br_crop_mask(tmp_path, monkeypatch):
    """A deliberately unmasked rebuild must not inherit the serving default."""
    _write_upload(tmp_path, "20260828-101010-0001-calibrate")

    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    clean = _rebuild_crops(tmp_path, None)
    monkeypatch.setenv("BR_CROP_MASK", "gray")
    with_ambient = _rebuild_crops(tmp_path, None)

    assert all(
        np.array_equal(a, b) for a, b in zip(clean, with_ambient, strict=True)
    )


# --- the rebuilt bank -------------------------------------------------------


def _rebuild(tmp_path, mask=None, embedder: Embedder | None = None) -> Rebuild:
    uploads, _ = load_calibrate_uploads(tmp_path)
    return rebuild_bank(
        uploads,
        embedder if embedder is not None else LiteEmbedder(),
        mask=mask,
        backbone="lite",
        geometry=FLAT,
    )


def test_rebuild_accumulates_every_upload(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    _write_upload(tmp_path, "b-calibrate")

    result = _rebuild(tmp_path)

    assert result.used == ("a-calibrate", "b-calibrate")
    assert result.skipped == ()
    assert result.exemplars == 2 * PER_BOARD
    assert set(result.classifier.labels.tolist()) == set(CLASSES)


def test_rebuild_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    _write_upload(tmp_path, "b-calibrate")

    first, second = _rebuild(tmp_path, "gray"), _rebuild(tmp_path, "gray")

    np.testing.assert_array_equal(first.classifier.exemplars, second.classifier.exemplars)
    np.testing.assert_array_equal(first.classifier.labels, second.classifier.labels)


def test_rebuild_skips_an_undecodable_photo_and_keeps_going(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "good-calibrate")
    _write_upload(tmp_path, "corrupt-calibrate")
    (tmp_path / "corrupt-calibrate.jpg").write_bytes(b"not a jpeg")

    result = _rebuild(tmp_path)

    assert result.used == ("good-calibrate",)
    assert [s.reason for s in result.skipped] == [
        "photo corrupt-calibrate.jpg is not a decodable image"
    ]
    assert result.exemplars == PER_BOARD


def test_rebuild_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="no calibrate uploads"):
        rebuild_bank([], LiteEmbedder(), mask=None)


def test_rebuild_rejects_a_corpus_of_only_unreadable_photos(tmp_path):
    _write_upload(tmp_path, "corrupt-calibrate")
    (tmp_path / "corrupt-calibrate.jpg").write_bytes(b"not a jpeg")
    uploads, _ = load_calibrate_uploads(tmp_path)

    with pytest.raises(ValueError, match="nothing to embed"):
        rebuild_bank(uploads, LiteEmbedder(), mask=None, backbone="lite")


def test_rebuilt_bank_reads_back_as_the_bank_the_estimator_expects(
    tmp_path, monkeypatch
):
    """The npz round-trips through the loader the gateway restores banks with."""
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    result = _rebuild(tmp_path)
    out = tmp_path / "bank.npz"

    result.classifier.save(out)
    loaded = load_classifier(out)

    assert isinstance(loaded, ExemplarClassifier)
    assert loaded.backbone == "lite"
    assert loaded.crop_mode == "photo"
    assert loaded.temperature == 0.1
    assert (loaded.box_headroom, loaded.box_side_pad) == (FLAT.headroom, FLAT.side_pad)
    # `ExemplarClassifier.__init__` re-normalises on ingest, so a round-tripped
    # row differs from the saved one by float noise — never by more than that.
    np.testing.assert_allclose(
        loaded.exemplars, result.classifier.exemplars, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(loaded.labels, result.classifier.labels)

    # …and drives a CentroidEstimator, which is what serving wraps a bank in.
    estimator = CentroidEstimator(LiteEmbedder(), loaded, geometry=FLAT)
    photo, quad = _angled_photo()
    read = estimator.estimate_photo(
        Frame(board_id="A1", seq=0, capture_ts=0.0, image=photo), quad
    )
    assert np.asarray(read.square_dist).shape == (GRID, GRID, len(CLASSES))


def test_rebuild_records_the_shipped_bank_geometry_by_default(tmp_path, monkeypatch):
    """A rebuild must be swappable for the deployed bank: same geometry metadata."""
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    uploads, _ = load_calibrate_uploads(tmp_path)

    result = rebuild_bank(uploads, LiteEmbedder(), mask=None, backbone="lite")

    assert result.classifier.box_headroom == FRESH_BANK_GEOMETRY.headroom
    assert result.classifier.box_side_pad == FRESH_BANK_GEOMETRY.side_pad


# --- the CLI ----------------------------------------------------------------


def test_cli_rebuild_bank_writes_a_loadable_bank(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    uploads = tmp_path / "uploads"
    _write_upload(uploads, "a-calibrate")
    _write_upload(uploads, "b-calibrate")
    out = tmp_path / "bank.npz"

    rc = main(
        [
            "rebuild-bank",
            "--uploads", str(uploads),
            "--mask", "gray",
            "--backbone", "lite",
            "--headroom", "0.0",
            "--side-pad", "0.0",
            "-o", str(out),
        ]
    )

    assert rc == 0
    bank = load_classifier(out)
    assert isinstance(bank, ExemplarClassifier)
    assert bank.exemplars.shape[0] == 2 * PER_BOARD
    stdout = capsys.readouterr().out
    assert f"rebuilt {2 * PER_BOARD} exemplars (13/13 classes)" in stdout
    assert "mask=gray" in stdout


def test_cli_rebuild_bank_refuses_an_empty_corpus(tmp_path, capsys):
    empty = tmp_path / "uploads"
    empty.mkdir()

    rc = main(["rebuild-bank", "--uploads", str(empty), "-o", str(tmp_path / "b.npz")])

    assert rc == 2
    assert "no usable calibrate uploads" in capsys.readouterr().err
    assert not (tmp_path / "b.npz").exists()


def test_calibrate_upload_is_the_calibrate_from_call_shape(tmp_path):
    """The loader hands over exactly what ``EstimatorSession.calibrate_from`` takes."""
    _write_upload(tmp_path, "a-calibrate")
    (upload,) = load_calibrate_uploads(tmp_path)[0]

    assert isinstance(upload, CalibrateUpload)
    image = cv2.imread(str(upload.photo), cv2.IMREAD_COLOR)
    assert image is not None and image.ndim == 3
    assert upload.corners.shape == (4, 2)
