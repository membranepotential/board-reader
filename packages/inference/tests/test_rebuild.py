"""Offline rebuild of the session exemplar bank from retained calibrate uploads."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    START_PLACEMENT,
    PositionEvent,
    parse_placement,
)
from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_core.wire import Frame

from board_reader_inference.centroid import (
    ExemplarClassifier,
    build_exemplars,
    load_classifier,
)
from board_reader_inference.cli import main
from board_reader_inference.embed import Embedder, LiteEmbedder
from board_reader_inference.estimator import CentroidEstimator
from board_reader_inference.labels import rotate_placement
from board_reader_inference.rebuild import (
    CalibrateUpload,
    Rebuild,
    crop_mask,
    is_rectified_warp,
    load_calibrate_uploads,
    parse_mask,
    rebuild_bank,
)
from board_reader_inference.session import (
    FRESH_BANK_GEOMETRY,
    Calibration,
    EstimatorSession,
)
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    full_frame_corners,
)

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
    photo: np.ndarray | None = None,
    kind: str = "calibrate",
    write_photo: bool = True,
) -> Path:
    """Write one ``<id>.jpg``/``<id>.json`` pair the way ``server._save_upload`` does."""
    directory.mkdir(parents=True, exist_ok=True)
    default_photo, quad = _angled_photo()
    if photo is None:
        photo = default_photo
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
    assert skip.upload_id == f"{name}-calibrate"
    assert skip.outcome == "unusable"
    assert reason_fragment in skip.detail


def test_unreadable_sidecar_is_skipped_not_fatal(tmp_path):
    _write_upload(tmp_path, "good-calibrate")
    (tmp_path / "broken-calibrate.json").write_text("{ truncated")

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert [u.upload_id for u in uploads] == ["good-calibrate"]
    assert [s.detail for s in skipped] == ["sidecar is not readable JSON"]


def test_estimate_sidecars_are_not_calibrate_data(tmp_path):
    """`/estimate` uploads carry no truth — only `*-calibrate.json` may seed a bank."""
    _write_upload(tmp_path, "20260828-101010-0001-estimate", placement=None)
    _write_upload(tmp_path, "mislabelled-calibrate", kind="estimate")

    uploads, skipped = load_calibrate_uploads(tmp_path)

    assert uploads == []
    assert [s.detail for s in skipped] == ["kind is 'estimate', not 'calibrate'"]


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
    rebuild_bank(
        uploads, embedder, mask=mask, guard=None, backbone="lite", geometry=geometry
    )
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


def _rebuild(
    tmp_path, mask=None, embedder: Embedder | None = None, guard=None
) -> Rebuild:
    uploads, _ = load_calibrate_uploads(tmp_path)
    return rebuild_bank(
        uploads,
        embedder if embedder is not None else LiteEmbedder(),
        mask=mask,
        guard=guard,
        backbone="lite",
        geometry=FLAT,
    )


def test_rebuild_accumulates_every_upload(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    _write_upload(tmp_path, "b-calibrate")

    result = _rebuild(tmp_path)

    assert [v.upload_id for v in result.of("used")] == ["a-calibrate", "b-calibrate"]
    assert {v.outcome for v in result.verdicts} == {"used"}
    assert result.exemplars == 2 * PER_BOARD
    assert result.classes == len(CLASSES)
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

    assert [v.upload_id for v in result.of("used")] == ["good-calibrate"]
    assert [v.detail for v in result.of("unusable")] == [
        "photo corrupt-calibrate.jpg is not a decodable image"
    ]
    assert result.exemplars == PER_BOARD


def test_rebuild_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="no calibrate uploads"):
        rebuild_bank([], LiteEmbedder(), mask=None, guard=None)


def test_rebuild_rejects_a_corpus_of_only_unreadable_photos(tmp_path):
    _write_upload(tmp_path, "corrupt-calibrate")
    (tmp_path / "corrupt-calibrate.jpg").write_bytes(b"not a jpeg")
    uploads, _ = load_calibrate_uploads(tmp_path)

    with pytest.raises(ValueError, match="nothing to embed"):
        rebuild_bank(uploads, LiteEmbedder(), mask=None, guard=None, backbone="lite")


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

    result = rebuild_bank(
        uploads, LiteEmbedder(), mask=None, guard=None, backbone="lite"
    )

    assert result.classifier.box_headroom == FRESH_BANK_GEOMETRY.headroom
    assert result.classifier.box_side_pad == FRESH_BANK_GEOMETRY.side_pad


# --- the label guard --------------------------------------------------------


class FixedProbe:
    """A :class:`PhotoReader` that always reads ``placement``, whatever the photo."""

    def __init__(self, placement: str) -> None:
        self.placement = placement

    def estimate_photo(self, frame, corners):
        grid = [cell for row in parse_placement(self.placement) for cell in row]
        dist = np.zeros((GRID, GRID, len(CLASSES)), dtype=np.float32)
        for i, cell in enumerate(grid):
            dist[i // GRID, i % GRID, CLASSES.index(cell)] = 1.0
        return PositionEvent(
            board_id="probe", seq=0, utc="1970-01-01T00:00:00Z",
            placement=self.placement, square_dist=dist.tolist(), confidence=1.0,
            status="ok",
        )


def test_guard_rejects_an_upload_whose_photo_does_not_match_its_labels(tmp_path):
    """A mid-game photo labelled 'start' is poison — the live path refuses it too."""
    _write_upload(tmp_path, "honest-calibrate")
    _write_upload(tmp_path, "liar-calibrate")
    uploads, _ = load_calibrate_uploads(tmp_path)

    # The probe reads an empty board, so 'start' labels agree on 0 piece squares
    # and every upload falls below CALIBRATE_MIN_PIECE_AGREE.
    with pytest.raises(ValueError, match="nothing to embed"):
        rebuild_bank(
            uploads,
            LiteEmbedder(),
            mask=None,
            guard=FixedProbe("8/8/8/8/8/8/8/8"),
            backbone="lite",
            geometry=FLAT,
        )


def test_guard_rejection_is_reported_per_upload(tmp_path):
    """One bad upload must not take the rebuild down with it — it is named and dropped."""
    _write_upload(tmp_path, "honest-calibrate")
    # A mid-game claim over a start-position photo: its four piece squares read
    # 'empty' at every rotation, so piece agreement is 0/4.
    _write_upload(tmp_path, "liar-calibrate", placement="8/8/3qk3/8/3QK3/8/8/8")

    result = _rebuild(tmp_path, guard=FixedProbe(START_PLACEMENT))

    assert [v.upload_id for v in result.of("used")] == ["honest-calibrate"]
    (rejected,) = result.of("rejected")
    assert rejected.upload_id == "liar-calibrate"
    assert "label guard" in rejected.detail
    assert result.exemplars == PER_BOARD


def test_guard_rotates_labels_that_were_stored_a_quarter_turn_off(tmp_path):
    """Corner order is semantic: a rolled quad rotates every label silently.

    The live guard resolves that against the model's own reading; the rebuild must
    apply the same correction rather than embedding colour-swapped exemplars.
    """
    rotated = rotate_placement(START_PLACEMENT, 1)
    _write_upload(tmp_path, "rolled-calibrate", placement=rotated)

    # The probe sees the true start position, so the stored labels fit at k=3.
    result = _rebuild(tmp_path, guard=FixedProbe(START_PLACEMENT))

    (used,) = result.of("used")
    assert used.detail == "labels rotated 270deg"
    # …and the exemplars carry the CORRECTED labels, not the stored ones.
    assert result.classifier.labels.tolist() == [
        lab
        for cell in [c for row in parse_placement(START_PLACEMENT) for c in row]
        for lab in [cell] * 4
    ]


def test_guard_decisions_do_not_depend_on_the_mask(tmp_path, monkeypatch):
    """A gray and a none rebuild of one directory must contain the SAME photos.

    The guard probes through ``crop_squares_photo`` too, so if it inherited the
    bank's masking (or the shell's) the two halves of a controlled masked-vs-
    unmasked comparison could quietly be built from different corpora.
    """
    _write_upload(tmp_path, "a-calibrate")
    _write_upload(tmp_path, "b-calibrate")
    probe = FixedProbe(START_PLACEMENT)

    monkeypatch.setenv("BR_CROP_MASK", "blur")
    gray = _rebuild(tmp_path, "gray", guard=probe)
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    none = _rebuild(tmp_path, None, guard=probe)

    assert [(v.upload_id, v.outcome, v.detail) for v in gray.verdicts] == [
        (v.upload_id, v.outcome, v.detail) for v in none.verdicts
    ]
    # …and the banks still differ, or the comparison would be measuring nothing.
    assert not np.array_equal(gray.classifier.exemplars, none.classifier.exemplars)


def test_guard_off_embeds_the_stored_labels_unchecked(tmp_path):
    rotated = rotate_placement(START_PLACEMENT, 1)
    _write_upload(tmp_path, "rolled-calibrate", placement=rotated)

    result = _rebuild(tmp_path, guard=None)

    (used,) = result.of("used")
    assert used.detail == "labels as given"
    a8_as_stored = next(c for row in parse_placement(rotated) for c in row)
    assert result.classifier.labels.tolist()[:4] == [a8_as_stored] * 4


# --- the rectified-warp filter ----------------------------------------------


def test_is_rectified_warp_recognises_full_frame_corners():
    board = render_board(START_PLACEMENT, square_px=48)
    assert is_rectified_warp(board, full_frame_corners(board))
    # Rotation-invariant: which physical corner is called a8 varies by upload.
    assert is_rectified_warp(board, np.roll(full_frame_corners(board), 2, axis=0))


def test_is_rectified_warp_rejects_a_real_angled_photo():
    photo, quad = _angled_photo()
    assert not is_rectified_warp(photo, quad)


def test_rectified_warps_are_excluded_by_default(tmp_path, monkeypatch):
    """They dilute the gray-vs-none contrast: masking is inert at uniform depth."""
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    board = render_board(START_PLACEMENT, square_px=48)
    _write_upload(tmp_path, "photo-calibrate")
    _write_upload(
        tmp_path, "warp-calibrate", photo=board, corners=full_frame_corners(board)
    )

    result = _rebuild(tmp_path)

    assert [v.upload_id for v in result.of("used")] == ["photo-calibrate"]
    (excluded,) = result.of("rectified")
    assert excluded.upload_id == "warp-calibrate"
    assert "image border" in excluded.detail
    assert result.exemplars == PER_BOARD


def test_include_rectified_keeps_them(tmp_path, monkeypatch):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    board = render_board(START_PLACEMENT, square_px=48)
    _write_upload(tmp_path, "photo-calibrate")
    _write_upload(
        tmp_path, "warp-calibrate", photo=board, corners=full_frame_corners(board)
    )
    uploads, _ = load_calibrate_uploads(tmp_path)

    result = rebuild_bank(
        uploads,
        LiteEmbedder(),
        mask=None,
        guard=None,
        include_rectified=True,
        backbone="lite",
        geometry=FLAT,
    )

    assert len(result.of("used")) == 2
    assert result.of("rectified") == ()
    assert result.exemplars == 2 * PER_BOARD


def test_masking_is_inert_on_a_rectified_warp(tmp_path, monkeypatch):
    """The reason they are excluded, pinned as a fact rather than a claim."""
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    board = render_board(START_PLACEMENT, square_px=48)
    quad = full_frame_corners(board)

    masked = crop_squares_photo(board, quad, geometry=FLAT, size=64, mask="gray")
    unmasked = crop_squares_photo(board, quad, geometry=FLAT, size=64)

    assert all(
        np.array_equal(a, b) for a, b in zip(masked, unmasked, strict=True)
    ), "every square is at the same camera depth, so nothing can occlude anything"


# --- parity with the live /calibrate path -----------------------------------


def test_rebuild_matches_a_live_calibrate_from(tmp_path, monkeypatch):
    """The spec's central claim: a rebuild differs from a live calibrate ONLY in
    masking and corpus. Same photo, same embedder, same geometry -> same rows.

    Guards against a silent drift in ``build_exemplars``' defaults (augmentation,
    temperature, label expansion) that would make a rebuilt bank quietly
    incomparable with the deployed one.
    """
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    _write_upload(tmp_path, "a-calibrate")
    (upload,) = load_calibrate_uploads(tmp_path)[0]
    # Read the photo back off disk so both paths see identical (JPEG-decoded) pixels.
    image = cv2.imread(str(upload.photo), cv2.IMREAD_COLOR)

    # A live session whose bank member fixes the embedder and box geometry, the
    # way an occupancy bank does on the real node.
    seed = build_exemplars(
        LiteEmbedder(),
        crop_squares_photo(image, upload.corners, geometry=FLAT),
        START_PLACEMENT,
        backbone="lite",
        box_headroom=FLAT.headroom,
        box_side_pad=FLAT.side_pad,
    )
    session = EstimatorSession(CentroidEstimator(LiteEmbedder(), seed, geometry=FLAT))
    outcome = session.calibrate_from(image, upload.corners, START_PLACEMENT)
    assert isinstance(outcome, Calibration)
    live = session.centroid_part().classifier.exemplars[-PER_BOARD:]

    rebuilt = _rebuild(tmp_path).classifier.exemplars

    assert rebuilt.shape == live.shape
    np.testing.assert_allclose(rebuilt, live, rtol=1e-5, atol=1e-6)


# --- the CLI ----------------------------------------------------------------


def _cli(uploads: Path, out: Path, *extra: str) -> int:
    """`rebuild-bank` on the lite backbone with the guard off (no ONNX in tests)."""
    return main(
        [
            "rebuild-bank",
            "--uploads", str(uploads),
            "--backbone", "lite",
            "--no-verify-labels",
            "--headroom", "0.0",
            "--side-pad", "0.0",
            "-o", str(out),
            *extra,
        ]
    )


def test_cli_rebuild_bank_writes_a_loadable_bank(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BR_CROP_MASK", raising=False)
    uploads = tmp_path / "uploads"
    _write_upload(uploads, "a-calibrate")
    _write_upload(uploads, "b-calibrate")
    out = tmp_path / "bank.npz"

    rc = _cli(uploads, out, "--mask", "gray")

    assert rc == 0
    bank = load_classifier(out)
    assert isinstance(bank, ExemplarClassifier)
    assert bank.exemplars.shape[0] == 2 * PER_BOARD
    stdout = capsys.readouterr().out
    assert f"rebuilt {2 * PER_BOARD} exemplars (13/13 classes)" in stdout
    assert "2 used" in stdout
    assert "mask=gray" in stdout
    assert "guard=off" in stdout


def test_cli_rebuild_bank_refuses_an_empty_corpus(tmp_path, capsys):
    empty = tmp_path / "uploads"
    empty.mkdir()

    rc = _cli(empty, tmp_path / "b.npz")

    assert rc == 2
    assert "no readable calibrate uploads" in capsys.readouterr().err
    assert not (tmp_path / "b.npz").exists()


def test_cli_rebuild_bank_refuses_to_overwrite_without_force(tmp_path, capsys):
    """A pinned bank is untracked and unrecoverable — never clobber it silently."""
    uploads = tmp_path / "uploads"
    _write_upload(uploads, "a-calibrate")
    pinned = tmp_path / "bank-baseline.npz"
    pinned.write_bytes(b"the pinned baseline")

    rc = _cli(uploads, pinned)

    assert rc == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert pinned.read_bytes() == b"the pinned baseline", "the file must be untouched"

    assert _cli(uploads, pinned, "--force") == 0
    assert load_classifier(pinned).exemplars.shape[0] == PER_BOARD


def test_cli_rebuild_bank_checks_the_target_before_doing_the_work(tmp_path, capsys):
    """The overwrite refusal must come before the GPU work, not after it."""
    uploads = tmp_path / "uploads"
    _write_upload(uploads, "a-calibrate")
    pinned = tmp_path / "bank.npz"
    pinned.write_bytes(b"pinned")

    embedder_calls: list[int] = []

    class Tripwire(LiteEmbedder):
        def embed(self, crops):  # pragma: no cover - must never run
            embedder_calls.append(len(crops))
            return super().embed(crops)

    import board_reader_inference.embed as embed_mod

    original = embed_mod.make_embedder
    embed_mod.make_embedder = lambda name: Tripwire()
    try:
        rc = _cli(uploads, pinned)
    finally:
        embed_mod.make_embedder = original

    assert rc == 2
    assert embedder_calls == []


def test_cli_rebuild_bank_exits_nonzero_on_a_class_incomplete_bank(tmp_path, capsys):
    """A class with no exemplar can never be predicted — that must not exit 0."""
    uploads = tmp_path / "uploads"
    # A board of pawns only: 3 of the 13 classes get exemplars.
    _write_upload(uploads, "a-calibrate", placement="pppppppp/8/8/8/8/8/8/PPPPPPPP")
    out = tmp_path / "bank.npz"

    rc = _cli(uploads, out)

    assert rc == 1
    assert "covers only 3/13 classes" in capsys.readouterr().err
    assert out.exists(), "the bank is still written — the exit code is the signal"

    assert _cli(uploads, out, "--force", "--allow-incomplete") == 0


def test_cli_rebuild_bank_reports_a_guard_model_it_cannot_load(tmp_path, capsys):
    uploads = tmp_path / "uploads"
    _write_upload(uploads, "a-calibrate")

    rc = main(
        [
            "rebuild-bank",
            "--uploads", str(uploads),
            "--backbone", "lite",
            "--guard-model", str(tmp_path / "nope.onnx"),
            "-o", str(tmp_path / "bank.npz"),
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "label guard needs a usable model" in err
    assert "--no-verify-labels" in err
    assert not (tmp_path / "bank.npz").exists()


def test_calibrate_upload_is_the_calibrate_from_call_shape(tmp_path):
    """The loader hands over exactly what ``EstimatorSession.calibrate_from`` takes."""
    _write_upload(tmp_path, "a-calibrate")
    (upload,) = load_calibrate_uploads(tmp_path)[0]

    assert isinstance(upload, CalibrateUpload)
    image = cv2.imread(str(upload.photo), cv2.IMREAD_COLOR)
    assert image is not None and image.ndim == 3
    assert upload.corners.shape == (4, 2)
