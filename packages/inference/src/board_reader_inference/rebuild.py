"""Rebuild the session exemplar bank offline from retained ``/calibrate`` uploads.

The deployed session bank grows *live*: every ``POST /calibrate`` embeds the
photo's 64 squares and appends them (:meth:`~.session.EstimatorSession.calibrate_from`).
That makes the bank a function of history — including the crop pipeline that was
running at the time. When occlusion masking shipped
(``BR_CROP_MASK``, :func:`~.squares.crop_squares_photo`) the *queries* became
masked while the exemplars on disk stayed unmasked, so a masked query is scored
against an unmasked bank. Re-embedding the bank from the same corpus under the
same masking is the domain-aligned fix.

The corpus is already on disk: ``BR_SAVE_UPLOADS`` retains every calibrate photo
as ``<stamp>-calibrate.jpg`` next to a ``<stamp>-calibrate.json`` sidecar holding
``corners`` (``x0,y0,…,x3,y3`` in a8, h8, h1, a1 order) and ``placement`` (FEN
field 1). That pair is exactly what ``calibrate_from`` receives, so a rebuild runs
the *same* machinery — :func:`~.session.verify_labels` on the operator's labels,
:func:`~.squares.crop_squares_photo` at :data:`~.session.FRESH_BANK_GEOMETRY`,
then :func:`~.centroid.build_exemplars` (photometric augmentation on, temperature
0.1) — and differs from a live-built bank in only two ways: which masking was
applied, and which photos are in it.

Two filters stand between the corpus and the bank, and both default to ON because
each has already produced a silently-wrong bank once:

* **The label guard** (:func:`~.session.verify_labels`, the same call
  ``calibrate_from`` makes). Corner order is *semantic* — naming the wrong point
  ``a8`` rotates every label while the geometry still grid-aligns perfectly — and
  19 of the 54 sidecars in the corpus needed exactly that repair. Nothing marks a
  sidecar as audited, so an unguarded rebuild embeds the next bad one in silence.
* **The rectified-warp filter** (:func:`is_rectified_warp`). Some uploads are
  already-rectified top-down 384x384 warps whose "corners" are the image border.
  Occlusion masking is inert on them by construction — with every square at the
  same camera depth nothing occludes anything (``squares._DEPTH_EPS``) — and their
  flat squares are out of domain for the tall ``hr=0.3`` photo boxes the bank is
  cut with. Left in, they dilute the very gray-vs-none contrast a masked rebuild
  exists to measure.

One departure from the live path: the live bank is built by chaining
:meth:`~.centroid.ExemplarClassifier.extended`, which re-normalises the whole bank
on every append; building once from the concatenated embeddings is the same bank
up to float noise in that repeated normalisation of already-unit rows (locked by
``test_rebuild.test_rebuild_matches_a_live_calibrate_from``).

Masking is applied through the *env* choke point rather than the ``mask=``
parameter (see :func:`crop_mask`), so a rebuild resolves its fill through exactly
the code path that serving does — and an ambient ``BR_CROP_MASK`` can never leak
into a deliberately unmasked rebuild.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from board_reader_core.corners import parse_corners
from board_reader_core.protocol import CLASSES, parse_placement

from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
from board_reader_inference.embed import Embedder
from board_reader_inference.routing import PhotoReader
from board_reader_inference.session import (
    FRESH_BANK_GEOMETRY,
    CalibrationRejected,
    verify_labels,
)
from board_reader_inference.squares import (
    BboxGeometry,
    OcclusionFill,
    crop_squares_photo,
    full_frame_corners,
)

#: Sidecar filenames a calibrate upload is retained under (``server._save_upload``).
CALIBRATE_SUFFIX = "-calibrate.json"

#: What ``--mask`` accepts on the CLI; ``"none"`` maps to ``None`` (masking off).
MASK_CHOICES = ("gray", "blur", "none")

#: The deployed classifier (``BR_MODEL`` in compose.yaml), used as the label
#: guard's probe. It must be the model that will *serve* against this bank: the
#: guard reads the photo through its own crop geometry, and a different model
#: would vet the labels under crops nothing else will ever use.
DEFAULT_GUARD_MODEL = "models/run_e/model.onnx"

#: How far a corner may sit from the image border and still count as "on it",
#: as a fraction of the image diagonal. The corpus splits at 0.0 (13 rectified
#: warps, corners exactly on the border) versus 0.16 at the closest real photo,
#: so anything in this range separates them; 1% is nowhere near either cluster.
RECTIFIED_TOLERANCE = 0.01


def parse_mask(spec: str) -> OcclusionFill | None:
    """``gray`` | ``blur`` | ``none`` -> the fill :func:`crop_mask` should force."""
    if spec == "none":
        return None
    if spec in ("gray", "blur"):
        return spec  # type: ignore[return-value]
    raise ValueError(f"mask must be one of {MASK_CHOICES}, got {spec!r}")


@contextlib.contextmanager
def crop_mask(fill: OcclusionFill | None) -> Iterator[None]:
    """Force ``crop_squares_photo``'s ``BR_CROP_MASK`` choke point to ``fill``.

    ``crop_squares_photo(mask=None)`` means *read the environment*, not *do not
    mask* — so an unmasked rebuild cannot be expressed by the parameter alone
    while ``BR_CROP_MASK`` is exported (which it is, in every deployed shell).
    Setting the variable for the duration both makes ``None`` mean "off"
    unambiguously and routes the choice through the same resolution serving uses.

    **CLI use only.** This mutates process-global ``os.environ`` and is therefore
    not thread-safe: the server crops inside a worker thread
    (``asyncio.to_thread(session.calibrate_from, …)``), so calling this in the
    gateway would change the mask under a concurrent read. The offline rebuild is
    single-threaded, which is the only reason it is sound here.
    """
    previous = os.environ.get("BR_CROP_MASK")
    os.environ["BR_CROP_MASK"] = fill or ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BR_CROP_MASK", None)
        else:
            os.environ["BR_CROP_MASK"] = previous


def is_rectified_warp(
    image: np.ndarray, corners: np.ndarray, *, tolerance: float = RECTIFIED_TOLERANCE
) -> bool:
    """Are these "corners" just the image border — i.e. an already-rectified warp?

    Such an upload is a top-down board crop posted with
    :func:`~.squares.full_frame_corners`, not a photo of a board in a scene. Two
    consequences make it the wrong thing to embed into a photo-geometry bank:
    every square recovers the same camera depth, so occlusion masking is a no-op
    by construction, and the flat squares are not what ``hr=0.3`` tall boxes are
    cut for. Compared rotation-invariantly, since which physical corner is called
    ``a8`` is exactly what varies across uploads.
    """
    quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    border = full_frame_corners(image)
    limit = tolerance * float(np.hypot(*image.shape[:2]))
    return any(
        float(np.abs(quad - np.roll(border, k, axis=0)).max()) <= limit
        for k in range(4)
    )


@dataclass(frozen=True, slots=True)
class CalibrateUpload:
    """One retained calibrate photo and the labels it was posted with."""

    upload_id: str  # the sidecar stem, e.g. ``20260828-111421-0007-calibrate``
    photo: Path
    corners: np.ndarray  # (4, 2) float32, a8/h8/h1/a1 in ``photo`` pixels
    placement: str  # FEN field 1, in the frame ``corners`` defines


#: Why an upload did or did not contribute exemplars.
#:
#: ``used``       embedded (``detail`` names the label rotation applied)
#: ``unusable``   sidecar or photo could not be read
#: ``rectified``  an already-rectified warp, excluded by default
#: ``rejected``   the label guard says the photo does not show the claimed position
Outcome = Literal["used", "unusable", "rectified", "rejected"]


@dataclass(frozen=True, slots=True)
class UploadVerdict:
    """What became of one upload, and why — the rebuild's whole audit trail."""

    upload_id: str
    outcome: Outcome
    detail: str


def _upload_from_sidecar(sidecar: Path) -> CalibrateUpload | UploadVerdict:
    """Parse one ``*-calibrate.json`` sidecar; never raises.

    A corpus assembled by a year of live requests contains half-written sidecars,
    photos deleted by hand, and (before the corner audit) entries with no corners
    at all. Every one of those is a *skip with a reason*, not a crashed rebuild.
    """
    upload_id = sidecar.name[: -len(".json")]

    def unusable(detail: str) -> UploadVerdict:
        return UploadVerdict(upload_id=upload_id, outcome="unusable", detail=detail)

    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return unusable("sidecar is not readable JSON")
    if not isinstance(meta, dict):
        return unusable("sidecar is not a JSON object")
    if meta.get("kind") != "calibrate":
        return unusable(f"kind is {meta.get('kind')!r}, not 'calibrate'")
    try:
        corners = parse_corners(meta.get("corners"))
    except ValueError as e:
        return unusable(f"corners: {e}")
    placement = meta.get("placement")
    if not isinstance(placement, str):
        return unusable("no placement")
    try:
        parse_placement(placement)
    except ValueError as e:
        return unusable(f"placement: {e}")
    photo = sidecar.with_name(f"{upload_id}.jpg")
    if not photo.exists():
        return unusable(f"photo {photo.name} is missing")
    return CalibrateUpload(
        upload_id=upload_id, photo=photo, corners=corners, placement=placement
    )


def load_calibrate_uploads(
    directory: str | Path,
) -> tuple[list[CalibrateUpload], list[UploadVerdict]]:
    """Every readable calibrate upload in ``directory``, plus what was skipped.

    Sorted by sidecar name — which is a timestamp-and-sequence stem, so the order
    is the order the operator calibrated in and is stable across runs.
    """
    kept: list[CalibrateUpload] = []
    skipped: list[UploadVerdict] = []
    for sidecar in sorted(Path(directory).glob(f"*{CALIBRATE_SUFFIX}")):
        parsed = _upload_from_sidecar(sidecar)
        if isinstance(parsed, CalibrateUpload):
            kept.append(parsed)
        else:
            skipped.append(parsed)
    return kept, skipped


@dataclass(frozen=True, slots=True)
class Rebuild:
    """A rebuilt bank and the per-upload audit trail that produced it."""

    classifier: ExemplarClassifier
    mask: OcclusionFill | None
    verdicts: tuple[UploadVerdict, ...]

    @property
    def exemplars(self) -> int:
        return int(self.classifier.exemplars.shape[0])

    @property
    def classes(self) -> int:
        """How many of the 13 classes have at least one exemplar."""
        return len(set(self.classifier.labels.tolist()) & set(CLASSES))

    def of(self, outcome: Outcome) -> tuple[UploadVerdict, ...]:
        return tuple(v for v in self.verdicts if v.outcome == outcome)


def rebuild_bank(
    uploads: Sequence[CalibrateUpload],
    embedder: Embedder,
    *,
    mask: OcclusionFill | None,
    guard: PhotoReader | None,
    include_rectified: bool = False,
    backbone: str = "dinov2",
    geometry: BboxGeometry = FRESH_BANK_GEOMETRY,
    square_size: int = 224,
) -> Rebuild:
    """Embed every accepted upload's 64 labelled squares into one exemplar bank.

    The per-upload work is ``calibrate_from``'s, verbatim: vet the labels with
    :func:`~.session.verify_labels` against ``guard``, cut the photo boxes at
    ``geometry`` (masked per ``mask``), then :func:`~.centroid.build_exemplars`
    with its shipped defaults (photometric augmentation, temperature 0.1).

    ``guard=None`` embeds the operator's labels unchecked — the escape hatch for a
    corpus that has been audited by other means, never the default. Every upload's
    fate lands in :attr:`Rebuild.verdicts`.
    """
    if not uploads:
        raise ValueError("no calibrate uploads to rebuild from")
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    verdicts: list[UploadVerdict] = []

    def verdict(upload: CalibrateUpload, outcome: Outcome, detail: str) -> None:
        verdicts.append(
            UploadVerdict(upload_id=upload.upload_id, outcome=outcome, detail=detail)
        )

    for upload in uploads:
        image = cv2.imread(str(upload.photo), cv2.IMREAD_COLOR)
        if image is None:
            verdict(
                upload, "unusable", f"photo {upload.photo.name} is not a decodable image"
            )
            continue
        if not include_rectified and is_rectified_warp(image, upload.corners):
            verdict(
                upload,
                "rectified",
                f"corners are the {image.shape[1]}x{image.shape[0]} image border",
            )
            continue
        placement = upload.placement
        rotation = 0
        if guard is not None:
            # The guard probes UNMASKED, whatever `mask` the bank is built with
            # and whatever BR_CROP_MASK the shell exports. Its job is to vet
            # labels, not to model a serving config — and pinning it keeps the
            # accepted corpus a function of the uploads alone, so a `--mask gray`
            # and a `--mask none` rebuild of the same directory are guaranteed to
            # contain the same photos. Without that, the controlled comparison a
            # masked/unmasked bank pair exists for could silently compare two
            # different corpora.
            with crop_mask(None):
                checked = verify_labels(guard, image, upload.corners, placement)
            if isinstance(checked, CalibrationRejected):
                # Name the rotated-fit cause when there is one: a board turned in
                # the frame reads as a total mismatch, and "would pass turned 90"
                # is the difference between a mislabeled upload and a whole
                # afternoon of portrait-held-phone calibrates (2026-08-27).
                fit = checked.rotated_fit
                cause = (
                    f"; the photo would pass turned {90 * fit.quarter_turns}deg "
                    f"({fit.piece_agree}/{checked.piece_squares}) — board rotated "
                    "in the frame"
                    if fit is not None
                    else ""
                )
                verdict(
                    upload,
                    "rejected",
                    f"label guard: only {checked.piece_agree}/{checked.piece_squares} "
                    f"claimed piece squares agree{cause}",
                )
                continue
            rotation, placement = checked
        with crop_mask(mask):
            crops = crop_squares_photo(
                image, upload.corners, geometry=geometry, size=square_size
            )
        board = build_exemplars(
            embedder,
            crops,
            placement,
            backbone=backbone,
            box_headroom=geometry.headroom,
            box_side_pad=geometry.side_pad,
        )
        embeddings.append(board.exemplars)
        labels.extend(str(lab) for lab in board.labels)
        verdict(
            upload,
            "used",
            f"labels rotated {rotation * 90}deg" if rotation else "labels as given",
        )
    if not embeddings:
        raise ValueError("every calibrate upload was skipped; nothing to embed")
    classifier = ExemplarClassifier(
        np.concatenate(embeddings, axis=0),
        labels,
        backbone=backbone,
        box_headroom=geometry.headroom,
        box_side_pad=geometry.side_pad,
    )
    return Rebuild(classifier=classifier, mask=mask, verdicts=tuple(verdicts))
