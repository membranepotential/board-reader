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
field 1). That pair is exactly what ``calibrate_from`` receives, so a rebuild can
run the *same* machinery — :func:`~.squares.crop_squares_photo` with
:data:`~.session.FRESH_BANK_GEOMETRY`, then
:func:`~.centroid.build_exemplars` (photometric augmentation on, temperature 0.1)
— and differ from a live-built bank in only two ways: which masking was applied,
and which photos are in it.

Two departures from the live path, both deliberate:

* **No calibrate guard.** ``calibrate_from`` probes the photo against a trained
  model before trusting the operator's labels. Here the corners have already been
  audited and repaired offline (``tools/repair_upload_orientation.py``), and
  requiring a model would tie a bank rebuild to a model choice. Every sidecar's
  labels are taken as given.
* **One construction, not 54 appends.** The live bank is built by chaining
  :meth:`~.centroid.ExemplarClassifier.extended`, which re-normalises the whole
  bank on every append; building once from the concatenated embeddings is the
  same bank up to float noise in that repeated normalisation of already-unit rows.

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

import cv2
import numpy as np

from board_reader_core.corners import parse_corners
from board_reader_core.protocol import parse_placement

from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
from board_reader_inference.embed import Embedder
from board_reader_inference.session import FRESH_BANK_GEOMETRY
from board_reader_inference.squares import (
    BboxGeometry,
    OcclusionFill,
    crop_squares_photo,
)

#: Sidecar filenames a calibrate upload is retained under (``server._save_upload``).
CALIBRATE_SUFFIX = "-calibrate.json"

#: What ``--mask`` accepts on the CLI; ``"none"`` maps to ``None`` (masking off).
MASK_CHOICES = ("gray", "blur", "none")


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


@dataclass(frozen=True, slots=True)
class CalibrateUpload:
    """One retained calibrate photo and the labels it was posted with."""

    upload_id: str  # the sidecar stem, e.g. ``20260828-111421-0007-calibrate``
    photo: Path
    corners: np.ndarray  # (4, 2) float32, a8/h8/h1/a1 in ``photo`` pixels
    placement: str  # FEN field 1, in the frame ``corners`` defines


@dataclass(frozen=True, slots=True)
class SkippedUpload:
    """A sidecar that could not become a :class:`CalibrateUpload`, and why."""

    sidecar: Path
    reason: str


def _upload_from_sidecar(sidecar: Path) -> CalibrateUpload | SkippedUpload:
    """Parse one ``*-calibrate.json`` sidecar; never raises.

    A corpus assembled by a year of live requests contains half-written sidecars,
    photos deleted by hand, and (before the corner audit) entries with no corners
    at all. Every one of those is a *skip with a reason*, not a crashed rebuild.
    """

    def skip(reason: str) -> SkippedUpload:
        return SkippedUpload(sidecar=sidecar, reason=reason)

    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return skip("sidecar is not readable JSON")
    if not isinstance(meta, dict):
        return skip("sidecar is not a JSON object")
    if meta.get("kind") != "calibrate":
        return skip(f"kind is {meta.get('kind')!r}, not 'calibrate'")
    try:
        corners = parse_corners(meta.get("corners"))
    except ValueError as e:
        return skip(f"corners: {e}")
    placement = meta.get("placement")
    if not isinstance(placement, str):
        return skip("no placement")
    try:
        parse_placement(placement)
    except ValueError as e:
        return skip(f"placement: {e}")
    upload_id = sidecar.name[: -len(".json")]
    photo = sidecar.with_name(f"{upload_id}.jpg")
    if not photo.exists():
        return skip(f"photo {photo.name} is missing")
    return CalibrateUpload(
        upload_id=upload_id, photo=photo, corners=corners, placement=placement
    )


def load_calibrate_uploads(
    directory: str | Path,
) -> tuple[list[CalibrateUpload], list[SkippedUpload]]:
    """Every usable calibrate upload in ``directory``, plus what was skipped.

    Sorted by sidecar name — which is a timestamp-and-sequence stem, so the order
    is the order the operator calibrated in and is stable across runs.
    """
    kept: list[CalibrateUpload] = []
    skipped: list[SkippedUpload] = []
    for sidecar in sorted(Path(directory).glob(f"*{CALIBRATE_SUFFIX}")):
        parsed = _upload_from_sidecar(sidecar)
        if isinstance(parsed, CalibrateUpload):
            kept.append(parsed)
        else:
            skipped.append(parsed)
    return kept, skipped


@dataclass(frozen=True, slots=True)
class Rebuild:
    """A rebuilt bank and the provenance needed to say what it is."""

    classifier: ExemplarClassifier
    mask: OcclusionFill | None
    used: tuple[str, ...]  # upload ids that contributed exemplars
    skipped: tuple[SkippedUpload, ...]

    @property
    def exemplars(self) -> int:
        return int(self.classifier.exemplars.shape[0])


def rebuild_bank(
    uploads: Sequence[CalibrateUpload],
    embedder: Embedder,
    *,
    mask: OcclusionFill | None,
    backbone: str = "dinov2",
    geometry: BboxGeometry = FRESH_BANK_GEOMETRY,
    square_size: int = 224,
) -> Rebuild:
    """Embed every upload's 64 labelled squares into one exemplar bank.

    The per-upload work is ``calibrate_from``'s, verbatim: cut the photo boxes at
    ``geometry`` (masked per ``mask``), then :func:`~.centroid.build_exemplars`
    with its shipped defaults (photometric augmentation, temperature 0.1). The
    per-upload banks are concatenated rather than chained through
    :meth:`~.centroid.ExemplarClassifier.extended` — see the module docstring.
    """
    if not uploads:
        raise ValueError("no calibrate uploads to rebuild from")
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    used: list[str] = []
    skipped: list[SkippedUpload] = []
    for upload in uploads:
        image = cv2.imread(str(upload.photo), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append(
                SkippedUpload(
                    sidecar=upload.photo.with_suffix(".json"),
                    reason=f"photo {upload.photo.name} is not a decodable image",
                )
            )
            continue
        with crop_mask(mask):
            crops = crop_squares_photo(
                image, upload.corners, geometry=geometry, size=square_size
            )
        board = build_exemplars(
            embedder,
            crops,
            upload.placement,
            backbone=backbone,
            box_headroom=geometry.headroom,
            box_side_pad=geometry.side_pad,
        )
        embeddings.append(board.exemplars)
        labels.extend(str(lab) for lab in board.labels)
        used.append(upload.upload_id)
    if not embeddings:
        raise ValueError("every calibrate upload was unreadable; nothing to embed")
    classifier = ExemplarClassifier(
        np.concatenate(embeddings, axis=0),
        labels,
        backbone=backbone,
        box_headroom=geometry.headroom,
        box_side_pad=geometry.side_pad,
    )
    return Rebuild(
        classifier=classifier,
        mask=mask,
        used=tuple(used),
        skipped=tuple(skipped),
    )
