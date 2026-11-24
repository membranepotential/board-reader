"""``make models``: fetch the recognizer weights pinned by their release descriptors.

Each ``models/chessq/*.json`` is a ChessQ release descriptor (committed; the weights
are not — PolyForm Noncommercial, THIRD-PARTY.md). It names the artifact, the
Hugging Face URL at a pinned ``distribution_revision``, its size and its SHA-256.
For every descriptor this:

* checks an existing file and leaves it alone when its SHA-256 matches;
* otherwise downloads the URL to a temporary file beside the target, verifies size
  and SHA-256, and only then moves it into place — a failed or tampered download
  never replaces anything;
* refuses a descriptor whose URL is not https at its own pinned revision, so a
  moving branch such as ``main`` can never be fetched.

A descriptor with an :data:`EXPORT_KEY` block describes a file we *build* rather
than download (the fp32 ONNX, ``tools/chessq_export``); this script leaves those
alone, and ``make models`` runs the export after it.

    uv run python tools/fetch_models.py            # what `make models` runs
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTORS = "models/chessq/*.json"
#: The block that marks a descriptor as an export's provenance, not a download pin.
EXPORT_KEY = "export"
_CHUNK = 1 << 20


def is_export(meta: dict) -> bool:
    """Whether a descriptor describes a file built here rather than fetched."""
    return EXPORT_KEY in meta


def release_descriptors(root: Path) -> list[Path]:
    """The descriptors under ``root`` that pin a download, in name order."""
    return [
        path
        for path in sorted(root.glob(DESCRIPTORS))
        if not is_export(json.loads(path.read_text()))
    ]


@dataclass(frozen=True)
class Pin:
    """One artifact as its release descriptor pins it."""

    descriptor: Path
    filename: str
    url: str
    revision: str
    size: int
    sha256: str

    @classmethod
    def load(cls, descriptor: Path) -> Pin:
        meta = json.loads(descriptor.read_text())
        artifact = meta["artifact"]
        pin = cls(
            descriptor=descriptor,
            filename=artifact["filename"],
            url=artifact["url"],
            revision=artifact["distribution_revision"],
            size=int(artifact["artifact_bytes"]),
            sha256=meta["model"]["artifact_sha256"],
        )
        if not pin.url.startswith("https://") or f"/resolve/{pin.revision}/" not in pin.url:
            raise ValueError(
                f"{descriptor}: {pin.url} is not an https URL at the pinned revision "
                f"{pin.revision}"
            )
        if Path(pin.filename).name != pin.filename:
            raise ValueError(f"{descriptor}: artifact filename {pin.filename!r} is a path")
        return pin

    @property
    def target(self) -> Path:
        return self.descriptor.parent / self.filename


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(pin: Pin) -> str:
    """Make ``pin.target`` hold the pinned bytes; returns what was done."""
    if pin.target.exists() and sha256_of(pin.target) == pin.sha256:
        return f"ok        {pin.filename} (sha256 verified)"
    partial = pin.target.with_name(pin.target.name + ".partial")
    try:
        with urllib.request.urlopen(pin.url) as response, partial.open("wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
        size, sha = partial.stat().st_size, sha256_of(partial)
        if (size, sha) != (pin.size, pin.sha256):
            raise ValueError(
                f"{pin.url}: got {size} bytes sha256 {sha}, "
                f"the descriptor pins {pin.size} bytes sha256 {pin.sha256}"
            )
        partial.replace(pin.target)
    finally:
        partial.unlink(missing_ok=True)
    return f"fetched   {pin.target.name} ({pin.size} bytes, sha256 verified)"


def main(root: Path = ROOT) -> int:
    descriptors = release_descriptors(root)
    if not descriptors:
        print(f"fetch_models: no descriptors match {DESCRIPTORS}", file=sys.stderr)
        return 2
    failed = 0
    for descriptor in descriptors:
        try:
            print(fetch(Pin.load(descriptor)))
        except (OSError, ValueError, KeyError) as e:
            failed += 1
            print(f"FAILED    {descriptor.name}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
