"""The fp32 export's descriptor: what was built, from which bytes, by which code.

Standard library only, so the workspace suite tests it without PyTorch
(``tests/test_chessq_export.py``); ``export.py`` is the torch half.

The export is a file we *build*, so its descriptor is provenance rather than a
download pin: the official release it was exported from (URL at the pinned
revision, SHA-256), the ``chessqueries`` commit that loaded it, the library
versions, the export's own SHA-256 and the parity check it passed.
``tools/fetch_models.py`` skips it by its :data:`fetch_models.EXPORT_KEY` block.

The export is byte-reproducible under the locked environment (re-exported
2026-11-24: same SHA-256), so the committed descriptor doubles as a check: a
re-export on another machine that produces different bytes rewrites it, and the
difference shows up in ``git diff`` rather than passing silently.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# tools/fetch_models.py: the one SHA-256 and release-pin parser.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_models as fm

#: ``model.format`` of a descriptor this tool writes.
FORMAT = "board-reader-chessq-onnx-export-v1"

#: The official release the export is built from (committed beside the others;
#: ``make models`` fetches its safetensors first).
SOURCE_DESCRIPTOR = "cq-vit-s-644-v6-seed1-safetensors-fp16-r1.json"

#: Our artifact's name: the source release with the precision we serve.
RELEASE = "cq-vit-s-644-v6-seed1-onnx-fp32"

#: ONNX opset of the export (``torch.onnx.export(opset_version=...)``).
OPSET = 17


@dataclass(frozen=True)
class Package:
    """The code that loaded the source weights."""

    name: str
    url: str
    commit: str


@dataclass(frozen=True)
class Verification:
    """Per-square argmax of the export against the package's own prediction."""

    samples: int
    squares: int
    squares_identical: int
    max_abs_dlogit: float

    @property
    def passed(self) -> bool:
        return self.samples > 0 and self.squares_identical == self.squares


def artifact_path(models_dir: Path) -> Path:
    return models_dir / f"{RELEASE}.onnx"


def descriptor_path(models_dir: Path) -> Path:
    return models_dir / f"{RELEASE}.json"


def load_source(models_dir: Path) -> tuple[fm.Pin, dict]:
    """The source release's pin and descriptor; raises if its weights are not the pinned bytes."""
    path = models_dir / SOURCE_DESCRIPTOR
    pin = fm.Pin.load(path)
    if not pin.target.is_file():
        raise FileNotFoundError(
            f"{pin.target} is missing; fetch it first (uv run python tools/fetch_models.py)"
        )
    sha = fm.sha256_of(pin.target)
    if sha != pin.sha256:
        raise ValueError(
            f"{pin.target} has sha256 {sha}, {path.name} pins {pin.sha256}; "
            "re-fetch it (delete the file, then tools/fetch_models.py)"
        )
    return pin, json.loads(path.read_text())


def build(
    *,
    source: fm.Pin,
    source_meta: dict,
    artifact: Path,
    package: Package,
    versions: dict[str, str],
    verification: Verification,
) -> dict:
    """The descriptor for ``artifact``. Refuses an export that failed its parity check."""
    if not verification.passed:
        raise ValueError(
            f"the export disagrees with the package on "
            f"{verification.squares - verification.squares_identical} of "
            f"{verification.squares} squares; not describing it"
        )
    return {
        "schema_version": 1,
        "model": {
            "release": RELEASE,
            "format": FORMAT,
            "artifact_sha256": fm.sha256_of(artifact),
        },
        "variant": source_meta["variant"],
        "precision": "fp32",
        "artifact": {"filename": artifact.name, "artifact_bytes": artifact.stat().st_size},
        "preprocessing_version": source_meta["preprocessing_version"],
        "input": {"name": "image", "dtype": "float32", "shape": [1, 3, 644, 644]},
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": [1, 64, 13],
            "class_symbols": source_meta["output"]["class_symbols"],
            "square_order": source_meta["output"]["square_order"],
        },
        "viewing_direction": {"name": "direction", "dtype": "float32", "shape": [1, 2]},
        fm.EXPORT_KEY: {
            "tool": "tools/chessq_export/export.py",
            "source": {
                "release": source_meta["model"]["release"],
                "url": source.url,
                "artifact_sha256": source.sha256,
                "artifact_bytes": source.size,
            },
            "package": asdict(package),
            "versions": versions,
            "opset": OPSET,
            "verification": asdict(verification),
        },
        "license": source_meta["license"],
    }


def is_current(models_dir: Path, source: fm.Pin) -> bool:
    """Whether the export on disk is the one its descriptor describes, from this source."""
    path, artifact = descriptor_path(models_dir), artifact_path(models_dir)
    if not (path.is_file() and artifact.is_file()):
        return False
    meta = json.loads(path.read_text())
    return (
        fm.is_export(meta)
        and meta[fm.EXPORT_KEY]["source"]["artifact_sha256"] == source.sha256
        and meta["model"]["artifact_sha256"] == fm.sha256_of(artifact)
    )


def write(models_dir: Path, descriptor: dict) -> Path:
    path = descriptor_path(models_dir)
    path.write_text(json.dumps(descriptor, indent=2) + "\n")
    return path
