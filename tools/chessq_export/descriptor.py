"""The descriptors of the files ``make models`` builds: what was built, from which
bytes, by which code.

Two kinds of built file:

* the **fp32 export** (``export.py``), our ONNX export of the official FP16
  safetensors;
* the **attention files** (``attention.py``), one per served release file (the fp32
  export and the author's int8): the same graph with one extra output,
  :data:`ATTENTION_OUTPUT`, the decoder cross-attention of the heads that localise
  the squares (docs/reports/2026-11-25-autocrop-probe.md). Their ``logits`` and
  ``direction`` are bit-identical to the source file's; that check is recorded.

Standard library only, so the workspace suite tests it without PyTorch
(``tests/test_chessq_export.py``); ``export.py`` is the torch half.

A built file's descriptor is provenance rather than a
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


def _sha_matches(models_dir: Path, meta: dict) -> bool:
    """Whether the file a descriptor names holds the bytes it records."""
    artifact = models_dir / meta["artifact"]["filename"]
    return artifact.is_file() and meta["model"]["artifact_sha256"] == fm.sha256_of(artifact)


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


# ------------------------------------------------------- the attention files

#: ``model.format`` of an attention file's descriptor.
ATTENTION_FORMAT = "board-reader-chessq-onnx-attention-v1"

#: The served release files that get an attention copy: the GPU file (our fp32
#: export) and the CPU file (the author's int8).
ATTENTION_SOURCES = (RELEASE, "cq-vit-s-644-v6-seed1-onnx-int8-r1")

#: The extra output's name.
ATTENTION_OUTPUT = "square_attention"

#: The decoder cross-attention heads ``(layer, head)`` the output stacks, in order:
#: the three that localise the squares (median R^2 0.997; every other head is diffuse).
ATTENTION_HEADS = ((1, 0), (2, 0), (3, 0))

#: Patches a side of the 644 px input (14 px patches): the attention's key grid.
PATCHES = 46

#: ``(1, heads, 64 queries, 46, 46)``: per head and square query, its attention
#: over the encoder's patch grid, the CLS token dropped.
ATTENTION_SHAPE = [1, len(ATTENTION_HEADS), 64, PATCHES, PATCHES]


def attention_release(source_release: str) -> str:
    """The attention file's release name: the source's, plus ``-attn``."""
    return f"{source_release}-attn"


@dataclass(frozen=True)
class Parity:
    """The attention file against its source, on the same inputs, on the CPU."""

    samples: int
    #: Whether ``logits`` and ``direction`` were bit-identical on every sample.
    outputs_identical: bool

    @property
    def passed(self) -> bool:
        return self.samples > 0 and self.outputs_identical


def load_attention_source(models_dir: Path, source_release: str) -> tuple[Path, dict]:
    """A source release file and its descriptor; raises unless it holds the recorded bytes."""
    path = models_dir / f"{source_release}.json"
    meta = json.loads(path.read_text())
    artifact = models_dir / meta["artifact"]["filename"]
    if not artifact.is_file():
        raise FileNotFoundError(f"{artifact} is missing; run `make models` from the top")
    if not _sha_matches(models_dir, meta):
        raise ValueError(
            f"{artifact} does not hold the bytes {path.name} records; re-fetch or re-export it"
        )
    return artifact, meta


def build_attention(
    *,
    source: Path,
    source_meta: dict,
    artifact: Path,
    versions: dict[str, str],
    parity: Parity,
) -> dict:
    """The descriptor of the attention file ``artifact`` built from ``source``.

    Refuses a file whose release outputs differ from the source's."""
    if not parity.passed:
        raise ValueError(
            f"{artifact.name}: logits/direction differ from {source.name}; not describing it"
        )
    release = attention_release(source_meta["model"]["release"])
    return {
        "schema_version": 1,
        "model": {
            "release": release,
            "format": ATTENTION_FORMAT,
            "artifact_sha256": fm.sha256_of(artifact),
        },
        "variant": source_meta["variant"],
        "precision": source_meta["precision"],
        "artifact": {"filename": artifact.name, "artifact_bytes": artifact.stat().st_size},
        "preprocessing_version": source_meta["preprocessing_version"],
        "input": source_meta["input"],
        "output": source_meta["output"],
        "viewing_direction": source_meta["viewing_direction"],
        "square_attention": {
            "name": ATTENTION_OUTPUT,
            "dtype": "float32",
            "shape": ATTENTION_SHAPE,
            "heads": [f"L{layer}H{head}" for layer, head in ATTENTION_HEADS],
            "meaning": (
                "softmax cross-attention of each square query (FEN order) over the "
                "46x46 encoder patch grid of the 644x644 input, CLS key dropped"
            ),
        },
        fm.EXPORT_KEY: {
            "tool": "tools/chessq_export/attention.py",
            "source": {
                "release": source_meta["model"]["release"],
                "filename": source.name,
                "artifact_sha256": source_meta["model"]["artifact_sha256"],
                "artifact_bytes": source_meta["artifact"]["artifact_bytes"],
            },
            "versions": versions,
            "verification": asdict(parity),
        },
        "license": source_meta["license"],
    }


def attention_descriptor_path(models_dir: Path, source_release: str) -> Path:
    return models_dir / f"{attention_release(source_release)}.json"


def attention_is_current(models_dir: Path, source_meta: dict) -> bool:
    """Whether the attention file on disk is the one its descriptor describes,
    built from the source bytes ``source_meta`` records."""
    path = attention_descriptor_path(models_dir, source_meta["model"]["release"])
    if not path.is_file():
        return False
    meta = json.loads(path.read_text())
    return (
        meta["model"]["format"] == ATTENTION_FORMAT
        and meta[fm.EXPORT_KEY]["source"]["artifact_sha256"]
        == source_meta["model"]["artifact_sha256"]
        and _sha_matches(models_dir, meta)
    )
