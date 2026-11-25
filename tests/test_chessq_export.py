"""``tools/chessq_export/descriptor.py``: the fp32 export's provenance, without torch.

The export itself (``export.py``) needs PyTorch and runs in its own uv project, so
it is exercised by ``make models``, not here. What is tested here is everything
that decides whether an export is trusted and described: the source SHA-256
check, the refusal to describe a failed parity check, and the idempotence test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools/chessq_export/descriptor.py"


def _load():
    spec = importlib.util.spec_from_file_location("chessq_export_descriptor", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


desc = _load()
fm = desc.fm
WEIGHTS = b"fp16 weights" * 50
REV = "a0393fdd5a7dfafd1a6d91e0d9b7c50aeacff025"
PASSED = desc.Verification(samples=3, squares=192, squares_identical=192, max_abs_dlogit=3e-5)
PACKAGE = desc.Package(name="chessqueries", url="https://github.com/JSeytre/chessqueries",
                       commit="f54c07b50ae27ec5516bc80efdfb4e0600adc007")


def _models(tmp_path: Path, weights: bytes = WEIGHTS) -> Path:
    """A models dir holding the committed source descriptor, re-pinned to ``WEIGHTS``."""
    models = tmp_path / "models/chessq"
    models.mkdir(parents=True)
    meta = json.loads((_ROOT / "models/chessq" / desc.SOURCE_DESCRIPTOR).read_text())
    meta["model"]["artifact_sha256"] = hashlib.sha256(WEIGHTS).hexdigest()
    meta["artifact"]["artifact_bytes"] = len(WEIGHTS)
    (models / desc.SOURCE_DESCRIPTOR).write_text(json.dumps(meta))
    (models / meta["artifact"]["filename"]).write_bytes(weights)
    return models


def _describe(models: Path, verification=PASSED) -> dict:
    source, meta = desc.load_source(models)
    return desc.build(source=source, source_meta=meta, artifact=desc.artifact_path(models),
                      package=PACKAGE, versions={"torch": "2.8.0"}, verification=verification)


def test_the_source_must_be_the_pinned_bytes(tmp_path):
    with pytest.raises(ValueError, match="re-fetch"):
        desc.load_source(_models(tmp_path, weights=b"tampered" * 75))


def test_a_missing_source_names_the_fetch_step(tmp_path):
    models = _models(tmp_path)
    for weights in models.glob("*.safetensors"):
        weights.unlink()
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        desc.load_source(models)


def test_the_descriptor_records_source_package_and_export(tmp_path):
    models = _models(tmp_path)
    desc.artifact_path(models).write_bytes(b"onnx bytes")
    meta = _describe(models)
    assert meta["model"]["artifact_sha256"] == hashlib.sha256(b"onnx bytes").hexdigest()
    assert meta["artifact"] == {"filename": f"{desc.RELEASE}.onnx", "artifact_bytes": 10}
    export = meta[fm.EXPORT_KEY]
    assert export["source"]["artifact_sha256"] == hashlib.sha256(WEIGHTS).hexdigest()
    assert f"/resolve/{REV}/" in export["source"]["url"]
    assert export["package"]["commit"] == PACKAGE.commit
    assert export["verification"]["squares_identical"] == 192
    assert fm.is_export(meta)  # so fetch_models never tries to download it


def test_a_failed_parity_check_is_never_described(tmp_path):
    models = _models(tmp_path)
    desc.artifact_path(models).write_bytes(b"onnx bytes")
    broken = desc.Verification(samples=3, squares=192, squares_identical=191, max_abs_dlogit=2.0)
    with pytest.raises(ValueError, match="1 of 192 squares"):
        _describe(models, broken)


def test_current_only_while_the_bytes_and_the_source_match(tmp_path):
    models = _models(tmp_path)
    artifact = desc.artifact_path(models)
    artifact.write_bytes(b"onnx bytes")
    source, _ = desc.load_source(models)
    assert not desc.is_current(models, source)  # no descriptor yet
    desc.write(models, _describe(models))
    assert desc.is_current(models, source)
    artifact.write_bytes(b"other bytes")
    assert not desc.is_current(models, source)


def test_the_committed_descriptor_describes_the_gated_export():
    """The fp32 file the replay gate measured (cq-v6-fp32), from the pinned V6 source."""
    meta = json.loads(desc.descriptor_path(_ROOT / "models/chessq").read_text())
    assert meta["model"]["artifact_sha256"] == (
        "b9ebec59cafc7d706f7403baff93642daa3a0daa61796b8e231fa2caba659f2b"
    )
    source = fm.Pin.load(_ROOT / "models/chessq" / desc.SOURCE_DESCRIPTOR)
    assert meta[fm.EXPORT_KEY]["source"]["artifact_sha256"] == source.sha256
    assert meta[fm.EXPORT_KEY]["verification"]["squares_identical"] == (
        meta[fm.EXPORT_KEY]["verification"]["squares"]
    )


# --------------------------------------------------------------------------- #
# The attention files (attention.py): the served copies with square attention
# --------------------------------------------------------------------------- #

#: The int8 release's descriptor, as the source of an attention file.
INT8 = "cq-vit-s-644-v6-seed1-onnx-int8-r1"


def _int8_source(tmp_path: Path, payload: bytes = b"int8 onnx" * 40) -> Path:
    """A models dir holding the committed int8 descriptor, re-pinned to ``payload``."""
    models = tmp_path / "models/chessq"
    models.mkdir(parents=True)
    meta = json.loads((_ROOT / "models/chessq" / f"{INT8}.json").read_text())
    meta["model"]["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
    meta["artifact"]["artifact_bytes"] = len(payload)
    (models / f"{INT8}.json").write_text(json.dumps(meta))
    (models / meta["artifact"]["filename"]).write_bytes(payload)
    return models


IDENTICAL = desc.Parity(samples=3, outputs_identical=True)


def _attention_meta(models: Path, parity=IDENTICAL) -> dict:
    source, meta = desc.load_attention_source(models, INT8)
    artifact = models / f"{desc.attention_release(INT8)}.onnx"
    artifact.write_bytes(b"attn bytes")
    return desc.build_attention(source=source, source_meta=meta, artifact=artifact,
                                versions={"onnx": "1.23.2"}, parity=parity)


def test_an_attention_file_describes_its_source_and_its_extra_output(tmp_path):
    models = _int8_source(tmp_path)
    meta = _attention_meta(models)
    assert meta["model"]["release"] == f"{INT8}-attn"
    assert meta["artifact"]["filename"] == f"{INT8}-attn.onnx"
    assert meta["square_attention"]["shape"] == [1, 3, 64, 46, 46]
    assert meta["square_attention"]["heads"] == ["L1H0", "L2H0", "L3H0"]
    source = meta[fm.EXPORT_KEY]["source"]
    assert source["release"] == INT8
    assert source["artifact_sha256"] == hashlib.sha256(b"int8 onnx" * 40).hexdigest()
    assert fm.is_export(meta)  # fetch_models never tries to download it


def test_an_attention_file_with_changed_outputs_is_never_described(tmp_path):
    with pytest.raises(ValueError, match="differ"):
        _attention_meta(_int8_source(tmp_path), desc.Parity(samples=3, outputs_identical=False))


def test_an_attention_source_must_hold_its_recorded_bytes(tmp_path):
    models = _int8_source(tmp_path)
    (models / f"{INT8}.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not hold the bytes"):
        desc.load_attention_source(models, INT8)


def test_an_attention_file_is_current_only_for_its_bytes_and_source(tmp_path):
    models = _int8_source(tmp_path)
    _, source_meta = desc.load_attention_source(models, INT8)
    assert not desc.attention_is_current(models, source_meta)  # no descriptor yet
    desc.attention_descriptor_path(models, INT8).write_text(json.dumps(_attention_meta(models)))
    assert desc.attention_is_current(models, source_meta)
    (models / f"{INT8}-attn.onnx").write_bytes(b"other bytes")
    assert not desc.attention_is_current(models, source_meta)


def test_the_committed_attention_descriptors_pin_the_served_files():
    """Both served files, built from the committed release files' recorded bytes,
    with logits and direction bit-identical to them."""
    models = _ROOT / "models/chessq"
    pinned = {
        desc.RELEASE: "eef9e49b52a1577b0f74702a8451b0098c6a94ad795571ade304f662cef3309a",
        INT8: "ed3dbd48242f9c5a8f48d3631d05394282677249dcd19f1d55030592a71b250b",
    }
    assert set(desc.ATTENTION_SOURCES) == set(pinned)
    for release, sha in pinned.items():
        meta = json.loads(desc.attention_descriptor_path(models, release).read_text())
        source = json.loads((models / f"{release}.json").read_text())
        assert meta["model"]["artifact_sha256"] == sha
        assert meta[fm.EXPORT_KEY]["source"]["artifact_sha256"] == source["model"]["artifact_sha256"]
        assert meta[fm.EXPORT_KEY]["verification"] == {"samples": 3, "outputs_identical": True}
