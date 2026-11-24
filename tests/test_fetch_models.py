"""``tools/fetch_models.py`` (``make models``): pinned, verified, never half-written."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "fetch_models.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_models", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fm = _load()
PAYLOAD = b"weights" * 100
REV = "a0393fdd5a7dfafd1a6d91e0d9b7c50aeacff025"


def _descriptor(root: Path, *, url: str | None = None, sha: str | None = None) -> Path:
    path = root / "models/chessq/cq.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": {"artifact_sha256": sha or hashlib.sha256(PAYLOAD).hexdigest()},
        "artifact": {
            "filename": "cq.onnx",
            "artifact_bytes": len(PAYLOAD),
            "url": url or f"https://huggingface.co/x/y/resolve/{REV}/cq.onnx",
            "distribution_revision": REV,
        },
    }))
    return path


@pytest.fixture
def served(monkeypatch):
    """Serve ``PAYLOAD`` (or whatever a test puts in the list) for any URL."""
    body = [PAYLOAD]
    asked: list[str] = []

    def urlopen(url):
        asked.append(url)
        return io.BytesIO(body[0])

    monkeypatch.setattr(fm.urllib.request, "urlopen", urlopen)
    return body, asked


def test_fetches_verifies_and_then_leaves_a_good_file_alone(tmp_path, served):
    _, asked = served
    _descriptor(tmp_path)
    assert fm.main(tmp_path) == 0
    assert (tmp_path / "models/chessq/cq.onnx").read_bytes() == PAYLOAD
    assert fm.main(tmp_path) == 0
    assert len(asked) == 1  # the second run checked the SHA-256 and fetched nothing


def test_a_tampered_download_never_replaces_anything(tmp_path, served):
    body, _ = served
    body[0] = b"evil" * 175  # same size as PAYLOAD, different bytes
    _descriptor(tmp_path)
    assert fm.main(tmp_path) == 1
    assert list((tmp_path / "models/chessq").iterdir()) == [tmp_path / "models/chessq/cq.json"]


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/x/y/resolve/main/cq.onnx",  # a moving branch
        f"http://huggingface.co/x/y/resolve/{REV}/cq.onnx",  # not https
    ],
)
def test_an_unpinned_url_is_refused(tmp_path, url):
    with pytest.raises(ValueError, match="pinned revision"):
        fm.Pin.load(_descriptor(tmp_path, url=url))


def test_the_committed_descriptor_pins_the_v6_release():
    [pin] = [fm.Pin.load(p) for p in sorted(fm.ROOT.glob(fm.DESCRIPTORS))]
    assert pin.filename == "cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx"
    assert pin.sha256 == "b5469a816bbc0c0574c4c02da6aff95286b1e3c90cb233e4e3d8750e1eed6e51"
