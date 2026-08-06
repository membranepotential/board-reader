"""Tests for the embedder interface + lightweight embedder (WP-B2)."""

from __future__ import annotations

from pathlib import Path

import sys
import types

import numpy as np
import pytest

from board_reader_core.sim import render_board
from board_reader_inference.embed import (
    DEFAULT_DINOV2_ONNX,
    DinoV2Embedder,
    Embedder,
    LiteEmbedder,
    l2_normalize,
    make_embedder,
)
from board_reader_inference.squares import crop_squares

_HAS_DINOV2 = Path(DEFAULT_DINOV2_ONNX).exists()
requires_dinov2 = pytest.mark.skipif(
    not _HAS_DINOV2,
    reason=f"DINOv2 ONNX not present at {DEFAULT_DINOV2_ONNX} (run tools/export_dinov2_onnx.py)",
)


def test_downsample_embedder_satisfies_protocol():
    assert isinstance(LiteEmbedder(), Embedder)


def test_embed_shape_and_l2_normalised():
    emb = LiteEmbedder(side=16)
    crops = crop_squares(render_board())
    out = emb.embed(crops)
    assert out.shape == (64, emb.dim)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_empty_returns_well_shaped_array():
    emb = LiteEmbedder(side=8)
    out = emb.embed([])
    assert out.shape == (0, emb.dim)


def test_distinct_glyphs_get_distinct_embeddings():
    # An occupied square and an empty square embed differently.
    crops = crop_squares(render_board())
    out = LiteEmbedder().embed(crops)
    a8, a4 = out[0], out[32]  # rook square vs empty middle square
    assert float(a8 @ a4) < 0.999  # not identical directions


def test_l2_normalize_handles_zero_rows():
    x = np.zeros((2, 4), dtype=np.float32)
    out = l2_normalize(x, axis=1)
    assert np.isfinite(out).all()


def test_make_embedder_lite():
    assert isinstance(make_embedder("lite"), LiteEmbedder)


def test_make_embedder_rejects_unknown():
    with pytest.raises(ValueError):
        make_embedder("nope")


@requires_dinov2
def test_dinov2_embedder_emits_384d_normalised_embeddings():
    emb = DinoV2Embedder()
    assert isinstance(emb, Embedder)
    assert emb.dim == 384
    out = emb.embed(crop_squares(render_board(), size=None))
    assert out.shape == (64, 384)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


@requires_dinov2
def test_make_embedder_dinov2():
    assert isinstance(make_embedder("dinov2"), DinoV2Embedder)


def test_embedder_resolves_a_symbolic_output_width_by_probing(monkeypatch, tmp_path):
    """An ONNX whose output width is a symbolic name, not an int, must still load.

    torch's legacy exporter names a reduce output's width symbolically
    (``'ReduceMeanembedding_dim_1'``); reading it as an int raised ValueError and
    made the central-column DINOv2 export unloadable despite being correct.
    """
    from board_reader_inference import embed as embed_mod

    class _FakeOutput:
        shape = ["batch", "ReduceMeanembedding_dim_1"]

    class _FakeInput:
        name = "pixel_values"

    class _FakeSession:
        def __init__(self, *a, **kw):
            pass

        def get_inputs(self):
            return [_FakeInput()]

        def get_outputs(self):
            return [_FakeOutput()]

        def run(self, _names, feed):
            n = feed["pixel_values"].shape[0]
            return [np.ones((n, 384), dtype=np.float32)]

    model = tmp_path / "sym.onnx"
    model.write_bytes(b"stub")
    fake_ort = types.SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    emb = embed_mod.DinoV2Embedder(model, providers=["CPUExecutionProvider"])
    assert emb.dim == 384
    assert emb.embed([np.zeros((8, 8, 3), dtype=np.uint8)] * 3).shape == (3, 384)
