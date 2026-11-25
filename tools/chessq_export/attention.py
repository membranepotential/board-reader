"""``make models``, step 3: give each served ChessQ file its square attention as an output.

Auto-frame finds the board in a whole phone frame from ChessQ's own attention
(docs/reports/2026-11-25-autocrop-probe.md): the decoder cross-attention heads
L1H0, L2H0 and L3H0 put each square query's attention on that square, so their
centroids give the board's grid. This writes, beside each served release file (the
fp32 export and the author's int8, :data:`descriptor.ATTENTION_SOURCES`), a copy
whose graph has one more output, ``square_attention`` (1, 3, 64, 46, 46): those
three heads, stacked, the CLS key dropped. Nothing else in the graph changes, so
``logits`` and ``direction`` stay bit-identical; this checks that on the CPU and
records it in the copy's descriptor (``<release>-attn.json``).

The cross-attention softmaxes are found by shape, not by name: the int8 release
renamed every tensor (``v890``...), so the fp32 names
(``/decoder/layers.<L>/multihead_attn/Softmax_output_0``) do not exist there. A
cross-attention softmax is the one whose last two axes are (64 queries, 1+46*46
keys); the decoder's self-attention has 64 keys, the encoder's 2117 queries. They
appear in graph order, layer 0 first.

Idempotent: a copy whose bytes match its descriptor, built from the source bytes
that descriptor records, is left alone (``--force`` rebuilds). A failed check
replaces nothing.

    uv run --project tools/chessq_export --frozen python tools/chessq_export/attention.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import descriptor as desc
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]

QUERIES = 64
KEYS = 1 + desc.PATCHES * desc.PATCHES  # the CLS token, then the patch grid

#: How many inputs the parity check feeds both files.
PARITY_SAMPLES = 3


def cross_attention_softmaxes(model: onnx.ModelProto) -> list[str]:
    """The decoder cross-attention softmax outputs, layer 0 first."""
    inferred = shape_inference.infer_shapes(model)
    shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in [*inferred.graph.value_info, *inferred.graph.output]
    }
    found = [
        n.output[0]
        for n in inferred.graph.node
        if n.op_type == "Softmax" and shapes.get(n.output[0], [])[-2:] == [QUERIES, KEYS]
    ]
    if len(found) != 4:
        raise ValueError(f"expected 4 decoder cross-attention softmaxes, found {found}")
    return found


def with_square_attention(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` with :data:`desc.ATTENTION_OUTPUT` added (in place, and returned)."""
    softmaxes = cross_attention_softmaxes(model)
    graph = model.graph
    name = desc.ATTENTION_OUTPUT

    def const(suffix: str, value: list[int]) -> str:
        tensor = f"{name}/{suffix}"
        graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), tensor))
        return tensor

    heads = []
    for layer, head in desc.ATTENTION_HEADS:
        # A 1-D index keeps the head axis: (1, 1, 64, 2117).
        out = f"{name}/L{layer}H{head}"
        graph.node.append(
            helper.make_node(
                "Gather", [softmaxes[layer], const(f"L{layer}H{head}_index", [head])], [out],
                axis=1,
            )
        )
        heads.append(out)
    graph.node.extend([
        helper.make_node("Concat", heads, [f"{name}/stacked"], axis=1),
        helper.make_node(
            "Slice",
            [f"{name}/stacked", const("starts", [1]), const("ends", [KEYS]), const("axes", [3])],
            [f"{name}/patches"],
        ),
        helper.make_node(
            "Reshape", [f"{name}/patches", const("shape", desc.ATTENTION_SHAPE)], [name]
        ),
    ])
    graph.output.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, desc.ATTENTION_SHAPE)
    )
    return model


def check_parity(source: Path, built: Path) -> desc.Parity:
    """Both files on the same seeded inputs, on the CPU: ``logits`` and ``direction``
    must be bit-identical, and the attention a softmax slice of the expected shape."""
    providers = ["CPUExecutionProvider"]
    old = ort.InferenceSession(str(source), providers=providers)
    new = ort.InferenceSession(str(built), providers=providers)
    rng = np.random.default_rng(0)
    identical = True
    for _ in range(PARITY_SAMPLES):
        x = rng.standard_normal((1, 3, 644, 644)).astype(np.float32)
        want = old.run(["logits", "direction"], {"image": x})
        *got, attention = new.run(["logits", "direction", desc.ATTENTION_OUTPUT], {"image": x})
        identical &= all(np.array_equal(a, b) for a, b in zip(want, got, strict=True))
        if list(attention.shape) != desc.ATTENTION_SHAPE:
            raise ValueError(f"{desc.ATTENTION_OUTPUT} has shape {attention.shape}")
        mass = attention.sum(axis=(-2, -1))  # each query's mass on the patches, CLS left out
        if not (np.all(mass > 0) and np.all(mass <= 1 + 1e-4)):
            raise ValueError(f"{desc.ATTENTION_OUTPUT} is not a softmax slice")
    return desc.Parity(samples=PARITY_SAMPLES, outputs_identical=bool(identical))


def build(models_dir: Path, source_release: str, force: bool) -> int:
    """Build one attention file; 0 on success or when already current."""
    try:
        source, source_meta = desc.load_attention_source(models_dir, source_release)
    except (OSError, ValueError, KeyError) as e:
        print(f"FAILED    {desc.attention_release(source_release)}: {e}", file=sys.stderr)
        return 2
    target = models_dir / f"{desc.attention_release(source_release)}.onnx"
    if not force and desc.attention_is_current(models_dir, source_meta):
        print(f"ok        {target.name} (sha256 verified against its descriptor)")
        return 0
    partial = target.with_name(target.name + ".partial")
    try:
        model = with_square_attention(onnx.load(str(source)))
        onnx.checker.check_model(model)
        onnx.save(model, str(partial))
        parity = check_parity(source, partial)
        if not parity.passed:
            print(
                f"FAILED    {target.name}: logits/direction differ from {source.name}; "
                "nothing replaced",
                file=sys.stderr,
            )
            return 1
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    meta = desc.build_attention(
        source=source,
        source_meta=source_meta,
        artifact=target,
        versions={"onnx": onnx.__version__, "onnxruntime": ort.__version__},
        parity=parity,
    )
    path = desc.attention_descriptor_path(models_dir, source_release)
    before = json.loads(path.read_text())["model"]["artifact_sha256"] if path.is_file() else None
    path.write_text(json.dumps(meta, indent=2) + "\n")
    sha = meta["model"]["artifact_sha256"]
    print(f"built     {target.name} ({meta['artifact']['artifact_bytes']} bytes, "
          f"logits/direction bit-identical to {source.name} on {parity.samples} inputs)")
    if before is not None and before != sha:
        print(f"!! {target.name}'s sha256 {sha} differs from the descriptor's {before}: "
              "the descriptor was rewritten — review it in `git diff`")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models/chessq")
    ap.add_argument("--force", action="store_true", help="rebuild even when current")
    args = ap.parse_args(argv)
    return max(build(args.models_dir, release, args.force) for release in desc.ATTENTION_SOURCES)


if __name__ == "__main__":
    sys.exit(main())
