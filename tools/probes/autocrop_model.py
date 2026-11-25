"""Write a scratch copy of a ChessQ V6 ONNX release file (fp32 or int8) whose
decoder cross-attention is an extra output, for the auto-frame probe
(docs/reports/2026-11-25-autocrop-probe.md).

Needs `onnx`, which only the export project has:

    uv run --project tools/chessq_export --frozen python tools/probes/autocrop_model.py \
        SOURCE.onnx OUT.onnx [LAYER:HEAD ...]

With no ``LAYER:HEAD`` it adds every head of every decoder layer as
``attn<L>`` (1, 8, 64, 1+46*46), for choosing the head. With ``2:0 3:0`` it adds
only those heads, sliced inside the graph, as ``attn<L>h<H>`` (1, 64, 1+46*46):
the shape a shipped export would carry (0.5 MB a head instead of 4.3 MB a layer).

The cross-attention softmaxes are found by shape, not by name: the int8 release
renamed every tensor (``v890``...), so the fp32 names
(``/decoder/layers.<L>/multihead_attn/Softmax_output_0``) do not exist there. A
cross-attention softmax is the one whose last axis has 1+46*46 keys (the encoder
tokens); the decoder's self-attention has 64 and the encoder's 2117 queries too,
so the query axis (64) tells them apart. They appear in graph order, layer 0 first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
TOKENS = 1 + 46 * 46
QUERIES = 64


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
        if n.op_type == "Softmax" and shapes.get(n.output[0], [])[-2:] == [QUERIES, TOKENS]
    ]
    if len(found) != 4:
        raise SystemExit(f"expected 4 cross-attention softmaxes, found {found}")
    return found


def main(source: Path, out: Path, picks: list[tuple[int, int]]) -> None:
    if out.resolve().parent == (ROOT / "models/chessq").resolve():
        raise SystemExit("refusing to write into models/: pass a scratch path")
    model = onnx.load(str(source))
    layers = cross_attention_softmaxes(model)
    graph = model.graph
    if not picks:
        for layer, tensor in enumerate(layers):
            graph.node.append(helper.make_node("Identity", [tensor], [f"attn{layer}"]))
            graph.output.append(
                helper.make_tensor_value_info(f"attn{layer}", TensorProto.FLOAT, None)
            )
    for layer, head in picks:
        name = f"attn{layer}h{head}"
        index = f"{name}_index"
        graph.initializer.append(numpy_helper.from_array(np.array(head, np.int64), index))
        # Gather on the head axis with a scalar index drops that axis: (1, 64, 2117).
        graph.node.append(
            helper.make_node("Gather", [layers[layer], index], [name], axis=1)
        )
        graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    onnx.save(model, str(out))
    print(
        f"wrote {out}: cross-attention {layers}; outputs {[o.name for o in graph.output]}"
    )


if __name__ == "__main__":
    src, dst, *rest = sys.argv[1:]
    main(
        Path(src),
        Path(dst),
        [(int(a), int(b)) for a, b in (p.split(":") for p in rest)],
    )
