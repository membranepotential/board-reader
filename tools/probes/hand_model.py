"""Write a scratch copy of the ChessQ V6 fp32 ONNX with intermediate tensors as
extra outputs, for the hand-signal probe (docs/reports/2026-11-24-hand-signal-probe.md).

Needs `onnx`, which only the export project has:

    uv run --project tools/chessq_export --frozen python tools/probes/hand_model.py OUT.onnx

Extra outputs, by the names the graph gives them:

* ``query``  — ``/norm_1`` output, (1, 64, 384): each square query's final
  embedding, the input of the shared linear head.
* ``memory`` — ``/norm`` output, (1, 1+2116, 384): the encoder's tokens (CLS, then
  the 46x46 patch grid, row-major).
* ``attn0..attn3`` — each decoder layer's cross-attention softmax, per head.

The original ``logits`` and ``direction`` outputs are kept unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx"

EXTRA = {
    "query": "/norm_1/LayerNormalization_output_0",
    "memory": "/norm/LayerNormalization_output_0",
    **{
        f"attn{i}": f"/decoder/layers.{i}/multihead_attn/Softmax_output_0" for i in range(4)
    },
}


def main(out: Path) -> None:
    if out.resolve().parent == SOURCE.parent:
        raise SystemExit("refusing to write into models/: pass a scratch path")
    model = onnx.load(str(SOURCE))
    produced = {o for n in model.graph.node for o in n.output}
    for name, tensor in EXTRA.items():
        if tensor not in produced:
            raise SystemExit(f"{tensor} not in the graph")
        # An Identity gives the internal tensor a stable public name.
        model.graph.node.append(helper.make_node("Identity", [tensor], [name]))
        model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )
    onnx.save(model, str(out))
    print(f"wrote {out} with outputs {[o.name for o in model.graph.output]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
