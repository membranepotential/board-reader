"""``make models``, step 2: export ChessQ Lite V6 to fp32 ONNX and check it.

The served model on a GPU is our own export of the official FP16 safetensors
(docs/reports/2026-11-24-chessq-gpu-formats.md): the author's int8 ONNX has no
CUDA kernels for its integer matmuls, and fp16 is slower than fp32 on a card
without tensor cores. This:

1. checks the safetensors ``tools/fetch_models.py`` fetched against the SHA-256
   its committed release descriptor pins;
2. loads it with the official ``chessqueries`` package (pinned in this
   directory's ``pyproject.toml``), promotes it to fp32 and exports it with
   ``torch.onnx.export`` (opset 17) under the same I/O contract as the int8
   release: ``image`` (1,3,644,644) -> ``logits`` (1,64,13), ``direction`` (1,2);
3. verifies it: on every sample image the export's per-square argmax must equal
   the package's own PyTorch prediction (its CPU fp32 path), fed the package's
   own preprocessing — so the check isolates the export, not our preprocessing;
4. writes ``models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.json`` (descriptor.py)
   and only then moves the export into place. A failed check replaces nothing.

Idempotent: an export whose bytes match its descriptor, from the pinned source,
is left alone (``--force`` re-exports).

    uv run --project tools/chessq_export --frozen python tools/chessq_export/export.py
    ... export.py --samples data/eval/*.jpg   # also verify on real photos

The default samples are drawn here (a few flat boards, deterministic), so the
check runs on a fresh clone with no data; ``--samples`` adds real photos.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import descriptor as desc
import numpy as np
import onnx
import onnxruntime as ort
import timm
import timm.models.vision_transformer as tvit
import torch
from chessqueries.models.chessqueries_model import OrientationMode
from chessqueries.models.predictor import Predictor
from PIL import Image, ImageDraw
from torch import nn

ROOT = Path(__file__).resolve().parents[2]

# timm resamples pos_embed with antialiased bicubic, which has no ONNX op. For a
# fixed input size the result depends only on the weights: compute it eagerly
# once, and the trace embeds it as a constant — the same values as the package's
# eager path.
_orig_resample, _resampled = tvit.resample_abs_pos_embed, {}


def _cached_resample(posemb, new_size, *args, **kwargs):
    key = tuple(int(v) for v in new_size)  # traced shapes are tensors
    if key not in _resampled:
        _resampled[key] = _orig_resample(posemb, new_size, *args, **kwargs).detach()
    return _resampled[key]


tvit.resample_abs_pos_embed = _cached_resample
# nn.MultiheadAttention's fused eval fast path has no ONNX symbolic; same maths without it.
torch.backends.mha.set_fastpath_enabled(False)


class _Exported(nn.Module):
    """The graph we export: image -> (logits, direction), both float32."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if model.orientation_mode is not OrientationMode.AUXILIARY:
            raise ValueError("expected V6 Lite's auxiliary viewing-direction head")
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The casts are no-ops in fp32, but they are graph nodes: they keep this
        # export byte-identical to the file the replay gate measured (cq-v6-fp32).
        memory = self.model.encode(image.to(torch.float32))
        logits = self.model.head(self.model.decode_features(memory)).float()
        return logits, self.model.orientation_predictor(memory).float()


def export(model: nn.Module, out: Path) -> None:
    wrapped = _Exported(model).eval()
    dummy = torch.randn(1, 3, 644, 644)  # only its shape enters the graph
    with torch.no_grad():
        wrapped(dummy)  # the eager pass fills the pos_embed cache
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(out),
            input_names=["image"],
            output_names=["logits", "direction"],
            opset_version=desc.OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(out)))


#: Placements for the drawn samples: a start position, a middlegame, a sparse ending.
_SAMPLE_PLACEMENTS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2QK2R",
    "8/5k2/8/3K4/8/8/2R5/8",
)


def drawn_samples(out_dir: Path) -> list[Path]:
    """A few flat boards with lettered discs, drawn deterministically.

    Not photographs — the model reads them worse than real boards — but the check
    is export-vs-package parity on the same input, not accuracy, and these exist
    on a fresh clone.
    """
    paths = []
    for i, placement in enumerate(_SAMPLE_PLACEMENTS):
        side, margin = 640, 40
        cell = (side - 2 * margin) // 8
        image = Image.new("RGB", (side + 160, side), (90, 70, 50))
        draw = ImageDraw.Draw(image)
        for r, rank in enumerate(_expand(placement)):
            for c, piece in enumerate(rank):
                x, y = margin + c * cell + 80, margin + r * cell
                draw.rectangle(
                    (x, y, x + cell, y + cell),
                    fill=(238, 220, 180) if (r + c) % 2 == 0 else (150, 110, 70),
                )
                if piece != ".":
                    ink = (245, 245, 240) if piece.isupper() else (25, 25, 25)
                    draw.ellipse((x + 8, y + 8, x + cell - 8, y + cell - 8), fill=ink)
                    draw.text((x + cell // 2 - 4, y + cell // 2 - 6), piece.upper(),
                              fill=(200, 30, 30))
        path = out_dir / f"drawn-{i}.png"
        image.save(path)
        paths.append(path)
    return paths


def _expand(placement: str) -> list[str]:
    return ["".join("." * int(ch) if ch.isdigit() else ch for ch in rank)
            for rank in placement.split("/")]


def verify(predictor: Predictor, onnx_path: Path, samples: list[Path]) -> desc.Verification:
    """The export's per-square argmax against the package's own prediction."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    identical = squares = 0
    max_diff = 0.0
    with torch.no_grad():
        for path in samples:
            x = predictor.prepare_image(path)[None]
            want = predictor.model.recognize(x).labels.reshape(-1).numpy()
            ref_logits = predictor.model(x).float().numpy()
            got_logits = session.run(["logits"], {"image": x.numpy()})[0]
            got = got_logits.reshape(-1, got_logits.shape[-1]).argmax(-1)
            identical += int((got == want).sum())
            squares += want.size
            max_diff = max(max_diff, float(np.abs(got_logits - ref_logits).max()))
            print(f"  {path.name}: {int((got == want).sum())}/{want.size} squares agree")
    return desc.Verification(
        samples=len(samples),
        squares=squares,
        squares_identical=identical,
        max_abs_dlogit=float(f"{max_diff:.3g}"),
    )


def _package() -> desc.Package:
    direct = json.loads(
        importlib.metadata.distribution("chessqueries").read_text("direct_url.json") or "{}"
    )
    return desc.Package(
        name="chessqueries",
        url=direct.get("url", ""),
        commit=direct.get("vcs_info", {}).get("commit_id", ""),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models/chessq")
    ap.add_argument("--samples", type=Path, nargs="*", default=[],
                    help="extra images to verify on, beside the drawn ones")
    ap.add_argument("--force", action="store_true", help="re-export even when current")
    args = ap.parse_args(argv)

    try:
        source, source_meta = desc.load_source(args.models_dir)
    except (OSError, ValueError, KeyError) as e:
        print(f"chessq_export: {e}", file=sys.stderr)
        return 2
    target = desc.artifact_path(args.models_dir)
    if not args.force and desc.is_current(args.models_dir, source):
        print(f"ok        {target.name} (export sha256 verified against its descriptor)")
        return 0

    committed = desc.descriptor_path(args.models_dir)
    before = json.loads(committed.read_text())["model"]["artifact_sha256"] if committed.is_file() else None
    predictor = Predictor.from_checkpoint(source.target, device="cpu")  # fp32 on CPU
    partial = target.with_name(target.name + ".partial")
    try:
        export(predictor.model, partial)
        with tempfile.TemporaryDirectory() as tmp:
            samples = drawn_samples(Path(tmp)) + list(args.samples)
            verification = verify(predictor, partial, samples)
        if not verification.passed:
            print(
                f"FAILED    {target.name}: the export disagrees with the package on "
                f"{verification.squares - verification.squares_identical} of "
                f"{verification.squares} squares; nothing replaced",
                file=sys.stderr,
            )
            return 1
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    descriptor = desc.build(
        source=source,
        source_meta=source_meta,
        artifact=target,
        package=_package(),
        versions={
            "torch": torch.__version__,
            "timm": timm.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        verification=verification,
    )
    desc.write(args.models_dir, descriptor)
    sha = descriptor["model"]["artifact_sha256"]
    print(f"exported  {target.name} ({descriptor['artifact']['artifact_bytes']} bytes, "
          f"{verification.squares_identical}/{verification.squares} squares agree with "
          f"the package, max |dlogit| {verification.max_abs_dlogit})")
    if before is not None and before != sha:
        print(f"!! the export's sha256 {sha} differs from the descriptor's {before}: "
              "the descriptor was rewritten — review it in `git diff`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
