# ChessQ V6 on the GPU: which format to serve (2026-11-24)

> Experiment dir: `scratch/experiments/chessq-fp16/` (gitignored). Cut short on purpose
> once int8 and float were seen to read the same; ViT-B accuracy and a matched int8
> rerun were not finished.

## Question

The served model is the author's **int8 ONNX** (`cq-vit-s-644-v6-seed1-onnx-int8-r1`),
which the model card targets at "ONNX Runtime, including browser WASM". Its matmuls are
`MatMulInteger` + `DynamicQuantizeLinear`, which have no CUDA kernels in onnxruntime,
so on CUDA it runs at ~0.4–0.7 s/frame with 222 memcpy nodes. The card's GPU path
is the **FP16 safetensors** under PyTorch. Is there a faster format with the same reads?

## Method

The user approved the official `chessqueries` package, installed in an isolated uv
project pinned to the model card's commit. It loaded the official FP16 safetensors
(V6 Lite, and ViT-B from `joelseytre/chessqueries`) and exported them with
`torch.onnx.export` (opset 17) to fp32 and fp16 ONNX, keeping the same I/O contract
as the int8 file (`image` 1×3×644×644 → `logits` 1×64×13).

## Latency (GTX 1650, 4 GB; machine shared with other jobs, so absolute values read high)

| model | runtime | precision | ms/frame |
|---|---|---|---|
| V6 int8 ONNX | onnxruntime CPU | int8 | ~600–1400 |
| V6 int8 ONNX | onnxruntime CUDA | int8 | ~430–720 (CPU fallbacks) |
| V6 ONNX | onnxruntime CUDA | fp16 | 547 |
| **V6 ONNX** | **onnxruntime CUDA** | **fp32** | **205** (818 MiB peak) |
| ViT-B ONNX | onnxruntime CUDA | fp32 / fp16 | 582 / 1611 |
| V6, the package itself | PyTorch CUDA | fp32 / fp16 | 194 / 463 end to end |
| ViT-B, the package itself | PyTorch CUDA | fp32 / fp16 | 557 / 1371 |

The GTX 1650 (TU117) has no tensor cores, so **fp16 is slower than fp32** in both runtimes.

## Accuracy (probe set: `cq_bench.py`, whole frame)

| V6 variant | held-out stills | curated stills | recording frames per-sq | perfect frames |
|---|---|---|---|---|
| int8 ONNX (truth before the 11-24 fixes) | 11/11 | 9/10 | 99.19 % | 324/413 |
| fp32 ONNX (fixed truth) | 11/11 | 9/10 | 99.40 % | 364/446 |
| fp16 ONNX (fixed truth) | 11/11 | 9/10 | 99.40 % | 364/446 |

The recording sets differ: the truth fixes added frames. So the int8 row is indicative
only. The like-for-like check is the replay gate.

## Decision

Serve **V6 fp32 ONNX on CUDA** through the existing onnxruntime recognizer. It is
~3× faster than int8, reads at least as well, and needs no PyTorch at runtime. The
package as runtime is equally fast, but adds PyTorch and CUDA wheels to the inference
image for no speed gain. The fp32 file is our own export, so `make models` gains an
export step: the pinned package in an isolated uv project, from the official
SHA-256-checked safetensors. The switch must pass the 4-take replay gate against `cq-v6`.
