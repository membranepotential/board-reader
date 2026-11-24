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

## Update 2026-11-24: shipped, and the replay gate

The fp32 export is now what a GPU serves (`make dev`, `make gate`); the int8 file
stays the CPU model (`make dev-cpu`, `compose.yaml`, which declares no GPU device).
`make models` reproduces the export: `tools/chessq_export` (its own uv project, the
same pinned package) exports byte-identically to the file measured here
(SHA-256 `b9ebec59cafc7d706f7403baff93642daa3a0daa61796b8e231fa2caba659f2b`), and its per-square argmax equals the package's own CPU fp32
prediction on 3 drawn boards (192/192 squares, max |Δlogit| 2.7e-5) and on 5 real
eval photos (320/320, max 4.8e-5).

**CPU, measured the same day** (onnxruntime CPU EP, 12 threads, machine loaded):
int8 663 ms, fp32 986 ms per read. So int8 stays the CPU choice.

**Replay gate** (`make gate GATE_NAME=cq-v6-fp32`, burst-end, settle 2.5 s / 2 reads):

| take | raw per-sq | raw perfect | prior perfect | settled perfect | wall (int8 → fp32) |
|---|---|---|---|---|---|
| 20260813-193308 | .9924 | 25/31 | 25/31 | 31/31 | 21.2 → 11.0 s |
| 20260827-175539 | .9948 | 11/12 | 11/12 | 8/12 | 6.9 → 3.8 s |
| 20260828-111421 | .9985 | 20/21 | 20/21 | 18/21 | 13.8 → 6.2 s |
| 20260827-181802 | .9996 | 36/37 | 36/37 | 27/37 | 22.6 → 10.0 s |
| **total** | .9966 | 92/101 | 92/101 | 84/101 | 64.5 → 31.1 s |

Every cell is identical to `cq-v6` (int8). Not just the totals: all 133 published
placements and all 133 settled boards are the same strings. The wall time halves
(it includes JPEG decode and the restamp of 08-13's 748 frames, so the per-read
gain is larger than 2×).

One caveat on provenance: the int8 descriptor names a different
`source_checkpoint_sha256` (`48d0366d492f6949dc551921863a6239e56af987bda39e052518ebf0f39019fb`) than the safetensors
(`9b007ccea09348ee011d13f95f7a9d33dfdf92aca4c9557bfbefd17a52fc0ba5`). They
may be two serialisations of one training run, or two checkpoints; the identical
gate reads say they behave as one model on our takes, but that is not proof.
