"""ONNX Runtime execution-provider selection, shared by every ONNX session here.

Why this exists, part one: the PyPI ``onnxruntime-gpu`` wheels are built against
CUDA 12, but a machine's *system* CUDA may be newer (13.x ships no
``libcublasLt.so.12``). The matching CUDA 12 libraries are already installed in
the venv as the ``nvidia-*-cu12`` wheels — they are simply not on the dynamic
loader path, so the CUDA EP fails to load and ORT silently falls back to CPU.
``ort.preload_dlls()`` is ONNX Runtime's own API for resolving those wheels, and
calling it before the first session removes any need for ``LD_LIBRARY_PATH``
juggling at the launcher.

Why this exists, part two: **the card is usually also driving a screen.** On a
laptop the display pipeline and this node share one GPU, and the display is the
side that cannot degrade — it has no CPU fallback. Left at ORT's defaults the
arena takes what it likes and the display loses; :func:`cuda_options` is what
keeps the two apart. See the class comment on ``BR_GPU_MEM_LIMIT_MB`` below.
"""

from __future__ import annotations

import os

#: A provider as ``InferenceSession(providers=...)`` accepts it: a bare name, or
#: a name paired with that provider's options.
Provider = str | tuple[str, dict[str, str | int]]

CPU_ONLY: list[Provider] = ["CPUExecutionProvider"]

#: Per-session ceiling on the CUDA arena, in MiB (``BR_GPU_MEM_LIMIT_MB``).
#:
#: The limit is **per session**, and serving builds three of them in one process
#: (the cascade's two models plus the DINOv2 embedder — each gets its own arena;
#: ORT only shares one across sessions via the env-allocator API, which we do not
#: use). So the default is a band, and 1024 MiB is the round number inside it:
#:
#: * not below ~896 MiB — measured 2026-08-13, DINOv2 ViT-S/14 on a 64-crop board
#:   fails at 768 and passes at 896 (a batch-64 attention matrix is 101 MB on its
#:   own). The two ResNet18s at 224px need less, but not much less.
#: * not above ~1365 MiB — three of them must still leave ~1 GiB of a 4 GiB card
#:   for everything else, which is what an X server needs for a 4K reverse-PRIME
#:   surface and its composition pipeline.
#:
#: The ceiling is not a reservation. Measured peak for the real three-session
#: shape at this default is ~1.6 GiB, well under 3x. Its job is to make a runaway
#: extension fail as a *catchable* ORT allocation error instead of silently
#: starving the display — 2026-08-13, when a calibration batch took the card and
#: X could no longer allocate its scanout surfaces, blacking out an external
#: monitor twice ("Failed to allocate reverse prime surface: out of memory").
#:
#: ``BR_GPU_MEM_LIMIT_MB=0`` removes the cap, for a card that drives no display.
_DEFAULT_GPU_MEM_LIMIT_MB = 1024


def _gpu_mem_limit_bytes() -> int | None:
    """The configured arena ceiling in bytes, or ``None`` for uncapped.

    An unparseable value is a typo in an operator's environment, not a reason to
    hand the display GPU an unbounded arena, so it falls back to the default.
    """
    raw = os.environ.get("BR_GPU_MEM_LIMIT_MB")
    if raw is None:
        mb = _DEFAULT_GPU_MEM_LIMIT_MB
    else:
        try:
            mb = int(raw)
        except ValueError:
            print(
                f"BR_GPU_MEM_LIMIT_MB={raw!r} is not an integer; "
                f"using the default {_DEFAULT_GPU_MEM_LIMIT_MB} MiB"
            )
            mb = _DEFAULT_GPU_MEM_LIMIT_MB
    if mb <= 0:
        return None
    return mb * 1024 * 1024


def cuda_options() -> dict[str, str | int]:
    """CUDA EP options that keep a shared display GPU usable.

    Three departures from ORT's defaults, all of them about co-existing with
    whatever else holds VRAM rather than about throughput:

    ``gpu_mem_limit``
        Ceiling on the arena — see ``BR_GPU_MEM_LIMIT_MB``. Absent when the cap
        is disabled, since ORT reads *no key* as "no limit".
    ``arena_extend_strategy=kSameAsRequested``
        The default ``kNextPowerOfTwo`` rounds every extension up to the next
        power of two, so a batch that needs 600 MB takes 1 GiB and never gives
        it back (BFCArena does not return memory to the driver).
    ``cudnn_conv_algo_search=HEURISTIC``
        The default ``EXHAUSTIVE`` benchmarks every cuDNN convolution algorithm
        on first use, allocating large transient workspaces to do it. On models
        this small the chosen algorithm is the same one; the spike is not.
    """
    options: dict[str, str | int] = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    limit = _gpu_mem_limit_bytes()
    if limit is not None:
        options["gpu_mem_limit"] = limit
    return options


def cuda_providers() -> list[Provider]:
    """Preferred provider list: CUDA (with CPU fallback) when it is usable.

    Never raises: an unusable GPU degrades to CPU-only, which is slower but
    correct. Safe to call repeatedly — ``preload_dlls`` is idempotent.
    """
    import onnxruntime as ort

    # An EMPTY CUDA_VISIBLE_DEVICES is an operator saying "not this card" (what
    # `make dev-cpu` and `make test` do). ORT would still *try* — the provider is
    # in the build regardless of what is plugged in — fail with "CUDA failure
    # 100: no CUDA-capable device is detected", fall back to CPU anyway, and
    # print that block once per session. Correct, but it reads as a broken
    # install rather than as the choice it is. Unset still means "all devices".
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return list(CPU_ONLY)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return list(CPU_ONLY)
    try:
        ort.preload_dlls()
    except Exception:
        # An ORT build without preload_dlls, or libs it cannot find. The system
        # loader may still resolve them, so try CUDA anyway — the session
        # constructor downgrades to CPU on its own if it cannot initialise.
        pass
    return [("CUDAExecutionProvider", cuda_options()), *CPU_ONLY]
