"""ONNX Runtime execution-provider selection, shared by every ONNX session here.

Why this exists: the PyPI ``onnxruntime-gpu`` wheels are built against CUDA 12,
but a machine's *system* CUDA may be newer (13.x ships no ``libcublasLt.so.12``).
The matching CUDA 12 libraries are already installed in the venv as the
``nvidia-*-cu12`` wheels — they are simply not on the dynamic loader path, so the
CUDA EP fails to load and ORT silently falls back to CPU. ``ort.preload_dlls()``
is ONNX Runtime's own API for resolving those wheels, and calling it before the
first session removes any need for ``LD_LIBRARY_PATH`` juggling at the launcher.
"""

from __future__ import annotations

CPU_ONLY: list[str] = ["CPUExecutionProvider"]


def cuda_providers() -> list[str]:
    """Preferred provider list: CUDA (with CPU fallback) when it is usable.

    Never raises: an unusable GPU degrades to CPU-only, which is slower but
    correct. Safe to call repeatedly — ``preload_dlls`` is idempotent.
    """
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return list(CPU_ONLY)
    try:
        ort.preload_dlls()
    except Exception:
        # An ORT build without preload_dlls, or libs it cannot find. The system
        # loader may still resolve them, so try CUDA anyway — the session
        # constructor downgrades to CPU on its own if it cannot initialise.
        pass
    return ["CUDAExecutionProvider", *CPU_ONLY]
