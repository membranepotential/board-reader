"""Provider selection must never raise, never drop the CPU fallback, and never
hand an unbounded arena to a GPU that is also driving a display."""

from __future__ import annotations

import pytest

from board_reader_inference.onnxrt import (
    _DEFAULT_GPU_MEM_LIMIT_MB,
    CPU_ONLY,
    Provider,
    cuda_options,
    cuda_providers,
)

ort = pytest.importorskip("onnxruntime")

_MIB = 1024 * 1024


def _names(providers: list[Provider]) -> list[str]:
    """Provider names, with the options stripped off the configured ones."""
    return [p if isinstance(p, str) else p[0] for p in providers]


def test_cpu_fallback_always_last() -> None:
    """Whatever the machine offers, CPU stays available as the last resort."""
    assert _names(cuda_providers())[-1] == "CPUExecutionProvider"


def test_only_advertised_providers_selected() -> None:
    available = set(ort.get_available_providers())
    assert set(_names(cuda_providers())) <= available


def test_cuda_absent_yields_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert cuda_providers() == CPU_ONLY


def test_preload_failure_still_tries_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken/absent preload_dlls must not cost us the GPU or raise."""
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def boom() -> None:
        raise RuntimeError("no nvidia wheels here")

    monkeypatch.setattr(ort, "preload_dlls", boom)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _names(cuda_providers()) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_an_empty_cuda_visible_devices_is_taken_at_its_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`make dev-cpu` hides the card; we must not ask for it anyway.

    ORT would try regardless (the EP is in the build, not on the bus), fail with
    "no CUDA-capable device is detected", fall back to CPU, and print that once
    per session — a clean choice that looks like a broken install.
    """
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cuda_providers() == CPU_ONLY


def test_devices_named_or_unset_still_get_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the EMPTY value means "no card" — unset means "all of them"."""
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _names(cuda_providers())[0] == "CUDAExecutionProvider"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert _names(cuda_providers())[0] == "CUDAExecutionProvider"


def test_cuda_is_selected_with_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CUDA entry must carry the options, not be a bare name.

    A bare ``"CUDAExecutionProvider"`` is exactly the shape that took out an
    external monitor on 2026-08-13, and it is a one-character edit away.
    """
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cuda = cuda_providers()[0]
    assert isinstance(cuda, tuple)
    assert cuda == ("CUDAExecutionProvider", cuda_options())


def test_arena_never_rounds_up_or_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both non-default knobs are load-bearing on a shared card.

    ``kNextPowerOfTwo`` rounds an extension up to the next power of two and
    BFCArena never returns it; ``EXHAUSTIVE`` benchmarks every conv algorithm
    with large transient workspaces. Either one alone can take the display's
    headroom while the steady-state footprint still looks modest.
    """
    monkeypatch.delenv("BR_GPU_MEM_LIMIT_MB", raising=False)
    options = cuda_options()
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    assert options["cudnn_conv_algo_search"] == "HEURISTIC"


def test_default_cap_stays_inside_the_measured_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is squeezed from both sides, and both sides bite.

    Too low and the node cannot serve: DINOv2 ViT-S/14 on a 64-crop board was
    measured failing at 768 MiB and passing at 896 (2026-08-13). Too high and
    the guarantee evaporates: serving builds THREE sessions in one process and
    the limit is per session, so 3x of it must still leave a 4 GiB card the
    ~1 GiB an X server needs for a 4K reverse-PRIME surface and its composition
    pipeline. Anything outside the band is a regression in one direction or the
    other, which is why this asserts the band and not the number.
    """
    monkeypatch.delenv("BR_GPU_MEM_LIMIT_MB", raising=False)
    limit = cuda_options()["gpu_mem_limit"]
    assert isinstance(limit, int)
    assert limit >= 896 * _MIB, "too small to run the DINOv2 embedder on a board"
    assert 3 * limit <= 3 * 1024 * _MIB, "three arenas would crowd out the display"


def test_cap_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BR_GPU_MEM_LIMIT_MB", "1536")
    assert cuda_options()["gpu_mem_limit"] == 1536 * _MIB


def test_zero_removes_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card that drives no display should not be rationed.

    ORT reads an *absent* key as "no limit", so the escape hatch has to drop the
    key rather than pass a sentinel.
    """
    monkeypatch.setenv("BR_GPU_MEM_LIMIT_MB", "0")
    assert "gpu_mem_limit" not in cuda_options()


def test_a_typo_falls_back_to_the_default_not_to_uncapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing open here would put the display back at risk over a typo."""
    monkeypatch.setenv("BR_GPU_MEM_LIMIT_MB", "1.5g")
    assert cuda_options()["gpu_mem_limit"] == _DEFAULT_GPU_MEM_LIMIT_MB * _MIB
