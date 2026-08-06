"""Provider selection must never raise and never drop the CPU fallback."""

from __future__ import annotations

import pytest

from board_reader_inference.onnxrt import CPU_ONLY, cuda_providers

ort = pytest.importorskip("onnxruntime")


def test_cpu_fallback_always_last() -> None:
    """Whatever the machine offers, CPU stays available as the last resort."""
    assert cuda_providers()[-1] == "CPUExecutionProvider"


def test_only_advertised_providers_selected() -> None:
    available = set(ort.get_available_providers())
    assert set(cuda_providers()) <= available


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
    assert cuda_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
