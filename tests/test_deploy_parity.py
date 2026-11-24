"""Keep the two ways the system is run from drifting apart.

There are exactly two entry points that start a real inference node: the
canonical demo (``docker compose up``, compose.yaml) and the dev path
(``make dev``, the Makefile). Both choose the *recognizer* — which model file
loads — out of the environment, and that choice is the single biggest lever on
how a board is read. Since DESIGN D10 both serve ChessQ (``BR_CHESSQ``) and
nothing of the old crop-classifier stack.

They serve the same *release* (ChessQ Lite V6) but not always the same *file*:
the file follows the hardware. A GPU node serves our fp32 export, a CPU node the
author's int8 (docs/reports/2026-11-24-chessq-gpu-formats.md — they read every
gate frame identically). compose declares no GPU device, so it serves int8; so
does ``make dev-cpu``; ``make dev`` serves fp32. The tests below pin that
pairing, so "different file" can only ever mean "different hardware".

They drifted once, silently. The confidence-gated cascade landed in compose.yaml
alone (``2b64ada``); ``make dev`` went on serving ``BR_MODEL`` by itself, so
every live phone test ran a measurably weaker reader than the demo it was
supposed to be rehearsing, and nothing failed. These tests are the guard.

Both files are read as *text and dry-run output* — never by starting a server —
so the whole module is fast and works on a fresh clone with no ``models/``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE = _ROOT / "compose.yaml"
_MAKEFILE = _ROOT / "Makefile"

#: The viewer's entry point (packages/viewer/server/main.ts), run by Node.
_VIEWER_ENTRY = "server/main.ts"

# The env family that decides *how boards are read*. These must agree between
# the two entry points: a difference here means the demo and the dev path
# disagree about what the system is, which is exactly the bug this module
# exists to catch.
_RECOGNITION_KEYS = frozenset({"BR_CHESSQ"})

# The deleted crop-classifier stack's settings. Nothing reads them any more, so one
# left in either entry point would be a dead knob that reads as if it mattered.
_LEGACY_KEYS = frozenset(
    {
        "BR_MODEL",
        "BR_MODEL_B",
        "BR_MODEL_TUNED",
        "BR_CASCADE_TAU",
        "BR_CROP_MASK",
    }
)

# Everything else may legitimately differ, because it describes the *environment*
# rather than the reader. Each entry has to carry its reason: an undocumented
# difference is indistinguishable from the drift above, so the test demands a
# justification rather than a name.
_ENVIRONMENT_SPECIFIC: dict[str, str] = {
    "BR_SAVE_UPLOADS": (
        "make dev only. Snapshot photos are saved into the working tree to grow "
        "the eval corpus; a container should not write into the repo."
    ),
    "BR_ALLOW_ORIGINS": (
        "compose only. The demo publishes the node's port for debugging, so it "
        "relaxes CORS; the dev path is reached through the viewer's relay."
    ),
    "BR_RECORD_DIR": (
        "compose only. Recording is on by default for `serve`; compose must point "
        "it at a bind mount or the recording dies with the container."
    ),
    "BR_RECORD_MAX_MB": "compose only, for the same reason as BR_RECORD_DIR.",
    "BR_CAMERA_CONFIG": (
        "compose only. `serve` defaults it to data/camera-config.json, which in a "
        "container would die with it; compose points it into the ./data bind mount, "
        "the same host file make dev's default names."
    ),
}


def _compose_inference_env() -> dict[str, str]:
    """The environment compose.yaml gives the inference service."""
    spec = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    env = spec["services"]["inference"].get("environment", {})
    return {k: str(v) for k, v in env.items()}


def _make_dev_env(target: str = "dev", **overrides: str) -> dict[str, str]:
    """The environment ``make <target>`` would give ``board-reader-inference serve``.

    Read from ``make -n`` rather than by pattern-matching the Makefile, so the
    conditional wiring (``CASCADE_ENV`` and friends) is exercised as make itself
    resolves it. ``-n`` prints recipes without running any of them, so the
    ``ui`` prerequisite costs nothing here.
    """
    cmd = ["make", "-n", target, *(f"{k}={v}" for k, v in overrides.items())]
    proc = subprocess.run(
        cmd, cwd=_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    # Assignments accumulate until an entry point consumes them; only the inference
    # server's block is ours (the viewer's BR_INFERENCE_* pair is a different
    # service's configuration).
    pending: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        for key, value in re.findall(r"\b(BR_[A-Z0-9_]+)=(\S*)", line):
            pending[key] = value
        if "uv run board-reader-inference serve" in line:
            return pending
        if "uv run board-reader-" in line or _VIEWER_ENTRY in line:
            pending = {}
    raise AssertionError("`make -n dev` never starts board-reader-inference serve")


def _model_relative(path: str) -> str:
    """Normalise a model path across the container and host layouts.

    compose mounts ``./models`` at ``/models``, so the same artifact is
    ``/models/chessq/<release>.onnx`` there and ``models/chessq/<release>.onnx`` here.
    """
    return path.split("models/", 1)[-1]


#: The V6 file each kind of hardware serves (Makefile ``BR_CHESSQ_GPU``/``_CPU``).
_GPU_MODEL = "chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx"
_CPU_MODEL = "chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx"


def _compose_declares_a_gpu() -> bool:
    """Whether compose reserves a GPU device for the inference service."""
    spec = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    devices = (
        spec["services"]["inference"].get("deploy", {}).get("resources", {})
        .get("reservations", {}).get("devices", [])
    )
    return any("gpu" in d.get("capabilities", []) for d in devices)


def test_dev_and_compose_declare_the_same_recognition_keys() -> None:
    """The dev path and the canonical demo configure recognition with the same keys."""
    compose_keys = _RECOGNITION_KEYS & _compose_inference_env().keys()
    dev_keys = _RECOGNITION_KEYS & _make_dev_env().keys()
    assert compose_keys == dev_keys == _RECOGNITION_KEYS, (
        "the recognition composition drifted between compose.yaml and `make dev`: "
        f"compose sets {sorted(compose_keys)}, make dev {sorted(dev_keys)}"
    )


def test_each_entry_point_serves_the_v6_file_for_its_hardware() -> None:
    """GPU -> the fp32 export, CPU -> the int8 release; nothing else differs."""
    compose = _model_relative(_compose_inference_env()["BR_CHESSQ"])
    assert compose == (_GPU_MODEL if _compose_declares_a_gpu() else _CPU_MODEL), (
        "compose.yaml must serve the fp32 export exactly when it reserves a GPU "
        f"and the int8 release otherwise; it serves {compose!r}"
    )
    assert _model_relative(_make_dev_env("dev")["BR_CHESSQ"]) == _GPU_MODEL
    assert _model_relative(_make_dev_env("dev-cpu")["BR_CHESSQ"]) == _CPU_MODEL


def test_both_entry_points_serve_chessq_and_nothing_of_the_old_stack() -> None:
    """DESIGN D10: ChessQ is the default recognizer, set even on a fresh clone
    (missing weights degrade to the stub inside the node, loudly)."""
    for name, env in (("compose.yaml", _compose_inference_env()), ("make dev", _make_dev_env())):
        assert "cq-vit-s-644-v6-seed1-onnx-" in env.get("BR_CHESSQ", ""), (
            f"{name} does not serve ChessQ V6"
        )
        stale = sorted(_LEGACY_KEYS & env.keys())
        assert not stale, f"{name} still sets the pre-D10 stack's {stale}"


def test_the_served_weights_are_the_ones_make_models_produces() -> None:
    """Every served file is the artifact of a committed descriptor — fetched or
    exported — and both are V6: the export names the V6 release it was built from."""
    import json

    descriptors = {
        f"chessq/{meta['artifact']['filename']}": meta
        for meta in (
            json.loads(p.read_text()) for p in sorted((_ROOT / "models/chessq").glob("*.json"))
        )
    }
    assert {_GPU_MODEL, _CPU_MODEL} <= descriptors.keys()
    assert descriptors[_CPU_MODEL]["model"]["release"].startswith("cq-vit-s-644-v6-seed1-")
    exported_from = descriptors[_GPU_MODEL]["export"]["source"]["release"]
    assert exported_from.startswith("cq-vit-s-644-v6-seed1-")
    assert exported_from in {meta["model"]["release"] for meta in descriptors.values()}


def test_every_inference_env_var_is_shared_or_justified() -> None:
    """A var set by only one entry point needs a recorded reason."""
    compose = _compose_inference_env()
    dev = _make_dev_env()

    only_one_side = set(compose) ^ set(dev)
    undeclared = {k for k in only_one_side if k not in _ENVIRONMENT_SPECIFIC}
    assert not undeclared, (
        f"{sorted(undeclared)} is set by only one entry point. If that is "
        "deliberate, add it to _ENVIRONMENT_SPECIFIC with the reason; if not, it "
        "is the compose/Makefile drift this module exists to catch."
    )

    stale = {k for k in _ENVIRONMENT_SPECIFIC if k not in only_one_side}
    assert not stale, (
        f"{sorted(stale)} is recorded as an environment-specific difference but "
        "is no longer one — drop the entry so the list stays trustworthy."
    )


def test_dev_disables_opencv_opencl_on_the_host_gpu() -> None:
    """The host path must not let OpenCV's OpenCL backend abort the node.

    Env parity with compose is not enough, because the two run on different
    hardware: compose declares no GPU device and runs the recognizer on CPU, while
    `make dev` puts its ONNX session on the host card. When OpenCV's OpenCL
    `warpAffine` then cannot allocate, it throws a cv::Exception through C++
    terminate — the node *aborts* mid-stream (status 134) instead of degrading.
    Measured 2026-08-13 on a 4GB GTX 1650. `disabled` is the strong form: the
    runtime never loads, so it claims no VRAM at all.
    """
    proc = subprocess.run(
        ["make", "-n", "dev"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if "OPENCV_OPENCL_RUNTIME=" in line:
            assert re.search(r"OPENCV_OPENCL_RUNTIME=disabled", line), (
                f"OpenCL must be disabled for the host GPU path, got {line.strip()!r}"
            )
            return
    pytest.fail("`make dev` no longer sets OPENCV_OPENCL_RUNTIME at all")


def _inference_launch_block(target: str) -> str:
    """The shell text ``make -n <target>`` runs to start the inference server.

    Everything up to and including the ``uv run board-reader-inference serve``
    line — which is exactly the set of assignments that command inherits.
    """
    proc = subprocess.run(
        ["make", "-n", target],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    head, sep, _ = proc.stdout.partition("uv run board-reader-inference serve")
    assert sep, f"`make -n {target}` never starts board-reader-inference serve"
    return head + sep


def test_dev_cpu_hides_the_card_and_dev_keeps_it() -> None:
    """`make dev-cpu` must deny the node a GPU — and `make dev` must not.

    On a laptop this card is also the display pipeline, and the display has no
    fallback to degrade to: on 2026-08-13 an inference batch took the VRAM an X
    server needed for a 4K external monitor's reverse-PRIME surface and the
    screen went dark twice. `dev-cpu` is the escape hatch, so the assertion is
    on the *value*: CUDA reads an EMPTY CUDA_VISIBLE_DEVICES as "no devices",
    while unset means "all of them" — the two are one character apart and only
    one of them is the feature.
    """
    hidden = re.search(r"\bCUDA_VISIBLE_DEVICES=(\S*)", _inference_launch_block("dev-cpu"))
    assert hidden is not None, "`make dev-cpu` no longer sets CUDA_VISIBLE_DEVICES"
    assert hidden.group(1) == "", (
        "CUDA_VISIBLE_DEVICES must be set to the EMPTY string to hide the card, "
        f"got {hidden.group(1)!r} — a non-empty value names devices to use"
    )
    assert "CUDA_VISIBLE_DEVICES" not in _inference_launch_block("dev"), (
        "`make dev` must leave the GPU visible; hiding it there would silently "
        "make the default dev path slower than the demo it rehearses"
    )


def test_dev_cpu_differs_from_dev_only_in_the_hardware_choice() -> None:
    """The CPU escape hatch must differ in hardware only, never in recognition.

    A `dev-cpu` that quietly served a different composition would be the same
    class of bug as the compose/`make dev` cascade drift this module exists to
    catch — and harder to see, because the two targets look identical. The one
    allowed difference is the V6 file, pinned by the test above.
    """
    cpu, gpu = _make_dev_env("dev-cpu"), _make_dev_env("dev")
    assert {k: v for k, v in cpu.items() if k != "BR_CHESSQ"} == {
        k: v for k, v in gpu.items() if k != "BR_CHESSQ"
    }


def test_both_entry_points_start_the_same_viewer_against_the_node() -> None:
    """The Node viewer, pointed at the node by the same two variables, in both."""
    spec = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]["viewer"]
    assert _VIEWER_ENTRY in " ".join(spec["command"])
    assert {"BR_INFERENCE_HTTP", "BR_INFERENCE_WS"} <= spec["environment"].keys()

    dry_run = subprocess.run(
        ["make", "-n", "dev"], cwd=_ROOT, capture_output=True, text=True, timeout=120, check=True
    ).stdout
    viewer = next((line for line in dry_run.splitlines() if _VIEWER_ENTRY in line), None)
    assert viewer is not None, "`make dev` no longer starts the Node viewer"
    head = dry_run[: dry_run.index(viewer) + len(viewer)]
    tail = head.rsplit("uv run board-reader-inference serve", 1)[-1]
    assert "BR_INFERENCE_HTTP=" in tail and "BR_INFERENCE_WS=" in tail
