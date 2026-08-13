"""Keep the two ways the system is run from drifting apart.

There are exactly two entry points that start a real inference node: the
canonical demo (``docker compose up``, compose.yaml) and the dev path
(``make dev``, the Makefile). Both choose the *recognition composition* — which
model files load and how they combine — out of the environment, and that choice
is the single biggest lever on how a board is read.

They drifted once, silently. The confidence-gated cascade landed in compose.yaml
alone (``3a55d41``); ``make dev`` went on serving ``BR_MODEL`` by itself, so
every live phone test ran a measurably weaker reader than the demo it was
supposed to be rehearsing, and nothing failed. These tests are the guard, and
``aspects.yaml`` carries the Makefile inside the ``deploy`` aspect so ``/drift``
reasons about it too.

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

# The env family that decides *how boards are read*. These must agree between
# the two entry points: a difference here means the demo and the dev path
# disagree about what the system is, which is exactly the bug this module
# exists to catch.
_RECOGNITION_KEYS = frozenset(
    {"BR_MODEL", "BR_MODEL_B", "BR_MODEL_TUNED", "BR_CASCADE_TAU", "BR_CENTROIDS"}
)

# Everything else may legitimately differ, because it describes the *environment*
# rather than the reader. Each entry has to carry its reason: an undocumented
# difference is indistinguishable from the drift above, so the test demands a
# justification rather than a name.
_ENVIRONMENT_SPECIFIC: dict[str, str] = {
    "BR_SESSION_BANK": (
        "make dev only. The session bank holds exemplars of a *physical* board; "
        "the compose demo streams synthetic renders, where a real set's bank is "
        "noise rather than signal."
    ),
    "BR_SAVE_UPLOADS": (
        "make dev only. Calibrate frames are saved into the working tree to grow "
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
}


def _compose_inference_env() -> dict[str, str]:
    """The environment compose.yaml gives the inference service."""
    spec = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    env = spec["services"]["inference"].get("environment", {})
    return {k: str(v) for k, v in env.items()}


def _make_dev_env(**overrides: str) -> dict[str, str]:
    """The environment ``make dev`` would give ``board-reader-inference serve``.

    Read from ``make -n`` rather than by pattern-matching the Makefile, so the
    conditional wiring (``CASCADE_ENV`` and friends) is exercised as make itself
    resolves it. ``-n`` prints recipes without running any of them, so the
    ``ui`` prerequisite costs nothing here.
    """
    cmd = ["make", "-n", "dev", *(f"{k}={v}" for k, v in overrides.items())]
    proc = subprocess.run(
        cmd, cwd=_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    # Assignments accumulate until a `uv run <entry point>` consumes them; only
    # the inference server's block is ours (the viewer's BR_INFERENCE_* pair is
    # a different service's configuration).
    pending: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        for key, value in re.findall(r"\b(BR_[A-Z0-9_]+)=(\S*)", line):
            pending[key] = value
        if "uv run board-reader-inference serve" in line:
            return pending
        if "uv run board-reader-" in line:
            pending = {}
    raise AssertionError("`make -n dev` never starts board-reader-inference serve")


def _model_relative(path: str) -> str:
    """Normalise a model path across the container and host layouts.

    compose mounts ``./models`` at ``/models``, so the same artifact is
    ``/models/run_e/model.onnx`` there and ``models/run_e/model.onnx`` here.
    """
    return path.split("models/", 1)[-1]


def test_dev_and_compose_declare_the_same_recognition_models() -> None:
    """The dev path and the canonical demo must read boards the same way."""
    compose = _compose_inference_env()
    # Force both members "present" so this asserts declared intent and stays
    # green on a fresh clone, where models/ does not exist at all.
    existing = str(_ROOT / "pyproject.toml")
    dev = _make_dev_env(BR_MODEL=existing, BR_MODEL_B=existing)

    compose_keys = _RECOGNITION_KEYS & compose.keys()
    dev_keys = _RECOGNITION_KEYS & dev.keys()
    assert compose_keys == dev_keys, (
        "the recognition composition drifted between compose.yaml and `make dev`: "
        f"only in compose {sorted(compose_keys - dev_keys)}, "
        f"only in make {sorted(dev_keys - compose_keys)}"
    )

    # ...and they must name the same artifacts, not merely the same variables.
    defaults = _make_dev_env()
    for key in sorted(compose_keys):
        if key not in defaults:  # withheld because the file is absent locally
            continue
        assert _model_relative(defaults[key]) == _model_relative(compose[key]), (
            f"{key} points at different artifacts: compose {compose[key]!r} vs "
            f"make dev {defaults[key]!r}"
        )


def test_every_inference_env_var_is_shared_or_justified() -> None:
    """A var set by only one entry point needs a recorded reason."""
    compose = _compose_inference_env()
    existing = str(_ROOT / "pyproject.toml")
    dev = _make_dev_env(BR_MODEL=existing, BR_MODEL_B=existing)

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


def test_dev_withholds_the_cascade_when_a_member_is_missing() -> None:
    """A partial pair is a hard startup error; `make dev` must not trip it.

    Mirrors ``server._trained_component``: BR_MODEL_B present with an unusable
    BR_MODEL (or vice versa) raises rather than quietly serving one model. Running
    `make dev` on a half-populated models/ dir should degrade, not crash.
    """
    dev = _make_dev_env(
        BR_MODEL=str(_ROOT / "pyproject.toml"), BR_MODEL_B="models/does-not-exist.onnx"
    )
    assert "BR_MODEL_B" not in dev
    assert dev["BR_MODEL"].endswith("pyproject.toml")


def test_dev_withholds_the_cascade_under_a_tuned_model() -> None:
    """BR_MODEL_TUNED and BR_MODEL_B together are an unmeasured stack.

    The node refuses the combination outright, so the default cascade member has
    to step aside when an operator asks for the calibration average.
    """
    dev = _make_dev_env(BR_MODEL_TUNED="models/tuned.onnx")
    assert dev.get("BR_MODEL_TUNED") == "models/tuned.onnx"
    assert "BR_MODEL_B" not in dev, (
        "make dev forwarded both BR_MODEL_TUNED and BR_MODEL_B — the server "
        "treats that as a hard error, so the dev path would not start"
    )


@pytest.mark.parametrize("path", [_COMPOSE, _MAKEFILE])
def test_both_entry_points_are_registered_in_the_deploy_aspect(path: Path) -> None:
    """`/drift` can only reason about files aspects.yaml claims.

    The Makefile was missing from the ``deploy`` aspect, which is why the cascade
    drift went unreported: the aspect described "how the system is actually run"
    but listed only the container half of it.
    """
    spec = yaml.safe_load((_ROOT / "aspects.yaml").read_text(encoding="utf-8"))
    files = spec["aspects"]["deploy"]["files"]
    assert path.name in files, (
        f"{path.name} is an entry point that starts a real node, but the deploy "
        f"aspect only claims {files}"
    )
