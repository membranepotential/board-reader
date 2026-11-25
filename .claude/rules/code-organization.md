---
paths:
  - 'packages/*/src/**'
  - 'packages/*/tests/**'
---

# Organizing a package's source

- Keep a package's `src/<package>/` flat only while it holds a handful of modules with one concern (core and edge today, 5 and 7 modules). Once it grows past ~8 modules or several distinct concerns, group the modules into subpackages by concern, each with a one-line docstring in its `__init__.py`, and keep only the entry points (the app factory, the CLI) at the package root, because a flat folder of 17 modules (the inference node, 2026-11-24) hides which parts belong together and which may import which. The user asked for this explicitly.
- Organize `board_reader_inference` like this, and put a new module into the group whose concern it serves:
  - `recognition/` — what turns pixels into a distribution, and where the board lies: `recognizer.py`, `chessq.py`, `onnxrt.py`, `locate.py`.
  - `stream/` — the live read path after recognition: `ingest.py`, `schedule.py`, `stabilize.py`, `rules.py`, `hub.py`.
  - `control/` — the camera control channel: `camera.py`.
  - `measurement/` — replay, scoring and labels: `replay.py`, `trial.py`, `evaluate.py`, `labels.py`, `calibration.py`.
  - the root — `server.py` (the app factory), `cli.py`, and `positions.py` (the headless `/viewer` client).
- Move modules with `git mv` in one change that also updates every import, `pyproject.toml` entry points, the `paths:` globs in `.claude/rules/`, doc links, and the learnings store's scopes (`learnings rescope`, so `learnings audit` stays at 0), because a move is what breaks those references, not an edit. Add no re-export shims for old import paths; update the callers instead.
- Keep dependencies between the groups one-directional: `recognition` imports nothing from the others, `stream` may import `recognition`, `measurement` may import both, `control` imports none of them, and only the root wires everything together, because that is what keeps the replay harness running the same read path as the live node. Keep `event_line` (the `events.ndjson` line writer) in `stream/ingest.py` for the same reason: in `server.py` it made `replay` import the root. `tests/test_inference_layers.py` checks the rule with an AST scan.
- Mirror the source tree in the tests: a module at `src/<package>/<group>/<module>.py` is tested in `tests/<group>/test_<module>.py` (for example `board_reader_inference/stream/stabilize.py` → `packages/inference/tests/stream/test_stabilize.py`), and a test that spans groups (the gateway, end-to-end) lives at the `tests/` root next to `conftest.py`, because the user wants to find a module's tests by its path, and a flat tests folder hides which group a test covers. Move tests with `git mv` in the same change that moves their module. pytest imports test files by basename unless the tests are packages or run in `--import-mode=importlib`, so keep every test file name unique across the workspace, or switch the suite to `importlib` mode in `pyproject.toml` deliberately (then import shared helpers through `conftest.py` fixtures or a package, not by sibling module name). The frontend already follows this: each `*.test.ts` sits next to the module it tests.
