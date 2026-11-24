---
paths:
  - 'Makefile'
  - 'compose.yaml'
  - 'docker/**'
  - '.dockerignore'
  - 'tools/stop.sh'
  - 'tests/test_deploy_parity.py'
  - 'tests/test_stop.py'
---

# Running the stack: Makefile, compose, Docker

- Keep the two entry points that start a real node, `docker compose up` (compose.yaml) and `make dev` (the Makefile), on the same `BR_*` recognition env, and extend `tests/test_deploy_parity.py` when you add a variable, because they once drifted for two days and every live phone test rehearsed a weaker reader than the demo. The test reads the env from `make -n dev` with overrides on the command line (a file that exists), so it checks the Makefile's conditionals and stays green on a fresh clone without `models/`.
- Never `kill 0` in a shell trap; kill the captured PIDs and run `trap - EXIT INT TERM` first in the handler, because the group kill re-enters its own trap until bash overflows its stack (every Ctrl-C of `make dev` once exited 139 with a core dump).
- Wait on a multi-process recipe with `wait -n`, not bare `wait`, so the recipe tears down the survivors and exits with the real status when one server dies at startup (a bound port, say), instead of serving a dead backend.
- Kill a server by the PID of its `uv run`, because `uv run` forwards SIGTERM and the child shuts down gracefully; never `pkill` on a loose pattern (use `make stop`).
- Attribute a segfault with `coredumpctl info <pid>` before you blame onnxruntime or OpenCV, because the `make dev` "Segmentation fault" was bash's trap recursion while the Python servers had exited cleanly.
- Install `libglib2.0-0` in the Python image (`docker/Dockerfile`), because `opencv-python-headless` still needs it on `python:3.12-slim`, and build with `uv sync --frozen`, so the image uses the same seven-day-seasoned lock as the workspace.
- Build the viewer image (`docker/viewer.Dockerfile`) with `pnpm install --frozen-lockfile` and ship only `build/`, `server/`, `package.json` and the production `node_modules` (`pnpm prune --prod`), because Node runs `server/main.ts` directly and adapter-node bundles everything else; a package the server imports must therefore sit under `dependencies`.
- Kill the viewer by the PID of its `node` process and stop it with the `[v]iewer/ui/server/main.ts` pattern, because `make dev` starts it as `node packages/viewer/ui/server/main.ts`, not through `uv run`.
