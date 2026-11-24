---
paths:
  - 'tests/**'
  - 'packages/*/tests/**'
  - 'tools/wtest.sh'
---

# The Python test suite

- Drive WS `/ingest` and `/viewer` through the `gateway` and `drain_ingest` fixtures in `packages/inference/tests/conftest.py`, and enter any Starlette `TestClient` as a context manager, never `TestClient(app)` plus sleep-polling, because a bare client makes a new event loop per call, so the ingest worker dies with the socket, positions are lost under parallel load, and a read of `/viewer` can hang forever.
- Never assert an asynchronously published result right after a synchronous one; poll with a deadline on a positive completion signal (`GET /boards/{id}`, `/healthz`'s ingest counters), because `/ingest` acks a frame when it is decoded and records it inline, while the read runs later in the worker, and such an assert flakes about once in three runs under `-n auto`.
- Never poll for something that must not happen; wait on a signal that is true on every branch, because a negative wait burns its whole deadline on every run.
- Wrap async socket and server tests in `async with asyncio.timeout(10)`, and let a fake websockets handler return instead of awaiting a bare `Future`, because pytest-timeout is not installed and websockets' `Server.__aexit__` waits for handlers that a closed connection never wakes, so the run hangs silently.
- Start e2e uvicorn servers with `timeout_graceful_shutdown=1`, because graceful shutdown waits for open WebSockets, and the e2e tests once spent most of their time in teardown. When a test's runtime sits near a join or shutdown timeout, suspect teardown first.
- Use `tests/_sim_recognizer.SimRecognizer` (template match against `render_board` squares, exact through a JPEG round trip) in any `tests/` test that needs real pixels read, because ChessQ reads 0 of 14 `core.sim` renders perfectly. Package tests cannot import it; they use `StubRecognizer` or a `FrameRead` fake. To make a fake single out one frame, compare the image it gets with `decode_jpeg(frame.encode()[1])`, which is exactly what `/ingest` decodes.
- For the shipped schedule in a test, `monkeypatch.setenv("BR_SCHEDULE", DEFAULT_POLICY_SPEC)` (the pipe e2e included); to test `cmd_serve` itself, swap `os.environ` for a private dict first (`monkeypatch.setattr(os, "environ", dict(os.environ))`), because monkeypatch cannot restore a key the code under test sets. Assert an every-frame policy through a real socket as `estimated + dropped + unscheduled == N`, never `estimated == N`, because the latest-wins inbox drops frames even with a stub.
- Isolate every test that can make the node write to disk (`tmp_path` for `BR_RECORD_DIR`/`BR_SAVE_UPLOADS`, or `monkeypatch.chdir(tmp_path)`), because a test run from the repo root once overwrote the operator's real state.
- Cover a quarter turn (90 or 270 degrees) in any rotation test, never 180 alone, because `np.rot90(x, 2) == np.rot90(x, -2)`, so a 180-only test passes when the rotation direction is inverted.
- Make a seeded property test prove itself: generate drifting boards from a small alphabet at a spacing that lets runs span the window, keep a subclass with the pre-fix logic and assert it fails, and assert the set of code paths taken, because a first version made 580 calls and zero commits and passed with the rule deleted. An ordering test needs two candidates eligible at the same instant that would give different boards.
- Do not chase `test_replay.py::test_default_name_tracks_the_model_environment` when the tree is dirty: `replay.default_name()` refuses an uncommitted tree by design, so the test fails with uncommitted work and passes after the commit.
- Keep `tests/test_layer_separation.py` parsing imports with `ast` (`Import`/`ImportFrom` roots), never substring matching, because docstrings legitimately name the forbidden packages.
- In a git worktree run tests with `tools/wtest.sh`, not `uv run pytest`, because `uv run` builds the whole workspace into a worktree-local `.venv` (minutes per run), while the script runs main's `.venv` with the worktree's `packages/*/src` first on `PYTHONPATH`.
