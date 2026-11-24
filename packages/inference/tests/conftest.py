"""Shared gateway-test plumbing for the inference package.

One thing about ``WS /ingest`` shapes every test that asserts on a *published*
position: the ack means **received**, not **estimated**. The handler submits the
frame to a per-board worker task and acks immediately
(:mod:`board_reader_inference.ingest`), so the position reaches the hub some time
*after* the ack the test just read.

That alone would only call for patience. Starlette's ``TestClient`` turns it into
two real failures, both of which this module exists to prevent:

* **A ``TestClient`` that is not entered as a context manager gets a fresh event
  loop per call** (``_portal_factory`` only reuses a portal once ``__enter__`` has
  set one). The ingest worker is an ``asyncio`` task on the *websocket's* loop, so
  it dies with the websocket and can never publish. A later
  ``GET /boards/A1`` then 404s.
* **``WebSocketTestSession.__exit__`` cancels the app mid-flight.** It sends the
  disconnect and then immediately cancels the scope running the handler, so the
  handler's own ``await ingest_.drain()`` — the thing meant to let the last frame
  land — is often cancelled before it gets to run.

Together those lose roughly 1 frame in 150 *under load*, which is why this only
ever bit the parallel (``make test``) runs. The two symptoms are the same bug:

* ``GET /boards/{id}`` 404s where the test expects 200 — a visible flake;
* a test that reads the retained latest off ``WS /viewer`` **blocks forever**,
  because nothing was ever published and ``receive_text()`` has no timeout. That
  is the whole-suite hang that has cost several sessions.

:func:`gateway` and :func:`drain_ingest` remove both. ``gateway`` keeps one portal
— one event loop — for the client's whole life, so workers outlive their socket
and the lifespan still shuts them down; ``drain_ingest`` then waits on the ingest
workers' own idle events. It is deterministic: no ``sleep``-polling, no deadline
to tune, and it returns the moment the work is genuinely done.

Rule of thumb for a new test: if it asserts on anything the *estimator* produced
(``/boards/{id}``, a ``/viewer`` event, ``frames_estimated``),
build the client with ``gateway`` and call ``drain_ingest`` after the last ack.
If it only asserts on what the handler wrote *before* acking — the raw-frame
retention behind ``/boards/{id}/frame``, or the ingress recording — a plain
``TestClient`` is fine and cheaper.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

# Generous by design: on a healthy run every drain returns in microseconds, so
# this bound is never approached. It exists so a genuinely stuck worker fails one
# test with a traceback instead of hanging the whole suite with no output — the
# failure mode this module was written to kill.
DRAIN_TIMEOUT = 10.0


@pytest.fixture
def gateway() -> Iterator[Callable[[FastAPI], TestClient]]:
    """Factory for a gateway ``TestClient`` whose event loop outlives its websockets.

    Use instead of a bare ``TestClient(app)`` whenever the test asserts on
    something the ingest workers produce — see this module's docstring. Clients
    are closed (and the app's lifespan run) when the test ends.
    """
    with contextlib.ExitStack() as stack:
        yield lambda app: stack.enter_context(TestClient(app))


@pytest.fixture
def drain_ingest() -> Callable[[TestClient], None]:
    """Block until every ingest worker has finished the frames it was handed.

    The deterministic replacement for "sleep until the hub looks right": it waits
    on :meth:`FrameIngest.drain`, which is set exactly when a worker's inbox is
    empty and nothing is in flight. The client must come from :func:`gateway` —
    a client with no shared portal has no live loop left to drain.
    """

    def _drain(client: TestClient) -> None:
        if client.portal is None:
            # Without a shared portal every call gets its own event loop, so the
            # worker holding this frame is already gone and no amount of waiting
            # will publish it. Say that, rather than raising AttributeError on None.
            raise RuntimeError(
                "drain_ingest needs a client from the `gateway` fixture: a plain "
                "TestClient has no shared event loop for the ingest workers to "
                "outlive the websocket on. See this module's docstring."
            )
        ingest = client.app.state.ingest

        async def _wait() -> None:
            await asyncio.wait_for(ingest.drain(), timeout=DRAIN_TIMEOUT)

        client.portal.call(_wait)

    return _drain
