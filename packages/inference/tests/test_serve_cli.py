"""`serve`'s environment: the recording directory (the flag, then BR_RECORD_DIR, then the
default) and the node URL its streams connect back to."""

from __future__ import annotations

import os

import pytest
import uvicorn

from board_reader_inference import cli


@pytest.fixture
def serve(monkeypatch):
    # A private environment: monkeypatch cannot restore a key the code under test sets.
    env = dict(os.environ)
    env.pop("BR_RECORD_DIR", None)
    env.pop("BR_NODE_URL", None)
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    def run(*flags: str) -> dict[str, str]:
        cli.cmd_serve(cli.build_parser().parse_args(["serve", *flags]))
        return env

    return run, env


def test_serve_records_to_the_default_when_nothing_says_otherwise(serve):
    run, _ = serve
    assert run()["BR_RECORD_DIR"] == cli.DEFAULT_RECORD_DIR


def test_an_inherited_record_dir_outranks_the_default(serve):
    run, env = serve
    env["BR_RECORD_DIR"] = "/elsewhere"
    assert run()["BR_RECORD_DIR"] == "/elsewhere"


def test_the_flag_outranks_the_environment(serve):
    run, env = serve
    env["BR_RECORD_DIR"] = "/elsewhere"
    assert run("--record-dir", "/flag")["BR_RECORD_DIR"] == "/flag"


def test_no_record_clears_an_inherited_record_dir(serve):
    run, env = serve
    env["BR_RECORD_DIR"] = "/elsewhere"
    assert "BR_RECORD_DIR" not in run("--no-record")


@pytest.mark.parametrize(
    ("flags", "url"),
    [
        ((), "ws://127.0.0.1:8000"),
        (("--host", "0.0.0.0", "--port", "9000"), "ws://127.0.0.1:9000"),  # every address
        (("--host", "192.168.1.5"), "ws://192.168.1.5:8000"),
        (("--host", "::1"), "ws://[::1]:8000"),
    ],
)
def test_serve_tells_its_streams_where_the_node_listens(serve, flags, url):
    run, _ = serve
    assert run(*flags)["BR_NODE_URL"] == url
