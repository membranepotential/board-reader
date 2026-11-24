"""`serve`'s recording directory: the flag, then BR_RECORD_DIR, then the default."""

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
