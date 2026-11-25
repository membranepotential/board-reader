"""Writes the one photo every labeled test board points at."""

from __future__ import annotations

import _evaluate_builders
import pytest

from board_reader_core.sim import render_board
from board_reader_core.wire import encode_jpeg


@pytest.fixture(autouse=True, scope="session")
def _photo_on_disk(tmp_path_factory):
    path = tmp_path_factory.mktemp("photos") / "board.jpg"
    path.write_bytes(encode_jpeg(render_board(_evaluate_builders.MIDGAME)))
    _evaluate_builders.PHOTO = path
    yield
