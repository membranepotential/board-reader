"""The streams file: whole or nothing, beside the camera config."""

from __future__ import annotations

import json
import logging

from board_reader_inference.control.streams.store import (
    StreamRecord,
    load_streams,
    save_streams,
    streams_path,
)

RECORD = StreamRecord("stream-1", "https://video.test/v", 60.0, "kitchen-stream-id", 1.5)


def test_the_streams_round_trip_through_their_file(tmp_path):
    path = streams_path(tmp_path / "camera-config.json")
    assert path == tmp_path / "streams.json"
    save_streams(path, (RECORD,))
    assert load_streams(path) == (RECORD,)
    assert [p.name for p in tmp_path.iterdir()] == ["streams.json"]  # no tmp left behind


def test_no_file_is_no_streams(tmp_path):
    assert load_streams(tmp_path / "streams.json") == ()


def test_a_file_that_does_not_parse_in_full_is_ignored_aloud(tmp_path, caplog):
    path = tmp_path / "streams.json"
    good = RECORD.to_dict()
    for body in (
        "{not json",
        json.dumps({"streams": [good, {**good, "url": "file:///etc/passwd"}]}),
        json.dumps({"streams": [good, {**good, "device_id": "other-stream-id"}]}),  # one id twice
        json.dumps({"streams": [good], "extra": 1}),
    ):
        path.write_text(body)
        with caplog.at_level(logging.WARNING):
            assert load_streams(path) == ()
    assert caplog.text.count("ignoring the corrupt streams file") == 4
