"""Tests for the crop registry — the part that fails silently if it drifts.

Nothing here touches ``data/`` (git-ignored, may be absent) or a real video;
every fixture is built in a tmp_path.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streams import (
    COLUMNS,
    DEFAULT_TABLE,
    CropBox,
    CropTableError,
    Stream,
    StreamStatus,
    UnknownStreamError,
    append_stream,
    load_streams,
    parse_video_id,
    slug_from_title,
    stream_for_video,
    title_from_path,
    video_id_from_path,
)

HEXAMIND = (
    "Hexamind Chess Team ｜ Day 1 ｜ FIDE World Team Blitz Chess Championship 2026"
    " [JTmovXvZnPM].mp4"
)


def make_stream(**overrides: object) -> Stream:
    fields: dict[str, object] = {
        "video_id": parse_video_id("JTmovXvZnPM"),
        "slug": "hexamind",
        "title": "Hexamind Chess Team | Day 1",
        "url": "https://www.youtube.com/watch?v=JTmovXvZnPM",
        "crop": CropBox(x=379, y=305, width=1116, height=646),
        "frame_width": 1920,
        "frame_height": 1080,
        "preview_at_s": 3600.0,
        "median_aspect": 0.65,
        "status": StreamStatus.KEEP,
        # Naive to match what crop_selector actually writes (a local operator
        # stamp); a tz-aware fixture would not round-trip through the registry.
        "recorded_at": datetime(2026, 8, 6, 23, 15, 45),  # noqa: DTZ001
    }
    fields.update(overrides)
    return Stream(**fields)  # type: ignore[arg-type]


# --- the committed table ---------------------------------------------------


# These assert *invariants* of the committed table, never its exact contents.
# Recording a new crop is the toolset's whole purpose and happens between test
# runs, so pinning the row set (or "exactly one stream survives the cull") makes
# ordinary use turn the suite red — it did, the first time someone added a
# broadcast. Facts about which streams exist today belong in
# docs/youtube-streams.md, which is prose and expects to be edited.
def test_committed_table_loads_and_is_keyed_by_id() -> None:
    streams = load_streams(DEFAULT_TABLE)
    assert streams, "the committed registry should never be empty"
    # The mapping key is the id, and it agrees with the row it points at.
    assert all(vid == s.video_id for vid, s in streams.items())
    assert all(len(vid) == 11 for vid in streams)


def test_committed_table_retains_the_originally_audited_streams() -> None:
    """The five absorbed broadcasts stay recoverable — their videos are deleted,
    so this table plus a video id is the *only* way back to any of them."""
    streams = load_streams(DEFAULT_TABLE)
    audited = {
        "JTmovXvZnPM": ("hexamind", StreamStatus.KEEP),
        "ZoYhhoG6dIY": ("divya-r14", StreamStatus.REJECTED_LOW_ANGLE),
        "1O2ahFZh6G8": ("coffeechess", StreamStatus.REJECTED_LOW_ANGLE),
        "Fed1aBzUMcs": ("wtr-day2", StreamStatus.REJECTED_LOW_ANGLE),
        "62h7O9Ns_eA": ("grandswiss", StreamStatus.REJECTED_LOW_ANGLE),
    }
    for video_id, (slug, status) in audited.items():
        assert video_id in streams, f"{video_id} dropped from the registry"
        assert streams[video_id].slug == slug
        assert streams[video_id].status is status


def test_committed_crops_fit_their_recorded_frame() -> None:
    for stream in load_streams(DEFAULT_TABLE).values():
        assert stream.crop.fits_within(stream.frame_width, stream.frame_height)


# --- round-trip ------------------------------------------------------------


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    original = make_stream()
    append_stream(original, table)

    assert table.read_text().splitlines()[0] == ",".join(COLUMNS)
    assert load_streams(table)["JTmovXvZnPM"] == original


def test_missing_median_aspect_round_trips_as_none(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    append_stream(make_stream(median_aspect=None), table)
    assert load_streams(table)["JTmovXvZnPM"].median_aspect is None


def test_later_row_supersedes_earlier_one_for_same_id(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    append_stream(make_stream(), table)
    append_stream(make_stream(crop=CropBox(x=0, y=0, width=100, height=100)), table)
    assert load_streams(table)["JTmovXvZnPM"].crop.width == 100


def test_ffmpeg_filter_is_wxh_at_xy() -> None:
    assert CropBox(379, 305, 1116, 646).ffmpeg_filter == "crop=1116:646:379:305"


# --- validation at the boundary --------------------------------------------


def test_crop_wider_than_its_frame_is_rejected() -> None:
    with pytest.raises(CropTableError, match="falls outside"):
        make_stream(crop=CropBox(x=1000, y=0, width=1000, height=100))


def test_crop_taller_than_its_frame_is_rejected() -> None:
    with pytest.raises(CropTableError, match="falls outside"):
        make_stream(crop=CropBox(x=0, y=900, width=100, height=200))


def test_crop_exactly_filling_the_frame_is_accepted() -> None:
    make_stream(crop=CropBox(x=0, y=0, width=1920, height=1080))


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"height": 0}, {"width": -4}, {"height": -4}]
)
def test_degenerate_crop_extent_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(CropTableError, match="positive extent"):
        CropBox(**{"x": 0, "y": 0, "width": 10, "height": 10, **kwargs})


def test_negative_crop_origin_is_rejected() -> None:
    with pytest.raises(CropTableError, match="non-negative"):
        CropBox(x=-1, y=0, width=10, height=10)


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    append_stream(make_stream(), table)
    table.write_text(table.read_text().replace(",keep,", ",maybe,"))
    with pytest.raises(CropTableError, match="unknown status"):
        load_streams(table)


def test_unexpected_header_is_rejected(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    table.write_text("recorded_at,video,x,y\n2026-01-01,a.mp4,1,2\n")
    with pytest.raises(CropTableError, match="unexpected header"):
        load_streams(table)


def test_missing_table_names_the_selector(tmp_path: Path) -> None:
    with pytest.raises(CropTableError, match=re.escape("crop_selector.py")):
        load_streams(tmp_path / "nope.csv")


@pytest.mark.parametrize("bad", ["short", "waytoolongforanid", "has spaces", ""])
def test_bad_video_id_is_rejected(bad: str) -> None:
    with pytest.raises(CropTableError, match="not a YouTube video id"):
        parse_video_id(bad)


# --- resolving a video path to a row ---------------------------------------


def test_video_id_comes_from_the_bracketed_suffix() -> None:
    assert video_id_from_path(HEXAMIND) == "JTmovXvZnPM"


def test_underscores_and_dashes_are_part_of_the_id() -> None:
    assert video_id_from_path("Grand Swiss [62h7O9Ns_eA].mp4") == "62h7O9Ns_eA"
    assert video_id_from_path("Some Talk [-b3xQoJ_1cA].mkv") == "-b3xQoJ_1cA"


def test_the_last_bracketed_group_wins() -> None:
    # Broadcast titles carry brackets of their own; yt-dlp appends the id last.
    assert (
        video_id_from_path("Round 11 [Board 57] [JTmovXvZnPM].mp4") == "JTmovXvZnPM"
    )


def test_path_without_an_id_says_what_to_do() -> None:
    with pytest.raises(UnknownStreamError, match="no YouTube id"):
        video_id_from_path("chess.mp4")


def test_unregistered_video_points_at_the_selector(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    append_stream(make_stream(), table)
    with pytest.raises(UnknownStreamError, match=re.escape("crop_selector.py")):
        stream_for_video("Other [Fed1aBzUMcs].mp4", table)


def test_stream_for_video_resolves_through_the_filename(tmp_path: Path) -> None:
    table = tmp_path / "crops.csv"
    append_stream(make_stream(), table)
    assert stream_for_video(f"/videos/{HEXAMIND}", table).slug == "hexamind"


def test_frames_dir_is_slug_underscore_id() -> None:
    assert make_stream().frames_dir.name == "hexamind_JTmovXvZnPM"


# --- title / slug derivation ------------------------------------------------


def test_title_unmangles_yt_dlp_filename_substitutions() -> None:
    assert title_from_path(HEXAMIND) == (
        "Hexamind Chess Team | Day 1 | FIDE World Team Blitz Chess Championship 2026"
    )


def test_slug_is_short_lowercase_and_hyphenated() -> None:
    assert slug_from_title("Hexamind Chess Team | Day 1") == "hexamind-chess-team-day"


def test_slug_stops_before_the_length_budget_rather_than_truncating_a_word() -> None:
    assert slug_from_title(
        "Day 2 | FIDE World Team Rapid Chess Championship 2026"
    ) == "day-2-fide-world-team"


def test_slug_keeps_the_first_word_whole_even_past_the_budget() -> None:
    assert slug_from_title("Supercalifragilisticexpialidocious Chess") == (
        "supercalifragilisticexpialidocious"
    )


def test_slug_falls_back_when_the_title_has_no_words() -> None:
    assert slug_from_title("!!! ???") == "stream"
