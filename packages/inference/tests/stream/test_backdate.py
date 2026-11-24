"""Tests for the pixel back-dating of ``changed_utc`` (:mod:`board_reader_inference.stream.backdate`).

The claims: the search returns the start of the unbroken run of frames that look like
the new reference on the changed pixels, ending at the new read; a hand over the
square is neither board, so it ends the run; a frame that flickers to "new" before a
hand frame does not count; nothing to judge (no frames, references alike, another
frame size) back-dates nothing. And the clock finds its two references in its own
read log and keeps the back-dated time until the settled board changes again.

The frames are synthetic: a two-level checkerboard with a dark "piece" block, so the
change between the references is exactly the two blocks.
"""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.stream.backdate import (
    HOLD_SECONDS,
    ChangeClock,
    FrameRing,
    HeldFrame,
    first_new_frame,
    grey_of,
)

E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"

OLD, NEW, HAND = "old", "new", "hand"


def _image(kind: str) -> np.ndarray:
    """A 96 x 160 BGR board: the piece on the left square (old), on the right (new),
    or a bright hand over both."""
    yy, xx = np.mgrid[0:96, 0:160]
    image = np.where(((yy // 16) + (xx // 16)) % 2 == 0, 90, 150).astype(np.uint8)
    if kind in (OLD, NEW):
        x = 20 if kind == OLD else 100
        image[36:60, x : x + 24] = 30
    else:
        image[20:80, 10:140] = 225
    return np.repeat(image[:, :, None], 3, axis=2)


def _held(seq: int, kind: str) -> HeldFrame:
    return HeldFrame(seq, float(seq), grey_of(_image(kind)))


def _search(kinds: str) -> int | None:
    """``kinds`` is the window, one letter per frame (o/n/h); the seq of the answer."""
    names = {"o": OLD, "n": NEW, "h": HAND}
    window = [_held(i + 1, names[k]) for i, k in enumerate(kinds)]
    found = first_new_frame(window, _held(0, OLD), _held(len(kinds) + 1, NEW))
    return None if found is None else found.seq


def test_the_answer_starts_the_run_of_new_frames_that_ends_at_the_read():
    assert _search("oonnn") == 3


def test_a_hand_still_over_the_square_is_not_the_new_board():
    assert _search("ohhnn") == 4


def test_a_flicker_to_new_before_the_hand_left_does_not_count():
    assert _search("nhn") == 3


def test_nothing_to_back_date_to():
    assert _search("") is None
    assert _search("onh") is None  # the frame right before the read is a hand


def test_references_that_look_alike_or_differ_in_size_judge_nothing():
    old = _held(0, NEW)
    assert first_new_frame([_held(1, NEW)], old, _held(2, NEW)) is None
    small = HeldFrame(1, 1.0, np.zeros((8, 8), np.uint8))
    assert first_new_frame([_held(1, NEW)], small, _held(2, NEW)) is None
    # A frame of another size ends the search: only the later frame counts.
    assert first_new_frame([_held(1, NEW), small, _held(3, NEW)], _held(0, OLD), _held(4, NEW)).seq == 3


def test_auto_exposure_does_not_hide_the_new_board():
    darker = (_image(NEW).astype(np.float32) * 0.6).astype(np.uint8)
    window = [HeldFrame(1, 1.0, grey_of(darker))]
    assert first_new_frame(window, _held(0, OLD), _held(2, NEW)).seq == 1


def test_the_ring_holds_a_bounded_time_and_restarts_with_the_stream():
    ring = FrameRing()
    for seq in range(30):
        ring.push(seq, seq * 1.0, _image(OLD))
    held = ring.snapshot()
    assert held[-1].seq == 29 and held[-1].ts - held[0].ts <= HOLD_SECONDS
    ring.push(0, 100.0, _image(OLD))  # seq restarts: a new streaming session
    assert [f.seq for f in ring.snapshot()] == [0]


def _drive(clock: ChangeClock, ring: FrameRing, seq: int, read: str, settled: str, changed: float):
    return clock.observe(seq, float(seq), read, settled, changed, ring.snapshot())


def test_the_clock_back_dates_a_commit_and_keeps_it_while_the_board_holds():
    """Frames 0-2 old, 3-4 hand, 5-9 new; reads at 0, 2, 7, 8, 9; E4 settles at 8
    from the run that began at 7. The frames between the last old read (2) and the
    first new one (7) put the change at 5."""
    ring, clock = FrameRing(), ChangeClock()
    kinds = [OLD] * 3 + [HAND] * 2 + [NEW] * 5
    published = {}
    for seq, kind in enumerate(kinds):
        ring.push(seq, float(seq), _image(kind))
        if seq in (0, 2):
            published[seq] = _drive(clock, ring, seq, START_PLACEMENT, START_PLACEMENT, 0.0)
        elif seq == 7:
            published[seq] = _drive(clock, ring, seq, E4, START_PLACEMENT, 0.0)
        elif seq in (8, 9):
            published[seq] = _drive(clock, ring, seq, E4, E4, 7.0)
    assert published == {0: 0.0, 2: 0.0, 7: 0.0, 8: 5.0, 9: 5.0}


def test_the_clock_keeps_the_stabilizers_time_without_frames_to_search():
    ring, clock = FrameRing(), ChangeClock()
    for seq, kind in enumerate([OLD, NEW, NEW]):
        ring.push(seq, float(seq), _image(kind))
    assert _drive(clock, ring, 0, START_PLACEMENT, START_PLACEMENT, 0.0) == 0.0
    assert _drive(clock, ring, 1, E4, START_PLACEMENT, 0.0) == 0.0
    assert _drive(clock, ring, 2, E4, E4, 1.0) == 1.0  # reads 0 and 1 are adjacent frames
