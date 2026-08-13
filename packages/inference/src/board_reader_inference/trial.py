"""Score a recorded live session against operator-authored ground truth.

Every served run records itself (:mod:`board_reader_core.recording`): the frames
that crossed the wire, and the ``PositionEvent``\\ s the gateway published from
them. That makes a *trial* — a person sitting at a real board with a phone —
into eval data, but only once someone writes down what was actually on the board.
That file is ``truth.json``, and this module is what reads it.

**Truth is a step function over frame seq.** A trial is not a set of photos; it is
one board changing a handful of times. So truth is a list of *segments*, each
holding from its ``from_seq`` until the next one starts::

    {"format": 1,
     "board_id": "phone",
     "segments": [{"from_seq": 0,   "placement": "rnbqkbnr/pppppppp/8/8/..."},
                  {"from_seq": 130, "placement": "rnbqkbnr/pppppppp/8/8/..."}],
     "guard_seqs": 12}

**The guard is the honest part.** A move takes a hand over the board for about a
second, and during it there is no ground truth to speak of — the operator's
segment boundary is one instant, the occlusion is a window. Events whose frame
falls within ``guard_seqs`` frames *after* a boundary are therefore excluded from
scoring rather than counted as errors (12 seqs is ~1.5 s at the 08-13 trial's
7.5 fps). Excluded counts are reported, because a guard that swallows half the
session is a measurement, not a detail.

**Three streams, scored apart.** A published event carries up to three boards and
they fail differently: the raw argmax (``placement``), the hard-constraint repair
(``prior.placement``, :mod:`.prior`) and the temporal debounce
(``settled.placement``, :mod:`.stabilize`). Scoring them side by side is the whole
point — it is what says whether the repair and the debounce earn their keep on a
real stream, which no single-photo eval can answer. Recordings made before the
settled block existed get it **derived** here, by replaying the same
:class:`~board_reader_inference.stabilize.BoardStabilizer` the gateway now runs
live; the report says so, because a replayed number is not a recorded one.

Scoring reads only ``events.ndjson`` — no model, no GPU, a fraction of a second.
``--export-crops`` is the other half: it goes back to the recorded JPEGs and cuts
per-square training crops labelled from the same truth, in the layout
``tools/train`` discovers (``labels_normalized.json``).

    board-reader-inference eval-session recordings/20260813-193308
    board-reader-inference eval-session recordings/ --export-crops data/external/trial
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

from board_reader_core.protocol import GRID, parse_placement, square_name
from board_reader_core.recording import EVENTS_NAME, Recording

from board_reader_inference.evaluate import square_errors
from board_reader_inference.squares import (
    DEFAULT_GEOMETRY,
    CropGeometry,
    crop_squares_photo,
)
from board_reader_inference.stabilize import (
    SETTLE_MIN_READS,
    SETTLE_SECONDS,
    BoardStabilizer,
)

#: Where a session's ground truth lives, inside the recording it describes.
TRUTH_NAME = "truth.json"
TRUTH_FORMAT = 1

#: Frames after a segment boundary that are not scored — the hand-over-the-board
#: window. ~1.5 s at the 08-13 trial's 7.5 fps.
DEFAULT_GUARD_SEQS = 12

#: Keep every Nth scored frame when exporting crops. A trial publishes several
#: reads a second off a board that changes every few *minutes*, so consecutive
#: frames are near-duplicates and a session would otherwise dump tens of
#: thousands of them.
DEFAULT_EVERY = 5

#: Crop size and geometry for the export. This is the deployed cut — ``run_e``'s
#: sidecar is ``headroom=1.6, side_pad=0.15, square_size=224`` — so exported crops
#: are the same pixels the served model sees.
EXPORT_SIZE = 224

#: The manifest ``tools/train/styles.py`` discovers, and the record of what cut it.
MANIFEST_NAME = "labels_normalized.json"
EXPORT_META_NAME = "export.json"

SQUARES = GRID * GRID


class TruthError(ValueError):
    """A ``truth.json`` that cannot be trusted to score anything."""


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TruthSegment:
    """One board, and the frame seq it starts holding from."""

    from_seq: int
    placement: str
    #: ``placement`` parsed once — the export cuts 64 crops per frame off it.
    grid: tuple[tuple[str, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.from_seq < 0:
            raise TruthError(f"from_seq must be >= 0, got {self.from_seq}")
        object.__setattr__(
            self, "grid", tuple(tuple(row) for row in parse_placement(self.placement))
        )


@dataclass(frozen=True)
class SessionTruth:
    """What was on the board, over the whole session."""

    board_id: str
    segments: tuple[TruthSegment, ...]
    guard_seqs: int = DEFAULT_GUARD_SEQS
    starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.board_id:
            raise TruthError("board_id must be a non-empty string")
        if not self.segments:
            raise TruthError("truth has no segments")
        if self.guard_seqs < 0:
            raise TruthError(f"guard_seqs must be >= 0, got {self.guard_seqs}")
        starts = tuple(s.from_seq for s in self.segments)
        if any(b <= a for a, b in pairwise(starts)):
            raise TruthError(f"segment from_seqs must be strictly increasing, got {list(starts)}")
        object.__setattr__(self, "starts", starts)

    @property
    def boundaries(self) -> int:
        """Segment changes — one fewer than the segments, the first is the opening."""
        return len(self.segments) - 1

    def index_for(self, frame_seq: int) -> int:
        """Index of the segment holding ``frame_seq``; ``-1`` before the first."""
        return bisect_right(self.starts, frame_seq) - 1

    def in_guard(self, frame_seq: int) -> bool:
        """Whether ``frame_seq`` sits in the occlusion window after a boundary.

        The *first* segment's start is never guarded: a session opening on a board
        is news, not a move with a hand over it.
        """
        i = self.index_for(frame_seq)
        if i <= 0:
            return False
        return frame_seq - self.starts[i] < self.guard_seqs

    def check_covers(self, first_seq: int) -> None:
        """Reject truth that starts after the stream does.

        A first segment beginning mid-session would silently drop every event
        before it, which reads as "the guard was generous" rather than as the
        authoring mistake it is. ``0`` is always accepted so a truth file can be
        written without looking up the recording's first seq.
        """
        if self.starts[0] not in (0, first_seq):
            raise TruthError(
                f"the first segment starts at seq {self.starts[0]}, but the stream's "
                f"first frame is seq {first_seq} — the first from_seq must be 0 or "
                f"{first_seq}, or the events before it have no truth"
            )

    def to_dict(self) -> dict:
        return {
            "format": TRUTH_FORMAT,
            "board_id": self.board_id,
            "segments": [
                {"from_seq": s.from_seq, "placement": s.placement} for s in self.segments
            ],
            "guard_seqs": self.guard_seqs,
        }


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise TruthError(message)


def load_truth(path: str | Path) -> SessionTruth:
    """Parse and validate a ``truth.json``. Raises :class:`TruthError` on anything off."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise TruthError(f"cannot read {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise TruthError(f"{p} is not valid JSON: {e}") from e
    _require(isinstance(data, dict), f"{p} must hold a JSON object")
    _require(
        data.get("format") == TRUTH_FORMAT,
        f"unsupported truth format {data.get('format')!r} (this reader speaks {TRUTH_FORMAT})",
    )
    board_id = data.get("board_id")
    _require(isinstance(board_id, str) and bool(board_id), "board_id must be a non-empty string")
    raw = data.get("segments")
    _require(isinstance(raw, list) and bool(raw), "segments must be a non-empty list")
    segments: list[TruthSegment] = []
    for i, rec in enumerate(raw):
        _require(isinstance(rec, dict), f"segment {i} must be an object")
        _require("from_seq" in rec and "placement" in rec, f"segment {i} needs from_seq and placement")
        try:
            segments.append(TruthSegment(int(rec["from_seq"]), str(rec["placement"])))
        except (TypeError, ValueError) as e:
            raise TruthError(f"segment {i}: {e}") from e
    guard = data.get("guard_seqs", DEFAULT_GUARD_SEQS)
    try:
        guard_seqs = int(guard)
    except (TypeError, ValueError) as e:
        raise TruthError(f"guard_seqs must be an integer, got {guard!r}") from e
    return SessionTruth(
        board_id=str(board_id), segments=tuple(segments), guard_seqs=guard_seqs
    )


def truth_skeleton(recording: Recording) -> str:
    """A fill-in-the-blanks ``truth.json`` for ``recording``, as JSON text.

    Printed when the file is missing: the two things the operator would otherwise
    have to go dig out of ``frames.ndjson`` — the board id and the first seq — are
    already filled in.
    """
    from board_reader_core.protocol import START_PLACEMENT

    board_id, first_seq = "board", 0
    for rec in recording.frames():
        board_id, first_seq = rec.board_id, rec.seq
        break
    return json.dumps(
        {
            "format": TRUTH_FORMAT,
            "board_id": board_id,
            "segments": [
                {"from_seq": first_seq, "placement": START_PLACEMENT},
                {"from_seq": first_seq + 100, "placement": "<the board after the 1st move>"},
            ],
            "guard_seqs": DEFAULT_GUARD_SEQS,
        },
        indent=2,
    )


# --------------------------------------------------------------------------- #
# Recorded events
# --------------------------------------------------------------------------- #


class Stream(StrEnum):
    """The three boards a published event can carry, scored side by side."""

    RAW = "raw"  # placement — the estimator's per-square argmax
    PRIOR = "prior"  # prior.placement — the hard-constraint repair
    SETTLED = "settled"  # settled.placement — the K-consecutive debounce


@dataclass(frozen=True)
class RecordedEvent:
    """One line of ``events.ndjson``, parsed down to what scoring needs.

    ``square_dist`` is stripped by the writer, so there is no distribution here
    and no per-square confidence — only the boards themselves.
    """

    board_id: str
    seq: int
    frame_seq: int
    capture_ts: float
    confidence: float
    placement: str
    prior_placement: str | None
    settled_placement: str | None


def read_events(recording: Recording, *, board_id: str | None = None) -> list[RecordedEvent]:
    """Every published event in a recording, in the order it was published.

    Unparseable lines are skipped like :meth:`Recording.frames` skips them — a
    session killed mid-write is still worth scoring.
    """
    path = recording.path / EVENTS_NAME
    if not path.exists():
        return []
    out: list[RecordedEvent] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                event = RecordedEvent(
                    board_id=str(d["board_id"]),
                    seq=int(d["seq"]),
                    frame_seq=int(d["frame_seq"]),
                    capture_ts=float(d.get("capture_ts", 0.0)),
                    confidence=float(d.get("confidence", 0.0)),
                    placement=str(d["placement"]),
                    prior_placement=_block_placement(d.get("prior")),
                    settled_placement=_block_placement(d.get("settled")),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if board_id is not None and event.board_id != board_id:
                continue
            out.append(event)
    return out


def _block_placement(block: object) -> str | None:
    """The ``placement`` of an advisory block (``prior``/``settled``), if it has one."""
    if not isinstance(block, dict):
        return None
    placement = block.get("placement")
    return str(placement) if isinstance(placement, str) else None


@dataclass(frozen=True)
class ScoredRead:
    """One event joined to its truth segment, with all three boards resolved."""

    event: RecordedEvent
    segment: int
    raw: str
    prior: str
    settled: str

    def board(self, stream: Stream) -> str:
        if stream is Stream.RAW:
            return self.raw
        if stream is Stream.PRIOR:
            return self.prior
        return self.settled


def derive_settled(events: list[RecordedEvent]) -> list[str]:
    """Replay :class:`BoardStabilizer` over a board's events, in order.

    What the gateway does live, for recordings that predate it doing so. Fed the
    *repaired* placement, exactly as the live path feeds it (see
    :mod:`.stabilize`), and fed **every** event including the ones scoring will
    exclude — the gateway saw those too, and a debounce that skipped them would
    settle differently.
    """
    stabilizer = BoardStabilizer()
    return [
        stabilizer.observe(e.prior_placement or e.placement, e.capture_ts).placement
        for e in events
    ]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass
class SegmentScore:
    """One truth segment's slice of one stream."""

    index: int
    from_seq: int
    to_seq: int | None  # exclusive; None = runs to the end of the session
    placement: str
    events: int = 0
    correct: int = 0
    perfect: int = 0

    @property
    def squares(self) -> int:
        return self.events * SQUARES

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "from_seq": self.from_seq,
            "to_seq": self.to_seq,
            "placement": self.placement,
            "events": self.events,
            "correct": self.correct,
            "squares": self.squares,
            "perfect": self.perfect,
        }


@dataclass
class StreamScore:
    """How one of the three boards did over the whole scored session."""

    stream: Stream
    events: int
    correct: int
    perfect: int
    errors: Counter
    confusions: Counter
    segments: list[SegmentScore]
    #: True when this stream was reconstructed offline rather than read off the
    #: recording. Only ever the settled stream, and only for old recordings.
    derived: bool = False

    @property
    def squares(self) -> int:
        return self.events * SQUARES

    @property
    def accuracy(self) -> float:
        return self.correct / self.squares if self.squares else 0.0

    @property
    def perfect_rate(self) -> float:
        return self.perfect / self.events if self.events else 0.0

    def to_dict(self) -> dict:
        return {
            "stream": self.stream.value,
            "derived": self.derived,
            "events": self.events,
            "squares": self.squares,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "perfect": self.perfect,
            "perfect_rate": round(self.perfect_rate, 4),
            "errors_by_kind": dict(self.errors),
            "confusions": dict(self.confusions.most_common()),
            "per_segment": [s.to_dict() for s in self.segments],
        }


def _score_stream(
    stream: Stream, reads: list[ScoredRead], truth: SessionTruth, *, derived: bool
) -> StreamScore:
    segments = [
        SegmentScore(
            index=i,
            from_seq=s.from_seq,
            to_seq=truth.starts[i + 1] if i + 1 < len(truth.segments) else None,
            placement=s.placement,
        )
        for i, s in enumerate(truth.segments)
    ]
    errors: Counter = Counter()
    confusions: Counter = Counter()
    correct = perfect = 0
    for read in reads:
        seg = segments[read.segment]
        misses = square_errors(truth.segments[read.segment].placement, read.board(stream))
        hit = SQUARES - len(misses)
        correct += hit
        seg.events += 1
        seg.correct += hit
        if not misses:
            perfect += 1
            seg.perfect += 1
        for m in misses:
            errors[m.kind] += 1
            confusions[f"{m.truth}->{m.predicted}"] += 1
    return StreamScore(
        stream=stream,
        events=len(reads),
        correct=correct,
        perfect=perfect,
        errors=errors,
        confusions=confusions,
        segments=segments,
        derived=derived,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class CropExport:
    """What ``--export-crops`` wrote."""

    directory: Path
    style: str
    geometry: CropGeometry
    every: int
    frames: int
    crops: int
    class_counts: Counter
    skipped_no_corners: int

    def to_dict(self) -> dict:
        return {
            "directory": str(self.directory),
            "style": self.style,
            "geometry": self.geometry.to_dict(),
            "every": self.every,
            "frames": self.frames,
            "crops": self.crops,
            "class_counts": dict(self.class_counts.most_common()),
            "skipped_no_corners": self.skipped_no_corners,
        }


@dataclass
class SessionReport:
    """A whole trial, scored — and the reads that were scored, for the export."""

    session: str
    board_id: str
    truth: SessionTruth
    streams: list[StreamScore]
    reads: list[ScoredRead]
    events_total: int
    excluded_guard: int
    excluded_untruthed: int
    excluded_other_board: int
    prior_missing: int
    settled_derived: bool
    export: CropExport | None = None

    @property
    def events_scored(self) -> int:
        return len(self.reads)

    def stream(self, stream: Stream) -> StreamScore:
        return next(s for s in self.streams if s.stream is stream)

    def to_dict(self) -> dict:
        return {
            "session": self.session,
            "board_id": self.board_id,
            "truth": self.truth.to_dict(),
            "events": {
                "total": self.events_total,
                "scored": self.events_scored,
                "excluded_guard": self.excluded_guard,
                "excluded_untruthed": self.excluded_untruthed,
                "excluded_other_board": self.excluded_other_board,
                "prior_missing": self.prior_missing,
            },
            "settled_derived": self.settled_derived,
            "streams": [s.to_dict() for s in self.streams],
            "export": self.export.to_dict() if self.export else None,
        }

    # -- rendering ---------------------------------------------------------- #

    def _render_header(self) -> list[str]:
        t = self.truth
        lines = [
            f"session:   {self.session} · board {self.board_id!r}",
            f"truth:     {len(t.segments)} segment(s), {t.boundaries} boundary change(s)"
            f" · guard {t.guard_seqs} seqs after each",
            f"events:    {self.events_total} published · {self.events_scored} scored · "
            f"{self.excluded_guard + self.excluded_untruthed + self.excluded_other_board} excluded"
            + _excluded_detail(self),
        ]
        if self.settled_derived:
            lines.append(
                f"settled:   DERIVED here (BoardStabilizer, {SETTLE_SECONDS}s / "
                f">={SETTLE_MIN_READS} reads) — this recording predates the settled "
                "block, so that row is what the gateway would publish today, not "
                "what it published then"
            )
        if self.prior_missing:
            lines.append(
                f"prior:     {self.prior_missing} event(s) carried no repair; the "
                "prior row falls back to the raw board for those"
            )
        return lines

    def _render_streams(self) -> list[str]:
        lines = ["", "stream       per-square              perfect boards"]
        for s in self.streams:
            mark = "*" if s.derived else " "
            lines.append(
                f"  {s.stream.value:<9}{mark}{s.correct:>6}/{s.squares:<7}"
                f"{100.0 * s.accuracy:5.1f}%   {s.perfect:>4}/{s.events:<6}"
                f"{100.0 * s.perfect_rate:5.1f}%"
            )
        if any(s.derived for s in self.streams):
            lines.append("  * derived offline, not read off the recording")
        return lines

    def _render_errors(self) -> list[str]:
        lines: list[str] = []
        for s in self.streams:
            if not s.errors:
                lines += ["", f"{s.stream.value}: no wrong squares"]
                continue
            lines += ["", f"{s.stream.value} · errors by kind"]
            lines += [f"  {kind:<14}{n:>5}" for kind, n in s.errors.most_common()]
            lines.append(f"{s.stream.value} · top confusions")
            lines += [f"  {pair:<14}{n:>5}" for pair, n in s.confusions.most_common(6)]
        return lines

    def _render_segments(self) -> list[str]:
        lines = [
            "",
            "per segment (per-square accuracy, then perfect boards)",
            "   #  from_seq      n" + "".join(f"{s.stream.value:>22}" for s in self.streams),
        ]
        for i, seg in enumerate(self.streams[0].segments):
            cells = ""
            for s in self.streams:
                cell = s.segments[i]
                acc = 100.0 * cell.correct / cell.squares if cell.squares else 0.0
                cells += f"{acc:>10.1f}%{cell.perfect:>5}/{cell.events:<5}"
            lines.append(f"  {i:>2}{seg.from_seq:>10}{seg.events:>7}{cells}".rstrip())
            lines.append(f"      {seg.placement}")
        return lines

    def render(self) -> str:
        if not self.reads:
            return "\n".join(
                [
                    *self._render_header(),
                    "",
                    "no events left to score — every one was excluded or off-board",
                ]
            )
        lines = self._render_header()
        lines += self._render_streams()
        lines += self._render_errors()
        lines += self._render_segments()
        if self.export is not None:
            e = self.export
            lines += [
                "",
                f"crops:     {e.crops} from {e.frames} frame(s) (every {e.every}th scored) "
                f"-> {e.directory}",
                f"           style {e.style!r}, geometry {e.geometry.to_dict()}"
                + (
                    f", {e.skipped_no_corners} frame(s) skipped (no corners)"
                    if e.skipped_no_corners
                    else ""
                ),
            ]
        return "\n".join(lines)


def _excluded_detail(report: SessionReport) -> str:
    bits = [
        (report.excluded_guard, "in a boundary guard"),
        (report.excluded_untruthed, "before the first segment"),
        (report.excluded_other_board, "on another board"),
    ]
    present = [f"{n} {what}" for n, what in bits if n]
    return f"  ({', '.join(present)})" if present else ""


def score_session(recording: Recording, truth: SessionTruth) -> SessionReport:
    """Join a recording's events to ``truth`` and score all three streams."""
    all_events = read_events(recording)
    mine = [e for e in all_events if e.board_id == truth.board_id]
    if not mine:
        boards = sorted({e.board_id for e in all_events})
        raise TruthError(
            f"no events for board {truth.board_id!r} in {recording.path}"
            + (f" — it recorded {boards}" if boards else " — it recorded no events")
        )
    truth.check_covers(mine[0].frame_seq)

    # Derived as one homogeneous stream or not at all: a settled row that is half
    # recorded and half replayed would not be a measurement of either.
    settled_derived = any(e.settled_placement is None for e in mine)
    settled = derive_settled(mine) if settled_derived else [
        e.settled_placement or e.placement for e in mine
    ]

    reads: list[ScoredRead] = []
    excluded_guard = excluded_untruthed = 0
    prior_missing = 0
    for event, settled_placement in zip(mine, settled):
        index = truth.index_for(event.frame_seq)
        if index < 0:
            excluded_untruthed += 1
            continue
        if truth.in_guard(event.frame_seq):
            excluded_guard += 1
            continue
        if event.prior_placement is None:
            prior_missing += 1
        reads.append(
            ScoredRead(
                event=event,
                segment=index,
                raw=event.placement,
                prior=event.prior_placement or event.placement,
                settled=settled_placement,
            )
        )

    streams = [
        _score_stream(Stream.RAW, reads, truth, derived=False),
        _score_stream(Stream.PRIOR, reads, truth, derived=False),
        _score_stream(Stream.SETTLED, reads, truth, derived=settled_derived),
    ]
    return SessionReport(
        session=str(recording.path),
        board_id=truth.board_id,
        truth=truth,
        streams=streams,
        reads=reads,
        events_total=len(all_events),
        excluded_guard=excluded_guard,
        excluded_untruthed=excluded_untruthed,
        excluded_other_board=len(all_events) - len(mine),
        prior_missing=prior_missing,
        settled_derived=settled_derived,
    )


# --------------------------------------------------------------------------- #
# Crop export
# --------------------------------------------------------------------------- #


def export_crops(
    recording: Recording,
    truth: SessionTruth,
    reads: list[ScoredRead],
    out_dir: str | Path,
    *,
    every: int = DEFAULT_EVERY,
    geometry: CropGeometry = DEFAULT_GEOMETRY,
    size: int = EXPORT_SIZE,
) -> CropExport:
    """Cut per-square training crops from the scored frames, labelled from truth.

    Written as a crop-level ``labels_normalized.json`` style — the schema
    ``tools/train/styles.py`` discovers under ``data/external`` and
    ``tools/train/cropcache.py`` turns into shards. Only *scored* reads are
    eligible, so nothing a guard window excluded can be labelled with a board
    that was not there yet.

    ``geometry`` defaults to the deployed cut (``run_e``'s sidecar), and is
    recorded in ``export.json`` beside the manifest, because a crop is only
    reusable if what cut it is known.
    """
    import cv2

    if every < 1:
        raise ValueError(f"--every must be >= 1, got {every}")
    out = Path(out_dir)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    wanted = {r.event.frame_seq: r for r in reads[::every]}
    records: list[dict] = []
    class_counts: Counter = Counter()
    frames = skipped = 0
    for recorded in recording.frames(board_id=truth.board_id):
        read = wanted.pop(recorded.seq, None)
        if read is None:
            continue
        frame = recorded.frame()
        if frame.corners is None:
            skipped += 1
            continue
        grid = truth.segments[read.segment].grid
        squares = crop_squares_photo(
            frame.image, frame.corners, geometry=geometry, size=size
        )
        for i, crop in enumerate(squares):
            r, c = divmod(i, GRID)
            cls = grid[r][c]
            name = f"{recorded.seq:06d}_{square_name(r, c)}_{cls}.jpg"
            cv2.imwrite(str(crops_dir / name), crop)
            records.append(
                {"image": f"crops/{name}", "piece": cls, "chess_set": truth.board_id}
            )
            class_counts[cls] += 1
        frames += 1

    (out / MANIFEST_NAME).write_text(json.dumps(records, indent=1), encoding="utf-8")
    export = CropExport(
        directory=out,
        # What tools/train will call it: <manifest dir>_<chess_set>.
        style=f"{out.name}_{truth.board_id}",
        geometry=geometry,
        every=every,
        frames=frames,
        crops=len(records),
        class_counts=class_counts,
        skipped_no_corners=skipped,
    )
    (out / EXPORT_META_NAME).write_text(
        json.dumps(
            {
                "source_recording": str(recording.path),
                "square_size": size,
                **export.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return export


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(
    recording: Recording,
    *,
    truth: SessionTruth | None = None,
    json_out: str | None = None,
    export_dir: str | None = None,
    every: int = DEFAULT_EVERY,
) -> SessionReport:
    """Score ``recording`` against its ``truth.json``, optionally exporting crops."""
    report = score_session(recording, truth or load_truth(recording.path / TRUTH_NAME))
    if export_dir and report.reads:
        report.export = export_crops(
            recording, report.truth, report.reads, export_dir, every=every
        )
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
