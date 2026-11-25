"""A whole scored trial, as JSON and as the text the CLI prints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from board_reader_core.protocol import EVIDENCE_TOP_K

from board_reader_inference.measurement.trial.evidence import EvidenceScore
from board_reader_inference.measurement.trial.score import (
    ScoredRead,
    Stream,
    StreamScore,
    TruthFrame,
)
from board_reader_inference.measurement.trial.stats import quantile
from board_reader_inference.measurement.trial.timing import Timing
from board_reader_inference.measurement.trial.truth import SessionTruth
from board_reader_inference.stream.stabilize import SETTLE_MIN_READS, SETTLE_SECONDS


@dataclass
class SessionReport:
    """A whole trial, scored — and the reads that were scored."""

    session: str
    board_id: str
    truth: SessionTruth
    streams: list[StreamScore]
    reads: list[ScoredRead]
    events_total: int
    excluded_guard: int
    excluded_untruthed: int
    excluded_other_board: int
    settled_derived: bool
    evidence: EvidenceScore
    timing: Timing
    frame: TruthFrame
    #: Where :func:`run` wrote the JSON, once it has.
    written_to: Path | None = None

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
            },
            "settled_derived": self.settled_derived,
            "truth_frame": self.frame.to_dict(),
            "streams": [s.to_dict() for s in self.streams],
            "evidence": self.evidence.to_dict(),
            "timing": self.timing.to_dict(),
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
        f = self.frame
        if f.reads and f.verdict:
            lines.append(
                f"frame:     ! WARNING {f.by_rotation[f.verdict]}/{f.reads} raw reads agree "
                f"with the truth best turned {90 * f.verdict}° — the truth and the "
                "reads are not in the same frame"
            )
        elif f.reads:
            lines.append(f"frame:     {f.by_rotation[0]}/{f.reads} raw reads agree with the truth unturned")
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

    def _render_evidence(self) -> list[str]:
        e = self.evidence
        if not e.recorded:
            return [
                "",
                "evidence:  none recorded — this session predates the top-"
                f"{EVIDENCE_TOP_K} digest. Re-run the trial to capture why each "
                "square read the way it did.",
            ]
        covered = f"{e.events}/{e.events_total} scored events carried a digest"
        lines = ["", f"raw · evidence on the {e.wrong} wrong squares ({covered})"]
        if not e.wrong:
            return lines
        lines += [
            f"  truth was the runner-up   {e.truth_runner_up:>5}/{e.wrong}"
            f"{100.0 * e.truth_runner_up / e.wrong:>7.1f}%",
            f"  truth outside the top {EVIDENCE_TOP_K}   {e.wrong - e.truth_ranked:>5}/{e.wrong}"
            f"{100.0 * (e.wrong - e.truth_ranked) / e.wrong:>7.1f}%",
        ]
        if e.margins:
            lines.append(
                f"  margin p(read)-p(truth)   median {median(e.margins):.3f} · "
                f"p90 {quantile(e.margins, 0.9):.3f} · max {max(e.margins):.3f}"
            )
        lines.append("  by confusion          n   truth ranked   median margin")
        for c in e.by_confusion[:6]:
            margin = f"{c.median_margin:.3f}" if c.margins else "     —"
            lines.append(f"    {c.pair:<14}{c.n:>5}{c.truth_ranked:>13}{margin:>15}")
        return lines

    def _render_timing(self) -> list[str]:
        t = self.timing
        lines = ["", "timing"]
        if t.frames:
            lines.append(
                f"  input      {t.frames} frame(s) over {t.frames_span:.1f}s · "
                f"{t.fps:.2f} fps"
            )
        if t.events:
            lines.append(
                f"  reads      {t.events} published over {t.events_span:.1f}s · "
                f"{t.reads_per_s:.2f} reads/s"
                + (f" · 1 read per {t.frames_per_read:.1f} frames" if t.frames else "")
            )
        if t.latencies:
            lines.append(
                f"  latency    capture->publish  median {median(t.latencies):.2f}s · "
                f"p90 {quantile(t.latencies, 0.9):.2f}s · max {max(t.latencies):.2f}s"
            )
        else:
            lines.append("  latency    not stamped in this session")
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
        lines += self._render_evidence()
        lines += self._render_timing()
        lines += self._render_segments()
        if self.written_to is not None:
            lines += ["", f"report:    {self.written_to}"]
        return "\n".join(lines)


def _excluded_detail(report: SessionReport) -> str:
    bits = [
        (report.excluded_guard, "in a boundary guard"),
        (report.excluded_untruthed, "before the first segment"),
        (report.excluded_other_board, "on another board"),
    ]
    present = [f"{n} {what}" for n, what in bits if n]
    return f"  ({', '.join(present)})" if present else ""
