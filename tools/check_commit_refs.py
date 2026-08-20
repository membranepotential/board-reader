#!/usr/bin/env python3
"""Find commit SHAs cited in tracked files that no longer resolve.

Prose in this repo cites commits — ``STATUS.md`` credits work to them, docstrings
point at the change that explains a workaround. A history rewrite (or a rebased
branch, or a dropped commit) turns every one of those into a dangling reference
that nothing else catches: they are prose, so no test and no linter reads them.

Run it bare to audit the working tree::

    uv run tools/check_commit_refs.py            # exit 1 if anything dangles

Give it the pre-rewrite history to get the *replacement* for each dangling ref,
and optionally write them back::

    uv run tools/check_commit_refs.py --baseline ../old-repo.bundle
    uv run tools/check_commit_refs.py --baseline ../old-repo.bundle --apply

A baseline turns guesses into facts. Without one a dangling ref is only
"unresolvable"; with one, a ref that resolved *there* and not *here* is
definitively stale, and the rewrite is recoverable by pairing the two histories
in topological order — filter-branch preserves order, trees, and messages, so
the Nth commit of the old history is the Nth commit of the new one.

## Why the matching is fussy

A bare ``[0-9a-f]{7,40}`` scan over this repo returns ~400 hits and 7 of them are
commits. The rest are lockfile byte counts, learning ids, ``20260813``-style
date stamps and constants like ``4294967296``. The filters below exist to cut
that noise to zero; each one is load-bearing, so read the comment before
relaxing any of them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# A cited SHA is 7-40 hex chars. Word-bounded so `abc1234` inside a longer
# token (a base64 integrity hash, a filename) never matches.
SHA_RE = re.compile(rb"\b[0-9a-f]{7,40}\b")

# Files that contain hex-like noise and never cite a commit. Lockfiles are the
# whole problem: uv.lock's byte counts are all-decimal, which is valid hex.
SKIP_GLOBS = ("uv.lock", "pnpm-lock.yaml", "*.png", "*.csv")

# `.learnings.ndjson` keys every record by a 12-hex id, and records cite each
# other by that id in prose ("CORRECTION to the ... learning (85aab09c98c4)"),
# which is shape-identical to a short SHA. Reading the ids out of the store beats
# hardcoding them: the ignore set stays correct as learnings are added.
LEARNINGS_STORE = Path(".learnings.ndjson")

# Hex that IS a hash, just not a git one. Each is a third-party artifact
# fingerprint that will never resolve and must not be reported forever.
KNOWN_NON_COMMITS = frozenset(
    {
        "2a0d100e",  # torchvision checkpoint suffix, Stylized-ImageNet ResNet-50
        "99b5c721",  # ChessReD archive checksum (tools/build_exemplar_bank_chessred.py)
        "b2900b69a90f",  # ChessReD archive checksum, long form
    }
)


@dataclass(frozen=True, slots=True)
class Citation:
    """One hex token in one file that looks like it names a commit."""

    path: Path
    line: int
    sha: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.sha}"


@dataclass(frozen=True, slots=True)
class Stale(Citation):
    """A citation that resolved in the baseline history but not in this one."""

    replacement: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.sha} -> {self.replacement}"


def git(*args: str, repo: Path | None = None, check: bool = True) -> str:
    cmd = ["git", *(("-C", str(repo)) if repo else ()), *args]
    out = subprocess.run(cmd, capture_output=True, check=check)
    return out.stdout.decode(errors="replace")


def tracked_files(repo: Path) -> Iterator[Path]:
    """Every tracked file git considers text, minus the known-noisy ones."""
    for name in git("ls-files", "-z", repo=repo).split("\0"):
        if not name:
            continue
        path = Path(name)
        if any(path.match(g) for g in SKIP_GLOBS):
            continue
        yield path


def _is_plausible_sha(token: str) -> bool:
    """Reject hex that is really a decimal number.

    Every false positive from `uv.lock` and every `20260813` date stamp is
    all-digits. A commit abbreviation that happens to contain no a-f letter
    exists in theory (~1 in 900 for 7 chars) and would be missed here — an
    acceptable trade for removing ~380 spurious hits.
    """
    return not token.isdigit()


def learning_ids(repo: Path) -> frozenset[str]:
    """Every record id in the learnings store, or empty if there is no store.

    A 12-hex id could in principle collide with a commit abbreviation someone
    cites; at that length the odds are negligible, and the cost is one missed
    dangling ref rather than a wrong rewrite.
    """
    store = repo / LEARNINGS_STORE
    if not store.exists():
        return frozenset()
    ids: set[str] = set()
    for line in store.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            ids.add(record["id"])
    return frozenset(ids)


def scan(repo: Path, paths: Iterable[Path], ignore: frozenset[str]) -> list[Citation]:
    found: list[Citation] = []
    for path in paths:
        try:
            blob = (repo / path).read_bytes()
        except (OSError, ValueError):
            continue
        if b"\0" in blob[:8192]:  # binary
            continue
        for match in SHA_RE.finditer(blob):
            token = match.group().decode()
            if not _is_plausible_sha(token) or token in ignore:
                continue
            line = blob.count(b"\n", 0, match.start()) + 1
            found.append(Citation(path=path, line=line, sha=token))
    return found


def resolve(repo: Path, shas: Sequence[str]) -> dict[str, str | None]:
    """Map each candidate to its full commit SHA, or None if it names no commit."""
    if not shas:
        return {}
    stdin = "\n".join(f"{s}^{{commit}}" for s in shas).encode()
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch-check"],
        input=stdin,
        capture_output=True,
        check=False,
    )
    resolved: dict[str, str | None] = {}
    for sha, line in zip(shas, proc.stdout.decode().splitlines(), strict=False):
        head, _, kind = line.partition(" ")
        resolved[sha] = head if kind.startswith("commit") else None
    return resolved


def rewrite_map(old: Path, new: Path) -> dict[str, str]:
    """Pair two histories positionally to recover an old->new SHA mapping.

    Safe only because a filter-branch rewrite preserves the DAG shape and the
    commit order. Verified, not assumed: the pairing is rejected unless every
    pair agrees on tree hash, author date and subject.
    """
    fmt = "%H%x00%T%x00%aI%x00%s"
    olds = git("log", "--all", "--reverse", "--topo-order", f"--format={fmt}", repo=old)
    news = git("log", "--all", "--reverse", "--topo-order", f"--format={fmt}", repo=new)
    old_rows = [ln.split("\0") for ln in olds.splitlines() if ln]
    new_rows = [ln.split("\0") for ln in news.splitlines() if ln]
    # Work committed since the rewrite extends the tip, so the baseline is a
    # prefix of this history. Anything shorter means the baseline is not this
    # repo's past, and pairing would invent mappings.
    if len(new_rows) < len(old_rows):
        raise SystemExit(
            f"cannot pair histories: baseline has {len(old_rows)} commits but this "
            f"repo has only {len(new_rows)} — the baseline is not this repo's past"
        )
    new_rows = new_rows[: len(old_rows)]
    mapping: dict[str, str] = {}
    for o, n in zip(old_rows, new_rows, strict=True):
        if (o[1], o[2], o[3]) != (n[1], n[2], n[3]):
            raise SystemExit(
                f"cannot pair histories: {o[0][:9]} and {n[0][:9]} differ in "
                f"tree/date/subject — the rewrite changed content, not just identity"
            )
        mapping[o[0]] = n[0]
    return mapping


def apply(repo: Path, stale: Sequence[Stale]) -> int:
    """Replace each stale abbreviation in place, preserving its length."""
    by_path: dict[Path, list[Stale]] = {}
    for s in stale:
        by_path.setdefault(s.path, []).append(s)
    edited = 0
    for path, items in by_path.items():
        target = repo / path
        text = target.read_text()
        for s in items:
            # Keep the author's chosen abbreviation length.
            text = re.sub(rf"\b{s.sha}\b", s.replacement[: len(s.sha)], text)
        target.write_text(text)
        edited += 1
    return edited


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo", type=Path, default=Path.cwd(), help="repo to audit (default: cwd)"
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        help="pre-rewrite history (a bundle or a repo) — turns unresolvable refs into "
        "definitively stale ones and computes their replacements",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="rewrite stale refs in place (needs --baseline)",
    )
    ap.add_argument(
        "--dump-map",
        type=Path,
        metavar="PATH",
        help="write the full old->new SHA table as TSV and exit (needs --baseline). "
        "Keep it somewhere git-ignored: it is how an old SHA from a note, a "
        "bookmark or a stale checkout gets translated once the bundle is gone.",
    )
    args = ap.parse_args(argv)

    if (args.apply or args.dump_map) and not args.baseline:
        ap.error("--apply and --dump-map both need --baseline")

    if args.dump_map:
        old = args.baseline.resolve()
        mapping = rewrite_map(old, args.repo.resolve())
        fmt = "--format=%H%x00%aI%x00%s"
        rows = git("log", "--all", "--reverse", "--topo-order", fmt, repo=old)
        args.dump_map.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_map.open("w") as fh:
            fh.write("# old->new commit SHA map, board-reader author-email rewrite\n")
            fh.write("# Content is byte-identical across it; only identity changed.\n")
            fh.write("# Regenerate: check_commit_refs.py --baseline <old> --dump-map P\n")
            fh.write("old_sha\tnew_sha\tabbrev_old\tabbrev_new\tauthor_date\tsubject\n")
            for line in rows.splitlines():
                if not line:
                    continue
                old_sha, date, subject = line.split("\0")
                new_sha = mapping[old_sha]
                fh.write(
                    f"{old_sha}\t{new_sha}\t{old_sha[:7]}\t{new_sha[:7]}\t"
                    f"{date}\t{subject}\n"
                )
        print(f"wrote {len(mapping)} mappings to {args.dump_map}")
        return 0

    repo = args.repo.resolve()
    ignore = KNOWN_NON_COMMITS | learning_ids(repo)
    citations = scan(repo, tracked_files(repo), ignore)
    here = resolve(repo, sorted({c.sha for c in citations}))
    dangling = [c for c in citations if here.get(c.sha) is None]

    if not dangling:
        print(f"OK — {len({c.sha for c in citations})} cited commits, all resolve.")
        return 0

    if not args.baseline:
        print(f"{len(dangling)} dangling commit reference(s):\n")
        for c in sorted(dangling, key=lambda c: (str(c.path), c.line)):
            print(f"  {c}")
        print("\nRe-run with --baseline <old repo or bundle> to get replacements.")
        return 1

    mapping = rewrite_map(args.baseline.resolve(), repo)
    there = resolve(args.baseline.resolve(), sorted({c.sha for c in dangling}))
    stale, unknown = [], []
    for c in dangling:
        full_old = there.get(c.sha)
        if full_old and full_old in mapping:
            stale.append(Stale(c.path, c.line, c.sha, mapping[full_old]))
        else:
            unknown.append(c)

    if stale:
        print(f"{len(stale)} stale reference(s) — resolved in the baseline, not here:\n")
        for s in sorted(stale, key=lambda s: (str(s.path), s.line)):
            print(f"  {s}")
    if unknown:
        print(f"\n{len(unknown)} unresolvable in both — not from this rewrite:\n")
        for c in sorted(unknown, key=lambda c: (str(c.path), c.line)):
            print(f"  {c}")

    if args.apply and stale:
        print(f"\nrewrote {apply(repo, stale)} file(s).")
        return 0
    return 1 if (stale or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
