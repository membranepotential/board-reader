"""Draw numbered contact sheets of sampled frames, for labelling hands by eye.

    uv run python tools/probes/hand_sheets.py OUT_DIR [--rows TAKE:ROW,ROW ...]

Without ``--rows`` it draws the probe's stratified sample (``sample()``): per take,
frames near truth boundaries (where hands are likely) plus frames drawn uniformly.
A row is the 0-based line of the take's frames.ndjson. Each tile is captioned
``<take tail>:<row>``; the labels typed from these sheets live in hand_labels.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAKES = (
    "20260813-193308",
    "20260827-175539",
    "20260828-111421",
    "20260827-181802",
    "20260827-163352",
    "20260827-170333",
    "20260827-172736",
    "20261124-204025",
    "20261124-221950",
)
TILE_W, COLS, ROWS = 360, 4, 4


def frame_lines(take: str) -> list[dict]:
    text = (ROOT / "recordings" / take / "frames.ndjson").read_text()
    return [json.loads(s) for s in text.splitlines() if s.strip()]


def sample(
    take: str, rng: np.random.Generator, near: int = 12, uniform: int = 18
) -> list[int]:
    """Rows near truth boundaries (seq from_seq-3 .. from_seq+8) and uniform rows."""
    lines = frame_lines(take)
    n = len(lines)
    truth_path = ROOT / "recordings" / take / "truth.json"
    pool: list[int] = []
    if truth_path.exists():
        starts = [s["from_seq"] for s in json.loads(truth_path.read_text())["segments"][1:]]
        seq_rows = {ln["seq"]: i for i, ln in enumerate(lines)}  # last session wins
        pool = sorted(
            {seq_rows[s + d] for s in starts for d in range(-3, 9) if s + d in seq_rows}
        )
    else:
        # No truth: frames inside motion bursts stand in for "near a move".
        pool = [i for i, ln in enumerate(lines) if ln.get("burst_id")]
    chosen = (
        set(rng.choice(pool, size=min(near, len(pool)), replace=False).tolist())
        if pool
        else set()
    )
    rest = [i for i in range(n) if i not in chosen]
    chosen |= set(rng.choice(rest, size=min(uniform, len(rest)), replace=False).tolist())
    return sorted(chosen)


def tile(take: str, row: int, line: dict) -> np.ndarray:
    image = cv2.imread(str(ROOT / "recordings" / take / line["jpeg"]))
    h = int(image.shape[0] * TILE_W / image.shape[1])
    t = cv2.resize(image, (TILE_W, h), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 0, 300 - h if h < 300 else 0, 0, 0, cv2.BORDER_CONSTANT)[:300]
    label = f"{take[-6:]}:{row}"
    cv2.rectangle(t, (0, 0), (150, 26), (0, 0, 0), -1)
    cv2.putText(t, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return t


def sheets(items: list[tuple[str, int]], out: Path, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict]] = {}
    per = COLS * ROWS
    for k in range(0, len(items), per):
        tiles = []
        for take, row in items[k : k + per]:
            lines = cache.setdefault(take, frame_lines(take))
            tiles.append(tile(take, row, lines[row]))
        while len(tiles) < per:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * COLS : (r + 1) * COLS]) for r in range(ROWS)])
        cv2.imwrite(
            str(out / f"{prefix}-{k // per:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )


def main() -> None:
    out = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--rows":
        items = []
        for spec in sys.argv[3:]:
            take, rows = spec.split(":")
            items += [(take, int(r)) for r in rows.split(",")]
        sheets(items, out, "extra")
        return
    rng = np.random.default_rng(20261124)
    items = [(take, row) for take in TAKES for row in sample(take, rng)]
    (out / "sample.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(json.dumps(items))
    sheets(items, out, "sheet")
    print(f"{len(items)} frames")


if __name__ == "__main__":
    main()
