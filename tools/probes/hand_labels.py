"""Hand labels for the hand-signal probe, typed by eye from hand_sheets.py's
contact sheets (2026-11-24). Key: take -> {row: label}, where row is the 0-based
line of the take's frames.ndjson and the label is

* ``H`` — a hand or arm covers part of the playing area (any square);
* ``E`` — a hand is in frame but only beyond the board's edge;
* ``S`` — no hand, but an arm's shadow falls on the board;
* ``N`` — nothing foreign on the board.

The sample is hand_sheets.sample() with seed 20261124: per take up to 12 rows near
truth boundaries (or inside motion bursts, for takes without truth) plus 18 uniform.
"""

from __future__ import annotations

from typing import Literal

Label = Literal["H", "E", "S", "N"]


def _parse(text: str) -> dict[int, Label]:
    return {int(tok[:-1]): tok[-1] for tok in text.split()}  # type: ignore[misc]


LABELS: dict[str, dict[int, Label]] = {
    "20260813-193308": _parse(
        "8N 14N 36N 69N 70N 81N 192N 193N 194N 196N 197N 228S 244N 274S 286N 291N "
        "293N 313H 315H 328N 335N 342N 467N 611N 620H 623H 628H 642N 644N 678N"
    ),
    "20260827-175539": _parse(
        "38N 45N 74H 100H 104H 105H 128N 129N 175N 261H 262H 264H 265H 266H 286N "
        "288N 305N 316H 326H 332H 335N 349H 350H 352H 377H 390N 416N 420N 450N 472N"
    ),
    "20260828-111421": _parse(
        "11N 24N 36N 45N 49H 53H 77N 80N 87N 103H 109H 110H 116H 120N 127N 133H "
        "139H 142H 144H 162H 163H 169E 177H 180H 181H 185N 189N 196N 226N 228N"
    ),
    "20260827-181802": _parse(
        "7N 36N 37N 52H 54H 87H 90H 93H 101H 103H 106H 114E 118H 127E 141N 150H "
        "151H 156E 159N 162N 176N 203H 206H 208H 213H 216H 240N 244H 247N 264E"
    ),
    "20260827-163352": _parse(
        "8N 12N 31N 35H 39H 41H 70H 99H 107H 108H 111H 145N 150H 151H 159H 165H "
        "179N 186H 189H 220E 222H 229H 235H 237H 240H 241H 257N 264N 270N 309N"
    ),
    "20260827-170333": _parse(
        "47N 72N 99N 148N 165N 259N 265H 269H 309N 311N 314N 324H 337N 362N 398H "
        "401H 406N 416N 460H 461H 494N 502H 503H 511N 628N 666H 674H 683H 700N 712N"
    ),
    "20260827-172736": _parse(
        "45H 72N 126H 128H 199H 220N 267H 279N 296N 471H 490H 538N 677H 679H 749N "
        "752H 755H 771N 820N 829N 899H 901H 905H 949N 954N 1093N 1106N 1242E 1358N 1369H"
    ),
    "20261124-204025": _parse(
        "12N 13N 20N 22H 28H 34H 36H 42H 47H 73N 78N 81N 88H 90H 96H 116H 117E "
        "119H 120H 123H 136H 155N 160N 162N 185N 215E 221N 225E 227N 237N"
    ),
    "20261124-221950": _parse(
        "6N 7N 16H 65H 69H 74N 92N 114H 123N 172H 183N 218H 224N 276H 303N 328E "
        "373H 386N 417N 446H 469H 487H 501E 510H 512H 532H 536H 545H 559H 564H"
    ),
}
