"""A stand-in for ``board-reader-edge camera``, for the stream supervisor's tests.

What it does depends on the URL's last path segment:

* ``sleep`` — starts a child of its own (as yt-dlp starts deno), writes both pids
  next to the ``--state`` file and sleeps until it is stopped;
* ``crash`` — says why on stderr and exits with 3;
* ``end`` — exits cleanly, as a video that ended.

Every run appends its argv (JSON) to ``<state>.runs``, so a test can count the starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("url")
parser.add_argument("--node", required=True)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--start", type=float, default=0.0)
args = parser.parse_args()

with Path(f"{args.state}.runs").open("a") as runs:
    runs.write(json.dumps(sys.argv[1:]) + "\n")
action = args.url.rsplit("/", 1)[-1]
if action == "crash":
    print("boom: no such video", file=sys.stderr)
    raise SystemExit(3)
if action == "end":
    raise SystemExit(0)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(f"{args.state}.pids").write_text(f"{os.getpid()} {child.pid}\n")
print("fake edge up", flush=True)
time.sleep(60)
