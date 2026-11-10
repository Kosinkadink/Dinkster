"""Bounded repetition of the original three-file Windows pytest process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("holder", "logging"), required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    command = [
        args.uv,
        "run",
        "--no-sync",
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/test_produced_video_assets.py",
        "tests/test_environment_downloads.py",
        "tests/test_asset_cleanup.py",
    ]
    for attempt in range(1, args.attempts + 1):
        folder = args.evidence / f"attempt-{attempt}"
        folder.mkdir()
        env = {**os.environ, "DINKSTER_LEASE_PROBE_EVIDENCE": str(folder)}
        started = time.time()
        with (folder / "pytest.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    [*command, "--basetemp", str(folder / "temporary")],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                ).returncode
            except subprocess.TimeoutExpired:
                result = 124
        record = {
            "arm": args.arm,
            "attempt": attempt,
            "exit": result,
            "elapsed_seconds": time.time() - started,
            "command": command,
        }
        with (args.evidence / "attempts.jsonl").open("a", encoding="ascii") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if result:
            print((folder / "pytest.log").read_text(encoding="utf-8"), flush=True)
            raise SystemExit(result)


if __name__ == "__main__":
    main()
