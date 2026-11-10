"""Opt-in lease event records for an unmerged diagnostic probe."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def log_lease(event: str, lock: object) -> None:
    try:
        record = {
            "event": event,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "time_ns": time.time_ns(),
            "path": str(getattr(lock, "lock_file", "")),
            "locked": getattr(lock, "is_locked", None),
            "stack": traceback.format_stack(),
        }
        folder = Path(os.environ["DINKSTER_LEASE_PROBE_EVIDENCE"])
        with (folder / f"process-{os.getpid()}.jsonl").open("a", encoding="ascii") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as error:
        print(f"lease logging failed: {error!r}", file=sys.stderr)
