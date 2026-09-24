"""Run a Dinkster workflow benchmark while retaining every HTTP response."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools import workflow_benchmark as benchmark


def output_root(argv: Sequence[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument == "--output" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--output="):
            return Path(argument.split("=", 1)[1])
    raise ValueError("missing --output")


def transcript_request(
    root: Path,
    base: str,
    path: str,
    body: Any = None,
    *,
    timeout: float = 10,
) -> Any:
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    query = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.time_ns()
    try:
        with urllib.request.urlopen(query, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
        _append_transcript(root, query, path, started, status, raw)
        raise
    _append_transcript(root, query, path, started, status, raw)
    return json.loads(raw)


def _append_transcript(
    root: Path,
    query: urllib.request.Request,
    path: str,
    started: int,
    status: int,
    raw: bytes,
) -> None:
    record = {
        "start_epoch_ns": started,
        "method": query.get_method(),
        "path": path,
        "status": status,
        "response_utf8": raw.decode("utf-8", "replace"),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "http-transcript.jsonl").open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = output_root(arguments)

    def request(base: str, path: str, body: Any = None, *, timeout: float = 10) -> Any:
        return transcript_request(root, base, path, body, timeout=timeout)

    benchmark.request = request
    return benchmark.main("dinkster", arguments)


if __name__ == "__main__":
    raise SystemExit(main())
