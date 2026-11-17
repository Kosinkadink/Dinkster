"""The helper's stdout belongs exclusively to the pipe protocol."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from ._records import FORMAT, Records, object_value
from .client import API_VERSION, MAX_MESSAGE, LibraryError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    records: Records | None = None
    try:
        while line := sys.stdin.buffer.readline(MAX_MESSAGE + 1):
            try:
                if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                    raise LibraryError("message", "library request is too large or incomplete")
                request = object_value(json.loads(line), "request")
                api = request.get("api")
                if type(api) is not int or api != API_VERSION:
                    raise LibraryError(
                        "api",
                        f"Engine requests library API {api}; helper offers "
                        f"{API_VERSION} to {API_VERSION}.",
                    )
                if records is None:
                    records = Records(args.library)
                operation = request.get("operation")
                if not isinstance(operation, str):
                    raise LibraryError("invalid", "operation must be a string")
                arguments = object_value(request.get("arguments"), "arguments")
                if operation == "hello":
                    result: dict[str, object] = {
                        "apiMin": API_VERSION,
                        "apiMax": API_VERSION,
                        "capabilities": ["projects", "changes"],
                        "format": FORMAT,
                        "attached": records.attachments(args.library),
                    }
                else:
                    result = records.dispatch(operation, arguments)
                reply: dict[str, object] = {"result": result}
            except (LibraryError, ValueError, sqlite3.Error) as exc:
                reply = {
                    "error": {
                        "code": exc.code if isinstance(exc, LibraryError) else "storage",
                        "message": str(exc),
                    }
                }
            encoded = json.dumps(reply, allow_nan=False).encode("utf-8") + b"\n"
            if len(encoded) > MAX_MESSAGE:
                encoded = b'{"error":{"code":"message","message":"reply too large"}}\n'
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        pass
    finally:
        if records is not None:
            records.close()


if __name__ == "__main__":
    main()
