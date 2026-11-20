"""The helper's stdout belongs exclusively to the pipe protocol."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sqlite3
import sys
import threading
from pathlib import Path

from ._framing import MessageReader, message_frames
from ._records import FORMAT, Records, object_value
from .client import API_VERSION, CAPABILITIES, MAX_MESSAGE, LibraryError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps({"apiMin": API_VERSION, "apiMax": API_VERSION, "capabilities": CAPABILITIES})
        )
        return
    if args.library is None:
        parser.error("--library is required unless --describe is used")
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(0, os.O_BINARY)
    requests: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def receive() -> None:
        pending = bytearray()
        discarding_line = False
        while True:
            chunk = os.read(0, 64 * 1024)
            if not chunk:
                # EOF can arrive while the main thread waits for a SQLite lock.
                # Exiting releases OS locks and rolls back any unfinished write.
                os._exit(0)
            if discarding_line:
                end = chunk.find(b"\n")
                if end < 0:
                    continue
                chunk = chunk[end + 1 :]
                discarding_line = False
            pending.extend(chunk)
            while (end := pending.find(b"\n")) >= 0:
                requests.put(bytes(pending[: end + 1]))
                del pending[: end + 1]
            if len(pending) > MAX_MESSAGE:
                requests.put(bytes(pending))
                pending.clear()
                discarding_line = True

    threading.Thread(target=receive, daemon=True).start()
    records: Records | None = None
    from ._scan import Scans

    scans = Scans()
    reader = MessageReader(MAX_MESSAGE, chunked=True)
    try:
        while True:
            line = requests.get()
            chunked = False
            try:
                try:
                    request = reader.accept(line)
                except ValueError as error:
                    raise LibraryError("message", str(error)) from error
                if request is None:
                    continue
                chunked = request.get("transport") == "chunks"
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
                        "capabilities": CAPABILITIES,
                        "format": FORMAT,
                        "attached": records.attachments(args.library),
                    }
                elif operation == "index.hash":
                    from ._index import hash_file

                    result = hash_file(arguments.get("path"))
                elif operation == "index.scan":
                    result = scans.start(arguments)
                elif operation == "index.progress":
                    result = scans.progress(arguments.get("id"))
                else:
                    result = records.dispatch(operation, arguments)
                reply: dict[str, object] = {"result": result}
            except (LibraryError, ValueError, OSError, sqlite3.Error) as exc:
                reply = {
                    "error": {
                        "code": exc.code if isinstance(exc, LibraryError) else "storage",
                        "message": str(exc),
                    }
                }
            encoded = json.dumps(reply, allow_nan=False).encode("utf-8") + b"\n"
            if len(encoded) > MAX_MESSAGE and chunked:
                for frame in message_frames(encoded):
                    sys.stdout.buffer.write(frame)
            else:
                if len(encoded) > MAX_MESSAGE:
                    encoded = (
                        b'{"error":{"code":"message",'
                        b'"message":"reply too large; use transport.chunks"}}\n'
                    )
                sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        pass
    finally:
        if records is not None:
            records.close()


if __name__ == "__main__":
    main()
