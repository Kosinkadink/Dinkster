"""Serialized pipe calls, with an asyncio facade that does not block its event loop."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self, cast

MAX_MESSAGE = 16 * 1024 * 1024
API_VERSION = 1
CAPABILITIES = (
    "projects",
    "changes",
    "index.hash",
    "index.scan",
    "index.progress",
    "index.scan.selected",
    "compatibility.affected",
    "records.list",
    "assets.list",
    "projects.needs.workflows",
    "projects.environment-selection",
)
COMMAND_ENV = "DINKSTER_LIBRARY_COMMAND"
REQUEST_TIMEOUT = 60.0


class LibraryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def helper_command() -> list[str]:
    configured = os.environ.get(COMMAND_ENV)
    if configured is None:
        return [sys.executable, "-m", "dinkster_library"]
    value: object = json.loads(configured)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(part, str) or not part for part in cast("list[object]", value))
    ):
        raise LibraryError("command", f"{COMMAND_ENV} must be a JSON command array")
    return cast("list[str]", value)


class LibraryClient:
    def __init__(
        self, library: Path, *, command: Sequence[str] | None = None, api: int = API_VERSION
    ) -> None:
        self._lock = threading.Lock()
        self._api = api
        self._process = subprocess.Popen(
            [*(command if command is not None else helper_command()), "--library", str(library)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Diagnostics never share the wire stream, and cannot fill an unread pipe.
            stderr=None,
            close_fds=True,
        )
        try:
            self.hello = self.call("hello")
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    def call(self, operation: str, **arguments: object) -> dict[str, object]:
        encoded = (
            json.dumps(
                {"api": self._api, "operation": operation, "arguments": arguments},
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > MAX_MESSAGE:
            raise LibraryError("message", "library request is too large")
        with self._lock:
            source, destination = self._process.stdout, self._process.stdin
            assert source is not None and destination is not None
            expired = threading.Event()

            def expire() -> None:
                expired.set()
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass

            deadline = threading.Timer(REQUEST_TIMEOUT, expire)
            deadline.daemon = True
            deadline.start()
            try:
                destination.write(encoded)
                destination.flush()
                line = source.readline(MAX_MESSAGE + 1)
            except (OSError, ValueError) as exc:
                if expired.is_set():
                    raise LibraryError("timeout", "library helper request timed out") from exc
                raise LibraryError("disconnected", "library helper pipe closed") from exc
            finally:
                deadline.cancel()
                deadline.join()
            if expired.is_set():
                raise LibraryError("timeout", "library helper request timed out")
            if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise LibraryError("disconnected", "library helper closed or sent an invalid reply")
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise LibraryError("message", "library helper reply must be an object")
            reply = cast("dict[str, object]", value)
            if isinstance(reply.get("error"), dict):
                error = cast("dict[str, object]", reply["error"])
                raise LibraryError(str(error["code"]), str(error["message"]))
            result = reply.get("result")
            if not isinstance(result, dict):
                raise LibraryError("message", "library helper result must be an object")
            return cast("dict[str, object]", result)

    def close(self) -> None:
        with self._lock:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            if self._process.stdout is not None:
                self._process.stdout.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncLibraryClient:
    def __init__(self, client: LibraryClient) -> None:
        self.sync = client

    @classmethod
    async def start(
        cls, library: Path, *, command: Sequence[str] | None = None, api: int = API_VERSION
    ) -> Self:
        startup = asyncio.create_task(
            asyncio.to_thread(LibraryClient, library, command=command, api=api)
        )
        try:
            return cls(await asyncio.shield(startup))
        except asyncio.CancelledError:
            try:
                client = await startup
            except Exception:
                pass
            else:
                await asyncio.to_thread(client.close)
            raise

    async def call(self, operation: str, **arguments: object) -> dict[str, object]:
        return await asyncio.to_thread(self.sync.call, operation, **arguments)

    async def close(self) -> None:
        await asyncio.to_thread(self.sync.close)
