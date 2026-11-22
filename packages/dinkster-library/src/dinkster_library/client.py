"""Serialized pipe calls, with an asyncio facade that does not block its event loop."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from filelock import FileLock

from ._framing import MessageReader, message_frames

MAX_MESSAGE = 16 * 1024 * 1024
API_VERSION = 1
CAPABILITIES = (
    "transport.chunks",
    "transport.progress",
    "projects",
    "changes",
    "index.hash",
    "index.scan",
    "index.progress",
    "index.scan.selected",
    "compatibility.affected",
    "records.list",
    "records",
    "history",
    "dependencies",
    "derivations",
    "references",
    "references.collect",
    "references.retire",
    "references.engine.vaults",
    "references.engine.lifetime",
    "assets.list",
    "assets.record",
    "assets.recorded",
    "records.list.after",
    "assets.list.after",
    "assets.list.digest",
    "grants",
    "results.offers",
    "results.renditions",
    "results.terminal",
    "retention",
    "retention.departure",
    "retention.stopped-engines",
    "settings",
    "cleanup.policy",
    "folders.projects",
    "folders.grants",
    "folders.models",
    "folders.transactions",
    "permissions",
    "delegations",
    "sources.receipts",
    "sources.provenance",
    "sources.resolvers",
    "sources.resolvers.transactions",
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


class EngineLifetime:
    """An engine-held lock that survives helper failure; hold it until engine exit."""

    def __init__(self, library: Path, engine: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", engine) is None:
            raise LibraryError("invalid", "engine identity must be 32 lowercase hex digits")
        self._token = uuid.uuid4().hex
        self._lock = FileLock(library / f"engine-lifetime-{engine}-{self._token}.lock")
        self._lock.acquire(timeout=0)

    @property
    def token(self) -> str:
        return self._token

    def close(self) -> None:
        self._lock.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class LibraryClient:
    def __init__(
        self, library: Path, *, command: Sequence[str] | None = None, api: int = API_VERSION
    ) -> None:
        self._lock = threading.Lock()
        self._api = api
        self._chunked = False
        self._progress = False
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
            capabilities = self.hello.get("capabilities")
            self._chunked = isinstance(capabilities, list) and "transport.chunks" in cast(
                "list[object]", capabilities
            )
            self._progress = isinstance(capabilities, list) and "transport.progress" in cast(
                "list[object]", capabilities
            )
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    def call(self, operation: str, **arguments: object) -> dict[str, object]:
        chunked = self._chunked
        request: dict[str, object] = {
            "api": self._api,
            "operation": operation,
            "arguments": arguments,
        }
        if chunked:
            request["transport"] = "chunks"
        progress = self._progress and operation in ("references.collect", "references.retire")
        if progress:
            request["progress"] = True
        encoded = json.dumps(request, allow_nan=False).encode("utf-8") + b"\n"
        if len(encoded) > MAX_MESSAGE and not chunked:
            raise LibraryError("message", "library helper does not support large requests")
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
                if len(encoded) > MAX_MESSAGE:
                    for frame in message_frames(encoded):
                        destination.write(frame)
                else:
                    destination.write(encoded)
                destination.flush()
                reader = MessageReader(MAX_MESSAGE, chunked=chunked)
                while True:
                    reply = reader.accept(source.readline(MAX_MESSAGE + 1))
                    if reply is None:
                        continue
                    if progress and set(reply) == {"progress"} and reply["progress"] is True:
                        deadline.cancel()
                        deadline.join()
                        if expired.is_set():
                            raise LibraryError("timeout", "library helper request timed out")
                        deadline = threading.Timer(REQUEST_TIMEOUT, expire)
                        deadline.daemon = True
                        deadline.start()
                        continue
                    break
            except (OSError, ValueError) as exc:
                if expired.is_set():
                    raise LibraryError("timeout", "library helper request timed out") from exc
                self._process.kill()
                self._process.wait()
                raise LibraryError("disconnected", "library helper pipe closed") from exc
            finally:
                deadline.cancel()
                deadline.join()
            if expired.is_set():
                raise LibraryError("timeout", "library helper request timed out")
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
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
                except OSError as error:
                    # Windows can report EINVAL while flushing a disconnected pipe.
                    if error.errno != errno.EINVAL:
                        raise
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
