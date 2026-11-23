"""The HTTP route a button's ``Call`` step uses: one press, one streamed answer.

The response is newline-delimited JSON. Progress lines go to the presser only
while the function runs; the last line is the answer, a refusal, or, for a
file, its size followed by exactly that many bytes. Closing the request
cancels the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Final

from aiohttp import web
from dinkster_protocol.pack_functions import (
    FAILED_MESSAGE,
    PACK_FUNCTION_PATH,
    PACK_FUNCTION_REQUEST_MAX_BYTES,
    PAUSED_MESSAGE,
    TIMEOUT_MESSAGE,
    PackFunctionAnswer,
    PackFunctionRefused,
    PackFunctionRequest,
)
from dinkster_schema.body import MAX_PACK_FUNCTION_TIME_LIMIT

from .auth import principal_for

PackFunctionDispatch = Callable[
    [PackFunctionRequest, Callable[[float, str], None]], Awaitable[PackFunctionAnswer]
]

CALLS_RUNNING_PER_USER: Final = 8
CALLS_STARTED_PER_MINUTE_PER_USER: Final = 60
_HOST_MARGIN: Final = 5.0
_DISCONNECT_POLL: Final = 0.25


class CallBudget:
    """How many calls each user may have under way, and start in a minute, on this backend."""

    def __init__(
        self,
        running: int = CALLS_RUNNING_PER_USER,
        per_minute: int = CALLS_STARTED_PER_MINUTE_PER_USER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._running_limit = running
        self._per_minute = per_minute
        self._clock = clock
        self._running: dict[str, int] = {}
        self._started: dict[str, deque[float]] = {}

    def admit(self, principal: str) -> Callable[[], None] | None:
        """Take one call for ``principal``; ``None`` when over budget. Call the result once done."""
        now = self._clock()
        started = self._started.setdefault(principal, deque())
        while started and now - started[0] >= 60.0:
            started.popleft()
        if (
            self._running.get(principal, 0) >= self._running_limit
            or len(started) >= self._per_minute
        ):
            return None
        started.append(now)
        self._running[principal] = self._running.get(principal, 0) + 1
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            left = self._running[principal] - 1
            if left:
                self._running[principal] = left
            else:
                del self._running[principal]

        return release


def _line(data: dict[str, object]) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode() + b"\n"


def _refusal(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": code, "message": message}, status=status)


def install_pack_functions(
    app: web.Application, dispatch: PackFunctionDispatch | None, budget: CallBudget
) -> dict[tuple[str, str], str]:
    """Add the call route; the result maps it to the capability a request needs."""

    async def call(request: web.Request) -> web.StreamResponse:
        if dispatch is None:
            raise web.HTTPNotFound()
        principal = principal_for(request)
        if principal.kind != "human":
            return _refusal(403, "headless", "Buttons are pressed by a person in the editor.")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        data = await request.content.read(PACK_FUNCTION_REQUEST_MAX_BYTES + 1)
        if len(data) > PACK_FUNCTION_REQUEST_MAX_BYTES:
            raise web.HTTPRequestEntityTooLarge(
                max_size=PACK_FUNCTION_REQUEST_MAX_BYTES, actual_size=len(data)
            )
        try:
            call_request = PackFunctionRequest.from_wire(json.loads(data))
        except (ValueError, UnicodeError):
            return _refusal(400, "invalid", FAILED_MESSAGE)
        release = budget.admit(principal.principal_id)
        if release is None:
            return _refusal(429, "paused", PAUSED_MESSAGE)
        try:
            return await _stream(request, call_request, dispatch)
        finally:
            release()

    app.router.add_post(PACK_FUNCTION_PATH, call)
    return {("POST", PACK_FUNCTION_PATH): "jobs:submit"}


async def _stream(
    request: web.Request, call_request: PackFunctionRequest, dispatch: PackFunctionDispatch
) -> web.StreamResponse:
    lines: asyncio.Queue[bytes] = asyncio.Queue()

    def on_progress(fraction: float, text: str) -> None:
        lines.put_nowait(_line({"progress": fraction, "text": text}))

    response = web.StreamResponse(
        headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-store"}
    )
    await response.prepare(request)
    running = asyncio.ensure_future(dispatch(call_request, on_progress))
    deadline = time.monotonic() + MAX_PACK_FUNCTION_TIME_LIMIT + _HOST_MARGIN
    try:
        while not running.done():
            transport = request.transport
            if transport is None or transport.is_closing():
                return response
            if time.monotonic() > deadline:
                await response.write(_line({"error": "timeout", "message": TIMEOUT_MESSAGE}))
                return response
            next_line = asyncio.ensure_future(lines.get())
            await asyncio.wait(
                {running, next_line}, timeout=_DISCONNECT_POLL, return_when=asyncio.FIRST_COMPLETED
            )
            if next_line.done():
                await response.write(next_line.result())
            else:
                next_line.cancel()
        while not lines.empty():
            await response.write(lines.get_nowait())
        try:
            answer = running.result()
        except PackFunctionRefused as exc:
            await response.write(_line({"error": exc.code, "message": exc.message}))
        except Exception:
            await response.write(_line({"error": "failed", "message": FAILED_MESSAGE}))
        else:
            notes = list(answer.notes)
            if answer.file is not None:
                await response.write(_line({"file": len(answer.file), "notes": notes}))
                await response.write(answer.file)
            else:
                await response.write(_line({"answer": dict(answer.fields or {}), "notes": notes}))
        await response.write_eof()
        return response
    finally:
        if not running.done():
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await running
