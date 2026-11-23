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
from collections.abc import Awaitable, Callable
from typing import Final

from aiohttp import web
from dinkster_protocol.lookups import LookupCallError
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
from .call_budget import CallBudget

PackFunctionDispatch = Callable[
    [PackFunctionRequest, Callable[[float, str], None]], Awaitable[PackFunctionAnswer]
]

_HOST_MARGIN: Final = 5.0
_DISCONNECT_POLL: Final = 0.25


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
        # read(n) answers what is buffered, so read until the end or past the limit.
        data = bytearray()
        while len(data) <= PACK_FUNCTION_REQUEST_MAX_BYTES and (
            chunk := await request.content.read(PACK_FUNCTION_REQUEST_MAX_BYTES + 1 - len(data))
        ):
            data += chunk
        if len(data) > PACK_FUNCTION_REQUEST_MAX_BYTES:
            raise web.HTTPRequestEntityTooLarge(
                max_size=PACK_FUNCTION_REQUEST_MAX_BYTES, actual_size=len(data)
            )
        try:
            call_request = PackFunctionRequest.from_wire(json.loads(data))
        except (ValueError, UnicodeError, OverflowError, RecursionError):
            return _refusal(400, "invalid", FAILED_MESSAGE)
        try:
            release = budget.admit(principal.principal_id)
        except LookupCallError:
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
