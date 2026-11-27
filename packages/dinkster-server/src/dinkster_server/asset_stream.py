"""Cancellation-safe streaming from an authority-checked asset descriptor.

Digest-addressed endpoints verify content identity before streaming; local
bindings instead validate the live grant and file revision. Both stream from
that same descriptor: checking then reopening would recreate the replacement
window. Neither may leak the descriptor when aiohttp cancels the handler while
a worker-thread open or read is in flight. ``asyncio.to_thread`` does not
interrupt its worker on cancellation, so the pattern here is
shield-plus-handoff: the awaiter may be cancelled, but the descriptor's
closer becomes a completion callback on the still-running worker task -
never a concurrent close racing a read.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from aiohttp import web
from dinkster_assets import open_verified
from dinkster_values import MEBIBYTE

CHUNK = MEBIBYTE  # streamed read size: bounded memory for any asset size


def _open_sized(opener: Callable[[], BinaryIO]) -> tuple[BinaryIO, int]:
    handle = opener()
    try:
        return handle, os.fstat(handle.fileno()).st_size
    except BaseException:
        handle.close()
        raise


def _close_open_result(task: asyncio.Task[tuple[BinaryIO, int]]) -> None:
    if task.cancelled() or task.exception() is not None:
        return  # nothing was handed out
    handle, _ = task.result()
    handle.close()


async def open_verified_sized(path: Path, digest: str) -> tuple[BinaryIO, int]:
    """``open_verified`` plus size, off the event loop, cancellation-safe:
    if the request dies while a multi-GB hash runs, the eventual descriptor
    is closed by a completion callback instead of leaking. Raises whatever
    ``open_verified`` raises (AssetIntegrityError, AssetError, OSError)."""
    return await open_sized(lambda: open_verified(path, digest))


async def open_sized(opener: Callable[[], BinaryIO]) -> tuple[BinaryIO, int]:
    """Open through the caller's asset authority without leaking on cancellation."""
    task = asyncio.ensure_future(asyncio.to_thread(_open_sized, opener))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_close_open_result)
        raise


async def stream_verified(
    request: web.Request,
    handle: BinaryIO,
    size: int,
    *,
    headers: dict[str, str] | None = None,
) -> web.StreamResponse:
    """Stream a verified handle as the response body. Owns closing the
    handle on every path: success, write failure after prepare (client
    disconnect), and cancellation mid-read (closed after the worker read
    completes, never concurrently with it)."""
    handed_off = False
    try:
        response_headers = {**(headers or {}), "Accept-Ranges": "bytes"}
        start, stop = 0, size
        status = 200
        range_header = request.headers.get("Range", "")
        validator = request.headers.get("If-Range")
        # These responses have no Last-Modified date. Only an exact strong
        # ETag can establish that an If-Range validator still names these bytes.
        if (
            request.method == "GET"
            and range_header.startswith("bytes=")
            and "," not in range_header
            and (
                validator is None
                or (not validator.startswith("W/") and validator == response_headers.get("ETag"))
            )
        ):
            try:
                requested = request.http_range
                start = requested.start if requested.start is not None else 0
                if range_header.startswith("bytes=-") and start == 0:
                    raise ValueError("empty suffix range")
                if start < 0:
                    start = max(0, size + start)
                stop = min(size, requested.stop if requested.stop is not None else size)
                if start >= stop:
                    raise ValueError("unsatisfiable range")
            except ValueError:
                return web.Response(
                    status=416,
                    headers={**response_headers, "Content-Range": f"bytes */{size}"},
                )
            status = 206
            response_headers["Content-Range"] = f"bytes {start}-{stop - 1}/{size}"
        response = web.StreamResponse(status=status, headers=response_headers)
        response.content_type = "application/octet-stream"
        response.content_length = stop - start
        handle.seek(start)
        await response.prepare(request)
        remaining = 0 if request.method == "HEAD" else stop - start
        while remaining:
            read_task = asyncio.ensure_future(asyncio.to_thread(handle.read, min(CHUNK, remaining)))
            try:
                chunk = await asyncio.shield(read_task)
            except asyncio.CancelledError:
                handed_off = True
                read_task.add_done_callback(lambda _task: handle.close())
                raise
            if not chunk:
                break
            await response.write(chunk)
            remaining -= len(chunk)
        await response.write_eof()
        return response
    finally:
        if not handed_off:
            handle.close()
