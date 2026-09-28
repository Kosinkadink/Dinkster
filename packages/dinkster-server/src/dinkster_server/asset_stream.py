"""Verified asset streaming shared by the digest-addressed byte endpoints.

Two routes hand asset bytes to the network: the peer surface
(``/assets/{digest}``, app.py) and the client surface
(``/api/assets/{digest}``, library.py). Both must honor the same
invariant - the digest names the bytes actually served - and the same
mechanics: verify from an open descriptor (dinkster_assets.integrity),
stream FROM THAT DESCRIPTOR (verify-then-reopen would recreate the
replacement window a peer or browser would silently mirror), and never
leak the descriptor, including when aiohttp cancels the handler while a
worker-thread open or read is in flight. ``asyncio.to_thread`` does not
interrupt its worker on cancellation, so the pattern here is
shield-plus-handoff: the awaiter may be cancelled, but the descriptor's
closer becomes a completion callback on the still-running worker task -
never a concurrent close racing a read.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import BinaryIO

from aiohttp import web
from dinkster_assets import open_verified
from dinkster_values import MEBIBYTE

CHUNK = MEBIBYTE  # streamed read size: bounded memory for any asset size


def _open_sized(path: Path, digest: str) -> tuple[BinaryIO, int]:
    handle = open_verified(path, digest)
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
    task = asyncio.ensure_future(asyncio.to_thread(_open_sized, path, digest))
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
        response = web.StreamResponse(headers=headers or {})
        response.content_type = "application/octet-stream"
        response.content_length = size
        await response.prepare(request)
        while True:
            read_task = asyncio.ensure_future(asyncio.to_thread(handle.read, CHUNK))
            try:
                chunk = await asyncio.shield(read_task)
            except asyncio.CancelledError:
                handed_off = True
                read_task.add_done_callback(lambda _task: handle.close())
                raise
            if not chunk:
                break
            await response.write(chunk)
        await response.write_eof()
        return response
    finally:
        if not handed_off:
            handle.close()
