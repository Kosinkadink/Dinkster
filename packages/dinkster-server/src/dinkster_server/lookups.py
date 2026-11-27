"""The host route that calls a node's declared lookups for the editor.

The route checks that the node type declares the function, keeps each user
within the lookup budget on this backend, and lets equal requests share one
call. Answers are not kept here: the editor keeps them until an explicit
trigger, and the host tells it which triggers happened.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass

from aiohttp import web
from dinkster_protocol.lookups import (
    LOOKUP_ANSWER_MAX_BYTES,
    LOOKUP_ERRORS,
    LOOKUP_REQUEST_MAX_BYTES,
    LOOKUP_SNAPSHOT_HEADER,
    LOOKUPS_PATH,
    LookupCallError,
    LookupRequest,
    lookup_names,
    lookup_request_from_wire,
)
from dinkster_protocol.pack_functions import value_to_wire
from dinkster_schema import NodeSchema
from dinkster_schema.body import LookupMark

from .auth import principal_for
from .call_budget import CallBudgets, call_budgets

_log = logging.getLogger("dinkster.server.lookups")

LookupDispatch = Callable[[str, str, Sequence[object]], Coroutine[object, object, object]]
"""Calls ``function`` for ``node_type`` with JSON arguments and returns the JSON answer,
or raises ``LookupCallError``."""
LookupMarks = Callable[[str], Mapping[str, LookupMark]]

LOOKUP_DISPATCH_KEY: web.AppKey[LookupDispatch] = web.AppKey("lookup_dispatch")
"""The backend's own way to call a declared lookup, for resolving at queue time.
It draws from no user's budget and shares nothing with editor calls; present
only when the app was given a dispatch."""
"""The marks of the lookups the pack that owns a node type declares."""


@dataclass
class _Call:
    task: asyncio.Task[object]
    waiters: int = 0


class LookupCalls:
    """One backend's lookup calls: the per-user budgets and sharing of equal calls.

    Equal demand (same user, catalog, pack, function and typed arguments) is
    one call. A call is cancelled when the last request waiting for it leaves,
    never while another still waits."""

    def __init__(self, dispatch: LookupDispatch, budgets: CallBudgets | None = None) -> None:
        self._dispatch = dispatch
        self._budgets = budgets or call_budgets()
        self._calls: dict[tuple[str, int, str, str], _Call] = {}

    async def call(self, principal: str, epoch: int, pack: str, request: LookupRequest) -> object:
        key = (principal, epoch, pack, request.key())
        call = self._calls.get(key)
        if call is None:
            release = self._budgets[request.asked_by].admit(principal)
            task = asyncio.create_task(
                self._dispatch(request.node_type, request.function, request.args)
            )
            call = _Call(task)
            self._calls[key] = call

            def finished(_: asyncio.Task[object], call: _Call = call) -> None:
                release()
                if self._calls.get(key) is call:
                    del self._calls[key]

            task.add_done_callback(finished)
        call.waiters += 1
        try:
            return await asyncio.shield(call.task)
        finally:
            call.waiters -= 1
            if call.waiters == 0 and not call.task.done():
                call.task.cancel()
                if self._calls.get(key) is call:
                    del self._calls[key]


def lookups_changed(
    node_types: Iterable[str],
    marks: LookupMarks,
    *,
    asset_kinds: Iterable[str] = (),
    after_run: bool = False,
) -> list[str]:
    """The functions whose answers a host event makes old: those that watch one
    of ``asset_kinds``, or, for a successful run, those marked ``after_run``."""
    kinds = set(asset_kinds)
    changed: set[str] = set()
    for node_type in node_types:
        for name, mark in marks(node_type).items():
            if (after_run and mark.after_run) or kinds.intersection(mark.watches):
                changed.add(name)
    return sorted(changed)


AssetKindsChanged = Callable[[Iterable[str] | None], None]
ASSET_KINDS_CHANGED_KEY: web.AppKey[AssetKindsChanged] = web.AppKey("lookup_asset_kinds_changed")


def announce_asset_kinds(app: web.Application, kinds: Iterable[str] | None) -> None:
    """Tell editors that the host's catalog committed a change to assets of
    ``kinds`` (None when the kinds are not known), so the lookups that watch
    them are asked again. Call it after the commit, on the event loop."""
    announce = app.get(ASSET_KINDS_CHANGED_KEY)
    if announce is not None:
        announce(kinds)


def recorded_asset_kinds(kind: str, media_type: str) -> set[str]:
    """The kinds an asset recorded as ``kind`` is listed under: its recorded
    kind and, for audio, images and video, its media kind."""
    major = media_type.partition("/")[0]
    return {kind, f"media/{major}"} if major in {"audio", "image", "video"} else {kind}


def install_lookups(
    app: web.Application,
    *,
    schemas: Callable[[], Mapping[str, NodeSchema]],
    node_pack: Callable[[str], str],
    epoch: Callable[[], int],
    dispatch: LookupDispatch | None,
    budgets: CallBudgets | None = None,
    marks: LookupMarks | None = None,
    publish: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[tuple[str, str], str]:
    """``POST /api/lookups``: one declared lookup's answer for the editor."""
    calls = None if dispatch is None else LookupCalls(dispatch, budgets)
    if dispatch is not None:
        app[LOOKUP_DISPATCH_KEY] = dispatch

    def failure(code: str) -> web.Response:
        return web.json_response(
            {"error": code},
            status=LOOKUP_ERRORS[code],
            headers={LOOKUP_SNAPSHOT_HEADER: str(epoch())},
        )

    async def handle(request: web.Request) -> web.Response:
        current = epoch()
        expected = request.headers.get("If-Match")
        if expected is not None and expected != str(current):
            raise web.HTTPPreconditionFailed(headers={LOOKUP_SNAPSHOT_HEADER: str(current)})
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        data = bytearray()
        async for chunk in request.content.iter_chunked(LOOKUP_REQUEST_MAX_BYTES + 1):
            data.extend(chunk)
            if len(data) > LOOKUP_REQUEST_MAX_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=LOOKUP_REQUEST_MAX_BYTES, actual_size=len(data)
                )
        try:
            parsed = lookup_request_from_wire(json.loads(data))
        except (ValueError, UnicodeError):
            return failure("lookup-invalid-arguments")
        schema = schemas().get(parsed.node_type)
        if calls is None or schema is None or parsed.function not in lookup_names(schema):
            return failure("lookup-unknown")
        try:
            answer = await calls.call(
                principal_for(request).principal_id,
                current,
                node_pack(parsed.node_type),
                parsed,
            )
            typed_answer = value_to_wire(answer)
            if (
                len(json.dumps(typed_answer, ensure_ascii=False).encode("utf-8"))
                > LOOKUP_ANSWER_MAX_BYTES
            ):
                return failure("lookup-wrong-shape")
        except LookupCallError as exc:
            if exc.detail:
                _log.info("lookup %s: %s", parsed.function, exc.detail)
            return failure(exc.code)
        except Exception:
            _log.warning("lookup %s failed", parsed.function, exc_info=True)
            return failure("lookup-failed")
        return web.json_response(
            {"answer": answer, "typedAnswer": typed_answer},
            headers={LOOKUP_SNAPSHOT_HEADER: str(current)},
        )

    if marks is not None and publish is not None:

        def asset_kinds_changed(kinds: Iterable[str] | None) -> None:
            node_types = list(schemas())
            if kinds is None:
                changed = sorted(
                    {name for t in node_types for name, mark in marks(t).items() if mark.watches}
                )
            else:
                changed = lookups_changed(node_types, marks, asset_kinds=kinds)
            if changed:
                publish({"type": "lookups_changed", "functions": changed})

        app[ASSET_KINDS_CHANGED_KEY] = asset_kinds_changed
    app.router.add_post(LOOKUPS_PATH, handle)
    return {("POST", LOOKUPS_PATH): "jobs:read"}
