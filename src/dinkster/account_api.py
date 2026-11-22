"""Operator-only browser controls for the install's native account session."""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from aiohttp import web
from dinkster_server.auth import principal_for

from .account import AccountError, AccountSession, AccountSignIn

_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}


@dataclass
class _SignIn:
    owner: str
    baseline: dict[str, object]
    attempt: AccountSignIn = field(default_factory=AccountSignIn)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    url: str = field(default="", repr=False)
    error: str | None = None


def add_account_routes(app: web.Application, account: AccountSession | None) -> None:
    pending: _SignIn | None = None
    control = asyncio.Lock()

    def authority(request: web.Request) -> str:
        principal = principal_for(request)
        if not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}',
                content_type="application/json",
                headers=_HEADERS,
            )
        return principal.principal_id

    def reply(value: dict[str, object], status: int = 200) -> web.Response:
        return web.json_response(value, status=status, headers=_HEADERS)

    async def snapshot() -> dict[str, object]:
        if account is None:
            return {
                "available": False,
                "signedIn": False,
                "profile": None,
                "signIn": {"state": "idle"},
            }
        if pending is not None and pending.task is not None and not pending.task.done():
            return {"available": True, **pending.baseline, "signIn": {"state": "pending"}}
        try:
            result = await asyncio.to_thread(account.status)
        except (AccountError, OSError):
            raise web.HTTPServiceUnavailable(
                text='{"error":"account-status-unavailable"}',
                content_type="application/json",
                headers=_HEADERS,
            ) from None
        state: dict[str, object] = {"state": "idle"}
        if pending is not None and pending.error is not None:
            state = {"state": "failed", "error": pending.error}
        return {"available": True, **result, "signIn": state}

    async def stop(owner: str | None) -> None:
        nonlocal pending
        flow = pending
        if flow is None:
            return
        if owner is not None and flow.owner != owner:
            raise web.HTTPForbidden(
                text='{"error":"account-sign-in-owner-required"}',
                content_type="application/json",
                headers=_HEADERS,
            )
        await asyncio.to_thread(flow.attempt.cancel)
        if flow.task is not None:
            await asyncio.shield(flow.task)
        pending = None

    async def status(request: web.Request) -> web.Response:
        authority(request)
        return reply(await snapshot())

    async def start(request: web.Request) -> web.Response:
        nonlocal pending
        owner = authority(request)
        try:
            host = urlsplit(f"http://{request.host}").hostname or ""
            local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
            local_peer = ipaddress.ip_address(request.remote or "").is_loopback
        except ValueError:
            local_host = local_peer = False
        if not local_host or not local_peer:
            return reply({"error": "account-sign-in-requires-engine-machine"}, 409)
        try:
            if await request.json() != {}:
                return reply({"error": "invalid-account-request"}, 400)
        except ValueError:
            return reply({"error": "invalid-account-request"}, 400)
        if account is None:
            return reply({"error": "account-unavailable"}, 409)
        async with control:
            if pending is not None and pending.task is not None and not pending.task.done():
                if pending.owner != owner:
                    return reply({"error": "account-sign-in-owner-required"}, 403)
                return reply({**await snapshot(), "authorizationUrl": pending.url}, 202)
            current = await snapshot()
            flow = _SignIn(owner, {"signedIn": current["signedIn"], "profile": current["profile"]})
            pending = flow
            loop = asyncio.get_running_loop()

            def ready(url: str) -> None:
                def publish() -> None:
                    flow.url = url
                    flow.ready.set()

                loop.call_soon_threadsafe(publish)

            async def login() -> None:
                assert account is not None
                try:
                    await asyncio.to_thread(account.login, open_browser=ready, sign_in=flow.attempt)
                except (AccountError, OSError):
                    flow.error = "account-sign-in-failed"
                finally:
                    flow.ready.set()

            flow.task = asyncio.create_task(login())
            try:
                await flow.ready.wait()
            except asyncio.CancelledError:
                await stop(owner)
                raise
            if not flow.url or flow.task.done():
                return reply(await snapshot(), 409)
            return reply({**await snapshot(), "authorizationUrl": flow.url}, 202)

    async def cancel(request: web.Request) -> web.Response:
        owner = authority(request)
        async with control:
            await stop(owner)
            return reply(await snapshot())

    async def sign_out(request: web.Request) -> web.Response:
        owner = authority(request)
        try:
            if await request.json() != {}:
                return reply({"error": "invalid-account-request"}, 400)
        except ValueError:
            return reply({"error": "invalid-account-request"}, 400)
        async with control:
            await stop(owner)
            if account is not None:
                try:
                    await asyncio.to_thread(account.logout)
                except (AccountError, OSError):
                    return reply({"error": "account-sign-out-failed"}, 503)
            return reply(await snapshot())

    async def cleanup(_: web.Application) -> None:
        await stop(None)

    app.router.add_get("/api/account", status)
    app.router.add_post("/api/account/sign-in", start)
    app.router.add_delete("/api/account/sign-in", cancel)
    app.router.add_post("/api/account/sign-out", sign_out)
    app.on_cleanup.append(cleanup)
