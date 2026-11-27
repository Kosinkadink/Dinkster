"""Operator-only browser controls for the install's native account session."""

from __future__ import annotations

import asyncio
import ipaddress
import secrets
from dataclasses import dataclass, field, replace
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID

from aiohttp import web
from dinkster_library import LibraryError
from dinkster_server.auth import BROWSER_AUTH_KEY, Principal, principal_for
from dinkster_server.projects import PROJECTS_KEY

from .account import AccountError, AccountSession, AccountSignIn

_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
_COOKIE = "dinkster_account"


def _same_origin(request: web.Request) -> bool:
    origin = request.headers.get("Origin")
    return request.headers.get("Sec-Fetch-Site") != "cross-site" and (
        origin is None or origin == f"{request.scheme}://{request.host}"
    )


@dataclass
class _BrowserAccount:
    account: AccountSession
    token: str = field(default="", repr=False)
    revision: str = field(default_factory=lambda: secrets.token_hex(16))
    native_session: str | None = field(default=None, repr=False)
    principal: Principal | None = None
    initiator: Principal | None = None

    def owns(self, request: web.Request) -> bool:
        token = request.cookies.get(_COOKIE, "")
        return bool(self.token) and secrets.compare_digest(token.encode(), self.token.encode())

    def begin(self, principal: Principal) -> None:
        if principal.browser_session_id == self.revision and self.initiator is not None:
            principal = self.initiator
        self.token = secrets.token_urlsafe(32)
        self.revision = secrets.token_hex(16)
        self.native_session = None
        self.principal = None
        self.initiator = principal

    def end(self) -> None:
        self.token = ""
        self.revision = secrets.token_hex(16)
        self.native_session = None
        self.principal = None
        self.initiator = None

    def current(self, principal: Principal) -> bool:
        if principal.browser_session_id != self.revision or self.native_session is None:
            return False
        if not self.account.matches_session(self.native_session):
            self.end()
            return False
        return True

    def authenticate(self, request: web.Request) -> Principal | None:
        if _COOKIE not in request.cookies:
            return None
        if not _same_origin(request):
            raise web.HTTPForbidden(
                text='{"error":"same-origin-required"}', content_type="application/json"
            )
        if self.owns(request) and self.principal is not None and self.current(self.principal):
            return self.principal
        if self.owns(request) and self.principal is None:
            return None
        # These account controls clear a stale binding; they never grant account authority.
        if request.path in {"/api/account", "/api/account/sign-in", "/api/account/sign-out"}:
            return None
        raise web.HTTPUnauthorized(
            text='{"error":"account-session-ended"}', content_type="application/json"
        )

    def cookie(self, response: web.Response, request: web.Request) -> None:
        response.set_cookie(
            _COOKIE, self.token, httponly=True, secure=request.secure, samesite="Strict", path="/"
        )


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
    browser = _BrowserAccount(account) if account is not None else None
    if browser is not None:
        app[BROWSER_AUTH_KEY] = browser

    def authority(request: web.Request) -> str:
        if request.method not in {"GET", "HEAD"} and not _same_origin(request):
            raise web.HTTPForbidden(
                text='{"error":"same-origin-required"}', content_type="application/json"
            )
        principal = principal_for(request)
        if not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}',
                content_type="application/json",
                headers=_HEADERS,
            )
        if browser is not None:
            if principal == browser.initiator and (not principal.local or browser.owns(request)):
                return browser.revision
            if browser.owns(request) and principal.browser_session_id == browser.revision:
                return browser.revision
        return principal.principal_id

    def reply(value: dict[str, object], status: int = 200) -> web.Response:
        return web.json_response(value, status=status, headers=_HEADERS)

    def local_browser(request: web.Request) -> bool:
        try:
            host = urlsplit(f"http://{request.host}").hostname or ""
            local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
            return local_host and ipaddress.ip_address(request.remote or "").is_loopback
        except ValueError:
            return False

    async def snapshot(request: web.Request) -> dict[str, object]:
        reason = None
        if account is None:
            reason = "account-unavailable"
        elif not local_browser(request):
            reason = "account-sign-in-requires-engine-machine"
        principal = principal_for(request)
        availability = {
            "canStartSignIn": reason is None,
            "signInUnavailableReason": reason,
            "principalRevision": (principal.browser_session_id or "local")
            if principal.authorization_error is None
            else "local",
        }
        if account is None:
            return {
                "available": False,
                **availability,
                "signedIn": False,
                "profile": None,
                "signIn": {"state": "idle"},
            }
        if pending is not None and pending.task is not None and not pending.task.done():
            return {
                "available": True,
                **availability,
                **pending.baseline,
                "signIn": {"state": "pending"},
            }
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
        return {"available": True, **availability, **result, "signIn": state}

    async def stop(owner: str | None) -> None:
        nonlocal pending
        flow = pending
        if flow is None:
            return
        if (
            owner is not None
            and flow.owner != owner
            and flow.task is not None
            and not flow.task.done()
        ):
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
        response = reply(await snapshot(request))
        if browser is not None and _COOKIE in request.cookies and not browser.owns(request):
            response.del_cookie(_COOKIE, path="/")
        return response

    async def start(request: web.Request) -> web.Response:
        nonlocal pending
        owner = authority(request)
        if not local_browser(request):
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
                return reply({**await snapshot(request), "authorizationUrl": pending.url}, 202)
            current = await snapshot(request)
            assert browser is not None
            browser.begin(principal_for(request))
            owner = browser.revision
            flow = _SignIn(owner, {"signedIn": current["signedIn"], "profile": current["profile"]})
            pending = flow
            loop = asyncio.get_running_loop()

            def ready(url: str) -> None:
                def publish() -> None:
                    flow.url = url
                    flow.ready.set()

                loop.call_soon_threadsafe(publish)

            async def login() -> None:
                assert account is not None and browser is not None
                try:
                    await asyncio.to_thread(account.login, open_browser=ready, sign_in=flow.attempt)
                    identity = await asyncio.to_thread(account.authenticated_identity)
                    if identity is not None and browser.revision == flow.owner:
                        session_id, subject = identity
                        browser.native_session = session_id
                        browser.principal = replace(
                            cast(Principal, browser.initiator),
                            principal_id=subject,
                            browser_session_id=browser.revision,
                        )
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
                response = reply(await snapshot(request), 409)
            else:
                response = reply({**await snapshot(request), "authorizationUrl": flow.url}, 202)
            browser.cookie(response, request)
            return response

    async def cancel(request: web.Request) -> web.Response:
        owner = authority(request)
        async with control:
            was_pending = (
                pending is not None and pending.task is not None and not pending.task.done()
            )
            await stop(owner)
            if was_pending and browser is not None:
                browser.end()
            response = reply(await snapshot(request))
            if was_pending:
                response.del_cookie(_COOKIE, path="/")
            return response

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
            if browser is not None:
                browser.end()
            response = reply(await snapshot(request))
            response.del_cookie(_COOKIE, path="/")
            return response

    async def project_choices(request: web.Request) -> web.Response:
        authority(request)
        if account is None:
            return reply({"error": "account-unavailable"}, 409)
        async with control:
            if pending is not None and pending.task is not None and not pending.task.done():
                return reply({"error": "account-sign-in-pending"}, 409)
            try:
                rows = await asyncio.to_thread(account.projects)
                projects: list[dict[str, object]] = []
                for row in rows:
                    identifier = str(UUID(str(row.get("id", ""))))
                    name = row.get("display_name")
                    projects.append(
                        {
                            "id": identifier,
                            "displayName": name.strip() or None
                            if isinstance(name, str) and len(name) <= 2048
                            else None,
                        }
                    )
                return reply({"issuer": account.services.identity_base, "projects": projects})
            except (ValueError, OSError):
                return reply({"error": "account-projects-unavailable"}, 503)

    async def link_project(request: web.Request) -> web.Response:
        authority(request)
        try:
            body = await request.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"revision", "accountProject"}
                or type(body["revision"]) is not int
                or (
                    body["accountProject"] is not None
                    and not isinstance(body["accountProject"], str)
                )
            ):
                return reply({"error": "invalid-account-request"}, 400)
            identifier = (
                str(UUID(body["accountProject"])) if body["accountProject"] is not None else None
            )
        except ValueError:
            return reply({"error": "invalid-account-request"}, 400)
        client = request.app.get(PROJECTS_KEY)
        if client is None:
            return reply({"error": "project-library-unavailable"}, 503)
        async with control:
            if pending is not None and pending.task is not None and not pending.task.done():
                return reply({"error": "account-sign-in-pending"}, 409)
            try:
                result = await client.call("projects.get", id=request.match_info["project"])
                project = cast("dict[str, object] | None", result["project"])
                if project is None or project["deleted"]:
                    return reply({"error": "project-unavailable"}, 404)
                if project["revision"] != body["revision"]:
                    return reply({"error": "project-changed"}, 409)
                link = None
                if identifier is not None:
                    if account is None:
                        return reply({"error": "account-unavailable"}, 409)
                    link = await asyncio.to_thread(account.project_link, identifier)
                    if link is None:
                        return reply({"error": "account-project-unavailable"}, 400)
                result = await client.call(
                    "projects.patch",
                    id=request.match_info["project"],
                    revision=body["revision"],
                    patch={"accountProject": link},
                )
                return reply(result)
            except LibraryError as error:
                status = {"stale": 409, "missing": 404, "invalid": 400}.get(error.code, 503)
                return reply({"error": "account-project-save-failed"}, status)
            except (ValueError, OSError):
                return reply({"error": "account-projects-unavailable"}, 503)

    async def cleanup(_: web.Application) -> None:
        await stop(None)

    app.router.add_get("/api/account", status)
    app.router.add_get("/api/account/projects", project_choices)
    app.router.add_put("/api/account/projects/{project}", link_project)
    app.router.add_post("/api/account/sign-in", start)
    app.router.add_delete("/api/account/sign-in", cancel)
    app.router.add_post("/api/account/sign-out", sign_out)
    app.on_cleanup.append(cleanup)
