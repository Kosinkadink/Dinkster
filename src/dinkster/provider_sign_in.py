"""Owner-bound Hugging Face PKCE and device-code sign-in."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

from aiohttp import web
from dinkster_assets import AssetError
from dinkster_assets.fetch import post_huggingface_form
from dinkster_server.auth import principal_for, resolve_scope

from .provider_credentials import (
    HUGGINGFACE_CLIENT_ID,
    ProviderCredentials,
    ProviderCredentialsError,
    oauth_credential,
)

_SCOPE = "openid profile gated-repos"
_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}


@dataclass
class _SignIn:
    owner: str
    scope: str
    revision: str
    expires: float
    method: str
    verifier: str = field(default="", repr=False)
    state: str = field(default="", repr=False)
    code: str = field(default="", repr=False)
    device_code: str = field(default="", repr=False)
    redirect: str = ""
    interval: int = 5
    next_poll: float = 0
    status: str = "pending"
    runner: web.AppRunner | None = field(default=None, repr=False)
    timeout: asyncio.TimerHandle | None = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def add_provider_sign_in_routes(
    app: web.Application, credentials: ProviderCredentials, changed: Callable[[], None]
) -> None:
    pending: dict[str, _SignIn] = {}
    active: set[asyncio.Task[None]] = set()
    creating = asyncio.Lock()

    async def exchange(endpoint: str, fields: dict[str, str]) -> dict[str, object]:
        task = asyncio.create_task(asyncio.to_thread(post_huggingface_form, endpoint, fields))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def authority(request: web.Request) -> tuple[str, str]:
        principal = principal_for(request)
        if not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}', content_type="application/json"
            )
        return principal.principal_id, resolve_scope(
            principal, "settings:write", request.query.get("scope")
        )

    def reply(value: dict[str, object], status: int = 200) -> web.Response:
        return web.json_response(value, status=status, headers=_HEADERS)

    async def discard(flow_id: str) -> None:
        flow = pending.pop(flow_id, None)
        if flow is not None:
            if flow.timeout is not None:
                flow.timeout.cancel()
            if flow.runner is not None:
                await flow.runner.cleanup()
            flow.code = flow.verifier = flow.device_code = flow.state = ""

    async def stop(flow_id: str) -> None:
        flow = pending.get(flow_id)
        if flow is not None:
            if flow.status == "pending":
                flow.status = "cancelled"
            async with flow.lock:
                await discard(flow_id)

    async def complete(flow: _SignIn, response: dict[str, object]) -> None:
        if (
            flow not in pending.values()
            or flow.expires <= time.monotonic()
            or flow.status != "pending"
        ):
            raise ProviderCredentialsError("Sign-in expired")
        token, expires, refresh = oauth_credential(response)
        flow.status = "saving"
        await asyncio.to_thread(
            credentials.set_token,
            token,
            expires_at=expires,
            refresh_token=refresh,
            expected_revision=flow.revision,
        )
        changed()

    async def start(request: web.Request) -> web.Response:
        authority(request)
        if creating.locked():
            return reply({"error": "provider-sign-in-busy"}, 429)
        async with creating:
            task = asyncio.create_task(begin(request))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                result = await task
                if result.status == 201:
                    for flow_id in tuple(pending):
                        if (pending[flow_id].owner, pending[flow_id].scope) == authority(request):
                            await stop(flow_id)
                raise

    async def begin(request: web.Request) -> web.Response:
        owner, scope = authority(request)
        flow_id = ""
        try:
            body: object = await request.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"method"}
                or not isinstance(body["method"], str)
                or body["method"] not in {"pkce", "device"}
            ):
                return reply({"error": "invalid-provider-sign-in"}, 400)
            method = str(body["method"])
            if method == "pkce":
                try:
                    host = urlsplit(f"http://{request.host}").hostname or ""
                    local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
                    local_peer = ipaddress.ip_address(request.remote or "").is_loopback
                except ValueError:
                    local_host = local_peer = False
                if not local_host or not local_peer:
                    return reply({"error": "device-sign-in-required"}, 409)
            for flow_id, flow in tuple(pending.items()):
                if flow.expires <= time.monotonic() or (flow.owner, flow.scope) == (owner, scope):
                    await stop(flow_id)
            if len(pending) >= 4:
                return reply({"error": "provider-sign-in-busy"}, 429)
            revision = await asyncio.to_thread(credentials.revision)
            flow_id = secrets.token_urlsafe(32)
            flow = _SignIn(owner, scope, revision, time.monotonic() + 600, method)
            pending[flow_id] = flow
            result: dict[str, object] = {"flowId": flow_id, "method": method, "status": "pending"}
            if method == "pkce":
                flow.state = secrets.token_urlsafe(32)
                flow.verifier = secrets.token_urlsafe(48)

                async def callback(callback_request: web.Request) -> web.Response:
                    query = callback_request.query
                    if (
                        flow_id not in pending
                        or flow.expires <= time.monotonic()
                        or not flow.state
                        or len(query.getall("state", [])) != 1
                        or not secrets.compare_digest(
                            query.get("state", "").encode("utf-8"), flow.state.encode("ascii")
                        )
                        or len(query.getall("code", [])) != 1
                        or not 1 <= len(query.get("code", "")) <= 8192
                    ):
                        return web.Response(
                            text="Sign-in unavailable. Return to Dinkster and try again.",
                            status=400,
                            headers=_HEADERS,
                        )
                    flow.code = query["code"]
                    flow.state = ""
                    return web.Response(
                        text="Authorization received. Return to Dinkster to finish signing in.",
                        headers=_HEADERS,
                    )

                callback_app = web.Application(client_max_size=16384)
                callback_app.router.add_get("/oauth/callback/huggingface", callback)
                flow.runner = web.AppRunner(callback_app, access_log=None)
                await flow.runner.setup()
                site = web.TCPSite(flow.runner, "127.0.0.1", 0)
                await site.start()
                port = flow.runner.addresses[0][1]
                flow.redirect = f"http://127.0.0.1:{port}/oauth/callback/huggingface"
                challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(flow.verifier.encode("ascii")).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )
                result["authorizationUrl"] = "https://huggingface.co/oauth/authorize?" + urlencode(
                    {
                        "client_id": HUGGINGFACE_CLIENT_ID,
                        "response_type": "code",
                        "scope": _SCOPE,
                        "redirect_uri": flow.redirect,
                        "state": flow.state,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    }
                )
            else:
                response = await exchange(
                    "device",
                    {
                        "client_id": HUGGINGFACE_CLIENT_ID,
                        "scope": _SCOPE,
                    },
                )
                device, user, uri = (
                    response.get("device_code"),
                    response.get("user_code"),
                    response.get("verification_uri"),
                )
                lifetime, interval = response.get("expires_in"), response.get("interval", 5)
                if (
                    not isinstance(device, str)
                    or not 1 <= len(device) <= 8192
                    or not isinstance(user, str)
                    or not 1 <= len(user) <= 128
                    or not isinstance(uri, str)
                    or len(uri) > 2048
                    or urlsplit(uri).scheme != "https"
                    or urlsplit(uri).netloc not in {"huggingface.co", "hf.co"}
                    or type(lifetime) is not int
                    or not 1 <= lifetime <= 3600
                    or type(interval) is not int
                    or not 1 <= interval <= 60
                ):
                    raise ProviderCredentialsError("Invalid Hugging Face device response")
                flow.device_code = device
                flow.expires = time.monotonic() + lifetime
                flow.interval = interval
                flow.next_poll = time.monotonic() + interval
                result.update(
                    userCode=user, verificationUri=uri, interval=interval, expiresIn=lifetime
                )

            def expire() -> None:
                task = asyncio.create_task(stop(flow_id))
                active.add(task)
                task.add_done_callback(active.discard)

            flow.timeout = asyncio.get_running_loop().call_later(
                max(0, flow.expires - time.monotonic()), expire
            )
            return reply(result, 201)
        except (ValueError, AssetError, OSError):
            await discard(flow_id)
            return reply({"error": "provider-sign-in-unavailable"}, 409)

    async def poll(request: web.Request) -> web.Response:
        owner, scope = authority(request)
        flow_id = request.match_info["flowId"]
        flow = pending.get(flow_id)
        if flow is None or (flow.owner, flow.scope) != (owner, scope):
            return reply({"error": "provider-sign-in-unavailable"}, 404)
        if request.method == "DELETE":
            if flow.status == "saving":
                return reply({"error": "provider-sign-in-completing"}, 409)
            flow.status = "cancelled"
            await discard(flow_id)
            return reply({"status": "cancelled"})
        async with flow.lock:
            if pending.get(flow_id) is not flow or flow.expires <= time.monotonic():
                await discard(flow_id)
                return reply({"error": "provider-sign-in-expired"}, 409)
            try:
                if await asyncio.to_thread(credentials.revision) != flow.revision:
                    await discard(flow_id)
                    return reply({"error": "provider-credentials-changed"}, 409)
                if flow.method == "pkce":
                    if not flow.code:
                        return reply({"status": "pending"})
                    fields = {
                        "grant_type": "authorization_code",
                        "code": flow.code,
                        "code_verifier": flow.verifier,
                        "redirect_uri": flow.redirect,
                    }
                    flow.code = ""
                else:
                    if time.monotonic() < flow.next_poll:
                        return reply({"status": "pending", "interval": flow.interval})
                    flow.next_poll = time.monotonic() + flow.interval
                    fields = {
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": flow.device_code,
                    }
                response = await exchange("token", {"client_id": HUGGINGFACE_CLIENT_ID, **fields})
                error = response.get("error")
                if (
                    flow.method == "device"
                    and isinstance(error, str)
                    and error in {"authorization_pending", "slow_down"}
                ):
                    if error == "slow_down":
                        flow.interval = min(flow.interval + 5, 300)
                    flow.next_poll = time.monotonic() + flow.interval
                    return reply({"status": "pending", "interval": flow.interval})
                if error:
                    await discard(flow_id)
                    return reply({"error": "provider-sign-in-refused"}, 409)
                saving = asyncio.create_task(complete(flow, response))
                try:
                    await asyncio.shield(saving)
                except asyncio.CancelledError:
                    await saving
                    raise
                await discard(flow_id)
                return reply({"status": "signed-in", "provider": "huggingface"})
            except asyncio.CancelledError:
                await discard(flow_id)
                raise
            except (ValueError, AssetError, OSError):
                await discard(flow_id)
                return reply({"error": "provider-sign-in-unavailable"}, 409)

    async def cleanup(_app: web.Application) -> None:
        for flow_id in tuple(pending):
            await stop(flow_id)
        if active:
            await asyncio.gather(*active)

    app.on_cleanup.append(cleanup)
    app.router.add_post("/api/models/providers/huggingface/sign-in", start)
    app.router.add_post("/api/models/providers/huggingface/sign-in/{flowId}", poll)
    app.router.add_delete("/api/models/providers/huggingface/sign-in/{flowId}", poll)
