"""Owner-bound provider PKCE and Hugging Face device-code sign-in."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import ipaddress
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

from aiohttp import web
from dinkster_assets import AssetError
from dinkster_assets.fetch import post_provider_form
from dinkster_server.auth import principal_for, resolve_scope

from .frontend import load_provider_callback
from .provider_credentials import (
    HUGGINGFACE_CLIENT_ID,
    ProviderCredentials,
    ProviderCredentialsError,
    ProviderOAuthResponseError,
    oauth_credential,
)

_SCOPE = "openid profile gated-repos"
_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
_log = logging.getLogger(__name__)


def valid_public_client_id(client_id: str) -> bool:
    return 1 <= len(client_id) <= 256 and all(
        33 <= ord(character) <= 126 for character in client_id
    )


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
    error: str = ""
    failure_message: str = "Dinkster could not complete sign-in."
    runner: web.AppRunner | None = field(default=None, repr=False)
    timeout: asyncio.TimerHandle | None = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def add_provider_sign_in_routes(
    app: web.Application,
    credentials: ProviderCredentials,
    changed: Callable[[], None],
    *,
    provider: str = "huggingface",
    client_id: str = "",
) -> None:
    if provider not in {"huggingface", "civitai"}:
        raise ValueError("Unsupported sign-in provider")
    if provider == "huggingface":
        client_id = HUGGINGFACE_CLIENT_ID
    pending: dict[str, _SignIn] = {}
    active: set[asyncio.Task[None]] = set()
    creating = asyncio.Lock()

    async def exchange(endpoint: str, fields: dict[str, str]) -> dict[str, object]:
        task = asyncio.create_task(
            asyncio.to_thread(post_provider_form, endpoint, fields)
            if provider == "huggingface"
            else asyncio.to_thread(post_provider_form, endpoint, fields, provider=provider)
        )
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
        _log.info("Provider sign-in: provider=%s stage=credential-validation", provider)
        token, expires, refresh = oauth_credential(response, provider=provider)
        flow.status = "saving"
        _log.info("Provider sign-in: provider=%s stage=credential-save", provider)
        flow.revision = await asyncio.to_thread(
            credentials.set_token,
            token,
            provider=provider,
            expires_at=expires,
            refresh_token=refresh,
            expected_revision=flow.revision,
            client_id=client_id if provider == "civitai" else None,
        )
        changed()
        flow.status = "signed-in"

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
            if not valid_public_client_id(client_id):
                return reply({"error": "provider-sign-in-unavailable"}, 409)
            if provider == "civitai" and method != "pkce":
                return reply({"error": "invalid-provider-sign-in"}, 400)
            if method == "pkce":
                try:
                    host = urlsplit(f"http://{request.host}").hostname or ""
                    local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
                    local_peer = ipaddress.ip_address(request.remote or "").is_loopback
                except ValueError:
                    local_host = local_peer = False
                if not local_host or not local_peer:
                    return reply(
                        {
                            "error": "device-sign-in-required"
                            if provider == "huggingface"
                            else "local-sign-in-required"
                        },
                        409,
                    )
            template = csp = return_url = ""
            if provider == "civitai":
                origins = request.headers.getall("Origin", [])
                origin = origins[0] if len(origins) == 1 else f"{request.scheme}://{request.host}"
                if (
                    len(origins) > 1
                    or not 1 <= len(origin) <= 2048
                    or any(
                        ord(character) < 33 or ord(character) > 126 or character == "\\"
                        for character in origin
                    )
                ):
                    return reply({"error": "invalid-provider-sign-in"}, 400)
                try:
                    destination = urlsplit(origin)
                    hostname = destination.hostname or ""
                    local_origin = (
                        hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
                    )
                    port = destination.port
                except ValueError:
                    return reply({"error": "invalid-provider-sign-in"}, 400)
                if (
                    not local_origin
                    or destination.scheme not in {"http", "https"}
                    or not destination.netloc
                    or destination.username is not None
                    or destination.password is not None
                    or destination.path
                    or destination.query
                    or destination.fragment
                    or "?" in origin
                    or "#" in origin
                    or port == 0
                    or destination.netloc.endswith(":")
                ):
                    return reply({"error": "invalid-provider-sign-in"}, 400)
                return_url = destination.geturl() + "/"
                template, csp = load_provider_callback()
            for flow_id, flow in tuple(pending.items()):
                if flow.expires <= time.monotonic() or (flow.owner, flow.scope) == (owner, scope):
                    await stop(flow_id)
            if len(pending) >= 4:
                return reply({"error": "provider-sign-in-busy"}, 429)
            revision = await asyncio.to_thread(credentials.revision, provider)
            flow_id = secrets.token_urlsafe(32)
            flow = _SignIn(owner, scope, revision, time.monotonic() + 600, method)
            pending[flow_id] = flow
            result: dict[str, object] = {"flowId": flow_id, "method": method, "status": "pending"}
            if method == "pkce":
                flow.state = secrets.token_urlsafe(32)
                flow.verifier = secrets.token_urlsafe(48)

                def terminal(heading: str, message: str, status: int = 200) -> web.Response:
                    if provider != "civitai":
                        return web.Response(text=message, status=status, headers=_HEADERS)
                    values = {
                        "__CALLBACK_HEADING__": heading,
                        "__CALLBACK_MESSAGE__": message,
                        "__CALLBACK_RETURN_LABEL__": "Return to Dinkster",
                        "__CALLBACK_RETURN_URL__": return_url,
                    }
                    document = re.sub(
                        "|".join(map(re.escape, values)),
                        lambda match: html.escape(values[match[0]], quote=True),
                        template,
                    )
                    return web.Response(
                        text=document,
                        status=status,
                        content_type="text/html",
                        charset="utf-8",
                        headers={
                            **_HEADERS,
                            "X-Content-Type-Options": "nosniff",
                            "Content-Security-Policy": csp + "; frame-ancestors 'none'",
                        },
                    )

                async def callback(callback_request: web.Request) -> web.Response:
                    query = callback_request.query
                    codes = query.getall("code", [])
                    errors = query.getall("error", [])
                    has_code = len(codes) == 1 and 1 <= len(codes[0]) <= 8192 and not errors
                    refused = len(errors) == 1 and 1 <= len(errors[0]) <= 8192 and not codes
                    if (
                        flow_id not in pending
                        or flow.expires <= time.monotonic()
                        or not flow.state
                        or len(query.getall("state", [])) != 1
                        or not secrets.compare_digest(
                            query.get("state", "").encode("utf-8"), flow.state.encode("ascii")
                        )
                        or not (has_code or refused)
                    ):
                        return terminal(
                            "Sign-in unavailable",
                            "Sign-in unavailable. Return to Dinkster and try again.",
                            400,
                        )
                    flow.state = ""
                    _log.info(
                        "Provider sign-in: provider=%s stage=callback outcome=%s",
                        provider,
                        "refused" if refused else "accepted",
                    )
                    if refused or provider == "civitai":
                        await site.stop()
                        runner, flow.runner = flow.runner, None
                        if runner is not None:
                            # Cleanup waits for this response, so it runs outside the handler.
                            task = asyncio.create_task(runner.cleanup())
                            active.add(task)
                            task.add_done_callback(active.discard)
                    if refused:
                        flow.status = "refused"
                        flow.verifier = ""
                        return terminal(
                            "Sign-in refused", "Sign-in refused. Return to Dinkster and try again."
                        )
                    if provider == "huggingface":
                        flow.code = query["code"]
                        return web.Response(
                            text="Authorization received. Return to Dinkster to finish signing in.",
                            headers=_HEADERS,
                        )
                    finishing = asyncio.create_task(advance(flow_id, flow, query["code"]))
                    try:
                        result = await asyncio.shield(finishing)
                    except asyncio.CancelledError:
                        await finishing
                        raise
                    return terminal(
                        "Signed in" if result.status == 200 else "Sign-in failed",
                        "Signed in. Return to Dinkster."
                        if result.status == 200
                        else flow.failure_message + " Return to Dinkster and check sign-in status.",
                        result.status,
                    )

                callback_app = web.Application(client_max_size=16384)
                callback_app.router.add_get(f"/oauth/callback/{provider}", callback)
                flow.runner = web.AppRunner(callback_app, access_log=None)
                await flow.runner.setup()
                site = web.TCPSite(flow.runner, "127.0.0.1", 15493 if provider == "civitai" else 0)
                await site.start()
                port = flow.runner.addresses[0][1]
                host = "localhost" if provider == "civitai" else "127.0.0.1"
                flow.redirect = f"http://{host}:{port}/oauth/callback/{provider}"
                challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(flow.verifier.encode("ascii")).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )
                authorize = (
                    "https://huggingface.co/oauth/authorize"
                    if provider == "huggingface"
                    else "https://auth.civitai.com/api/auth/oauth/authorize"
                )
                result["authorizationUrl"] = (
                    authorize
                    + "?"
                    + urlencode(
                        {
                            "client_id": client_id,
                            "response_type": "code",
                            "scope": _SCOPE if provider == "huggingface" else "4",
                            "redirect_uri": flow.redirect,
                            "state": flow.state,
                            "code_challenge": challenge,
                            "code_challenge_method": "S256",
                        }
                    )
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
        result = await advance(flow_id, flow)
        if flow.error or flow.status == "signed-in":
            await discard(flow_id)
        return result

    async def advance(flow_id: str, flow: _SignIn, code: str | None = None) -> web.Response:
        async with flow.lock:
            if pending.get(flow_id) is not flow or flow.expires <= time.monotonic():
                flow.error = "provider-sign-in-expired"
                flow.failure_message = "This sign-in session expired before it could finish."
                return reply({"error": flow.error}, 409)
            if code is not None:
                flow.code = code
            stage = "credential-revision"
            try:
                if flow.status == "refused":
                    flow.error = "provider-sign-in-refused"
                if flow.error:
                    return reply({"error": flow.error}, 409)
                if await asyncio.to_thread(credentials.revision, provider) != flow.revision:
                    flow.error = "provider-credentials-changed"
                    flow.failure_message = (
                        "Sign-in settings changed while authorization was in progress."
                    )
                    return reply({"error": flow.error}, 409)
                if flow.status == "signed-in":
                    return reply({"status": "signed-in", "provider": provider})
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
                stage = "token-exchange"
                _log.info("Provider sign-in: provider=%s stage=%s", provider, stage)
                response = await exchange("token", {"client_id": client_id, **fields})
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
                    reason = (
                        error
                        if isinstance(error, str)
                        and error
                        in {
                            "invalid_grant",
                            "invalid_client",
                            "invalid_request",
                            "unsupported_grant_type",
                            "invalid_scope",
                            "access_denied",
                            "expired_token",
                        }
                        else "provider-error"
                    )
                    _log.warning(
                        "Provider sign-in: provider=%s stage=%s outcome=refused reason=%s",
                        provider,
                        stage,
                        reason,
                    )
                    flow.failure_message = {
                        "invalid_grant": "The provider rejected this authorization.",
                        "invalid_client": "The provider did not recognize Dinkster's registration.",
                        "invalid_request": "The provider rejected Dinkster's sign-in request.",
                        "unsupported_grant_type": (
                            "The provider does not support this sign-in method."
                        ),
                        "invalid_scope": "The provider did not allow the requested model access.",
                        "access_denied": "The provider denied this sign-in.",
                        "expired_token": "The provider reported that this authorization expired.",
                    }.get(reason, "The provider refused to complete sign-in.")
                    flow.error = "provider-sign-in-refused"
                    return reply({"error": flow.error}, 409)
                stage = "credential-validation"
                saving = asyncio.create_task(complete(flow, response))
                try:
                    await asyncio.shield(saving)
                except asyncio.CancelledError:
                    await saving
                    raise
                flow.verifier = flow.device_code = ""
                _log.info("Provider sign-in: provider=%s outcome=signed-in", provider)
                return reply({"status": "signed-in", "provider": provider})
            except asyncio.CancelledError:
                _log.info(
                    "Provider sign-in: provider=%s stage=%s outcome=cancelled", provider, stage
                )
                await discard(flow_id)
                raise
            except (ValueError, AssetError, OSError) as exc:
                if isinstance(exc, ProviderOAuthResponseError):
                    flow.failure_message = str(exc)
                    _log.warning(
                        "Provider sign-in: provider=%s stage=credential-validation reason=%s",
                        provider,
                        exc.reason,
                    )
                elif flow.status == "saving":
                    flow.failure_message = (
                        "Dinkster could not save sign-in credentials on this engine."
                    )
                elif stage == "token-exchange":
                    flow.failure_message = (
                        "Dinkster could not obtain a valid sign-in response from the provider."
                    )
                elif stage == "credential-validation":
                    flow.failure_message = (
                        "The sign-in session changed or expired before credentials could be saved."
                    )
                else:
                    flow.failure_message = (
                        "Dinkster could not read the sign-in settings on this engine."
                    )
                _log.warning(
                    "Provider sign-in: provider=%s stage=%s outcome=failed exception=%s",
                    provider,
                    "credential-save" if flow.status == "saving" else stage,
                    type(exc).__name__,
                )
                flow.error = "provider-sign-in-unavailable"
                return reply({"error": flow.error}, 409)
            finally:
                if flow.error:
                    flow.status = "failed"
                    flow.code = flow.verifier = flow.device_code = ""

    async def cleanup(_app: web.Application) -> None:
        for flow_id in tuple(pending):
            await stop(flow_id)
        if active:
            await asyncio.gather(*active)

    app.on_cleanup.append(cleanup)
    path = f"/api/models/providers/{provider}/sign-in"
    app.router.add_post(path, start)
    app.router.add_post(path + "/{flowId}", poll)
    app.router.add_delete(path + "/{flowId}", poll)
