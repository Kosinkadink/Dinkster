"""Inbound principals, capabilities, and agent permission categories.

Agent token grants are intersected with the currently enabled categories.
The category model is defined by ``PERMISSION_CATEGORIES``; humans are never
masked. Session creation and ending are governed by session-level ownership,
not a category, because all session writes share one capability. Whether an
agent may end a session is therefore a session ACL concern.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from aiohttp import web
from aiohttp.web_urldispatcher import ResourceRoute, SystemRoute
from dinkster_token_verifier import TokenVerifier  # pyright: ignore[reportMissingTypeStubs]

from .federated_assets_v1 import ErrorResponseV1, ErrorV1, encode_error_response

_BEARER_TOKEN = re.compile(r"[A-Za-z0-9\-._~+/]+=*")
_AUTHENTICATED = object()
WS_TICKET_TTL_SECONDS = 30

CAPABILITIES = frozenset(
    {
        "jobs:submit",
        "jobs:read",
        "jobs:cancel",
        "history:read",
        "assets:read",
        "assets:write",
        "settings:read",
        "settings:write",
        "sessions:read",
        "sessions:write",
        "queue:control",
        "memory:read",
        "cache:read",
        "training:read",
        "principals:manage",
    }
)

PERMISSION_CATEGORIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "edit": frozenset({"sessions:write"}),
        "execute": frozenset({"jobs:submit", "jobs:cancel"}),
        "read": frozenset(
            {
                "jobs:read",
                "history:read",
                "sessions:read",
                "settings:read",
                "assets:read",
                "memory:read",
                "cache:read",
                "training:read",
            }
        ),
        "assets": frozenset({"assets:write"}),
        "settings": frozenset({"settings:write"}),
        "queue": frozenset({"queue:control"}),
    }
)
AGENT_PERMISSION_DEFAULTS: Mapping[str, bool] = MappingProxyType(
    {
        "edit": True,
        "execute": True,
        "read": True,
        "assets": True,
        "settings": False,
        "queue": False,
    }
)


def _federated_error(
    code: str,
    reason: str,
    status: int,
    *,
    headers: Mapping[str, str] | None = None,
) -> web.Response:
    body = encode_error_response(ErrorResponseV1(ErrorV1(code, reason=reason)), status=status)
    return web.json_response(body, status=status, headers=headers)


class AuthError(ValueError):
    """An inbound authentication configuration is malformed."""


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated identity with immutable capabilities per scope."""

    principal_id: str
    grants: Mapping[str, frozenset[str]]
    kind: str = "human"
    expires_at: float | None = None
    delegation_id: str | None = None
    session_id: str | None = None
    display_name: str | None = None
    local: bool = False

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id must be non-empty")
        if self.kind not in {"human", "agent"}:
            raise ValueError("kind must be 'human' or 'agent'")
        frozen = {scope: frozenset(capabilities) for scope, capabilities in self.grants.items()}
        object.__setattr__(self, "grants", MappingProxyType(frozen))

    def allows(self, capability: str) -> bool:
        """Authorize a route when any scope grants the capability."""
        return any(capability in capabilities for capabilities in self.grants.values())

    def allows_in(self, scope: str, capability: str) -> bool:
        """Authorize one capability against one resource scope."""
        if self.local:
            return self.allows(capability)
        return capability in self.grants.get(scope, ())

    def scopes_for(self, capability: str) -> frozenset[str]:
        """Every scope granting a capability."""
        return frozenset(
            scope for scope, capabilities in self.grants.items() if capability in capabilities
        )


class Authenticator(Protocol):
    """Resolve one opaque bearer credential without framework coupling."""

    async def authenticate(self, token: str) -> Principal | None: ...


class StaticBearerAuthenticator:
    """Operator-managed static bearer credentials loaded from auth.toml."""

    def __init__(self, principals: Mapping[str, Principal]) -> None:
        self._principals = dict(principals)

    async def authenticate(self, token: str) -> Principal | None:
        # Compare opaque secrets without reflecting them in diagnostics.
        if _BEARER_TOKEN.fullmatch(token) is None:
            return None
        found = None
        for candidate, principal in self._principals.items():
            if secrets.compare_digest(token, candidate):
                found = principal
        return found

    def known_principals(self) -> tuple[Principal, ...]:
        """Return known identities without exposing their bearer tokens."""
        by_id: dict[str, Principal] = {}
        for principal in self._principals.values():
            by_id.setdefault(principal.principal_id, principal)
        return tuple(sorted(by_id.values(), key=lambda principal: principal.principal_id))

    def principal(self, principal_id: str) -> Principal | None:
        """Look up a known identity by public principal ID."""
        return next(
            (
                principal
                for principal in self._principals.values()
                if principal.principal_id == principal_id
            ),
            None,
        )


class TokenAuthenticator:
    """Verify identity-service access tokens against its cached JWKS."""

    def __init__(self, jwks_url: str, issuer: str, audience: str) -> None:
        self._verifier = TokenVerifier(
            jwks_url=jwks_url,
            issuer=issuer,
            audience=audience,
        )

    async def authenticate(self, token: str) -> Principal | None:
        verified = await self._verifier.verify(token)
        if verified is None:
            return None
        # Read expiry only after the verifier has checked this exact JWT.
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return Principal(
            principal_id=verified.principal_id,
            grants=verified.grants,
            kind=verified.kind,
            expires_at=float(claims["exp"]),
        )


class CompositeAuthenticator:
    """Try inbound credential authenticators in operator-defined order."""

    def __init__(self, *authenticators: Authenticator) -> None:
        if not authenticators:
            raise ValueError("at least one authenticator is required")
        self._authenticators = authenticators

    async def authenticate(self, token: str) -> Principal | None:
        for authenticator in self._authenticators:
            principal = await authenticator.authenticate(token)
            if principal is not None:
                return principal
        return None

    @property
    def static_authenticator(self) -> StaticBearerAuthenticator | None:
        """Return the static component used by principal administration."""
        return next(
            (
                authenticator
                for authenticator in self._authenticators
                if isinstance(authenticator, StaticBearerAuthenticator)
            ),
            None,
        )


class PrincipalPermissionStore:
    """Explicit agent permission overrides, optionally persisted in SQLite."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[str, bool]] = {}
        self._conn: sqlite3.Connection | None = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            with self._lock, self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS principal_permissions (
                        principal_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        enabled INTEGER NOT NULL,
                        PRIMARY KEY (principal_id, category)
                    )
                    """
                )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def categories(self, principal_id: str) -> dict[str, bool]:
        resolved = dict(AGENT_PERMISSION_DEFAULTS)
        with self._lock:
            if self._conn is None:
                resolved.update(self._values.get(principal_id, {}))
            else:
                rows = self._conn.execute(
                    "SELECT category, enabled FROM principal_permissions WHERE principal_id = ?",
                    (principal_id,),
                ).fetchall()
                resolved.update({str(category): bool(enabled) for category, enabled in rows})
        return resolved

    def update(self, principal_id: str, changes: Mapping[str, bool]) -> None:
        with self._lock:
            if self._conn is None:
                self._values.setdefault(principal_id, {}).update(changes)
                return
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO principal_permissions (principal_id, category, enabled)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT(principal_id, category) DO UPDATE SET enabled = excluded.enabled",
                    [
                        (principal_id, category, int(enabled))
                        for category, enabled in changes.items()
                    ],
                )

    def apply(self, principal: Principal) -> Principal:
        """Derive the effective request principal from token grants and toggles."""
        if principal.kind == "human":
            return principal
        categories = self.categories(principal.principal_id)
        allowed = frozenset(
            capability
            for category, capabilities in PERMISSION_CATEGORIES.items()
            if categories[category]
            for capability in capabilities
        )
        return replace(
            principal,
            grants={
                scope: capabilities & allowed for scope, capabilities in principal.grants.items()
            },
        )


LOCAL_PRINCIPAL = Principal(
    principal_id="local",
    grants={"local": CAPABILITIES},
    local=True,
)
PRINCIPAL_KEY: web.RequestKey[Principal] = web.RequestKey("dinkster_principal")
_RAW_PRINCIPAL_KEY: web.RequestKey[Principal] = web.RequestKey("dinkster_raw_principal")


def principal_for(request: web.Request) -> Principal:
    """Read the request-bound principal; there is no ambient identity."""
    principal = request[PRINCIPAL_KEY]
    if principal is LOCAL_PRINCIPAL:
        return principal
    if _RAW_PRINCIPAL_KEY not in request:
        return principal
    principal = request[_RAW_PRINCIPAL_KEY]
    if (principal.expires_at is not None and principal.expires_at <= time.time()) or (
        principal.delegation_id is not None
        and not request.app[DELEGATIONS_KEY].active(principal.delegation_id)
    ):
        return replace(principal, grants={})
    return request.app[PRINCIPAL_PERMISSIONS_KEY].apply(principal)


def resolve_scope(principal: Principal, capability: str, explicit_scope: object | None) -> str:
    """Resolve an optional resource scope or raise the public RBAC refusal."""
    if explicit_scope is not None:
        if (
            not isinstance(explicit_scope, str)
            or not explicit_scope
            or explicit_scope != explicit_scope.strip()
            or any(ch.isspace() for ch in explicit_scope)
        ):
            raise web.HTTPBadRequest(
                text='{"error": "scope must be a non-empty whitespace-free string"}',
                content_type="application/json",
            )
        if principal.local and principal.allows(capability):
            return explicit_scope
        if not principal.allows_in(explicit_scope, capability):
            raise web.HTTPForbidden(
                text=json.dumps(
                    {
                        "error": "capability-required",
                        "capability": capability,
                        "scope": explicit_scope,
                        "message": f"scope {explicit_scope} requires {capability}",
                    }
                ),
                content_type="application/json",
            )
        return explicit_scope
    if principal.local:
        return "local"
    scopes = sorted(principal.scopes_for(capability))
    if len(scopes) == 1:
        return scopes[0]
    raise web.HTTPBadRequest(
        text=json.dumps(
            {
                "error": "scope-required",
                "capability": capability,
                "scopes": scopes,
                "message": "an explicit scope is required",
            }
        ),
        content_type="application/json",
    )


class WebSocketTicketStore:
    """Short-lived, single-use browser WebSocket credentials."""

    def __init__(self) -> None:
        self._tickets: dict[str, tuple[Principal, float]] = {}

    def mint(self, principal: Principal) -> str:
        self._sweep()
        ticket = secrets.token_urlsafe(32)
        self._tickets[ticket] = (principal, time.monotonic() + WS_TICKET_TTL_SECONDS)
        return ticket

    def redeem(self, ticket: str) -> Principal | None:
        now = time.monotonic()
        found = self._tickets.pop(ticket, None)
        self._sweep(now)
        if found is None or found[1] <= now:
            return None
        return found[0]

    def _sweep(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        for ticket, (_, expiry) in list(self._tickets.items()):
            if expiry <= current:
                del self._tickets[ticket]


WS_TICKETS_KEY: web.AppKey[WebSocketTicketStore] = web.AppKey("ws_tickets")


async def handle_ws_ticket(request: web.Request) -> web.Response:
    ticket = request.app[WS_TICKETS_KEY].mint(request[_RAW_PRINCIPAL_KEY])
    return web.json_response({"ticket": ticket, "expiresInSeconds": WS_TICKET_TTL_SECONDS})


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AuthError(f"{where} must be a table")
    return {str(key): item for key, item in cast("Mapping[object, object]", value).items()}


def _exact_keys(body: Mapping[str, object], expected: set[str], where: str) -> None:
    if set(body) != expected:
        raise AuthError(f"{where} must contain exactly {sorted(expected)}, got {sorted(body)}")


def _token_entry_keys(body: Mapping[str, object], where: str) -> None:
    required = {"token", "principalId", "grants"}
    allowed = required | {"kind"}
    if not required <= set(body) or not set(body) <= allowed:
        raise AuthError(
            f"{where} must contain {sorted(required)} and optional 'kind', got {sorted(body)}"
        )


def load_authenticator(path: Path) -> StaticBearerAuthenticator:
    """Load a strict v1 static-token file or refuse startup loudly."""
    try:
        raw = tomllib.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuthError(f"{path}: invalid auth TOML: {exc}") from exc
    body = _object(raw, str(path))
    _exact_keys(body, {"version", "tokens"}, str(path))
    if type(body["version"]) is not int or body["version"] != 1:
        raise AuthError(f"{path}: version must be 1")
    raw_tokens = body["tokens"]
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise AuthError(f"{path}: tokens must be a non-empty array of tables")

    principals: dict[str, Principal] = {}
    principal_kinds: dict[str, str] = {}
    for index, raw_entry in enumerate(cast("list[object]", raw_tokens)):
        where = f"{path}: tokens[{index}]"
        entry = _object(raw_entry, where)
        _token_entry_keys(entry, where)
        token = entry["token"]
        principal_id = entry["principalId"]
        kind = entry.get("kind", "human")
        if not isinstance(token, str) or _BEARER_TOKEN.fullmatch(token) is None:
            raise AuthError(f"{where}.token must use the Bearer token character set")
        if token in principals:
            raise AuthError(f"{path}: duplicate bearer token at tokens[{index}]")
        if (
            not isinstance(principal_id, str)
            or not principal_id
            or principal_id != principal_id.strip()
        ):
            raise AuthError(f"{where}.principalId must be a non-empty trimmed string")
        if kind not in {"human", "agent"}:
            raise AuthError(f"{where}.kind must be 'human' or 'agent'")
        previous_kind = principal_kinds.setdefault(principal_id, cast(str, kind))
        if previous_kind != kind:
            raise AuthError(
                f"{where}.kind must match other tokens for principalId {principal_id!r}"
            )
        raw_grants = _object(entry["grants"], f"{where}.grants")
        if not raw_grants:
            raise AuthError(f"{where}.grants must contain at least one scope")
        grants: dict[str, frozenset[str]] = {}
        for scope, raw_capabilities in raw_grants.items():
            if not scope or scope != scope.strip() or any(ch.isspace() for ch in scope):
                raise AuthError(f"{where}.grants scope names must be whitespace-free")
            if not isinstance(raw_capabilities, list):
                raise AuthError(f"{where}.grants[{scope!r}] must be an array of strings")
            capability_items = cast("list[object]", raw_capabilities)
            if not all(isinstance(item, str) for item in capability_items):
                raise AuthError(f"{where}.grants[{scope!r}] must be an array of strings")
            capabilities = cast("list[str]", capability_items)
            unknown = set(capabilities) - CAPABILITIES
            if unknown:
                raise AuthError(
                    f"{where}.grants[{scope!r}] has unknown capabilities: {sorted(unknown)}"
                )
            if len(set(capabilities)) != len(capabilities):
                raise AuthError(f"{where}.grants[{scope!r}] contains duplicates")
            grants[scope] = frozenset(capabilities)
        principals[token] = Principal(
            principal_id=principal_id, grants=grants, kind=cast(str, kind)
        )
    return StaticBearerAuthenticator(principals)


async def handle_principals(request: web.Request) -> web.Response:
    store = request.app[PRINCIPAL_PERMISSIONS_KEY]
    caller = principal_for(request)
    if caller.kind != "human":
        return web.json_response({"error": "human-session-required"}, status=403)
    principals = request.app[_KNOWN_PRINCIPALS_KEY].values()
    return web.json_response(
        [
            {
                "principalId": principal.principal_id,
                "kind": principal.kind,
                "local": principal.local,
                "categories": store.categories(principal.principal_id),
                "scopes": sorted(principal.grants),
                "self": principal.principal_id == caller.principal_id,
            }
            for principal in principals
            if principal.principal_id == caller.principal_id or caller.allows("principals:manage")
        ]
    )


async def handle_principal_permissions(request: web.Request) -> web.Response:
    principal_id = request.match_info["principal_id"]
    caller = principal_for(request)
    if caller.kind != "human" or (
        caller.principal_id != principal_id and not caller.allows("principals:manage")
    ):
        return web.json_response({"error": "permission-owner-required"}, status=403)
    principal = request.app[_KNOWN_PRINCIPALS_KEY].get(principal_id)
    if principal is None:
        return web.json_response({"error": "unknown-principal"}, status=404)
    try:
        body = await request.json()
    except (json.JSONDecodeError, web.HTTPBadRequest):
        return web.json_response({"error": "invalid-permissions"}, status=400)
    if not isinstance(body, Mapping):
        return web.json_response({"error": "invalid-permissions"}, status=400)
    body_mapping = cast("Mapping[object, object]", body)
    changes = {str(category): enabled for category, enabled in body_mapping.items()}
    unknown = set(changes) - set(PERMISSION_CATEGORIES)
    if unknown:
        return web.json_response(
            {"error": "unknown-permission-category", "categories": sorted(unknown)},
            status=400,
        )
    if not all(type(enabled) is bool for enabled in changes.values()):
        return web.json_response({"error": "invalid-permissions"}, status=400)
    store = request.app[PRINCIPAL_PERMISSIONS_KEY]
    store.update(principal_id, cast("Mapping[str, bool]", changes))
    return web.json_response(store.categories(principal_id))


PRINCIPAL_PERMISSIONS_KEY: web.AppKey[PrincipalPermissionStore] = web.AppKey(
    "principal_permissions"
)
_KNOWN_PRINCIPALS_KEY: web.AppKey[dict[str, Principal]] = web.AppKey("known_principals")


def add_principal_routes(
    app: web.Application,
    authenticator: Authenticator | None,
    store: PrincipalPermissionStore,
) -> None:
    app[PRINCIPAL_PERMISSIONS_KEY] = store
    static_authenticator = None
    if isinstance(authenticator, StaticBearerAuthenticator):
        static_authenticator = authenticator
    elif isinstance(authenticator, CompositeAuthenticator):
        static_authenticator = authenticator.static_authenticator
    if static_authenticator is not None:
        app[_KNOWN_PRINCIPALS_KEY].update(
            (principal.principal_id, principal)
            for principal in static_authenticator.known_principals()
        )
    app.router.add_get("/api/principals", handle_principals)
    app.router.add_put(
        "/api/principals/{principal_id}/permissions",
        handle_principal_permissions,
    )


class DelegationStore:
    """Bounded, short-lived credentials; restart invalidates all delegations."""

    def __init__(self) -> None:
        self._tokens: dict[str, Principal] = {}

    def _sweep(self) -> None:
        now = time.time()
        self._tokens = {
            digest: principal
            for digest, principal in self._tokens.items()
            if principal.expires_at is not None and principal.expires_at > now
        }

    def active(self, delegation_id: str) -> bool:
        self._sweep()
        return any(p.delegation_id == delegation_id for p in self._tokens.values())

    def authenticate(self, token: str) -> Principal | None:
        self._sweep()
        return self._tokens.get(hashlib.sha256(token.encode()).hexdigest())

    def list(self, principal_id: str) -> list[dict[str, object]]:
        self._sweep()
        return [
            {
                "id": p.delegation_id,
                "displayName": p.display_name,
                "scope": next(iter(p.grants)),
                "sessionId": p.session_id,
                "expiresAt": p.expires_at,
                "kind": "agent",
            }
            for p in self._tokens.values()
            if p.principal_id == principal_id
        ]

    def revoke(self, principal_id: str, delegation_id: str) -> None:
        self._tokens = {
            digest: p
            for digest, p in self._tokens.items()
            if p.principal_id != principal_id or p.delegation_id != delegation_id
        }

    def mint(
        self, principal: Principal, scope: str, name: str, session_id: str | None, lifetime: int
    ) -> dict[str, object]:
        self._sweep()
        if len(self._tokens) >= 4096 or len(self.list(principal.principal_id)) >= 32:
            raise web.HTTPTooManyRequests(
                text='{"error":"delegation-limit"}', content_type="application/json"
            )
        token = "dinkster_delegate_" + secrets.token_urlsafe(32)
        expiry = min(time.time() + lifetime, principal.expires_at or float("inf"))
        delegate = replace(
            principal,
            kind="agent",
            expires_at=expiry,
            delegation_id=secrets.token_urlsafe(16),
            session_id=session_id,
            display_name=name,
            grants={scope: principal.grants[scope] - {"principals:manage"}},
        )
        self._tokens[hashlib.sha256(token.encode()).hexdigest()] = delegate
        return {"token": token, "id": delegate.delegation_id, "expiresAt": expiry}


DELEGATIONS_KEY: web.AppKey[DelegationStore] = web.AppKey("delegations")


async def handle_delegations(request: web.Request) -> web.Response:
    principal = principal_for(request)
    if principal.kind != "human" or principal is LOCAL_PRINCIPAL:
        return web.json_response({"error": "authenticated-human-session-required"}, status=403)
    store = request.app[DELEGATIONS_KEY]
    if request.method == "GET":
        return web.json_response(store.list(principal.principal_id))
    if request.method == "DELETE":
        store.revoke(principal.principal_id, request.match_info["delegation_id"])
        return web.json_response({"revoked": True})
    try:
        body = await request.json()
    except (ValueError, web.HTTPBadRequest):
        return web.json_response({"error": "invalid-delegation"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid-delegation"}, status=400)
    data = cast("dict[str, object]", body)
    scope, name = data.get("scope"), data.get("displayName")
    session_id, lifetime = data.get("sessionId"), data.get("expiresInSeconds", 600)
    if (
        not isinstance(scope, str)
        or scope not in principal.grants
        or not isinstance(name, str)
        or not 1 <= len(name.strip()) <= 64
        or (session_id is not None and (not isinstance(session_id, str) or not session_id))
        or type(lifetime) is not int
        or not 1 <= lifetime <= 600
    ):
        return web.json_response({"error": "invalid-delegation"}, status=400)
    return web.json_response(
        store.mint(principal, scope, name.strip(), session_id, lifetime),
        status=201,
    )


_CATALOG_PATHS = frozenset(
    {
        "/api/nodes",
        "/api/workers",
        "/api/composition",
        "/api/templates",
        "/api/diagnostics",
        "/api/extensions/snapshot",
    }
)
_CATALOG_PREFIXES = ("/api/choices/", "/api/packs/")


def required_capability(method: str, path: str) -> str | object | None:
    """Map the existing HTTP route matrix onto the frozen vocabulary."""
    read = method in {"GET", "HEAD"}
    if read and (
        path in _CATALOG_PATHS or any(path.startswith(prefix) for prefix in _CATALOG_PREFIXES)
    ):
        return _AUTHENTICATED
    if path == "/api/auth/ws-ticket" or path.startswith("/api/auth/delegations"):
        return _AUTHENTICATED
    if path == "/api/principals" or path.startswith("/api/principals/"):
        return _AUTHENTICATED
    if path == "/api/compat/comfy/prompt":
        return "jobs:submit"
    if path == "/api/jobs":
        return "jobs:submit" if method == "POST" else "jobs:read"
    if path.startswith("/api/jobs/"):
        return "jobs:cancel" if method == "DELETE" else "jobs:read"
    if path in {"/api/generation/models", "/v1/models"}:
        return _AUTHENTICATED
    if path in {"/api/generation/models/load", "/api/generation/models/unload"}:
        return "settings:write"
    if path.startswith("/api/generation/sessions/"):
        return "sessions:write"
    if path == "/api/generation" or path in {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/responses",
    }:
        return "jobs:submit"
    if path == "/api/values" or path == "/api/events":
        return "jobs:read"
    if path.startswith("/api/runs/"):
        return "jobs:read"
    if path.startswith("/api/history"):
        return "history:read"
    if path.startswith("/api/training"):
        return "training:read"
    if path.startswith("/api/settings"):
        return "settings:read" if read else "settings:write"
    if path.startswith("/api/p2p"):
        return "settings:read" if read else "settings:write"
    if path.startswith("/api/sessions"):
        if path.endswith("/actors") and method == "POST":
            return "sessions:read"
        return "sessions:read" if read else "sessions:write"
    if path.startswith("/api/queue"):
        return "queue:control"
    if path.startswith("/memory"):
        return "memory:read"
    if path.startswith("/cache"):
        return "cache:read"
    if path.startswith("/api/install/activation"):
        return "settings:read" if read else "settings:write"
    if path.startswith("/api/packs/"):
        return "settings:write"
    if path == "/api/assets/guess":
        return "assets:read"
    if read and path == "/api/output-profiles/model":
        return "assets:read"
    if path.startswith(("/api/assets", "/api/library", "/api/mounts", "/assets")):
        return "assets:read" if read else "assets:write"
    return None


def install_auth(
    app: web.Application,
    authenticator: Authenticator | None = None,
    *,
    route_capabilities: Mapping[tuple[str, str], str] | None = None,
    federated_asset_paths: frozenset[str] = frozenset(),
    permission_store: PrincipalPermissionStore | None = None,
) -> None:
    """Attach one principal per request and enforce route capabilities."""
    app[WS_TICKETS_KEY] = WebSocketTicketStore()
    store = permission_store or PrincipalPermissionStore()
    app[PRINCIPAL_PERMISSIONS_KEY] = store
    app[_KNOWN_PRINCIPALS_KEY] = {}
    app[DELEGATIONS_KEY] = DelegationStore()
    app.router.add_get("/api/auth/delegations", handle_delegations)
    app.router.add_post("/api/auth/delegations", handle_delegations)
    app.router.add_delete("/api/auth/delegations/{delegation_id}", handle_delegations)

    @web.middleware
    async def authenticate(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        route = request.match_info.route
        resource = route.resource if isinstance(route, ResourceRoute) else None
        canonical = resource.canonical if resource is not None else None
        is_media_ingest = request.method == "POST" and canonical == "/api/assets/media"
        if authenticator is None:
            local = LOCAL_PRINCIPAL
            if request.headers.get("X-Dinkster-Actor-Kind") == "agent":
                local = replace(LOCAL_PRINCIPAL, kind="agent")
            if request.headers.get("Upgrade", "").lower() == "websocket":
                local = request.app[WS_TICKETS_KEY].redeem(request.query.get("ticket", "")) or local
            request[PRINCIPAL_KEY] = local
            request[_RAW_PRINCIPAL_KEY] = local
        elif request.path == "/api/health":
            return await handler(request)
        else:
            scheme, separator, token = request.headers.get("Authorization", "").partition(" ")
            principal = None
            if separator and scheme.lower() == "bearer" and _BEARER_TOKEN.fullmatch(token):
                principal = request.app[DELEGATIONS_KEY].authenticate(token)
                if principal is None:
                    principal = await authenticator.authenticate(token)
            is_ws_route = request.method == "GET" and canonical in (
                "/api/events",
                "/api/sessions/{session_id}/events",
            )
            if (
                principal is None
                and is_ws_route
                and request.headers.get("Upgrade", "").lower() == "websocket"
            ):
                ticket = request.query.get("ticket")
                if ticket is not None:
                    principal = request.app[WS_TICKETS_KEY].redeem(ticket)
            if principal is None:
                if canonical in federated_asset_paths:
                    return _federated_error(
                        "authentication-required",
                        "a valid Bearer credential is required",
                        401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                if is_media_ingest:
                    return web.json_response(
                        {
                            "error": {
                                "code": "asset.media.authentication_required",
                                "message": "a valid Bearer credential is required",
                            }
                        },
                        status=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return web.json_response(
                    {
                        "error": "authentication-required",
                        "message": "a valid Bearer credential is required",
                    },
                    status=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request[_RAW_PRINCIPAL_KEY] = principal
            request[PRINCIPAL_KEY] = store.apply(principal)

        raw = request[_RAW_PRINCIPAL_KEY]
        if (
            authenticator is not None
            and raw.kind != "agent"
            and request.headers.get("X-Dinkster-Actor-Kind") == "agent"
        ):
            return web.json_response({"error": "agent-delegation-required"}, status=403)
        if raw.kind == "human" or (not raw.local and raw.delegation_id is None):
            known = request.app[_KNOWN_PRINCIPALS_KEY]
            if len(known) >= 4096 and raw.principal_id not in known:
                known.pop(next(iter(known)))
            known[raw.principal_id] = raw
        if raw.session_id is not None and (
            request.path != "/api/auth/ws-ticket"
            and request.match_info.get("session_id") != raw.session_id
        ):
            return web.json_response({"error": "delegation-session-required"}, status=403)

        capability = (route_capabilities or {}).get(
            (request.method, cast(str, canonical))
        ) or required_capability(request.method, request.path)
        principal = principal_for(request)
        if (
            capability is None
            and authenticator is not None
            and not isinstance(request.match_info.route, SystemRoute)
        ):
            return web.json_response(
                {
                    "error": "authorization-policy-missing",
                    "message": "this route is not classified by the capability policy",
                },
                status=403,
            )
        if isinstance(capability, str) and not principal.allows(capability):
            if canonical in federated_asset_paths:
                return _federated_error(
                    "forbidden",
                    f"this route requires {capability}",
                    403,
                )
            if is_media_ingest:
                return web.json_response(
                    {
                        "error": {
                            "code": "asset.media.capability_required",
                            "message": "this route requires assets:write",
                        }
                    },
                    status=403,
                )
            return web.json_response(
                {
                    "error": "capability-required",
                    "capability": capability,
                    "message": f"this route requires {capability}",
                },
                status=403,
            )
        return await handler(request)

    app.middlewares.append(authenticate)
