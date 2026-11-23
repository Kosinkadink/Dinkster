import asyncio
import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import dinkster_token_verifier.verifier as verifier_module
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_server import (
    CompositeAuthenticator,
    Principal,
    StaticBearerAuthenticator,
    TokenAuthenticator,
    create_app,
    principal_for,
)
from joserfc import jwt
from joserfc.jwk import OctKey, OKPKey
from test_server import SCHEMAS, make_engine

from tests.async_scenarios import run_scenario

ISSUER = "https://identity.example"
AUDIENCE = "dinkster-session"
KID = "identity-key"
PRINCIPAL_ID = f"u_{UUID(int=1)}"
SCOPE = "project:01900000-0000-7000-8000-000000000001"


def _encode_segment(value: dict[str, object]) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _claims(**changes: object) -> dict[str, object]:
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": PRINCIPAL_ID,
        "kind": "human",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "org": None,
        "acct": str(UUID(int=7)),
        "key": None,
        "grants": {SCOPE: ["history:read", "jobs:read"]},
        "iat": now,
        "exp": now + 600,
        "jti": str(UUID(int=2)),
    }
    claims.update(changes)
    return claims


def _sign(key: OKPKey, claims: dict[str, object] | None = None) -> str:
    return jwt.encode(
        {"alg": "Ed25519", "kid": KID, "typ": "JWT"},
        claims or _claims(),
        key,
        algorithms=["Ed25519"],
    )


class _JwksService:
    def __init__(self, key: OKPKey) -> None:
        self.available = True
        self.requests = 0
        self.app = web.Application()
        self.app.router.add_get("/.well-known/jwks.json", self.handle)
        self.server = TestServer(self.app)
        self.public_jwk = key.as_dict(
            private=False,
            kid=KID,
            alg="Ed25519",
            use="sig",
        )

    async def handle(self, request: web.Request) -> web.Response:
        self.requests += 1
        if not self.available:
            raise web.HTTPServiceUnavailable()
        return web.json_response(
            {"keys": [self.public_jwk]},
            headers={"Cache-Control": "public, max-age=1"},
        )

    async def start(self) -> str:
        await self.server.start_server()
        return str(self.server.make_url("/.well-known/jwks.json"))

    async def close(self) -> None:
        await self.server.close()


def test_token_authenticator_rejects_invalid_credentials_without_raising() -> None:
    async def scenario() -> None:
        key = OKPKey.generate_key("Ed25519")
        service = _JwksService(key)
        jwks_url = await service.start()
        try:
            authenticator = TokenAuthenticator(jwks_url, ISSUER, AUDIENCE)
            valid = _sign(key)
            assert await authenticator.authenticate(valid) is not None

            header, payload, signature = valid.split(".")
            decoded_payload = json.loads(
                base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            )
            decoded_payload["grants"] = {SCOPE: ["invented:capability"]}
            tampered = f"{header}.{_encode_segment(decoded_payload)}.{signature}"

            now = int(time.time())
            expired = _sign(key, _claims(iat=now - 60, exp=now - 31))
            excess_lifetime = _sign(key, _claims(iat=now, exp=now + 601))
            symmetric_key = OctKey.generate_key(256)
            hs256 = jwt.encode(
                {"alg": "HS256", "kid": KID, "typ": "JWT"},
                _claims(),
                symmetric_key,
                algorithms=["HS256"],
            )
            none = f"{_encode_segment({'alg': 'none', 'kid': KID})}.{payload}."
            eddsa = f"{_encode_segment({'alg': 'EdDSA', 'kid': KID})}.{payload}.{signature}"

            for credential in (
                "not-a-token",
                expired,
                tampered,
                hs256,
                none,
                eddsa,
                excess_lifetime,
            ):
                assert await authenticator.authenticate(credential) is None

            wrong_audience = TokenAuthenticator(jwks_url, ISSUER, "other-audience")
            wrong_issuer = TokenAuthenticator(jwks_url, "https://other.example", AUDIENCE)
            assert await wrong_audience.authenticate(valid) is None
            assert await wrong_issuer.authenticate(valid) is None
        finally:
            await service.close()

    run_scenario(scenario())


def test_token_authenticator_shares_fresh_and_stale_jwks_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        cache_time = 100.0
        monkeypatch.setattr(
            verifier_module,
            "time",
            SimpleNamespace(monotonic=lambda: cache_time, time=time.time),
        )
        key = OKPKey.generate_key("Ed25519")
        service = _JwksService(key)
        jwks_url = await service.start()
        try:
            authenticator = TokenAuthenticator(jwks_url, ISSUER, AUDIENCE)
            token = _sign(key)
            principals = await asyncio.gather(
                *(authenticator.authenticate(token) for _ in range(5))
            )
            assert all(principal is not None for principal in principals)
            assert service.requests == 1

            service.available = False
            cache_time += 1.05
            for _ in range(3):
                assert await authenticator.authenticate(token) is not None
            assert service.requests == 2
        finally:
            await service.close()

    run_scenario(scenario())


def test_identity_bearer_reaches_protected_endpoint_with_verified_principal() -> None:
    async def scenario() -> None:
        key = OKPKey.generate_key("Ed25519")
        service = _JwksService(key)
        jwks_url = await service.start()
        authenticator = TokenAuthenticator(jwks_url, ISSUER, AUDIENCE)
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)

        async def whoami(request: web.Request) -> web.Response:
            principal = principal_for(request)
            return web.json_response(
                {
                    "principalId": principal.principal_id,
                    "kind": principal.kind,
                    "grants": {
                        scope: sorted(capabilities)
                        for scope, capabilities in principal.grants.items()
                    },
                }
            )

        app.router.add_get("/api/history/whoami", whoami)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            unauthenticated = await client.get("/api/history/whoami")
            assert unauthenticated.status == 401

            response = await client.get(
                "/api/history/whoami",
                headers={"Authorization": f"Bearer {_sign(key)}"},
            )
            assert response.status == 200
            assert await response.json() == {
                "principalId": PRINCIPAL_ID,
                "kind": "human",
                "grants": {SCOPE: ["history:read", "jobs:read"]},
            }
        finally:
            await client.close()
            await service.close()

    run_scenario(scenario())


@pytest.mark.parametrize("failure", ["disconnected", "migration"])
def test_helper_outage_preserves_human_authentication_and_federated_error_contract(
    tmp_path: Path, failure: str
) -> None:
    from dinkster_library import LibraryClient
    from dinkster_server.auth import DelegationStore, PrincipalPermissionStore, install_auth
    from dinkster_server.federated_assets_v1 import decode_error_response

    async def scenario() -> None:
        key = OKPKey.generate_key("Ed25519")
        service = _JwksService(key)
        url = await service.start()
        grants = {SCOPE: frozenset({"history:read", "assets:read"})}
        human = Principal(PRINCIPAL_ID, grants)
        authenticator = CompositeAuthenticator(
            StaticBearerAuthenticator(
                {
                    "human": human,
                    "agent": Principal("agent", grants, kind="agent"),
                }
            ),
            TokenAuthenticator(url, ISSUER, AUDIENCE),
        )
        try:
            with LibraryClient(tmp_path / "library") as helper:
                permissions = PrincipalPermissionStore(helper)
                delegated = DelegationStore(helper).mint(human, SCOPE, "Agent", None)
                app = web.Application()
                install_auth(
                    app,
                    authenticator,
                    permission_store=permissions,
                    federated_asset_paths=frozenset({"/api/assets/probe"}),
                )

                async def probe(request: web.Request) -> web.Response:
                    principal = principal_for(request)
                    return web.json_response({"principalId": principal.principal_id})

                app.router.add_get("/api/history/probe", probe)
                app.router.add_get("/api/assets/probe", probe)
                signed = _sign(key, _claims(grants={SCOPE: sorted(grants[SCOPE])}))
                async with TestClient(TestServer(app)) as client:
                    for token in (signed, "human", "agent", delegated["token"]):
                        response = await client.get(
                            "/api/assets/probe", headers={"Authorization": f"Bearer {token}"}
                        )
                        assert response.status == 200
                    if failure == "disconnected":
                        helper.close()
                    else:
                        (tmp_path / "library" / "principals.sqlite").write_bytes(b"legacy")
                    for token in (signed, "human"):
                        for path in ("/api/history/probe", "/api/assets/probe"):
                            response = await client.get(
                                path, headers={"Authorization": f"Bearer {token}"}
                            )
                            assert response.status == 200
                            assert await response.json() == {"principalId": PRINCIPAL_ID}
                    response = await client.get(
                        "/api/assets/probe", headers={"Authorization": "Bearer invalid"}
                    )
                    assert response.status == 401
                    assert (
                        decode_error_response(await response.json(), status=401).error.code
                        == "authentication-required"
                    )
                    for token in ("agent", delegated["token"]):
                        response = await client.get(
                            "/api/assets/probe", headers={"Authorization": f"Bearer {token}"}
                        )
                        assert response.status == 503
                        assert (
                            decode_error_response(await response.json(), status=503).error.code
                            == "service-unavailable"
                        )
                        response = await client.get(
                            "/api/history/probe", headers={"Authorization": f"Bearer {token}"}
                        )
                        assert response.status == 503
                        assert await response.json() == {"error": "authorization-store-unavailable"}
        finally:
            await service.close()

    run_scenario(scenario())


def test_static_principal_management_survives_composition() -> None:
    async def scenario() -> None:
        static = StaticBearerAuthenticator(
            {
                "operator": Principal(
                    "operator",
                    {"local": frozenset({"principals:manage"})},
                )
            }
        )

        class RejectingAuthenticator:
            async def authenticate(self, token: str) -> Principal | None:
                return None

        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=CompositeAuthenticator(static, RejectingAuthenticator()),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get(
                "/api/principals",
                headers={"Authorization": "Bearer operator"},
            )
            assert response.status == 200
            assert [entry["principalId"] for entry in await response.json()] == ["operator"]
        finally:
            await client.close()

    run_scenario(scenario())
