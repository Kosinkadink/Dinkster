import asyncio
import base64
import json
import time
from types import MappingProxyType
from uuid import UUID

import dinkster_server.auth as auth_module
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
from dinkster_token_verifier import VerifiedPrincipal
from joserfc import jwt
from joserfc.jwk import OctKey, OKPKey
from test_server import SCHEMAS, make_engine

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


def test_token_authenticator_maps_verified_principal_without_deriving_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grants = MappingProxyType({SCOPE: frozenset({"jobs:read", "settings:read"})})
    verified = VerifiedPrincipal(
        principal_id=f"k_{UUID(int=3)}",
        grants=grants,
        kind="agent",
        organization_id=None,
        account_id=UUID(int=4),
        api_key_id=UUID(int=5),
        api_key_scopes=frozenset({"jobs:read", "settings:read"}),
        jti=UUID(int=6),
        expires_at=int(time.time()) + 600,
    )
    instances: list[object] = []
    accepted = f"header.{_encode_segment({'exp': int(time.time()) + 600})}.signature"

    class StubVerifier:
        def __init__(self, **configuration: str) -> None:
            self.configuration = configuration
            self.tokens: list[str] = []
            instances.append(self)

        async def verify(self, token: str) -> VerifiedPrincipal | None:
            self.tokens.append(token)
            return verified if token == accepted else None

    monkeypatch.setattr(auth_module, "TokenVerifier", StubVerifier)
    authenticator = TokenAuthenticator(
        "https://identity.example/.well-known/jwks.json",
        ISSUER,
        AUDIENCE,
    )

    async def scenario() -> None:
        principal = await authenticator.authenticate(accepted)
        assert principal is not None
        assert principal.principal_id == verified.principal_id
        assert principal.grants == verified.grants
        assert principal.kind == verified.kind
        assert await authenticator.authenticate("rejected") is None
        with pytest.raises(TypeError):
            principal.grants[SCOPE] = frozenset()  # type: ignore[index]

    asyncio.run(scenario())

    assert len(instances) == 1
    verifier = instances[0]
    assert isinstance(verifier, StubVerifier)
    assert verifier.configuration == {
        "jwks_url": "https://identity.example/.well-known/jwks.json",
        "issuer": ISSUER,
        "audience": AUDIENCE,
    }
    assert verifier.tokens == [accepted, "rejected"]


def test_composite_authenticator_uses_declared_order() -> None:
    calls: list[str] = []
    static_principal = Principal("static", {"local": frozenset()})
    token_principal = Principal(PRINCIPAL_ID, {SCOPE: frozenset()})

    class StubAuthenticator:
        def __init__(self, name: str, principal: Principal | None) -> None:
            self.name = name
            self.principal = principal

        async def authenticate(self, token: str) -> Principal | None:
            calls.append(f"{self.name}:{token}")
            return self.principal

    async def scenario() -> None:
        static_first = CompositeAuthenticator(
            StubAuthenticator("static", static_principal),
            StubAuthenticator("token", token_principal),
        )
        assert await static_first.authenticate("shared") is static_principal
        assert calls == ["static:shared"]

        calls.clear()
        fallback = CompositeAuthenticator(
            StubAuthenticator("static", None),
            StubAuthenticator("token", token_principal),
        )
        assert await fallback.authenticate("jwt") is token_principal
        assert calls == ["static:jwt", "token:jwt"]

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_token_authenticator_shares_fresh_and_stale_jwks_cache() -> None:
    async def scenario() -> None:
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
            await asyncio.sleep(1.05)
            for _ in range(3):
                assert await authenticator.authenticate(token) is not None
            assert service.requests == 2
        finally:
            await service.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())
