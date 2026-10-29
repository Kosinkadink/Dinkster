from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import P2P_PIECE_LENGTH, P2P_PROTOCOL, P2PDescriptorV1, SeedGrantV1
from dinkster_server import (
    P2PAction,
    P2PActivityStatus,
    P2PNetworkStatus,
    P2PProviderConflict,
    P2PProviderUnavailable,
    P2PSeedAuthorization,
    P2PTotals,
    P2PTransfer,
    P2PTransferNotFound,
    create_app,
)
from dinkster_server.p2p_activity import add_p2p_action_routes
from test_server import SCHEMAS, make_engine

DIGEST = f"blake3:{'a' * 64}"
GRANT_ID = "b" * 64
SEED_GRANT = SeedGrantV1(
    version=1,
    grant_id=GRANT_ID,
    digest=DIGEST,
    source_type="declarative-resolver",
    source_id="resolver.example/models",
    source_revision="sha256:" + "c" * 64,
    license="Apache-2.0",
    descriptor=P2PDescriptorV1(
        protocol=P2P_PROTOCOL,
        info_hash="d" * 64,
        file_root="e" * 64,
        piece_length=P2P_PIECE_LENGTH,
    ),
    expires_at=4_000_000_000.0,
    evidence_type="public-acquisition-receipt",
    evidence_id="f" * 32,
)


def activity_status() -> P2PActivityStatus:
    return P2PActivityStatus(
        network=P2PNetworkStatus(
            system="metered",
            override="unmetered",
            effective="unmetered",
            paused=False,
        ),
        totals=P2PTotals(downloaded_bytes=1_500, uploaded_bytes=750),
        transfers=(
            P2PTransfer(
                digest=DIGEST,
                state="downloading",
                size_bytes=4_000,
                peers=3,
                download_rate_bytes_per_second=200,
                upload_rate_bytes_per_second=50,
                downloaded_bytes=1_500,
                uploaded_bytes=750,
                partial_bytes=1_500,
                seed_authorizations=(
                    P2PSeedAuthorization(
                        grant_id=GRANT_ID,
                        state="active",
                        grant=SEED_GRANT,
                    ),
                ),
                remaining_seed_ratio=0.5,
                remaining_seed_time_seconds=3_600,
            ),
        ),
    )


class Provider:
    def __init__(self) -> None:
        self.actions: list[tuple[str, P2PAction]] = []
        self.action_error: Exception | None = None

    async def perform_action(self, digest: str, action: P2PAction) -> None:
        self.actions.append((digest, action))
        if self.action_error is not None:
            raise self.action_error


def app_with_provider(provider: Provider | None, *, p2p_granted: bool = True):
    app = create_app(make_engine, SCHEMAS)
    add_p2p_action_routes(app, provider, p2p_granted=p2p_granted)
    return app


def test_activity_status_uses_the_exact_camel_case_wire_contract() -> None:
    assert activity_status().to_wire() == {
        "network": {
            "system": "metered",
            "override": "unmetered",
            "effective": "unmetered",
            "paused": False,
        },
        "totals": {"downloadedBytes": 1_500, "uploadedBytes": 750},
        "transfers": [
            {
                "digest": DIGEST,
                "state": "downloading",
                "sizeBytes": 4_000,
                "peers": 3,
                "downloadRateBytesPerSecond": 200,
                "uploadRateBytesPerSecond": 50,
                "downloadedBytes": 1_500,
                "uploadedBytes": 750,
                "partialBytes": 1_500,
                "seedAuthorizations": [
                    {
                        "grantId": GRANT_ID,
                        "state": "active",
                        "grant": SEED_GRANT.to_wire(),
                    }
                ],
                "remainingSeedRatio": 0.5,
                "remainingSeedTimeSeconds": 3_600,
            }
        ],
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: P2PSeedAuthorization(grant_id="bad", state="active", grant=SEED_GRANT),
        lambda: P2PSeedAuthorization(grant_id=GRANT_ID, state="active", grant=None),
        lambda: P2PSeedAuthorization(grant_id=GRANT_ID, state="revoked", grant=SEED_GRANT),
        lambda: P2PTotals(downloaded_bytes=-1, uploaded_bytes=0),
        lambda: P2PTotals(downloaded_bytes=1.5, uploaded_bytes=0),  # type: ignore[arg-type]
        lambda: P2PNetworkStatus(
            system="invalid",  # type: ignore[arg-type]
            override="auto",
            effective="unknown",
            paused=False,
        ),
        lambda: P2PTransfer(
            digest=f"blake3:{'A' * 64}",
            state="queued",
            size_bytes=0,
            peers=0,
            download_rate_bytes_per_second=0,
            upload_rate_bytes_per_second=0,
            downloaded_bytes=0,
            uploaded_bytes=0,
            partial_bytes=0,
            seed_authorizations=(),
            remaining_seed_ratio=None,
            remaining_seed_time_seconds=None,
        ),
        lambda: P2PActivityStatus(
            network=activity_status().network,
            totals=activity_status().totals,
            transfers=(activity_status().transfers[0], activity_status().transfers[0]),
        ),
    ),
)
def test_activity_values_reject_invalid_contract_data(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_network_status_rejects_an_effective_cost_that_ignores_the_override() -> None:
    with pytest.raises(ValueError, match="effective network cost"):
        P2PNetworkStatus(
            system="metered",
            override="unmetered",
            effective="metered",
            paused=False,
        )


def test_activity_status_freezes_mutable_transfer_inputs() -> None:
    transfers = [activity_status().transfers[0]]
    status = P2PActivityStatus(
        network=activity_status().network,
        totals=activity_status().totals,
        transfers=transfers,  # type: ignore[arg-type]
    )
    transfers.clear()
    assert status.transfers == activity_status().transfers


def test_absent_provider_rejects_actions() -> None:
    async def scenario() -> None:
        client = TestClient(TestServer(app_with_provider(None)))
        await client.start_server()
        try:
            response = await client.post(f"/api/p2p/transfers/{DIGEST}/pause")
            assert response.status == 503
            assert (await response.json())["error"]["code"] == "p2p-unavailable"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_actions_require_the_host_p2p_grant() -> None:
    async def scenario() -> None:
        provider = Provider()
        client = TestClient(TestServer(app_with_provider(provider, p2p_granted=False)))
        await client.start_server()
        try:
            response = await client.post(f"/api/p2p/transfers/{DIGEST}/pause")
            assert response.status == 403
            assert await response.json() == {
                "error": {
                    "code": "p2p-permission-denied",
                    "message": "P2P actions are disabled by this host",
                }
            }
            assert provider.actions == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_all_transfer_actions_are_wired_to_the_provider() -> None:
    async def scenario() -> None:
        provider = Provider()
        client = TestClient(TestServer(app_with_provider(provider)))
        await client.start_server()
        try:
            actions: tuple[P2PAction, ...] = (
                "pause",
                "resume",
                "stop",
                "remove-partial",
                "reset-budget",
                "continuous-seed",
            )
            for action in actions:
                response = await client.post(f"/api/p2p/transfers/{DIGEST}/{action}")
                assert response.status == 204
                assert await response.read() == b""
            assert provider.actions == [(DIGEST, action) for action in actions]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_invalid_transfer_inputs_never_reach_the_provider() -> None:
    async def scenario() -> None:
        provider = Provider()
        client = TestClient(TestServer(app_with_provider(provider)))
        await client.start_server()
        try:
            invalid_digest = await client.post(f"/api/p2p/transfers/blake3:{'A' * 64}/pause")
            assert invalid_digest.status == 400
            assert (await invalid_digest.json())["error"]["code"] == "invalid-p2p-digest"

            invalid_action = await client.post(f"/api/p2p/transfers/{DIGEST}/delete")
            assert invalid_action.status == 400
            assert (await invalid_action.json())["error"]["code"] == "invalid-p2p-action"
            assert provider.actions == []
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "status", "code"),
    (
        (P2PProviderConflict("secret path /private/model"), 409, "p2p-conflict"),
        (P2PTransferNotFound("secret path /private/model"), 404, "p2p-transfer-not-found"),
        (P2PProviderUnavailable("secret token"), 503, "p2p-unavailable"),
        (ValueError("secret token"), 500, "p2p-internal-error"),
    ),
)
def test_action_errors_are_structured_without_provider_details(
    error: Exception, status: int, code: str
) -> None:
    async def scenario() -> None:
        provider = Provider()
        provider.action_error = error
        client = TestClient(TestServer(app_with_provider(provider)))
        await client.start_server()
        try:
            response = await client.post(f"/api/p2p/transfers/{DIGEST}/pause")
            assert response.status == status
            body = await response.json()
            assert body["error"]["code"] == code
            assert "secret" not in str(body)
            assert "/private" not in str(body)
        finally:
            await client.close()

    asyncio.run(scenario())
