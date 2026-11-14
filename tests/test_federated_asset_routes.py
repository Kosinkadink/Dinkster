from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets.acquisition_plan import AcquisitionSource
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import (
    AdvisoryAlias,
    Artifact,
    LocalMaterialization,
    LogicalModel,
    ModelVariant,
    MountMaterialization,
    ProviderMirror,
    ResolutionStore,
    SourceIdentity,
    VariantArtifact,
)
from dinkster_server import federated_assets_v1 as dto
from dinkster_server.app import create_app
from dinkster_server.auth import Principal
from test_server import SCHEMAS, make_engine

from tests.async_scenarios import run_scenario

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
DIGEST_C = "blake3:" + "c" * 64
KIND = "model/diffusion"
CATALOG_PATH = "/federated/assets/catalog-v1"
CANDIDATES_PATH = "/federated/assets/candidates-v1"


class _Authenticator:
    def __init__(self, principals: Mapping[str, Principal]) -> None:
        self.principals = dict(principals)

    async def authenticate(self, token: str) -> Principal | None:
        return self.principals.get(token)


def _context(*, accept: list[str] | None = None) -> dict[str, object]:
    return {
        "assetKind": KIND,
        "schema": {"nodeType": "dinkster.load", "inputId": "model"},
        "accept": accept or [],
    }


def _seed(store: ResolutionStore) -> tuple[AcquisitionSource, ...]:
    store.upsert_logical_model(LogicalModel("zeta/model", "zeta", KIND, 1))
    store.upsert_logical_model(LogicalModel("alpha/model", "alpha", KIND, 1))
    for logical_id, variant_id, digest in (
        ("zeta/model", "fp8", DIGEST_A),
        ("alpha/model", "bf16", DIGEST_B),
    ):
        store.upsert_variant(
            ModelVariant(
                logical_id,
                variant_id,
                "float8-e4m3fn" if variant_id == "fp8" else "bfloat16",
                "fp8" if variant_id == "fp8" else "none",
                "safetensors",
                "diffusion",
                updated_at=2,
            )
        )
        store.upsert_artifact(Artifact(digest))
        store.link_variant_artifact(VariantArtifact(logical_id, variant_id, digest))
    store.add_alias(AdvisoryAlias("logical", "Featured Model", "alpha/model"))
    store.upsert_local_materialization(
        LocalMaterialization(
            "workspace-a",
            AssetRef(
                DIGEST_A,
                "renamed-local.bin",
                11,
                "application/octet-stream",
                "models/renamed-local.bin",
            ),
            3,
            4,
        )
    )
    store.upsert_local_materialization(
        LocalMaterialization(
            "unlisted",
            AssetRef(
                DIGEST_A,
                "private-local.bin",
                13,
                "application/octet-stream",
                "models/private-local.bin",
            ),
            3,
            4,
        )
    )
    store.upsert_mirror(
        ProviderMirror(SourceIdentity("provider-b", "mirror-z"), DIGEST_B, "available")
    )
    store.upsert_mirror(
        ProviderMirror(SourceIdentity("provider-a", "mirror-a"), DIGEST_B, "available")
    )
    store.upsert_mirror(
        ProviderMirror(SourceIdentity("provider-c", "metadata-only"), DIGEST_B, "available")
    )
    store.upsert_mirror(
        ProviderMirror(SourceIdentity("provider-x", "unauthorized"), DIGEST_B, "available")
    )
    return (
        AcquisitionSource(
            SourceIdentity("provider-b", "mirror-z"),
            DIGEST_B,
            KIND,
            "available",
            "https://provider-b.example/model",
            22,
            "application/safetensors",
            license_requirement_id="license-b",
        ),
        AcquisitionSource(
            SourceIdentity("provider-a", "mirror-a"),
            DIGEST_B,
            KIND,
            "available",
            "https://provider-a.example/model",
            22,
            "application/safetensors",
            credential_requirement_id="credential-a",
        ),
        AcquisitionSource(
            SourceIdentity("provider-x", "unauthorized"),
            DIGEST_B,
            KIND,
            "available",
            "https://provider-x.example/model",
            22,
            "application/safetensors",
        ),
        AcquisitionSource(
            SourceIdentity("provider-d", "trusted-no-mirror"),
            DIGEST_B,
            KIND,
            "available",
            "https://provider-d.example/model",
            22,
            "application/safetensors",
        ),
    )


async def _client(
    tmp_path: Path,
    *,
    now: list[float] | None = None,
) -> tuple[TestClient, ResolutionStore]:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    sources = _seed(store)
    authenticator = _Authenticator(
        {
            "reader": Principal(
                "reader",
                {"workspace-a": frozenset({"assets:read"})},
            ),
            "other": Principal(
                "other",
                {"workspace-a": frozenset({"assets:read"})},
            ),
            "writer": Principal(
                "writer",
                {"workspace-a": frozenset({"assets:write"})},
            ),
            "unlisted": Principal(
                "unlisted",
                {"unlisted": frozenset({"assets:read"})},
            ),
            "multi": Principal(
                "multi",
                {
                    "workspace-a": frozenset({"assets:read"}),
                    "workspace-b": frozenset({"assets:read"}),
                },
            ),
        }
    )
    clock = (lambda: now[0]) if now is not None else None
    app = create_app(
        make_engine,
        SCHEMAS,
        authenticator=authenticator,
        federated_asset_paths={"catalog": CATALOG_PATH, "candidates": CANDIDATES_PATH},
        federated_asset_store=store,
        federated_asset_sources=sources,
        federated_asset_provider_policy={
            "workspace-a": frozenset({"provider-a", "provider-b", "provider-c", "provider-d"}),
            "workspace-b": frozenset({"provider-a"}),
        },
        _federated_asset_cursor_key=b"deterministic-test-cursor-key-32b",
        _federated_asset_clock=clock,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, store


def _headers(token: str = "reader") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _mount_row(
    scope: str,
    digest: str,
    *,
    mount_id: str,
    priority: int,
    kind: str,
    name: str,
    size: int = 17,
    media_type: str = "application/octet-stream",
    virtual_path: str | None = None,
) -> MountMaterialization:
    return MountMaterialization(
        scope,
        mount_id,
        priority,
        kind,
        AssetRef(
            digest,
            name,
            size,
            media_type,
            virtual_path or f"mounts/{mount_id}/{name}",
        ),
    )


def _replace_mounts(store: ResolutionStore, rows: tuple[MountMaterialization, ...]) -> None:
    mirrored = tuple(
        MountMaterialization(
            "workspace-b",
            row.mount_id,
            row.priority,
            row.asset_kind,
            row.ref,
        )
        for row in rows
    )
    store.replace_mount_snapshot(("workspace-a", "workspace-b"), (*rows, *mirrored))


def test_federated_routes_refuse_scopes_absent_from_provider_policy(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, store = await _client(tmp_path)
        try:
            for path, request in (
                (CATALOG_PATH, {"contractVersion": 1, "scope": "unlisted"}),
                (
                    CANDIDATES_PATH,
                    {
                        "contractVersion": 1,
                        "scope": "unlisted",
                        "context": _context(),
                    },
                ),
            ):
                response = await client.post(path, headers=_headers("unlisted"), json=request)
                assert response.status == 403
                assert await response.json() == {
                    "contractVersion": 1,
                    "error": {
                        "code": "forbidden",
                        "reason": "scope not authorized by federated asset policy",
                    },
                }
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())


def test_auth_off_federated_routes_serve_only_policy_scopes(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ResolutionStore(tmp_path / "resolution.sqlite")
        _seed(store)
        app = create_app(
            make_engine,
            SCHEMAS,
            federated_asset_paths={"catalog": CATALOG_PATH, "candidates": CANDIDATES_PATH},
            federated_asset_store=store,
            federated_asset_provider_policy={"local": frozenset()},
            _federated_asset_cursor_key=b"deterministic-test-cursor-key-32b",
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            allowed = await client.post(CATALOG_PATH, json={"contractVersion": 1})
            assert allowed.status == 200
            forbidden = await client.post(
                CATALOG_PATH,
                json={"contractVersion": 1, "scope": "unlisted"},
            )
            assert forbidden.status == 403
            assert (await forbidden.json())["error"] == {
                "code": "forbidden",
                "reason": "scope not authorized by federated asset policy",
            }
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())


def test_catalog_snapshot_folds_authorized_providers_filters_and_orders(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client, store = await _client(tmp_path)
        try:
            response = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "context": _context(),
                },
            )
            assert response.status == 200
            items = (await response.json())["items"]
            assert [(item["logicalId"], item["variantId"], item["digest"]) for item in items] == [
                ("alpha/model", "bf16", DIGEST_B),
                ("zeta/model", "fp8", DIGEST_A),
            ]
            remote, local = items
            assert remote["availability"] == {
                "status": "downloadable",
                "reason": "authorized provider mirror available",
            }
            assert [row["source"] for row in remote["providerSources"]] == [
                {"providerId": "provider-a", "sourceId": "mirror-a"},
                {"providerId": "provider-b", "sourceId": "mirror-z"},
                {"providerId": "provider-d", "sourceId": "trusted-no-mirror"},
            ]
            assert remote["providerSources"][0]["requires"] == {"credential": "credential-a"}
            assert remote["providerSources"][1]["requires"] == {"license": "license-b"}
            assert remote["providerSources"][2] == {
                "source": {"providerId": "provider-d", "sourceId": "trusted-no-mirror"},
                "status": "unavailable",
                "reason": "provider mirror unavailable",
                "requires": {},
            }
            assert local["assetRef"] == {
                "digest": DIGEST_A,
                "name": "renamed-local.bin",
                "size": 11,
                "mediaType": "application/octet-stream",
                "virtualPath": "models/renamed-local.bin",
            }

            filtered = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "query": "featured",
                    "availability": ["downloadable"],
                    "context": _context(),
                },
            )
            assert [item["digest"] for item in (await filtered.json())["items"]] == [DIGEST_B]
            filename = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "query": "renamed-local",
                },
            )
            assert (await filename.json())["items"] == []
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())


def test_mount_candidates_page_past_one_hundred_with_global_strict_order(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client, store = await _client(tmp_path)
        rows = tuple(
            _mount_row(
                "workspace-a",
                "blake3:" + f"{index + 1000:064x}",
                mount_id="bulk",
                priority=0,
                kind="",
                name=f"asset-{index:03d}.bin",
            )
            for index in range(103)
        )
        _replace_mounts(store, rows)
        request: dict[str, object] = {
            "contractVersion": 1,
            "scope": "workspace-a",
            "limit": 100,
        }
        try:
            first_body = await (
                await client.post(CATALOG_PATH, headers=_headers(), json=request)
            ).json()
            first = dto.decode_catalog_response(first_body)
            assert len(first.items) == 100
            assert first.next_cursor is not None
            second_body = await (
                await client.post(
                    CATALOG_PATH,
                    headers=_headers(),
                    json={**request, "cursor": first.next_cursor},
                )
            ).json()
            second = dto.decode_catalog_response(second_body)
            assert len(second.items) == 5
            assert second.next_cursor is None
            keys = [
                (item.logical_id, item.variant_id, item.digest)
                for item in (*first.items, *second.items)
            ]
            assert keys == sorted(set(keys))
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())


def test_candidates_selection_expected_context_and_hints_are_authoritative_only_where_frozen(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client, store = await _client(tmp_path)
        try:
            base = {
                "contractVersion": 1,
                "scope": "workspace-a",
                "context": _context(),
                "selection": {
                    "logicalId": "zeta/model",
                    "variantId": "fp8",
                    "digest": DIGEST_A,
                },
            }
            response = await client.post(CANDIDATES_PATH, headers=_headers(), json=base)
            body = await response.json()
            assert response.status == 200
            assert body["status"] == "resolved"
            assert body["selectedCandidate"] == {
                "logicalId": "zeta/model",
                "variantId": "fp8",
                "digest": DIGEST_A,
                "reason": "explicit-selection",
            }

            implicit = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "context": _context(),
                },
            )
            assert (await implicit.json())["status"] == "ambiguous"

            mirror_only = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    **base,
                    "selection": {
                        **base["selection"],
                        "source": {
                            "providerId": "provider-c",
                            "sourceId": "metadata-only",
                        },
                    },
                },
            )
            assert mirror_only.status == 409
            assert (await mirror_only.json())["error"]["code"] == "source-unavailable"

            unauthorized_source = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    **base,
                    "selection": {
                        **base["selection"],
                        "source": {
                            "providerId": "provider-x",
                            "sourceId": "unauthorized",
                        },
                    },
                },
            )
            assert unauthorized_source.status == 403
            assert (await unauthorized_source.json())["error"]["code"] == "forbidden"

            unauthorized_unknown_candidate = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    **base,
                    "selection": {
                        "logicalId": "unknown/model",
                        "variantId": "unknown",
                        "digest": DIGEST_C,
                        "source": {
                            "providerId": "provider-x",
                            "sourceId": "unauthorized",
                        },
                    },
                },
            )
            assert unauthorized_unknown_candidate.status == 403
            assert (await unauthorized_unknown_candidate.json())["error"]["code"] == "forbidden"

            hinted = dict(base)
            hinted["hints"] = {
                "source": "provider-x",
                "reference": "wrong",
                "loaderPath": "alpha/model",
                "modelType": "other",
                "displayName": "Featured Model",
            }
            assert (
                await (await client.post(CANDIDATES_PATH, headers=_headers(), json=hinted)).json()
            )["selectedCandidate"] == body["selectedCandidate"]

            mismatch = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "context": _context(),
                    "expected": {"digest": DIGEST_A, "size": 12},
                },
            )
            assert mismatch.status == 422
            assert await mismatch.json() == {
                "contractVersion": 1,
                "error": {
                    "code": "integrity-mismatch",
                    "expectedSize": 12,
                    "observedSize": 11,
                },
            }

            store.upsert_logical_model(LogicalModel("other/model", "other", "model/lora", 1))
            store.upsert_variant(
                ModelVariant(
                    "other/model",
                    "default",
                    "float16",
                    "none",
                    "safetensors",
                    "lora",
                    updated_at=2,
                )
            )
            store.upsert_artifact(Artifact(DIGEST_C))
            store.link_variant_artifact(VariantArtifact("other/model", "default", DIGEST_C))
            wrong_kind = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "context": _context(),
                    "expected": {"digest": DIGEST_C},
                },
            )
            assert wrong_kind.status == 422
            assert (await wrong_kind.json())["error"] == {
                "code": "wrong-kind",
                "expectedKind": KIND,
                "actualKind": "model/lora",
            }

            store.link_variant_artifact(VariantArtifact("other/model", "default", DIGEST_A))
            shared_digest = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    "contractVersion": 1,
                    "scope": "workspace-a",
                    "context": _context(),
                    "expected": {"digest": DIGEST_A},
                },
            )
            assert shared_digest.status == 200
            assert (await shared_digest.json())["selectedCandidate"]["digest"] == DIGEST_A

            incompatible = await client.post(
                CANDIDATES_PATH,
                headers=_headers(),
                json={
                    **base,
                    "context": _context(accept=["image/png"]),
                },
            )
            incompatible_body = await incompatible.json()
            assert incompatible_body["status"] == "incompatible"
            assert incompatible_body["items"][1]["availability"]["status"] == "local"
            assert incompatible_body["items"][1]["compatibility"]["status"] == "incompatible"
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())


def test_signed_cursor_binds_principal_scope_query_generation_and_expiry(tmp_path: Path) -> None:
    async def scenario() -> None:
        now = [1000.0]
        client, store = await _client(tmp_path, now=now)
        request = {
            "contractVersion": 1,
            "context": _context(accept=["application/safetensors", "application/safetensors"]),
            "limit": 1,
        }
        try:
            first = await (await client.post(CATALOG_PATH, headers=_headers(), json=request)).json()
            cursor = first["nextCursor"]
            assert isinstance(cursor, str) and len(cursor) <= 4096
            second = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={
                    **request,
                    "scope": "workspace-a",
                    "context": _context(accept=["application/safetensors"]),
                    "query": "",
                    "availability": ["unavailable", "downloadable", "local"],
                    "compatibility": ["incompatible", "compatible", "unknown"],
                    "cursor": cursor,
                },
            )
            assert [item["digest"] for item in (await second.json())["items"]] == [DIGEST_A]

            for changed, token in (
                ({**request, "query": "alpha", "cursor": cursor}, "reader"),
                ({**request, "cursor": cursor}, "other"),
            ):
                response = await client.post(CATALOG_PATH, headers=_headers(token), json=changed)
                assert response.status == 400
                assert (await response.json())["error"] == {
                    "code": "cursor-invalid",
                    "reason": "query-mismatch",
                }

            malformed = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={**request, "cursor": cursor[:-1] + "x"},
            )
            assert (await malformed.json())["error"]["reason"] == "malformed"

            store.upsert_local_materialization(
                LocalMaterialization(
                    "workspace-a",
                    AssetRef(DIGEST_B, "new.bin", 22, "application/safetensors", "models/new.bin"),
                )
            )
            stale = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={**request, "cursor": cursor},
            )
            assert (await stale.json())["error"]["reason"] == "stale-snapshot"

            fresh = await (await client.post(CATALOG_PATH, headers=_headers(), json=request)).json()
            now[0] += 301
            expired = await client.post(
                CATALOG_PATH,
                headers=_headers(),
                json={**request, "cursor": fresh["nextCursor"]},
            )
            assert (await expired.json())["error"]["reason"] == "expired"
        finally:
            await client.close()
            store.close()

    run_scenario(scenario())
