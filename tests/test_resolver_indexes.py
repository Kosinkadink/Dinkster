"""Declarative resolver-index parsing, subscriptions, and API wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    RESOLVER_INDEX_MAX_ENTRIES,
    RESOLVER_INDEX_MAX_URLS,
    AssetVault,
    LibraryStore,
    ProvenanceStore,
    ResolverIndexError,
    ResolverSubscriptionError,
    ResolverSubscriptionStore,
    digest_bytes,
    parse_resolver_index,
    resolver_index_from_wire,
)
from dinkster_server import ServerLibrary, create_app
from test_server import SCHEMAS, make_engine

from dinkster.guess_api import add_guess_routes
from dinkster.resolver_api import add_resolver_index_routes

MODEL_BYTES = b"resolver-index model bytes" * 32
MODEL_DIGEST = digest_bytes(MODEL_BYTES)
OTHER_BYTES = b"updated resolver-index model bytes" * 32
OTHER_DIGEST = digest_bytes(OTHER_BYTES)


def resolver_entry(
    *,
    digest: str = MODEL_DIGEST,
    name: str = "models/example.safetensors",
    urls: list[str] | None = None,
    regions: dict[str, list[str]] | None = None,
    components: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "digest": digest,
        "name": name,
        "urls": urls or ["https://models.example/default.safetensors"],
        "kind": "model/diffusion",
        "size": len(MODEL_BYTES),
        "license": "apache-2.0",
        "notes": "verified community mirror",
        "regions": regions or {"cn": ["https://cn.example/model.safetensors"]},
        "family": "example",
        "variant": {"format": "safetensors", "options": {"precision": "fp16"}},
    }
    if components is not None:
        entry["components"] = components
    return entry


def resolver_document(*entries: dict[str, object]) -> dict[str, object]:
    return {
        "dinksterResolver": 1,
        "name": "community-models",
        "description": "Community mirror index",
        "homepage": "https://models.example",
        "updated": "2026-09-01T00:00:00Z",
        "entries": list(entries or (resolver_entry(),)),
    }


def write_document(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document), "utf-8")


def test_parse_resolver_index_round_trip_and_http_requirements() -> None:
    index = parse_resolver_index(json.dumps(resolver_document()))
    entry = index.entries[0]
    assert entry.urls_for("cn") == (
        "https://cn.example/model.safetensors",
        "https://models.example/default.safetensors",
    )
    assert entry.license == "apache-2.0"
    assert parse_resolver_index(json.dumps(index.to_wire())) == index

    for urls in (
        [],
        ["http://models.example/model"],
        ["https://user:password@models.example/model"],
    ):
        document = resolver_document()
        document["entries"][0]["urls"] = urls
        with pytest.raises(ResolverIndexError):
            resolver_index_from_wire(document)


@pytest.mark.parametrize("gated", [None, 0, 1, "true", [], {}])
def test_resolver_gated_metadata_requires_boolean(gated: object) -> None:
    entry = resolver_entry()
    entry["gated"] = gated
    with pytest.raises(ResolverIndexError, match="gated must be a boolean"):
        resolver_index_from_wire(resolver_document(entry))


def test_parse_resolver_index_retains_components_and_ignores_unknown_optional_fields() -> None:
    components = [{"kind": "model/diffusion", "architecture": "example-transformer"}]
    document = resolver_document(resolver_entry(components=components))
    document["futureDocumentField"] = {"version": 2}
    document["entries"][0]["futureEntryField"] = True

    parsed = resolver_index_from_wire(document)

    assert parsed.entries[0].component_manifest is not None
    assert parsed.entries[0].component_manifest.to_wire() == components
    assert "futureDocumentField" not in parsed.to_wire()


def test_parse_resolver_index_enforces_url_entry_and_json_limits() -> None:
    with pytest.raises(ResolverIndexError, match="duplicate JSON field"):
        parse_resolver_index('{"dinksterResolver":1,"dinksterResolver":1,"entries":[]}')
    with pytest.raises(ResolverIndexError, match=str(RESOLVER_INDEX_MAX_ENTRIES)):
        resolver_index_from_wire(
            {
                "dinksterResolver": 1,
                "entries": [resolver_entry()] * (RESOLVER_INDEX_MAX_ENTRIES + 1),
            }
        )
    urls = [f"https://models.example/{number}" for number in range(RESOLVER_INDEX_MAX_URLS + 1)]
    with pytest.raises(ResolverIndexError, match="at most"):
        resolver_index_from_wire(resolver_document(resolver_entry(urls=urls)))


def test_local_subscriptions_refresh_suggestions_and_reload(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    write_document(source, resolver_document(resolver_entry(name="folder/Model.safetensors")))
    store = ResolverSubscriptionStore(state, provenance)
    subscription = store.subscribe(str(source))
    assert provenance.sources(MODEL_DIGEST) == ("https://models.example/default.safetensors",)
    assert [suggestion.digest for suggestion in store.suggest("foo\\Model.safetensors")] == [
        MODEL_DIGEST
    ]
    assert store.suggest("model.safetensors") == ()

    write_document(
        source,
        resolver_document(
            resolver_entry(
                digest=OTHER_DIGEST,
                name="replacement.safetensors",
                urls=["https://models.example/replacement"],
            )
        ),
    )
    (refreshed,) = store.refresh(subscription.id)
    assert refreshed["error"] == ""
    assert provenance.sources(MODEL_DIGEST) == ()
    assert provenance.sources(OTHER_DIGEST) == ("https://models.example/replacement",)
    assert ResolverSubscriptionStore(state, provenance).subscriptions()[0].id == subscription.id


def test_persisted_unused_optional_field_keeps_http_entries_and_suggestions(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    write_document(source, resolver_document())
    store = ResolverSubscriptionStore(state, provenance)
    store.subscribe(str(source))

    persisted = json.loads(state.read_text("utf-8"))
    persisted["subscriptions"][0]["retiredOptionalField"] = {"unused": True}
    state.write_text(json.dumps(persisted), "utf-8")

    reloaded = ResolverSubscriptionStore(state, provenance)
    assert provenance.sources(MODEL_DIGEST)
    assert [suggestion.digest for suggestion in reloaded.suggest("example.safetensors")] == [
        MODEL_DIGEST
    ]


def test_subscription_state_rejects_invalid_known_fields(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    write_document(source, resolver_document())
    ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json")).subscribe(
        str(source)
    )
    persisted = json.loads(state.read_text("utf-8"))
    persisted["subscriptions"][0]["checkedAt"] = float("inf")
    state.write_text(json.dumps(persisted), "utf-8")
    with pytest.raises(
        ResolverSubscriptionError, match="invalid resolver subscription JSON number"
    ):
        ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "reload.json"))


def test_subscription_source_validation_is_closed(tmp_path: Path) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json", ProvenanceStore(tmp_path / "provenance.json")
    )
    for source in (
        "",
        " relative.json ",
        "http://models.example/index.json",
        "ftp://models.example/index.json",
        "https://user:pass@models.example/index.json",
    ):
        with pytest.raises(ResolverSubscriptionError):
            store.subscribe(source)


def test_hosted_subscription_revalidates_with_etag(tmp_path: Path) -> None:
    async def scenario() -> None:
        now = [100.0]
        requests: list[str | None] = []

        async def index(request: web.Request) -> web.Response:
            requests.append(request.headers.get("If-None-Match"))
            if request.headers.get("If-None-Match") == '"one"':
                return web.Response(status=304, headers={"ETag": '"one"'})
            return web.json_response(resolver_document(), headers={"ETag": '"one"'})

        app = web.Application()
        app.router.add_get("/index.json", index)
        server = TestServer(app)
        await server.start_server()
        try:
            store = ResolverSubscriptionStore(
                tmp_path / "subscriptions.json",
                ProvenanceStore(tmp_path / "provenance.json"),
                clock=lambda: now[0],
            )
            await asyncio.to_thread(store.subscribe, str(server.make_url("/index.json")))
            now[0] += 6 * 60 * 60
            (refreshed,) = await asyncio.to_thread(store.refresh)
            assert refreshed["etag"] == '"one"'
            assert requests == [None, '"one"']
        finally:
            await server.close()

    asyncio.run(scenario())


def test_official_bootstrap_pins_provider_identity(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def index(_: web.Request) -> web.Response:
            return web.json_response(resolver_document())

        app = web.Application()
        app.router.add_get("/resolver", index)
        server = TestServer(app)
        await server.start_server()
        try:
            store = ResolverSubscriptionStore(
                tmp_path / "subscriptions.json", ProvenanceStore(tmp_path / "provenance.json")
            )
            subscription = await asyncio.to_thread(
                store.bootstrap_official, str(server.make_url("/resolver")), "community-models"
            )
            assert subscription is not None
            assert subscription.index.name == "community-models"
            with pytest.raises(ResolverSubscriptionError, match="configuration differs"):
                store.bootstrap_official(str(server.make_url("/resolver")), "different")
        finally:
            await server.close()

    asyncio.run(scenario())


def test_resolver_management_and_filename_guess_api(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry(name="Model.safetensors")))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    library = ServerLibrary(
        AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite"), provenance
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, library=library)
        add_resolver_index_routes(app, store)
        add_guess_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/assets/resolver-indexes", json={"source": str(source)}
            )
            assert response.status == 201
            subscription = await response.json()
            response = await client.get("/api/assets/resolver-indexes")
            assert (await response.json())["subscriptions"][0]["entryCount"] == 1
            response = await client.post("/api/assets/guess", json={"names": ["Model.safetensors"]})
            candidate = (await response.json())["matches"][0]["candidates"][0]
            assert candidate["digest"] == MODEL_DIGEST
            response = await client.delete(f"/api/assets/resolver-indexes/{subscription['id']}")
            assert response.status == 200
            assert provenance.sources(MODEL_DIGEST) == ()
        finally:
            await client.close()

    asyncio.run(scenario())
