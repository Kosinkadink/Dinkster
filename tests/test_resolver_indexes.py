"""Declarative resolver-index parsing, subscriptions, and API wiring."""

from __future__ import annotations

import asyncio
import http.client
import json
import re
import socket
import ssl
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    RESOLVER_INDEX_MAX_BYTES,
    RESOLVER_INDEX_MAX_ENTRIES,
    RESOLVER_INDEX_MAX_URLS,
    RESOLVER_REVALIDATE_SECONDS,
    RESOLVER_SUBSCRIPTIONS_MAX_BYTES,
    AssetNeed,
    AssetVault,
    LibraryStore,
    ProvenanceRecord,
    ProvenanceStore,
    ResolverIndexError,
    ResolverSubscription,
    ResolverSubscriptionError,
    ResolverSubscriptionStore,
    acquire_need,
    digest_bytes,
    parse_resolver_index,
    resolver_index_from_wire,
)
from dinkster_assets import resolver_subscription as resolver_transport
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
    kind: str = "model/diffusion",
    size: int | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "digest": digest,
        "name": name,
        "urls": urls or ["https://models.example/default.safetensors"],
        "kind": kind,
        "size": len(MODEL_BYTES) if size is None else size,
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


@pytest.fixture
def valid_document() -> dict[str, object]:
    return resolver_document(resolver_entry())


def test_parse_resolver_index_round_trip_and_region_priority(
    valid_document: dict[str, object],
) -> None:
    index = parse_resolver_index(json.dumps(valid_document))
    assert index.name == "community-models"
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.digest == MODEL_DIGEST
    assert entry.kind == "model/diffusion"
    assert entry.urls_for("cn") == (
        "https://cn.example/model.safetensors",
        "https://models.example/default.safetensors",
    )
    assert entry.urls_for("us") == ("https://models.example/default.safetensors",)
    with pytest.raises(TypeError):
        entry.regions["us"] = ("https://us.example/model.safetensors",)  # type: ignore[index]
    with pytest.raises(TypeError):
        entry.variant["format"] = "pickle"  # type: ignore[index]
    assert parse_resolver_index(json.dumps(index.to_wire())) == index


@pytest.mark.parametrize("gated", [None, 0, 1, "true", [], {}])
def test_resolver_gated_metadata_requires_boolean(gated: object) -> None:
    entry = resolver_entry()
    entry["gated"] = gated
    with pytest.raises(ResolverIndexError, match="gated must be a boolean"):
        resolver_index_from_wire(resolver_document(entry))


def test_parse_resolver_index_uses_shared_component_manifest() -> None:
    document = resolver_document(
        resolver_entry(
            name="combined.safetensors",
            kind="model/checkpoint",
            components=[
                {
                    "kind": "model/diffusion",
                    "architecture": "example-transformer",
                    "dtype": "float16",
                    "metadata": {"role": "denoiser"},
                },
                {"kind": "model/vae"},
            ],
        )
    )
    index = parse_resolver_index(json.dumps(document))
    manifest = index.entries[0].component_manifest
    assert manifest is not None
    assert [component.kind for component in manifest.components] == [
        "model/diffusion",
        "model/vae",
    ]
    assert index.to_wire()["entries"][0]["components"] == document["entries"][0]["components"]  # type: ignore[index]
    assert parse_resolver_index(json.dumps(index.to_wire())) == index

    invalid = resolver_document(resolver_entry(components=[{"kind": "diffusion"}]))
    with pytest.raises(ResolverIndexError, match="asset kind"):
        parse_resolver_index(json.dumps(invalid))


def test_parse_resolver_index_ignores_unknown_optional_fields() -> None:
    baseline = resolver_document(
        resolver_entry(
            components=[
                {
                    "kind": "model/vae",
                    "architecture": "example-vae",
                }
            ]
        )
    )
    extended = json.loads(json.dumps(baseline))
    extended["futureTopLevel"] = {"added": True}
    extended_entry = extended["entries"][0]
    extended_entry["futureEntry"] = [1, 2, 3]
    extended_entry["components"][0]["futureComponent"] = "optional"

    expected = parse_resolver_index(json.dumps(baseline))
    parsed = parse_resolver_index(json.dumps(extended))

    assert parsed == expected
    assert parsed.to_wire() == expected.to_wire()


def test_resolver_index_rejects_non_json_variant_values() -> None:
    entry = resolver_entry()
    entry["variant"] = {"invalid": object()}
    with pytest.raises(ResolverIndexError, match="only JSON values"):
        resolver_index_from_wire(resolver_document(entry))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document.update(dinksterResolver=2),
            "unsupported resolver index version",
        ),
        (
            lambda document: document["entries"][0].update(digest="sha256:abc"),
            "digest",
        ),
        (
            lambda document: document["entries"][0].update(urls=["http://models.example/model"]),
            "HTTPS URL",
        ),
        (
            lambda document: document["entries"][0].update(
                urls=["https://models.example/model file"]
            ),
            "raw whitespace",
        ),
        (lambda document: document.update(description=None), "must be a string"),
        (lambda document: document["entries"][0].update(components=None), "must be a list"),
        (lambda document: document["entries"][0].update(kind="checkpoint"), "asset kind"),
        (lambda document: document["entries"][0].update(size=-2), "non-negative integer"),
        (lambda document: document.update(updated="2026-09-01T00:00:00"), "RFC 3339"),
    ],
)
def test_parse_resolver_index_rejects_invalid_documents(
    valid_document: dict[str, object],
    mutate: object,
    message: str,
) -> None:
    document = json.loads(json.dumps(valid_document))
    assert isinstance(document, dict)
    mutate(document)  # type: ignore[operator]
    with pytest.raises(ResolverIndexError, match=message):
        parse_resolver_index(json.dumps(document))


@pytest.mark.parametrize("regional", [False, True])
def test_parse_resolver_index_accepts_64_urls_and_rejects_65(regional: bool) -> None:
    def document_with_url_count(count: int) -> dict[str, object]:
        urls = [f"https://models.example/{number}" for number in range(count)]
        entry = resolver_entry()
        if regional:
            entry["regions"] = {"us": urls}
        else:
            entry["urls"] = urls
        return resolver_document(entry)

    parsed = parse_resolver_index(json.dumps(document_with_url_count(RESOLVER_INDEX_MAX_URLS)))
    entry = parsed.entries[0]
    assert len(entry.regions["us"] if regional else entry.urls) == RESOLVER_INDEX_MAX_URLS

    field_name = "entries[0].regions['us']" if regional else "entries[0].urls"
    with pytest.raises(
        ResolverIndexError,
        match=rf"{re.escape(field_name)} accepts at most {RESOLVER_INDEX_MAX_URLS} URLs",
    ):
        parse_resolver_index(json.dumps(document_with_url_count(RESOLVER_INDEX_MAX_URLS + 1)))


def test_parse_resolver_index_rejects_duplicate_fields_and_digests(
    valid_document: dict[str, object],
) -> None:
    with pytest.raises(ResolverIndexError, match="duplicate JSON field 'dinksterResolver'"):
        parse_resolver_index('{"dinksterResolver":1,"dinksterResolver":1,"entries":[]}')
    document = json.loads(json.dumps(valid_document))
    document["entries"].append(dict(document["entries"][0]))
    with pytest.raises(ResolverIndexError, match="duplicates digest"):
        parse_resolver_index(json.dumps(document))


def test_parse_resolver_index_enforces_document_limits() -> None:
    with pytest.raises(ResolverIndexError, match=str(RESOLVER_INDEX_MAX_BYTES)):
        parse_resolver_index(b" " * (RESOLVER_INDEX_MAX_BYTES + 1))
    entry = resolver_entry()
    with pytest.raises(ResolverIndexError, match=str(RESOLVER_INDEX_MAX_ENTRIES)):
        resolver_index_from_wire(
            {
                "dinksterResolver": 1,
                "entries": [entry] * (RESOLVER_INDEX_MAX_ENTRIES + 1),
            }
        )


def test_provenance_named_sources_are_replaceable_isolated_and_persistent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.json"
    store = ProvenanceStore(path)
    store.add(
        ProvenanceRecord(
            MODEL_DIGEST,
            ("https://official.example/model",),
            license="official",
        )
    )
    store.replace_source(
        "resolver-index:first",
        (
            ProvenanceRecord(
                MODEL_DIGEST,
                ("https://first.example/model",),
                note="first",
            ),
        ),
    )
    store.replace_source(
        "resolver-index:second",
        (ProvenanceRecord(MODEL_DIGEST, ("https://second.example/model",)),),
    )
    assert store.sources(MODEL_DIGEST) == (
        "https://official.example/model",
        "https://first.example/model",
        "https://second.example/model",
    )
    assert store.remove_source("resolver-index:first")
    assert store.sources(MODEL_DIGEST) == (
        "https://official.example/model",
        "https://second.example/model",
    )
    reloaded = ProvenanceStore(path)
    assert reloaded.source_names() == ("resolver-index:second",)
    assert reloaded.get(MODEL_DIGEST).license == "official"  # type: ignore[union-attr]


def test_provenance_legacy_list_migrates_on_next_write(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps([ProvenanceRecord(MODEL_DIGEST, ("https://legacy.example/model",)).to_wire()]),
        "utf-8",
    )
    store = ProvenanceStore(path)
    store.replace_source(
        "resolver-index:new",
        (ProvenanceRecord(OTHER_DIGEST, ("https://new.example/model",)),),
    )
    persisted = json.loads(path.read_text("utf-8"))
    assert persisted["dinksterProvenance"] == 1
    assert store.sources(MODEL_DIGEST) == ("https://legacy.example/model",)
    assert store.sources(OTHER_DIGEST) == ("https://new.example/model",)


def test_local_subscriptions_replace_remove_and_reload(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    subscriptions_path = tmp_path / "resolver-indexes.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_document(first_path, resolver_document(resolver_entry()))
    write_document(
        second_path,
        resolver_document(
            resolver_entry(
                name="other/example.safetensors",
                urls=["https://second.example/default"],
                regions={"cn": ["https://second.example/cn"]},
            )
        ),
    )
    provenance = ProvenanceStore(provenance_path)
    provenance.add(ProvenanceRecord(MODEL_DIGEST, ("https://official.example/model",)))
    store = ResolverSubscriptionStore(subscriptions_path, provenance, region="cn")
    first = store.subscribe(str(first_path))
    second = store.subscribe(str(second_path))
    assert provenance.sources(MODEL_DIGEST) == (
        "https://official.example/model",
        "https://cn.example/model.safetensors",
        "https://models.example/default.safetensors",
        "https://second.example/cn",
        "https://second.example/default",
    )
    assert store.unsubscribe(first.id)
    assert provenance.sources(MODEL_DIGEST) == (
        "https://official.example/model",
        "https://second.example/cn",
        "https://second.example/default",
    )
    reloaded_provenance = ProvenanceStore(provenance_path)
    reloaded = ResolverSubscriptionStore(subscriptions_path, reloaded_provenance, region="cn")
    assert [subscription.id for subscription in reloaded.subscriptions()] == [second.id]
    assert reloaded_provenance.sources(MODEL_DIGEST) == provenance.sources(MODEL_DIGEST)


def test_persisted_unused_optional_field_keeps_http_entries_and_suggestions(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    write_document(source, resolver_document())
    ResolverSubscriptionStore(state, provenance).subscribe(str(source))

    persisted = json.loads(state.read_text("utf-8"))
    subscriptions = persisted["subscriptions"]
    assert isinstance(subscriptions, list) and isinstance(subscriptions[0], dict)
    subscriptions[0]["retiredOptionalField"] = {"unused": True}
    state.write_text(json.dumps(persisted), "utf-8")

    reloaded = ResolverSubscriptionStore(state, provenance)
    assert provenance.sources(MODEL_DIGEST)
    assert [suggestion.digest for suggestion in reloaded.suggest("example.safetensors")] == [
        MODEL_DIGEST
    ]


def test_subscription_state_rejects_ambiguous_nonfinite_and_oversized_json(
    tmp_path: Path,
) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    write_document(source, resolver_document(resolver_entry()))
    store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))
    subscription = store.subscribe(str(source))
    valid = state.read_text("utf-8")
    checked_at = f'"checkedAt": {subscription.checked_at}'

    duplicate = valid.replace(
        '"dinksterResolverSubscriptions": 1,',
        '"dinksterResolverSubscriptions": 1,\n "dinksterResolverSubscriptions": 1,',
    )
    for invalid, message in (
        (duplicate, "duplicate resolver subscription JSON field"),
        (valid.replace(checked_at, '"checkedAt": Infinity'), "invalid.*number"),
        (valid.replace(checked_at, '"checkedAt": 1e400'), "checkedAt is invalid"),
        (
            valid.replace(checked_at, '"checkedAt": ' + "1" * 5000),
            "invalid resolver subscription JSON",
        ),
    ):
        state.write_text(invalid, "utf-8")
        with pytest.raises(ResolverSubscriptionError, match=message):
            ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "reload.json"))

    with state.open("wb") as handle:
        handle.truncate(RESOLVER_SUBSCRIPTIONS_MAX_BYTES + 1)
    with pytest.raises(ResolverSubscriptionError, match="file exceeds"):
        ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "oversized.json"))


def test_local_refresh_replaces_entries_and_preserves_last_good_on_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry()))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    subscription = store.subscribe(str(source))

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

    source.write_text("{invalid", "utf-8")
    (failed,) = store.refresh(subscription.id)
    assert "invalid resolver index JSON" in str(failed["error"])
    assert provenance.sources(OTHER_DIGEST) == ("https://models.example/replacement",)


def test_multi_source_refresh_reads_concurrently(tmp_path: Path) -> None:
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    for position in range(4):
        source = tmp_path / f"index-{position}.json"
        write_document(source, resolver_document(resolver_entry()))
        store.subscribe(str(source))

    barrier = threading.Barrier(4)
    refresh_one = store._refresh_one

    def synchronized_refresh(
        subscription: ResolverSubscription,
        *,
        now: float,
    ) -> tuple[ResolverSubscription, bool]:
        barrier.wait(timeout=5)
        return refresh_one(subscription, now=now)

    with patch.object(store, "_refresh_one", side_effect=synchronized_refresh):
        refreshed = store.refresh()

    assert len(refreshed) == 4


def test_unsubscribe_rolls_back_when_subscription_persistence_fails(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry()))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    subscription = store.subscribe(str(source))

    with (
        patch.object(store, "_save", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        store.unsubscribe(subscription.id)

    assert [row.id for row in store.subscriptions()] == [subscription.id]
    assert provenance.sources(MODEL_DIGEST) == ("https://models.example/default.safetensors",)


def test_refresh_rolls_back_prior_sources_when_later_source_update_fails(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_document(first_path, resolver_document(resolver_entry()))
    write_document(
        second_path,
        resolver_document(
            resolver_entry(
                digest=OTHER_DIGEST,
                name="other.safetensors",
                urls=["https://models.example/other"],
            )
        ),
    )
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    store.subscribe(str(first_path))
    store.subscribe(str(second_path))
    write_document(
        first_path,
        resolver_document(resolver_entry(urls=["https://models.example/first-updated"])),
    )
    write_document(
        second_path,
        resolver_document(
            resolver_entry(
                digest=OTHER_DIGEST,
                name="other.safetensors",
                urls=["https://models.example/second-updated"],
            )
        ),
    )
    replace_source = provenance.replace_source
    calls = 0

    def fail_second_update(source_name: str, records: tuple[ProvenanceRecord, ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        replace_source(source_name, records)

    with (
        patch.object(provenance, "replace_source", side_effect=fail_second_update),
        pytest.raises(OSError, match="disk full"),
    ):
        store.refresh()

    assert provenance.sources(MODEL_DIGEST) == ("https://models.example/default.safetensors",)
    assert provenance.sources(OTHER_DIGEST) == ("https://models.example/other",)


def test_filename_suggestions_are_exact_case_sensitive_basenames(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry(name="folder/Model.safetensors")))
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )
    store.subscribe(str(source))
    assert [item.digest for item in store.suggest("foo\\bar\\Model.safetensors")] == [MODEL_DIGEST]
    assert store.suggest("model.safetensors") == ()
    assert store.suggest("Model.ckpt") == ()


def test_subscription_source_validation_is_closed(tmp_path: Path) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )
    for source in (
        "",
        " relative.json ",
        "http://models.example/index.json",
        "ftp://models.example/index.json",
        "https://user:pass@models.example/index.json",
        "https://models.example/index.json#fragment",
    ):
        with pytest.raises(ResolverSubscriptionError):
            store.subscribe(source)


def test_future_checked_at_does_not_suppress_revalidation(
    tmp_path: Path,
    valid_document: dict[str, object],
) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )
    subscription = ResolverSubscription(
        id="0" * 32,
        source="https://models.example/index.json",
        source_type="url",
        index=parse_resolver_index(json.dumps(valid_document)),
        checked_at=200.0,
        complete_snapshot=True,
    )
    with patch(
        "dinkster_assets.resolver_subscription._fetch_url",
        return_value=(None, ""),
    ) as fetch:
        refreshed, changed = store._refresh_one(subscription, now=100.0)
    assert changed
    assert refreshed.checked_at == 100.0
    assert not refreshed.error
    fetch.assert_called_once()


def test_nonfinite_clock_and_serialized_timestamp_are_rejected(
    tmp_path: Path,
    valid_document: dict[str, object],
) -> None:
    state = tmp_path / "subscriptions.json"
    store = ResolverSubscriptionStore(
        state,
        ProvenanceStore(tmp_path / "provenance.json"),
        clock=lambda: float("nan"),
    )
    with (
        patch(
            "dinkster_assets.resolver_subscription._fetch_url",
            return_value=(json.dumps(valid_document).encode(), ""),
        ),
        pytest.raises(ResolverSubscriptionError, match="clock.*invalid timestamp"),
    ):
        store.subscribe("https://models.example/index.json")
    assert not state.exists()

    source = tmp_path / "index.json"
    write_document(source, valid_document)
    valid_store = ResolverSubscriptionStore(
        state,
        ProvenanceStore(tmp_path / "valid-provenance.json"),
    )
    valid_store.subscribe(str(source))
    with (
        patch.object(
            ResolverSubscription,
            "to_wire",
            return_value={"checkedAt": float("inf")},
        ),
        pytest.raises(ResolverSubscriptionError, match="cannot serialize"),
    ):
        valid_store._save()


def test_hosted_subscription_revalidates_with_etag_at_most_every_six_hours(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        now = [100.0]
        state: dict[str, object] = {
            "document": resolver_document(resolver_entry()),
            "etag": '"one"',
            "requests": [],
        }

        async def index(request: web.Request) -> web.Response:
            requests = state["requests"]
            assert isinstance(requests, list)
            assert "Authorization" not in request.headers
            assert "Cookie" not in request.headers
            requests.append(request.headers.get("If-None-Match"))
            if request.headers.get("If-None-Match") == state["etag"]:
                return web.Response(status=304, headers={"ETag": str(state["etag"])})
            return web.json_response(
                state["document"],
                headers={"ETag": str(state["etag"])},
            )

        app = web.Application()
        app.router.add_get("/index.json", index)
        server = TestServer(app)
        await server.start_server()
        try:
            provenance = ProvenanceStore(tmp_path / "provenance.json")
            store = ResolverSubscriptionStore(
                tmp_path / "subscriptions.json",
                provenance,
                clock=lambda: now[0],
                fetch_timeout=2.0,
            )
            subscription = await asyncio.to_thread(
                store.subscribe, str(server.make_url("/index.json"))
            )
            assert state["requests"] == [None]
            assert subscription.etag == '"one"'
            assert store.refresh() == ()
            assert state["requests"] == [None]

            now[0] += RESOLVER_REVALIDATE_SECONDS
            (not_modified,) = await asyncio.to_thread(store.refresh)
            assert not_modified["etag"] == '"one"'
            assert state["requests"] == [None, '"one"']

            state["etag"] = '"two"'
            state["document"] = resolver_document(
                resolver_entry(
                    digest=OTHER_DIGEST,
                    name="replacement.safetensors",
                    urls=["https://models.example/replacement"],
                )
            )
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (updated,) = await asyncio.to_thread(store.refresh)
            assert updated["etag"] == '"two"'
            assert provenance.sources(MODEL_DIGEST) == ()
            assert provenance.sources(OTHER_DIGEST) == ("https://models.example/replacement",)

            state["etag"] = '"three"'
            state["document"] = {"dinksterResolver": 99, "entries": []}
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (failed,) = await asyncio.to_thread(store.refresh)
            assert "unsupported resolver index version" in str(failed["error"])
            assert provenance.sources(OTHER_DIGEST) == ("https://models.example/replacement",)
        finally:
            await server.close()

    asyncio.run(scenario())


@asynccontextmanager
async def resolver_http_server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get("/dinkster/v1/export/resolver", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/dinkster/v1/export/resolver"))
    finally:
        await server.close()


@pytest.mark.parametrize(
    "source,provider_id",
    [
        (None, None),
        (None, "fixture"),
        ("https://fixture.invalid/export", None),
        ("https://user:password@fixture.invalid/export", "fixture"),
        ("file:///tmp/export.json", "fixture"),
        ("https://fixture.invalid/export", " padded "),
        ("https://fixture.invalid/export", "x" * 513),
        ("https://fixture.invalid/export", "bad\nID"),
    ],
)
def test_official_bootstrap_requires_valid_explicit_configuration(
    tmp_path: Path, source: str | None, provider_id: str | None
) -> None:
    state = tmp_path / "subscriptions.json"
    store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))
    with patch.object(
        resolver_transport, "_fetch_url", side_effect=AssertionError("unexpected network")
    ):
        with pytest.raises(ResolverSubscriptionError):
            store.bootstrap_official(source, provider_id)
    assert not state.exists() and not store.subscriptions()


@pytest.mark.parametrize("cursor", ["", 1, False, [], {}, "x" * 2049, "\ud800"])
def test_hosted_resolver_rejects_malformed_cursor(tmp_path: Path, cursor: object) -> None:
    async def scenario() -> None:
        async def index(_: web.Request) -> web.Response:
            return web.json_response({**resolver_document(), "nextCursor": cursor})

        async with resolver_http_server(index) as url:
            state = tmp_path / "subscriptions.json"
            store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))
            with pytest.raises(ResolverIndexError, match="nextCursor"):
                await asyncio.to_thread(store.subscribe, url)
            assert not state.exists()

    asyncio.run(scenario())


def test_first_page_not_modified_requires_a_cached_complete_index(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def index(_: web.Request) -> web.Response:
            return web.Response(status=304)

        async with resolver_http_server(index) as url:
            store = ResolverSubscriptionStore(
                tmp_path / "subscriptions.json", ProvenanceStore(tmp_path / "provenance.json")
            )
            with pytest.raises(ResolverSubscriptionError, match="cached complete document"):
                await asyncio.to_thread(store.subscribe, url)
            assert store.subscriptions() == ()

    asyncio.run(scenario())


@contextmanager
def resolver_fetch_threads() -> Iterator[list[threading.Thread]]:
    workers: list[threading.Thread] = []

    class TrackedThread(threading.Thread):
        def start(self) -> None:
            super().start()
            if self.name == "resolver-fetch":
                workers.append(self)

    # Track creation, not arrival at a mocked socket call: TLS setup may still
    # be running when the caller times out. Join before restoring its mocks.
    with patch.object(threading, "Thread", TrackedThread):
        try:
            yield workers
        finally:
            for worker in workers:
                worker.join(2.0)
                assert not worker.is_alive()


@pytest.mark.parametrize("pages", [1, 2])
def test_http_resolver_does_not_initialize_tls(pages: int) -> None:
    async def scenario() -> None:
        calls = 0

        async def index(request: web.Request) -> web.Response:
            nonlocal calls
            calls += 1
            if pages == 2 and "cursor" not in request.query:
                return web.json_response({**resolver_document(), "nextCursor": "next"})
            return web.json_response(resolver_document(resolver_entry(digest=OTHER_DIGEST)))

        async with resolver_http_server(index) as url:
            with patch(
                "ssl._create_default_https_context", side_effect=AssertionError("HTTP loaded TLS")
            ):
                data, _etag = await asyncio.to_thread(
                    resolver_transport._fetch_url,
                    url,
                    "",
                    timeout=resolver_transport.RESOLVER_FETCH_TIMEOUT,
                )
            assert data is not None
            assert len(parse_resolver_index(data).entries) == pages
            assert calls == pages

    asyncio.run(scenario())


def test_https_resolver_creates_one_verified_tls_context() -> None:
    context = ssl.create_default_context()
    connections: list[http.client.HTTPSConnection] = []

    def connect(connection: http.client.HTTPSConnection) -> None:
        connections.append(connection)
        assert connection._context is context  # type: ignore[reportPrivateUsage]
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        raise OSError("stop before network access")

    with (
        patch("ssl._create_default_https_context", return_value=context) as create_context,
        patch.object(http.client.HTTPSConnection, "connect", connect),
        resolver_fetch_threads(),
    ):
        with pytest.raises(ResolverSubscriptionError, match="stop before network access"):
            resolver_transport._fetch_url("https://models.example/index", "", timeout=1.0)
        assert len(connections) == 1
        create_context.assert_called_once_with()


def test_fetch_returns_capacity_to_the_semaphore_it_acquired() -> None:
    original = threading.BoundedSemaphore(1)
    replacement = threading.BoundedSemaphore(1)
    entered = threading.Event()
    release = threading.Event()

    def blocked(*_args: object, **_kwargs: object) -> tuple[bytes | None, str]:
        entered.set()
        assert release.wait(2.0)
        return None, "late"

    with (
        patch.object(resolver_transport, "_FETCH_SLOTS", original),
        patch.object(resolver_transport, "_fetch_url_until_deadline", side_effect=blocked),
        resolver_fetch_threads() as workers,
    ):
        try:
            with pytest.raises(ResolverSubscriptionError, match="time limit"):
                resolver_transport._fetch_url("http://localhost/index", "", timeout=0.05)
            assert entered.wait(1.0)
            assert not original.acquire(blocking=False)
            with patch.object(resolver_transport, "_FETCH_SLOTS", replacement):
                release.set()
                for worker in workers:
                    worker.join(1.0)
                    assert not worker.is_alive()
            assert original.acquire(blocking=False)
            assert not original.acquire(blocking=False)
            assert replacement.acquire(blocking=False)
            assert not replacement.acquire(blocking=False)
        finally:
            release.set()


def test_cursor_pages_share_one_timeout_and_preserve_previous_refresh(tmp_path: Path) -> None:
    mode = "initial"
    elapsed = [0.0]
    timeouts: list[float] = []
    deadlines: list[float] = []

    def page(
        url: str, _etag: str, *, timeout: float, deadline: float, max_bytes: int
    ) -> tuple[bytes, str]:
        timeouts.append(timeout)
        deadlines.append(deadline)
        if mode == "initial":
            document = resolver_document()
        else:
            elapsed[0] += 0.35
            document = (
                {**resolver_document(), "nextCursor": "next"}
                if "cursor=" not in url
                else resolver_document(resolver_entry(digest=OTHER_DIGEST))
            )
        data = json.dumps(document).encode()
        assert len(data) < max_bytes
        return data, ""

    now = [100.0]
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
        clock=lambda: now[0],
        fetch_timeout=0.5,
    )
    # Advance page time explicitly so the test proves the shared budget rather
    # than assuming the OS schedules page two inside a short wall-clock window.
    with (
        patch.object(resolver_transport, "time", wraps=time) as clock,
        patch.object(resolver_transport, "_fetch_page", side_effect=page),
        resolver_fetch_threads(),
    ):
        clock.monotonic.side_effect = lambda: elapsed[0]
        prior = store.subscribe("http://localhost/index")
        mode = "slow"
        now[0] += RESOLVER_REVALIDATE_SECONDS
        (failed,) = store.refresh()
        assert "time limit" in str(failed["error"])
        assert timeouts == pytest.approx([0.5, 0.5, 0.15])
        assert deadlines == [0.5, 0.5, 0.5]
        assert store.subscriptions()[0].refreshed_at == prior.refreshed_at
        assert store.subscriptions()[0].index == prior.index


@pytest.mark.parametrize("part", ["headers", "chunk-framing", "body"])
def test_refresh_deadline_interrupts_trickling_http_response(tmp_path: Path, part: str) -> None:
    async def scenario() -> None:
        finished = asyncio.Event()

        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.readuntil(b"\r\n\r\n")
                if part == "headers":
                    writer.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                elif part == "chunk-framing":
                    writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1;")
                else:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
                with suppress(ConnectionError):
                    for _ in range(100):
                        writer.write(b"x")
                        await writer.drain()
                        await asyncio.sleep(0.03)
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
                finished.set()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            store = ResolverSubscriptionStore(
                tmp_path / "subscriptions.json",
                ProvenanceStore(tmp_path / "provenance.json"),
                fetch_timeout=0.2,
            )
            started = time.monotonic()
            with pytest.raises(ResolverSubscriptionError):
                await asyncio.to_thread(store.subscribe, f"http://127.0.0.1:{port}/index")
            assert time.monotonic() - started < 1.5
            await asyncio.wait_for(finished.wait(), 1.0)
            assert store.subscriptions() == ()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["dns", "http-connect", "tls-context", "tls-connect"])
def test_fetch_deadline_covers_blocking_connection_establishment(phase: str) -> None:
    release = threading.Event()
    context = ssl.create_default_context()

    def blocked(*_args: object, **_kwargs: object) -> None:
        assert release.wait(2.0)
        raise socket.gaierror("delayed connection failure")

    target = {
        "dns": "socket.getaddrinfo",
        "http-connect": "http.client.HTTPConnection.connect",
        "tls-context": "ssl._create_default_https_context",
        "tls-connect": "http.client.HTTPSConnection.connect",
    }[phase]
    url = "https://models.example/index" if phase.startswith("tls-") else "http://localhost/index"
    with (
        patch("ssl._create_default_https_context", return_value=context),
        patch(target, side_effect=blocked) as blocked_phase,
        resolver_fetch_threads() as workers,
    ):
        try:
            started = time.monotonic()
            with pytest.raises(ResolverSubscriptionError, match="time limit"):
                resolver_transport._fetch_url(url, "", timeout=0.05)
            assert time.monotonic() - started < 0.5
            blocked_phase.assert_called_once()
            assert len(workers) == 1
            assert workers[0].is_alive() and workers[0].daemon
        finally:
            release.set()


def test_fetch_capacity_is_held_until_timed_out_dns_workers_exit() -> None:
    release = threading.Event()
    capacity = resolver_transport._REFRESH_WORKERS
    slots = threading.BoundedSemaphore(capacity)

    def blocked(*_args: object, **_kwargs: object) -> None:
        assert release.wait(3.0)
        raise socket.gaierror("delayed DNS failure")

    with (
        patch.object(resolver_transport, "_FETCH_SLOTS", slots),
        patch("socket.getaddrinfo", side_effect=blocked) as blocked_dns,
        resolver_fetch_threads() as workers,
    ):
        try:
            for _ in range(capacity):
                with pytest.raises(ResolverSubscriptionError, match="time limit"):
                    resolver_transport._fetch_url("http://localhost/index", "", timeout=0.05)
            assert len(workers) == capacity
            assert blocked_dns.call_count == capacity
            assert all(worker.daemon and worker.is_alive() for worker in workers)
            started = time.monotonic()
            for _ in range(capacity * 2):
                with pytest.raises(ResolverSubscriptionError, match="capacity exhausted"):
                    resolver_transport._fetch_url("http://localhost/index", "", timeout=15.0)
            assert time.monotonic() - started < 0.5
            assert len(workers) == capacity
        finally:
            release.set()
            for worker in workers:
                worker.join(2.0)
                assert not worker.is_alive()
        for _ in range(capacity):
            assert slots.acquire(blocking=False)
        assert not slots.acquire(blocking=False)
        for _ in range(capacity):
            slots.release()
        with patch.object(
            resolver_transport,
            "_fetch_page",
            return_value=(json.dumps(resolver_document()).encode(), "fresh"),
        ):
            assert (
                resolver_transport._fetch_url("http://localhost/index", "", timeout=1.0)[1]
                == "fresh"
            )


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_resolver_timeout_must_be_finite_and_positive(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ResolverSubscriptionStore(
            tmp_path / "subscriptions.json",
            ProvenanceStore(tmp_path / "provenance.json"),
            fetch_timeout=timeout,
        )


@pytest.mark.parametrize("marker", [None, 0, 1, "true", [], {}])
def test_persisted_completion_marker_requires_boolean(tmp_path: Path, marker: object) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    write_document(source, resolver_document())
    ResolverSubscriptionStore(state, provenance).subscribe(str(source))
    persisted = json.loads(state.read_text())
    persisted["subscriptions"][0]["completeSnapshot"] = marker
    state.write_text(json.dumps(persisted))
    with pytest.raises(ResolverSubscriptionError, match="completeSnapshot must be a boolean"):
        ResolverSubscriptionStore(state, provenance)


def test_local_partial_resolver_cannot_replace_or_reload_complete_authority(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    now = [100.0]
    terminal = {**resolver_document(), "nextCursor": None}
    write_document(source, terminal)
    store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
    prior = store.subscribe(str(source))
    assert prior.complete_snapshot
    assert "nextCursor" not in prior.index.to_wire()
    persisted = json.loads(state.read_text())
    persisted["subscriptions"][0]["index"] = terminal
    state.write_text(json.dumps(persisted))
    store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
    assert store.subscriptions()[0] == prior
    partial = {**resolver_document(), "nextCursor": "nonterminal"}
    write_document(source, partial)
    now[0] += 10
    (failed,) = store.refresh()
    assert "partial resolver index" in str(failed["error"])
    assert store.subscriptions()[0].refreshed_at == prior.refreshed_at
    with pytest.raises(ResolverIndexError, match="partial resolver index"):
        resolver_index_from_wire(partial)
    persisted = json.loads(state.read_text())
    persisted["subscriptions"][0]["index"] = partial
    state.write_text(json.dumps(persisted))
    with pytest.raises(ResolverIndexError, match="partial resolver index"):
        ResolverSubscriptionStore(state, provenance)


class _BytesResponse:
    def __init__(self, data: bytes) -> None:
        self._chunks: Iterator[bytes] = iter((data, b""))

    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return next(self._chunks)


class _IncompleteResponse:
    status = 200
    headers: dict[str, str] = {}

    def __enter__(self) -> _IncompleteResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://models.example/index.json"

    def read1(self, _: int) -> bytes:
        raise http.client.IncompleteRead(b"partial", 100)


def test_hosted_body_protocol_failure_is_a_subscription_error() -> None:
    class Opener:
        def open(self, *_: object, **__: object) -> _IncompleteResponse:
            return _IncompleteResponse()

    with (
        patch(
            "dinkster_assets.resolver_subscription.urllib.request.build_opener",
            return_value=Opener(),
        ),
        pytest.raises(ResolverSubscriptionError, match="resolver index request failed"),
    ):
        from dinkster_assets.resolver_subscription import _fetch_url

        _fetch_url("https://models.example/index.json", "", timeout=1.0)


@pytest.mark.parametrize(
    ("served", "expected_status"),
    [(MODEL_BYTES, "acquired"), (b"tampered bytes", "failed")],
)
def test_resolver_acquisition_is_digest_safe(
    tmp_path: Path,
    served: bytes,
    expected_status: str,
) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry()))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    store.subscribe(str(source))
    vault = AssetVault(tmp_path / "vault")
    with patch(
        "dinkster_assets.fetch.urllib.request.urlopen",
        return_value=_BytesResponse(served),
    ):
        result = acquire_need(
            AssetNeed(name="example.safetensors", digest=MODEL_DIGEST),
            vault,
            provenance=provenance,
        )
    assert result.status == expected_status
    assert vault.has(MODEL_DIGEST) is (served == MODEL_BYTES)
    if served != MODEL_BYTES:
        assert not [path for path in vault.root.rglob("*") if path.is_file()]


def test_acquisition_survives_provenance_persistence_failure(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry()))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance).subscribe(str(source))
    vault = AssetVault(tmp_path / "vault")
    with (
        patch(
            "dinkster_assets.fetch.urllib.request.urlopen",
            return_value=_BytesResponse(MODEL_BYTES),
        ),
        patch.object(provenance, "add", side_effect=OSError("disk full")),
    ):
        result = acquire_need(
            AssetNeed(name="example.safetensors", digest=MODEL_DIGEST),
            vault,
            provenance=provenance,
        )
    assert result.status == "acquired"
    assert vault.has(MODEL_DIGEST)


def test_resolver_management_and_filename_guess_api(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    components: list[dict[str, object]] = [
        {
            "kind": "model/diffusion",
            "architecture": "example-transformer",
            "dtype": "float16",
        },
        {"kind": "model/vae"},
    ]
    write_document(
        source,
        resolver_document(
            resolver_entry(
                name="Model.safetensors",
                kind="model/checkpoint",
                components=components,
            )
        ),
    )
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        provenance=provenance,
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, library=library)
        add_resolver_index_routes(app, store)
        add_guess_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/assets/resolver-indexes",
                json={"source": str(source)},
            )
            assert response.status == 201, await response.text()
            subscription = await response.json()

            response = await client.get("/api/assets/resolver-indexes")
            assert response.status == 200
            assert (await response.json())["subscriptions"][0]["entryCount"] == 1

            response = await client.get(f"/api/assets/{MODEL_DIGEST}/sources")
            assert response.status == 200
            provenance_wire = await response.json()
            assert provenance_wire["sources"] == ["https://models.example/default.safetensors"]
            assert provenance_wire["metadata"]["components"] == components

            response = await client.post(
                "/api/assets/guess",
                json={"names": ["foo/bar/Model.safetensors", "model.safetensors"]},
            )
            assert response.status == 200
            exact, wrong_case = (await response.json())["matches"]
            (candidate,) = exact["candidates"]
            assert candidate["digest"] == MODEL_DIGEST
            assert candidate["confidence"] == "name"
            assert candidate["resolverIndex"] == subscription["id"]
            assert candidate["held"] is False
            assert candidate["components"] == components
            assert wrong_case["candidates"] == []
            assert provenance.get(MODEL_DIGEST).metadata["components"] == components  # type: ignore[union-attr]

            response = await client.delete(f"/api/assets/resolver-indexes/{subscription['id']}")
            assert response.status == 200
            assert provenance.sources(MODEL_DIGEST) == ()
            response = await client.post(
                "/api/assets/guess",
                json={"names": ["Model.safetensors"]},
            )
            assert (await response.json())["matches"][0]["candidates"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_resolver_api_refreshes_local_subscriptions_on_startup(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document(resolver_entry()))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(tmp_path / "subscriptions.json", provenance)
    store.subscribe(str(source))
    write_document(
        source,
        resolver_document(
            resolver_entry(
                digest=OTHER_DIGEST,
                name="new.safetensors",
                urls=["https://models.example/new"],
            )
        ),
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        add_resolver_index_routes(app, store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert provenance.sources(MODEL_DIGEST) == ()
            assert provenance.sources(OTHER_DIGEST) == ("https://models.example/new",)
            response = await client.get("/api/assets/resolver-indexes")
            assert response.status == 200
            assert (await response.json())["subscriptions"][0]["entryCount"] == 1
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", 1),
        ("source", "file:///tmp/export"),
        ("providerId", ""),
        ("providerId", " padded "),
        ("subscriptionId", "invalid"),
        ("extra", True),
        (None, None),
    ],
)
def test_official_bootstrap_rejects_invalid_persisted_binding(
    tmp_path: Path, field: str | None, value: object
) -> None:
    bootstrap: dict[str, object] = {
        "source": "https://fixture.invalid/export",
        "providerId": "fixture",
        "subscriptionId": "a" * 32,
    }
    if field is not None:
        bootstrap[field] = value
    state = tmp_path / "subscriptions.json"
    write_document(
        state,
        {
            "dinksterResolverSubscriptions": 1,
            "subscriptions": [],
            "officialBootstrap": bootstrap if field is not None else None,
        },
    )
    with pytest.raises(ResolverSubscriptionError, match="official resolver"):
        ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))


def test_resolver_api_startup_refresh_persistence_failure_is_nonfatal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        add_resolver_index_routes(app, store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/assets/resolver-indexes")
            assert response.status == 200
        finally:
            await client.close()

    with patch.object(store, "refresh", side_effect=OSError("disk full")):
        asyncio.run(scenario())
    assert "resolver index startup refresh failed: disk full" in caplog.text


def test_resolver_management_api_validates_requests(tmp_path: Path) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        add_resolver_index_routes(app, store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for body in ([], {}, {"source": 42}, {"source": "x", "extra": True}):
                response = await client.post("/api/assets/resolver-indexes", json=body)
                assert response.status == 400
            response = await client.post(
                "/api/assets/resolver-indexes/refresh",
                json={"id": "unknown"},
            )
            assert response.status == 404
            response = await client.delete("/api/assets/resolver-indexes/unknown")
            assert response.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_refresh_io_does_not_block_resolver_reads(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document())
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json", ProvenanceStore(tmp_path / "provenance.json")
    )
    subscription = store.subscribe(str(source))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    read_local = resolver_transport._read_local

    def blocked_read(path: str) -> bytes:
        started.set()
        assert release.wait(3.0)
        return read_local(path)

    with patch.object(resolver_transport, "_read_local", side_effect=blocked_read):
        refresher = threading.Thread(target=lambda: store.refresh(subscription.id), daemon=True)
        refresher.start()
        assert started.wait(1.0)
        reader = threading.Thread(
            target=lambda: (
                store.subscriptions(),
                store.suggest("example.safetensors"),
                finished.set(),
            ),
            daemon=True,
        )
        reader.start()
        try:
            assert finished.wait(0.5)
        finally:
            release.set()
            refresher.join(2.0)
            reader.join(2.0)
    assert not refresher.is_alive() and not reader.is_alive()


@pytest.mark.parametrize("failure", ["apply", "save"])
def test_refresh_failure_restores_every_observable(tmp_path: Path, failure: str) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_document(first_path, resolver_document())
    write_document(second_path, resolver_document(resolver_entry(digest=OTHER_DIGEST)))
    state = tmp_path / "subscriptions.json"
    provenance_path = tmp_path / "provenance.json"
    provenance = ProvenanceStore(provenance_path)
    store = ResolverSubscriptionStore(state, provenance, clock=lambda: 100.0)
    store.subscribe(str(first_path))
    store.subscribe(str(second_path))
    before = store.subscriptions()
    saved = state.read_bytes()
    provenance_saved = provenance_path.read_bytes()
    records = provenance.records()
    suggestions = store.suggest("example.safetensors")
    write_document(first_path, resolver_document(resolver_entry(name="updated-first.safetensors")))
    write_document(
        second_path,
        resolver_document(resolver_entry(digest=OTHER_DIGEST, name="updated-second.safetensors")),
    )
    apply, save = store._apply, store._save
    calls = 0

    def fail_during_apply(subscription: ResolverSubscription) -> None:
        nonlocal calls
        calls += 1
        apply(subscription)
        if calls == 2:
            raise OSError("injected apply failure")

    def fail_after_save() -> None:
        save()
        raise OSError("injected save failure")

    fault = (
        patch.object(store, "_apply", side_effect=fail_during_apply)
        if failure == "apply"
        else patch.object(store, "_save", side_effect=fail_after_save)
    )
    with fault, pytest.raises(OSError, match=f"injected {failure} failure"):
        store.refresh()
    current = store.subscriptions()
    assert current == before
    assert all(after is old for after, old in zip(current, before, strict=True))
    assert state.read_bytes() == saved
    assert provenance_path.read_bytes() == provenance_saved
    assert provenance.records() == records
    assert store.suggest("example.safetensors") == suggestions
    assert store.suggest("updated-first.safetensors") == ()
    assert store.suggest("updated-second.safetensors") == ()


def test_refresh_does_not_overwrite_concurrent_subscription_mutation(tmp_path: Path) -> None:
    source = tmp_path / "index.json"
    write_document(source, resolver_document())
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json", ProvenanceStore(tmp_path / "provenance.json")
    )
    subscription = store.subscribe(str(source))
    write_document(source, resolver_document(resolver_entry(name="updated.safetensors")))
    started, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    read_local = resolver_transport._read_local

    def blocked_read(path: str) -> bytes:
        started.set()
        assert release.wait(3.0)
        return read_local(path)

    def refresh() -> None:
        try:
            store.refresh(subscription.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with patch.object(resolver_transport, "_read_local", side_effect=blocked_read):
        refresher = threading.Thread(target=refresh, daemon=True)
        refresher.start()
        assert started.wait(1.0)
        assert store.unsubscribe(subscription.id)
        release.set()
        refresher.join(2.0)
    assert isinstance(errors[0], ResolverSubscriptionError)
    assert "changed during refresh" in str(errors[0])
    mutated = store.subscribe(str(source))
    assert store.subscriptions() == (mutated,)


def test_official_bootstrap_identity_configuration_and_unsubscribe(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = tmp_path / "subscriptions.json"
        store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))

        async def index(_: web.Request) -> web.Response:
            return web.json_response(resolver_document(), headers={"ETag": '"one"'})

        async with resolver_http_server(index) as url:
            subscription = await asyncio.to_thread(
                store.bootstrap_official, url, "community-models"
            )
            assert subscription is not None and subscription.complete_snapshot
            saved = state.read_bytes()
            assert (
                await asyncio.to_thread(store.bootstrap_official, url, "community-models")
                == subscription
            )
            assert state.read_bytes() == saved
            with pytest.raises(ResolverSubscriptionError, match="configuration differs"):
                await asyncio.to_thread(
                    store.bootstrap_official, url + "?different=1", "community-models"
                )
            assert store.unsubscribe(subscription.id)
            reloaded = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "reloaded.json"))
            assert (
                await asyncio.to_thread(reloaded.bootstrap_official, url, "community-models")
                is None
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["identity", "json", "partial", "save"])
def test_official_bootstrap_failure_is_atomic_and_retryable(tmp_path: Path, failure: str) -> None:
    async def scenario() -> None:
        failing = True

        async def index(request: web.Request) -> web.Response:
            if failing and failure == "json":
                return web.Response(text="{")
            document = resolver_document()
            if failing and failure == "identity":
                document["name"] = "wrong"
            if failing and failure == "partial" and "cursor" not in request.query:
                document["nextCursor"] = "next"
            elif failing and failure == "partial":
                return web.Response(status=500)
            return web.json_response(document)

        state = tmp_path / "subscriptions.json"
        store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))
        async with resolver_http_server(index) as url:
            with ExitStack() as stack:
                if failure == "save":
                    stack.enter_context(
                        patch.object(store, "_save", side_effect=OSError("disk full"))
                    )
                with pytest.raises((ResolverIndexError, ResolverSubscriptionError, OSError)):
                    await asyncio.to_thread(store.bootstrap_official, url, "community-models")
            assert not state.exists() and not store.subscriptions()
            failing = False
            assert (
                await asyncio.to_thread(store.bootstrap_official, url, "community-models")
                is not None
            )

    asyncio.run(scenario())


def test_cursor_refresh_refetches_all_pages_despite_unchanged_first_page_etag(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        now = [100.0]
        requests: list[tuple[str, str | None]] = []
        cursor = "opaque+/=&?%# cursor"
        state = tmp_path / "subscriptions.json"
        store = ResolverSubscriptionStore(
            state, ProvenanceStore(tmp_path / "provenance.json"), clock=lambda: now[0]
        )

        async def index(request: web.Request) -> web.Response:
            requests.append((request.raw_path, request.headers.get("If-None-Match")))
            if "cursor" in request.query:
                assert request.query["cursor"] == cursor
                assert request.headers.get("If-None-Match") is None
                return web.json_response(resolver_document(resolver_entry(digest=OTHER_DIGEST)))
            return web.json_response(
                {**resolver_document(), "nextCursor": cursor}, headers={"ETag": '"first"'}
            )

        async with resolver_http_server(index) as url:
            subscription = await asyncio.to_thread(store.subscribe, url + "?limit=1&cursor=stale")
            assert len(subscription.index.entries) == 2 and subscription.etag == ""
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (refreshed,) = await asyncio.to_thread(store.refresh)
            assert refreshed["error"] == ""
            assert len(store.subscriptions()[0].index.entries) == 2
            assert requests == [
                ("/dinkster/v1/export/resolver?limit=1", None),
                ("/dinkster/v1/export/resolver?limit=1&cursor=" + quote(cursor, safe=""), None),
                ("/dinkster/v1/export/resolver?limit=1", None),
                ("/dinkster/v1/export/resolver?limit=1&cursor=" + quote(cursor, safe=""), None),
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("206", "HTTP 206"),
        ("400", "HTTP 400"),
        ("500", "HTTP 500"),
        ("304", "not-modified"),
        ("json", "invalid resolver index JSON"),
        ("truncated", "body is incomplete"),
        ("duplicate", "duplicates digest"),
        ("cycle", "cursor cycle"),
        ("name", "metadata changed"),
        ("updated", "metadata changed"),
        ("description", "metadata changed"),
        ("homepage", "metadata changed"),
        ("providerProtocol", "metadata changed"),
        ("entries", "at most 1 entries"),
        ("pages", "exceeds 1 pages"),
        ("bytes", "bytes"),
        ("chunked-bytes", "bytes"),
    ],
)
def test_failed_cursor_refresh_retains_http_state(
    tmp_path: Path, failure: str, message: str
) -> None:
    async def scenario() -> None:
        entry = resolver_entry()
        removed = resolver_entry(digest=OTHER_DIGEST, size=len(OTHER_BYTES))
        now = [100.0]
        mode = "initial"
        first = {**resolver_document(resolver_entry(digest=OTHER_DIGEST)), "nextCursor": "next"}
        first["providerProtocol"] = 1
        second = {**resolver_document(entry), "providerProtocol": 1}
        state_path = tmp_path / "subscriptions.json"
        provenance = ProvenanceStore(tmp_path / "provenance.json")
        store = ResolverSubscriptionStore(state_path, provenance, clock=lambda: now[0])

        async def index(request: web.Request) -> web.StreamResponse:
            assert "Authorization" not in request.headers
            assert "Cookie" not in request.headers
            if mode == "initial":
                return web.json_response(
                    resolver_document(entry, removed), headers={"ETag": '"old"'}
                )
            if mode == "remove":
                return web.json_response(resolver_document(entry), headers={"ETag": '"old"'})
            if "cursor" not in request.query:
                assert request.headers.get("If-None-Match") == '"old"'
                return web.json_response(first, headers={"ETag": '"candidate"'})
            assert request.headers.get("If-None-Match") is None
            if failure in {"400", "500", "304"}:
                return web.Response(status=int(failure), text="cursor-invalid")
            if failure == "206":
                return web.json_response(second, status=206)
            if failure == "json":
                return web.Response(text="{bad")
            if failure == "truncated":
                body = json.dumps(second).encode()
                response = web.StreamResponse(headers={"Content-Length": str(len(body) + 1)})
                response.force_close()
                await response.prepare(request)
                await response.write(body)
                await response.write_eof()
                return response
            if failure == "duplicate":
                return web.json_response({**second, "entries": first["entries"]})
            if failure == "cycle":
                return web.json_response({**second, "nextCursor": "next"})
            if failure in {"name", "updated", "description", "homepage", "providerProtocol"}:
                changed = dict(second)
                if failure == "providerProtocol":
                    del changed[failure]
                elif failure == "updated":
                    changed[failure] = "2026-09-02T00:00:00Z"
                elif failure == "homepage":
                    changed[failure] = "https://different.example"
                else:
                    changed[failure] = "different"
                return web.json_response(changed)
            if failure == "chunked-bytes":
                response = web.StreamResponse()
                response.enable_chunked_encoding()
                await response.prepare(request)
                await response.write(json.dumps(second).encode())
                await response.write_eof()
                return response
            return web.json_response(second)

        async with resolver_http_server(index) as url:
            await asyncio.to_thread(store.subscribe, url + "?limit=1")
            mode = "remove"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            await asyncio.to_thread(store.refresh)
            prior = store.subscriptions()[0]
            assert prior.index.entries == (
                parse_resolver_index(json.dumps(resolver_document(entry))).entries[0],
            )
            mode = "fail"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            limits: dict[str, int] = {}
            if failure == "entries":
                limits["RESOLVER_INDEX_MAX_ENTRIES"] = 1
            elif failure == "pages":
                limits["RESOLVER_FETCH_MAX_PAGES"] = 1
            elif failure in {"bytes", "chunked-bytes"}:
                limits["RESOLVER_INDEX_MAX_BYTES"] = len(json.dumps(first).encode()) + 10
            with ExitStack() as stack:
                for name, value in limits.items():
                    stack.enter_context(
                        patch("dinkster_assets.resolver_subscription." + name, value)
                    )
                (failed,) = await asyncio.to_thread(store.refresh)
            assert message in str(failed["error"])
            retained = store.subscriptions()[0]
            assert retained.checked_at == now[0]
            assert retained.refreshed_at == prior.refreshed_at
            assert retained.index == prior.index
            assert retained.etag == prior.etag
            assert provenance.sources(MODEL_DIGEST)
            assert not provenance.sources(OTHER_DIGEST)
            reloaded = ResolverSubscriptionStore(state_path, provenance, clock=lambda: now[0])
            assert reloaded.subscriptions() == store.subscriptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("late_response", ["new-snapshot", "not-modified"])
def test_refresh_deadline_releases_store_lock_and_rejects_late_results(
    tmp_path: Path, late_response: str
) -> None:
    now = [100.0]
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0], fetch_timeout=0.1)
    initial = json.dumps(resolver_document()).encode()
    with patch.object(resolver_transport, "_fetch_page", return_value=(initial, "old")):
        for source in ("https://first.example/index", "https://second.example/index"):
            store.subscribe(source)
    prior = store.subscriptions()
    now[0] += RESOLVER_REVALIDATE_SECONDS
    release = threading.Event()
    workers: list[threading.Thread] = []

    def blocked(*_args: object, **_kwargs: object) -> tuple[bytes | None, str]:
        workers.append(threading.current_thread())
        assert release.wait(3.0)
        return (
            None
            if late_response == "not-modified"
            else json.dumps(resolver_document(resolver_entry(digest=OTHER_DIGEST))).encode(),
            "late",
        )

    with patch.object(resolver_transport, "_fetch_page", side_effect=blocked):
        try:
            started = time.monotonic()
            failed = store.refresh()
            assert time.monotonic() - started < 0.75
            assert len(failed) == 2 and all(row["error"] for row in failed)
            assert len(workers) == 2 and all(worker.is_alive() for worker in workers)
            read_finished = threading.Event()

            def read_store() -> None:
                store.subscriptions()
                read_finished.set()

            reader = threading.Thread(target=read_store, daemon=True)
            reader.start()
            assert read_finished.wait(0.5)
            reader.join(0.5)
            retained = store.subscriptions()
            saved = state.read_bytes()
        finally:
            release.set()
            for worker in workers:
                worker.join(2.0)
                assert not worker.is_alive()
    assert store.subscriptions() == retained
    assert state.read_bytes() == saved
    for old, current in zip(prior, retained, strict=True):
        assert current.refreshed_at == old.refreshed_at
        assert current.index == old.index
        assert current.etag == old.etag
    assert not provenance.sources(OTHER_DIGEST)
    reloaded = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
    assert reloaded.subscriptions() == retained


def test_legacy_resolver_cache_requires_complete_unconditional_refresh(tmp_path: Path) -> None:
    async def scenario() -> None:
        entry = resolver_entry()
        mode = "initial"
        state = tmp_path / "subscriptions.json"
        now = [100.0]

        async def index(request: web.Request) -> web.Response:
            if mode == "revalidate":
                assert request.headers.get("If-None-Match") == '"whole"'
                return web.Response(status=304)
            assert "If-None-Match" not in request.headers
            if mode == "complete-single":
                return web.json_response(
                    resolver_document(entry, resolver_entry(digest=OTHER_DIGEST)),
                    headers={"ETag": '"whole"'},
                )
            if mode == "legacy-304":
                return web.Response(status=304)
            if mode == "initial":
                return web.json_response(resolver_document(entry), headers={"ETag": '"first"'})
            if "cursor" not in request.query:
                return web.json_response(
                    {**resolver_document(entry), "nextCursor": "next"},
                    headers={"ETag": '"first"'},
                )
            assert json.loads(state.read_text())["subscriptions"][0]["completeSnapshot"] is False
            if mode == "failed-page":
                return web.Response(status=400)
            return web.json_response(resolver_document(resolver_entry(digest=OTHER_DIGEST)))

        async with resolver_http_server(index) as url:
            provenance = ProvenanceStore(tmp_path / "provenance.json")
            store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            await asyncio.to_thread(store.subscribe, url)
            legacy = json.loads(state.read_text())
            del legacy["subscriptions"][0]["completeSnapshot"]
            state.write_text(json.dumps(legacy))
            store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            prior = store.subscriptions()[0]
            assert not prior.complete_snapshot
            assert provenance.sources(MODEL_DIGEST)
            assert store.public_sources(MODEL_DIGEST)
            for failure_mode in ("legacy-304", "failed-page"):
                mode = failure_mode
                now[0] += RESOLVER_REVALIDATE_SECONDS
                (failed,) = await asyncio.to_thread(store.refresh)
                assert failed["error"]
                retained = store.subscriptions()[0]
                assert retained.checked_at == now[0]
                assert retained.refreshed_at == prior.refreshed_at
                assert retained.index == prior.index
                assert not retained.complete_snapshot
                store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
                assert store.subscriptions()[0] == retained
            mode = "complete"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (refreshed,) = await asyncio.to_thread(store.refresh)
            assert not refreshed["error"]
            assert len(store.subscriptions()[0].index.entries) == 2
            assert store.subscriptions()[0].complete_snapshot
            assert store.subscriptions()[0].etag == ""
            store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            mode = "complete-single"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (refreshed,) = await asyncio.to_thread(store.refresh)
            assert not refreshed["error"]
            assert store.subscriptions()[0].etag == '"whole"'
            mode = "revalidate"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (revalidated,) = await asyncio.to_thread(store.refresh)
            assert not revalidated["error"]
            assert store.subscriptions()[0].refreshed_at == now[0]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "encoded-url",
        "url-encoding",
        "unicode-metadata",
        "metadata-encoding",
        "absent",
        "identity",
        "json",
        "save",
    ],
)
def test_resolver_api_official_bootstrap_with_fixture_provider(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    async def scenario() -> None:
        async def index(_: web.Request) -> web.Response:
            if mode == "json":
                return web.Response(text="{")
            document = resolver_document()
            if mode == "metadata-encoding":
                document["description"] = "bad\ud800"
            elif mode == "unicode-metadata":
                document["description"] = "r\u00e9solver"
            return web.json_response(document)

        state = tmp_path / "subscriptions.json"
        store = ResolverSubscriptionStore(state, ProvenanceStore(tmp_path / "provenance.json"))
        async with resolver_http_server(index) as url:
            app = create_app(make_engine, SCHEMAS)
            source = None if mode == "absent" else url
            if mode == "url-encoding":
                source = url + "?label=r\u00e9solver"
            elif mode == "encoded-url":
                source = url + "?label=r%C3%A9solver"
            add_resolver_index_routes(
                app,
                store,
                official_url=source,
                official_provider_id="wrong" if mode == "identity" else "community-models",
            )
            client = TestClient(TestServer(app))
            try:
                with ExitStack() as stack:
                    if mode == "save":
                        stack.enter_context(
                            patch.object(store, "_save", side_effect=OSError("disk full"))
                        )
                    elif mode == "url-encoding":
                        stack.enter_context(
                            patch(
                                "socket.socket.connect",
                                side_effect=AssertionError(
                                    "invalid configuration must not connect"
                                ),
                            )
                        )
                    await client.start_server()
                response = await client.get("/api/assets/resolver-indexes")
                assert response.status == 200
                subscriptions = (await response.json())["subscriptions"]
                if mode in {"valid", "encoded-url", "unicode-metadata"}:
                    (subscription,) = subscriptions
                    assert subscription["entryCount"] == 1
                    assert "official resolver bootstrap refused" not in caplog.text
                else:
                    assert not subscriptions and not state.exists()
                    assert "official resolver bootstrap refused" in caplog.text
                    if mode == "url-encoding":
                        assert "percent-encode" in caplog.text
            finally:
                await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_null", [False, True])
def test_official_bootstrap_complete_snapshot_identity(tmp_path: Path, terminal_null: bool) -> None:
    async def scenario() -> None:
        state = tmp_path / "subscriptions.json"
        provenance = ProvenanceStore(tmp_path / "provenance.json")
        now = [100.0]
        mode = "initial"
        calls: list[str | None] = []
        store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])

        async def index(request: web.Request) -> web.Response:
            assert not any(
                name in request.headers
                for name in ("Authorization", "Cookie", "Proxy-Authorization")
            )
            assert "If-None-Match" not in request.headers
            cursor = request.query.get("cursor")
            calls.append(cursor)
            document = resolver_document()
            document["entries"] = [resolver_entry()] if mode == "initial" and cursor else []
            if not cursor:
                document["nextCursor"] = "last"
            elif terminal_null:
                document["nextCursor"] = None
            if mode == "wrong-identity":
                document["name"] = "other-provider"
            if mode == "initial":
                assert not state.exists()
            return web.json_response(document, headers={"ETag": '"page"', "Set-Cookie": "x=y"})

        async with resolver_http_server(index) as url:
            subscription = await asyncio.to_thread(
                store.bootstrap_official, url, "community-models"
            )
            assert subscription is not None
            assert subscription.complete_snapshot and subscription.etag == ""
            assert calls == [None, "last"]
            assert provenance.sources(MODEL_DIGEST)
            saved = state.read_bytes()
            store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            assert (
                await asyncio.to_thread(store.bootstrap_official, url, "community-models")
                == subscription
            )
            assert calls == [None, "last"]
            assert state.read_bytes() == saved

            mode = "wrong-identity"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (failed,) = await asyncio.to_thread(store.refresh)
            assert "identity" in str(failed["error"])
            retained = store.subscriptions()[0]
            assert retained.refreshed_at == subscription.refreshed_at
            assert retained.index == subscription.index
            assert provenance.sources(MODEL_DIGEST)

            mode = "empty"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            await asyncio.to_thread(store.refresh)
            assert not provenance.sources(MODEL_DIGEST)
            store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            assert not store.subscriptions()[0].index.entries

            corrupt = json.loads(state.read_bytes())
            corrupt["officialBootstrap"]["providerId"] = "other-provider"
            write_document(state, corrupt)
            with pytest.raises(ResolverSubscriptionError, match="identity"):
                ResolverSubscriptionStore(state, provenance)

    asyncio.run(scenario())
