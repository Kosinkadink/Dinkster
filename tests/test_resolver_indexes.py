"""Declarative resolver-index parsing, subscriptions, and API wiring."""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import ssl
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    RESOLVER_REVALIDATE_SECONDS,
    AssetNeed,
    AssetVault,
    LibraryStore,
    ProvenanceRecord,
    ProvenanceStore,
    ResolverIndexError,
    ResolverSubscriptionError,
    ResolverSubscriptionStore,
    acquire_need,
    digest_bytes,
    parse_resolver_index,
)
from dinkster_assets import resolver_subscription as resolver_transport
from dinkster_server import RuntimeSettings, ServerLibrary, create_app
from test_server import SCHEMAS, make_engine

from dinkster.guess_api import add_guess_routes
from dinkster.resolver_api import add_resolver_index_routes
from tests.async_scenarios import run_scenario

MODEL_BYTES = b"resolver-index model bytes" * 32
MODEL_DIGEST = digest_bytes(MODEL_BYTES)
OTHER_BYTES = b"updated resolver-index model bytes" * 32
OTHER_DIGEST = digest_bytes(OTHER_BYTES)


def resolver_settings() -> RuntimeSettings:
    return RuntimeSettings(
        {
            "memory-budgets": {},
            "memory-headroom": 256 * 1024**2,
            "aimdo-policy": "auto",
            "dtype-policy": {"diffusion": "auto", "textEncoder": "auto", "vae": "auto"},
            "fp8-matmul": False,
            "worker-comfy-args": (),
            "jobs": {"maxRunningJobs": 1},
            "logging": {"level": "info", "overrides": {}},
        },
        {
            category: "default"
            for category in (
                "memory-budgets",
                "memory-headroom",
                "aimdo-policy",
                "dtype-policy",
                "fp8-matmul",
                "worker-comfy-args",
                "jobs",
                "logging",
            )
        },
        features={"resolver": {"enabled": True}},
    )


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

    run_scenario(scenario())


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


@pytest.mark.parametrize("part", ["headers", "chunk-framing", "body"])
def test_refresh_deadline_interrupts_trickling_http_response(tmp_path: Path, part: str) -> None:
    async def scenario() -> None:
        finished = asyncio.Event()
        sent = 0

        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal sent
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
                        sent += 1
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
            with pytest.raises(ResolverSubscriptionError):
                await asyncio.to_thread(store.subscribe, f"http://127.0.0.1:{port}/index")
            await asyncio.wait_for(finished.wait(), 1.0)
            assert 0 < sent < 100
            assert store.subscriptions() == ()

    run_scenario(scenario())


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
            with pytest.raises(ResolverSubscriptionError, match="time limit"):
                resolver_transport._fetch_url(url, "", timeout=0.05)
            blocked_phase.assert_called_once()
            assert len(workers) == 1
            assert workers[0].is_alive() and workers[0].daemon
        finally:
            release.set()


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
        app = create_app(make_engine, SCHEMAS, library=library, settings=resolver_settings())
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

    run_scenario(scenario())


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
        app = create_app(make_engine, SCHEMAS, settings=resolver_settings())
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

    run_scenario(scenario())


def test_resolver_api_startup_refresh_persistence_failure_is_nonfatal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, settings=resolver_settings())
        add_resolver_index_routes(app, store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/assets/resolver-indexes")
            assert response.status == 200
        finally:
            await client.close()

    with patch.object(store, "refresh", side_effect=OSError("disk full")):
        run_scenario(scenario())
    assert "resolver index startup refresh failed: disk full" in caplog.text


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

    run_scenario(scenario())


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

    run_scenario(scenario())


@pytest.mark.parametrize("terminal_null", [False, True])
def test_cursor_refresh_refetches_all_pages_despite_unchanged_first_page_etag(
    tmp_path: Path, terminal_null: bool
) -> None:
    async def scenario() -> None:
        now = [100.0]
        entries = [
            {**resolver_entry(digest=digest_bytes(content), size=len(content)), "gated": True}
            for content in (MODEL_BYTES, OTHER_BYTES, b"gone")
        ]
        mode = "initial"
        requests: list[tuple[str, str | None]] = []
        saved = b""
        cursor = "opaque+/=&?%# cursor"
        query = "limit=1&search=a%2fb+z&tag=a&tag=b&blank=&flag"
        state = tmp_path / "subscriptions.json"
        provenance = ProvenanceStore(tmp_path / "provenance.json")
        store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])

        async def index(request: web.Request) -> web.Response:
            assert "Authorization" not in request.headers
            assert "Cookie" not in request.headers
            assert "Proxy-Authorization" not in request.headers
            assert request.headers["User-Agent"] == "dinkster-resolver-index"
            requests.append((request.raw_path, request.headers.get("If-None-Match")))
            if mode == "not-modified" or request.headers.get("If-None-Match") == '"first"':
                assert "cursor" not in request.query
                return web.Response(status=304, headers={"ETag": '"first"'})
            if "cursor" in request.query:
                assert request.query["cursor"] == cursor
                assert request.headers.get("If-None-Match") is None
                if mode == "initial":
                    assert not state.exists()
                    page_entries = entries[1:]
                else:
                    assert state.read_bytes() == saved
                    assert provenance.sources(str(entries[2]["digest"]))
                    page_entries = entries[1:2]
                terminal = {**resolver_document(*page_entries), "providerProtocol": 1}
                if terminal_null:
                    terminal["nextCursor"] = None
                return web.json_response(terminal, headers={"ETag": '"last"'})
            return web.json_response(
                {
                    **resolver_document(entries[0]),
                    "providerProtocol": 1,
                    "nextCursor": cursor,
                },
                headers={"ETag": '"first"', "Set-Cookie": "session=do-not-forward"},
            )

        async with resolver_http_server(index) as url:
            subscription = await asyncio.to_thread(
                store.subscribe, url + "?" + query + "&cursor=stale&%63ursor=also-stale"
            )
            assert len(subscription.index.entries) == 3
            assert subscription.etag == ""
            saved = state.read_bytes()
            mode = "remove"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (refreshed,) = await asyncio.to_thread(store.refresh)
            assert refreshed["error"] == ""
            after = store.subscriptions()[0]
            assert after.refreshed_at == now[0]
            assert [row.digest for row in after.index.entries] == [
                str(row["digest"]) for row in entries[:2]
            ]
            assert not provenance.sources(str(entries[2]["digest"]))
            assert "nextCursor" not in state.read_text()
            reloaded = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
            assert reloaded.subscriptions()[0].etag == ""
            assert reloaded.subscriptions()[0] == after
            mode = "not-modified"
            now[0] += RESOLVER_REVALIDATE_SECONDS
            (failed,) = await asyncio.to_thread(reloaded.refresh)
            assert "not-modified" in str(failed["error"])
            assert reloaded.subscriptions()[0].index == after.index
            assert reloaded.subscriptions()[0].refreshed_at == after.refreshed_at
            first_path = "/dinkster/v1/export/resolver?" + query
            second_path = first_path + "&cursor=" + quote(cursor, safe="")
            assert requests == [
                (first_path, None),
                (second_path, None),
                (first_path, None),
                (second_path, None),
                (first_path, None),
            ]

    run_scenario(scenario())


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

    run_scenario(scenario())


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
            app = create_app(make_engine, SCHEMAS, settings=resolver_settings())
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

    run_scenario(scenario())


@pytest.mark.parametrize(
    "license_text",
    ["", " ", " custom license ", *["x" * n for n in (256, 300, 512, 513, 4096)]],
    ids=["empty", "space", "custom-padded", "256", "300", "512", "513", "4096"],
)
def test_license_metadata_survives_refresh_and_reload(tmp_path: Path, license_text: str) -> None:
    source = tmp_path / "resolver.json"
    state = tmp_path / "subscriptions.json"
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    now = [100.0]
    entry = {**resolver_entry(), "license": license_text, "gated": True}
    write_document(source, resolver_document(entry))
    store = ResolverSubscriptionStore(state, provenance, clock=lambda: now[0])
    subscription = store.subscribe(str(source))
    assert subscription.index.entries[0].license == license_text
    assert subscription.index.entries[0].gated
    assert ResolverSubscriptionStore(state, provenance).subscriptions() == (subscription,)
    for value in ("other metadata", license_text):
        entry["license"] = value
        write_document(source, resolver_document(entry))
        now[0] += 1
        store.refresh()
        current = store.subscriptions()[0]
        assert current.index.entries[0].license == value
        assert current.index.entries[0].gated
        assert ResolverSubscriptionStore(state, provenance).subscriptions() == (current,)
