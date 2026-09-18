"""M6: the shared remote cache - one instance's computed entries are
another's hits.

Instance A persists cache entries in a DiskCacheStore and serves them over
the cache-sharing endpoints (read-only pull). Instance B composes a
PeerCacheStore behind its local layers: a peer hit is verified against its
digests, rehydrated through the same wire code a disk hit uses, promoted
onto B's local layers, and keys identically because fingerprints travel
verbatim (hazard H4). Every failure - miss, tamper, malformed digest - is
conservative: a miss, never a corrupt Value.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_engine import Engine
from dinkster_schema import build_schemas
from dinkster_server import PeerCacheStore, create_app
from dinkster_values import TypeRegistry, register_core_types
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types
from test_cas import PoisonWorker
from test_engine import InProcessWorkerFactory, image_graph
from test_server import SCHEMAS, make_engine


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    return registry


async def serve_cache(store: DiskCacheStore) -> TestServer:
    """A live instance A: real app, its disk store exported."""
    server = TestServer(create_app(make_engine, SCHEMAS, cache_export=store))
    await server.start_server()
    return server


def test_cache_endpoints(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("known-key", {"out": registry.wrap("core.int", 5)})
        server = await serve_cache(store)
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.get(server.make_url("/cache/entry/known-key"))
                assert resp.status == 200
                manifest = await resp.json()
                assert manifest["key"] == "known-key"
                digest = manifest["outputs"]["out"]["digest"]

                resp = await http.get(server.make_url("/cache/entry/unknown"))
                assert resp.status == 404

                resp = await http.get(server.make_url(f"/cache/cas/{digest}"))
                assert resp.status == 200
                assert (await resp.read()) == b"json:5"

                resp = await http.get(server.make_url("/cache/cas/not-a-digest"))
                assert resp.status == 400

                resp = await http.get(server.make_url("/cache/cas/blake3:" + "0" * 64))
                assert resp.status == 404
        finally:
            await server.close()

    asyncio.run(scenario())


def test_endpoints_absent_without_export() -> None:
    async def scenario() -> None:
        server = TestServer(create_app(make_engine, SCHEMAS))
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.get(server.make_url("/cache/entry/x"))
                assert resp.status == 404  # route not registered
        finally:
            await server.close()

    asyncio.run(scenario())


def test_peer_cache_store_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        value = registry.wrap("core.string", "computed on A")
        await store.put("shared-key", {"out": value})
        server = await serve_cache(store)
        try:
            async with PeerCacheStore(str(server.make_url("")), make_registry()) as peer:
                hit = await peer.get("shared-key")
                assert hit is not None
                assert hit["out"].fingerprint == value.fingerprint  # H4
                assert hit["out"].resolve() == "computed on A"
                assert peer.hits == 1

                assert await peer.get("never-computed") is None
                assert peer.misses == 1

                await peer.put("shared-key", hit)  # read-only: a no-op
        finally:
            await server.close()

    asyncio.run(scenario())


class TamperingExport:
    """A peer whose blob bytes do not match their digests."""

    def __init__(self, inner: DiskCacheStore) -> None:
        self._inner = inner

    async def entry_wire(self, key: str) -> Mapping[str, Any] | None:
        return await self._inner.entry_wire(key)

    async def blob(self, digest: str) -> bytes | None:
        data = await self._inner.blob(digest)
        return None if data is None else data + b"!"


def test_peer_cache_store_rejects_tampered_bytes(tmp_path: Path) -> None:
    """Bytes from the wire are only trusted to be what they claim: a blob
    that does not hash to its manifest digest is a miss, never a Value."""

    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key", {"out": registry.wrap("core.int", 5)})
        server = TestServer(create_app(make_engine, SCHEMAS, cache_export=TamperingExport(store)))
        await server.start_server()
        try:
            async with PeerCacheStore(str(server.make_url("")), make_registry()) as peer:
                assert await peer.get("key") is None
                assert peer.misses == 1
        finally:
            await server.close()

    asyncio.run(scenario())


def test_peer_down_is_a_miss() -> None:
    async def scenario() -> None:
        async with PeerCacheStore(
            "http://127.0.0.1:9", make_registry(), request_timeout=2.0
        ) as peer:
            assert await peer.get("any-key") is None  # conservative, no raise

    asyncio.run(scenario())


def test_second_instance_hits_shared_cache(tmp_path: Path) -> None:
    """The demo that matters: instance A runs the graph; instance B - a
    different process's worth of state, a worker that cannot execute at all
    - runs the same graph entirely from A's cache, and the entry it pulled
    is promoted onto B's local layers so the network is crossed once."""

    async def scenario() -> None:
        schemas = build_schemas(SCAFFOLD_NODES)

        # Instance A computes and persists.
        registry_a = make_registry()
        store_a = DiskCacheStore(tmp_path / "a", registry_a)
        engine_a = Engine(
            schemas=schemas,
            registry=registry_a,
            worker=InProcessWorkerFactory(registry_a),
            cache=store_a,
        )
        result_a = await engine_a.run(image_graph(), ["s"])
        assert set(result_a.executed) == {"g", "i", "b", "s"}

        server = await serve_cache(store_a)
        try:
            # Instance B: fresh registry, local layers, A as the last resort.
            registry_b = make_registry()
            memory_b = MemoryLRUCache()
            disk_b = DiskCacheStore(tmp_path / "b", registry_b)
            peer = PeerCacheStore(str(server.make_url("")), registry_b)
            engine_b = Engine(
                schemas=schemas,
                registry=registry_b,
                worker=PoisonWorker(),
                cache=LayeredCache(memory_b, disk_b, peer),
            )
            try:
                result_b = await engine_b.run(image_graph(), ["s"])
                assert result_b.executed == ()
                assert set(result_b.cached) == {"g", "i", "b", "s"}
                assert result_b.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
                assert peer.hits == 4

                # Promotion: the second run never leaves this process.
                again = await engine_b.run(image_graph(), ["s"])
                assert set(again.cached) == {"g", "i", "b", "s"}
                assert peer.hits == 4  # unchanged - local layers answered
            finally:
                await peer.close()
        finally:
            await server.close()

    asyncio.run(scenario())
