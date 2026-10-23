"""Runtime expansion uses the same invocation and value ownership on every worker."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_memory import (
    GovernorReservationService,
    MemoryGovernor,
    ReservationRequest,
)
from dinkster_protocol import (
    BlockedOutput,
    CachedResult,
    CurrentNodeRef,
    CurrentOutputRef,
    ExpandedReturn,
    Invocation,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    PresentOutput,
    Worker,
)
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr, build_node_types
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    TypeRegistry,
    Value,
    ValueMeta,
    register_core_types,
)
from dinkster_workers import GovernedWorker, InProcessWorker, IsolatedWorker, RemoteWorker


class WireExpansionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "expansion.test",
            outputs=(
                OutputSpec("result", TypeExpr.concrete("core.int")),
                OutputSpec("seed", TypeExpr.concrete("core.int")),
            ),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls) -> Any:
        values = TypeRegistry()
        register_core_types(values)
        return ExpandedReturn(
            LocalExpansion(
                (LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),)
            ),
            (
                ("result", LocalOutputRef("child", "value")),
                ("seed", PresentOutput(values.wrap("core.int", 17))),
            ),
        )


PACK_NODES = [WireExpansionNode]


@pytest.mark.parametrize(
    ("worker_kind", "use_shm"),
    [
        ("in-process", False),
        ("governed", False),
        ("isolated", False),
        ("isolated", True),
        ("remote", False),
    ],
)
def test_expansion_result_survives_worker_round_trip(
    tmp_path: Path, worker_kind: str, use_shm: bool
) -> None:
    from test_remote import TOKEN, start_service, stop_service

    async def scenario() -> None:
        values = TypeRegistry()
        register_core_types(values)
        boundary_worker: IsolatedWorker | RemoteWorker | None = None
        service: asyncio.subprocess.Process | None = None
        worker: Worker
        if worker_kind in ("isolated", "remote"):
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "expansion"\nnamespaces = ["expansion"]\n'
                '[pack.entry]\nnodes = "tests.test_runtime_expansion:PACK_NODES"\n',
                encoding="ascii",
            )
            root = Path(__file__).resolve().parent.parent
            if worker_kind == "remote":
                service, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
                boundary_worker = RemoteWorker(host, port, TOKEN, values, name="expansion")
            else:
                boundary_worker = IsolatedWorker(
                    manifest,
                    values,
                    use_shm=use_shm,
                    shm_threshold=1,
                    extra_env={"PYTHONPATH": str(root)},
                )
            worker = boundary_worker
        else:
            worker = InProcessWorker(build_node_types(PACK_NODES), values)
            if worker_kind == "governed":
                worker = GovernedWorker(
                    worker,
                    planner=lambda _: (ReservationRequest("ram", 1),),
                    reservations=GovernorReservationService(MemoryGovernor({"ram": 16})),
                )
        try:
            if boundary_worker is not None:
                await boundary_worker.start()
            for index in range(2):
                result = await worker.invoke(
                    Invocation(
                        f"invocation-{index}",
                        "parent",
                        "expansion.test",
                        {},
                        WireExpansionNode.schema(),
                        job_ref="run",
                    )
                )
                assert result.error is None
                assert result.outputs is not None
                assert result.outputs["seed"].resolve() == 17
                assert result.expansion is not None
                bindings = dict(result.expansion.bindings)
                assert bindings["result"] == LocalOutputRef("child", "value")
                assert isinstance(bindings["seed"], PresentOutput)
                assert bindings["seed"].value is result.outputs["seed"]
                assert result.expansion.expansion.nodes == (
                    LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),
                )
        finally:
            try:
                if boundary_worker is not None:
                    await boundary_worker.close()
            finally:
                if service is not None:
                    await stop_service(service)

    asyncio.run(scenario())


def cached_expansion(seed: Value) -> CachedResult:
    registry = TypeRegistry()
    register_core_types(registry)
    other = registry.wrap("core.int", -4)
    return CachedResult(
        {"seed": seed, "other": other},
        ExpandedReturn(
            LocalExpansion(
                (
                    LocalNode(
                        "child",
                        "expansion.child",
                        (("value", CurrentOutputRef("seed")),),
                        parent=CurrentNodeRef(),
                        display=LocalNodeRef("display"),
                    ),
                    LocalNode("display", "expansion.display"),
                )
            ),
            (
                ("blocked", BlockedOutput("stop here")),
                ("other", PresentOutput(other)),
                ("result", LocalOutputRef("child", "value")),
                ("seed", PresentOutput(seed)),
            ),
        ),
    )


@pytest.mark.parametrize("cache_kind", ["memory", "disk", "layered", "spill", "peer"])
def test_cached_expansion_survives_persistence_and_promotion(
    tmp_path: Path, cache_kind: str
) -> None:
    from dinkster_server import PeerCacheStore
    from test_cache_share import serve_cache

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        original = cached_expansion(registry.wrap("core.int", 17))
        memory = MemoryLRUCache(max_bytes=1 if cache_kind == "spill" else 1024)
        disk = DiskCacheStore(tmp_path / "disk", registry)
        server = None
        peer = None
        try:
            if cache_kind == "memory":
                await memory.put("parent", original)
                cache = memory
            elif cache_kind == "disk":
                await disk.put("parent", original)
                cache = DiskCacheStore(tmp_path / "disk", registry)
            elif cache_kind == "peer":
                await disk.put("parent", original)
                server = await serve_cache(disk)
                peer = PeerCacheStore(str(server.make_url("")), registry)
                cache = LayeredCache(memory, DiskCacheStore(tmp_path / "local", registry), peer)
            elif cache_kind == "spill":
                costed = replace(original["seed"], meta=ValueMeta({COST_META_KEY: {"ram": 50}}))
                original = CachedResult({**original, "seed": costed}, original.expansion)
                cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
                await cache.put("parent", original)
                assert len(memory) == 0
            else:
                await disk.put("parent", original)
                cache = LayeredCache(memory, DiskCacheStore(tmp_path / "disk", registry))

            for _ in range(2):
                hit = await cache.get("parent")
                assert hit is not None and hit.expansion is not None
                assert set(hit) == {"seed", "other"}
                assert hit["seed"].resolve() == 17
                assert hit["other"].resolve() == -4
                bindings = dict(hit.expansion.bindings)
                assert bindings["blocked"] == BlockedOutput("stop here")
                assert bindings["result"] == LocalOutputRef("child", "value")
                assert isinstance(bindings["seed"], PresentOutput)
                assert bindings["seed"].value is hit["seed"]
                assert original.expansion is not None
                assert hit.expansion.expansion == original.expansion.expansion
            if peer is not None:
                assert peer.hits == 1
                restarted = await DiskCacheStore(tmp_path / "local", registry).get("parent")
                assert restarted is not None and restarted.expansion is not None
                assert restarted["seed"].resolve() == 17
        finally:
            if peer is not None:
                await peer.close()
            if server is not None:
                await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "damage", ["document", "missing-output", "extra-output", "missing-field", "index"]
)
def test_incomplete_cached_expansion_is_a_whole_entry_miss(tmp_path: Path, damage: str) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("parent", cached_expansion(registry.wrap("core.int", 17)))
        path = next((tmp_path / "entries").glob("*.json"))
        manifest = json.loads(path.read_text())
        if damage == "document":
            manifest["expansion"]["expansion"]["nodes"][0]["nodeType"] = "bad type"
        elif damage == "missing-output":
            del manifest["outputs"]["seed"]
        elif damage == "extra-output":
            manifest["outputs"]["extra"] = manifest["outputs"]["seed"]
        elif damage == "index":
            manifest["expansion"]["bindings"][3]["value"]["valueIndex"] = True
        else:
            del manifest["expansion"]
        path.write_text(json.dumps(manifest))
        cache = LayeredCache(MemoryLRUCache(), DiskCacheStore(tmp_path, registry))
        assert await cache.get("parent") is None
        assert not path.exists()

    asyncio.run(scenario())


def test_cached_expansion_accounts_and_invalidates_present_values_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        seed = registry.wrap("core.int", 17)
        seed = replace(seed, meta=ValueMeta({COST_META_KEY: {"ram": 50}}))
        memory = MemoryLRUCache(max_bytes=100)
        disk = DiskCacheStore(tmp_path, registry)
        cache = LayeredCache(memory, disk)
        await cache.put("parent", cached_expansion(seed))
        assert memory.footprint("ram") == 50
        resident = replace(seed, meta=ValueMeta({RESOURCE_ID_META_KEY: "resident"}))
        await cache.put("parent", cached_expansion(resident))
        assert memory.footprint("ram") == 0
        assert await disk.get("parent") is None
        assert cache.drop_referencing("resident") == 1
        assert await cache.get("parent") is None

    asyncio.run(scenario())
