"""M3: the isolated-worker memory relay. A governor in the engine process
governs a ResidentPool living in a worker child: footprints and details
arrive as pushed snapshots, vram pressure crosses as a live shed round
trip, and ram pressure crosses only through the two-phase release gate -
pins, cache invalidation, in-flight holds, use tokens - and returns zero
when no ReleaseGuard is wired (DESIGN 3.10)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import ResidentPool, resident_resource_id
from dinkster_engine import Engine, Invocation
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import (
    FullReleaseCommitResult,
    FullReleaseResult,
    GovernorReservationService,
    MemoryGovernor,
    PressureSignal,
    ReleaseCandidate,
)
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import ResourcePins, TypeRegistry, register_core_types
from dinkster_workers import DeviceMap, GroupIsolatedWorker, IsolatedWorker, ReleaseGuard
from dinkster_workers.boundary import ValueCodec, read_frame, write_frame
from dinkster_workers.execution import current_execution_context
from dinkster_workers.host import serve_connection
from dinkster_workers.in_process import InProcessWorker
from dinkster_workers.relay import MemoryRelay
from dinkster_workers.session import BoundarySession, WorkerDied

TESTS_DIR = Path(__file__).parent

VRAM0 = "vram:cuda:0"
VRAM1 = "vram:cuda:1"
CONSUMER = "memorypack:models"


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


@asynccontextmanager
async def hosted_pool(pool: ResidentPool, guard: ReleaseGuard) -> AsyncIterator[BoundarySession]:
    registry = core_registry()
    accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        accepted.set_result(task)
        await serve_connection(
            reader,
            writer,
            pack_name="pool",
            worker=InProcessWorker({}, registry),
            schemas={},
            planner=None,
            consumers={"pool": pool},
            codec=ValueCodec(registry),
        )

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(address[0], address[1])
    host_task = await accepted
    parent_registry = core_registry()
    session = BoundarySession(
        parent_registry,
        role="test",
        pack="pool",
        codec=ValueCodec(parent_registry),
        release_guard=guard,
    )
    try:
        await session.begin(reader, writer, timeout=2)
        yield session
    finally:
        with suppress(Exception):
            await session.send({"type": "shutdown"}, [])
        try:
            await session.close()
            await asyncio.wait_for(host_task, 2)
        finally:
            server.close()
            await server.wait_closed()


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
        'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
        'consumers = "memorypack_nodes:memory_consumers"\n'
    )
    return manifest


def write_reserved_manifest(tmp_path: Path) -> Path:
    manifest = write_manifest(tmp_path)
    with manifest.open("a") as stream:
        stream.write('reservations = "dinkster_compat_comfy.memory:plan_reservations"\n')
    return manifest


def relay_worker(
    tmp_path: Path, registry: TypeRegistry, governor: MemoryGovernor, **kwargs: object
) -> IsolatedWorker:
    return IsolatedWorker(
        write_manifest(tmp_path),
        registry,
        extra_env={"PYTHONPATH": str(TESTS_DIR)},
        governor=governor,
        **kwargs,  # type: ignore[arg-type]
    )


def make_engine(
    registry: TypeRegistry,
    worker: IsolatedWorker,
    *,
    cache: MemoryLRUCache | None = None,
    pins: ResourcePins | None = None,
) -> Engine:
    return Engine(
        schemas=dict(worker.schemas),
        registry=registry,
        worker=worker,
        cache=cache if cache is not None else MemoryLRUCache(),
        pins=pins,
    )


def load_graph(name: str = "sd15.safetensors", vram: int = 400, ram: int = 300) -> Graph:
    return Graph(
        nodes={
            "load": GraphNode("mem.load", {"name": name, "vram": vram, "ram": ram}),
        }
    )


async def eventually(predicate: Callable[[], bool], timeout: float = 8.0) -> bool:
    """Snapshot pushes race the invocation result; poll for the frame."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def consumer_footprint(governor: MemoryGovernor, device: str) -> int:
    consumers = governor.status().get(device, {}).get("consumers", {})
    return int(consumers.get(CONSUMER, 0))  # type: ignore[union-attr, arg-type]


def test_relay_announces_reports_and_details(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            # Registered from the hello handshake, before any invocation:
            # the consumer's name is discoverable through status.
            assert CONSUMER in governor.status()[VRAM0]["consumers"]  # type: ignore[operator]

            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400, ram=300), ["load"])

            # The post-invocation report lands: declared costs on both lanes.
            assert await eventually(lambda: consumer_footprint(governor, VRAM0) == 400)
            assert consumer_footprint(governor, "ram") == 300

            # The detail contract crossed too: stable ID, human name.
            items = governor.details()[CONSUMER]
            assert len(items) == 1
            assert items[0].display_name == "sd15.safetensors"
            assert items[0].bytes_by_residency[VRAM0] == 400
            assert items[0].bytes_by_residency["ram"] == 300
        finally:
            await worker.close()
        # Close unregisters the proxies: the dead worker's bytes are gone
        # from the governor's view, not frozen at their last report.
        assert CONSUMER not in governor.status()[VRAM0]["consumers"]  # type: ignore[operator]

    asyncio.run(scenario())


def test_vram_shed_crosses_and_rescores_immediately(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            result = await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: consumer_footprint(governor, VRAM0) == 400)

            freed = await governor.shed(VRAM0, 400)
            assert freed == 400
            # The shed reply carried a fresh snapshot, applied before the
            # proxy returned: admission rescoring sees the freed bytes now,
            # not at the next push.
            assert consumer_footprint(governor, VRAM0) == 0
            # Advisory unloading only: the host/offload copy stays.
            assert consumer_footprint(governor, "ram") == 300

            # Identity survived - the stub still resolves in the child.
            model = result.outputs["load"]["model"]
            use = Graph(
                nodes={
                    "load": GraphNode("mem.load", {"name": "sd15.safetensors"}),
                    "use": GraphNode("mem.use", {"model": Link("load", "model")}),
                }
            )
            assert model.fingerprint.startswith("resident:")
            rerun = await engine.run(use, ["use"])
            assert rerun.outputs["use"]["name"].resolve() == "sd15.safetensors"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_compat_vram_reservation_gates_isolated_admission(tmp_path: Path) -> None:
    """The compat planner's child-local request crosses the real lease
    protocol and is refused by a parent budget smaller than the model."""

    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 300, "ram": 1000})
        registry = core_registry()
        worker = IsolatedWorker(
            write_reserved_manifest(tmp_path),
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            loaded = await engine.run(load_graph(vram=400, ram=300), ["load"])
            model = loaded.outputs["load"]["model"]
            result = await worker.invoke(
                Invocation(
                    invocation_id="use-denied",
                    node_id="use",
                    node_type="mem.use",
                    inputs={"model": model},
                    effective_schema=worker.schemas["mem.use"],
                )
            )
            assert result.error is not None
            assert "memory admission failed" in result.error.message
            assert governor.reserved(VRAM0) == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_ram_pressure_is_gated_at_the_proxy(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: consumer_footprint(governor, "ram") == 300)

            # ram release would drop the child's strong reference while this
            # process may still hold stubs - the proxy refuses (DESIGN 3.10),
            # and nothing in the child changed.
            freed = await governor.shed("ram", 300)
            assert freed == 0
            assert consumer_footprint(governor, "ram") == 300
            assert consumer_footprint(governor, VRAM0) == 400
        finally:
            await worker.close()

    asyncio.run(scenario())


def wire_gate(cache: MemoryLRUCache) -> tuple[ResourcePins, ReleaseGuard]:
    """The full gate composition (DESIGN 3.10): one ResourcePins shared by the
    engine and the guard, and every parent cache as an invalidator."""
    pins = ResourcePins()
    return pins, ReleaseGuard(pins=pins, invalidators=(cache.drop_referencing,))


def test_ram_release_crosses_the_gate_and_recomputes(tmp_path: Path) -> None:
    """With the guard wired and no run holding pins, --cache-ram works
    across the boundary: the child's strong reference drops, and the next
    run recomputes through an honest cache miss - never a dangling stub."""

    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        cache = MemoryLRUCache()
        pins, guard = wire_gate(cache)
        worker = relay_worker(tmp_path, registry, governor, release_guard=guard)
        await worker.start()
        try:
            engine = make_engine(registry, worker, cache=cache, pins=pins)
            await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: consumer_footprint(governor, "ram") == 300)

            freed = await governor.shed("ram", 300)
            assert freed == 300
            # The result frame's snapshot arrived with the reply: releasing
            # drops the resident outright, every lane included.
            assert consumer_footprint(governor, "ram") == 0
            assert consumer_footprint(governor, VRAM0) == 0

            # The gate invalidated the parent cache before the child
            # dropped: rerunning recomputes the loader and works.
            use = Graph(
                nodes={
                    "load": GraphNode("mem.load", {"name": "sd15.safetensors"}),
                    "use": GraphNode("mem.use", {"model": Link("load", "model")}),
                }
            )
            rerun = await engine.run(use, ["use"])
            assert "load" in rerun.executed  # recompute, not a stale stub
            assert rerun.outputs["use"]["name"].resolve() == "sd15.safetensors"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_full_release_visits_all_consumers_and_preserves_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        cache = MemoryLRUCache()
        pins, guard = wire_gate(cache)
        worker = relay_worker(tmp_path, registry, governor, release_guard=guard)
        await worker.start()
        try:
            engine = make_engine(registry, worker, cache=cache, pins=pins)
            await engine.run(load_graph(vram=0, ram=0), ["load"])
            token = worker.instance_token
            assert token is not None
            assert len(cache) == 1
            await asyncio.sleep(0.05)

            results = await worker.full_release("maintenance-1", token)
            assert results.status == "complete"
            assert results.consumers == (
                {"consumer": "models", "status": "complete"},
                {"consumer": "zero-cost", "status": "complete"},
            )
            assert len(cache) == 0
            assert worker.instance_token == token

            use = Graph(
                nodes={
                    "load": GraphNode("mem.load", {"name": "sd15.safetensors"}),
                    "use": GraphNode("mem.use", {"model": Link("load", "model")}),
                }
            )
            rerun = await engine.run(use, ["use"])
            assert "load" in rerun.executed
            assert rerun.outputs["use"]["name"].resolve() == "sd15.safetensors"
            assert worker.instance_token == token

            await asyncio.sleep(0.05)
            repeated = await worker.full_release("maintenance-2", token)
            assert repeated.status == "complete"
            assert all(result["status"] == "complete" for result in repeated.consumers)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_consumer_factory_is_enumerated_once_and_shared_with_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = relay_worker(
            tmp_path,
            registry,
            MemoryGovernor({"ram": 1000}),
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await worker.start()
        try:
            cache = MemoryLRUCache()
            engine = make_engine(registry, worker, cache=cache)
            graph = Graph(nodes={"state": GraphNode("mem.consumer_state", {})})
            before = await engine.run(graph, ["state"])
            assert before.outputs["state"]["created"].resolve() == 1
            assert before.outputs["state"]["live"].resolve() == 1
            token = worker.instance_token
            assert token is not None
            await asyncio.sleep(0.05)
            result = await worker.full_release("factory", token)
            assert result.status == "complete"
            cache.clear()
            after = await engine.run(graph, ["state"])
            assert after.outputs["state"]["created"].resolve() == 1
            assert after.outputs["state"]["live"].resolve() == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_full_release_acknowledges_worker_with_no_consumers(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
        )
        worker = IsolatedWorker(
            manifest,
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await worker.start()
        try:
            token = worker.instance_token
            assert token is not None
            result = await worker.full_release("maintenance-empty", token)
            assert result.status == "complete"
            assert result.consumers == ()
            assert worker.alive
            assert worker.instance_token == token
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_full_release_refuses_active_worker_invocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 1000})
        registry = core_registry()
        worker = relay_worker(
            tmp_path,
            registry,
            governor,
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            running = asyncio.create_task(
                engine.run(
                    Graph(nodes={"hold": GraphNode("mem.hold", {"milliseconds": 200})}),
                    ["hold"],
                )
            )
            await asyncio.sleep(0.05)
            token = worker.instance_token
            assert token is not None
            results = await worker.full_release("maintenance-active", token)
            assert results.status == "busy"
            assert results.consumers == (
                {"consumer": "models", "status": "busy"},
                {"consumer": "zero-cost", "status": "busy"},
            )
            assert (await running).outputs["hold"]["done"].resolve() == "done"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_full_release_refuses_pinned_resident(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        cache = MemoryLRUCache()
        pins, guard = wire_gate(cache)
        worker = relay_worker(tmp_path, registry, governor, release_guard=guard)
        await worker.start()
        try:
            engine = make_engine(registry, worker, cache=cache, pins=pins)
            await engine.run(load_graph(), ["load"])
            assert await eventually(lambda: len(governor.details().get(CONSUMER, ())) == 1)
            resource_id = resident_resource_id(governor.details()[CONSUMER][0].item_id)
            assert pins.pin(resource_id)
            token = worker.instance_token
            assert token is not None
            await asyncio.sleep(0.05)

            results = await worker.full_release("maintenance-busy", token)
            assert results.status == "complete"
            assert results.consumers[0] == {"consumer": "models", "status": "busy"}
            assert results.consumers[1]["status"] == "complete"
            assert not pins.condemned(resource_id)
            assert consumer_footprint(governor, "ram") == 300

            pins.unpin(resource_id)
            results = await worker.full_release("maintenance-complete", token)
            assert results.status == "complete"
            assert all(result["status"] == "complete" for result in results.consumers)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_full_release_covers_two_live_worker_processes(tmp_path: Path) -> None:
    async def scenario() -> None:
        workers: list[tuple[IsolatedWorker, Engine, str]] = []
        for index in range(2):
            governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
            registry = core_registry()
            cache = MemoryLRUCache()
            pins, guard = wire_gate(cache)
            worker = relay_worker(tmp_path, registry, governor, release_guard=guard)
            await worker.start()
            engine = make_engine(registry, worker, cache=cache, pins=pins)
            await engine.run(load_graph(name=f"model-{index}", ram=0), ["load"])
            token = worker.instance_token
            assert token is not None
            workers.append((worker, engine, token))
        try:
            await asyncio.sleep(0.05)
            results = await asyncio.gather(
                *(
                    worker.full_release(f"maintenance-{index}", token)
                    for index, (worker, _engine, token) in enumerate(workers)
                )
            )
            assert all(len(result.consumers) == 2 for result in results)
            assert all(
                consumer["status"] == "complete"
                for result in results
                for consumer in result.consumers
            )
            assert all(worker.instance_token == token for worker, _engine, token in workers)
        finally:
            await asyncio.gather(*(worker.close() for worker, _engine, _token in workers))

    asyncio.run(scenario())


def test_group_full_release_stays_busy_until_cancelled_thread_settles(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifests: list[Path] = []
        for pack in ("alpha", "beta"):
            root = tmp_path / pack
            root.mkdir()
            manifest = root / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname = "{pack}"\n\n[pack.entry]\n'
                'nodes = "memorypack_nodes:NODES"\n'
                'types = "memorypack_nodes:register_types"\n'
                'consumers = "memorypack_nodes:memory_consumers"\n'
            )
            manifests.append(manifest)
        registry = core_registry()
        group = GroupIsolatedWorker(
            "shared",
            manifests,
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await group.start()
        entered = tmp_path / "entered"
        finish = tmp_path / "finish"
        alpha = group.members["alpha"]
        beta = group.members["beta"]
        running = asyncio.create_task(
            make_engine(registry, cast("Any", alpha)).run(
                Graph(
                    nodes={
                        "hold": GraphNode(
                            "mem.thread_hold",
                            {"entered": str(entered), "finish": str(finish)},
                        )
                    }
                ),
                ["hold"],
            )
        )
        try:
            assert await eventually(entered.exists)
            running.cancel()
            await asyncio.sleep(0.02)
            close_alpha = asyncio.create_task(cast("Any", alpha)._session.close())
            await asyncio.sleep(0.05)
            token = beta.instance_token
            assert token is not None
            refused = await beta.full_release("while-thread-active", token)
            assert refused.status == "busy"
            assert all(item["status"] == "busy" for item in refused.consumers)
            finish.touch()
            with pytest.raises(asyncio.CancelledError):
                await running
            await close_alpha
            await asyncio.sleep(0.05)
            completed = await beta.full_release("after-thread", token)
            assert completed.status == "complete"
        finally:
            finish.touch()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            await group.close()

    asyncio.run(scenario())


def test_full_release_ignores_late_duplicate_reply() -> None:
    async def scenario() -> None:
        sent: list[dict[str, object]] = []
        query_reply: dict[str, object] = {}
        relay: MemoryRelay

        async def send(header: dict[str, object]) -> None:
            nonlocal query_reply
            sent.append(dict(header))
            if header["type"] == "memoryFreeQuery":
                query_reply = {
                    **header,
                    "type": "memoryFreeCandidates",
                    "status": "complete",
                    "consumers": [{"consumer": "cache", "status": "ready", "candidates": []}],
                }
                relay.on_release_reply(query_reply)
            else:
                relay.on_release_reply(query_reply)
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeResult",
                        "status": "complete",
                        "consumers": [
                            {
                                "consumer": "cache",
                                "status": "complete",
                                "released": [],
                            }
                        ],
                    }
                )

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        result = await relay.full_release("operation", "worker-instance")
        assert result is not None
        assert result.status == "complete"
        assert result.consumers == ({"consumer": "cache", "status": "complete"},)
        assert sent[0]["requestId"] != sent[1]["requestId"]

    asyncio.run(scenario())


def test_full_release_refuses_mismatched_acknowledgement() -> None:
    async def scenario() -> None:
        relay: MemoryRelay

        async def send(header: dict[str, object]) -> None:
            relay.on_release_reply(
                {
                    **header,
                    "type": "memoryFreeCandidates",
                    "status": "complete",
                    "operationRequestId": "another-operation",
                    "consumers": [],
                }
            )

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        assert await relay.full_release("operation", "worker-instance") is None

    asyncio.run(scenario())


def test_full_release_aborts_malformed_query_before_reuse() -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        malformed = True
        active_operation: str | None = None

        async def send(header: dict[str, object]) -> None:
            nonlocal active_operation, malformed
            if header["type"] == "memoryFreeAbort":
                assert header["operationRequestId"] == active_operation
                active_operation = None
                relay.on_release_reply(
                    {**header, "type": "memoryFreeAborted", "status": "complete"}
                )
                return
            if header["type"] == "memoryFreeQuery":
                assert active_operation is None
                active_operation = str(header["operationRequestId"])
                reply = {
                    **header,
                    "type": "memoryFreeCandidates",
                    "status": "complete",
                    "consumers": [],
                }
                if malformed:
                    reply["error"] = "invalid"
                    malformed = False
                relay.on_release_reply(reply)
                return
            assert header["operationRequestId"] == active_operation
            active_operation = None
            relay.on_release_reply(
                {
                    **header,
                    "type": "memoryFreeResult",
                    "status": "complete",
                    "consumers": [],
                }
            )

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        assert await relay.full_release("malformed", "worker-instance") is None
        result = await relay.full_release("retry", "worker-instance")
        assert result is not None
        assert result.status == "complete"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "override",
    (
        {"status": "complete", "error": "malformed"},
        {"status": "error", "error": "refused"},
        {"workerInstance": "other"},
    ),
)
def test_full_release_rejects_malformed_abort_acknowledgement(
    override: dict[str, object],
) -> None:
    async def scenario() -> None:
        relay: MemoryRelay

        async def send(header: dict[str, object]) -> None:
            relay.on_release_reply(
                {
                    **header,
                    "type": "memoryFreeAborted",
                    "status": "complete",
                    **override,
                }
            )

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        assert not await relay._abort_full_release("operation", "worker-instance")

    asyncio.run(scenario())


def test_host_aborts_malformed_query_before_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    corrupted = False

    async def corrupt_query(writer, header, blobs):
        nonlocal corrupted
        if header.get("type") == "memoryFreeCandidates" and not corrupted:
            header = {**header, "error": "malformed"}
            corrupted = True
        await write_frame(writer, header, blobs)

    monkeypatch.setattr("dinkster_workers.host.write_frame", corrupt_query)

    async def scenario() -> None:
        child_registry = core_registry()
        child_worker = InProcessWorker({}, child_registry)
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader, writer) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await serve_connection(
                reader,
                writer,
                pack_name="empty",
                worker=child_worker,
                schemas={},
                planner=None,
                consumers={},
                codec=ValueCodec(child_registry),
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        parent_registry = core_registry()
        session = BoundarySession(
            parent_registry,
            role="test",
            pack="empty",
            codec=ValueCodec(parent_registry),
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await session.begin(reader, writer, timeout=2)
        try:
            token = session.instance_token
            assert token is not None
            with pytest.raises(WorkerDied):
                await asyncio.wait_for(session.full_release("malformed", token), 2)
            assert session.alive
            retried = await asyncio.wait_for(session.full_release("retry", token), 2)
            assert retried.status == "complete"
            assert retried.consumers == ()
        finally:
            with suppress(Exception):
                await session.send({"type": "shutdown"}, [])
            await session.close()
            await asyncio.wait_for(host_task, 2)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_host_transport_close_clears_only_uncommitted_reservation() -> None:
    async def scenario() -> None:
        child_registry = core_registry()
        child_worker = InProcessWorker({}, child_registry)
        resource_tasks: set[asyncio.Task[None]] = set()
        maintenance_operations: set[tuple[object, str]] = set()
        accepted: asyncio.Queue[asyncio.Task[None]] = asyncio.Queue()

        async def accept(reader, writer) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.put_nowait(task)
            await serve_connection(
                reader,
                writer,
                pack_name="empty",
                worker=child_worker,
                schemas={},
                planner=None,
                consumers={},
                codec=ValueCodec(child_registry),
                process_resource_tasks=resource_tasks,
                process_maintenance_operations=maintenance_operations,
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        first_reader, first_writer = await asyncio.open_connection(address[0], address[1])
        first_host = await accepted.get()
        hello = await read_frame(first_reader)
        assert hello is not None
        worker_instance = hello[0]["workerInstance"]
        assert isinstance(worker_instance, str)
        await write_frame(
            first_writer,
            {
                "type": "memoryFreeQuery",
                "requestId": "query",
                "operationRequestId": "abandoned",
                "workerInstance": worker_instance,
            },
            [],
        )
        candidates = await read_frame(first_reader)
        assert candidates is not None
        assert candidates[0]["type"] == "memoryFreeCandidates"
        assert maintenance_operations
        first_writer.close()
        await first_writer.wait_closed()
        await asyncio.wait_for(first_host, 2)
        assert not maintenance_operations

        second_reader, second_writer = await asyncio.open_connection(address[0], address[1])
        second_host = await accepted.get()
        parent_registry = core_registry()
        session = BoundarySession(
            parent_registry,
            role="test",
            pack="empty",
            codec=ValueCodec(parent_registry),
            release_guard=ReleaseGuard(ResourcePins(), ()),
        )
        await session.begin(second_reader, second_writer, timeout=2)
        try:
            token = session.instance_token
            assert token is not None
            assert token == worker_instance
            result = await session.full_release("retry", token)
            assert result.status == "complete"
        finally:
            with suppress(Exception):
                await session.send({"type": "shutdown"}, [])
            await session.close()
            await asyncio.wait_for(second_host, 2)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_host_malformed_commit_identity_clears_its_reservation_for_reuse() -> None:
    async def scenario() -> None:
        child_registry = core_registry()
        child_worker = InProcessWorker({}, child_registry)
        maintenance_operations: set[tuple[object, str]] = set()
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader, writer) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await serve_connection(
                reader,
                writer,
                pack_name="empty",
                worker=child_worker,
                schemas={},
                planner=None,
                consumers={},
                codec=ValueCodec(child_registry),
                process_maintenance_operations=maintenance_operations,
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        hello = await read_frame(reader)
        assert hello is not None
        worker_instance = hello[0]["workerInstance"]
        assert isinstance(worker_instance, str)

        async def query(operation_id: str) -> dict[str, object]:
            await write_frame(
                writer,
                {
                    "type": "memoryFreeQuery",
                    "requestId": operation_id,
                    "operationRequestId": operation_id,
                    "workerInstance": worker_instance,
                },
                [],
            )
            frame = await read_frame(reader)
            assert frame is not None
            return frame[0]

        try:
            candidates = await query("malformed-commit")
            assert candidates["status"] == "complete"
            assert maintenance_operations
            await write_frame(
                writer,
                {
                    "type": "memoryFreeCommit",
                    "requestId": "malformed-commit",
                    "operationRequestId": "malformed-commit",
                    "workerInstance": "wrong-instance",
                },
                [],
            )
            result = await read_frame(reader)
            assert result is not None
            assert result[0]["status"] == "error"
            assert result[0]["error"] == "full-release commit identity is invalid"
            assert not maintenance_operations

            retried = await query("retry")
            assert retried["status"] == "complete"
            await write_frame(
                writer,
                {
                    "type": "memoryFreeAbort",
                    "requestId": "retry",
                    "operationRequestId": "retry",
                    "workerInstance": worker_instance,
                },
                [],
            )
            aborted = await read_frame(reader)
            assert aborted is not None
            assert aborted[0]["status"] == "complete"
            assert not maintenance_operations
        finally:
            await write_frame(writer, {"type": "shutdown"}, [])
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(host_task, 2)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


async def _exercise_host_duplicate_commit(duplicate_worker_instance: str | None) -> None:
    class BlockingConsumer:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.finish = asyncio.Event()
            self.calls = 0

        def footprint(self, device: str) -> int:
            del device
            return 0

        async def shed(self, pressure: object) -> int:
            del pressure
            return 0

        async def full_release(self) -> FullReleaseResult:
            self.calls += 1
            self.entered.set()
            await self.finish.wait()
            return FullReleaseResult("complete")

    registry = core_registry()
    worker = InProcessWorker({}, registry)
    consumer = BlockingConsumer()
    maintenance_operations: set[tuple[object, str]] = set()
    accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    async def accept(reader, writer) -> None:
        task = asyncio.current_task()
        assert task is not None
        accepted.set_result(task)
        await serve_connection(
            reader,
            writer,
            pack_name="empty",
            worker=worker,
            schemas={},
            planner=None,
            consumers={"blocking": consumer},
            codec=ValueCodec(registry),
            process_maintenance_operations=maintenance_operations,
        )

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(address[0], address[1])
    host_task = await accepted
    hello = await read_frame(reader)
    assert hello is not None
    worker_instance = hello[0]["workerInstance"]
    assert isinstance(worker_instance, str)
    operation_id = "blocked-commit"

    async def read_request(request_id: str) -> dict[str, object]:
        while True:
            frame = await asyncio.wait_for(read_frame(reader), 2)
            assert frame is not None
            if frame[0].get("requestId") == request_id:
                return frame[0]

    try:
        await write_frame(
            writer,
            {
                "type": "memoryFreeQuery",
                "requestId": "query",
                "operationRequestId": operation_id,
                "workerInstance": worker_instance,
            },
            [],
        )
        candidates = await read_request("query")
        assert candidates["status"] == "complete"
        candidate_consumers = cast("list[dict[str, object]]", candidates["consumers"])
        commit_selection = [
            {
                "consumer": item["consumer"],
                "queryStatus": item["status"],
                "candidates": item.get("candidates", []),
            }
            for item in candidate_consumers
        ]
        await write_frame(
            writer,
            {
                "type": "memoryFreeCommit",
                "requestId": "first-commit",
                "operationRequestId": operation_id,
                "workerInstance": worker_instance,
                "workerStatus": candidates["status"],
                "consumers": commit_selection,
            },
            [],
        )
        await asyncio.wait_for(consumer.entered.wait(), 2)
        assert consumer.calls == 1
        assert maintenance_operations

        await write_frame(
            writer,
            {
                "type": "memoryFreeCommit",
                "requestId": "duplicate-commit",
                "operationRequestId": operation_id,
                "workerInstance": duplicate_worker_instance or worker_instance,
                "workerStatus": candidates["status"],
                "consumers": commit_selection,
            },
            [],
        )
        duplicate = await read_request("duplicate-commit")
        assert duplicate == {
            "type": "memoryFreeResult",
            "requestId": "duplicate-commit",
            "operationRequestId": operation_id,
            "workerInstance": worker_instance,
            "status": "error",
            "error": "full-release commit is already active",
            "consumers": [],
            "blobs": [],
        }
        assert consumer.calls == 1
        assert maintenance_operations

        consumer.finish.set()
        completed = await read_request("first-commit")
        assert completed["status"] == "complete"
        assert consumer.calls == 1
        assert not maintenance_operations
    finally:
        consumer.finish.set()
        with suppress(Exception):
            await write_frame(writer, {"type": "shutdown"}, [])
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(host_task, 2)
        server.close()
        await server.wait_closed()


def test_host_duplicate_valid_commit_preserves_first_commit_fence() -> None:
    asyncio.run(_exercise_host_duplicate_commit(None))


def test_host_duplicate_wrong_instance_commit_preserves_first_commit_fence() -> None:
    asyncio.run(_exercise_host_duplicate_commit("wrong-instance"))


def test_cancelled_in_process_invocation_refuses_late_source_materialization() -> None:
    async def scenario() -> None:
        registry = core_registry()
        schema = NodeSchema(
            node_type="test.late_source",
            outputs=(OutputSpec("result", TypeExpr.concrete("core.string")),),
        )
        entered = threading.Event()
        finished = threading.Event()
        materialization: list[str] = []

        class LateSourceNode(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return schema

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                context = current_execution_context()
                assert context is not None and context.materialize_source is not None
                entered.set()
                while not context.cancelled():
                    time.sleep(0.001)
                try:
                    context.materialize_source(cast("Any", object()), "media/image", "input")
                except RuntimeError:
                    materialization.append("refused")
                else:
                    materialization.append("allowed")
                finally:
                    finished.set()
                return cls.outputs(result="done")

        class StagingSession:
            def materialize(self, asset: Any, kind: str, category: str) -> str:
                del asset, kind, category
                return ".dinkster-source/late"

            def close(self) -> None:
                pass

        class StagingProvider:
            def sweep(self) -> None:
                pass

            def open(self, invocation_id: str, authorities: Sequence[object]) -> StagingSession:
                del invocation_id, authorities
                return StagingSession()

        worker = InProcessWorker({schema.node_type: LateSourceNode}, registry)
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader, writer) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await serve_connection(
                reader,
                writer,
                pack_name="test",
                worker=worker,
                schemas={schema.node_type: schema},
                planner=None,
                consumers={},
                codec=ValueCodec(registry),
                source_staging=StagingProvider(),  # type: ignore[arg-type]
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        session = BoundarySession(
            registry,
            role="test",
            pack="test",
            codec=ValueCodec(registry),
        )
        await session.begin(reader, writer, timeout=2)
        running = asyncio.create_task(
            session.invoke(
                Invocation(
                    invocation_id="late-source",
                    node_id="late-source",
                    node_type=schema.node_type,
                    inputs={},
                    effective_schema=schema,
                )
            )
        )
        try:
            assert await eventually(entered.is_set)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert await eventually(finished.is_set)
            assert materialization == ["refused"]
        finally:
            with suppress(Exception):
                await session.send({"type": "shutdown"}, [])
            await session.close()
            await asyncio.wait_for(host_task, 2)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_full_release_refuses_malformed_completion_and_retains_condemnation() -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        pins = ResourcePins()
        resource_id = "resident:cache:item"

        async def send(header: dict[str, object]) -> None:
            if header["type"] == "memoryFreeQuery":
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeCandidates",
                        "status": "complete",
                        "consumers": [
                            {
                                "consumer": "cache",
                                "status": "ready",
                                "candidates": [
                                    {
                                        "itemId": "item",
                                        "resourceId": resource_id,
                                        "nbytes": 0,
                                        "token": "unchanged",
                                    }
                                ],
                            }
                        ],
                    }
                )
                return
            assert pins.condemned(resource_id)
            relay.on_release_reply(
                {
                    **header,
                    "type": "memoryFreeResult",
                    "status": "complete",
                    "consumers": [
                        {
                            "consumer": "cache",
                            "status": "complete",
                            "error": "cleanup failed",
                            "released": [],
                        }
                    ],
                }
            )

        relay = MemoryRelay(send, None, ReleaseGuard(pins, ()))
        assert await relay.full_release("operation", "worker-instance") is None
        assert pins.condemned(resource_id)
        assert not pins.pin(resource_id)

    asyncio.run(scenario())


def test_full_release_timeout_waits_for_terminal_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        commit_started = asyncio.Event()
        commit_finished = asyncio.Event()

        async def finish_commit(header: dict[str, object]) -> None:
            await asyncio.sleep(0.05)
            relay.on_release_reply(
                {
                    **header,
                    "type": "memoryFreeResult",
                    "status": "complete",
                    "consumers": [{"consumer": "cache", "status": "complete", "released": []}],
                }
            )
            commit_finished.set()

        async def send(header: dict[str, object]) -> None:
            if header["type"] == "memoryFreeQuery":
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeCandidates",
                        "status": "complete",
                        "consumers": [{"consumer": "cache", "status": "ready", "candidates": []}],
                    }
                )
                return
            commit_started.set()
            asyncio.create_task(finish_commit(dict(header)))

        monkeypatch.setattr("dinkster_workers.relay._RELEASE_TIMEOUT_S", 0.01)
        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        result = await relay.full_release("operation", "worker-instance")
        assert commit_started.is_set()
        assert commit_finished.is_set()
        assert result is not None
        assert result.status == "error"
        assert result.error == "worker full release timed out"

    asyncio.run(scenario())


def test_relay_close_keeps_release_unsettled_until_terminal_acknowledgement() -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        commit: dict[str, object] | None = None
        commit_started = asyncio.Event()

        async def send(header: dict[str, object]) -> None:
            nonlocal commit
            if header["type"] == "memoryFreeQuery":
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeCandidates",
                        "status": "complete",
                        "consumers": [],
                    }
                )
                return
            commit = dict(header)
            commit_started.set()

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        release = asyncio.create_task(relay.full_release("operation", "worker-instance"))
        await commit_started.wait()
        release.cancel()
        await asyncio.sleep(0)
        release.cancel()
        relay.close()
        drain = asyncio.create_task(relay.drain_releases())
        await asyncio.sleep(0)
        drain.cancel()
        await asyncio.sleep(0)
        drain.cancel()
        assert not release.done()
        assert not drain.done()
        assert commit is not None
        relay.on_release_reply(
            {
                **commit,
                "type": "memoryFreeResult",
                "status": "complete",
                "consumers": [],
            }
        )
        with pytest.raises(asyncio.CancelledError):
            await drain
        with pytest.raises(asyncio.CancelledError):
            await release

    asyncio.run(scenario())


def test_release_send_drain_failure_waits_for_terminal_acknowledgement() -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        commit: dict[str, object] | None = None
        commit_delivered = asyncio.Event()

        class FailedDrainWriter:
            def write(self, data: bytes) -> None:
                assert data

            async def drain(self) -> None:
                commit_delivered.set()
                raise ConnectionError("drain failed after delivery")

        writer = cast("asyncio.StreamWriter", FailedDrainWriter())

        async def send(header: dict[str, object]) -> None:
            nonlocal commit
            if header["type"] == "memoryFreeQuery":
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeCandidates",
                        "status": "complete",
                        "consumers": [],
                    }
                )
                return
            commit = dict(header)
            await write_frame(writer, header, [])

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        release = asyncio.create_task(relay.full_release("operation", "worker-instance"))
        await commit_delivered.wait()
        await asyncio.sleep(0)
        assert not release.done()
        assert commit is not None
        relay.on_release_reply(
            {
                **commit,
                "type": "memoryFreeResult",
                "status": "complete",
                "consumers": [],
            }
        )
        result = await release
        assert result is not None
        assert result.status == "complete"

    asyncio.run(scenario())


def test_release_send_cancellation_waits_for_proven_process_death() -> None:
    async def scenario() -> None:
        relay: MemoryRelay
        commit_delivered = asyncio.Event()

        class BlockingDrainWriter:
            def write(self, data: bytes) -> None:
                assert data

            async def drain(self) -> None:
                commit_delivered.set()
                await asyncio.Event().wait()

        writer = cast("asyncio.StreamWriter", BlockingDrainWriter())

        async def send(header: dict[str, object]) -> None:
            if header["type"] == "memoryFreeQuery":
                relay.on_release_reply(
                    {
                        **header,
                        "type": "memoryFreeCandidates",
                        "status": "complete",
                        "consumers": [],
                    }
                )
                return
            await write_frame(writer, header, [])

        relay = MemoryRelay(send, None, ReleaseGuard(ResourcePins(), ()))
        release = asyncio.create_task(relay.full_release("operation", "worker-instance"))
        await commit_delivered.wait()
        release.cancel()
        await asyncio.sleep(0)
        release.cancel()
        await asyncio.sleep(0)
        assert not release.done()
        relay.process_died()
        with pytest.raises(asyncio.CancelledError):
            await release

    asyncio.run(scenario())


@pytest.mark.parametrize("gate_status", ("complete", "busy", "error"))
def test_full_release_consumer_exception_keeps_sent_candidates_condemned(gate_status: str) -> None:
    async def scenario() -> None:
        class FailingPool(ResidentPool):
            async def release_full(
                self, candidates: Sequence[ReleaseCandidate]
            ) -> FullReleaseCommitResult:
                await super().release_full(candidates)
                raise RuntimeError("post-release cleanup failed")

        pool = FailingPool(cost_of=lambda _: {"ram": 0})
        rid = pool.rid_for(bytearray(8))
        resource_id = resident_resource_id(rid)
        pins = ResourcePins()
        blocked_id = (
            resident_resource_id(pool.rid_for(bytearray(8))) if gate_status != "complete" else None
        )
        if gate_status == "busy":
            assert blocked_id is not None
            assert pins.pin(blocked_id)

        def invalidate(candidate_id: str) -> None:
            if gate_status == "error" and candidate_id == blocked_id:
                raise RuntimeError("cache invalidation failed")

        async with hosted_pool(pool, ReleaseGuard(pins, (invalidate,))) as session:
            token = session.instance_token
            assert token is not None
            result = await session.full_release("operation", token)
            assert result.consumers[0]["status"] == ("busy" if gate_status == "busy" else "error")
            assert all(candidate.item_id != rid for candidate in pool.propose_full_release())
            assert session.alive and session.instance_token == token
            assert pins.condemned(resource_id)
            assert not pins.pin(resource_id)

    asyncio.run(scenario())


def test_full_release_proven_refusal_allows_surviving_reference_reuse() -> None:
    async def scenario() -> None:
        class UsedPool(ResidentPool):
            async def release_full(
                self, candidates: Sequence[ReleaseCandidate]
            ) -> FullReleaseCommitResult:
                for candidate in candidates:
                    self.get(candidate.item_id)
                return await super().release_full(candidates)

        pool = UsedPool(cost_of=lambda _: {"ram": 0})
        resident = bytearray(8)
        rid = pool.rid_for(resident)
        resource_id = resident_resource_id(rid)
        pins = ResourcePins()
        async with hosted_pool(pool, ReleaseGuard(pins, ())) as session:
            token = session.instance_token
            assert token is not None
            result = await session.full_release("operation", token)
            assert result.consumers == ({"consumer": "pool", "status": "busy"},)
            assert pool.get(rid) is resident
            assert not pins.condemned(resource_id)
            assert pins.pin(resource_id)

    asyncio.run(scenario())


def test_legacy_shed_timeout_does_not_cancel_session_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dinkster_workers.relay._RELEASE_TIMEOUT_S", 0.1)

    async def scenario() -> None:
        entered, finish = asyncio.Event(), asyncio.Event()

        class SlowPool(ResidentPool):
            async def release(
                self, device: str, candidates: Sequence[ReleaseCandidate]
            ) -> dict[str, int]:
                entered.set()
                await finish.wait()
                return await super().release(device, candidates)

        pool = SlowPool(cost_of=lambda _: {"ram": 8})
        rid = pool.rid_for(bytearray(8))
        async with hosted_pool(pool, ReleaseGuard(ResourcePins(), ())) as session:
            relay = session._relay
            assert relay is not None
            release = asyncio.create_task(relay.shed("pool", PressureSignal("ram", 8, (rid,))))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                close = asyncio.create_task(session.close())
                done, _ = await asyncio.wait({close}, timeout=1)
                assert close in done
                assert not close.cancelled()
                await close
                assert session._writer is None
                assert session._relay is None
                assert await release == 0
            finally:
                finish.set()
                await asyncio.gather(release, return_exceptions=True)

    asyncio.run(scenario())


def test_pinned_reference_refuses_ram_release(tmp_path: Path) -> None:
    """A resource pinned by a live run cannot be released - condemnation
    fails, the child keeps the resident, zero is reported - and the same
    pressure crosses the moment the pin unwinds."""

    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000, "ram": 1000})
        registry = core_registry()
        cache = MemoryLRUCache()
        pins, guard = wire_gate(cache)
        worker = relay_worker(tmp_path, registry, governor, release_guard=guard)
        await worker.start()
        try:
            engine = make_engine(registry, worker, cache=cache, pins=pins)
            await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: len(governor.details().get(CONSUMER, ())) == 1)
            item_id = governor.details()[CONSUMER][0].item_id
            resource_id = resident_resource_id(item_id)

            # The stand-in for a run mid-flight: engine.run pins every
            # resource-referencing envelope for the run's lifetime.
            assert pins.pin(resource_id)
            freed = await governor.shed("ram", 300)
            assert freed == 0  # condemnation refused: a live run holds it
            assert consumer_footprint(governor, "ram") == 300
            assert not pins.condemned(resource_id)  # nothing left standing

            # Item-targeted pressure at an ID the child cannot resolve
            # frees nothing, pinned or not.
            freed = await governor.shed("ram", 300, consumers=[CONSUMER], items=["no-such-item"])
            assert freed == 0

            pins.unpin(resource_id)
            freed = await governor.shed("ram", 300)
            assert freed == 300  # the pin was the only thing in the way
            assert consumer_footprint(governor, "ram") == 0
            assert pins.condemned(resource_id)  # tombstone: rids never reuse
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_item_targeted_shed_resolves_only_real_items(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: len(governor.details().get(CONSUMER, ())) == 1)
            item_id = governor.details()[CONSUMER][0].item_id

            # An item ID the child cannot resolve frees nothing.
            freed = await governor.shed(VRAM0, 400, consumers=[CONSUMER], items=["no-such-item"])
            assert freed == 0
            assert consumer_footprint(governor, VRAM0) == 400

            # The real ID is "unload this model", across the boundary.
            freed = await governor.shed(VRAM0, 400, consumers=[CONSUMER], items=[item_id])
            assert freed == 400
            assert consumer_footprint(governor, VRAM0) == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_admission_sheds_the_child_pool(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=800, ram=100), ["load"])
            assert await eventually(lambda: consumer_footprint(governor, VRAM0) == 800)

            # 800 held by the child pool + 500 requested > 1000: admission
            # must shed across the boundary, see the freed bytes in the
            # rescore, and grant - not time out on a stale snapshot.
            async with governor.reserve(VRAM0, 500, timeout=8.0):
                assert governor.reserved(VRAM0) == 500
                assert consumer_footprint(governor, VRAM0) == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_device_map_translates_snapshot_and_gates_wrong_silicon(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        # A worker pinned to physical GPU 1: its honest "cuda:0" is the
        # parent's "cuda:1".
        governor = MemoryGovernor({VRAM1: 1000})
        registry = core_registry()
        worker = relay_worker(
            tmp_path,
            registry,
            governor,
            device_map=DeviceMap({"cuda:0": "cuda:1"}),
        )
        await worker.start()
        try:
            assert worker.device_map_wire == {
                "mapping": {"cuda:0": "cuda:1"},
                "qualifier": None,
            }
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400, ram=300), ["load"])

            # Snapshot keys crossed into the parent's namespace.
            assert await eventually(lambda: consumer_footprint(governor, VRAM1) == 400)
            assert consumer_footprint(governor, VRAM0) == 0

            # Pressure on the parent's cuda:0 has no counterpart inside this
            # child; forwarding the string through would unload the wrong
            # GPU, so the proxy sheds nothing.
            governor.set_budget(VRAM0, 1000)
            assert await governor.shed(VRAM0, 400) == 0
            assert consumer_footprint(governor, VRAM1) == 400

            # Pressure on the parent's cuda:1 inverts to the child's cuda:0.
            assert await governor.shed(VRAM1, 400) == 400
            assert consumer_footprint(governor, VRAM1) == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_dead_worker_holds_nothing_and_frees_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({VRAM0: 1000})
        registry = core_registry()
        worker = relay_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            engine = make_engine(registry, worker)
            await engine.run(load_graph(vram=400, ram=300), ["load"])
            assert await eventually(lambda: consumer_footprint(governor, VRAM0) == 400)
            proc = worker._proc  # noqa: SLF001 - killing the child on purpose
            assert proc is not None
            proc.kill()
            # The read loop notices and closes the relay: the snapshot dies
            # with the process instead of freezing at its last report.
            assert await eventually(lambda: consumer_footprint(governor, VRAM0) == 0)
            assert await governor.shed(VRAM0, 400) == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_bad_consumers_entry_fails_start(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:NODES"\n'  # a list, not a factory
        )
        worker = IsolatedWorker(manifest, core_registry(), extra_env={"PYTHONPATH": str(TESTS_DIR)})
        with pytest.raises(RuntimeError, match="failed to start"):
            await worker.start()
        await worker.close()

    asyncio.run(scenario())
