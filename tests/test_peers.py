"""start_server + PeerClient (DESIGN 3.10): a real instance on a real
socket announcing into discovery, and the typed client peers use to speak
the coordination surface. Verdicts relay: a peer's 507/503 raise the same
exceptions the local governor would."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_memory import (
    BudgetExceeded,
    ConsumerItem,
    MemoryGovernor,
    PressureSignal,
    ReservationTimeout,
    Shedder,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import (
    InstanceRegistry,
    PeerClient,
    PeerUngoverned,
    RunningServer,
    start_server,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING = TypeExpr.concrete("core.string")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


NODES: list[type[Node]] = [Echo]
SCHEMAS = build_schemas(NODES)


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


class TrimmableCache(Shedder):
    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        freed = min(self.holding, pressure.bytes_needed)
        self.holding -= freed
        return freed


async def governed_instance(
    governor: MemoryGovernor,
    *,
    registry: InstanceRegistry | None = None,
    instance_id: str | None = None,
    heartbeat_interval: float = 5.0,
) -> RunningServer:
    return await start_server(
        make_engine,
        SCHEMAS,
        governor=governor,
        registry=registry,
        instance_id=instance_id,
        heartbeat_interval=heartbeat_interval,
    )


# -- serve lifecycle -----------------------------------------------------------


def test_server_announces_real_endpoint_and_close_withdraws(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = InstanceRegistry(tmp_path)
        server = await start_server(make_engine, SCHEMAS, registry=registry, instance_id="inst-a")
        reader = InstanceRegistry(tmp_path)
        try:
            peers = reader.peers()
            assert [p.instance_id for p in peers] == ["inst-a"]
            # The advertised endpoint is the bound one, and it answers.
            assert peers[0].endpoint == server.endpoint
            async with PeerClient(server.endpoint) as client:
                data = await client.status()
                assert "queue" in data
        finally:
            await server.close()
        # Withdrawn from discovery before anyone can find a corpse.
        assert reader.peers() == []
        await server.close()  # idempotent

    asyncio.run(scenario())


def test_heartbeat_refreshes_the_registry_entry(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = InstanceRegistry(tmp_path)
        server = await start_server(
            make_engine,
            SCHEMAS,
            registry=registry,
            instance_id="inst-a",
            heartbeat_interval=0.05,
        )
        try:
            path = tmp_path / "inst-a.json"
            first = json.loads(path.read_text())["heartbeatAt"]
            async with asyncio.timeout(2):
                while json.loads(path.read_text())["heartbeatAt"] <= first:
                    await asyncio.sleep(0.02)
        finally:
            await server.close()

    asyncio.run(scenario())


def test_server_runs_without_discovery() -> None:
    async def scenario() -> None:
        async with await start_server(make_engine, SCHEMAS) as server:
            async with PeerClient(server.endpoint) as client:
                assert "devices" in await client.status()

    asyncio.run(scenario())


# -- peer client round trips ---------------------------------------------------


def test_peer_lease_round_trip_and_verdict_relay() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        async with (
            await governed_instance(governor) as server,
            PeerClient(server.endpoint) as client,
        ):
            lease = await client.reserve("vram:cuda:0", 60, ttl=30.0)
            assert governor.reserved("vram:cuda:0") == 60
            assert lease.expires_in > 29

            status = await client.status()
            assert [entry["reservationId"] for entry in status["leases"]] == [lease.lease_id]

            renewed = await client.renew(lease.lease_id, ttl=45.0)
            assert renewed is not None and renewed.expires_in > 44

            assert await client.release(lease.lease_id) is True
            assert governor.reserved("vram:cuda:0") == 0
            assert await client.release(lease.lease_id) is False
            assert await client.renew(lease.lease_id) is None

            # The peer's verdicts arrive as the local exceptions.
            with pytest.raises(BudgetExceeded):
                await client.reserve("vram:cuda:0", 200)
            blocker = await client.reserve("vram:cuda:0", 100)
            with pytest.raises(ReservationTimeout):
                await client.reserve("vram:cuda:0", 50, timeout=0.05)
            await client.release(blocker.lease_id)

    asyncio.run(scenario())


def test_peer_shed_and_trim() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        cache = TrimmableCache("ram", 40)
        pool = TrimmableCache("ram", 30)
        governor.register_shedder(cache, name="cache", priority=0)
        governor.register_shedder(pool, name="pool", priority=1)
        async with (
            await governed_instance(governor) as server,
            PeerClient(server.endpoint) as client,
        ):
            assert await client.shed("ram", 50) == 50
            assert (cache.holding, pool.holding) == (0, 20)
            cache.holding = 40
            assert await client.trim("ram", consumers=["cache"]) == 40
            assert (cache.holding, pool.holding) == (0, 20)
            assert await client.trim("ram") == 20  # full maintenance

    asyncio.run(scenario())


class ItemPool(TrimmableCache):
    """Detail-contract consumer: one named model, unloadable by stable ID."""

    def __init__(self, device: str, item_id: str, nbytes: int) -> None:
        super().__init__(device, nbytes)
        self.item_id = item_id

    async def shed(self, pressure: PressureSignal) -> int:
        if pressure.items is not None and self.item_id not in pressure.items:
            return 0
        freed, self.holding = self.holding, 0
        return freed

    def details(self) -> list[ConsumerItem]:
        if self.holding == 0:
            return []
        return [
            ConsumerItem(
                item_id=self.item_id,
                display_name="sd15.safetensors",
                bytes_by_residency={self.device: self.holding},
            )
        ]


def test_peer_details_and_item_trim() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        pool = ItemPool("vram:cuda:0", "ckpt-1", 40)
        governor.register_shedder(pool, name="models")
        async with (
            await governed_instance(governor) as server,
            PeerClient(server.endpoint) as client,
        ):
            assert "consumerDetails" not in await client.status()
            data = await client.status(details=True)
            (item,) = data["consumerDetails"]["models"]
            assert item["itemId"] == "ckpt-1"
            assert item["displayName"] == "sd15.safetensors"
            # A frontend's "unload this model", spoken peer-to-peer.
            freed = await client.trim("vram:cuda:0", consumers=["models"], items=[item["itemId"]])
            assert freed == 40
            data = await client.status(details=True)
            assert data["consumerDetails"]["models"] == []

    asyncio.run(scenario())


def test_ungoverned_peer_raises_peer_ungoverned() -> None:
    async def scenario() -> None:
        async with (
            await start_server(make_engine, SCHEMAS) as server,
            PeerClient(server.endpoint) as client,
        ):
            with pytest.raises(PeerUngoverned):
                await client.reserve("ram", 1)
            with pytest.raises(PeerUngoverned):
                await client.shed("ram", 1)
            # Status still answers, honestly ungoverned.
            assert (await client.status())["memoryGovernor"] is None

    asyncio.run(scenario())


def test_held_renews_past_the_original_ttl_and_releases_on_exit() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        async with (
            await governed_instance(governor) as server,
            PeerClient(server.endpoint) as client,
        ):
            async with client.held("ram", 60, ttl=0.2):
                assert governor.reserved("ram") == 60
                await asyncio.sleep(0.5)  # several original lifetimes
                assert governor.reserved("ram") == 60  # renewal kept it alive
            assert governor.reserved("ram") == 0  # exit released it

    asyncio.run(scenario())


# -- discovery to coordination, end to end --------------------------------------


def test_instance_discovers_peer_and_takes_a_lease(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor_a = MemoryGovernor({"vram:cuda:0": 100})
        a = await governed_instance(
            governor_a,
            registry=InstanceRegistry(tmp_path),
            instance_id="inst-a",
        )
        b_registry = InstanceRegistry(tmp_path)
        b = await start_server(make_engine, SCHEMAS, registry=b_registry, instance_id="inst-b")
        try:
            # B finds A through the per-machine registry, never config.
            (peer,) = b_registry.peers()
            assert peer.instance_id == "inst-a"
            async with PeerClient(peer.endpoint) as client:
                async with client.held("vram:cuda:0", 80, ttl=30.0):
                    assert governor_a.reserved("vram:cuda:0") == 80
                assert governor_a.reserved("vram:cuda:0") == 0
        finally:
            await b.close()
            await a.close()

    asyncio.run(scenario())
