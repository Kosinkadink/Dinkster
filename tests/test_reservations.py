"""Reservation-before-allocation at the worker boundary (DESIGN 3.10).

The engine's lanes bound *execution*; these tests prove the second gate -
*allocation* - holds everywhere it must: the request contracts, the
governor-backed service, the GovernedWorker wrapper, the isolated lease
protocol over the wire, and the compat pack's checkpoint policy. The
invariant throughout: a reservation is held across execution and released
on every exit path - success, node error, denial, cancellation, and worker
death.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_assets import ASSET_TYPE, AssetError, AssetRef, AssetResolver, register_asset_type
from dinkster_compat_comfy.memory import (
    CHECKPOINT_RAM_FACTOR,
    NATIVE_CHECKPOINT_RAM_FACTOR,
    ReservationPlanError,
    plan_reservations,
)
from dinkster_compat_comfy.pool import ResidentPool
from dinkster_engine import Invocation, InvocationResult, OnInvocationEvent
from dinkster_memory import (
    BudgetExceeded,
    GovernorReservationService,
    InvocationView,
    MemoryGovernor,
    PressureSignal,
    ReservationRequest,
)
from dinkster_schema import InputSpec, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    TypeRegistry,
    Value,
    make_list_value,
    register_core_types,
)
from dinkster_workers import (
    GovernedWorker,
    IsolatedWorker,
    ManifestError,
    load_manifest,
)
from dinkster_workers.boundary import ValueCodec
from dinkster_workers.session import BoundarySession, WorkerDied

TESTS_DIR = Path(__file__).parent

DIGEST = "blake3:" + "ab" * 32


# -- ReservationRequest contracts ----------------------------------------


def test_reservation_request_validates() -> None:
    with pytest.raises(ValueError, match="residency"):
        ReservationRequest(residency="", nbytes=1)
    with pytest.raises(ValueError, match="nbytes"):
        ReservationRequest(residency="ram", nbytes=-1)
    assert ReservationRequest(residency="ram", nbytes=0).nbytes == 0


# -- GovernorReservationService ------------------------------------------


def test_service_merges_classes_and_releases_on_exit() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100, "disk": 20})
        service = GovernorReservationService(governor)
        requests = (
            ReservationRequest("ram", 40),
            ReservationRequest("disk", 10),
            ReservationRequest("ram", 30),  # merged with the first
        )
        async with service.reserve(requests):
            assert governor.reserved("ram") == 70
            assert governor.reserved("disk") == 10
        assert governor.reserved("ram") == 0
        assert governor.reserved("disk") == 0

    asyncio.run(scenario())


def test_service_unwinds_partial_acquisition_on_denial() -> None:
    async def scenario() -> None:
        # disk fits, ram never can: the already-acquired disk reservation
        # must unwind when the ram leg raises.
        governor = MemoryGovernor({"ram": 100, "disk": 20})
        service = GovernorReservationService(governor)
        requests = (
            ReservationRequest("disk", 10),
            ReservationRequest("ram", 200),
        )
        with pytest.raises(BudgetExceeded):
            async with service.reserve(requests):
                raise AssertionError("must not be granted")
        assert governor.reserved("ram") == 0
        assert governor.reserved("disk") == 0

    asyncio.run(scenario())


# -- GovernedWorker -------------------------------------------------------


INT = TypeExpr.concrete(CORE_INT)
HOG_SCHEMA = NodeSchema(
    node_type="test.hog",
    display_name="Hog",
    category="test",
    inputs=(InputSpec("nbytes", INT),),
    outputs=(OutputSpec("nbytes", INT),),
)


def make_invocation(nbytes: int, invocation_id: str = "i1") -> Invocation:
    registry = TypeRegistry()
    register_core_types(registry)
    return Invocation(
        invocation_id=invocation_id,
        node_id="n1",
        node_type="test.hog",
        inputs={"nbytes": registry.wrap(CORE_INT, nbytes)},
        effective_schema=HOG_SCHEMA,
    )


def plan_from_input(invocation: InvocationView) -> Sequence[ReservationRequest]:
    nbytes = invocation.inputs["nbytes"].resolve()
    assert isinstance(nbytes, int)
    return (ReservationRequest("ram", nbytes),) if nbytes else ()


class FakeWorker:
    """Inner worker that records the governor's state at execute time."""

    def __init__(self, governor: MemoryGovernor) -> None:
        self.governor = governor
        self.reserved_during: list[int] = []
        self.prepared: list[str] = []
        self.gate: asyncio.Event | None = None
        self.fail_with: Exception | None = None

    async def prepare(self, node_types: Sequence[str]) -> None:
        self.prepared.extend(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.reserved_during.append(self.governor.reserved("ram"))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        return InvocationResult(outputs={})


def governed(
    governor: MemoryGovernor, inner: FakeWorker | None = None
) -> tuple[GovernedWorker, FakeWorker]:
    fake = inner or FakeWorker(governor)
    worker = GovernedWorker(
        fake,
        planner=plan_from_input,
        reservations=GovernorReservationService(governor),
    )
    return worker, fake


def test_governed_worker_holds_reservation_across_execute() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        worker, fake = governed(governor)
        result = await worker.invoke(make_invocation(60))
        assert result.error is None
        assert fake.reserved_during == [60]  # held while executing
        assert governor.reserved("ram") == 0  # released after

    asyncio.run(scenario())


def test_governed_worker_skips_reservation_for_empty_plan() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        worker, fake = governed(governor)
        result = await worker.invoke(make_invocation(0))
        assert result.error is None
        assert fake.reserved_during == [0]

    asyncio.run(scenario())


def test_governed_worker_denial_is_a_node_error_and_skips_execute() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        worker, fake = governed(governor)
        result = await worker.invoke(make_invocation(200))
        assert result.error is not None
        assert "memory admission failed" in result.error.message
        assert result.error.node_id == "n1"
        assert fake.reserved_during == []  # execute never ran
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


def test_governed_worker_planner_failure_is_a_node_error() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        fake = FakeWorker(governor)

        def bad_planner(invocation: InvocationView) -> Sequence[ReservationRequest]:
            raise RuntimeError("no size declared")

        worker = GovernedWorker(
            fake,
            planner=bad_planner,
            reservations=GovernorReservationService(governor),
        )
        result = await worker.invoke(make_invocation(60))
        assert result.error is not None
        assert "reservation planning failed" in result.error.message
        assert "no size declared" in result.error.message
        assert fake.reserved_during == []

    asyncio.run(scenario())


def test_governed_worker_releases_when_inner_raises() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        fake = FakeWorker(governor)
        fake.fail_with = RuntimeError("execute blew up")
        worker, _ = governed(governor, fake)
        with pytest.raises(RuntimeError, match="execute blew up"):
            await worker.invoke(make_invocation(60))
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


def test_governed_worker_releases_on_cancellation() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        fake = FakeWorker(governor)
        fake.gate = asyncio.Event()  # never set: invoke blocks until cancelled
        worker, _ = governed(governor, fake)
        task = asyncio.create_task(worker.invoke(make_invocation(60)))
        while governor.reserved("ram") == 0:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


def test_governed_worker_serializes_over_a_constrained_budget() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        fake = FakeWorker(governor)
        fake.gate = asyncio.Event()
        worker, _ = governed(governor, fake)
        first = asyncio.create_task(worker.invoke(make_invocation(60, "i1")))
        while governor.reserved("ram") != 60:
            await asyncio.sleep(0.001)
        # Second does not fit until the first releases: both cannot believe
        # the same bytes are free.
        second = asyncio.create_task(worker.invoke(make_invocation(60, "i2")))
        await asyncio.sleep(0.02)
        assert governor.reserved("ram") == 60
        assert fake.reserved_during == [60]  # second not executing
        fake.gate.set()
        results = await asyncio.gather(first, second)
        assert all(r.error is None for r in results)
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


# -- manifest entry -------------------------------------------------------


def test_manifest_reservations_entry_is_optional_and_validated(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    plain.write_text('[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\n')
    assert load_manifest(plain).reservations_entry is None

    with_planner = tmp_path / "planner.toml"
    with_planner.write_text(
        '[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\nreservations = "m:plan"\n'
    )
    assert load_manifest(with_planner).reservations_entry == "m:plan"

    malformed = tmp_path / "bad.toml"
    malformed.write_text(
        '[pack]\nname = "p"\n\n[pack.entry]\nnodes = "m:N"\nreservations = "noattr"\n'
    )
    with pytest.raises(ManifestError, match="reservations"):
        load_manifest(malformed)


# -- isolated lease protocol ----------------------------------------------


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
        'reservations = "isopack_nodes:plan_reservations"\n'
    )
    return manifest


def iso_worker(
    tmp_path: Path, registry: TypeRegistry, governor: MemoryGovernor | None
) -> IsolatedWorker:
    return IsolatedWorker(
        write_manifest(tmp_path),
        registry,
        extra_env={"PYTHONPATH": str(TESTS_DIR)},
        reservations=(GovernorReservationService(governor) if governor is not None else None),
    )


def hog_invocation(
    worker: IsolatedWorker,
    registry: TypeRegistry,
    nbytes: int,
    seconds: float = 0.0,
    invocation_id: str = "i1",
    residency: str = "ram",
) -> Invocation:
    return Invocation(
        invocation_id=invocation_id,
        node_id="hog",
        node_type="iso.hog",
        inputs={
            "nbytes": registry.wrap(CORE_INT, nbytes),
            "seconds": registry.wrap(CORE_FLOAT, seconds),
            "residency": registry.wrap(CORE_STRING, residency),
        },
        effective_schema=worker.schemas["iso.hog"],
    )


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def _feed_frame(reader: asyncio.StreamReader, header: dict[str, object]) -> None:
    encoded = json.dumps({**header, "blobs": []}, separators=(",", ":")).encode()
    reader.feed_data(len(encoded).to_bytes(4, "big") + encoded)


@pytest.mark.parametrize("capture_outcome", [None, "success", "error"])
def test_result_waits_for_lease_cleanup_when_peer_dies(
    monkeypatch: pytest.MonkeyPatch, capture_outcome: str | None
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        session = BoundarySession(
            registry,
            role="test worker",
            pack="test",
            codec=ValueCodec(registry, use_shm=False),
            reservations=GovernorReservationService(governor),
        )

        async def drop_frame(header: Mapping[str, object], *_args: object) -> None:
            if header.get("type") in ("shmAck", "resultAck") and reader.at_eof():
                raise WorkerDied()

        class Writer:
            def close(self) -> None:
                return None

            async def wait_closed(self) -> None:
                return None

        session._send_unlocked = drop_frame  # type: ignore[method-assign]  # noqa: SLF001
        session._writer = Writer()  # type: ignore[assignment]  # noqa: SLF001
        session._alive = True  # noqa: SLF001 - drive a negotiated session
        reader = asyncio.StreamReader()
        session._reader = reader  # noqa: SLF001 - drive the production receive path
        session._reader_task = asyncio.create_task(session._read_loop())  # noqa: SLF001
        entered, release = threading.Event(), threading.Event()
        capture = session._produced_assets.capture

        def blocked_capture(
            outputs: Mapping[str, Value],
            transferred: AssetResolver | None = None,
            inputs: Mapping[str, Value] | None = None,
        ) -> None:
            entered.set()
            assert release.wait(5)
            if capture_outcome == "error":
                raise AssetError("adoption failed after peer EOF")
            capture(outputs, transferred, inputs)

        if capture_outcome is not None:
            monkeypatch.setattr(session._produced_assets, "capture", blocked_capture)
        invocation = make_invocation(60)
        pending = asyncio.create_task(session.invoke(invocation))
        await asyncio.sleep(0)
        session._start_lease(  # noqa: SLF001 - peer memoryReserve frame
            {
                "requestId": invocation.invocation_id,
                "requests": [{"residency": "ram", "nbytes": 60}],
            }
        )
        while governor.reserved("ram") != 60:
            await asyncio.sleep(0)

        _feed_frame(
            reader,
            {
                "type": "result",
                "invocationId": invocation.invocation_id,
                "executeMs": 0.0,
                "outputs": {},
                "outputStats": {},
            },
        )
        try:
            if capture_outcome is not None:
                # Complete the lease so adoption can start before the transport dies.
                _feed_frame(
                    reader, {"type": "memoryRelease", "requestId": invocation.invocation_id}
                )
                assert await asyncio.to_thread(entered.wait, 5)
            reader.feed_eof()
            if capture_outcome is not None:
                async with asyncio.timeout(1):
                    while not session._closing:
                        await asyncio.sleep(0)
                assert not pending.done()
                assert session._asset_adoptions
            release.set()
            result = await asyncio.wait_for(pending, 1)

            if capture_outcome == "error":
                assert result.error is not None
                assert result.error.message == "adoption failed after peer EOF"
            else:
                assert result.error is None
                assert result.outputs == {}
            assert governor.reserved("ram") == 0
            assert session._leases == {}  # noqa: SLF001
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await session.close()

    asyncio.run(scenario())


def test_isolated_lease_grant_hold_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            task = asyncio.create_task(
                worker.invoke(hog_invocation(worker, registry, 60, seconds=0.2))
            )
            # The reservation is visible in the parent's governor while the
            # child executes...
            async with asyncio.timeout(5):
                while governor.reserved("ram") != 60:
                    await asyncio.sleep(0.005)
            result = await task
            assert result.error is None
            # ...and gone once the child releases.
            async with asyncio.timeout(5):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_denial_is_a_node_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            result = await worker.invoke(hog_invocation(worker, registry, 200))
            assert result.error is not None
            assert "memory admission failed" in result.error.message
            assert governor.reserved("ram") == 0
            # The worker survives a denial: the next fitting hog runs.
            ok = await worker.invoke(hog_invocation(worker, registry, 50, invocation_id="i2"))
            assert ok.error is None
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_released_when_node_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            invocation = Invocation(
                invocation_id="i1",
                node_id="b",
                node_type="iso.boom",
                inputs={"tag": registry.wrap(CORE_STRING, "t")},
                effective_schema=worker.schemas["iso.boom"],
            )
            result = await worker.invoke(invocation)
            assert result.error is not None
            assert "boom" in result.error.message
            async with asyncio.timeout(5):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_leases_serialize_over_a_constrained_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            tasks = [
                asyncio.create_task(
                    worker.invoke(
                        hog_invocation(worker, registry, 60, seconds=0.15, invocation_id=f"i{n}")
                    )
                )
                for n in (1, 2)
            ]
            # Two 60-byte hogs on a 100-byte budget can never both hold.
            while not all(t.done() for t in tasks):
                assert governor.reserved("ram") <= 100
                await asyncio.sleep(0.005)
            results = [t.result() for t in tasks]
            assert all(r.error is None for r in results)
            assert governor.reserved("ram") == 0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_survives_cancellation_while_blocked(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            async with governor.reserve("ram", 100):  # occupy the whole budget
                blocked = asyncio.create_task(worker.invoke(hog_invocation(worker, registry, 60)))
                await asyncio.sleep(0.1)  # child is waiting on its grant
                assert not blocked.done()
                blocked.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await blocked
            # Budget free again and nothing leaked: a new hog runs to completion.
            async with asyncio.timeout(5):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
            ok = await worker.invoke(hog_invocation(worker, registry, 50, invocation_id="i2"))
            assert ok.error is None
            async with asyncio.timeout(5):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_lease_without_service_grants_unaccounted(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor=None)
        await worker.start()
        try:
            result = await worker.invoke(
                hog_invocation(worker, registry, 10**12)  # nothing budgets this
            )
            assert result.error is None
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_close_unwinds_held_leases(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        task = asyncio.create_task(
            worker.invoke(hog_invocation(worker, registry, 60, seconds=30.0))
        )
        async with asyncio.timeout(5):
            while governor.reserved("ram") != 60:
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await worker.close()  # must not hang; leases die with the worker
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


# -- compat checkpoint policy ----------------------------------------------


def asset_invocation(size: int, *, arm: str | None = None) -> Invocation:
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry)
    ref = AssetRef(digest=DIGEST, name="model.safetensors", size=size)
    return Invocation(
        invocation_id="i1",
        node_id="load",
        node_type="dinkster.load_checkpoint",
        inputs={"checkpoint": registry.wrap(ASSET_TYPE, ref)},
        effective_schema=NodeSchema(
            node_type="dinkster.load_checkpoint",
            display_name="Load Checkpoint",
            category="compat",
            inputs=(InputSpec("checkpoint", TypeExpr.concrete(ASSET_TYPE)),),
            outputs=(),
        ),
        arm=arm,
    )


def test_comfy_arm_checkpoint_planner_keeps_2_6_ram_factor() -> None:
    plan = plan_reservations(asset_invocation(1_000))
    assert len(plan) == 1
    assert plan[0].residency == "ram"
    assert plan[0].nbytes == int(1_000 * CHECKPOINT_RAM_FACTOR)


def test_native_arm_checkpoint_planner_uses_1_5_ram_and_zero_vram() -> None:
    plan = plan_reservations(asset_invocation(1_001, arm="native"))
    assert plan == (ReservationRequest("ram", math.ceil(1_001 * NATIVE_CHECKPOINT_RAM_FACTOR)),)


def test_compat_planner_refuses_sizeless_assets() -> None:
    with pytest.raises(ReservationPlanError, match="byte size"):
        plan_reservations(asset_invocation(0))


def test_compat_planner_ignores_other_node_types() -> None:
    invocation = asset_invocation(1_000)
    other = Invocation(
        invocation_id=invocation.invocation_id,
        node_id=invocation.node_id,
        node_type="comfy.vae_decode",
        inputs=invocation.inputs,
        effective_schema=invocation.effective_schema,
    )
    assert plan_reservations(other) == ()


def resident_input(
    resource_id: str | None,
    costs: dict[str, object],
    *,
    type_id: str = "test.resident",
) -> Value:
    registry = TypeRegistry()
    registry.register(
        type_id,
        meta=lambda obj: {
            COST_META_KEY: costs,
            **({RESOURCE_ID_META_KEY: resource_id} if resource_id is not None else {}),
        },
    )
    return registry.wrap(type_id, object())


def invocation_with_inputs(**inputs: Value) -> Invocation:
    return Invocation(
        invocation_id="i-vram",
        node_id="use",
        node_type="comfy.use_model",
        inputs=inputs,
        effective_schema=NodeSchema(
            node_type="comfy.use_model",
            display_name="Use Model",
            category="test",
            inputs=(),
            outputs=(),
        ),
    )


def test_compat_planner_reserves_unloaded_and_unknown_residents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidentPool(cost_of=lambda obj: {COST_META_KEY: {"vram:cuda:0": 100, "ram": 50}})
    rid = pool.rid_for(object())
    asyncio.run(pool.shed(PressureSignal(device="vram:cuda:0", bytes_needed=100)))
    monkeypatch.setattr("dinkster_compat_comfy.memory.default_pool", lambda: pool)
    invocation = invocation_with_inputs(
        known=resident_input(f"resident:{rid}", {"vram:cuda:0": 100, "ram": 50}),
        unknown=resident_input("resident:old-worker", {"vram:cuda:1": 200}),
    )
    assert plan_reservations(invocation) == (
        ReservationRequest("vram:cuda:0", 100),
        ReservationRequest("vram:cuda:1", 200),
    )


def test_compat_planner_skips_loaded_residents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidentPool(cost_of=lambda obj: {COST_META_KEY: {"vram:cuda:0": 100}})
    rid = pool.rid_for(object())
    monkeypatch.setattr("dinkster_compat_comfy.memory.default_pool", lambda: pool)
    invocation = invocation_with_inputs(
        model=resident_input(f"resident:{rid}", {"vram:cuda:0": 100})
    )
    assert plan_reservations(invocation) == ()


def test_compat_planner_deduplicates_resident_ids_and_walks_nested_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_compat_comfy.memory.default_pool", lambda: ResidentPool())
    child = resident_input("resident:shared", {"vram:cuda:0": 100})
    inner = make_list_value("test.resident", (child, child))
    outer = make_list_value("list<test.resident>", (inner,))
    invocation = invocation_with_inputs(first=child, nested=outer)
    assert plan_reservations(invocation) == (ReservationRequest("vram:cuda:0", 100),)


def test_compat_planner_reserves_anonymous_vram_costs_without_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_compat_comfy.memory.default_pool", lambda: ResidentPool())
    invocation = invocation_with_inputs(
        first=resident_input(None, {"vram:cuda:0": 75}),
        second=resident_input(None, {"vram:cuda:0": 75}),
    )
    assert plan_reservations(invocation) == (
        ReservationRequest("vram:cuda:0", 75),
        ReservationRequest("vram:cuda:0", 75),
    )


def test_checkpoint_ram_policy_composes_with_input_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_compat_comfy.memory.default_pool", lambda: ResidentPool())
    base = asset_invocation(1_000)
    invocation = Invocation(
        invocation_id=base.invocation_id,
        node_id=base.node_id,
        node_type=base.node_type,
        inputs={
            **base.inputs,
            "model": resident_input("resident:old-worker", {"vram:cuda:0": 125}),
        },
        effective_schema=base.effective_schema,
    )
    assert plan_reservations(invocation) == (
        ReservationRequest("vram:cuda:0", 125),
        ReservationRequest("ram", int(1_000 * CHECKPOINT_RAM_FACTOR)),
    )
