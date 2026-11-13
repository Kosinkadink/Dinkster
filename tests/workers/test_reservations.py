"""Reservation-before-allocation at the worker boundary (DESIGN 3.10).

The engine's lanes bound *execution*; these tests prove the second gate -
*allocation* - holds everywhere it must: the request contracts, the
governor-backed service, the GovernedWorker wrapper, the isolated lease
protocol over the wire, and the native checkpoint policy. The
invariant throughout: a reservation is held across execution and released
on every exit path - success, node error, denial, cancellation, and worker
death.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
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
)
from dinkster_workers.reservations import (
    plan_reservations,
)
from dinkster_workers.resident_pool import ResidentPool

from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parents[1]

DIGEST = "blake3:" + "ab" * 32


# -- ReservationRequest contracts ----------------------------------------


# -- GovernorReservationService ------------------------------------------


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

    run_scenario(scenario())


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


def observe_reservation_wait(
    governor: MemoryGovernor, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Event:
    waiting = asyncio.Event()
    original_wait = governor._cond.wait

    async def wait() -> bool:
        waiting.set()
        return await original_wait()

    monkeypatch.setattr(governor._cond, "wait", wait)
    return waiting


def test_governed_worker_holds_reservation_across_execute() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        worker, fake = governed(governor)
        result = await worker.invoke(make_invocation(60))
        assert result.error is None
        assert fake.reserved_during == [60]  # held while executing
        assert governor.reserved("ram") == 0  # released after

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_governed_worker_releases_when_inner_raises() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        fake = FakeWorker(governor)
        fake.fail_with = RuntimeError("execute blew up")
        worker, _ = governed(governor, fake)
        with pytest.raises(RuntimeError, match="execute blew up"):
            await worker.invoke(make_invocation(60))
        assert governor.reserved("ram") == 0

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_governed_worker_serializes_over_a_constrained_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        waiting = observe_reservation_wait(governor, monkeypatch)
        fake = FakeWorker(governor)
        fake.gate = asyncio.Event()
        worker, _ = governed(governor, fake)
        first = asyncio.create_task(worker.invoke(make_invocation(60, "i1")))
        async with asyncio.timeout(5):
            while governor.reserved("ram") != 60:
                await asyncio.sleep(0)
        # Second does not fit until the first releases: both cannot believe
        # the same bytes are free.
        second = asyncio.create_task(worker.invoke(make_invocation(60, "i2")))
        await asyncio.wait_for(waiting.wait(), 5)
        assert governor.reserved("ram") == 60
        assert fake.reserved_during == [60]  # second not executing
        fake.gate.set()
        results = await asyncio.gather(first, second)
        assert all(r.error is None for r in results)
        assert governor.reserved("ram") == 0

    run_scenario(scenario())


# -- manifest entry -------------------------------------------------------


# -- isolated lease protocol ----------------------------------------------


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
        'type-namespaces = ["iso"]\n\n[pack.entry]\n'
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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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
            async with asyncio.timeout(5):
                while not all(t.done() for t in tasks):
                    assert governor.reserved("ram") <= 100
                    await asyncio.sleep(0.005)
            results = [t.result() for t in tasks]
            assert all(r.error is None for r in results)
            assert governor.reserved("ram") == 0
        finally:
            await worker.close()

    run_scenario(scenario())


def test_isolated_lease_survives_cancellation_while_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        waiting = observe_reservation_wait(governor, monkeypatch)
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, governor)
        await worker.start()
        try:
            async with governor.reserve("ram", 100):  # occupy the whole budget
                blocked = asyncio.create_task(worker.invoke(hog_invocation(worker, registry, 60)))
                await asyncio.wait_for(waiting.wait(), 5)
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

    run_scenario(scenario())


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

    run_scenario(scenario())


# -- checkpoint policy -----------------------------------------------------


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


def test_checkpoint_planner_reserves_unloaded_and_unknown_residents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidentPool(cost_of=lambda obj: {COST_META_KEY: {"vram:cuda:0": 100, "ram": 50}})
    rid = pool.rid_for(object())
    run_scenario(pool.shed(PressureSignal(device="vram:cuda:0", bytes_needed=100)))
    monkeypatch.setattr("dinkster_workers.reservations.default_pool", lambda: pool)
    invocation = invocation_with_inputs(
        known=resident_input(f"resident:{rid}", {"vram:cuda:0": 100, "ram": 50}),
        unknown=resident_input("resident:old-worker", {"vram:cuda:1": 200}),
    )
    assert plan_reservations(invocation) == (
        ReservationRequest("vram:cuda:0", 100),
        ReservationRequest("vram:cuda:1", 200),
    )


def test_checkpoint_planner_skips_loaded_residents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidentPool(cost_of=lambda obj: {COST_META_KEY: {"vram:cuda:0": 100}})
    rid = pool.rid_for(object())
    monkeypatch.setattr("dinkster_workers.reservations.default_pool", lambda: pool)
    invocation = invocation_with_inputs(
        model=resident_input(f"resident:{rid}", {"vram:cuda:0": 100})
    )
    assert plan_reservations(invocation) == ()


def test_checkpoint_planner_deduplicates_resident_ids_and_walks_nested_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_workers.reservations.default_pool", lambda: ResidentPool())
    child = resident_input("resident:shared", {"vram:cuda:0": 100})
    inner = make_list_value("test.resident", (child, child))
    outer = make_list_value("list<test.resident>", (inner,))
    invocation = invocation_with_inputs(first=child, nested=outer)
    assert plan_reservations(invocation) == (ReservationRequest("vram:cuda:0", 100),)
