"""ServingComposer enrollment for stage-6 execution dispatch."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy.workgroup import SingleJobWorkGroupHandler
from dinkster_engine import (
    Engine,
    EngineEvent,
    ExecutionError,
    ExecutionSelection,
    ProviderResolutionError,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import ReservationRequest
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    CacheKey,
    Invocation,
    InvocationEvent,
    InvocationResult,
    NodeError,
    ReleaseWorkGroup,
    ReplicaId,
    RunWorkUnit,
    WorkGroupDefinition,
    WorkGroupLifecycle,
    WorkGroupMessage,
    WorkGroupState,
    WorkUnitId,
    derive_attention_route_token,
)
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComboOption,
    ComboWidget,
    ComfyAliasConfidence,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    schema_signature,
)
from dinkster_server import PackInfo
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_PRODUCER_ARM_META_KEY,
    RESOURCES_META_KEY,
    TypeRegistry,
    Value,
    ValueMeta,
    list_children,
)
from dinkster_values.model import PyObjPayload
from dinkster_workers import DispatchWorker, GroupMemberWorker, ReplicaEndpoint, load_manifest

from dinkster.compose import (
    ArmRecord,
    CompositionError,
    PackSpec,
    ServingComposer,
    _ReplicaLane,
    _ResidencyDomain,
    _SingleJobWorkerPool,
    compose_serving,
)


def _resource_value(resource_id: str, rank: int) -> Value:
    return Value(
        "test.resource",
        "same-logical-value",
        ValueMeta(
            {
                RESOURCE_ID_META_KEY: resource_id,
                RESOURCE_OWNER_META_KEY: f"owner-{rank}",
                RESOURCE_PRODUCER_ARM_META_KEY: f"arm-{rank}",
                RESOURCES_META_KEY: {"gpu": f"cuda:{rank}"},
            }
        ),
        PyObjPayload(f"resource-{rank}"),
    )


class _SingleJobFakeWorker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.calls: list[Invocation] = []
        self.closed = False
        self.block = False
        self.error: NodeError | None = None
        self.exception: BaseException | None = None
        self.cancelled = False
        self.gate: asyncio.Event | None = None
        self.close_delay = 0.0

    async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
        self.calls.append(invocation)
        if on_event is not None:
            on_event(InvocationEvent("rank", {"rank": self.rank}))
        try:
            if self.block:
                await asyncio.Event().wait()
            if self.exception is not None:
                raise self.exception
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            return InvocationResult(error=self.error)
        return InvocationResult(
            outputs={"out": _resource_value(f"resource-{self.rank}", self.rank)}
        )

    async def close(self) -> None:
        self.closed = True
        await asyncio.sleep(self.close_delay)


class _WorkGroupFakeWorker(_SingleJobFakeWorker):
    workgroup_capabilities = frozenset({WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY})
    schemas: Mapping[str, NodeSchema] = {}
    combo_choices: Mapping[str, tuple[str, ...]] = {}
    lazy_choice_ids: tuple[str, ...] = ()
    compat_skips: Mapping[str, object] = {}
    body_arms = None
    extension_contributions = None
    attention_capabilities = None
    attention_route_token = None

    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        self.instance_token = f"worker-{rank}"
        self.handler = SingleJobWorkGroupHandler()
        self.unbound = 0
        self.release_failure = False
        self.run_transport_failure = False
        self.started = False
        self.start_error: BaseException | None = None

    @property
    def alive(self) -> bool:
        return self.started and not self.closed

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def bind_workgroup_endpoint(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
    ) -> ReplicaEndpoint:
        queue: asyncio.Queue[WorkGroupMessage] = asyncio.Queue()

        async def send(message: WorkGroupMessage) -> None:
            if self.run_transport_failure and type(message) is RunWorkUnit:
                raise RuntimeError("run transport failed")
            replies = await self.handler(message)
            if self.release_failure and type(message) is ReleaseWorkGroup:
                raise RuntimeError("release transport failed")
            for reply in replies:
                await queue.put(reply)

        return ReplicaEndpoint.bind(definition, replica, send=send, receive=queue.get)

    def unbind_workgroup_endpoint(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
    ) -> None:
        del definition, replica
        if not self.closed:
            assert not self.handler._attempts  # pyright: ignore[reportPrivateUsage]
        self.unbound += 1

    async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
        await self.handler.before_invocation(invocation.invocation_id)
        result = await super().invoke(invocation, on_event=on_event)
        await self.handler.after_invocation(
            invocation.invocation_id,
            None if result.error is None else result.error.message,
        )
        return result


class _Reservations:
    @asynccontextmanager
    async def reserve(self, requests: tuple[ReservationRequest, ...]):
        assert requests == ()
        yield


class _RecordingReservations:
    def __init__(self) -> None:
        self.calls: list[tuple[ReservationRequest, ...]] = []

    @asynccontextmanager
    async def reserve(self, requests: tuple[ReservationRequest, ...]):
        self.calls.append(requests)
        yield


class _RefusingReservations:
    def __init__(self) -> None:
        self.calls: list[tuple[ReservationRequest, ...]] = []

    @asynccontextmanager
    async def reserve(self, requests: tuple[ReservationRequest, ...]):
        self.calls.append(requests)
        raise RuntimeError("admission refused")
        yield


def _single_job_pool(
    tmp_path: Path,
) -> tuple[_SingleJobWorkerPool, tuple[_SingleJobFakeWorker, ...]]:
    workers = (_SingleJobFakeWorker(0), _SingleJobFakeWorker(1))
    lanes = tuple(_ReplicaLane(rank, worker) for rank, worker in enumerate(workers))  # type: ignore[arg-type]
    pool = _SingleJobWorkerPool(lanes, tmp_path / "rendezvous", "guidance")
    pool._invoke_workgroup = pool._invoke_ranks  # type: ignore[method-assign]
    return pool, workers


def _single_job_invocation(inputs: Mapping[str, Value] | None = None) -> Invocation:
    return Invocation(
        "invocation",
        "node",
        "test.node",
        {} if inputs is None else inputs,
        NodeSchema(node_type="test.node"),
    )


def test_single_job_pool_preserves_rank_local_resources_and_rank_zero_events(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        pool, workers = _single_job_pool(tmp_path)
        events: list[InvocationEvent] = []

        produced = await pool.invoke(_single_job_invocation(), on_event=events.append)
        assert produced.outputs is not None
        leader = produced.outputs["out"]
        assert leader.meta.get(RESOURCES_META_KEY) == {"gpu": ("cuda:0", "cuda:1")}
        assert Engine._occupancy(  # pyright: ignore[reportPrivateUsage]
            NodeSchema(node_type="test.gpu", occupies=("gpu",)),
            {"resource": leader},
            None,
        ) == ("gpu:cuda:0", "gpu:cuda:1")
        schema = NodeSchema(node_type="test.gpu", occupies=("gpu",))
        engine = Engine(
            schemas={schema.node_type: schema},
            registry=TypeRegistry(),
            worker=workers[0],  # type: ignore[arg-type]
            cache=MemoryLRUCache(),
        )
        competing_entered = asyncio.Event()

        async def compete_for_rank_one() -> None:
            async with engine._admission(  # pyright: ignore[reportPrivateUsage]
                schema,
                {"resource": _resource_value("competing", 1)},
                None,
            ):
                competing_entered.set()

        async with engine._admission(  # pyright: ignore[reportPrivateUsage]
            schema,
            {"resource": leader},
            None,
        ):
            competitor = asyncio.create_task(compete_for_rank_one())
            await asyncio.sleep(0)
            assert not competing_entered.is_set()
        await asyncio.wait_for(competing_entered.wait(), timeout=1.0)
        await competitor
        await pool.invoke(_single_job_invocation({"resource": leader}))

        assert events == [InvocationEvent("rank", {"rank": 0})]
        for rank, worker in enumerate(workers):
            received = worker.calls[1].inputs["resource"]
            assert received.resolve() == f"resource-{rank}"
            assert received.meta.get(RESOURCE_OWNER_META_KEY) == f"owner-{rank}"
            assert received.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == f"arm-{rank}"

    asyncio.run(scenario())


def test_single_job_pool_cancels_siblings_and_closes_after_rank_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        pool, workers = _single_job_pool(tmp_path)
        workers[0].block = True
        workers[1].error = NodeError("node", "test.node", "rank failed")

        result = await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)

        assert result.error == workers[1].error
        assert workers[0].cancelled
        assert all(worker.closed for worker in workers)

    asyncio.run(scenario())


def test_single_job_pool_replaces_every_rank_after_workgroup_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        generations: list[tuple[_WorkGroupFakeWorker, ...]] = []

        def make_lanes() -> tuple[_ReplicaLane, ...]:
            workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
            generations.append(workers)
            return tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            )

        pool = _SingleJobWorkerPool(
            make_lanes(),  # type: ignore[arg-type]
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
            make_lanes,
        )

        first = await pool.invoke(_single_job_invocation())
        assert first.error is None
        generations[0][0].error = NodeError("node", "test.node", "rank failed")
        generations[0][1].block = True
        failed = await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)
        assert failed.error == generations[0][0].error
        assert all(worker.closed for worker in generations[0])
        assert generations[0][1].cancelled
        assert all(worker.started for worker in generations[1])

        recovered = await pool.invoke(_single_job_invocation())
        assert recovered.error is None
        assert [worker.unbound for worker in generations[1]] == [1, 1]
        assert all(not worker.closed for worker in generations[1])

    asyncio.run(scenario())


def test_single_job_pool_surfaces_rank_reason_when_workgroup_concludes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
        for worker in workers:
            worker.block = True

        class FailedCoordinator:
            def __init__(self, reservations: object) -> None:
                del reservations

            async def execute(
                self,
                definition: WorkGroupDefinition,
                *args: object,
                started: asyncio.Event,
                **kwargs: object,
            ) -> WorkGroupLifecycle:
                del args
                started.set()
                while not all(worker.calls for worker in workers):
                    await asyncio.sleep(0)
                on_failure = kwargs["on_failure"]
                assert callable(on_failure)
                on_failure()
                replicas = tuple(member.replica for member in definition.members)
                units = tuple(unit.unit for unit in definition.units)
                return WorkGroupLifecycle(
                    definition,
                    state=WorkGroupState.FAILED,
                    prepared=replicas,
                    dispatched=units,
                    cancelled=replicas,
                    released=replicas,
                    failures=((WorkUnitId("sample-1"), "CUDA kernel exploded"),),
                )

        monkeypatch.setattr("dinkster.compose.WorkGroupCoordinator", FailedCoordinator)
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
        )

        with pytest.raises(
            RuntimeError,
            match="workgroup concluded with rank failures: sample-1: CUDA kernel exploded",
        ):
            await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)

        assert all(worker.cancelled for worker in workers)

    asyncio.run(scenario())


def test_single_job_pool_awaits_rank_replies_after_successful_workgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        replies_allowed = asyncio.Event()

        class DelayedReplyWorker(_WorkGroupFakeWorker):
            async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
                self.calls.append(invocation)
                await replies_allowed.wait()
                return InvocationResult(
                    outputs={"out": _resource_value(f"resource-{self.rank}", self.rank)}
                )

        workers = (DelayedReplyWorker(0), DelayedReplyWorker(1))

        class SuccessfulCoordinator:
            def __init__(self, reservations: object) -> None:
                del reservations

            async def execute(
                self,
                definition: WorkGroupDefinition,
                *args: object,
                started: asyncio.Event,
                **kwargs: object,
            ) -> WorkGroupLifecycle:
                del args, kwargs
                started.set()
                while not all(worker.calls for worker in workers):
                    await asyncio.sleep(0)
                asyncio.get_running_loop().call_later(0.01, replies_allowed.set)
                replicas = tuple(member.replica for member in definition.members)
                units = tuple(unit.unit for unit in definition.units)
                return WorkGroupLifecycle(
                    definition,
                    state=WorkGroupState.SUCCEEDED,
                    prepared=replicas,
                    dispatched=units,
                    completed=units,
                    released=replicas,
                )

        monkeypatch.setattr("dinkster.compose.WorkGroupCoordinator", SuccessfulCoordinator)
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
        )

        result = await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)

        assert result.error is None
        assert result.outputs is not None
        assert result.outputs["out"].meta.get(RESOURCES_META_KEY) == {"gpu": ("cuda:0", "cuda:1")}
        assert all(not worker.closed for worker in workers)

    asyncio.run(scenario())


@pytest.mark.parametrize("rank_exception", [False, True], ids=("node-error", "exception"))
@pytest.mark.parametrize(
    "coordinator_first", [False, True], ids=("rank-first", "coordinator-first")
)
def test_single_job_pool_preserves_rank_failure_when_coordinator_transport_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_exception: bool,
    coordinator_first: bool,
) -> None:
    async def scenario() -> None:
        workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
        for worker in workers:
            worker.close_delay = 0.1
        if rank_exception:
            workers[0].exception = RuntimeError("primary rank failure")
        else:
            workers[0].error = NodeError("node", "test.node", "primary rank failure")
        workers[1].block = True

        class FailingCoordinator:
            def __init__(self, reservations: object) -> None:
                del reservations

            async def execute(
                self, *args: object, started: asyncio.Event, **kwargs: object
            ) -> None:
                del args
                on_failure = kwargs["on_failure"]
                assert callable(on_failure)
                started.set()
                if coordinator_first:
                    on_failure()
                while not workers[0].closed:
                    await asyncio.sleep(0)
                if not coordinator_first:
                    on_failure()
                raise RuntimeError("transport lost")

        monkeypatch.setattr("dinkster.compose.WorkGroupCoordinator", FailingCoordinator)
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
        )

        if coordinator_first:
            with pytest.raises(RuntimeError, match="transport lost"):
                await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)
        elif rank_exception:
            with pytest.raises(RuntimeError, match="primary rank failure"):
                await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)
        else:
            result = await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)
            assert result.error == workers[0].error
        assert workers[1].cancelled

    asyncio.run(scenario())


def test_single_job_pool_retries_failed_replacement_startup(tmp_path: Path) -> None:
    async def scenario() -> None:
        generations: list[tuple[_WorkGroupFakeWorker, ...]] = []

        def make_lanes() -> tuple[_ReplicaLane, ...]:
            workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
            if len(generations) == 1:
                workers[0].start_error = RuntimeError("replacement startup failed")
            generations.append(workers)
            return tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            )

        pool = _SingleJobWorkerPool(
            make_lanes(),
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
            make_lanes,
        )
        await pool.start()
        generations[0][0].error = NodeError("node", "test.node", "rank failed")

        with pytest.raises(RuntimeError, match="replacement startup failed"):
            await pool.invoke(_single_job_invocation())

        assert pool.alive
        assert pool.instance_token is not None
        recovered = await pool.invoke(_single_job_invocation())
        assert recovered.error is None
        assert len(generations) == 3
        assert all(worker.started for worker in generations[2])

    asyncio.run(scenario())


def test_single_job_pool_serializes_complete_workgroup_attempts(tmp_path: Path) -> None:
    async def scenario() -> None:
        pool, workers = _single_job_pool(tmp_path)
        gate = asyncio.Event()
        for worker in workers:
            worker.gate = gate

        first = asyncio.create_task(pool.invoke(_single_job_invocation()))
        while any(len(worker.calls) < 1 for worker in workers):
            await asyncio.sleep(0)
        second = asyncio.create_task(pool.invoke(_single_job_invocation()))
        await asyncio.sleep(0.01)
        assert [len(worker.calls) for worker in workers] == [1, 1]

        gate.set()
        await asyncio.gather(first, second)
        assert [len(worker.calls) for worker in workers] == [2, 2]

    asyncio.run(scenario())


def test_single_job_pool_closes_all_ranks_when_attempt_teardown_fails(tmp_path: Path) -> None:
    async def scenario() -> None:
        workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
        workers[1].release_failure = True
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
        )

        with pytest.raises(RuntimeError, match="transport lost"):
            await pool.invoke(_single_job_invocation())

        assert all(worker.closed for worker in workers)
        assert [worker.unbound for worker in workers] == [1, 1]

    asyncio.run(scenario())


def test_single_job_pool_replaces_blocked_ranks_when_coordinator_fails(tmp_path: Path) -> None:
    async def scenario() -> None:
        generations: list[tuple[_WorkGroupFakeWorker, ...]] = []

        def make_lanes() -> tuple[_ReplicaLane, ...]:
            workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
            generations.append(workers)
            return tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            )

        pool = _SingleJobWorkerPool(
            make_lanes(),  # type: ignore[arg-type]
            tmp_path / "rendezvous",
            "guidance",
            _Reservations(),  # type: ignore[arg-type]
            make_lanes,
        )
        generations[0][0].run_transport_failure = True

        with pytest.raises(RuntimeError, match="transport lost"):
            await asyncio.wait_for(pool.invoke(_single_job_invocation()), timeout=1.0)

        assert all(worker.closed for worker in generations[0])
        assert all(worker.started for worker in generations[1])
        recovered = await pool.invoke(_single_job_invocation())
        assert recovered.error is None

    asyncio.run(scenario())


def test_single_job_pool_batches_all_rank_vram_reservations(tmp_path: Path) -> None:
    async def scenario() -> None:
        workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
        reservations = _RecordingReservations()
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            reservations,  # type: ignore[arg-type]
        )
        resident = Value(
            "test.resource",
            "resident",
            ValueMeta(
                {
                    RESOURCE_ID_META_KEY: "model",
                    COST_META_KEY: {"vram:cuda:0": 100},
                }
            ),
            PyObjPayload("resident"),
        )

        result = await pool.invoke(_single_job_invocation({"runtime": resident}))

        assert result.error is None
        assert reservations.calls == [
            (
                ReservationRequest("vram:cuda:0", 100),
                ReservationRequest("vram:cuda:0", 100),
            )
        ]

    asyncio.run(scenario())


def test_single_job_pool_starts_no_rank_when_atomic_admission_is_refused(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workers = (_WorkGroupFakeWorker(0), _WorkGroupFakeWorker(1))
        reservations = _RefusingReservations()
        pool = _SingleJobWorkerPool(
            tuple(
                _ReplicaLane(rank, worker)  # type: ignore[arg-type]
                for rank, worker in enumerate(workers)
            ),
            tmp_path / "rendezvous",
            "guidance",
            reservations,  # type: ignore[arg-type]
        )

        with pytest.raises(RuntimeError, match="admission refused"):
            await pool.invoke(_single_job_invocation())

        assert reservations.calls == [()]
        assert all(not worker.calls for worker in workers)
        assert all(not worker.closed for worker in workers)
        assert [worker.unbound for worker in workers] == [1, 1]

    asyncio.run(scenario())


def _attention_token(*, policy: str = "auto") -> AttentionRouteToken:
    return derive_attention_route_token(
        _attention_capabilities(),
        AttentionPolicyConfig(policy),  # type: ignore[arg-type]
    )


def _attention_capabilities(*, kitchen: bool = False) -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa", "dinkster_kitchen_int8") if kitchen else ("sdpa",),
        provider_versions=(
            (("dinkster-kitchen", "0.2.31"), ("torch", "2.13.0"))
            if kitchen
            else (("torch", "2.13.0"),)
        ),
    )


def _write_flash_attention_provider(root: Path) -> None:
    (root / "dinkster_inference_torch.py").write_text(
        """from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    derive_attention_route_token,
)

EVIDENCE = AttentionCapabilityEvidence(
    version=1,
    device_kind="cpu",
    device_sm=None,
    sdpa_torch_runtime="2.13.0",
    adapter_contract_revision="dinkster.attention-kernel.v1",
    available_policies=("sdpa", "flash"),
    provider_versions=(("torch", "2.13.0"),),
)

def discover_attention_capabilities():
    return EVIDENCE

def discover_attention_route_token(policy="auto"):
    return derive_attention_route_token(EVIDENCE, AttentionPolicyConfig(policy))
""",
        encoding="utf-8",
    )


def _write_pack(
    root: Path,
    name: str,
    node_type: str,
    *,
    executes: tuple[str, ...] = (),
    own_type: str | None = None,
    mismatched: bool = False,
    body_arm: str | None = None,
    schema_only: bool = False,
    lazy: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    module = name.replace("-", "_") + "_nodes"
    extra_input = ', InputSpec("extra", STRING, default="")' if mismatched else ""
    lazy_input = ", lazy=True" if lazy else ""
    lazy_hook = (
        """
    @classmethod
    def check_lazy_status(cls, *, value: str, **kwargs: object) -> tuple[str, ...]:
        return ()
"""
        if lazy
        else ""
    )
    own_class = ""
    nodes = "Echo"
    arm_class = ""
    arm_nodes = ""
    if body_arm is not None:
        arm_class = """

class AltEcho(Echo):
    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        await __import__("asyncio").sleep(0)
        from dinkster_workers import current_execution_context
        context = current_execution_context()
        identity = context.expected_execution_identity if context else "missing"
        fp8 = context.fp8_matmul if context else "missing"
        return cls.outputs(value="alt:" + value + ":" + str(identity) + ":" + str(fp8))
"""
        arm_nodes = f"\nARM_NODES = {{{body_arm!r}: [AltEcho]}}\n"
    if own_type is not None:
        own_class = f"""

class Own(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type={own_type!r},
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value="own:" + value)
"""
        nodes += ", Own"
    (root / f"{module}.py").write_text(
        f"""from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry

STRING = TypeExpr.concrete("core.string")

def register_types(registry: TypeRegistry) -> None:
    pass

class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type={node_type!r},
            inputs=(InputSpec("value", STRING{lazy_input}){extra_input},),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(value={name!r} + ":" + value)
{lazy_hook}
{own_class}
{arm_class}
NODES = [{nodes}]
{arm_nodes}
"""
    )
    namespace = (
        own_type.partition(".")[0]
        if own_type
        else name
        if executes
        else node_type.partition(".")[0]
    )
    executes_line = (
        "executes = [" + ", ".join(f'"{item}"' for item in executes) + "]\n" if executes else ""
    )
    schema_only_line = f'schema-only = ["{node_type}"]\n' if schema_only else ""
    arms_table = f'\n[pack.arms]\n{body_arm} = ["{node_type}"]\n' if body_arm else ""
    arm_entry = f'arm_nodes = "{module}:ARM_NODES"\n' if body_arm else ""
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["{namespace}"]\n'
        f"{executes_line}{schema_only_line}{arms_table}\n[pack.entry]\n"
        f'nodes = "{module}:NODES"\ntypes = "{module}:register_types"\n{arm_entry}'
    )
    return manifest


def test_server_attention_default_is_not_forwarded_to_pack_workers() -> None:
    composer = ServingComposer(
        worker_env={"DINKSTER_ATTENTION_POLICY": "flash", "INHERITED": "yes"}
    )
    spec = PackSpec(
        "pack.toml",
        env={"DINKSTER_ATTENTION_POLICY": "sage", "EXPLICIT": "yes"},
    )

    environment = composer._worker_environment(spec, ())

    assert "DINKSTER_ATTENTION_POLICY" not in environment
    assert environment["INHERITED"] == "yes"
    assert environment["EXPLICIT"] == "yes"


def _write_vision_provider_packs(
    root: Path, *, provider_choice: str = "vision.providers"
) -> tuple[Path, Path]:
    from dinkster_assets import digest_bytes

    owner_root = root / "owner"
    provider_root = root / "provider"
    owner_root.mkdir(parents=True)
    provider_root.mkdir(parents=True)
    common = """from collections.abc import Mapping

from dinkster_api.v1 import ComboWidget, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

STRING = TypeExpr.concrete("core.string")
COMBO = TypeExpr.concrete("core.combo")

class Vision(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision.process",
            inputs=(
                InputSpec("value", STRING),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route="/api/choices/vision.providers"),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("value", STRING),),
        )
"""
    (owner_root / "vision_owner_nodes.py").write_text(
        common
        + """
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("vision provider required")

NODES = [Vision]

def choices() -> dict[str, tuple[str, ...]]:
    return {"vision.providers": ()}
""",
        encoding="utf-8",
    )
    owner_manifest = owner_root / "dinkster-pack.toml"
    owner_manifest.write_text(
        """[pack]
name = "vision-owner"
namespaces = ["vision"]
[pack.entry]
nodes = "vision_owner_nodes:NODES"
choices = "vision_owner_nodes:choices"
""",
        encoding="utf-8",
    )

    model_bytes = b"provider-model"
    assets = provider_root / "assets"
    assets.mkdir()
    (assets / "model.bin").write_bytes(model_bytes)
    (provider_root / "vision_provider_nodes.py").write_text(
        common
        + """
    @classmethod
    def execute(cls, *, value: str, **_inputs: object) -> Mapping[str, object]:
        from dinkster_api.v1 import declared_asset
        model = declared_asset("model").read_bytes().decode("utf-8")
        return cls.outputs(value=f"{model}:{value}")

NODES = [Vision]
""",
        encoding="utf-8",
    )
    provider_manifest = provider_root / "dinkster-pack.toml"
    provider_manifest.write_text(
        f"""[pack]
name = "depth-provider"
namespaces = []
executes = ["vision.process"]
[[pack.vision-providers]]
choice = "{provider_choice}"
node = "vision.process"
devices = ["cpu"]
dtypes = ["float32"]
batching = "batch"
artifacts = ["model"]
[[pack.assets]]
id = "model"
name = "Test model"
digest = "{digest_bytes(model_bytes)}"
file = "assets/model.bin"
[pack.entry]
nodes = "vision_provider_nodes:NODES"
""",
        encoding="utf-8",
    )
    return owner_manifest, provider_manifest


def test_vision_provider_manifest_populates_choices_and_routes_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        import sys

        from dinkster_assets import AssetVault, digest_bytes

        owner_manifest, provider_manifest = _write_vision_provider_packs(tmp_path)
        vault = AssetVault(tmp_path / "vault")
        with vault.writer(digest_bytes(b"provider-model")) as writer:
            writer.write(b"provider-model")
            writer.commit()
        worker_env = _env(owner_manifest.parent, provider_manifest.parent)
        worker_env["DINKSTER_ASSET_VAULT"] = str(vault.root)
        composer = ServingComposer(worker_env=worker_env)
        try:
            owner_delta = await composer.add_pack(owner_manifest)
            assert owner_delta.derived_choices == {}
            assert composer.composition.choices["vision.providers"] == ()
            owner_signature = schema_signature(composer.composition.schemas["vision.process"])

            provider_delta = await composer.add_pack(provider_manifest)
            assert provider_delta.derived_choices == {"vision.providers": ("depth-provider",)}
            assert composer.composition.choices["vision.providers"] == ("depth-provider",)
            assert (
                schema_signature(composer.composition.schemas["vision.process"]) == owner_signature
            )
            assert load_manifest(provider_manifest).vision_providers[0].artifacts == ("model",)
            assert set(
                composer.composition.asset_catalog.needs_for_nodes(
                    (),
                    provider_selections={("vision.process", "depth-provider")},
                )
            ) == {digest_bytes(b"provider-model")}
            assert (
                composer.composition.asset_catalog.needs_for_nodes(
                    (),
                    provider_selections={("vision.process", "missing")},
                )
                == {}
            )
            assert "vision_provider_nodes" not in sys.modules
            reloaded = await composer.reload_pack("depth-provider")
            assert reloaded.delta.derived_choices == {}
            assert composer.composition.choices["vision.providers"] == ("depth-provider",)

            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            graph = Graph(
                nodes={
                    "n": GraphNode(
                        "vision.process",
                        {"value": "input", "provider": "depth-provider"},
                    )
                }
            )
            result = await engine.run(graph, ["n"])
            assert result.outputs["n"]["value"].resolve() == "provider-model:input"

            automatic = Graph(nodes={"n": GraphNode("vision.process", {"value": "automatic"})})
            automatic_result = await engine.run(automatic, ["n"])
            assert automatic_result.outputs["n"]["value"].resolve() == ("provider-model:automatic")
            finished = next(
                event
                for event in reversed(events)
                if event.kind == "node_finished" and event.node_id == "n"
            )
            assert finished.detail["provider"] == "depth-provider"
            assert finished.detail["pack"] == "depth-provider"
            assert finished.detail["worker"] == "local"

            unavailable = Graph(
                nodes={
                    "n": GraphNode(
                        "vision.process",
                        {"value": "input", "provider": "missing"},
                    )
                }
            )
            with pytest.raises(
                ProviderResolutionError,
                match="compatible vision-processing implementation is unavailable",
            ):
                await engine.run(unavailable, ["n"])

            removal = await composer.remove_pack("depth-provider")
            assert removal.derived_choices == {"vision.providers": ()}
            assert composer.composition.choices["vision.providers"] == ()
            with pytest.raises(
                ProviderResolutionError,
                match="compatible vision-processing implementation is unavailable",
            ):
                await composer.composition.make_engine(lambda _event: None).run(graph, ["n"])
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_vision_provider_remote_placement_uses_observable_old_peer_fallback() -> None:
    worker = SimpleNamespace(alive=True)
    topology = {
        "vision.process": (
            ArmRecord(
                name="box1",
                worker=worker,
                domain=_ResidencyDomain(worker, "box1"),
                instance_token=lambda: "worker-token",
                default_cache_tag="unversioned",
                default_arm="box1",
                owner_worker=worker,
                remote="box1",
                implementation_pack="legacy-depth-pack",
                vision_provider="legacy-depth-pack",
                provider_declared=False,
            ),
        )
    }
    graph = Graph(nodes={"n": GraphNode("vision.process", {"value": "input"})})

    resolved = ServingComposer()._resolve_providers(
        graph,
        topology=topology,
        routes={},
        models={},
        provider_nodes={"vision.process"},
        generation_nodes=(),
        placement={"n": "box1"},
        remote_names={"box1"},
        schemas={"vision.process": NodeSchema("vision.process", display_name="Vision Process")},
    )

    assert resolved.nodes["n"].inputs["provider"] == "legacy-depth-pack"


def test_vision_provider_remote_authoritative_empty_evidence_is_unsupported() -> None:
    worker = SimpleNamespace(alive=True)
    topology = {
        "vision.process": (
            ArmRecord(
                name="box1",
                worker=worker,
                domain=_ResidencyDomain(worker, "box1"),
                instance_token=lambda: "worker-token",
                default_cache_tag="unversioned",
                default_arm="box1",
                owner_worker=worker,
                remote="box1",
                implementation_pack="ordinary-pack",
            ),
        )
    }
    graph = Graph(nodes={"n": GraphNode("vision.process", {"value": "input"})})

    with pytest.raises(ProviderResolutionError, match="Vision Process cannot run"):
        ServingComposer()._resolve_providers(
            graph,
            topology=topology,
            routes={},
            models={},
            provider_nodes={"vision.process"},
            generation_nodes=(),
            placement={"n": "box1"},
            remote_names={"box1"},
            schemas={"vision.process": NodeSchema("vision.process", display_name="Vision Process")},
        )


def test_vision_provider_resolution_refuses_dead_group_member() -> None:
    session = SimpleNamespace(alive=False, instance_token=None)
    worker = GroupMemberWorker(SimpleNamespace(), session)  # type: ignore[arg-type]
    topology = {
        "vision.process": (
            ArmRecord(
                name="depth-provider",
                worker=worker,
                domain=_ResidencyDomain(worker, "depth-provider"),
                instance_token=lambda: worker.instance_token,
                default_cache_tag="unversioned",
                default_arm="depth-provider",
                owner_worker=worker,
            ),
        )
    }
    composer = ServingComposer()

    for inputs in (
        {"value": "automatic"},
        {"value": "pinned", "provider": "depth-provider"},
    ):
        graph = Graph(nodes={"n": GraphNode("vision.process", inputs)})
        with pytest.raises(ProviderResolutionError) as raised:
            composer._resolve_providers(
                graph,
                topology=topology,
                routes={("vision.process", "depth-provider"): "depth-provider"},
                models={("vision.process", "depth-provider"): None},
                provider_nodes={"vision.process"},
                generation_nodes=(),
                placement={},
                remote_names=(),
                schemas={
                    "vision.process": NodeSchema("vision.process", display_name="Vision Process")
                },
            )
        assert raised.value.node_id == "n"
        assert raised.value.node_type == "vision.process"
        assert raised.value.title == "Vision Process"
        assert "compatible vision-processing implementation" in str(raised.value)
        assert "depth-provider" not in str(raised.value)


def test_vision_provider_choice_mismatch_fails_atomically(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_manifest, provider_manifest = _write_vision_provider_packs(
            tmp_path, provider_choice="vision.other"
        )
        composer = ServingComposer(worker_env=_env(owner_manifest.parent, provider_manifest.parent))
        try:
            await composer.add_pack(owner_manifest)
            with pytest.raises(CompositionError, match="does not match.*provider input"):
                await composer.add_pack(provider_manifest)
            assert set(composer._records) == {"vision-owner"}
            assert composer.composition.choices["vision.providers"] == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_node_cannot_use_both_vision_and_generation_provider_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_manifest, provider_manifest = _write_vision_provider_packs(tmp_path)
        provider_manifest.write_text(
            provider_manifest.read_text().replace(
                "[[pack.assets]]",
                "[[pack.generation-providers]]\n"
                'choice = "vision.providers"\n'
                'node = "vision.process"\n'
                "[[pack.assets]]",
            )
        )
        composer = ServingComposer(worker_env=_env(owner_manifest.parent, provider_manifest.parent))
        try:
            await composer.add_pack(owner_manifest)
            with pytest.raises(CompositionError, match="both vision and generation"):
                await composer.add_pack(provider_manifest)
            assert set(composer._records) == {"vision-owner"}
        finally:
            await composer.close()

    asyncio.run(scenario())


def _write_generation_provider_packs(root: Path) -> tuple[Path, Path, Path]:
    owner_root = root / "generation-owner"
    native_root = root / "generation-native"
    external_root = root / "generation-external"
    for directory in (owner_root, native_root, external_root):
        directory.mkdir(parents=True)
    schema = """from collections.abc import Mapping

from dinkster_api.v1 import ComboWidget, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

STRING = TypeExpr.concrete("core.string")
COMBO = TypeExpr.concrete("core.combo")

class Generate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.generate",
            inputs=(
                InputSpec("clip", STRING, required=False, lazy=True),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route="/api/choices/fixture.generation.providers"),
                    hidden=True,
                ),
                InputSpec("value", STRING),
            ),
            outputs=(OutputSpec("value", STRING),),
        )
"""
    (owner_root / "generation_owner.py").write_text(
        schema
        + """
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("generation provider required")

class Fail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.fail",
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("lazy clip branch executed")

class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.source",
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value="resident")

NODES = [Generate, Fail, Source]

def choices() -> dict[str, tuple[str, ...]]:
    return {"fixture.generation.providers": ()}
""",
        encoding="utf-8",
    )
    owner_manifest = owner_root / "dinkster-pack.toml"
    owner_manifest.write_text(
        """[pack]
name = "generation-owner"
namespaces = ["fixture"]
schema-only = ["fixture.generate"]
[pack.capabilities]
"dinkster.generation.schemas" = "1.0.0"
[pack.entry]
nodes = "generation_owner:NODES"
choices = "generation_owner:choices"
""",
        encoding="utf-8",
    )

    (native_root / "generation_native.py").write_text(
        schema
        + """
    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, *, clip: str, value: str, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value=f"native:{clip}:{value}")

NODES = [Generate]
""",
        encoding="utf-8",
    )
    native_manifest = native_root / "dinkster-pack.toml"
    native_manifest.write_text(
        """[pack]
name = "generation-native"
namespaces = []
executes = ["fixture.generate"]
[pack.entry]
nodes = "generation_native:NODES"
""",
        encoding="utf-8",
    )

    (external_root / "generation_external.py").write_text(
        schema
        + """
    @classmethod
    def check_lazy_status(cls, **_inputs: object) -> tuple[str, ...]:
        return ()

    @classmethod
    def execute(cls, *, value: str, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value=f"external:{value}")

NODES = [Generate]
""",
        encoding="utf-8",
    )
    external_manifest = external_root / "dinkster-pack.toml"
    external_manifest.write_text(
        """[pack]
name = "generation-external"
namespaces = []
executes = ["fixture.generate"]
[[pack.generation-providers]]
choice = "fixture.generation.providers"
node = "fixture.generate"
[pack.entry]
nodes = "generation_external:NODES"
""",
        encoding="utf-8",
    )
    return owner_manifest, native_manifest, external_manifest


def test_generation_provider_is_explicit_and_suppresses_lazy_native_inputs(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner, native, external = _write_generation_provider_packs(tmp_path)
        composer = ServingComposer(worker_env=_env(owner.parent, native.parent, external.parent))
        try:
            await composer.add_pack(owner)
            delta = await composer.add_pack(external)
            await composer.add_pack(native)
            assert delta.derived_choices == {
                "fixture.generation.providers": ("generation-external",)
            }
            provider_input = composer.composition.schemas["fixture.generate"].input("provider")
            assert provider_input is not None
            assert provider_input.widget == ComboWidget(
                options=(
                    ComboOption("builtin", "Built-in"),
                    ComboOption("generation-external", "generation-external"),
                )
            )
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            native_graph = Graph(
                nodes={
                    "source": GraphNode("fixture.source"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {"clip": Link("source", "value"), "value": "prompt"},
                    ),
                }
            )
            native_result = await engine.run(native_graph, ["generate"])
            assert native_result.outputs["generate"]["value"].resolve() == (
                "native:resident:prompt"
            )
            native_finished = next(
                event
                for event in events
                if event.kind == "node_finished" and event.node_id == "generate"
            )
            assert native_finished.detail["provider"] == "builtin"
            builtin_graph = Graph(
                nodes={
                    "source": GraphNode("fixture.source"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {
                            "clip": Link("source", "value"),
                            "provider": "builtin",
                            "value": "builtin prompt",
                        },
                    ),
                }
            )
            before_builtin = len(events)
            builtin_result = await engine.run(builtin_graph, ["generate"])
            assert builtin_result.outputs["generate"]["value"].resolve() == (
                "native:resident:builtin prompt"
            )
            builtin_finished = next(
                event
                for event in events[before_builtin:]
                if event.kind == "node_finished" and event.node_id == "generate"
            )
            assert builtin_finished.detail["provider"] == "builtin"

            external_graph = Graph(
                nodes={
                    "fail": GraphNode("fixture.fail"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {
                            "clip": Link("fail", "value"),
                            "provider": "generation-external",
                            "value": "prompt",
                        },
                    ),
                }
            )
            external_result = await engine.run(external_graph, ["generate"])
            assert external_result.outputs["generate"]["value"].resolve() == "external:prompt"

            removal = await composer.remove_pack("generation-external")
            assert removal.derived_choices == {"fixture.generation.providers": ()}
            provider_input = composer.composition.schemas["fixture.generate"].input("provider")
            assert provider_input is not None
            assert provider_input.widget == ComboWidget(
                options=(ComboOption("builtin", "Built-in"),)
            )
            with pytest.raises(ExecutionError, match="unavailable generation provider"):
                await composer.composition.make_engine(lambda _event: None).run(
                    external_graph, ["generate"]
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_composer_rejects_native_hello_without_attention_evidence(tmp_path: Path) -> None:
    manifest = load_manifest(
        _write_pack(tmp_path / "native", "nativepack", "nativepack.echo", body_arm="native")
    )
    worker = SimpleNamespace(
        body_arms={"native": ("nativepack.echo",)},
        attention_route_token=None,
    )
    composer = ServingComposer()

    with pytest.raises(CompositionError, match="omitted attention route evidence"):
        composer._validate_body_arms(manifest, worker)


@pytest.mark.parametrize("with_policy", [False, True])
def test_composer_fallback_propagates_arm_attention_policy_and_token(
    with_policy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        token = _attention_token(policy="sdpa")
        worker = SimpleNamespace(alive=True)
        domain = _ResidencyDomain(worker, "nativepack")
        arm = ArmRecord(
            name="nativepack@native",
            worker=worker,
            domain=domain,
            instance_token=lambda: "worker-token",
            default_cache_tag="native:identity",
            default_arm="nativepack",
            owner_worker=worker,
            attention_route_token=token,
        )
        composer = ServingComposer()
        selected = await composer._plan_execution(
            "nativepack.echo",
            NodeSchema(node_type="nativepack.echo"),
            {},
            topology={"nativepack.echo": (arm,)},
        )

        assert selected is not None
        assert selected.attention_policy == "sdpa"
        assert selected.attention_route_token is token
        assert selected.attention_diagnostic is None

        if with_policy:

            async def select(*args: object, **kwargs: object) -> ExecutionSelection:
                return ExecutionSelection(
                    target=arm.name,
                    cache_tag=arm.default_cache_tag,
                    attention_policy=token.requested_policy,
                    attention_route_token=token,
                )

            monkeypatch.setattr(composer, "_native_policy", SimpleNamespace(select=select))
        selected = await composer._plan_execution(
            "nativepack.echo",
            NodeSchema(node_type="nativepack.echo"),
            {},
            attention_config=AttentionPolicyConfig("sage"),
            topology={"nativepack.echo": (arm,)},
        )
        assert selected is not None
        assert selected.attention_route_token is token
        assert selected.attention_policy == "sdpa"
        assert selected.attention_diagnostic is not None
        assert "missing capability evidence" in selected.attention_diagnostic
        assert "'requestedPolicy': 'sage'" in selected.attention_diagnostic
        assert "unet=sdpa" in selected.attention_diagnostic

    asyncio.run(scenario())


@pytest.mark.parametrize("remote", [None, "remote-utility"])
def test_composer_keeps_utility_arms_for_explicit_attention(remote: str | None) -> None:
    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        arm = ArmRecord(
            name="utility",
            worker=worker,
            domain=_ResidencyDomain(worker, "utility"),
            instance_token=lambda: "worker-token",
            default_cache_tag="utility:identity",
            default_arm="utility",
            owner_worker=worker,
            remote=remote,
        )
        composer = ServingComposer()
        topology = {"utility.echo": (arm,)}

        selected = await composer._plan_execution(
            "utility.echo",
            NodeSchema(node_type="utility.echo"),
            {},
            topology=topology,
            preferred_worker=remote,
            remote_names=frozenset({remote}) if remote is not None else frozenset(),
        )
        assert selected is not None
        assert selected.attention_policy == "auto"
        assert selected.attention_route_token is None

        for config in (
            AttentionPolicyConfig("flash"),
            AttentionPolicyConfig(requested_role_policies=(("flux", "sage"),)),
        ):
            selected = await composer._plan_execution(
                "utility.echo",
                NodeSchema(node_type="utility.echo"),
                {},
                attention_config=config,
                topology=topology,
                preferred_worker=remote,
                remote_names=frozenset({remote}) if remote is not None else frozenset(),
            )
            assert selected is not None
            assert selected.attention_policy == "auto"
            assert selected.attention_route_token is None
            assert selected.attention_diagnostic is not None
            assert "missing capability evidence" in selected.attention_diagnostic
            assert "default auto route without an attention token" in selected.attention_diagnostic
            assert config.requested_policy in selected.attention_diagnostic

    asyncio.run(scenario())


@pytest.mark.parametrize("existing_token", [False, True])
@pytest.mark.parametrize("remote", [None, "remote-fallback"])
def test_composer_keeps_arm_when_requested_route_derivation_fails(
    existing_token: bool, remote: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _attention_capabilities()
    token = derive_attention_route_token(evidence, AttentionPolicyConfig())

    def derive(
        capabilities: AttentionCapabilityEvidence, config: AttentionPolicyConfig
    ) -> AttentionRouteToken:
        if config != AttentionPolicyConfig():
            raise ValueError("requested route not implemented")
        return derive_attention_route_token(capabilities, config)

    monkeypatch.setattr("dinkster.compose.derive_attention_route_token", derive)

    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        arm = ArmRecord(
            name="fallback",
            worker=worker,
            domain=_ResidencyDomain(worker, "fallback"),
            instance_token=lambda: "worker-token",
            default_cache_tag="native:identity",
            default_arm="fallback",
            owner_worker=worker,
            remote=remote,
            attention_capabilities=evidence,
            attention_route_token=token if existing_token else None,
        )
        selected = await ServingComposer()._plan_execution(
            "fallback.echo",
            NodeSchema(node_type="fallback.echo"),
            {},
            attention_config=AttentionPolicyConfig(requested_role_policies=(("flux", "sage"),)),
            topology={"fallback.echo": (arm,)},
            preferred_worker=remote,
            remote_names=frozenset({remote}) if remote is not None else frozenset(),
        )
        assert selected is not None
        assert selected.worker == (remote or "local")
        assert selected.attention_route_token == token
        assert selected.attention_policy == "auto"
        assert selected.attention_diagnostic is not None
        assert "requested route not implemented" in selected.attention_diagnostic
        assert "['flux', 'sage']" in selected.attention_diagnostic
        assert "flux=sdpa" in selected.attention_diagnostic

    asyncio.run(scenario())


def test_composer_propagates_default_route_derivation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _attention_capabilities()

    def derive(
        capabilities: AttentionCapabilityEvidence, config: AttentionPolicyConfig
    ) -> AttentionRouteToken:
        raise ValueError(
            "default route unavailable"
            if config == AttentionPolicyConfig()
            else "requested route unavailable"
        )

    monkeypatch.setattr("dinkster.compose.derive_attention_route_token", derive)

    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        arm = ArmRecord(
            name="fallback",
            worker=worker,
            domain=_ResidencyDomain(worker, "fallback"),
            instance_token=lambda: "worker-token",
            default_cache_tag="native:identity",
            default_arm="fallback",
            owner_worker=worker,
            attention_capabilities=evidence,
        )
        with pytest.raises(ValueError, match="default route unavailable"):
            await ServingComposer()._plan_execution(
                "fallback.echo",
                NodeSchema(node_type="fallback.echo"),
                {},
                attention_config=AttentionPolicyConfig("sage"),
                topology={"fallback.echo": (arm,)},
            )

    asyncio.run(scenario())


def test_composer_keeps_portable_attention_arms_and_respects_placement() -> None:
    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        domain = _ResidencyDomain(worker, "nativepack")

        def arm(name: str, *, kitchen: bool) -> ArmRecord:
            return ArmRecord(
                name=name,
                worker=worker,
                domain=domain,
                instance_token=lambda: "worker-token",
                default_cache_tag=f"native:{name}",
                default_arm=name,
                owner_worker=worker,
                attention_capabilities=_attention_capabilities(kitchen=kitchen),
                remote=name,
            )

        weak = arm("weak", kitchen=False)
        strong = arm("strong", kitchen=True)
        topology = {"nativepack.echo": (weak, strong)}
        config = AttentionPolicyConfig("dinkster_kitchen_int8")
        composer = ServingComposer()

        selected = await composer._plan_execution(
            "nativepack.echo",
            NodeSchema(node_type="nativepack.echo"),
            {},
            attention_config=config,
            topology=topology,
            remote_names=frozenset({"weak", "strong"}),
        )
        assert selected is not None
        assert selected.target == "weak"
        assert selected.attention_policy == "dinkster_kitchen_int8"
        assert selected.attention_route_token is not None
        assert selected.attention_route_token.requested_policy == "dinkster_kitchen_int8"
        assert selected.attention_route_token.version == 3
        assert all(route.primary == "sdpa" for route in selected.attention_route_token.routes)

        for preferred in ("weak", "strong"):
            selected = await composer._plan_execution(
                "nativepack.echo",
                NodeSchema(node_type="nativepack.echo"),
                {},
                attention_config=config,
                topology=topology,
                preferred_worker=preferred,
                remote_names=frozenset({"weak", "strong"}),
            )
            assert selected is not None
            assert selected.target == preferred
            assert selected.attention_route_token is not None
            assert selected.attention_route_token.version == (3 if preferred == "weak" else 1)

    asyncio.run(scenario())


def test_composer_follows_resident_producer_without_native_policy() -> None:
    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        domain = _ResidencyDomain(worker, "compat")

        def arm(name: str) -> ArmRecord:
            return ArmRecord(
                name=name,
                worker=worker,
                domain=domain,
                instance_token=lambda: "worker-token",
                default_cache_tag=f"identity:{name}",
                default_arm="compat",
                owner_worker=worker,
            )

        value = _resource_value("model", 1)
        value = replace(
            value,
            meta=ValueMeta(
                {
                    **value.meta.entries,
                    RESOURCE_PRODUCER_ARM_META_KEY: "dinkster-compat-comfy@native",
                }
            ),
        )
        selected = await ServingComposer()._plan_execution(
            "dinkster.model_sampling_aura_flow",
            NodeSchema(node_type="dinkster.model_sampling_aura_flow"),
            {"model": value},
            topology={
                "dinkster.model_sampling_aura_flow": (
                    arm("dinkster-compat-comfy"),
                    arm("dinkster-compat-comfy@native"),
                )
            },
        )

        assert selected is not None
        assert selected.target == "dinkster-compat-comfy@native"
        assert selected.cache_tag == "identity:dinkster-compat-comfy@native"

    asyncio.run(scenario())


def test_job_attention_policy_crosses_isolated_invocation_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "routed"
        manifest = _write_pack(root, "routed", "routed.echo", body_arm="native")
        _write_flash_attention_provider(root)
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            engine = composer.composition.make_engine(lambda _event: None)
            cases = (
                ("scalar", AttentionPolicyConfig("flash")),
                (
                    "role",
                    AttentionPolicyConfig(requested_role_policies=(("flux", "flash"),)),
                ),
            )
            for value, config in cases:
                result = await engine.run(
                    _graph("routed.echo", value),
                    ["n"],
                    attention_config=config,
                )
                assert result.outputs["n"]["value"].resolve() == f"routed:{value}"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_job_attention_policy_crosses_isolated_lazy_status_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "lazy-routed"
        manifest = _write_pack(
            root,
            "lazy-routed",
            "lazy-routed.echo",
            body_arm="native",
            lazy=True,
        )
        _write_flash_attention_provider(root)
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            engine = composer.composition.make_engine(lambda _event: None)
            cases = (
                ("default", None),
                ("non-default", AttentionPolicyConfig("flash")),
            )
            for value, config in cases:
                result = await engine.run(
                    _graph("lazy-routed.echo", value),
                    ["n"],
                    attention_config=config,
                )
                assert result.outputs["n"]["value"].resolve() == f"lazy-routed:{value}"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_composer_fallback_keeps_requested_policy_in_cache_identity() -> None:
    async def scenario() -> None:
        worker = SimpleNamespace(alive=True)
        domain = _ResidencyDomain(worker, "nativepack")
        arm = ArmRecord(
            name="nativepack@native",
            worker=worker,
            domain=domain,
            instance_token=lambda: "worker-token",
            default_cache_tag="native:identity",
            default_arm="nativepack",
            owner_worker=worker,
            attention_capabilities=_attention_capabilities(),
        )
        composer = ServingComposer()

        class CacheSpy(MemoryLRUCache):
            get_calls = 0
            put_calls = 0

            async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
                self.get_calls += 1
                return await super().get(key)

            async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
                self.put_calls += 1
                await super().put(key, outputs)

        class WorkerSpy:
            tokens: list[AttentionRouteToken] = []

            async def prepare(self, node_types: object) -> None:
                assert node_types == ["nativepack.echo"]

            async def invoke(
                self, invocation: Invocation, on_event: object = None
            ) -> InvocationResult:
                assert invocation.attention_route_token is not None
                self.tokens.append(invocation.attention_route_token)
                return InvocationResult(outputs={})

        schema = NodeSchema(node_type="nativepack.echo")
        cache = CacheSpy()
        execution_worker = WorkerSpy()

        async def plan(
            node_id: str,
            node_type: str,
            planned_schema: NodeSchema,
            inputs: object,
            run_id: str,
            attention_config: object,
        ) -> object:
            return await composer._plan_execution(
                node_type,
                planned_schema,
                inputs,  # type: ignore[arg-type]
                run_id,
                attention_config,  # type: ignore[arg-type]
                topology={node_type: (arm,)},
            )

        engine = Engine(
            schemas={schema.node_type: schema},
            registry=TypeRegistry(),
            worker=execution_worker,  # type: ignore[arg-type]
            cache=cache,
            plan_execution=plan,  # type: ignore[arg-type]
        )
        for policy in ("dinkster_kitchen_int8", "sdpa", "dinkster_kitchen_int8"):
            await engine.run(
                Graph(nodes={"n": GraphNode(schema.node_type, {})}),
                ["n"],
                attention_config=AttentionPolicyConfig(policy),
            )

        assert cache.get_calls == 3
        assert cache.put_calls == 2
        assert [token.requested_policy for token in execution_worker.tokens] == [
            "dinkster_kitchen_int8",
            "sdpa",
        ]
        assert execution_worker.tokens[0].version == 3
        assert execution_worker.tokens[1].version == 1

    asyncio.run(scenario())


def _env(*roots: Path) -> dict[str, str]:
    return {"PYTHONPATH": os.pathsep.join(str(root) for root in roots)}


def _write_resident_arm_pack(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "resident_nodes.py").write_text(
        """from collections.abc import Mapping
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import RESOURCE_HANDLE_TYPE, ResourceHandle, register_resource_handle_type

def register_types(registry):
    register_resource_handle_type(registry)

class Resident(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="resident.make",
            inputs=(),
            outputs=(OutputSpec("out", TypeExpr.list_of(TypeExpr.concrete(RESOURCE_HANDLE_TYPE))),),
        )
    @classmethod
    def execute(cls):
        return cls.outputs(out=[
            ResourceHandle(resource_id="local", kind="model", obj=object()),
            ResourceHandle(resource_id="foreign", kind="model", owner="foreign-session"),
        ])

class AltResident(Resident):
    pass

NODES = [Resident]
ARM_NODES = {"alt": [AltResident]}
"""
    )
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        """[pack]
name = "resident"
[pack.arms]
alt = ["resident.make"]
[pack.entry]
nodes = "resident_nodes:NODES"
types = "resident_nodes:register_types"
arm_nodes = "resident_nodes:ARM_NODES"
"""
    )
    return manifest


def _graph(node_type: str, value: str = "x") -> Graph:
    return Graph(nodes={"n": GraphNode(node_type, {"value": value})})


def test_enrollment_publishes_dispatch_and_keeps_claimed_schema_private(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        arm_root = tmp_path / "arm"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        arm_manifest = _write_pack(
            arm_root,
            "alternative",
            "owned.echo",
            executes=("owned.echo",),
            own_type="arm.own",
        )
        composer = ServingComposer(worker_env=_env(owner_root, arm_root))
        try:
            await composer.add_pack(owner_manifest)
            route = composer._routing._routes["owned.echo"]
            assert isinstance(route, DispatchWorker)
            token = composer._records["owner"].worker.instance_token
            worker = composer._records["owner"].worker

            delta = await composer.add_pack(arm_manifest)
            assert composer._records["owner"].worker is worker
            assert worker.instance_token == token
            assert isinstance(composer._routing._routes["owned.echo"], DispatchWorker)
            assert "owned.echo" not in delta.schemas
            assert "owned.echo" not in delta.node_packs
            assert "arm.own" in delta.schemas
            assert composer.composition.node_packs["owned.echo"] == "owner"
            assert composer.composition.node_packs["arm.own"] == "alternative"
            assert [arm.name for arm in composer._topology["owned.echo"]] == [
                "owner",
                "alternative",
            ]

            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(_graph("owned.echo", "ok"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "owner:ok"
            selection = await composer._plan_execution(
                "owned.echo", composer.composition.schemas["owned.echo"], {}
            )
            assert selection is not None
            assert selection.target == "owner"
            assert selection.cache_tag == "unversioned"
            assert token is not None and composer._owner_alive(token)

            alternative = composer._records["alternative"].worker
            alternative_token = alternative.instance_token
            assert alternative_token is not None
            alternative._session._instance_token = token
            with pytest.raises(RuntimeError, match="share instance token"):
                composer._owner_alive(token)
            dispatch = composer._routing._routes["owned.echo"]
            assert isinstance(dispatch, DispatchWorker)
            with pytest.raises(RuntimeError, match="share instance token"):
                dispatch._resolve_owner(token)
            alternative._session._instance_token = alternative_token
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_cross_pack_provider_may_publish_same_session_body_arms(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
            body_arm="alt",
        )
        composer = ServingComposer(worker_env=_env(owner_root, provider_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(provider_manifest)

            arms = composer._topology["owned.echo"]
            assert [arm.name for arm in arms] == ["owner", "provider", "provider@alt"]
            assert arms[1].domain is arms[2].domain
            assert composer.composition.node_packs["owned.echo"] == "owner"

            route = composer._routing._routes["owned.echo"]
            schema = composer.composition.schemas["owned.echo"]
            registry = composer.composition._registry

            def invocation(target: str) -> Invocation:
                return Invocation(
                    invocation_id=target,
                    node_id=target,
                    node_type="owned.echo",
                    inputs={"value": registry.wrap("core.string", "ok")},
                    effective_schema=schema,
                    executor=target,
                    expected_execution_identity="identity",
                )

            default = await route.invoke(invocation("provider"))
            alternate = await route.invoke(invocation("provider@alt"))
            assert default.outputs is not None
            assert default.outputs["value"].resolve() == "provider:ok"
            assert alternate.outputs is not None
            assert alternate.outputs["value"].resolve() == "alt:ok:identity:False"
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("optional_execution", [False, True])
def test_schema_only_owner_routes_only_after_provider_composes(
    tmp_path: Path, optional_execution: bool
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(
            owner_root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
            body_arm="alt",
        )
        composer = ServingComposer(worker_env=_env(owner_root, provider_root))
        try:
            owner_delta = await composer.add_pack(
                PackSpec(
                    manifest=owner_manifest,
                    optional_execution=("owned.echo",) if optional_execution else (),
                )
            )
            assert owner_delta.schemas == {}
            assert owner_delta.execution_arms == {}
            assert composer._topology["owned.echo"] == ()
            assert "owned.echo" not in composer._routing._routes
            with (
                nullcontext()
                if optional_execution
                else pytest.raises(CompositionError, match="have no execution provider")
            ):
                composer.validate_complete_generation()
            assert composer.incomplete_generation_removals() == (
                {} if optional_execution else {"owner": ("owned.echo",)}
            )
            with pytest.raises(RuntimeError, match="has no execution provider"):
                await composer._plan_execution(
                    "owned.echo", composer.composition.schemas["owned.echo"], {}
                )

            provider_delta = await composer.add_pack(provider_manifest)
            assert provider_delta.schemas == {
                "owned.echo": composer.composition.schemas["owned.echo"]
            }
            assert provider_delta.node_packs == {"owned.echo": "owner"}
            arms = composer._topology["owned.echo"]
            assert [arm.name for arm in arms] == ["provider", "provider@alt"]
            assert composer.composition.node_packs["owned.echo"] == "owner"
            composer.validate_complete_generation()

            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(_graph("owned.echo", "ok"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "provider:ok"

            removal = await composer.remove_pack("provider")
            assert removal.removed_types == ("owned.echo",)
            assert composer._topology["owned.echo"] == ()
            assert "owned.echo" not in composer._routing._routes
            with (
                nullcontext()
                if optional_execution
                else pytest.raises(CompositionError, match="have no execution provider")
            ):
                composer.validate_complete_generation()
            assert composer.incomplete_generation_removals() == (
                {} if optional_execution else {"owner": ("owned.echo",)}
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


def _fixture_alias_registry(carrier: str) -> ComfyAliasRegistry:
    string = TypeExpr.concrete("core.string")
    source = ComfyAliasSource("comfy-core", "Echo", "comfy.Echo", "b78cec87")
    return ComfyAliasRegistry(
        source_schemas=(
            ComfyAliasSourceSchema(
                NodeSchema(
                    source.node_type,
                    inputs=(InputSpec("value", string),),
                    outputs=(OutputSpec("value", string),),
                ),
                SCHEMA_WIRE_VERSION,
            ),
        ),
        records=(
            ComfyAliasRecord(
                id="comfy_alias:comfy-core/Echo",
                mapping_kind="op",
                carrier=carrier,
                source=source,
                replacement=ReplacementRule(
                    from_type=source.node_type,
                    cases=(
                        ReplacementCase.build(
                            carrier,
                            inputs={"value": MappingSource.copy("value")},
                            outputs={"value": "value"},
                        ),
                    ),
                ),
                confidence=ComfyAliasConfidence("exact", ("tests/test_compose_dispatch.py",)),
            ),
        ),
    )


def test_schema_only_owner_delta_carries_its_registry_before_the_provider(
    tmp_path: Path,
) -> None:
    """The owner's alias registry rides its pack entry while the schema-only
    carrier stays withheld until the executing provider composes - the shape
    dinkster-serve announces when the generation schema owner precedes the
    compat worker."""

    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo", schema_only=True)
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
        )
        owner_spec = PackSpec(
            manifest=owner_manifest,
            packs={
                "owner": PackInfo(
                    "Owner",
                    comfy_aliases=_fixture_alias_registry("owned.echo"),
                )
            },
        )
        composer = ServingComposer(worker_env=_env(owner_root, provider_root))
        try:
            owner_delta = await composer.add_pack(owner_spec)
            assert owner_delta.schemas == {}
            assert owner_delta.packs["owner"].comfy_aliases is not None
            provider_delta = await composer.add_pack(provider_manifest)
            assert provider_delta.node_packs == {"owned.echo": "owner"}
            composer.validate_complete_generation()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_reload_revalidates_registry_carriers_against_the_staged_attribution(
    tmp_path: Path,
) -> None:
    """A reload's registry is checked against the attribution as it would
    look after the swap: a carrier the pack declared before the reload but
    drops in the new spec is refused even though the pre-swap composition
    still attributes it, and a carrier the new spec keeps passes."""

    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(root, "owner", "owned.echo", own_type="owned.other")
        spec = PackSpec(
            manifest=manifest,
            packs={
                "owner": PackInfo(
                    "Owner",
                    comfy_aliases=_fixture_alias_registry("owned.other"),
                )
            },
        )
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(spec)
            assert composer.composition.node_packs["owned.other"] == "owner"

            _write_pack(root, "owner", "owned.echo")
            with pytest.raises(CompositionError, match="is not declared by pack"):
                await composer.reload_pack("owner", spec)
            assert composer.composition.node_packs["owned.other"] == "owner"

            kept = PackSpec(
                manifest=manifest,
                packs={
                    "owner": PackInfo(
                        "Owner",
                        comfy_aliases=_fixture_alias_registry("owned.echo"),
                    )
                },
            )
            await composer.reload_pack("owner", kept)
            assert "owned.other" not in composer.composition.node_packs
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_add_pack_rejects_registry_carrier_the_pack_never_declares(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo", schema_only=True)
        owner_spec = PackSpec(
            manifest=owner_manifest,
            packs={
                "owner": PackInfo(
                    "Owner",
                    comfy_aliases=_fixture_alias_registry("owned.missing"),
                )
            },
        )
        composer = ServingComposer(worker_env=_env(owner_root))
        try:
            with pytest.raises(CompositionError, match="is not declared by pack"):
                await composer.add_pack(owner_spec)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_in_process_schema_owner_and_provider_share_the_native_pack_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(
            owner_root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
        )
        monkeypatch.syspath_prepend(str(owner_root))
        monkeypatch.syspath_prepend(str(provider_root))
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(owner_manifest, in_process=True))
            await composer.add_pack(PackSpec(provider_manifest, in_process=True))
            assert composer._records["owner"].domain is composer._records["provider"].domain
            result = await composer.composition.make_engine(lambda _event: None).run(
                _graph("owned.echo", "ok"), ["n"]
            )
            assert result.outputs["n"]["value"].resolve() == "provider:ok"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_atomic_composition_refuses_schema_owner_without_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(
            root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        with pytest.raises(CompositionError, match="have no execution provider"):
            await compose_serving(
                (manifest,),
                include_default_packs=False,
                worker_env=_env(root),
            )

    asyncio.run(scenario())


def test_schema_only_owner_reload_adds_and_retracts_its_body(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(
            root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            assert not composer._routing.has_route("owned.echo")

            _write_pack(root, "owner", "owned.echo")
            await composer.reload_pack("owner")
            assert composer._routing.has_route("owned.echo")
            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(_graph("owned.echo", "ok"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "owner:ok"

            _write_pack(root, "owner", "owned.echo", schema_only=True)
            await composer.reload_pack("owner")
            assert not composer._routing.has_route("owned.echo")

            removed = await composer.remove_pack("owner")
            assert removed.removed_types == ()
            assert "owned.echo" not in composer.composition.schemas
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_incomplete_generation_removes_dependents_before_schema_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        consumer_root = tmp_path / "consumer"
        owner_manifest = _write_pack(
            owner_root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        owner_manifest.write_text(
            owner_manifest.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.capabilities]\n"owner.schemas" = "1.0.0"\n\n[pack.entry]',
            )
        )
        consumer_manifest = _write_pack(
            consumer_root,
            "consumer",
            "consumer.echo",
        )
        consumer_manifest.write_text(
            consumer_manifest.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.requirements.capabilities]\n"owner.schemas" = ">=1,<2"\n\n[pack.entry]',
            )
        )
        composer = ServingComposer(worker_env=_env(owner_root, consumer_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(consumer_manifest)
            assert composer.incomplete_generation_removals() == {
                "consumer": (),
                "owner": ("owned.echo",),
            }
            for pack in composer.incomplete_generation_removals():
                await composer.remove_pack(pack)
            assert composer.composition.packs == {}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_incomplete_generation_removes_execution_provider_before_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(
            owner_root,
            "z-owner",
            "owned.echo",
            own_type="owned.other",
            schema_only=True,
        )
        owner_manifest.write_text(
            owner_manifest.read_text().replace(
                'schema-only = ["owned.echo"]',
                'schema-only = ["owned.echo", "owned.other"]',
            )
        )
        provider_manifest = _write_pack(
            provider_root,
            "a-provider",
            "owned.echo",
            executes=("owned.echo",),
        )
        composer = ServingComposer(worker_env=_env(owner_root, provider_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(provider_manifest)
            assert composer.incomplete_generation_removals() == {
                "a-provider": (),
                "z-owner": ("owned.other",),
            }
            for pack in composer.incomplete_generation_removals():
                await composer.remove_pack(pack)
            assert composer.composition.packs == {}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_incomplete_generation_rechecks_for_newly_stranded_schema_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_a_root = tmp_path / "owner-a"
        owner_b_root = tmp_path / "owner-b"
        provider_root = tmp_path / "provider"
        owner_a = _write_pack(
            owner_a_root,
            "owner-a",
            "a.missing",
            schema_only=True,
        )
        owner_a.write_text(
            owner_a.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.capabilities]\n"a.schemas" = "1.0.0"\n\n[pack.entry]',
            )
        )
        owner_b = _write_pack(
            owner_b_root,
            "owner-b",
            "b.echo",
            schema_only=True,
        )
        provider = _write_pack(
            provider_root,
            "provider",
            "b.echo",
            executes=("b.echo",),
            own_type="provider.echo",
        )
        provider.write_text(
            provider.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.requirements.capabilities]\n"a.schemas" = ">=1,<2"\n\n[pack.entry]',
            )
        )
        composer = ServingComposer(worker_env=_env(owner_a_root, owner_b_root, provider_root))
        try:
            await composer.add_pack(owner_a)
            await composer.add_pack(owner_b)
            await composer.add_pack(provider)

            first = composer.incomplete_generation_removals()
            assert first == {"provider": (), "owner-a": ("a.missing",)}
            for pack in first:
                await composer.remove_pack(pack)

            second = composer.incomplete_generation_removals()
            assert second == {"owner-b": ("b.echo",)}
            for pack in second:
                await composer.remove_pack(pack)
            composer.validate_complete_generation()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_schema_only_owner_allows_provider_owned_lazy_inputs(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(
            root,
            "owner",
            "owned.echo",
            schema_only=True,
            lazy=True,
        )
        composer = ServingComposer(worker_env=_env(root))
        try:
            delta = await composer.add_pack(manifest)
            assert delta.schemas == {}
            assert composer.incomplete_generation_removals() == {"owner": ("owned.echo",)}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_claim_validation_is_atomic_for_mismatch_missing_and_core(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        roots = [owner_root]
        composer = ServingComposer(worker_env={})
        try:
            composer._worker_env = _env(*roots)
            await composer.add_pack(owner_manifest)
            cases = (
                ("mismatch", "owned.echo", True, "schema signature"),
                ("missing", "absent.echo", False, "no owning pack"),
                ("coreclaim", "std.math.add_ints", False, "no owning pack"),
            )
            for name, claim, mismatch, message in cases:
                root = tmp_path / name
                roots.append(root)
                composer._worker_env = _env(*roots)
                manifest = _write_pack(root, name, claim, executes=(claim,), mismatched=mismatch)
                before = composer._topology
                with pytest.raises(CompositionError, match=message):
                    await composer.add_pack(manifest)
                assert composer._topology is before
                assert name not in composer.composition.packs
                assert name not in composer._records
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_owner_dependencies_and_alternative_reload_are_revalidated(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        arm_root = tmp_path / "arm"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        arm_manifest = _write_pack(
            arm_root,
            "alternative",
            "owned.echo",
            executes=("owned.echo",),
        )
        composer = ServingComposer(worker_env=_env(owner_root, arm_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(arm_manifest)
            with pytest.raises(CompositionError, match="alternative"):
                await composer.remove_pack("owner")

            _write_pack(owner_root, "owner", "owned.echo", mismatched=True)
            with pytest.raises(CompositionError, match="alternative"):
                await composer.reload_pack("owner")
            assert not composer._records["owner"].worker.schemas["owned.echo"].inputs[1:]

            _write_pack(
                arm_root,
                "alternative",
                "owned.echo",
                executes=("owned.echo",),
                mismatched=True,
            )
            with pytest.raises(CompositionError, match="schema signature"):
                await composer.reload_pack("alternative")
            assert len(composer._topology["owned.echo"]) == 2

            _write_pack(
                arm_root,
                "alternative",
                "owned.echo",
                executes=("owned.echo",),
            )
            old_topology = composer._topology
            old_route = composer._routing._routes["owned.echo"]
            await composer.reload_pack("alternative")
            reloaded_topology = composer._topology
            assert reloaded_topology is not old_topology
            assert len(old_topology["owned.echo"]) == 2
            assert len(reloaded_topology["owned.echo"]) == 2

            await composer.remove_pack("alternative")
            assert composer._topology is not reloaded_topology
            assert len(reloaded_topology["owned.echo"]) == 2
            assert [arm.name for arm in composer._topology["owned.echo"]] == ["owner"]
            assert composer._routing._routes["owned.echo"] is not old_route
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(_graph("owned.echo", "still"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "owner:still"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_execution_identity_and_dead_owner_fail_loudly(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        arm_root = tmp_path / "arm"
        manifest = _write_pack(owner_root, "owner", "owned.echo")
        arm_manifest = _write_pack(
            arm_root,
            "alternative",
            "owned.echo",
            executes=("owned.echo",),
        )
        digest = "sha256:" + "ab" * 32
        spec = PackSpec(
            manifest=manifest,
            packs={"owner": PackInfo(display_name="Owner", artifact_digest=digest)},
        )
        composer = ServingComposer(worker_env=_env(owner_root, arm_root))
        try:
            await composer.add_pack(spec)
            await composer.add_pack(arm_manifest)
            worker = composer._records["owner"].worker
            alternative = composer._records["alternative"].worker
            token = worker.instance_token
            assert token is not None
            selection = await composer._plan_execution(
                "owned.echo", composer.composition.schemas["owned.echo"], {}
            )
            assert selection is not None and selection.cache_tag == digest

            assert worker._proc is not None
            worker._proc.kill()
            await worker._proc.wait()
            for _ in range(100):
                if not worker.alive:
                    break
                await asyncio.sleep(0.01)
            assert alternative.alive
            assert [arm.name for arm in composer._topology["owned.echo"]] == [
                "owner",
                "alternative",
            ]
            assert not composer._owner_alive(token)
            with pytest.raises(RuntimeError, match="not live"):
                await composer._plan_execution(
                    "owned.echo", composer.composition.schemas["owned.echo"], {}
                )
            engine = composer.composition.make_engine(lambda event: None)
            with pytest.raises(ExecutionError):
                await engine.run(_graph("owned.echo"), ["n"])
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_reload_rotates_current_owner_token(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        manifest = _write_pack(owner_root, "owner", "owned.echo")
        composer = ServingComposer(worker_env=_env(owner_root))
        try:
            await composer.add_pack(manifest)
            old_token = composer._records["owner"].worker.instance_token
            assert old_token is not None
            await composer.reload_pack("owner")
            new_token = composer._records["owner"].worker.instance_token
            assert new_token is not None and new_token != old_token
            assert not composer._owner_alive(old_token)
            assert composer._owner_alive(new_token)
            assert composer._resolve_owner(composer._topology, old_token) is None
            domain = composer._resolve_owner(composer._topology, new_token)
            assert domain is not None and domain.default_arm == "owner"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_policy_selected_native_arm_is_liveness_validated(tmp_path: Path) -> None:
    class SelectNative:
        target = "same@native"

        async def select(
            self,
            node_type: str,
            inputs: object,
            arms: tuple[str, object],
            *,
            run_id: str | None = None,
            attention_routes: object,
        ) -> ExecutionSelection:
            assert attention_routes is not None
            return ExecutionSelection(target=self.target, cache_tag="native:identity")

    async def scenario() -> None:
        root = tmp_path / "same"
        root.mkdir(parents=True)
        (root / "dinkster_inference_torch.py").write_text(
            """from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    derive_attention_route_token,
)

def discover_attention_capabilities():
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )

def discover_attention_route_token(policy="auto"):
    return derive_attention_route_token(
        discover_attention_capabilities(), AttentionPolicyConfig(policy)
    )
"""
        )
        manifest = _write_pack(root, "same", "same.echo", body_arm="native")
        policy = SelectNative()
        composer = ServingComposer(
            worker_env=_env(root),
            native_policy=policy,  # type: ignore[arg-type]
        )
        try:
            await composer.add_pack(manifest)
            policy.target = "same@unknown"
            with pytest.raises(RuntimeError, match="selected unknown arm"):
                await composer._plan_execution(
                    "same.echo", composer.composition.schemas["same.echo"], {}
                )
            policy.target = "same@native"
            worker = composer._records["same"].worker
            assert worker._proc is not None
            worker._proc.kill()
            await worker._proc.wait()
            for _ in range(100):
                if not worker.alive:
                    break
                await asyncio.sleep(0.01)
            with pytest.raises(RuntimeError, match="same@native.*not live"):
                await composer._plan_execution(
                    "same.echo", composer.composition.schemas["same.echo"], {}
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_same_session_body_arm_routes_without_selector_bleed(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "same"
        manifest = _write_pack(root, "same", "same.echo", body_arm="alt")
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            arms = composer._topology["same.echo"]
            assert [arm.name for arm in arms] == ["same", "same@alt"]
            assert arms[0].domain is arms[1].domain
            token = composer._records["same"].worker.instance_token
            assert token is not None
            assert composer._live_token_owners(composer._topology) == {token: arms[0].domain}
            assert composer._records["same"].worker.body_arms == {"alt": ("same.echo",)}
            loaded_manifest = composer._records["same"].worker._manifest
            session = composer._records["same"].worker._session
            session._body_arms = None
            with pytest.raises(CompositionError, match="omitted required bodyArms"):
                composer._validate_body_arms(loaded_manifest, composer._records["same"].worker)
            session._body_arms = {"alt": ("same.other",)}
            with pytest.raises(CompositionError, match="does not exactly match"):
                composer._validate_body_arms(loaded_manifest, composer._records["same"].worker)
            session._body_arms = {"alt": ("same.echo",)}
            route = composer._routing._routes["same.echo"]
            schema = composer.composition.schemas["same.echo"]
            registry = composer.composition._registry

            def invocation(index: int, target: str) -> Invocation:
                return Invocation(
                    invocation_id=f"inv-{index}",
                    node_id=f"n-{index}",
                    node_type="same.echo",
                    inputs={"value": registry.wrap("core.string", str(index))},
                    effective_schema=schema,
                    executor=target,
                    expected_execution_identity=f"identity-{index}",
                    fp8_matmul=True,
                )

            results = await asyncio.gather(
                *(
                    route.invoke(invocation(i, "same" if i % 2 == 0 else "same@alt"))
                    for i in range(20)
                )
            )
            assert [result.outputs["value"].resolve() for result in results if result.outputs] == [
                ("same:" + str(i) if i % 2 == 0 else f"alt:{i}:identity-{i}:True")
                for i in range(20)
            ]

            underlying = composer._records["same"].worker
            unknown = await underlying.invoke(replace(invocation(30, "same@alt"), arm="missing"))
            assert unknown.error is not None and "unknown body arm" in unknown.error.message
            survivor = await route.invoke(invocation(31, "same@alt"))
            assert survivor.outputs is not None
            assert survivor.outputs["value"].resolve() == "alt:31:identity-31:True"

            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(_graph("same.echo", "default"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "same:default"

            old_token = underlying.instance_token
            old_domain = composer._topology["same.echo"][0].domain
            await composer.reload_pack("same")
            new_arms = composer._topology["same.echo"]
            assert new_arms[0].domain is new_arms[1].domain
            assert new_arms[0].domain is not old_domain
            assert old_token is not None and not composer._owner_alive(old_token)
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
def test_replica_topology_exposes_one_native_arm_per_requested_device(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "replicas"
        root.mkdir(parents=True)
        (root / "dinkster_inference_torch.py").write_text(
            """from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    derive_attention_route_token,
)

def discover_attention_capabilities():
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )

def discover_attention_route_token(policy="auto"):
    return derive_attention_route_token(
        discover_attention_capabilities(), AttentionPolicyConfig(policy)
    )
"""
        )
        manifest = _write_pack(root, "same", "same.echo", body_arm="native")
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(PackSpec(manifest, replica_cuda_indices=(2, 0, 1)))
            arms = composer._topology["same.echo"]
            assert [arm.name for arm in arms] == [
                "same",
                "same@native:cuda:2",
                "same@native:cuda:0",
                "same@native:cuda:1",
            ]
            assert len({id(arm.domain) for arm in arms[1:]}) == 3
            assert len({arm.instance_token() for arm in arms[1:]}) == 3
            assert (
                composer._sampling_worker({"dinkster.ksampler": arms})
                is composer._records["same"].worker
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
def test_single_job_topology_follows_replacement_rank_workers(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "single-job"
        manifest = _write_pack(root, "same", "same.echo", body_arm="alt")
        composer = ServingComposer(
            worker_env=_env(root),
            reservations=_Reservations(),  # type: ignore[arg-type]
        )
        try:
            await composer.add_pack(PackSpec(manifest, single_job_cuda_indices=(0, 1)))
            pool = composer._records["same"].worker
            assert isinstance(pool, _SingleJobWorkerPool)
            arms = composer._topology["same.echo"]
            assert [arm.name for arm in arms] == [
                "same",
                "same@alt:single-job:2:auto",
            ]
            domain = arms[0].domain
            assert domain is arms[1].domain
            assert domain.worker is pool
            old_token = pool.instance_token
            assert old_token is not None

            await pool._close_failed_lanes()  # pyright: ignore[reportPrivateUsage]
            assert not composer._owner_alive(old_token)
            recovery_token = pool.instance_token
            assert recovery_token is not None and recovery_token != old_token
            assert composer._resolve_owner(composer._topology, recovery_token) is domain
            selection = await composer._plan_execution(
                "same.echo", composer.composition.schemas["same.echo"], {}
            )
            assert selection is not None and selection.target == "same"

            await pool._replace_failed_lanes()  # pyright: ignore[reportPrivateUsage]
            new_token = pool.instance_token
            assert new_token is not None and new_token not in (old_token, recovery_token)
            assert composer._resolve_owner(composer._topology, new_token) is domain
            assert all(arm.domain.worker.alive for arm in arms)

            pool._invoke_workgroup = pool._invoke_ranks  # type: ignore[method-assign]
            registry = composer.composition._registry
            result = await composer._routing._routes["same.echo"].invoke(
                Invocation(
                    invocation_id="replacement",
                    node_id="replacement",
                    node_type="same.echo",
                    inputs={"value": registry.wrap("core.string", "ok")},
                    effective_schema=composer.composition.schemas["same.echo"],
                    executor=arms[1].name,
                )
            )
            assert result.error is None and result.outputs is not None
            assert result.outputs["value"].resolve() == "alt:ok:None:False"
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
def test_single_job_provider_arms_share_the_pool_domain(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
            body_arm="alt",
        )
        composer = ServingComposer(
            worker_env=_env(owner_root, provider_root),
            reservations=_Reservations(),  # type: ignore[arg-type]
        )
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(PackSpec(provider_manifest, single_job_cuda_indices=(0, 1)))
            pool = composer._records["provider"].worker
            assert isinstance(pool, _SingleJobWorkerPool)
            arms = composer._topology["owned.echo"]
            assert [arm.name for arm in arms] == [
                "owner",
                "provider",
                "provider@alt:single-job:2:auto",
            ]
            assert arms[1].domain is arms[2].domain
            assert arms[1].domain is composer._records["provider"].domain
            token = pool.instance_token
            assert token is not None
            assert composer._owner_alive(token)
            assert composer._resolve_owner(composer._topology, token) is arms[1].domain
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_worker_host_stamps_nested_residents_for_selected_body(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "resident"
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(_write_resident_arm_pack(root))
            worker = composer._records["resident"].worker
            schema = composer.composition.schemas["resident.make"]

            async def invoke(arm: str | None) -> tuple[Value, Value]:
                result = await worker.invoke(
                    Invocation(
                        invocation_id=f"resident-{arm}",
                        node_id="resident",
                        node_type="resident.make",
                        inputs={},
                        effective_schema=schema,
                        arm=arm,
                    )
                )
                assert result.outputs is not None
                children = list_children(result.outputs["out"])
                assert children is not None and len(children) == 2
                return children[0], children[1]

            default_local, default_foreign = await invoke(None)
            alt_local, alt_foreign = await invoke("alt")
            token = worker.instance_token
            assert default_local.meta.get(RESOURCE_OWNER_META_KEY) == token
            assert default_local.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "resident"
            assert alt_local.meta.get(RESOURCE_OWNER_META_KEY) == token
            assert alt_local.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "resident@alt"
            for foreign in (default_foreign, alt_foreign):
                assert foreign.meta.get(RESOURCE_OWNER_META_KEY) == "foreign-session"
                assert foreign.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) is None
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_native_flux_patch_graph_follows_producer_through_real_dispatch(tmp_path: Path) -> None:
    """The real native patch executes over the existing weight-free worker fixture."""
    from dinkster.native_policy import NativeDispatchPolicy

    patch_type = "dinkster.model_sampling_flux"
    read_type = "test.flux_shift"
    common = (
        "from dataclasses import replace\n"
        "from flux_sampling_pack_nodes import "
        "GenerationModelSamplingFlux, LoadModel, ReadShift, register_types\n"
        "\nclass Load(LoadModel):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return replace(super().define_schema(), node_type='producer.load')\n"
    )
    roots = [tmp_path / name for name in ("owner", "producer")]
    for root in roots:
        root.mkdir()
        nodes = "[GenerationModelSamplingFlux, ReadShift]"
        if root.name == "producer":
            nodes = "[Load, GenerationModelSamplingFlux, ReadShift]"
        (root / f"{root.name}_nodes.py").write_text(common + f"\nNODES = {nodes}\n")
        claims = ["dinkster", "test"] if root.name == "owner" else ["producer"]
        executes = [] if root.name == "owner" else [patch_type, read_type]
        executes_line = f"executes = {json.dumps(executes)}\n" if executes else ""
        (root / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{root.name}"\nnamespaces = {json.dumps(claims)}\n'
            f"{executes_line}[pack.entry]\n"
            f'nodes = "{root.name}_nodes:NODES"\n'
            f'types = "{root.name}_nodes:register_types"\n'
        )

    async def scenario() -> None:
        policy = NativeDispatchPolicy(lambda _digest: None, lambda _diagnostic: None)
        composer = ServingComposer(
            worker_env=_env(*roots, Path(__file__).parent), native_policy=policy
        )
        try:
            for root in roots:
                await composer.add_pack(PackSpec(root / "dinkster-pack.toml", trust_reserved=True))
            assert composer._topology[patch_type][0].name == "owner"
            assert composer._topology[read_type][0].name == "owner"
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode("producer.load"),
                    "patch": GraphNode(
                        patch_type,
                        {
                            "model": Link("load", "model"),
                            "max_shift": 1.15,
                            "base_shift": 0.5,
                            "width": 768,
                            "height": 1024,
                        },
                    ),
                    "read": GraphNode(read_type, {"model": Link("patch", "model")}),
                }
            )
            result = await engine.run(graph, ["load", "patch", "read"])
            assert result.outputs["read"]["shift"].resolve() == 0.9766666666666666
            for node_id in ("load", "patch"):
                model = result.outputs[node_id]["model"]
                assert model.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "producer"
                assert model.meta.get(RESOURCE_OWNER_META_KEY) == (
                    composer._records["producer"].worker.instance_token
                )
            again = await engine.run(graph, ["load", "patch", "read"])
            assert again.executed == ()
            assert again.outputs["read"]["shift"].resolve() == 0.9766666666666666
        finally:
            await composer.close()

    asyncio.run(scenario())
