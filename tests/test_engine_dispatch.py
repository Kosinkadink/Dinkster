"""Stage-6 execution dispatch, engine side: the plan_execution hook runs
BEFORE cache lookup, its cache_tag partitions cache identity per executing
implementation, and owner-liveness admission rejects results that reference
dead worker lifetimes (cached entries recompute; fresh results fail loudly)."""

from __future__ import annotations

import asyncio

from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    Engine,
    EngineEvent,
    ExecutionSelection,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
    PlanExecution,
)
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    derive_attention_route_token,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_schemas,
)
from dinkster_values import (
    RESOURCE_HANDLE_TYPE,
    ResourceHandle,
    TypeRegistry,
    Value,
    register_core_types,
    register_resource_handle_type,
)

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")
HANDLE_LIST = TypeExpr.list_of(TypeExpr.concrete(RESOURCE_HANDLE_TYPE))

_WRAP_REGISTRY = TypeRegistry()
register_core_types(_WRAP_REGISTRY)
register_resource_handle_type(_WRAP_REGISTRY)


class Produce(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.produce",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )


class Load(Node):
    """Resident-producing shape: outputs a LIST of resource handles, so the
    owner sits inside a value tree (traversal, not top-envelope reads)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.load",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", HANDLE_LIST),),
        )


def _string(text: str) -> Value:
    return _WRAP_REGISTRY.wrap("STRING", text)


def _stamped(rid: str, owner: str) -> Value:
    """A list-of-handles Value whose single element references ``rid`` with
    producer-stamped owner provenance ``owner`` (a non-local handle relays
    the token its producer stamped)."""
    handle = ResourceHandle(resource_id=rid, kind="model", owner=owner)
    return _WRAP_REGISTRY.wrap(f"list<{RESOURCE_HANDLE_TYPE}>", [handle])


class RecordingWorker:
    """Fake worker: records every invocation and returns outputs built by
    a per-call factory, so tests control resource/owner stamping exactly."""

    def __init__(self, make_out) -> None:
        self.invocations: list[Invocation] = []
        self._make_out = make_out

    async def prepare(self, node_types) -> None:
        pass

    async def invoke(
        self, invocation: Invocation, on_event: OnInvocationEvent | None = None
    ) -> InvocationResult:
        self.invocations.append(invocation)
        return InvocationResult(outputs={"out": self._make_out(len(self.invocations))})


class TrackingWorker(RecordingWorker):
    """Records the peak number of concurrent worker invocations."""

    def __init__(self) -> None:
        super().__init__(lambda n: _string(f"v{n}"))
        self.active = 0
        self.peak = 0

    async def invoke(
        self, invocation: Invocation, on_event: OnInvocationEvent | None = None
    ) -> InvocationResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return await super().invoke(invocation, on_event)
        finally:
            self.active -= 1


def _engine(
    worker: RecordingWorker,
    *,
    plan_execution: PlanExecution | None = None,
    run_finished=None,
    owner_alive=None,
    cache: MemoryLRUCache | None = None,
    events: list[EngineEvent] | None = None,
    explain_misses: bool = False,
) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_resource_handle_type(registry)
    return Engine(
        schemas=build_schemas([Produce, Load]),
        registry=registry,
        worker=worker,
        cache=cache if cache is not None else MemoryLRUCache(),
        on_event=events.append if events is not None else None,
        plan_execution=plan_execution,
        run_finished=run_finished,
        owner_alive=owner_alive,
        explain_misses=explain_misses,
    )


def _graph(tag: str = "x") -> Graph:
    return Graph(nodes={"p": GraphNode("test.produce", {"tag": tag})})


def _load_graph(tag: str = "x") -> Graph:
    return Graph(nodes={"p": GraphNode("test.load", {"tag": tag})})


def test_attention_route_partitions_cache_entries_on_the_same_arm() -> None:
    async def scenario() -> None:
        evidence = AttentionCapabilityEvidence(
            version=1,
            device_kind="cpu",
            device_sm=None,
            sdpa_torch_runtime="2.13.0",
            adapter_contract_revision="dinkster.attention-kernel.v1",
            available_policies=("sdpa", "flash"),
            provider_versions=(("torch", "2.13.0"),),
        )

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config, single_job_multi_gpu
        ) -> ExecutionSelection:
            config = attention_config or AttentionPolicyConfig()
            token = derive_attention_route_token(evidence, config)
            return ExecutionSelection(
                target="native",
                cache_tag="native@1",
                attention_policy=token.requested_policy,
                attention_route_token=token,
            )

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        engine = _engine(worker, plan_execution=plan)
        scalar = AttentionPolicyConfig("flash")
        role = AttentionPolicyConfig(requested_role_policies=(("flux", "flash"),))

        results = []
        for config in (scalar, scalar, role, role):
            result = await engine.run(_graph(), ["p"], attention_config=config)
            results.append(result.outputs["p"]["out"].resolve())

        assert results == ["v1", "v1", "v2", "v2"]
        assert len(worker.invocations) == 2
        assert [inv.attention_route_token for inv in worker.invocations] == [
            derive_attention_route_token(evidence, scalar),
            derive_attention_route_token(evidence, role),
        ]

    run_scenario(scenario())
