"""Stage-6 execution dispatch, engine side: the plan_execution hook runs
BEFORE cache lookup, its cache_tag partitions cache identity per executing
implementation, and owner-liveness admission rejects results that reference
dead worker lifetimes (cached entries recompute; fresh results fail loudly)."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    Engine,
    EngineEvent,
    ExecutionError,
    ExecutionSelection,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
    PlanExecution,
)
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import (
    ActiveExtension,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    ExtensionSnapshot,
    MediaSourceAuthority,
    derive_attention_route_token,
    extension_behavior_hash,
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

STRING = TypeExpr.concrete("core.string")
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
    return _WRAP_REGISTRY.wrap("core.string", text)


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


def test_execution_selection_rejects_blank_fields() -> None:
    with pytest.raises(ValueError):
        ExecutionSelection(target="", cache_tag="native@1")
    with pytest.raises(ValueError):
        ExecutionSelection(target="native", cache_tag="")
    with pytest.raises(ValueError):
        ExecutionSelection(target="native", cache_tag="native@1", execution_arm="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attention_diagnostic"):
        ExecutionSelection(target="native", cache_tag="native@1", attention_diagnostic="")


@pytest.mark.parametrize("diagnostic", [None, "using default attention"])
def test_selected_arm_is_reported_for_execution_and_cache_reuse(diagnostic: str | None) -> None:
    async def scenario() -> None:
        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            return ExecutionSelection(
                target="compat",
                cache_tag="compat@1",
                execution_arm="comfyui",
                attention_diagnostic=f"{diagnostic}: {run_id}" if diagnostic is not None else None,
            )

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        events: list[EngineEvent] = []
        engine = _engine(worker, plan_execution=plan, events=events)
        await engine.run(_graph(), ["p"], run_id="executed")
        await engine.run(_graph(), ["p"], run_id="cached")

        arms = [
            (event.kind, event.run_id, event.detail.get("executionArm"))
            for event in events
            if event.kind in ("node_started", "node_finished", "node_cached")
        ]
        assert arms == [
            ("node_started", "executed", "comfyui"),
            ("node_finished", "executed", "comfyui"),
            ("node_cached", "cached", "comfyui"),
        ]
        for event in events:
            if event.kind in ("node_started", "node_finished", "node_cached"):
                if diagnostic is None:
                    assert "attentionDiagnostic" not in event.detail
                else:
                    assert event.detail["attentionDiagnostic"] == f"{diagnostic}: {event.run_id}"
        assert len(worker.invocations) == 1

    asyncio.run(scenario())


def test_plan_runs_once_before_lookup_and_target_rides_invocation() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        configs: list[AttentionPolicyConfig | None] = []

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            calls.append(node_type)
            configs.append(attention_config)
            return ExecutionSelection(target="native", cache_tag="native@1", fp8_matmul=True)

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        engine = _engine(worker, plan_execution=plan)
        config = AttentionPolicyConfig("flash")
        await engine.run(_graph(), ["p"], attention_config=config)
        # Miss: planned once, invoked once, selection rode the invocation.
        assert calls == ["test.produce"]
        assert configs == [config]
        assert [inv.executor for inv in worker.invocations] == ["native"]
        assert [inv.expected_execution_identity for inv in worker.invocations] == ["native@1"]
        assert [inv.fp8_matmul for inv in worker.invocations] == [True]
        await engine.run(_graph(), ["p"], attention_config=config)
        # Hit: planned again (exactly once per attempt), nothing invoked.
        assert calls == ["test.produce", "test.produce"]
        assert configs == [config, config]
        assert len(worker.invocations) == 1

    asyncio.run(scenario())


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
            node_id, node_type, schema, inputs, run_id, attention_config
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

    asyncio.run(scenario())


def test_selected_plain_nodes_admit_per_execution_arm() -> None:
    async def scenario() -> None:
        targets = {"a": "cuda:0", "b": "cuda:0"}

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            target = targets[run_id]
            return ExecutionSelection(target=target, cache_tag=f"native:{target}")

        worker = TrackingWorker()
        engine = _engine(worker, plan_execution=plan)
        await asyncio.gather(
            engine.run(_graph("first"), ["p"], run_id="a"),
            engine.run(_graph("second"), ["p"], run_id="b"),
        )
        assert worker.peak == 1
        assert "compute:cuda:0" in engine._resource_slots

        worker.peak = 0
        targets["b"] = "cuda:1"
        await asyncio.gather(
            engine.run(_graph("third"), ["p"], run_id="a"),
            engine.run(_graph("fourth"), ["p"], run_id="b"),
        )
        assert worker.peak == 2
        assert "compute:cuda:1" in engine._resource_slots

    asyncio.run(scenario())


def test_media_sources_propagate_per_run_without_changing_cache_identity() -> None:
    async def scenario() -> None:
        authority = MediaSourceAuthority(
            "blake3:" + "a" * 64,
            "media/image",
            "image/png",
            "png",
            17,
        )
        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        engine = _engine(worker)
        await engine.run(_graph(), ["p"], media_sources=(authority,))
        await engine.run(_graph(), ["p"])
        assert len(worker.invocations) == 1
        assert worker.invocations[0].media_sources == (authority,)

    asyncio.run(scenario())


def test_malformed_media_sources_do_not_poison_run_state() -> None:
    async def scenario() -> None:
        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        engine = _engine(worker)
        with pytest.raises(ValueError, match="MediaSourceAuthority"):
            await engine.run(
                _graph(),
                ["p"],
                run_id="reusable",
                media_sources=(object(),),  # type: ignore[arg-type]
            )
        result = await engine.run(_graph(), ["p"], run_id="reusable")
        assert result.run_id == "reusable"

    asyncio.run(scenario())


def test_plan_failure_fails_the_node_loudly() -> None:
    async def scenario() -> None:
        finished: list[str] = []

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            raise RuntimeError("no owner agreement")

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        engine = _engine(worker, plan_execution=plan, run_finished=finished.append)
        with pytest.raises(ExecutionError):
            await engine.run(_graph(), ["p"], run_id="failed-run")
        assert worker.invocations == []  # never fell through to a guess
        assert finished == ["failed-run"]

    asyncio.run(scenario())


def test_cache_tag_partitions_entries_and_rotation_is_named() -> None:
    async def scenario() -> None:
        tag = "compat@1"

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            return ExecutionSelection(target="arm", cache_tag=tag)

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        cache = MemoryLRUCache()
        events: list[EngineEvent] = []
        engine = _engine(
            worker,
            plan_execution=plan,
            cache=cache,
            events=events,
            explain_misses=True,
        )
        await engine.run(_graph(), ["p"])
        await engine.run(_graph(), ["p"])
        assert len(worker.invocations) == 1  # same tag: ordinary hit

        tag = "native@1"  # different implementation: different computation
        await engine.run(_graph(), ["p"])
        assert len(worker.invocations) == 2

        tag = "native@2"  # identity rotation: keys rotate, named as such
        await engine.run(_graph(), ["p"])
        assert len(worker.invocations) == 3
        rotation = [
            e
            for e in events
            if e.kind == "cache_miss" and e.detail.get("reason") == "executor-changed"
        ]
        assert rotation  # the rotation was explained, not misreported

    asyncio.run(scenario())


def test_target_alone_partitions_entries_even_with_a_shared_tag() -> None:
    """The key-to-arm binding is structural: two arms announcing the SAME
    cache_tag still never share an entry, so a compat result can never be
    served for a native selection (cache poisoning across arms)."""

    async def scenario() -> None:
        target = "compat"

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            return ExecutionSelection(target=target, cache_tag="impl@1")

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        events: list[EngineEvent] = []
        engine = _engine(worker, plan_execution=plan, events=events, explain_misses=True)
        await engine.run(_graph(), ["p"])
        assert [inv.executor for inv in worker.invocations] == ["compat"]

        target = "native"  # same tag, different arm: different computation
        await engine.run(_graph(), ["p"])
        assert [inv.executor for inv in worker.invocations] == ["compat", "native"]
        rotation = [
            e
            for e in events
            if e.kind == "cache_miss" and e.detail.get("reason") == "executor-changed"
        ]
        assert rotation  # named as a selection change, not misreported

    asyncio.run(scenario())


def test_unenrolled_none_preserves_ordinary_cache_identity() -> None:
    async def scenario() -> None:
        async def plan(node_id, node_type, schema, inputs, run_id, attention_config) -> None:
            return None

        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        cache = MemoryLRUCache()
        engine_hooked = _engine(worker, plan_execution=plan, cache=cache)
        await engine_hooked.run(_graph(), ["p"])
        assert [inv.executor for inv in worker.invocations] == [None]
        # A hook-free engine sharing the cache hits the same entry: None
        # keyed nothing, so unenrolled identity is unchanged by the hook.
        engine_plain = _engine(worker, cache=cache)
        await engine_plain.run(_graph(), ["p"])
        assert len(worker.invocations) == 1

    asyncio.run(scenario())


def test_empty_extension_snapshot_has_stable_cache_key() -> None:
    async def scenario() -> None:
        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        events: list[EngineEvent] = []
        engine = _engine(worker, events=events)
        await engine.run(_graph(), ["p"])
        finished = next(event for event in events if event.kind == "node_finished")
        assert finished.detail["cache_key"] == "250e1d46c0d7a2c566da1e863ec20a18aa4f45da"
        assert worker.invocations[0].extension_snapshot_digest is None

    asyncio.run(scenario())


def test_extension_snapshot_pins_concurrent_runs_and_rotates_cache_identity() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedWorker(RecordingWorker):
            async def invoke(self, invocation, on_event=None) -> InvocationResult:
                self.invocations.append(invocation)
                if len(self.invocations) == 1:
                    entered.set()
                    await release.wait()
                return InvocationResult(outputs={"out": _string("same")})

        worker = GatedWorker(lambda _n: _string("same"))
        cache = MemoryLRUCache()
        engine = _engine(worker, cache=cache)
        base = engine.pin_execution()
        old_snapshot = ExtensionSnapshot(
            (
                ActiveExtension(
                    id="example.old",
                    version="1.0.0",
                    package_digest="sha256:" + "1" * 64,
                ),
            )
        )
        new_snapshot = ExtensionSnapshot(
            (
                ActiveExtension(
                    id="example.new",
                    version="1.0.0",
                    package_digest="sha256:" + "2" * 64,
                ),
            )
        )
        old_runtime = replace(base, extension_snapshot=old_snapshot)
        new_runtime = replace(base, extension_snapshot=new_snapshot)

        first = asyncio.create_task(engine.run(_graph(), ["p"], execution=old_runtime))
        await entered.wait()
        second_task = asyncio.create_task(engine.run(_graph(), ["p"], execution=new_runtime))
        await asyncio.sleep(0)
        release.set()
        _, second = await asyncio.gather(first, second_task)
        assert second.executed == ("p",)
        assert [inv.extension_snapshot_digest for inv in worker.invocations] == [
            "sha256:" + extension_behavior_hash(old_snapshot),
            "sha256:" + extension_behavior_hash(new_snapshot),
        ]

        # Same generation now hits, while the exact empty generation shares
        # the legacy key path and omits the new invocation boundary field.
        await engine.run(_graph(), ["p"], execution=new_runtime)
        assert len(worker.invocations) == 2
        empty_runtime = replace(base, extension_snapshot=ExtensionSnapshot())
        await engine.run(_graph(), ["p"], execution=empty_runtime)
        assert len(worker.invocations) == 3
        assert worker.invocations[-1].extension_snapshot_digest is None

    asyncio.run(scenario())


def test_enrollment_marker_does_not_collide_with_pre_enrollment_cache() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache()
        worker = RecordingWorker(lambda n: _string(f"v{n}"))
        plain = _engine(worker, cache=cache)
        await plain.run(_graph(), ["p"])

        async def plan(
            node_id, node_type, schema, inputs, run_id, attention_config
        ) -> ExecutionSelection:
            return ExecutionSelection(target="owner", cache_tag="unversioned")

        enrolled = _engine(worker, plan_execution=plan, cache=cache)
        await enrolled.run(_graph(), ["p"])
        assert len(worker.invocations) == 2

    asyncio.run(scenario())


def test_dead_owner_hit_recomputes_and_entry_is_overwritten() -> None:
    async def scenario() -> None:
        live = {"life-1"}
        owners = iter(["life-1", "life-2"])

        # Nested-list stamping: traversal must find the owner inside the tree.
        worker = RecordingWorker(lambda n: _stamped("res:model", next(owners)))
        cache = MemoryLRUCache()
        events: list[EngineEvent] = []
        engine = _engine(
            worker,
            owner_alive=lambda token: token in live,
            cache=cache,
            events=events,
            explain_misses=True,
        )
        await engine.run(_load_graph(), ["p"])
        assert len(worker.invocations) == 1

        live.clear()
        live.add("life-2")  # the first lifetime died; a new one is live
        await engine.run(_load_graph(), ["p"])
        assert len(worker.invocations) == 2  # dead-owner hit became a miss
        assert any(
            e.kind == "cache_miss" and e.detail.get("reason") == "dead-owner" for e in events
        )
        await engine.run(_load_graph(), ["p"])
        assert len(worker.invocations) == 2  # fresh result overwrote the entry

    asyncio.run(scenario())


def test_fresh_dead_owner_fails_loudly_and_is_not_cached() -> None:
    async def scenario() -> None:
        worker = RecordingWorker(lambda n: _stamped("res:model", "dead-life"))
        engine = _engine(worker, owner_alive=lambda token: False)
        with pytest.raises(ExecutionError) as excinfo:
            await engine.run(_load_graph(), ["p"])
        assert "ownership contract violation" in str(excinfo.value)

        # The doomed result never entered the cache: a live worker sharing
        # the cache computes fresh instead of hitting the poisoned entry.
        cache = MemoryLRUCache()
        dead_worker = RecordingWorker(lambda n: _stamped("res:model", "dead-life"))
        engine_dead = _engine(dead_worker, owner_alive=lambda token: False, cache=cache)
        with pytest.raises(ExecutionError):
            await engine_dead.run(_load_graph(), ["p"])
        live_worker = RecordingWorker(lambda n: _stamped("res:model", "live-life"))
        engine_live = _engine(
            live_worker, owner_alive=lambda token: token == "live-life", cache=cache
        )
        result = await engine_live.run(_load_graph(), ["p"])
        assert tuple(result.executed) == ("p",)

    asyncio.run(scenario())
