from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    ExportSnapshot,
    Invocation,
    derive_attention_route_token,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_event,
)
from dinkster_values import CORE_COMBO, CORE_STRING, TypeRegistry, register_core_types
from dinkster_workers import (
    DeviceMap,
    InProcessWorker,
    IsolatedWorker,
    LaunchSpec,
    current_execution_context,
)
from dinkster_workers.in_process import attention_route_token_matches_capabilities
from dinkster_workers.launch import Launcher


class SlowSyncNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="review.slow",
            display_name="Slow",
            category="test",
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        time.sleep(0.1)
        report_event("review.thread", {"ok": True})
        return cls.outputs(value="done")


class ComboAdmissionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        combo = TypeExpr.concrete(CORE_COMBO)
        return NodeSchema(
            node_type="review.combo_admission",
            inputs=(InputSpec("value", combo),),
            outputs=(OutputSpec("value", combo),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class StringAdmissionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="review.string_admission",
            inputs=(InputSpec("value", string),),
            outputs=(OutputSpec("value", string),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ComboListAdmissionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        combo_list = TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
        return NodeSchema(
            node_type="review.combo_list_admission",
            inputs=(InputSpec("values", combo_list),),
            outputs=(OutputSpec("values", combo_list),),
        )

    @classmethod
    def execute(cls, *, values: list[str]) -> Mapping[str, object]:
        return cls.outputs(values=values)


class ExecutionContextProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="review.execution_context",
            outputs=(OutputSpec("value", string),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        context = current_execution_context()
        assert context is not None
        return cls.outputs(
            value="|".join(
                (
                    context.node_id or "",
                    context.arm or "",
                    context.expected_execution_identity or "",
                    context.extension_snapshot_digest or "",
                    (
                        "none"
                        if context.export_snapshot is None
                        else str(context.export_snapshot.prompt)
                    ),
                    str(context.fp8_matmul),
                    str(context.cancelled()),
                )
            )
        )


def test_sync_node_does_not_block_loop_and_reports_from_thread() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas([SlowSyncNode]),
            registry=registry,
            worker=InProcessWorker(build_node_types([SlowSyncNode]), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        run = asyncio.create_task(
            engine.run(Graph(nodes={"slow": GraphNode("review.slow", {})}), ["slow"])
        )
        await asyncio.sleep(0.02)
        assert not run.done()
        await run
        assert any(
            event.kind == "node_event" and event.detail["name"] == "review.thread"
            for event in events
        )

    asyncio.run(scenario())


def test_in_process_worker_supplies_invocation_execution_context() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        node_types = build_node_types([ExecutionContextProbe])
        worker = InProcessWorker(node_types, registry)
        schema = ExecutionContextProbe.schema()
        digest = "sha256:" + "a" * 64
        snapshot = ExportSnapshot(
            prompt={"save": {"class_type": "Save", "inputs": {}}},
            extra_pnginfo={"workflow": {"nodes": [1]}},
        )
        result = await worker.invoke(
            Invocation(
                invocation_id="invocation",
                node_id="submitted-node-17",
                node_type=schema.node_type,
                inputs={},
                effective_schema=schema,
                arm="native",
                expected_execution_identity="identity",
                extension_snapshot_digest=digest,
                export_snapshot=snapshot,
                fp8_matmul=True,
            )
        )
        assert result.error is None
        assert result.outputs is not None
        assert result.outputs["value"].resolve() == (
            f"submitted-node-17|native|identity|{digest}|{snapshot.prompt}|True|False"
        )

    asyncio.run(scenario())


def test_worker_requires_exact_optional_startup_attention_evidence() -> None:
    capabilities = AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )
    token = derive_attention_route_token(capabilities, AttentionPolicyConfig())
    selectable_capabilities = AttentionCapabilityEvidence(
        version=1,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime=token.sdpa_torch_runtime,
        adapter_contract_revision=token.adapter_contract_revision,
        available_policies=("sdpa", "flash"),
        provider_versions=token.provider_versions,
    )
    selectable_startup_token = derive_attention_route_token(
        selectable_capabilities, AttentionPolicyConfig()
    )
    selected_flash_token = derive_attention_route_token(
        selectable_capabilities, AttentionPolicyConfig("flash")
    )
    fallback_flash_token = derive_attention_route_token(
        capabilities, AttentionPolicyConfig("flash")
    )
    inconsistent_capabilities = AttentionCapabilityEvidence(
        version=1,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime="9.9.9",
        adapter_contract_revision=token.adapter_contract_revision,
        available_policies=("sdpa",),
        provider_versions=(("torch", "9.9.9"),),
    )
    registry = TypeRegistry()
    register_core_types(registry)
    with pytest.raises(ValueError, match="capabilities do not match route token"):
        InProcessWorker(
            {},
            registry,
            attention_capabilities=inconsistent_capabilities,
            attention_route_token=token,
        )

    async def scenario() -> None:
        schema = ExecutionContextProbe.schema()
        forged = replace(token, provider_versions=(("torch", "forged"),))
        for worker_capabilities, worker_token, invocation_token, accepted in (
            (None, None, None, True),
            (capabilities, token, token, True),
            (capabilities, token, None, False),
            (None, None, token, False),
            (capabilities, token, forged, False),
            (selectable_capabilities, selectable_startup_token, selected_flash_token, True),
            (capabilities, token, fallback_flash_token, True),
            (
                capabilities,
                token,
                replace(fallback_flash_token, device_kind="forged"),
                False,
            ),
            (
                capabilities,
                token,
                replace(fallback_flash_token, adapter_contract_revision="forged"),
                False,
            ),
        ):
            worker = InProcessWorker(
                build_node_types([ExecutionContextProbe]),
                registry,
                attention_capabilities=worker_capabilities,
                attention_route_token=worker_token,
            )
            result = await worker.invoke(
                Invocation(
                    invocation_id="invocation",
                    node_id="authenticated",
                    node_type=schema.node_type,
                    inputs={},
                    effective_schema=schema,
                    attention_policy=(
                        "auto" if invocation_token is None else invocation_token.requested_policy
                    ),
                    attention_route_token=invocation_token,
                )
            )
            assert (result.error is None) is accepted
            if not accepted:
                assert result.error is not None
                assert result.error.message == (
                    "attention route token does not match worker startup evidence"
                )

    asyncio.run(scenario())


def test_attention_tokens_stay_bound_to_restart_and_concurrent_worker_evidence() -> None:
    first = AttentionCapabilityEvidence(
        version=1,
        device_kind="cuda",
        device_sm=120,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )
    restarted_same = replace(first)
    concurrent_other = replace(
        first,
        device_sm=121,
        sdpa_torch_runtime="2.14.0",
        provider_versions=(("torch", "2.14.0"),),
    )
    first_token = derive_attention_route_token(first, AttentionPolicyConfig())
    restarted_token = derive_attention_route_token(restarted_same, AttentionPolicyConfig())
    other_token = derive_attention_route_token(concurrent_other, AttentionPolicyConfig())

    assert attention_route_token_matches_capabilities(first, first_token)
    assert attention_route_token_matches_capabilities(restarted_same, restarted_token)
    assert attention_route_token_matches_capabilities(restarted_same, first_token)
    assert attention_route_token_matches_capabilities(concurrent_other, other_token)
    assert not attention_route_token_matches_capabilities(concurrent_other, first_token)
    assert not attention_route_token_matches_capabilities(first, other_token)
    assert attention_route_token_matches_capabilities(None, None)
    assert not attention_route_token_matches_capabilities(first, None)
    assert not attention_route_token_matches_capabilities(None, first_token)


def test_engine_keeps_export_snapshot_off_non_output_invocations() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        node_types = build_node_types([ExecutionContextProbe])
        engine = Engine(
            schemas=build_schemas([ExecutionContextProbe]),
            registry=registry,
            worker=InProcessWorker(node_types, registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(nodes={"probe": GraphNode("review.execution_context", {})}),
            ["probe"],
            export_snapshot=ExportSnapshot(prompt={"save": {"inputs": {}}}),
        )
        value = result.outputs["probe"]["value"].resolve()
        assert isinstance(value, str)
        assert value.split("|")[4] == "none"

    asyncio.run(scenario())


def test_worker_rejects_combo_boundary_mismatches_both_directions() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        nodes = (ComboAdmissionNode, StringAdmissionNode)
        worker = InProcessWorker(build_node_types(nodes), registry)
        schemas = build_schemas(nodes)
        for node_type, value in (
            ("review.combo_admission", registry.wrap(CORE_STRING, "x")),
            ("review.string_admission", registry.wrap(CORE_COMBO, "x")),
        ):
            result = await worker.invoke(
                Invocation(
                    invocation_id=node_type,
                    node_id="n",
                    node_type=node_type,
                    inputs={"value": value},
                    effective_schema=schemas[node_type],
                )
            )
            assert result.error is not None
            assert "explicit converter" in result.error.message

    asyncio.run(scenario())


def test_worker_preserves_combo_list_order_duplicates_and_fingerprint() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = ComboListAdmissionNode.schema()
        worker = InProcessWorker(build_node_types((ComboListAdmissionNode,)), registry)
        value = registry.wrap("list<core.combo>", ["b", "a", "b"])
        result = await worker.invoke(
            Invocation(
                invocation_id="combo-list",
                node_id="n",
                node_type=schema.node_type,
                inputs={"values": value},
                effective_schema=schema,
            )
        )
        assert result.error is None
        assert result.outputs is not None
        output = result.outputs["values"]
        assert output.resolve() == ["b", "a", "b"]
        assert output.fingerprint == value.fingerprint

    asyncio.run(scenario())


class RaisingLauncher(Launcher):
    endpoint: Path | None = None

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        self.endpoint = spec.endpoint_dir
        raise RuntimeError("launch failed")


def test_launch_failure_cleans_listener_and_temp_directory(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text('[pack]\nname = "review"\n\n[pack.entry]\nnodes = "missing:NODES"\n')
        registry = TypeRegistry()
        register_core_types(registry)
        launcher = RaisingLauncher()
        worker = IsolatedWorker(manifest, registry, launcher=launcher)
        with pytest.raises(RuntimeError, match="launch failed"):
            await worker.start()
        assert launcher.endpoint is not None
        assert not launcher.endpoint.exists()
        assert worker._listener is None

    asyncio.run(scenario())


def test_worker_mapping_dataclasses_snapshot_input(tmp_path: Path) -> None:
    mapping = {"cuda:0": "cuda:1"}
    device_map = DeviceMap(mapping)
    mapping["cuda:0"] = "cuda:2"
    assert device_map.device("cuda:0") == "cuda:1"

    env = {"TOKEN": "original"}
    spec = LaunchSpec(("python",), env, tmp_path, tmp_path, "python", False)
    env["TOKEN"] = "changed"
    assert spec.env["TOKEN"] == "original"
