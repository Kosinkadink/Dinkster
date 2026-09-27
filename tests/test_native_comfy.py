"""Standalone native registration and import metadata require no ComfyUI runtime."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy.schema_snapshot import core_schema_snapshot
from dinkster_compat_comfy.source_staging import ComfySourceStagingProvider, SourceStagingError
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_generation import GENERATION_NODES, generation_choices
from dinkster_schema import SCHEMA_WIRE_VERSION, build_schemas, schema_from_wire, schema_to_wire
from dinkster_server import STATE_KEY, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, SingleJobMultiGpuConfig
from dinkster_workers.manifest import load_manifest

from dinkster.comfy_compose import comfy_compat_specs
from dinkster.compat_api import add_comfy_compat_routes
from dinkster.compose import CompositionError
from tests.test_compat_prompt import LatentSink
from tools.gen_native_comfy_manifests import manifest_with_declared_arms, native_manifest


def test_snapshot_round_trips_and_records_provenance() -> None:
    snapshot = core_schema_snapshot()
    assert snapshot["sourceRepository"] == "https://github.com/Comfy-Org/ComfyUI"
    assert snapshot["sourceCommit"] == "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
    assert snapshot["schemaWireVersion"] == SCHEMA_WIRE_VERSION
    assert len(snapshot["schemas"]) == 642
    for name, wire in snapshot["schemas"].items():
        assert name == wire["nodeType"]
        assert schema_to_wire(schema_from_wire(wire)) == wire
    assert "comfy.KSampler" in snapshot["schemas"]
    assert "comfy.CLIPTextEncode" in snapshot["schemas"]
    assert snapshot["schemas"]["comfy.ComfySwitchNode"]["selector"] == {
        "input": "switch",
        "branches": {"false": "on_false", "true": "on_true"},
    }
    assert "ComfySwitchNode" not in snapshot["skipped"]


def test_native_entry_does_not_import_comfyui(tmp_path: Path) -> None:
    script = """
import importlib.abc
import json
import sys
class NoComfy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('comfy', 'comfy_extras', 'folder_paths', 'nodes', 'torch'):
            raise AssertionError('Runtime import attempted: ' + fullname)
        if fullname.split('.')[0] == 'dinkster_compat_comfy':
            raise AssertionError('Compatibility import attempted: ' + fullname)
sys.meta_path.insert(0, NoComfy())
from dinkster_native.entry import ARM_NODES, NATIVE_NODES, combo_choices, register_types
from dinkster_native.native_arm import (
    GenerationApplyTextureToMesh,
    GenerationBakeAmbientOcclusion,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeTextureFromVoxel,
    GenerationDecimateMesh,
    GenerationEstimateGeometry,
    GenerationKSampler,
    GenerationLoadBackgroundRemoval,
    GenerationLoadGeometryModel,
    GenerationPaintMesh,
    GenerationRemeshMesh,
    GenerationRemoveBackground,
    GenerationRenderUVAtlas,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationVoxelToMesh,
)
from dinkster_native.native_catalog import (
    COMFY_RUNTIME_NODE_IDS, NATIVE_SCHEDULING_NODE_TYPES,
    NATIVE_SCHEDULING_SOURCE_NODE_NAMES,
)
from dinkster_native.native_residency import NativeComponentHandle, NativeRuntimeHandle
from dinkster_native.pool import resident_advisory_unload
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_registry import execution_symbol
from dinkster_values import TypeRegistry, register_core_types
registry = TypeRegistry()
register_core_types(registry)
register_types(registry)
nodes = {node.schema().node_type: node for node in NATIVE_NODES}
assert nodes['dinkster.ksampler'] is GenerationKSampler
assert 'comfy.KSampler' not in nodes
assert 'comfy.CLIPTextEncode' not in nodes
assert not COMFY_RUNTIME_NODE_IDS.intersection(nodes)
assert nodes['dinkster.load_geometry_model'] is GenerationLoadGeometryModel
assert nodes['dinkster.estimate_geometry'] is GenerationEstimateGeometry
assert nodes['dinkster.load_background_removal'] is GenerationLoadBackgroundRemoval
assert nodes['dinkster.remove_background'] is GenerationRemoveBackground
assert nodes['dinkster.voxel_to_mesh'] is GenerationVoxelToMesh
assert nodes['dinkster.remesh_mesh'] is GenerationRemeshMesh
assert nodes['dinkster.decimate_mesh'] is GenerationDecimateMesh
assert nodes['dinkster.smooth_mesh_normals'] is GenerationSmoothMeshNormals
assert nodes['dinkster.unwrap_mesh'] is GenerationUnwrapMesh
assert nodes['dinkster.paint_mesh'] is GenerationPaintMesh
assert nodes['dinkster.bake_texture_from_voxel'] is GenerationBakeTextureFromVoxel
assert nodes['dinkster.bake_normal_map_from_mesh'] is GenerationBakeNormalMapFromMesh
assert nodes['dinkster.bake_ambient_occlusion'] is GenerationBakeAmbientOcclusion
assert nodes['dinkster.render_uv_atlas'] is GenerationRenderUVAtlas
assert nodes['dinkster.apply_texture_to_mesh'] is GenerationApplyTextureToMesh
assert 'dinkster.geometry_to_fov' in nodes
assert 'dinkster.preview_mask' in nodes
assert NATIVE_SCHEDULING_NODE_TYPES.keys() == NATIVE_SCHEDULING_SOURCE_NODE_NAMES.keys()
for symbol, node_type in NATIVE_SCHEDULING_NODE_TYPES.items():
    source_name = NATIVE_SCHEDULING_SOURCE_NODE_NAMES[symbol]
    assert node_type in nodes
    assert nodes[node_type].schema().aliases == (source_name,)
    assert f'comfy.{source_name}' not in nodes
assert all(nodes[node.schema().node_type] is node for node in ARM_NODES['native'])
assert 'euler' in combo_choices()['comfy.samplers']
for handle_type in (NativeComponentHandle, NativeRuntimeHandle):
    class Handle(handle_type):
        def __init__(self):
            self.unloaded = False
        def advisory_unload(self):
            self.unloaded = True
    handle = Handle()
    resident_advisory_unload(handle)
    assert handle.unloaded
for descriptor in default_component_registry():
    for reference in (descriptor.execution_resolver, descriptor.codec_adapter):
        if reference is not None:
            assert reference.startswith('dinkster_native.')
            assert execution_symbol(reference).__module__.startswith('dinkster_native.')
print(json.dumps(sorted(nodes)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "DINKSTER_COMFY_NATIVE_ONLY": "1", "DINKSTER_COMFYUI_ROOT": ""},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "dinkster.clip_text_encode" in json.loads(result.stdout)


def test_compat_entry_replaces_translated_generation_providers(tmp_path: Path) -> None:
    script = """
import importlib
import os
import sys
from dinkster_compat_comfy import bootstrap
from dinkster_compat_comfy.translate import CompatTranslation
from dinkster_compat_comfy.prompt import build_alias_index
from dinkster_native.nodes_conditioning import GenerationBlockSparseAttention
from dinkster_nodes_generation.nodes import BlockSparseAttention
from dinkster_schema import build_node_types, build_schemas

class TranslatedBlockSparseAttention(GenerationBlockSparseAttention):
    pass

translation = CompatTranslation()
translation.node_classes.append(TranslatedBlockSparseAttention)
bootstrap.load_comfyui_nodes = lambda *, required=(): translation
sys.modules.pop('dinkster_compat_comfy.entry', None)
entry = importlib.import_module('dinkster_compat_comfy.entry')
nodes = [
    node
    for node in entry.COMFY_NODES
    if node.schema().node_type == 'comfy.BlockSparseAttention'
]
assert nodes == [GenerationBlockSparseAttention]
assert nodes[0].schema() == BlockSparseAttention.schema()
build_node_types(entry.COMFY_NODES)
assert build_alias_index(build_schemas(entry.COMFY_NODES))['BlockSparseAttention'] == [
    'comfy.BlockSparseAttention'
]
"""
    env = {key: value for key, value in os.environ.items() if key != "DINKSTER_COMFY_NATIVE_ONLY"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name",
    (
        "audio",
        "devices",
        "image",
        "latent",
        "legacy_sources",
        "memory",
        "memory_policy",
        "model_aware_schedules",
        "native",
        "native_arm",
        "native_catalog",
        "native_residency",
        "pool",
        "preview_emit",
        "resident",
        "usdu",
        "video",
        "workgroup",
    ),
)
def test_compatibility_forwarders_preserve_native_module_identity(module_name: str) -> None:
    compatibility = importlib.import_module(f"dinkster_compat_comfy.{module_name}")
    native = importlib.import_module(f"dinkster_native.{module_name}")

    assert compatibility is native


def test_standalone_native_graph_executes_through_registered_provider() -> None:
    from dinkster_native.entry import NATIVE_NODES, register_types

    async def scenario() -> None:
        nodes = {node.schema().node_type: node for node in NATIVE_NODES}
        schemas = build_schemas(NATIVE_NODES)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(nodes, registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(
                nodes={
                    "preview": GraphNode(
                        "dinkster.preview_mask",
                        {"mask": [[0.0, 1.0], [1.0, 0.0]]},
                    )
                }
            ),
            ["preview"],
        )
        assert result.outputs["preview"]["mask"].resolve() == [[0.0, 1.0], [1.0, 0.0]]

    asyncio.run(scenario())


def test_native_entry_registers_every_schema_value_type() -> None:
    from dinkster_native.entry import NATIVE_NODES, register_types

    registry = TypeRegistry()
    register_core_types(registry)
    register_types(registry)
    referenced: set[str] = set()

    def collect(entries: Iterable[object]) -> None:
        for entry in entries:
            type_expr = getattr(entry, "type", None)
            if type_expr is not None:
                referenced.update(type_expr.types)

    for node in NATIVE_NODES:
        schema = node.schema()
        collect(schema.inputs)
        collect(schema.outputs)
        for family in schema.input_families:
            collect(family.template)
        for combo in schema.combos:
            for option in combo.options:
                collect(option.inputs)

    assert {type_id for type_id in referenced if type_id not in registry} == set()


@pytest.mark.parametrize("native_only", [True, False])
def test_native_manifest_catalogs_match_provider_claims(tmp_path: Path, native_only: bool) -> None:
    from dinkster_native.native_arm import (
        GENERATION_PROVIDER_NODES,
        NATIVE_ARM_NODES,
        NATIVE_ARM_TYPE_IDS,
    )
    from dinkster_native.native_catalog import COMFY_RUNTIME_NODE_IDS
    from dinkster_native.usdu import USDU_CARRIER_NODES
    from dinkster_nodes_generation import GENERATION_SCHEMA_NODES

    generation, provider = comfy_compat_specs(None if native_only else tmp_path)
    owner_manifest = load_manifest(generation.manifest)
    provider_manifest = load_manifest(provider.manifest)
    assert provider.packs is not None
    aliases = provider.packs["comfy"].comfy_aliases
    assert aliases is not None
    assert {
        record.source.node_class
        for record in aliases.records
        if record.source.revision == "b5cc8830279eae909a59de030af1e50761c36751"
    } == {
        "CLIPLoader",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3AddGuide",
        "ResolutionSelector",
    }
    excluded = COMFY_RUNTIME_NODE_IDS if native_only else frozenset()
    owner_types = {node.schema().node_type for node in GENERATION_SCHEMA_NODES}
    assert set(generation.optional_execution) == excluded
    assert COMFY_RUNTIME_NODE_IDS <= owner_types
    assert set(owner_manifest.schema_only) <= owner_types
    assert set(provider_manifest.executes) <= owner_types
    assert (
        set(provider_manifest.executes)
        == {node.schema().node_type for node in (*GENERATION_PROVIDER_NODES, *USDU_CARRIER_NODES)}
        - excluded
    )
    assert dict(provider_manifest.arms)["native"] == tuple(
        node.schema().node_type
        for node in NATIVE_ARM_NODES
        if node.schema().node_type not in excluded
    )
    assert NATIVE_ARM_TYPE_IDS == tuple(node.schema().node_type for node in NATIVE_ARM_NODES)
    if not native_only:
        assert COMFY_RUNTIME_NODE_IDS <= set(provider_manifest.executes)
    path = Path(provider.manifest)
    assert path.name == "dinkster-pack.toml"
    if native_only:
        compat_manifest = path.parents[1] / "dinkster-compat-comfy" / "dinkster-pack.toml"
        assert path.read_text() == native_manifest(compat_manifest.read_text())
    assert Path(generation.manifest).name == "dinkster-pack.toml"
    assert owner_manifest.nodes_entry == "dinkster_nodes_generation:GENERATION_SCHEMA_NODES"


def test_arm_declaration_drives_registry_and_both_manifests() -> None:
    from dinkster_native.native_arm import NATIVE_ARM_NODES, NATIVE_ARM_TYPE_IDS
    from dinkster_native.native_catalog import COMFY_RUNTIME_NODE_IDS

    compat_path = Path(__file__).parents[1] / "packages/dinkster-compat-comfy/dinkster-pack.toml"
    native_path = Path(__file__).parents[1] / "packages/dinkster-native/dinkster-pack.toml"
    compat_source = compat_path.read_text(encoding="utf-8")
    compat = load_manifest(compat_path)
    native = load_manifest(native_path)

    assert tuple(node.schema().node_type for node in NATIVE_ARM_NODES) == NATIVE_ARM_TYPE_IDS
    assert dict(compat.arms)["native"] == NATIVE_ARM_TYPE_IDS
    assert dict(native.arms)["native"] == tuple(
        type_id for type_id in NATIVE_ARM_TYPE_IDS if type_id not in COMFY_RUNTIME_NODE_IDS
    )
    assert compat_source == manifest_with_declared_arms(compat_source)
    assert native_path.read_text(encoding="utf-8") == native_manifest(compat_source)


def test_standalone_specs_preserve_native_worker_configuration(tmp_path: Path) -> None:
    config = SingleJobMultiGpuConfig(cuda_indices=(1, 0), mode="guidance")
    generation, native = comfy_compat_specs(
        asset_vault=tmp_path / "vault",
        mounts_snapshot=tmp_path / "mounts.json",
        single_job_multi_gpu=config,
        memory_budgets={"vram:cuda:1": 1024},
    )
    assert generation.in_process
    assert native.python == sys.executable
    assert native.env["DINKSTER_COMFY_NATIVE_ONLY"] == "1"
    assert native.env["DINKSTER_COMFYUI_ROOT"] == ""
    assert native.env["DINKSTER_ASSET_VAULT"] == str(tmp_path / "vault")
    assert native.single_job_cuda_indices == (1, 0)
    assert native.single_job_mode == "guidance"
    assert native.vram_budgets == {"vram:cuda:1": 1024}
    with pytest.raises(CompositionError, match="legacy packs require --comfy-root"):
        comfy_compat_specs(legacy_packs=[tmp_path / "custom_nodes"])


def test_legacy_pack_cli_requires_comfy_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster.serve",
            "--library-root",
            "",
            "--legacy-pack",
            str(tmp_path / "custom_nodes"),
        ],
        cwd=tmp_path,
        env={**os.environ, "DINKSTER_COMFYUI_ROOT": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--legacy-pack requires --comfy-root" in result.stderr


def test_standalone_source_staging_refuses_legacy_filenames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_COMFY_NATIVE_ONLY", "1")
    provider = ComfySourceStagingProvider()
    provider.sweep()
    with pytest.raises(SourceStagingError, match="require --comfy-root"):
        provider.open("invocation", [])


def test_import_catalog_tracks_active_aliases_without_registering_executable_ids() -> None:
    async def scenario() -> None:
        schemas = build_schemas((*GENERATION_NODES, LatentSink))
        registry = TypeRegistry()
        register_core_types(registry)

        def make_engine(on_event: EventListener | None = None) -> Engine:
            return Engine(
                schemas=schemas,
                registry=registry,
                worker=InProcessWorker({}, registry),
                cache=MemoryLRUCache(),
                on_event=on_event,
            )

        app = create_app(make_engine, schemas, choices=generation_choices())
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/compat/comfy/schemas")
            assert response.status == 200
            catalog = await response.json()
            for source, native in (
                ("KSampler", "dinkster.ksampler"),
                ("CLIPTextEncode", "dinkster.clip_text_encode"),
                ("EmptyLatentImage", "dinkster.empty_latent_image"),
            ):
                row = catalog["nodes"][f"comfy.{source}"]
                assert row["importable"] is True
                assert row["nativeNodeTypes"] == [native]
                assert f"comfy.{source}" not in app[STATE_KEY].schemas
            assert schemas["dinkster.empty_latent_image"].aliases == ("EmptyLatentImage",)
            translated = await client.post(
                "/api/compat/comfy/prompt?dryRun=1",
                json={
                    "5": {
                        "class_type": "comfy.EmptyLatentImage",
                        "inputs": {"width": 512, "height": 512, "batch_size": 1},
                    },
                    "9": {
                        "class_type": "LatentSink",
                        "inputs": {"latent": ["5", 0]},
                    },
                },
            )
            body = await translated.json()
            assert translated.status == 200, body
            nodes = body["graph"]["nodes"]
            assert nodes["5"]["nodeType"] == "dinkster.empty_latent_image"
            assert nodes["5"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
            assert nodes["9"]["inputs"]["latent"] == {"$link": {"node": "5", "output": "latent"}}
            assert body["targets"] == ["9"]
            assert catalog["nodes"]["comfy.CheckpointLoader"]["importable"] is False
            state = app[STATE_KEY]
            state.schemas = {**state.schemas}
            state.schemas.pop("dinkster.ksampler")
            catalog = await (await client.get("/api/compat/comfy/schemas")).json()
            assert catalog["nodes"]["comfy.KSampler"]["nativeNodeTypes"] == []
            state.schemas["test.other"] = replace(
                schemas["dinkster.clip_text_encode"], node_type="test.other"
            )
            catalog = await (await client.get("/api/compat/comfy/schemas")).json()
            assert catalog["nodes"]["comfy.CLIPTextEncode"]["importable"] is False
        finally:
            await client.close()

    asyncio.run(scenario())
