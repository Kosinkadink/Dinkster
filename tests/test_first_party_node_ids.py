"""The accepted pack cut preserves every ported class and a loadable identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_api.v1 import with_identity_replacement
from dinkster_graph import Graph, GraphNode, migrate_pure_node_type_replacements
from dinkster_schema import NodeSchema
from dinkster_schema.replace import ReplacementRule
from dinkster_schema.wire import schema_to_wire
from dinkster_text.text import StringConcatenate, StringFormat
from dinkster_workers import load_manifest

from dinkster.compose import default_pack_specs
from tests.async_scenarios import run_scenario


@pytest.mark.parametrize("display_name", ["", "Authored name"])
def test_identity_replacement_preserves_authored_id(display_name: str) -> None:
    schema = NodeSchema(
        node_type="independent.pack.join",
        display_name=display_name,
        search_terms=("join", "ExternalJoin"),
    )
    declared = with_identity_replacement(schema, "ExternalJoin")
    assert declared.node_type == "independent.pack.join"
    assert declared.display_name == (display_name or "ExternalJoin")
    assert declared.search_terms == ("join", "ExternalJoin")
    assert len(declared.replacements) == 1
    assert declared.replacements[0].from_type == "ExternalJoin"
    assert declared.replacements[0].cases[0].to == "independent.pack.join"
    assert schema.replacements == ()


def test_first_party_rules_preserve_flat_graphs_and_leave_families_frontend_owned() -> None:
    schema = StringConcatenate.schema()
    original = replace(schema, node_type="StringConcatenate", replacements=(), search_terms=())
    family = StringFormat.schema()
    graph = Graph(
        {
            "flat": GraphNode(
                "StringConcatenate", {"string_a": "left", "string_b": "RIGHT", "delimiter": "/"}
            ),
            "dynamic": GraphNode("StringFormat", {"values.z": "last", "f_string": "{z}"}),
        }
    )
    migrated = migrate_pure_node_type_replacements(
        graph, {schema.node_type: schema, family.node_type: family}
    )
    assert migrated.nodes["flat"] == GraphNode(
        "dinkster.text.string_concatenate",
        {"string_a": "left", "string_b": "RIGHT", "delimiter": "/"},
    )
    assert migrated.nodes["dynamic"] == graph.nodes["dynamic"]
    assert schema.inputs == original.inputs
    assert schema.outputs == original.outputs
    assert schema.aliases == ()
    assert schema.search_terms == (*original.search_terms, "StringConcatenate")
    assert family.replacements[0].cases[0].input_families[0][1].source_template == ("value",)
    duplicate = replace(schema, node_type="other.node")
    duplicate = replace(
        duplicate,
        replacements=(
            ReplacementRule(
                "StringConcatenate",
                (replace(schema.replacements[0].cases[0], to="other.node"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicting pure replacements"):
        migrate_pure_node_type_replacements(
            graph, {schema.node_type: schema, duplicate.node_type: duplicate}
        )


def test_producer_pack_type_ownership_survives_ordinary_loading() -> None:
    from dinkster_inference_wire import register_inference_types
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers.host import load_pack

    producers = {
        "wan": {"WAN_CAMERA_EMBEDDING", "TRACKS"},
        "ltx": {"IC_LORA_PARAMETERS"},
        "sd": {"PHOTOMAKER"},
        "video": {"INTERP_MODEL", "OPTICAL_FLOW"},
        "upscale": {"UPSCALE_MODEL"},
        "vision-depth-anything-3": {"DA3_MODEL", "DA3_GEOMETRY"},
        "vision-mediapipe": {"FACE_DETECTION_MODEL", "FACE_LANDMARKS"},
        "vision-sam3": {"SAM3_TRACK_DATA"},
        "vision-sam3d-body": {"SAM3D_BODY_MODEL", "MHR_POSE_DATA"},
    }
    registry = TypeRegistry()
    register_core_types(registry)
    register_inference_types(registry)
    for types in producers.values():
        for type_id in types:
            assert registry.owner_of(type_id) is None
            assert type_id not in registry
    root = Path(__file__).resolve().parents[1] / "packages"
    manifests = [load_manifest(spec.manifest) for spec in default_pack_specs()]
    expected_ids: set[str] = set()
    for manifest in manifests:
        name = manifest.name
        pack = name.removeprefix("dinkster-")
        _worker, isolated_registry, _nodes, _arms = load_pack(manifest)
        expected_ids.update(_worker.schemas)
        for schema in _worker.schemas.values():
            pending: list[object] = [schema_to_wire(schema)]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    if value.get("kind") == "concrete":
                        for atom in value["types"]:
                            assert atom in isolated_registry, (name, schema.node_type, atom)
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
        for type_id in producers.get(pack, ()):
            assert isolated_registry.owner_of(type_id) == name
            assert isolated_registry.spec(type_id).declared_codec
        if pack == "vision-sam3d-body":
            assert isolated_registry.owner_of("SAM3_TRACK_DATA") == "dinkster-vision-sam3"
        if pack == "wan":
            assert isolated_registry.owner_of("SAM3_TRACK_DATA") == "dinkster-vision-sam3"
        if pack == "void":
            assert isolated_registry.owner_of("OPTICAL_FLOW") == "dinkster-video"
    for pack, types in producers.items():
        manifest = load_manifest(root / f"dinkster-nodes-{pack}/dinkster-pack.toml")
        load_pack(manifest, registry=registry)
        for type_id in types:
            assert registry.owner_of(type_id) == manifest.name
            assert registry.spec(type_id).declared_codec
            with pytest.raises(ValueError, match="owned by"):
                registry.claim_types("unrelated", (type_id,))
    assert registry.owner_of("SAM3_TRACK_DATA") == "dinkster-vision-sam3"
    registry.remove_pack_types("dinkster-video")
    assert registry.owner_of("INTERP_MODEL") is None
    assert "INTERP_MODEL" not in registry
    assert registry.owner_of("PHOTOMAKER") == "dinkster-sd"
    assert registry.owner_of("IMAGE") == "core"

    async def compose_all() -> None:
        from dinkster_server import PackInfo

        from dinkster.compose import PackSpec, ServingComposer

        composer = ServingComposer()
        try:
            specs = [
                PackSpec(
                    manifest.path,
                    in_process=True,
                    trust_reserved=True,
                    packs={manifest.name: PackInfo(manifest.name, version="0.0.1")},
                )
                for manifest in manifests
            ]
            for spec in composer.order_pack_entries(specs):
                await composer.add_pack(spec)
            assert set(composer.composition.schemas) == expected_ids
        finally:
            await composer.close()

    run_scenario(compose_all())
