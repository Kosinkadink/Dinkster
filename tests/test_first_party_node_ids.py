"""The accepted pack cut preserves every ported class and a loadable identity."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from dinkster_api.v1 import with_identity_replacement
from dinkster_graph import Graph, GraphNode, migrate_pure_node_type_replacements
from dinkster_schema import Node, NodeSchema, claim_covers, validate_name
from dinkster_schema.replace import ReplacementRule, rule_to_wire, validate_replacement_references
from dinkster_schema.wire import schema_from_wire, schema_to_wire
from dinkster_text.text import StringConcatenate, StringFormat
from dinkster_workers import load_manifest
from dinkster_workers.manifest import resolve_entry, validate_registration_claims
from packaging.version import Version

from dinkster.compose import default_pack_specs
from tests.async_scenarios import run_scenario


def _source_schemas() -> dict[str, NodeSchema]:
    schemas = {}
    for spec in default_pack_specs():
        manifest = load_manifest(spec.manifest)
        nodes = resolve_entry(manifest.nodes_entry)
        assert isinstance(nodes, (list, tuple))
        for node in nodes:
            assert isinstance(node, type) and issubclass(node, Node)
            schema = node.schema()
            for rule in schema.replacements:
                assert rule.from_type not in schemas
                schemas[rule.from_type] = schema
    return schemas


def test_default_install_set_contains_every_comfy_origin_pack() -> None:
    expected = {
        load_manifest(path).name
        for path in Path("packages").glob("dinkster-nodes-*/dinkster-pack.toml")
        if path.parent.name != "dinkster-nodes-dev"
    }
    defaults = tomllib.loads(Path("src/dinkster/default-packs.toml").read_text())["packs"]
    assert defaults.keys() == expected
    declarations = {
        manifest.name: manifest.compatibility.version
        for path in Path("packages").glob("dinkster-nodes-*/dinkster-pack.toml")
        if (manifest := load_manifest(path)).compatibility is not None and manifest.name in expected
    }
    assert defaults.keys() == declarations.keys()
    assert all(
        Version(declarations[name]) >= Version(minimum) for name, minimum in defaults.items()
    )
    specs = default_pack_specs()
    assert {load_manifest(spec.manifest).name for spec in specs} == expected
    assert len(specs) == len(expected)
    assert all(not spec.in_process for spec in specs)


def test_first_party_node_ids_preserve_the_pack_cut() -> None:
    seen: set[str] = set()
    for spec in default_pack_specs():
        manifest = load_manifest(spec.manifest)
        validate_registration_claims(manifest)
        nodes = resolve_entry(manifest.nodes_entry)
        assert isinstance(nodes, (list, tuple))
        for node in nodes:
            schema = node.schema()
            assert validate_name(schema.node_type) is None
            assert schema.node_type not in seen
            seen.add(schema.node_type)
            assert any(
                claim_covers(namespace, schema.node_type) for namespace in manifest.namespaces
            )
            assert validate_replacement_references({schema.node_type: schema}) == ()
    assert len(seen) == 646
    sources = _source_schemas()
    assert len(sources) == 645
    assert len({schema.node_type for schema in sources.values()}) == 644


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


def test_retained_image_classes_have_ordinary_identity_rules() -> None:
    from dinkster_image.ported.compare import NativeImageCompare
    from dinkster_image.ported.glsl import NativeGlslShader

    schemas = []
    for node, expected, source in (
        (NativeImageCompare, "dinkster.image.image_compare", "ImageCompare"),
        (NativeGlslShader, "dinkster.image.glsl_shader", "GLSLShader"),
    ):
        renamed = node.schema()
        assert renamed.node_type == expected
        assert renamed.aliases == ()
        assert len(renamed.replacements) == 1
        assert renamed.replacements[0].from_type == source
        assert renamed.replacements[0].migration is None
        schemas.append(renamed)
    assert {schema.replacements[0].from_type for schema in schemas} == {
        "ImageCompare",
        "GLSLShader",
    }


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


def test_translation_interfaces_retain_positional_order_after_json_round_trip() -> None:
    flat, family = json.loads(
        json.dumps(
            [
                {"rule": rule_to_wire(schema.replacements[0]), "schema": schema_to_wire(schema)}
                for schema in (StringConcatenate.schema(), StringFormat.schema())
            ]
        )
    )
    assert [item["rule"]["from"] for item in (flat, family)] == [
        "StringConcatenate",
        "StringFormat",
    ]
    assert flat["schema"]["nodeType"] == "dinkster.text.string_concatenate"
    assert [entry["id"] for entry in flat["schema"]["interface"]] == [
        "string_a",
        "string_b",
        "delimiter",
        "STRING",
    ]
    assert [entry["id"] for entry in family["schema"]["interface"]] == [
        "values",
        "f_string",
        "STRING",
    ]
    assert family["schema"]["interface"][0]["memberNames"] == list("abcdefghijklmnopqrstuvwxyz")
    assert family["schema"]["interface"][0]["template"][0]["id"] == "value"
    assert family["schema"]["interface"][1]["default"] == "{a}"
    assert family["schema"]["interface"][1]["widget"] == {"type": "STRING", "multiline": True}
    decoded = schema_from_wire(family["schema"])
    assert decoded.node_type == "dinkster.text.string_format"
    assert tuple(entry.id for entry in decoded.input_entries) == ("values", "f_string")
    assert decoded.replacements[0].from_type == "StringFormat"


@pytest.mark.parametrize(
    "source",
    [
        "ReferenceMultiLatent",
        "TextEncodeMOSSImageEdit",
        "MergeSplat",
        "MergeMeshes",
        "ComfyAndNode",
        "ComfyOrNode",
        "ComfyMathExpression",
        "BatchImagesNode",
        "BatchMasksNode",
        "BatchLatentsNode",
        "CreateList",
        "ConcatenateVideo",
    ],
)
def test_renamed_prefixed_families_validate_without_serving_the_source(source: str) -> None:
    schema = _source_schemas()[source]
    original = replace(schema, replacements=())
    assert with_identity_replacement(original, source) == schema
    assert validate_replacement_references({schema.node_type: schema}) == ()
    assert schema.input_families == original.input_families
    assert len(schema.replacements) == 1
    assert schema.replacements[0].from_type == source
    assert schema.replacements[0].migration is None
    for case in schema.replacements[-1].cases:
        for family, mapping in case.input_families:
            assert mapping.source_family == family
            assert mapping.member_ids is None


def test_nested_rename_rules_map_only_the_selected_branch_inputs() -> None:
    schema = _source_schemas()["BuildPoseFile"]
    assert validate_replacement_references({schema.node_type: schema}) == ()
    rule = schema.replacements[-1]
    case = next(
        case
        for case in rule.cases
        if dict(case.slot_variants)
        == {
            "format": "glb",
            "format.mesh_style": "body_mesh",
            "format.mesh_style.bone_vis": "octahedrons",
            "format.mesh_style.shader": "rainbow",
        }
    )
    mappings = dict(case.inputs)
    assert (
        mappings["format.mesh_style.bone_vis.bone_vis_radius_m"].input
        == "format.mesh_style.bone_vis.bone_vis_radius_m"
    )
    assert (
        mappings["format.mesh_style.shader.rainbow_tilt_z"].input
        == "format.mesh_style.shader.rainbow_tilt_z"
    )
    assert "format.units" not in mappings
    assert len(rule.cases) == 12  # 2 bone modes * 4 shaders, 3 other GLB styles, BVH.


@pytest.mark.parametrize(
    ("pack", "required"),
    [
        ("3d", {"MESH", "SPLAT", "VOXEL", "LOAD3D_CAMERA", "LOAD3D_MODEL_INFO", "VIDEO"}),
        ("3d_models", {"MODEL", "VAE", "CONDITIONING", "LATENT", "SHAPE_SUBDIVIDES"}),
        ("vision_background_removal", {"ASSET", "BACKGROUND_REMOVAL"}),
        ("vision_marigold", {"IMAGE"}),
        ("vision_moge", {"ASSET", "MOGE_MODEL", "MOGE_GEOMETRY", "MESH"}),
        ("vision_rtdetr", {"MODEL", "BOUNDING_BOX"}),
        ("vision_sam3", {"MODEL", "CONDITIONING", "BOUNDING_BOX", "SAM3_TRACK_DATA"}),
        ("vision_sdpose", {"MODEL", "VAE", "BOUNDING_BOX", "POSE_KEYPOINT"}),
    ],
)
def test_geometry_and_vision_hooks_do_not_register_unrelated_family_codecs(
    pack: str, required: set[str]
) -> None:
    from dinkster_values import TypeRegistry, register_core_types

    registry = TypeRegistry()
    register_core_types(registry)
    hook = import_module(f"dinkster_{pack}").register_types
    hook(registry)
    assert required <= set(registry.type_ids())
    assert not {
        "DA3_MODEL",
        "DA3_GEOMETRY",
        "SAM3D_BODY_MODEL",
        "MHR_POSE_DATA",
        "FACE_DETECTION_MODEL",
        "FACE_LANDMARKS",
        "PHOTOMAKER",
        "INTERP_MODEL",
        "OPTICAL_FLOW",
        "WAN_CAMERA_EMBEDDING",
        "TRACKS",
        "IC_LORA_PARAMETERS",
    } & set(registry.type_ids())
    specs = {type_id: registry.spec(type_id) for type_id in required}
    hook(registry)
    assert all(registry.spec(type_id) is spec for type_id, spec in specs.items())


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
