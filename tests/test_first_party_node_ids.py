"""The accepted pack cut preserves every ported class and a loadable identity."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import replace
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, migrate_pure_node_type_replacements
from dinkster_schema import Node, NodeSchema, claim_covers, validate_name
from dinkster_schema.first_party import collect_comfy_translations, first_party_schema
from dinkster_schema.replace import ReplacementRule, rule_to_wire, validate_replacement_references
from dinkster_schema.wire import schema_from_wire, schema_to_wire
from dinkster_text.text import StringConcatenate, StringFormat
from dinkster_workers import load_manifest
from dinkster_workers.manifest import resolve_entry, validate_registration_claims
from packaging.version import Version

from dinkster.compose import default_pack_specs
from tests.async_scenarios import run_scenario
from tools.compare_nodes import SCOPE_RULES, compare_registry


def _source_schemas() -> dict[str, NodeSchema]:
    inventory = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    packs = {row["pack"] for row in inventory["nodes"]}
    schemas = {}
    for path in Path("packages").glob("dinkster-nodes-*/dinkster-pack.toml"):
        manifest = load_manifest(path)
        if manifest.name not in packs:
            continue
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
    inventory = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    expected = {row["pack"] for row in inventory["nodes"]} | {"dinkster-nodes-remote"}
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


def test_first_party_node_ids_preserve_the_pack_cut(tmp_path: Path) -> None:
    inventory = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    rows = inventory["nodes"]
    support = inventory["supportNodes"]
    assert support == [
        {
            "pack": "dinkster-utilities",
            "nodeType": node_type,
            "bodyNotFromUpstream": True,
        }
        for node_type in ("dinkster.utilities.make_list", "dinkster.utilities.list_element")
    ]
    assert not ({row["nodeType"] for row in support} & {row["nodeType"] for row in rows})
    assert collect_comfy_translations(
        [
            ("dinkster", "dinkster-utilities", "0.0.1", [NodeSchema(node_type=row["nodeType"])])
            for row in support
        ]
    ) == {"schemaVersion": 1, "items": []}
    assert inventory["schemaVersion"] == 1
    assert inventory["referenceCommit"] == "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
    assert len(rows) == len({row["from"] for row in rows}) == 644
    assert len({row["nodeType"] for row in rows}) == 644
    assert Counter(row["pack"].removeprefix("dinkster-") for row in rows[:636]) == {
        "sampling": 63,
        "models": 101,
        "conditioning": 29,
        "latent": 45,
        "image": 65,
        "upscale": 2,
        "audio": 15,
        "video": 12,
        "3d": 39,
        "text": 16,
        "llm": 1,
        "utilities": 25,
        "sd": 23,
        "flux": 10,
        "ltx": 30,
        "wan": 32,
        "hunyuan": 14,
        "qwen": 6,
        "hidream": 4,
        "minimax": 9,
        "cosmos": 3,
        "void": 5,
        "seedvr": 5,
        "audio-models": 9,
        "3d-models": 14,
        "misc-models": 28,
        "vision-background-removal": 2,
        "vision-depth-anything-3": 4,
        "vision-marigold": 1,
        "vision-mediapipe": 4,
        "vision-moge": 5,
        "vision-rtdetr": 2,
        "vision-sam3": 4,
        "vision-sam3d-body": 6,
        "vision-sdpose": 3,
    }
    schemas = _source_schemas()
    expected_sources = {row["from"]: row["nodeType"] for row in rows}
    expected_sources["ResizeImagesByLongerEdge"] = "dinkster.image.image_scale_to_max_dimension"
    assert expected_sources == {source: schema.node_type for source, schema in schemas.items()}
    assert {row["from"]: row["nodeType"] for row in rows[636:]} == {
        "CreateBoundingBoxes": "dinkster.image.create_bounding_boxes",
        "LayersFromBoundingBoxes": "dinkster.image.layers_from_bounding_boxes",
        "ImageCompositor": "dinkster.image.image_compositor",
        "WebcamCapture": "dinkster.image.webcam_capture",
        "LoadMoGeModel": "dinkster.vision.moge.load_mo_ge_model",
        "SplatToMesh": "dinkster.3d.splat_to_mesh",
        "ImageCompare": "dinkster.image.image_compare",
        "GLSLShader": "dinkster.image.glsl_shader",
    }
    assert {row["from"] for row in rows if row.get("bodyNotFromUpstream") is True} == {
        "CreateBoundingBoxes",
        "LayersFromBoundingBoxes",
        "ImageCompositor",
        "WebcamCapture",
        "LoadMoGeModel",
        "SplatToMesh",
    }
    assert all("bodyNotFromUpstream" not in row for row in rows[:636])
    reference = {
        row["class_name"]: row
        for row in json.loads(Path("tools/data/comfy_core_reference.json").read_text())["nodes"]
    }
    for row in rows[636:642]:
        assert row["backendPureRename"] is True
        source = reference[row["from"]]
        for group in ("required", "optional"):
            for spec in source["input_types"].get(group, {}).values():
                assert not str(spec[0]).startswith("COMFY_")
        assert all(not value.startswith("COMFY_") for value in source["return_types"])
    for row in rows[:636]:
        schema = schemas[row["from"]]
        assert row["backendPureRename"] is not bool(
            schema.input_families or schema.output_families or schema.combos or schema.slots
        )
    for row in [*rows, *support]:
        assert validate_name(row["nodeType"]) is None
        namespace, leaf = row["nodeType"].rsplit(".", 1)
        assert namespace == row["pack"].replace("-", ".")
        assert leaf and all(c.isascii() and (c.islower() or c.isdigit() or c == "_") for c in leaf)
        manifest_path = tmp_path / "dinkster-pack.toml"
        manifest_path.write_text(
            f'[pack]\nname = "{row["pack"]}"\npublisher = "dinkster"\n'
            f'namespaces = ["{namespace}"]\n'
            '[pack.entry]\nnodes = "example:NODES"\n'
        )
        manifest = load_manifest(manifest_path)
        validate_registration_claims(manifest)
        assert manifest.namespaces == (namespace,)
        assert claim_covers(manifest.namespaces[0], row["nodeType"])

    scope = json.loads(SCOPE_RULES.read_text())
    scope["migratedPacks"] = ["dinkster-image", "dinkster-vision-moge", "dinkster-3d"]
    census = compare_registry(
        {"nodes": [reference[row["from"]] for row in rows[636:]]},
        {"nodes": {}},
        scope=scope,
    )
    assert census["eligible_node_count"] == 8
    assert census["counts"] == {"absent": 8}
    assert census["pack_counts"] == {
        "dinkster-image": {"absent": 6},
        "dinkster-vision-moge": {"absent": 1},
        "dinkster-3d": {"absent": 1},
    }


@pytest.mark.parametrize("display_name", ["", "Join text explicitly"])
def test_pack_identity_preserves_the_effective_display_name(display_name: str) -> None:
    original = NodeSchema(node_type="StringConcatenate", display_name=display_name)
    renamed = first_party_schema(original)
    assert renamed.node_type == "dinkster.text.string_concatenate"
    assert schema_to_wire(renamed)["displayName"] == (display_name or "StringConcatenate")


def test_retained_image_classes_have_ordinary_identity_rules() -> None:
    from dinkster_image.compare import NativeImageCompare
    from dinkster_image.glsl import NativeGlslShader

    schemas = []
    for node, expected in (
        (NativeImageCompare, "dinkster.image.image_compare"),
        (NativeGlslShader, "dinkster.image.glsl_shader"),
    ):
        original = node.schema()
        renamed = first_party_schema(original)
        assert renamed.node_type == expected
        assert renamed.aliases == ()
        assert len(renamed.replacements) == 1
        assert renamed.replacements[0].from_type == original.node_type
        assert renamed.replacements[0].migration is None
        assert (
            replace(
                renamed,
                node_type=original.node_type,
                aliases=original.aliases,
                search_terms=original.search_terms,
                replacements=original.replacements,
            )
            == original
        )
        schemas.append(renamed)
    index = collect_comfy_translations([("dinkster", "dinkster-image", "0.0.1", schemas)])
    items = index["items"]
    assert isinstance(items, list)
    assert {item["rule"]["from"] for item in items} == {"ImageCompare", "GLSLShader"}


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


def test_translation_index_collects_schema_rules_and_refuses_wrong_claims() -> None:
    schema = StringConcatenate.schema()
    pack = ("dinkster", "dinkster-text", "0.0.1", (schema,))
    assert collect_comfy_translations([pack]) == {
        "schemaVersion": 1,
        "items": [
            {
                "pack": "dinkster-text",
                "version": "0.0.1",
                "rule": rule_to_wire(schema.replacements[0]),
                "schema": schema_to_wire(schema),
            }
        ],
    }
    with pytest.raises(ValueError, match="comfy-translations.publisher"):
        collect_comfy_translations([("outsider", *pack[1:])])
    with pytest.raises(ValueError, match="comfy-translations.owner"):
        collect_comfy_translations([("dinkster", "dinkster-image", "0.0.1", (schema,))])
    with pytest.raises(ValueError, match="comfy-translations.duplicate"):
        collect_comfy_translations([pack, pack])
    altered = replace(
        schema,
        replacements=(
            ReplacementRule(
                "StringConcatenate", (replace(schema.replacements[0].cases[0], outputs=()),)
            ),
        ),
    )
    with pytest.raises(ValueError, match="comfy-translations.rule"):
        collect_comfy_translations([("dinkster", "dinkster-text", "0.0.1", (altered,))])


def test_translation_interfaces_retain_positional_order_after_json_round_trip() -> None:
    index = json.loads(
        json.dumps(
            collect_comfy_translations(
                [
                    (
                        "dinkster",
                        "dinkster-text",
                        "0.0.1",
                        (StringFormat.schema(), StringConcatenate.schema()),
                    )
                ]
            )
        )
    )
    flat, family = index["items"]
    assert [item["rule"]["from"] for item in index["items"]] == [
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
    original = replace(schema, node_type=source, replacements=())
    assert first_party_schema(original) == schema
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
    "injected",
    [
        "import dinkster_inference.utils",
        "from dinkster_inference import utils",
        "from dinkster_inference.utils import common_upscale",
        "import importlib as loader; loader.import_module('dinkster_inference.utils')",
        "from importlib import import_module as load; load('dinkster_inference.utils')",
        "__import__('dinkster_inference.utils')",
    ],
)
def test_pack_inference_declaration_rejects_undeclared_import(
    tmp_path: Path, injected: str
) -> None:
    from scripts.check_pack_inference_imports import check_pack

    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text('[pack.inference]\nimports = []\ntested-version = "0.1.1"\n')
    source = tmp_path / "src"
    source.mkdir()
    code = source / "node.py"
    code.write_text("from .utils import local_helper\n")
    check_pack(tmp_path, "0.1.1")
    code.write_text(injected + "\n")
    with pytest.raises(ValueError, match="inference imports must be.*utils"):
        check_pack(tmp_path, "0.1.1")
    manifest.write_text('[pack.inference]\nimports = ["utils"]\ntested-version = "0.1.1"\n')
    check_pack(tmp_path, "0.1.1")
    with pytest.raises(ValueError, match="tested-version must be 0.1.2"):
        check_pack(tmp_path, "0.1.2")


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
    inventory = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    for name in sorted({row["pack"] for row in inventory["nodes"]}):
        pack = name.removeprefix("dinkster-")
        manifest = load_manifest(root / f"dinkster-nodes-{pack}/dinkster-pack.toml")
        _worker, isolated_registry, _nodes, _arms = load_pack(manifest)
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
                    root / f"dinkster-nodes-{name.removeprefix('dinkster-')}/dinkster-pack.toml",
                    in_process=True,
                    trust_reserved=True,
                    packs={name: PackInfo(name, version="0.0.1")},
                )
                for name in sorted({row["pack"] for row in inventory["nodes"]})
            ]
            for spec in composer.order_pack_entries(specs):
                await composer.add_pack(spec)
            assert set(composer.composition.schemas) == {
                row["nodeType"] for row in inventory["nodes"] + inventory["supportNodes"]
            }
        finally:
            await composer.close()

    run_scenario(compose_all())
