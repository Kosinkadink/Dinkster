"""The accepted pack cut preserves every ported class and a loadable identity."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, migrate_pure_node_type_replacements
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_text import NODES as TEXT_NODES
from dinkster_nodes_text.text import StringConcatenate, StringFormat
from dinkster_nodes_upscale import NODES as UPSCALE_NODES
from dinkster_schema import claim_covers, validate_name
from dinkster_schema.first_party import collect_comfy_translations, first_party_schema
from dinkster_schema.replace import ReplacementRule, rule_to_wire, validate_replacement_references
from dinkster_schema.wire import schema_from_wire, schema_to_wire
from dinkster_workers import load_manifest
from dinkster_workers.manifest import validate_registration_claims

from tools.compare_nodes import SCOPE_RULES, compare_registry


def test_first_party_node_ids_preserve_the_pack_cut(tmp_path: Path) -> None:
    inventory = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    rows = inventory["nodes"]
    assert inventory["schemaVersion"] == 1
    assert inventory["referenceCommit"] == "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
    assert len(rows) == len({row["from"] for row in rows}) == 642
    assert len({row["nodeType"] for row in rows}) == 642
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
    schemas = {node.schema().node_type: node.schema() for node in COMFY_NODES}
    schemas.update(
        {
            node.schema().replacements[-1].from_type: node.schema()
            for node in (*TEXT_NODES, *UPSCALE_NODES)
        }
    )
    assert {row["from"] for row in rows[:636]} == schemas.keys()
    assert {row["from"]: row["nodeType"] for row in rows[636:]} == {
        "CreateBoundingBoxes": "dinkster.image.create_bounding_boxes",
        "LayersFromBoundingBoxes": "dinkster.image.layers_from_bounding_boxes",
        "ImageCompositor": "dinkster.image.image_compositor",
        "WebcamCapture": "dinkster.image.webcam_capture",
        "LoadMoGeModel": "dinkster.vision.moge.load_mo_ge_model",
        "SplatToMesh": "dinkster.3d.splat_to_mesh",
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
    for row in rows[636:]:
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
    for row in rows:
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
    assert census["eligible_node_count"] == 6
    assert census["counts"] == {"absent": 6}
    assert census["pack_counts"] == {
        "dinkster-image": {"absent": 4},
        "dinkster-vision-moge": {"absent": 1},
        "dinkster-3d": {"absent": 1},
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
    original = next(node.schema() for node in COMFY_NODES if node.schema().node_type == source)
    schema = first_party_schema(original)
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
    original = next(
        node.schema() for node in COMFY_NODES if node.schema().node_type == "BuildPoseFile"
    )
    schema = first_party_schema(original)
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
