"""The accepted pack cut preserves every ported class and a loadable identity."""

from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files
from pathlib import Path

from dinkster_nodes_comfy import COMFY_NODES
from dinkster_schema import claim_covers, validate_name
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
    assert {row["from"] for row in rows[:636]} == schemas.keys()
    assert {row["from"]: row["nodeType"] for row in rows[636:]} == {
        "CreateBoundingBoxes": "dinkster.image.create_bounding_boxes",
        "LayersFromBoundingBoxes": "dinkster.image.layers_from_bounding_boxes",
        "ImageCompositor": "dinkster.image.image_compositor",
        "WebcamCapture": "dinkster.image.webcam_capture",
        "LoadMoGeModel": "dinkster.vision.moge.load_mo_ge_model",
        "SplatToMesh": "dinkster.3d.splat_to_mesh",
    }
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
