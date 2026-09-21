from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

from dinkster_model_triposplat import TRIPOSPLAT_MODEL_NODES
from dinkster_native.native import NATIVE_NODES
from dinkster_nodes_generation import GENERATION_SCHEMA_NODES
from dinkster_workers import load_pack_templates

from dinkster.compose import compose_serving

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "dinkster-nodes-generation": ROOT / "packages/dinkster-nodes-generation/dinkster-pack.toml",
    "dinkster-model-qwen-image": ROOT / "packages/dinkster-model-qwen-image/dinkster-pack.toml",
    "dinkster-model-wan": ROOT / "packages/dinkster-model-wan/dinkster-pack.toml",
    "dinkster-model-triposplat": ROOT / "packages/dinkster-model-triposplat/dinkster-pack.toml",
}

EXPECTED_FAMILIES = {
    "dinkster.sd15",
    "dinkster.sdxl",
    "dinkster.sdxl_refiner",
    "dinkster.chroma",
    "dinkster.chroma_radiance",
    "dinkster.flux_dev",
    "dinkster.flux_schnell",
    "dinkster.flux2_dev",
    "dinkster.flux2_klein_9b",
    "dinkster.flux2_klein_4b",
    "dinkster.wan21",
    "dinkster.wan22",
    "dinkster.ltxv",
    "dinkster.ltxav",
    "dinkster.qwen_image",
    "dinkster.z_image",
    "dinkster.z_image_pixel_space",
    "dinkster.minimax_h3",
    "dinkster.minimax_music3",
    "dinkster.krea2",
    "dinkster.ideogram4",
    "dinkster.seedvr2",
    "dinkster.anima",
    "dinkster.lumina2",
    "dinkster.triposplat",
    "dinkster.trellis2",
}


def test_generation_pack_has_one_complete_starter_per_supported_family() -> None:
    by_pack = {
        pack: load_pack_templates(manifest, pack=pack) for pack, manifest in MANIFESTS.items()
    }
    templates = tuple(template for values in by_pack.values() for template in values)
    assert len(templates) == len(EXPECTED_FAMILIES)
    assert {template.family for template in templates} == EXPECTED_FAMILIES
    assert [template.id for template in by_pack["dinkster-model-qwen-image"]] == ["qwen-image"]
    assert [template.id for template in by_pack["dinkster-model-wan"]] == [
        "wan21",
        "wan22",
    ]
    assert [template.id for template in by_pack["dinkster-model-triposplat"]] == ["triposplat"]
    assert all(
        template.description and template.models and template.thumbnail for template in templates
    )
    assert all(
        template.thumbnail and template.thumbnail.media_type == "image/png"
        for template in templates
    )
    node_type_sets: set[frozenset[str]] = set()
    for template in templates:
        document = cast("dict[str, object]", json.loads(template.data))
        graphs = cast("dict[str, dict[str, object]]", document["graphs"])
        nodes = cast("dict[str, dict[str, object]]", graphs["g0"]["nodes"])
        serialized_values = json.dumps(
            [node.get("values", {}) for node in nodes.values()], sort_keys=True
        )
        assert all(model in serialized_values for model in template.models), template.id
        node_type_sets.add(frozenset(cast("str", node["type"]) for node in nodes.values()))
    assert len(node_type_sets) >= 8


def test_every_starter_resolves_against_current_schemas() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            schemas = {
                **composition.schemas,
                **{node.schema().node_type: node.schema() for node in NATIVE_NODES},
                **{node.schema().node_type: node.schema() for node in GENERATION_SCHEMA_NODES},
                **{node.schema().node_type: node.schema() for node in TRIPOSPLAT_MODEL_NODES},
            }
            for pack, manifest in MANIFESTS.items():
                for template in load_pack_templates(manifest, pack=pack):
                    document = cast("dict[str, object]", json.loads(template.data))
                    graphs = cast("dict[str, dict[str, object]]", document["graphs"])
                    for graph in graphs.values():
                        nodes = cast("dict[str, dict[str, object]]", graph["nodes"])
                        links = cast("dict[str, dict[str, object]]", graph["links"])
                        linked_inputs: set[tuple[str, str]] = set()
                        for link in links.values():
                            source_ref = cast("dict[str, str]", link["from"])
                            target_ref = cast("dict[str, str]", link["to"])
                            source_node = nodes[source_ref["node"]]
                            target_node = nodes[target_ref["node"]]
                            source = schemas[cast("str", source_node["type"])]
                            target = schemas[cast("str", target_node["type"])]
                            output = next(
                                spec for spec in source.outputs if spec.id == source_ref["port"]
                            )
                            input_spec = next(
                                spec for spec in target.inputs if spec.id == target_ref["port"]
                            )
                            runtime_type = output.type.runtime_type_id()
                            assert runtime_type is not None, template.id
                            assert input_spec.type.accepts_concrete(runtime_type), (
                                template.id,
                                source_ref,
                                target_ref,
                                runtime_type,
                            )
                            linked_inputs.add((target_ref["node"], target_ref["port"]))
                        for node_id, node in nodes.items():
                            schema = schemas[cast("str", node["type"])]
                            values = cast("dict[str, object]", node.get("values", {}))
                            assert set(values) <= {spec.id for spec in schema.inputs}, template.id
                            for spec in schema.inputs:
                                if spec.required and spec.default is None:
                                    assert (
                                        spec.id in values
                                        or (
                                            node_id,
                                            spec.id,
                                        )
                                        in linked_inputs
                                    ), (template.id, node_id, spec.id)
        finally:
            await composition.close()

    asyncio.run(scenario())
