"""Core class names migrate to their first-party schemas without rewriting values."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from dinkster_graph import Graph, GraphNode, Link, migrate_pure_node_type_replacements
from dinkster_schema import MappingSource, schema_from_wire, schema_to_wire


@pytest.mark.parametrize(
    ("module", "source", "target"),
    [
        (
            "dinkster_nodes_image.bounding_boxes",
            "CreateBoundingBoxes",
            "dinkster.image.create_bounding_boxes",
        ),
        (
            "dinkster_nodes_image.layers_from_bounding_boxes",
            "LayersFromBoundingBoxes",
            "dinkster.image.layers_from_bounding_boxes",
        ),
        (
            "dinkster_nodes_image.image_compositor",
            "ImageCompositor",
            "dinkster.image.image_compositor",
        ),
        ("dinkster_nodes_image.webcam", "WebcamCapture", "dinkster.image.webcam_capture"),
        (
            "dinkster_nodes_vision_moge.loader",
            "LoadMoGeModel",
            "dinkster.vision.moge.load_mo_ge_model",
        ),
        ("dinkster_nodes_3d.splat_mesh", "SplatToMesh", "dinkster.3d.splat_to_mesh"),
    ],
)
def test_core_class_pure_rename_preserves_graph_state(
    module: str, source: str, target: str
) -> None:
    schema = getattr(importlib.import_module(module), source).define_schema()
    schema = schema_from_wire(schema_to_wire(schema))
    assert schema.node_type == target
    assert source in schema.search_terms
    assert schema.aliases == ()
    assert len(schema.replacements) == 1
    rule = schema.replacements[0]
    assert rule.from_type == source
    assert rule.migration is None
    assert len(rule.cases) == 1
    case = rule.cases[0]
    assert case.to == target
    assert case.unconditional
    assert dict(case.inputs) == {spec.id: MappingSource.copy(spec.id) for spec in schema.inputs}
    assert dict(case.outputs) == {spec.id: spec.id for spec in schema.outputs}

    inputs: dict[str, object] = {
        spec.id: {"stored": index + 17} for index, spec in enumerate(schema.inputs)
    }
    inputs[schema.inputs[0].id] = Link("upstream", "output")
    old = GraphNode(source, inputs)
    consumer = GraphNode("unrelated.consumer", {"value": Link("ported", schema.outputs[-1].id)})
    graph = Graph({"ported": old, "consumer": consumer})
    schemas = {target: schema}
    migrated = migrate_pure_node_type_replacements(graph, schemas)
    assert migrated.nodes["ported"] == replace(old, node_type=target)
    assert migrated.nodes["consumer"] is consumer
    assert graph.nodes["ported"] is old
    assert old.node_type == source
    assert migrate_pure_node_type_replacements(migrated, schemas) is migrated
