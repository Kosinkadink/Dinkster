"""Public interfaces of image-edit conditioning nodes, without Torch."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_engine import ParentGraphBudget, admit_parent_graph
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire, validate
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.boogu_encoders import TextEncodeBooguEdit
from dinkster_nodes_comfy.joyimage_encoders import TextEncodeJoyImageEdit
from dinkster_nodes_comfy.reference_conditioning import (
    InstructPixToPixConditioning,
    ReferenceLatent,
    ReferenceMultiLatent,
)
from dinkster_schema import elaborate


def test_reference_conditioning_interfaces_and_manifest() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text()
    )
    for node in (InstructPixToPixConditioning, ReferenceLatent, ReferenceMultiLatent):
        assert node in COMFY_NODES
        assert node.__name__ in manifest["pack"]["namespaces"]
    image_edit = InstructPixToPixConditioning.define_schema()
    assert [(port.id, port.type) for port in image_edit.inputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("vae", TypeExpr.concrete(VAE_TYPE)),
        ("pixels", TypeExpr.concrete(IMAGE_TYPE)),
    ]
    assert [(port.id, port.type) for port in image_edit.outputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("latent", TypeExpr.concrete(LATENT_TYPE)),
    ]
    reference = ReferenceLatent.define_schema()
    assert not reference.inputs[1].required
    conditioning = [["embedding", {"reference_latents": ["first"], "strength": 0.7}]]
    assert ReferenceLatent.execute(conditioning=conditioning)["CONDITIONING"] is conditioning


def test_hunyuan_vision_interfaces_are_native() -> None:
    from dinkster_api.v1 import CLIP_VISION_OUTPUT_TYPE
    from dinkster_nodes_comfy.hunyuan3d import HUNYUAN3D_CONDITIONING_NODES

    for node in HUNYUAN3D_CONDITIONING_NODES:
        assert node in COMFY_NODES
        schema = node.schema()
        assert schema.category == "model/conditioning/hunyuan 3d"
        assert [(port.id, port.type) for port in schema.outputs] == [
            ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
            ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ]
        assert all(
            port.type == TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE) for port in schema.inputs
        )
    single, multi = HUNYUAN3D_CONDITIONING_NODES
    assert [(port.id, port.required) for port in single.schema().inputs] == [
        ("clip_vision_output", True)
    ]
    assert [(port.id, port.required) for port in multi.schema().inputs] == [
        ("front", False),
        ("left", False),
        ("back", False),
        ("right", False),
    ]


def test_reference_multi_latent_exposes_resize_and_optional_ordered_images() -> None:
    schema = ReferenceMultiLatent.schema()
    assert [(port.id, port.required) for port in schema.inputs] == [
        ("positive", True),
        ("negative", False),
        ("vae", False),
    ]
    assert [(port.id, port.type) for port in schema.outputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("empty_latent", TypeExpr.concrete(LATENT_TYPE)),
    ]
    combo = schema.combos[0]
    assert combo.id == "resolution" and combo.default is None
    assert [option.key for option in combo.options] == ["keep resolution", "resize"]
    megapixels = combo.options[1].inputs[0]
    assert isinstance(megapixels, InputSpec)
    assert megapixels.id == "megapixels" and megapixels.default == 1.0
    assert isinstance(megapixels.widget, NumberWidget)
    assert megapixels.widget.min == 0.01
    assert megapixels.widget.max == 16.0
    family = schema.input_families[0]
    assert family.id == "images" and not family.required
    assert (family.min_members, family.max_members, family.member_prefix) == (1, 100, "image")
    assert isinstance(family.template[0], InputSpec)
    assert family.template[0].type == TypeExpr.concrete(IMAGE_TYPE)


@pytest.mark.parametrize("connected", [(), (1, 4)])
def test_joy_authored_image_members_are_optional_in_native_graphs(
    connected: tuple[int, ...],
) -> None:
    schema = schema_from_wire(schema_to_wire(TextEncodeJoyImageEdit.schema()))
    family = schema.input_families[0]
    assert (family.min_members, family.max_members, family.required) == (0, 6, False)
    members = tuple(f"images.m{index}" for index in range(6))
    effective = elaborate(schema, ("clip", "prompt", *members))
    source = NodeSchema(
        "test.source",
        outputs=(
            OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
        ),
    )
    sink = NodeSchema(
        "test.sink",
        inputs=(
            InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
            InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ),
    )
    image_links = {members[index]: Link("source", "image") for index in connected}
    graph = Graph(
        {
            "source": GraphNode(source.node_type),
            "positive": GraphNode(
                schema.node_type, {"clip": Link("source", "clip"), "prompt": "edit", **image_links}
            ),
            "negative": GraphNode(
                schema.node_type, {"clip": Link("source", "clip"), "prompt": "avoid", **image_links}
            ),
            "sink": GraphNode(
                sink.node_type,
                {
                    "positive": Link("positive", "CONDITIONING"),
                    "negative": Link("negative", "CONDITIONING"),
                },
            ),
        }
    )
    before = graph_to_wire(graph)
    schemas = {item.node_type: item for item in (source, effective, sink)}
    assert validate(graph, schemas, ("sink",)) == []
    admitted = admit_parent_graph(
        graph,
        graph_to_wire(Graph({})),
        namespace="test.joy",
        origins=(),
        virtual_joins=(),
        budget=ParentGraphBudget(),
        schemas=schemas,
    )
    assert graph_to_wire(admitted.graph) == before
    assert graph_to_wire(graph) == before
    assert [(port.id, port.type, port.required) for port in effective.inputs[3:]] == [
        (member, TypeExpr.concrete(IMAGE_TYPE), False) for member in members
    ]
    missing_clip = Graph(
        {**graph.nodes, "positive": GraphNode(schema.node_type, {"prompt": "edit"})}
    )
    assert [
        (item.code, item.node_id, item.input_id)
        for item in validate(missing_clip, schemas, ("sink",))
    ] == [("missing-input", "positive", "clip")]


@pytest.mark.parametrize("member_count,connected", [(1, ()), (16, (1, 13))])
def test_boogu_authored_image_members_are_optional_in_native_graphs(
    member_count: int, connected: tuple[int, ...]
) -> None:
    schema = schema_from_wire(schema_to_wire(TextEncodeBooguEdit.schema()))
    family = schema.input_families[0]
    assert family.required and family.min_members == 0
    assert family.member_names == tuple(f"image_{index}" for index in range(1, 17))
    members = tuple(f"images.image_{index}" for index in range(1, member_count + 1))
    required = ("clip", "prompt", "negative_prompt", "vae")
    effective = elaborate(schema, (*required, *members))
    absent = elaborate(schema, required)
    assert [port.id for port in absent.inputs] == list(required)
    source = NodeSchema(
        "test.source",
        outputs=(
            OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
        ),
    )
    sink = NodeSchema(
        "test.sink",
        inputs=(
            InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
            InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ),
    )
    inputs = {
        "clip": Link("source", "clip"),
        "vae": Link("source", "vae"),
        "prompt": "edit",
        "negative_prompt": "avoid",
        **{f"images.image_{index}": Link("source", "image") for index in connected},
    }
    graph = Graph(
        {
            "source": GraphNode(source.node_type),
            "encode": GraphNode(schema.node_type, inputs),
            "sink": GraphNode(
                sink.node_type,
                {"positive": Link("encode", "positive"), "negative": Link("encode", "negative")},
            ),
        }
    )
    schemas = {item.node_type: item for item in (source, effective, sink)}
    before = graph_to_wire(graph)
    assert validate(graph, schemas, ("sink",)) == []
    admitted = admit_parent_graph(
        graph,
        graph_to_wire(Graph({})),
        namespace="test.boogu",
        origins=(),
        virtual_joins=(),
        budget=ParentGraphBudget(),
        schemas=schemas,
    )
    assert graph_to_wire(admitted.graph) == graph_to_wire(graph) == before
    assert [(port.id, port.type, port.required) for port in effective.inputs[4:]] == [
        (member, TypeExpr.concrete(IMAGE_TYPE), False) for member in members
    ]
    for missing in ("clip", "vae"):
        incomplete = Graph(
            {
                **graph.nodes,
                "encode": GraphNode(
                    schema.node_type, {k: v for k, v in inputs.items() if k != missing}
                ),
            }
        )
        assert [
            (item.code, item.node_id, item.input_id)
            for item in validate(incomplete, schemas, ("sink",))
        ] == [("missing-input", "encode", missing)]
