from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_image_document.compat import from_comfy_layers, register_comfy_layers
from dinkster_image_document.document import (
    ImageDocument,
    append_raster,
    apply_commands,
    empty_document,
    flatten,
    raster_png,
)
from dinkster_image_document.format import (
    BLEND_MODES,
    IMAGE_DOCUMENT_MEDIA_TYPE,
    INLINE_RESOURCE_LIMIT,
    InvalidDocument,
    affine_from_components,
    canonical_json,
)
from dinkster_image_document.interchange import (
    export_ora,
    export_psd,
    import_ora,
    import_psd,
    load_document,
)
from dinkster_nodes_image import (
    AddLayer,
    CreateLayeredImage,
    EditLayers,
    FlattenLayers,
    LayersFromBoundingBoxes,
    LoadLayers,
    Region,
    SaveLayers,
    SplitLayers,
    register_image_types,
)
from dinkster_nodes_media_io import register_media_types
from dinkster_protocol import InvocationResult
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import (
    TypeRegistry,
    annotate_image,
    annotate_mask,
    media_semantics,
    register_core_types,
)
from dinkster_workers import GroupIsolatedWorker, InProcessWorker
from dinkster_workers.boundary import ValueCodec, decode_result_outputs, encode_result
from PIL import Image


def document() -> ImageDocument:
    return append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5]]], dtype=np.float32)),
    )


class LayerNodeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.layer-node-source",
            outputs=(
                OutputSpec("image", TypeExpr.concrete("dinkster.image")),
                OutputSpec(
                    "mask",
                    TypeExpr.concrete("dinkster.mask"),
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            image=annotate_image(
                np.array([[[[0.5, 0, 0, 0.5]]]], dtype=np.float32),
                alpha="premultiplied",
                color={"primaries": 9, "transfer": 16, "range": 1, "matrix": 9, "bit_depth": 10},
            ),
            mask=annotate_mask(
                np.zeros((1, 1, 1), dtype=np.float32),
                polarity="transparency",
                semantic="alpha",
            ),
        )


def execute_graph_node(node: type[Node], inputs: dict[str, object]) -> dict[str, object]:
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    nodes = [node]
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )

    async def run() -> dict[str, object]:
        graph = Graph(nodes={"node": GraphNode(node.define_schema().node_type, inputs)})
        result = await engine.run(graph, ["node"])
        return {name: value.resolve() for name, value in result.outputs["node"].items()}

    return asyncio.run(run())


def test_split_layer_integer_regions_survive_the_worker_boundary() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    register_image_types(registry)
    direct = cast("list[Region]", SplitLayers.execute(layers=document())["regions"])
    assert [region.to_record() for region in direct] == [{"x": 0, "y": 0, "width": 2, "height": 1}]
    assert all(
        type(coordinate) is int
        for coordinate in (direct[0].x, direct[0].y, direct[0].width, direct[0].height)
    )

    value = registry.wrap("list<dinkster.region>", direct)
    header, blobs, segments = encode_result(
        ValueCodec(registry, use_shm=False, accept_shm=False),
        InvocationResult(outputs={"regions": value}),
        "invocation",
        0.0,
    )
    assert not segments
    outputs, _ = decode_result_outputs(
        ValueCodec(registry, use_shm=False, accept_shm=False), header, blobs, []
    )
    decoded = cast("list[Region]", outputs["regions"].resolve())
    assert [region.to_record() for region in decoded] == [{"x": 0, "y": 0, "width": 2, "height": 1}]
    assert all(
        type(coordinate) is int
        for coordinate in (decoded[0].x, decoded[0].y, decoded[0].width, decoded[0].height)
    )


def test_native_and_layer_wire_share_one_canonical_document() -> None:
    value = document()
    registry = TypeRegistry()
    register_image_types(registry)
    register_comfy_layers(registry)
    for type_id in ("dinkster.layers", "comfy.LAYERS"):
        spec = registry.spec(type_id)
        encoded = spec.encode(value)
        assert encoded == value.data
        assert cast(ImageDocument, spec.decode(encoded)).data == encoded
    assert registry.equivalent_type("comfy.LAYERS") == "dinkster.layers"
    record = value.to_record()
    record["canvas"]["width"] = 100
    assert value.to_record()["canvas"]["width"] == 2


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
def test_document_rendition_needs_no_raster_resources(
    type_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = document().to_record()
    for resource in record["resources"].values():
        resource.pop("inline")
    expected = ImageDocument.from_record(record)

    def forbid_raster(*args, **kwargs):
        pytest.fail("document retrieval must not render or read raster resources")

    monkeypatch.setattr(ImageDocument, "render", forbid_raster)
    monkeypatch.setattr(ImageDocument, "read_resource", forbid_raster)
    registry = TypeRegistry()
    register = register_image_types if type_id == "dinkster.layers" else register_comfy_layers
    register(registry)
    register(registry)
    spec = registry.spec(type_id)
    value = registry.wrap(type_id, spec.decode(spec.encode(expected)))
    rendition = registry.render(value, "document")
    assert rendition.mime == IMAGE_DOCUMENT_MEDIA_TYPE
    assert rendition.data == expected.data
    assert [s.kind for s in registry.renditions_of(type_id) if s.default] == [
        "image" if type_id == "dinkster.layers" else "document"
    ]


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_every_document_blend_renders_deterministically(mode: str) -> None:
    value = append_raster(
        document(), raster_png(np.full((1, 2, 3), 0.25)), blend_mode=mode, opacity=0.4
    )
    first = value.render()
    assert first.png == value.render().png
    assert first.pixels.dtype == np.uint16
    assert np.all(first.pixels[..., :3] <= first.pixels[..., 3:4])


def test_canvas_background_is_the_blend_backdrop() -> None:
    value = append_raster(
        empty_document(1, 1), raster_png(np.full((1, 1, 3), 0.5)), blend_mode="multiply"
    )
    value = apply_commands(
        value, [{"op": "canvas", "changes": {"background": [32768, 32768, 32768, 65535]}}]
    )
    image, mask = flatten(value)
    np.testing.assert_allclose(image, 64 / 255, atol=1 / 255, rtol=0)
    np.testing.assert_array_equal(mask, 0)


def test_node_build_edit_split_selectors_and_api_flatten_match() -> None:
    source = np.array([[[[1, 0, 0], [0, 1, 0]]]], dtype=np.float32)
    expected_value = AddLayer.execute(image=source)["layers"]
    value = execute_graph_node(
        AddLayer, {"image": TypedLiteral("dinkster.image", source.tolist())}
    )["layers"]
    assert isinstance(value, ImageDocument)
    assert isinstance(expected_value, ImageDocument)
    assert value.data == expected_value.data
    record = value.to_record()
    identifier = record["rootLayerIds"][0]
    components = record["layers"][identifier]["transform"]["components"]
    edited = EditLayers.execute(
        layers=value,
        commands=json.dumps(
            [
                {
                    "op": "transform",
                    "id": identifier,
                    "components": {**components, "flipHorizontal": True},
                }
            ]
        ),
    )["layers"]
    assert isinstance(edited, ImageDocument)
    result = FlattenLayers.execute(layers=edited)
    np.testing.assert_array_equal(
        result["image"], np.array([[[[0, 1, 0], [1, 0, 0]]]], dtype=np.float32)
    )
    split_value = append_raster(
        edited, raster_png(np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32))
    )
    split_ids = split_value.to_record()["rootLayerIds"]
    split_value = apply_commands(split_value, [{"op": "group", "ids": split_ids}])
    expected_split = cast(dict[str, Any], SplitLayers.execute(layers=split_value))
    split = cast(
        dict[str, Any],
        execute_graph_node(
            SplitLayers, {"layers": TypedLiteral("dinkster.layers", split_value.to_record())}
        ),
    )
    assert len(split["images"]) == len(split["masks"]) == len(split["regions"]) == 2
    for name in ("images", "masks"):
        for actual, expected in zip(split[name], expected_split[name], strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert split["regions"] == expected_split["regions"]
    np.testing.assert_array_equal(split["images"][0], result["image"])
    np.testing.assert_array_equal(split["masks"][0], result["transparency_mask"])
    assert split["regions"][0].width == 2


def test_edit_layers_node_exposes_structural_and_layer_commands() -> None:
    value = AddLayer.execute(image=np.array([[[[1, 0, 0], [0, 1, 0]]]], dtype=np.float32))["layers"]
    value = AddLayer.execute(
        layers=value,
        image=np.array([[[[0, 0, 1], [0, 0, 1]]]], dtype=np.float32),
        name="Top",
    )["layers"]
    assert isinstance(value, ImageDocument)
    record = value.to_record()
    base, top = record["rootLayerIds"]
    components = {
        **record["layers"][top]["transform"]["components"],
        "x": 1,
        "width": 1,
        "rotation": 0.25,
        "flipHorizontal": True,
        "flipVertical": True,
    }
    commands = [
        {"op": "reorder", "ids": [top, base]},
        {"op": "group", "ids": [top, base], "name": "Node Group"},
        {
            "op": "layer",
            "id": top,
            "changes": {"name": "Edited", "blendMode": "screen", "opacity": 32768},
        },
        {
            "op": "layer",
            "id": base,
            "changes": {"clipping": "clip-to-previous"},
        },
        {"op": "transform", "id": top, "components": components},
    ]
    expected = apply_commands(value, commands)
    edited = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", value.to_record()),
            "commands": json.dumps(commands),
        },
    )["layers"]
    assert isinstance(edited, ImageDocument)
    assert edited.data == expected.data
    record = edited.to_record()
    group_id = record["rootLayerIds"][0]
    assert record["layers"][group_id]["name"] == "Node Group"
    assert record["layers"][group_id]["childLayerIds"] == [top, base]
    assert [record["layers"][identifier]["z_index"] for identifier in (top, base)] == [0, 1]
    assert record["layers"][top]["name"] == "Edited"
    assert record["layers"][top]["blendMode"] == "screen"
    assert record["layers"][top]["opacity"] == 32768
    assert record["layers"][base]["clipping"] == "clip-to-previous"
    transform = record["layers"][top]["transform"]
    assert transform["components"] == components
    assert {key: transform[key] for key in ("a", "b", "c", "d", "tx", "ty")} == (
        affine_from_components(components)
    )
    selected = execute_graph_node(
        FlattenLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "selector": f"layer:{top}",
        },
    )
    expected_image, expected_mask = flatten(edited, f"layer:{top}")
    np.testing.assert_array_equal(selected["image"], expected_image)
    np.testing.assert_array_equal(selected["transparency_mask"], expected_mask)

    remove = [{"op": "remove", "id": group_id}]
    expected_removed = apply_commands(edited, remove)
    removed = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "commands": json.dumps(remove),
        },
    )["layers"]
    assert isinstance(removed, ImageDocument)
    assert removed.data == expected_removed.data
    assert removed.to_record()["layers"] == {}


@pytest.mark.parametrize("combine", ("multiply", "add", "subtract", "intersect"))
@pytest.mark.parametrize("channel", ("alpha", "luminance"))
def test_edit_layers_node_exposes_raster_mask_commands(combine: str, channel: str) -> None:
    value = document()
    record = value.to_record()
    owner = record["rootLayerIds"][0]
    resource = record["layers"][owner]["resourceId"]
    commands = [
        {
            "op": "add_mask",
            "ownerLayerId": owner,
            "mask": {
                "resourceId": resource,
                "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
                "enabled": False,
                "invert": True,
                "opacity": 32768,
                "combineMode": combine,
                "channel": channel,
            },
        }
    ]
    expected = apply_commands(value, commands)
    edited = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", value.to_record()),
            "commands": json.dumps(commands),
        },
    )["layers"]
    assert isinstance(edited, ImageDocument)
    assert edited.data == expected.data
    record = edited.to_record()
    mask_id = record["layers"][owner]["maskIds"][0]
    assert record["masks"][mask_id] == {
        "id": mask_id,
        "kind": "raster",
        "ownerLayerId": owner,
        "resourceId": resource,
        "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
        "enabled": False,
        "invert": True,
        "opacity": 32768,
        "transform": {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000, "tx": 0, "ty": 0},
        "combineMode": combine,
        "channel": channel,
    }
    selected = execute_graph_node(
        FlattenLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "selector": f"mask:{mask_id}",
        },
    )
    expected_image, expected_mask = flatten(edited, f"mask:{mask_id}")
    np.testing.assert_array_equal(selected["image"], expected_image)
    np.testing.assert_array_equal(selected["transparency_mask"], expected_mask)

    components = {
        "x": 1,
        "y": 0,
        "width": 1,
        "height": 1,
        "rotation": 0,
        "flipHorizontal": True,
        "flipVertical": False,
        "sourceWidth": 2,
        "sourceHeight": 1,
    }
    enable = [
        {"op": "mask", "id": mask_id, "changes": {"enabled": True, "invert": False}},
        {"op": "transform", "kind": "mask", "id": mask_id, "components": components},
    ]
    expected_enabled = apply_commands(edited, enable)
    enabled = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "commands": json.dumps(enable),
        },
    )["layers"]
    assert isinstance(enabled, ImageDocument)
    assert enabled.data == expected_enabled.data
    assert enabled.to_record()["masks"][mask_id]["enabled"] is True
    assert enabled.to_record()["masks"][mask_id]["invert"] is False
    assert enabled.to_record()["masks"][mask_id]["transform"]["components"] == components
    remove = [{"op": "remove_mask", "id": mask_id}]
    expected_removed = apply_commands(enabled, remove)
    removed = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", enabled.to_record()),
            "commands": json.dumps(remove),
        },
    )["layers"]
    assert isinstance(removed, ImageDocument)
    assert removed.data == expected_removed.data
    assert removed.to_record()["masks"] == {}
    assert removed.to_record()["layers"][owner]["maskIds"] == []


def test_layer_creation_normalizes_premultiplied_pixels_before_png_storage() -> None:
    source = LayerNodeSource.execute()
    expected = AddLayer.execute(image=source["image"], mask=source["mask"])["layers"]
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    nodes: list[type[Node]] = [LayerNodeSource, AddLayer]
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )

    async def run() -> ImageDocument:
        result = await engine.run(
            Graph(
                nodes={
                    "source": GraphNode("test.layer-node-source"),
                    "add": GraphNode(
                        "dinkster.layers.add",
                        {"image": Link("source", "image"), "mask": Link("source", "mask")},
                    ),
                }
            ),
            ["add"],
        )
        value = result.outputs["add"]["layers"].resolve()
        assert isinstance(value, ImageDocument)
        return value

    layered = asyncio.run(run())
    assert isinstance(expected, ImageDocument)
    assert layered.data == expected.data
    record = layered.to_record()
    assert len(record["masks"]) == 1
    outputs = FlattenLayers.execute(layers=layered)
    assert media_semantics(outputs["image"])["color"] == {
        "primaries": 9,
        "transfer": 16,
        "range": 1,
        "matrix": 9,
        "bit_depth": 10,
    }
    assert media_semantics(outputs["transparency_mask"]) == {
        "polarity": "transparency",
        "semantic": "alpha",
    }
    actual = np.asarray(outputs["image"])
    np.testing.assert_array_equal(actual[..., :3], [[[[1, 0, 0]]]])
    np.testing.assert_allclose(actual[..., 3], 0.5, atol=1 / 255, rtol=0)


@pytest.mark.parametrize("combine", ("multiply", "add", "subtract", "intersect"))
@pytest.mark.parametrize("channel", ("alpha", "luminance"))
def test_mask_fields_groups_clipping_and_z_order(combine: str, channel: str) -> None:
    value = document()
    record = value.to_record()
    identifier = record["rootLayerIds"][0]
    resource = record["layers"][identifier]["resourceId"]
    mask = {
        "resourceId": resource,
        "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
        "channel": channel,
        "combineMode": combine,
        "enabled": True,
        "invert": True,
        "opacity": 32768,
    }
    masked = apply_commands(
        value,
        [
            {"op": "add_mask", "ownerLayerId": identifier, "mask": mask},
            {"op": "add_mask", "ownerLayerId": identifier, "mask": mask},
        ],
    )
    assert masked.render().png == masked.render().png
    mask_id = next(iter(masked.to_record()["masks"]))
    assert flatten(masked, f"mask:{mask_id}")[0].shape == (1, 1, 2, 3)
    disabled = apply_commands(
        masked,
        [
            {"op": "mask", "id": mid, "changes": {"enabled": False}}
            for mid in masked.to_record()["masks"]
        ],
    )
    assert disabled.render().png == value.render().png
    group = apply_commands(disabled, [{"op": "group", "ids": [identifier]}])
    assert group.render().png == value.render().png
    removed = apply_commands(group, [{"op": "remove", "id": group.to_record()["rootLayerIds"][0]}])
    assert not removed.to_record()["layers"] and not removed.to_record()["masks"]


def test_transform_components_cannot_compete_with_affine() -> None:
    record = document().to_record()
    layer = record["layers"][record["rootLayerIds"][0]]
    layer["transform"]["components"]["x"] = 2
    with pytest.raises(InvalidDocument, match="disagree"):
        ImageDocument.from_record(record)


def test_delta_codecs_bind_edits_and_reject_stale_documents() -> None:
    from dinkster_assets import digest_bytes
    from dinkster_nodes_image.layer_recipe import replay_compositor

    value = document()
    identifier = value.to_record()["rootLayerIds"][0]
    delta = {
        "version": 2,
        "documentDigest": digest_bytes(value.data),
        "commands": [{"op": "layer", "id": identifier, "changes": {"visible": False}}],
    }
    registry = TypeRegistry()
    register_image_types(registry)
    for type_id in ("dinkster.compositor", "comfy.COMPOSITOR"):
        spec = registry.spec(type_id)
        decoded = spec.decode(spec.encode(delta))
        assert decoded == delta
        edited, stale = replay_compositor(value, decoded, value)
        assert not stale
        assert np.all(flatten(edited)[1] == 1)
        unchanged, stale = replay_compositor(edited, decoded, edited)
        assert stale and unchanged.data == edited.data


@pytest.mark.parametrize("color_space", ["perceptual", "linear"])
def test_compositor_preview_digest_binds_repeated_apply_to_input(monkeypatch, color_space) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_nodes_image import compositor as module

    value = document()
    original = value.data
    identifier = value.to_record()["rootLayerIds"][0]
    events = []
    monkeypatch.setattr(module, "report_event", lambda name, data: events.append((name, data)))
    delta = {"version": 2, "documentDigest": None, "commands": []}
    for opacity in (65535, 32768, 16384):
        delta["commands"] = [{"op": "layer", "id": identifier, "changes": {"opacity": opacity}}]
        CreateLayeredImage.execute(layers=value, compositor=delta, color_space=color_space)
        name, state = events[-1]
        assert name == "dinkster.compositor.state"
        assert state["stale"] is False
        assert state["documentDigest"] == digest_bytes(original)
        assert state["document"]["layers"][identifier]["opacity"] == opacity
        assert state["document"]["canvas"]["compositing"] == (
            "linear-premultiplied-alpha" if color_space == "linear" else "premultiplied-alpha"
        )
        delta["documentDigest"] = state["documentDigest"]
    assert value.data == original

    changed = apply_commands(value, [{"op": "layer", "id": identifier, "changes": {"name": "New"}}])
    CreateLayeredImage.execute(layers=changed, compositor=delta, color_space=color_space)
    state = events[-1][1]
    assert state["stale"] is True
    assert state["documentDigest"] == digest_bytes(changed.data)
    assert state["document"]["layers"][identifier]["opacity"] == 65535


def test_clipping_and_grouping_use_z_order_not_object_key_order() -> None:
    value = append_raster(document(), raster_png(np.ones((1, 2, 3))), z_index=10)
    record = value.to_record()
    base, clipped = record["rootLayerIds"]
    record["layers"][base]["z_index"] = -10
    record["layers"][clipped]["clipping"] = "clip-to-previous"
    record["rootLayerIds"] = [clipped, base]
    value = ImageDocument.from_record(record)
    image, _ = flatten(value)
    assert image.shape[-1] == 4
    assert image[0, 0, 1, 3] < 1
    record["layers"][clipped]["clipping"] = "none"
    value = ImageDocument.from_record(record)
    grouped = apply_commands(value, [{"op": "group", "ids": [clipped]}])
    assert grouped.render().png == value.render().png
    split = cast(dict[str, Any], SplitLayers.execute(layers=grouped))
    np.testing.assert_array_equal(split["images"][0], flatten(value, f"layer:{base}")[0])


def test_comfy_missing_canvas_and_zero_dimensions_use_source_size() -> None:
    value = from_comfy_layers(
        {"version": 1, "layers": [{"image": np.ones((1, 2, 3, 3)).tolist(), "w": 0, "h": 0}]}
    )
    assert flatten(value)[0].shape == (1, 2, 3, 3)
    assert flatten(from_comfy_layers({"version": 1, "layers": []}))[0].shape == (1, 1, 1, 4)


def large_document() -> ImageDocument:
    value = empty_document(4096, 4096)
    for index in range(50):
        output = io.BytesIO()
        Image.new("RGB", (4096, 4096), (index, 34, 56)).save(output, "PNG")
        raster = output.getvalue()
        assert len(raster) > INLINE_RESOURCE_LIMIT
        value = append_raster(value, raster, name=f"Layer {index}")
    registry = TypeRegistry()
    register_image_types(registry)
    spec = registry.spec("dinkster.layers")
    assert spec.meta is not None
    meta = spec.meta(value)
    size = len(spec.encode(value)) + len(canonical_json(meta).encode())
    assert size < 1024 * 1024
    assert all("inline" not in resource for resource in value.to_record()["resources"].values())
    assert b"/tmp" not in value.data
    assert len(value.to_record()["resources"]) == 50
    print(f"50x4096x4096 document plus metadata: {size} bytes")
    return value


def test_ora_archive_exact_reimport_and_independent_stack_fixture() -> None:
    value = document()
    encoded = export_ora(value)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("mergedimage.png") == value.render().png
    imported = import_ora(encoded)
    assert imported.data == value.data
    assert imported.render().png == value.render().png
    fixture = io.BytesIO()
    pixels = raster_png(np.array([[[0.25, 0.5, 1, 1]]], dtype=np.float32))
    with zipfile.ZipFile(fixture, "w") as archive:
        archive.writestr("mimetype", "image/openraster")
        archive.writestr(
            "stack.xml",
            '<image w="2" h="1"><stack><layer name="Blue" src="data/a.png" x="1" y="0" '
            'opacity="1" composite-op="svg:src-over"/></stack></image>',
        )
        archive.writestr("data/a.png", pixels)
    actual = import_ora(fixture.getvalue())
    np.testing.assert_array_equal(
        flatten(actual)[0],
        np.array([[[[0, 0, 0, 0], [64 / 255, 128 / 255, 1, 1]]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("opacity", "expected_8_bit", "expected_decimal"),
    [
        (0, 0, "0.0"),
        (1, 0, "0.0"),
        (129, 1, "0.003921568627450981"),
        (32_768, 128, "0.5019607843137256"),
        (49_151, 191, "0.7490196078431374"),
        (65_535, 255, "1.0"),
    ],
)
def test_ora_export_rounds_opacity_for_krita_without_reducing_native_precision(
    opacity: int, expected_8_bit: int, expected_decimal: str
) -> None:
    pixels = raster_png(np.ones((1, 1, 4), dtype=np.float32))
    value = append_raster(empty_document(1, 1), pixels, opacity=opacity / 65_535)

    encoded = export_ora(value)

    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        emitted = list(stack)[0].attrib["opacity"]
    assert emitted == expected_decimal
    assert int(float(emitted) * 255) == expected_8_bit
    imported = import_ora(encoded)
    assert imported.data == value.data
    identifier = imported.to_record()["rootLayerIds"][0]
    assert imported.to_record()["layers"][identifier]["opacity"] == opacity


def test_ora_export_preserves_portable_layer_properties() -> None:
    def solid(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
        pixels = np.empty((height, width, 4), dtype=np.float32)
        pixels[:] = np.asarray(rgba, dtype=np.float32) / 255
        return raster_png(pixels)

    value = empty_document(8, 6)
    value = append_raster(value, solid(8, 6, (32, 64, 192, 64)), name="Base")
    value = append_raster(
        value,
        solid(4, 4, (224, 32, 16, 128)),
        name="Offset",
        x=-2,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        solid(2, 2, (16, 240, 64, 255)),
        name="Hidden",
        x=4,
        opacity=0.75,
        blend_mode="screen",
        visible=False,
    )
    value = append_raster(value, solid(1, 1, (0, 0, 0, 0)), name="Empty", x=6, y=5)
    encoded = export_ora(value)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert "Thumbnails/thumbnail.png" in archive.namelist()
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("name") for layer in layers] == ["Empty", "Hidden", "Offset", "Base"]
        opacity_values = [layer.get("opacity") for layer in layers]
        assert opacity_values == [
            "1.0",
            "0.7490196078431374",
            "0.5019607843137256",
            "1.0",
        ]
        # Krita converts the ORA decimal times 255 directly to an unsigned byte.
        assert [int(float(opacity) * 255) for opacity in opacity_values if opacity is not None] == [
            255,
            191,
            128,
            255,
        ]
        assert [
            {
                "visibility": layer.get("visibility"),
                "opacity": round(float(layer.get("opacity", "0")) * 65_535),
                "x": int(layer.get("x", "0")),
                "y": int(layer.get("y", "0")),
                "composite-op": layer.get("composite-op"),
            }
            for layer in layers
        ] == [
            {
                "visibility": "visible",
                "opacity": 65_535,
                "x": 6,
                "y": 5,
                "composite-op": "svg:src-over",
            },
            {
                "visibility": "hidden",
                "opacity": 49_087,
                "x": 4,
                "y": 0,
                "composite-op": "svg:screen",
            },
            {
                "visibility": "visible",
                "opacity": 32_896,
                "x": -2,
                "y": 1,
                "composite-op": "svg:multiply",
            },
            {
                "visibility": "visible",
                "opacity": 65_535,
                "x": 0,
                "y": 0,
                "composite-op": "svg:src-over",
            },
        ]
        with Image.open(io.BytesIO(archive.read(cast(str, layers[1].get("src"))))) as hidden:
            assert np.asarray(hidden.convert("RGBA"))[0, 0, 3] == 255
        standard = io.BytesIO()
        with zipfile.ZipFile(standard, "w") as stripped:
            for entry in archive.infolist():
                if not entry.filename.startswith("dinkster/"):
                    stripped.writestr(entry, archive.read(entry.filename))
    imported = import_ora(standard.getvalue())
    imported_record = imported.to_record()
    imported_layers = [
        imported_record["layers"][identifier]
        for identifier in reversed(imported_record["rootLayerIds"])
    ]
    assert [layer["name"] for layer in imported_layers] == ["Empty", "Hidden", "Offset", "Base"]
    assert [layer["visible"] for layer in imported_layers] == [True, False, True, True]
    assert [layer["opacity"] for layer in imported_layers] == [65_535, 49_087, 32_896, 65_535]
    assert [layer["transform"]["tx"] for layer in imported_layers] == [
        6_000_000,
        4_000_000,
        -2_000_000,
        0,
    ]
    assert [layer["transform"]["ty"] for layer in imported_layers] == [
        5_000_000,
        0,
        1_000_000,
        0,
    ]
    assert imported.render().png == value.render().png


def test_ora_export_bakes_only_nonportable_raster_geometry_and_masks() -> None:
    pixels = raster_png(np.ones((2, 2, 4), dtype=np.float32))
    mask = raster_png(np.zeros((2, 2), dtype=np.float32))
    value = append_raster(
        empty_document(4, 4),
        pixels,
        mask_data=mask,
        name="Baked",
        x=0.5,
        y=1,
        rotation=0.25,
        opacity=0.5,
        visible=False,
    )
    with zipfile.ZipFile(io.BytesIO(export_ora(value))) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layer = list(stack)[0]
        assert layer.attrib | {"src": "ignored"} == {
            "name": "Baked",
            "src": "ignored",
            "opacity": "0.5019607843137256",
            "visibility": "hidden",
            "x": "0",
            "y": "0",
            "composite-op": "svg:src-over",
        }
        with Image.open(io.BytesIO(archive.read(layer.attrib["src"]))) as baked:
            assert baked.size == (4, 4)
            assert np.max(np.asarray(baked)[..., 3]) > 0


def test_ora_export_uses_composite_for_empty_and_nonbaseline_stacks() -> None:
    empty = empty_document(4, 3)
    with zipfile.ZipFile(io.BytesIO(export_ora(empty))) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("name") for layer in layers] == ["Composite"]
        assert layers[0].get("src") == "data/composite.png"
        with Image.open(io.BytesIO(archive.read("data/composite.png"))) as composite:
            assert np.max(np.asarray(composite.convert("RGBA"))) == 0
    assert import_ora(export_ora(empty)).data == empty.data

    exclusion = append_raster(
        empty_document(1, 1),
        raster_png(np.ones((1, 1, 4), dtype=np.float32)),
        blend_mode="exclusion",
    )
    with pytest.warns(UserWarning, match="merged standard view"):
        encoded = export_ora(exclusion)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("composite-op") for layer in layers] == ["svg:src-over"]
        assert [name for name in archive.namelist() if name.startswith("data/")] == [
            "data/composite.png"
        ]


def test_seedream_style_document_uses_transparency_masks_and_z_index() -> None:
    background = np.zeros((1, 1, 2, 3), dtype=np.float32)
    foreground = np.ones((1, 1, 1, 3), dtype=np.float32)
    value = from_comfy_layers(
        {
            "version": 1,
            "canvas": (2, 1),
            "layers": [
                {
                    "type": "raster",
                    "image": foreground,
                    "mask": np.zeros((1, 1, 1)),
                    "x": 1,
                    "z_index": 10,
                },
                {"type": "raster", "image": background, "z_index": 0},
            ],
        }
    )
    np.testing.assert_array_equal(
        flatten(value)[0], np.array([[[[0, 0, 0], [1, 1, 1]]]], dtype=np.float32)
    )


def test_layer_aliases_execute() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/layer_document_goldens.json").read_text()
    )
    for case in fixture["cases"]:
        value = from_comfy_layers(case["layers"])
        assert value.to_record()["extensions"]["comfy"]["inputs"] == case["inputs"]
        result = cast(dict[str, Any], CreateLayeredImage.execute(layers=value, compositor={}))
        np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
        np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)
        background, subject = case["layers"]["layers"]
        built = AddLayer.execute(image=np.asarray(background["image"], dtype=np.float32))["layers"]
        built = LayersFromBoundingBoxes.execute(
            layers=built,
            image=np.asarray(subject["image"], dtype=np.float32),
            mask=np.asarray(subject["mask"], dtype=np.float32),
            bboxes=json.dumps([{"x": 1, "y": 1, "width": 2, "height": 2}]),
        )["layers"]
        assert isinstance(built, ImageDocument)
        identifier = built.to_record()["rootLayerIds"][-1]
        built = apply_commands(
            built,
            [
                {
                    "op": "layer",
                    "id": identifier,
                    "changes": {"opacity": round(subject["opacity"] * 65535)},
                }
            ],
        )
        result = cast(
            dict[str, Any],
            CreateLayeredImage.execute(layers=built, compositor={}, color_space="linear"),
        )
        np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
        np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)


@pytest.mark.parametrize("case_index", range(3))
def test_layer_node_source_goldens(case_index: int) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/layer_document_goldens.json").read_text()
    )
    case = fixture["aliasCases"][case_index]
    inputs = case["inputs"]
    inputs["layers"] = from_comfy_layers(inputs["layers"])
    for key in ("image", "mask"):
        if key in inputs:
            inputs[key] = np.asarray(inputs[key], dtype=np.float32)
    if case["node"] == "AddLayer":
        inputs["flip_horizontal"] = inputs.pop("flip_h")
        value = AddLayer.execute(**inputs)["layers"]
    elif case["node"] == "LayersFromBoundingBoxes":
        value = LayersFromBoundingBoxes.execute(**inputs)["layers"]
        assert isinstance(value, ImageDocument)
        record = value.to_record()
        assert record["layers"][record["rootLayerIds"][-1]]["z_index"] == 11
    else:
        value = inputs["layers"]
    result = cast(dict[str, Any], CreateLayeredImage.execute(layers=value, color_space="linear"))
    np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
    np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)


def test_psd_real_raster_fixture() -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGB", (2, 1))
    PixelLayer.frompil(Image.new("RGBA", (2, 1), (255, 0, 0, 255)), psd, name="Red")
    PixelLayer.frompil(Image.new("RGBA", (1, 1), (0, 255, 0, 255)), psd, name="Green", left=1)
    output = io.BytesIO()
    psd.save(output)
    loaded = import_psd(output.getvalue())
    assert [layer["name"] for layer in loaded.to_record()["layers"].values()] == ["Red", "Green"]
    np.testing.assert_array_equal(flatten(loaded)[0], np.array([[[[1, 0, 0], [0, 1, 0]]]]))


def test_psd_import_removes_krita_unicode_name_terminators() -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer

    psd = PSDImage.new("RGB", (2, 1))
    group = Group.new(psd, name="Group\0")
    PixelLayer.frompil(
        Image.new("RGBA", (2, 1), (255, 0, 0, 255)), group, name="Raster \u03a9 name\0"
    )
    output = io.BytesIO()
    psd.save(output)

    imported = import_psd(output.getvalue())
    record = imported.to_record()
    group_record = record["layers"][record["rootLayerIds"][0]]
    raster_record = record["layers"][group_record["childLayerIds"][0]]
    assert group_record["name"] == "Group"
    assert raster_record["name"] == "Raster \u03a9 name"
    np.testing.assert_array_equal(flatten(imported)[0], np.array([[[[1, 0, 0], [1, 0, 0]]]]))

    reexported = PSDImage.open(io.BytesIO(export_psd(imported)))
    assert [layer.name for layer in reexported] == ["Group"]
    assert [layer.name for layer in cast(Any, reexported[0])] == ["Raster \u03a9 name"]


def _psd_acceptance_document() -> ImageDocument:
    value = append_raster(
        empty_document(6, 4),
        raster_png(np.full((4, 6, 4), (32 / 255, 64 / 255, 1, 1), dtype=np.float32)),
        name="Backdrop",
    )
    value = append_raster(
        value,
        raster_png(np.full((2, 3, 4), (1, 32 / 255, 0, 191 / 255), dtype=np.float32)),
        mask_data=raster_png(np.zeros((2, 3), dtype=np.float32)),
        name="Masked offset",
        x=1,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 2, 4), (0, 1, 0, 1), dtype=np.float32)),
        name="Hidden",
        x=4,
        visible=False,
        blend_mode="screen",
    )
    record = value.to_record()
    masked, hidden = record["rootLayerIds"][-2:]
    value = apply_commands(
        value,
        [{"op": "group", "ids": [masked, hidden], "name": "Edits"}],
    )
    group = value.to_record()["rootLayerIds"][-1]
    return apply_commands(
        value,
        [{"op": "layer", "id": group, "changes": {"isolation": "pass-through"}}],
    )


def test_psd_export_round_trips_standard_layer_properties_and_pixels() -> None:
    from psd_tools import PSDImage

    value = append_raster(
        empty_document(4, 2),
        raster_png(np.full((2, 3, 3), (0, 0, 1), dtype=np.float32)),
        name="Backdrop",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 2, 4), (1, 0, 0, 0.5), dtype=np.float32)),
        mask_data=raster_png(np.array([[0, 0.5]], dtype=np.float32)),
        name="Masked offset",
        x=-1,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 1, 3), (0, 1, 0), dtype=np.float32)),
        name="Hidden clipped",
        x=2,
        visible=False,
        blend_mode="screen",
    )
    record = value.to_record()
    offset, hidden = record["rootLayerIds"][-2:]
    value = apply_commands(value, [{"op": "group", "ids": [offset, hidden], "name": "Edits"}])
    group = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [
            {
                "op": "layer",
                "id": group,
                "changes": {"isolation": "pass-through", "opacity": 49_151},
            },
            {
                "op": "layer",
                "id": hidden,
                "changes": {"clipping": "clip-to-previous"},
            },
        ],
    )

    encoded = export_psd(value)
    psd = PSDImage.open(io.BytesIO(encoded))
    with (
        Image.open(io.BytesIO(encoded)) as merged,
        Image.open(io.BytesIO(value.render().png)) as expected_merged,
    ):
        assert merged.format == "PSD"
        np.testing.assert_allclose(
            np.asarray(merged.convert("RGBA")),
            np.asarray(expected_merged.convert("RGBA")),
            atol=1,
            rtol=0,
        )
    assert [layer.name for layer in psd] == ["Backdrop", "Edits"]
    assert [layer.name for layer in cast(Any, psd[1])] == ["Masked offset", "Hidden clipped"]

    imported = import_psd(encoded)
    imported_record = imported.to_record()
    backdrop_id, group_id = imported_record["rootLayerIds"]
    imported_group = imported_record["layers"][group_id]
    masked_id, hidden_id = imported_group["childLayerIds"]
    masked = imported_record["layers"][masked_id]
    hidden_layer = imported_record["layers"][hidden_id]
    assert imported_record["layers"][backdrop_id]["name"] == "Backdrop"
    assert imported_group["name"] == "Edits"
    assert imported_group["isolation"] == "pass-through"
    assert imported_group["opacity"] == round(round(49_151 / 65_535 * 255) / 255 * 65_535)
    assert masked["name"] == "Masked offset"
    assert masked["transform"]["tx"] == -1_000_000
    assert masked["transform"]["ty"] == 1_000_000
    assert masked["blendMode"] == "multiply"
    assert masked["opacity"] == round(round(32_768 / 65_535 * 255) / 255 * 65_535)
    assert len(masked["maskIds"]) == 1
    assert hidden_layer["visible"] is False
    assert hidden_layer["blendMode"] == "screen"
    assert hidden_layer["clipping"] == "clip-to-previous"
    np.testing.assert_allclose(flatten(imported)[0], flatten(value)[0], atol=1 / 255, rtol=0)
    reexported = PSDImage.open(io.BytesIO(export_psd(imported)))
    assert [layer.name for layer in reexported] == ["Backdrop", "Edits"]


def test_psd_export_matches_fixed_editor_acceptance_contract() -> None:
    from psd_tools import PSDImage

    encoded = export_psd(_psd_acceptance_document())
    with Image.open(io.BytesIO(encoded)) as merged:
        pixels = merged.convert("RGBA").tobytes()
    assert len(pixels) == 96
    assert hashlib.sha256(pixels).hexdigest() == (
        "06590d2ccfd36d2a4b5337cde8755f7bd5332a37ce826c1a44ee21d68d029900"
    )
    psd = PSDImage.open(io.BytesIO(encoded))
    assert (psd.width, psd.height) == (6, 4)
    assert [layer.name for layer in psd] == ["Backdrop", "Edits"]
    group = cast(Any, psd[1])
    assert group.blend_mode.name == "PASS_THROUGH"
    assert [layer.name for layer in group] == ["Masked offset", "Hidden"]
    assert (group[0].left, group[0].top, group[0].opacity) == (1, 1, 128)
    assert group[0].mask is not None and not group[0].mask.disabled
    assert (group[1].left, group[1].top, group[1].visible) == (4, 0, False)


def test_psd_export_round_trips_disabled_mask_state() -> None:
    value = _psd_acceptance_document()
    record = value.to_record()
    group = record["layers"][record["rootLayerIds"][-1]]
    masked = record["layers"][group["childLayerIds"][0]]
    mask_id = masked["maskIds"][0]
    value = apply_commands(value, [{"op": "mask", "id": mask_id, "changes": {"enabled": False}}])

    imported = import_psd(export_psd(value))
    imported_record = imported.to_record()
    imported_group = imported_record["layers"][imported_record["rootLayerIds"][-1]]
    imported_masked = imported_record["layers"][imported_group["childLayerIds"][0]]
    imported_mask = imported_record["masks"][imported_masked["maskIds"][0]]
    assert imported_mask["enabled"] is False
    np.testing.assert_array_equal(flatten(imported)[0], flatten(value)[0])


@pytest.mark.parametrize("mode", [mode for mode in BLEND_MODES if not mode.startswith("grain_")])
def test_psd_export_preserves_every_representable_blend_mode(mode: str) -> None:
    value = append_raster(
        empty_document(1, 1), raster_png(np.ones((1, 1, 4), dtype=np.float32)), blend_mode=mode
    )
    imported = import_psd(export_psd(value))
    layer = imported.to_record()["layers"][imported.to_record()["rootLayerIds"][0]]
    assert layer["blendMode"] == mode


def test_psd_export_uses_merged_view_for_unsupported_structure() -> None:
    from psd_tools import PSDImage

    value = append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 0.5], [0, 1, 0, 1]]], dtype=np.float32)),
        blend_mode="grain_extract",
    )
    with pytest.warns(UserWarning, match="merged standard view"):
        encoded = export_psd(value)
    psd = PSDImage.open(io.BytesIO(encoded))
    assert [layer.name for layer in psd] == ["Composite"]
    np.testing.assert_array_equal(flatten(import_psd(encoded))[0], flatten(value)[0])

    value = append_raster(document(), raster_png(np.ones((1, 2, 4), dtype=np.float32)))
    top = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [top]}])
    group = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [{"op": "layer", "id": group, "changes": {"clipping": "clip-to-previous"}}],
    )
    with pytest.warns(UserWarning, match="group geometry, clipping, or mask"):
        encoded = export_psd(value)
    assert [layer.name for layer in PSDImage.open(io.BytesIO(encoded))] == ["Composite"]


@pytest.mark.parametrize("depth", [16, 32])
def test_psd_import_reports_rasterized_capability_gaps(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    from psd_tools import PSDImage
    from psd_tools.constants import ColorMode

    class Layer:
        def __init__(self, kind: str, name: str, vector_mask: bool) -> None:
            self.kind = kind
            self.name = name
            self.left = 0
            self.top = 0
            self.width = 1
            self.height = 1
            self.opacity = 255
            self.visible = True
            self.clipping = False
            self.blend_mode = SimpleNamespace(name="NORMAL")
            self.mask = None
            self._vector_mask = vector_mask

        def has_effects(self) -> bool:
            return False

        def has_vector_mask(self) -> bool:
            return self._vector_mask

        def is_group(self) -> bool:
            return False

        def topil(self) -> Image.Image:
            return Image.new("RGBA", (1, 1), (255, 0, 0, 255))

    class Document(list[Layer]):
        width = 1
        height = 1
        color_mode = ColorMode.CMYK

        def __init__(self, layers: list[Layer], depth: int) -> None:
            super().__init__(layers)
            self.depth = depth

    monkeypatch.setattr(
        PSDImage,
        "open",
        lambda _: Document(
            [Layer("type", "Caption", True), Layer("smartobject", "Placed", False)], depth
        ),
    )
    with pytest.warns(UserWarning) as caught:
        imported = import_psd(b"8BPS")
    messages = [str(item.message) for item in caught]
    assert any(f"{depth}-bit channels are rasterized to 8-bit RGBA" in item for item in messages)
    assert any("CMYK color is rasterized to RGBA" in item for item in messages)
    assert any(
        "type layer 'Caption' is rasterized as an ordinary layer" in item for item in messages
    )
    assert any(
        "smartobject layer 'Placed' is rasterized as an ordinary layer" in item for item in messages
    )
    assert any("layer 'Caption' vector mask is not preserved" in item for item in messages)
    assert len(imported.to_record()["rootLayerIds"]) == 2


@pytest.mark.parametrize("background", [0, 255])
def test_psd_mask_preserves_outside_bounds_background(background: int) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGB", (3, 1))
    layer = PixelLayer.frompil(Image.new("RGB", (3, 1), (255, 0, 0)), psd, name="Red")
    if layer.has_mask():
        layer.remove_mask()
    layer.create_mask(Image.new("L", (1, 1), 255), left=1, top=0)
    cast(Any, layer)._record.mask_data.background_color = background
    output = io.BytesIO()
    psd.save(output)
    rendered = np.asarray(Image.open(io.BytesIO(import_psd(output.getvalue()).render().png)))
    np.testing.assert_array_equal(rendered[..., 3], [[background, 255, background]])
    np.testing.assert_array_equal(
        rendered[..., 0], [[255 if background else 0, 255, 255 if background else 0]]
    )


def test_comfy_layer_coercion_preserves_premultiplied_color() -> None:
    color = {"primaries": 9, "transfer": 16, "range": 2}
    pixels = annotate_image(
        np.array([[[[0.5, 0, 0, 0.5]]]], np.float32), alpha="premultiplied", color=color
    )
    value = from_comfy_layers({"version": 1, "canvas": (1, 1), "layers": [{"image": pixels}]})
    actual, mask = flatten(value)
    np.testing.assert_array_equal(actual, [[[[1, 0, 0, np.float32(128 / 255)]]]])
    assert media_semantics(actual)["color"] == color
    np.testing.assert_array_equal(mask, 1 - actual[..., 3])


@pytest.mark.parametrize("opacity", [0, 128, 255])
@pytest.mark.parametrize("mask", [None, 0, 128, 255])
def test_psd_pass_through_group_preserves_backdrop_blending(opacity: int, mask: int | None) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.composite import composite
    from psd_tools.constants import BlendMode

    psd = PSDImage.new("RGB", (3, 1))
    PixelLayer.frompil(Image.new("RGB", (3, 1), (0, 0, 255)), psd, name="Blue")
    group = Group.new(psd, name="Pass through")
    group.blend_mode = BlendMode.PASS_THROUGH
    group.opacity = opacity
    layer = PixelLayer.frompil(Image.new("RGB", (1, 1), (255, 0, 0)), group, name="Multiply red")
    layer.blend_mode = BlendMode.MULTIPLY
    if mask is not None:
        group.create_mask(Image.new("L", (1, 1), mask))
    output = io.BytesIO()
    psd.save(output)
    color, _, alpha = composite(PSDImage.open(io.BytesIO(output.getvalue())), force=True)
    # Round reference float pixels; its PIL path truncates opaque 0.99999994 alpha to 254.
    reference = np.rint(np.concatenate((color, alpha), axis=-1) * 255).astype(np.uint8)
    value = import_psd(output.getvalue())
    actual = Image.open(io.BytesIO(value.render().png))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    identifier = value.to_record()["rootLayerIds"][-1]
    assert value.to_record()["layers"][identifier]["isolation"] == "pass-through"
    moved = apply_commands(
        value,
        [
            {
                "op": "layer",
                "id": identifier,
                "changes": {
                    "transform": {
                        "a": 1_000_000,
                        "b": 0,
                        "c": 0,
                        "d": 1_000_000,
                        "tx": 1_000_000,
                        "ty": 0,
                    }
                },
            }
        ],
    )
    shifted = np.asarray(Image.open(io.BytesIO(moved.render().png)))
    np.testing.assert_array_equal(shifted[0, 1], np.asarray(reference)[0, 0])
    np.testing.assert_array_equal(shifted[0, [0, 2]], [[0, 0, 255, 255]] * 2)
    with pytest.warns(UserWarning, match="merged standard view"):
        assert import_ora(export_ora(moved)).data == moved.data


@pytest.mark.parametrize("backdrop_alpha", [0, 128, 255])
@pytest.mark.parametrize("child_alpha", [128, 255])
@pytest.mark.parametrize("opacity", [0, 32768, 65535])
def test_linear_pass_through_normal_child_matches_isolated_group(
    backdrop_alpha: int, child_alpha: int, opacity: int
) -> None:
    value = append_raster(
        empty_document(1, 1),
        raster_png(np.array([[[0, 0, 1, backdrop_alpha / 255]]], np.float32)),
    )
    value = append_raster(value, raster_png(np.array([[[1, 0, 0, child_alpha / 255]]], np.float32)))
    leaf = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [leaf]}])
    record = value.to_record()
    record["canvas"]["compositing"] = "linear-premultiplied-alpha"
    group = record["layers"][record["rootLayerIds"][-1]]
    group["opacity"] = opacity
    isolated = ImageDocument.from_record(record).render()
    group["isolation"] = "pass-through"
    passed = ImageDocument.from_record(record).render()
    reference = np.asarray(Image.open(io.BytesIO(isolated.png)))
    actual = np.asarray(Image.open(io.BytesIO(passed.png)))
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(passed.pixels[..., 3], isolated.pixels[..., 3])
    if backdrop_alpha == child_alpha == 255 and opacity == 32768:
        np.testing.assert_array_equal(actual, [[[188, 0, 188, 255]]])


@pytest.mark.parametrize("mask_opacity", [32768, 65535])
@pytest.mark.parametrize("translation", [0, 1])
@pytest.mark.parametrize("clipping", ["none", "clip-to-previous"])
def test_linear_pass_through_group_preserves_mask_opacity_transform_and_clipping(
    mask_opacity: int, translation: int, clipping: str
) -> None:
    value = append_raster(
        empty_document(4, 1),
        raster_png(
            np.array([[[0, 0, 1, alpha / 255] for alpha in (0, 128, 255, 128)]], np.float32)
        ),
    )
    record = value.to_record()
    mask_resource = record["layers"][record["rootLayerIds"][0]]["resourceId"]
    value = append_raster(value, raster_png(np.array([[[1, 0, 0]] * 4], np.float32)))
    leaf = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [leaf]}])
    identifier = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [
            {
                "op": "add_mask",
                "ownerLayerId": identifier,
                "mask": {
                    "resourceId": mask_resource,
                    "sourceRect": {"x": 0, "y": 0, "width": 4, "height": 1},
                    "channel": "alpha",
                    "opacity": mask_opacity,
                },
            }
        ],
    )
    record = value.to_record()
    record["canvas"]["compositing"] = "linear-premultiplied-alpha"
    group = record["layers"][identifier]
    group["opacity"] = 32768
    group["transform"]["tx"] = translation * 1_000_000
    group["clipping"] = clipping
    isolated = ImageDocument.from_record(record).render()
    group["isolation"] = "pass-through"
    passed = ImageDocument.from_record(record).render()
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(passed.png))),
        np.asarray(Image.open(io.BytesIO(isolated.png))),
    )
    np.testing.assert_array_equal(passed.pixels[..., 3], isolated.pixels[..., 3])


def test_nested_pass_through_groups_visit_each_raster_once(monkeypatch) -> None:
    import dinkster_image_document.render as renderer

    value = append_raster(empty_document(1, 1), raster_png(np.array([[[0, 0, 1]]], np.float32)))
    value = append_raster(
        value, raster_png(np.array([[[1, 0, 0]]], np.float32)), blend_mode="multiply"
    )
    for _ in range(20):
        identifier = value.to_record()["rootLayerIds"][-1]
        value = apply_commands(value, [{"op": "group", "ids": [identifier]}])
        identifier = value.to_record()["rootLayerIds"][-1]
        value = apply_commands(
            value, [{"op": "layer", "id": identifier, "changes": {"isolation": "pass-through"}}]
        )
    visits = 0
    original = renderer._composite

    def counted(*args, **kwargs):
        nonlocal visits
        visits += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer, "_composite", counted)
    np.testing.assert_array_equal(flatten(value)[0], [[[[0, 0, 0]]]])
    assert visits == 2


@pytest.mark.parametrize("format", ["native", "ora", "psd"])
def test_layer_file_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes

    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(tmp_path), "mode": "readwrite"}]})
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    value = document()
    saved = cast(
        AssetRef,
        execute_graph_node(
            SaveLayers,
            {
                "layers": TypedLiteral("dinkster.layers", value.to_record()),
                "target": {"mount": "out", "prefix": "layers"},
                "format": format,
            },
        )["document"],
    )
    suffix, media_type = {
        "native": (".json", IMAGE_DOCUMENT_MEDIA_TYPE),
        "ora": (".ora", "image/openraster"),
        "psd": (".psd", "image/vnd.adobe.photoshop"),
    }[format]
    assert saved.name.endswith(suffix)
    assert saved.media_type == media_type
    content = (tmp_path / saved.name).read_bytes()
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest_bytes(content)) as writer:
        writer.write(content)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    bound = AssetRef.from_wire(saved.to_wire(), vault)
    expected = LoadLayers.execute(document=bound)["layers"]
    loaded = execute_graph_node(LoadLayers, {"document": bound.to_wire()})["layers"]
    assert isinstance(loaded, ImageDocument)
    assert isinstance(expected, ImageDocument)
    assert loaded.data == expected.data
    if format != "psd":
        assert loaded.data == value.data
    assert loaded.render().png == value.render().png


@pytest.mark.parametrize(
    ("format", "mode", "media_type", "alpha_mode"),
    [
        ("PNG", "RGBA", "image/png", "straight"),
        ("JPEG", "RGB", "image/jpeg", "opaque"),
        ("WEBP", "RGBA", "image/webp", "straight"),
    ],
)
def test_load_layers_node_creates_documents_from_raster_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    mode: str,
    media_type: str,
    alpha_mode: str,
) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes

    output = io.BytesIO()
    image = Image.new(mode, (3, 2), (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60))
    image.save(output, format, **({"lossless": True} if format == "WEBP" else {}))
    payload = output.getvalue()
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    reference = AssetRef(
        digest, f"source.{format.lower()}", len(payload), media_type, resolver=vault
    )
    loaded = execute_graph_node(LoadLayers, {"document": reference.to_wire()})["layers"]
    assert isinstance(loaded, ImageDocument)
    expected = load_document(payload)
    assert loaded.data == expected.data
    record = loaded.to_record()
    assert (record["canvas"]["width"], record["canvas"]["height"]) == (3, 2)
    assert len(record["rootLayerIds"]) == 1
    resource = record["resources"][record["layers"][record["rootLayerIds"][0]]["resourceId"]]
    assert (resource["mediaType"], resource["alphaMode"], resource["byteSize"]) == (
        media_type,
        alpha_mode,
        len(payload),
    )
    assert loaded.read_resource(resource) == payload
    assert flatten(loaded)[0].shape == (1, 2, 3, 4 if alpha_mode == "straight" else 3)


@pytest.mark.parametrize("large", [False, True])
def test_isolated_worker_shm_flatten_matches_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, large: bool
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))

    async def run() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        register_image_types(registry)
        packages = Path(__file__).parents[1] / "packages"
        group = GroupIsolatedWorker(
            "layer-test",
            [
                packages / name / "dinkster-pack.toml"
                for name in (
                    "dinkster-nodes-foundation",
                    "dinkster-nodes-media-io",
                    "dinkster-nodes-image",
                )
            ],
            registry,
            shm_threshold=1,
        )
        await group.start()
        worker = group.members["dinkster-nodes-image"]
        try:
            value = large_document() if large else document()
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            edited = await engine.run(
                Graph(
                    nodes={
                        "edit": GraphNode(
                            "dinkster.layers.edit",
                            {
                                "layers": TypedLiteral("dinkster.layers", value.to_record()),
                                "commands": "[]",
                            },
                        )
                    }
                ),
                ["edit"],
            )
            assert (
                cast(ImageDocument, edited.outputs["edit"]["layers"].resolve()).data == value.data
            )
            if large:
                return
            source = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
            built = await engine.run(
                Graph(
                    nodes={
                        "add": GraphNode(
                            "dinkster.layers.add",
                            {"image": TypedLiteral("dinkster.image", source.tolist())},
                        )
                    }
                ),
                ["add"],
            )
            published = cast(ImageDocument, built.outputs["add"]["layers"].resolve())
            resource = next(iter(published.to_record()["resources"].values()))
            assert "inline" not in resource
            expected = cast(ImageDocument, AddLayer.execute(image=source)["layers"])
            assert published.data == expected.data
            np.testing.assert_array_equal(flatten(published)[0], flatten(expected)[0])
            graph = Graph(
                nodes={
                    "flatten": GraphNode(
                        "dinkster.layers.flatten",
                        {"layers": TypedLiteral("dinkster.layers", value.to_record())},
                    )
                }
            )
            result = await engine.run(graph, ["flatten"])
            for name, expected in FlattenLayers.execute(layers=value).items():
                actual = result.outputs["flatten"][name].resolve()
                np.testing.assert_array_equal(actual, expected)
        finally:
            await group.close()

    asyncio.run(run())
