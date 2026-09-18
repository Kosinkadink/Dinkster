from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import CompositorWidget, TypeRegistry
from dinkster_image_document.document import ImageDocument
from dinkster_nodes_image import (
    COMPOSITOR_BLEND_MODES,
    COMPOSITOR_TYPE,
    IMAGE_NODES,
    LAYERS_TYPE,
    AddLayer,
    CompositorLayer,
    CompositorRecipe,
    CompositorSourceLayer,
    CompositorTransform,
    CreateLayeredImage,
    Detection,
    LayersFromBoundingBoxes,
    LayerStack,
    Region,
    register_image_types,
)
from dinkster_nodes_image.compositor_blend import blend, linear_to_srgb
from dinkster_nodes_image.compositor_types import (
    MAX_COMPOSITOR_DIMENSION,
    MAX_COMPOSITOR_LAYERS,
    coerce_compositor_recipe,
    compositor_recipe_meta,
    decode_compositor_recipe,
    decode_layer_stack,
    encode_compositor_recipe,
    encode_layer_stack,
    layer_stack_meta,
    source_layer_fingerprint,
)
from dinkster_schema import use_reporter

Captured = tuple[str, Mapping[str, object], bytes | None]
_LAYERS_MAGIC = b"DINKSTER-LAYERS\x00\x01"


def _solid(
    color: tuple[float, ...],
    *,
    width: int = 1,
    height: int = 1,
    batch: int = 1,
) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(color, dtype=np.float32),
        (batch, height, width, len(color)),
    ).copy()


def _source(
    color: tuple[float, ...] = (1.0, 0.0, 0.0),
    *,
    name: str = "Layer",
    width: int = 1,
    height: int = 1,
    x: float = 0.0,
    y: float = 0.0,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    mask: np.ndarray | None = None,
) -> CompositorSourceLayer:
    return CompositorSourceLayer(
        image=_solid(color, width=width, height=height),
        mask=mask,
        name=name,
        x=x,
        y=y,
        width=width,
        height=height,
        opacity=opacity,
        blend_mode=blend_mode,
    )


def _edit(
    source: CompositorSourceLayer,
    source_index: int,
    fingerprint: str,
    **overrides: object,
) -> CompositorLayer:
    values: dict[str, object] = {
        "id": f"layer-{source_index}-{fingerprint[:8]}",
        "source_index": source_index,
        "name": source.name,
        "visible": source.visible,
        "opacity": source.opacity,
        "blend_mode": source.blend_mode,
        "transform": CompositorTransform(
            source.x,
            source.y,
            source.width,
            source.height,
            source.rotation,
        ),
        "flip_horizontal": source.flip_horizontal,
        "flip_vertical": source.flip_vertical,
    }
    values.update(overrides)
    return CompositorLayer(**values)  # type: ignore[arg-type]


def _recipe(
    sources: tuple[CompositorSourceLayer, ...],
    *,
    width: int,
    height: int,
    layers: tuple[CompositorLayer, ...] | None = None,
    background_color: str = "#000000",
    background_opacity: float = 0.0,
    background_visible: bool = False,
) -> CompositorRecipe:
    fingerprints = tuple(source_layer_fingerprint(source) for source in sources)
    return CompositorRecipe(
        version=1,
        input_fingerprints=fingerprints,
        canvas_width=width,
        canvas_height=height,
        background_color=background_color,
        background_opacity=background_opacity,
        background_visible=background_visible,
        layers=(
            tuple(_edit(source, index, fingerprints[index]) for index, source in enumerate(sources))
            if layers is None
            else layers
        ),
    )


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_image_types(registry)
    register_image_types(registry)
    return registry


def _legacy_registry() -> TypeRegistry:
    registry = TypeRegistry()
    registry.register(
        LAYERS_TYPE, encode=encode_layer_stack, decode=decode_layer_stack, meta=layer_stack_meta
    )
    registry.register(
        COMPOSITOR_TYPE,
        encode=encode_compositor_recipe,
        decode=decode_compositor_recipe,
        coerce=coerce_compositor_recipe,
        meta=compositor_recipe_meta,
    )
    return registry


def _replace_layers_header(encoded: bytes, header: object) -> bytes:
    prefix = len(_LAYERS_MAGIC) + 4
    old_size = int.from_bytes(encoded[len(_LAYERS_MAGIC) : prefix], "little")
    payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    return b"".join(
        (
            _LAYERS_MAGIC,
            len(payload).to_bytes(4, "little"),
            payload,
            encoded[prefix + old_size :],
        )
    )


def _layers_header(encoded: bytes) -> dict[str, object]:
    prefix = len(_LAYERS_MAGIC) + 4
    size = int.from_bytes(encoded[len(_LAYERS_MAGIC) : prefix], "little")
    return cast("dict[str, object]", json.loads(encoded[prefix : prefix + size]))


def test_compositor_nodes_and_runtime_types_are_registered() -> None:
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_NODES}
    assert {
        "dinkster.layers.add",
        "dinkster.layers.from_bounding_boxes",
        "dinkster.image.create_layered",
    } <= schemas.keys()
    assert schemas["dinkster.layers.add"].idempotent is True
    assert schemas["dinkster.layers.from_bounding_boxes"].idempotent is True
    create = schemas["dinkster.image.create_layered"]
    assert create.output_node is True
    assert create.emits_previews is True
    assert create.idempotent is False
    compositor = next(input_spec for input_spec in create.inputs if input_spec.id == "compositor")
    assert compositor.type.types == (COMPOSITOR_TYPE,)
    assert compositor.required is False
    assert compositor.default == {"version": 2, "documentDigest": None, "commands": []}
    assert compositor.widget == CompositorWidget()
    boxes = schemas["dinkster.layers.from_bounding_boxes"].inputs[2].type
    assert boxes.kind == "list"
    assert boxes.element is not None
    assert boxes.element.kind == "union"
    assert boxes.element.types == ("dinkster.region", "dinkster.detection")

    registry = _registry()
    assert registry.spec(LAYERS_TYPE).type_id == LAYERS_TYPE
    assert registry.spec(COMPOSITOR_TYPE).type_id == COMPOSITOR_TYPE


def test_layer_stack_codec_round_trips_pixels_masks_placement_and_meta() -> None:
    image = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / 23
    mask = np.asarray([[[0.0, 1.0], [0.25, 0.75]]], dtype=np.float32)
    source = CompositorSourceLayer(
        image=image,
        mask=mask,
        name="Frames",
        x=-2.5,
        y=4.0,
        width=8,
        height=6,
        rotation=0.25,
        opacity=0.75,
        blend_mode="screen",
        visible=False,
        flip_horizontal=True,
        flip_vertical=False,
    )
    stack = LayerStack((source,), canvas_width=16, canvas_height=12)
    spec = _legacy_registry().spec(LAYERS_TYPE)
    encoded = spec.encode(stack)
    assert encoded == spec.encode(stack)
    decoded = cast("LayerStack", spec.decode(encoded))
    assert (decoded.canvas_width, decoded.canvas_height) == (16, 12)
    assert len(decoded.layers) == 1
    actual = decoded.layers[0]
    np.testing.assert_array_equal(actual.image, image)
    np.testing.assert_array_equal(actual.mask, mask)
    assert (
        actual.name,
        actual.x,
        actual.y,
        actual.width,
        actual.height,
        actual.rotation,
        actual.opacity,
        actual.blend_mode,
        actual.visible,
        actual.flip_horizontal,
        actual.flip_vertical,
    ) == ("Frames", -2.5, 4.0, 8.0, 6.0, 0.25, 0.75, "screen", False, True, False)
    assert spec.meta is not None
    assert spec.meta(stack) == {"layers": 2, "canvas": (16, 12)}
    assert [member.name for member in decoded.expanded()] == ["Frames 1", "Frames 2"]


def test_source_layers_own_immutable_pixels_and_masks() -> None:
    image = _solid((0.125, 0.375, 0.625))
    mask = np.full((1, 1, 1), 0.25, dtype=np.float32)
    source = CompositorSourceLayer(image, mask, "Layer", 0, 0, 1, 1)
    fingerprint = source_layer_fingerprint(source)
    image.fill(1.0)
    mask.fill(1.0)
    np.testing.assert_array_equal(source.image, _solid((0.125, 0.375, 0.625)))
    np.testing.assert_array_equal(source.mask, np.full((1, 1, 1), 0.25, dtype=np.float32))
    assert not source.image.flags.writeable
    assert source.mask is not None and not source.mask.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        source.image.setflags(write=True)
    assert source_layer_fingerprint(source) == fingerprint


def test_source_layers_enforce_dimension_and_operation_memory_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="dimensions exceed"):
        CompositorSourceLayer(
            np.zeros((1, 1, MAX_COMPOSITOR_DIMENSION + 1, 3), dtype=np.float32),
            None,
            "Wide",
            0,
            0,
            1,
            1,
        )
    with pytest.raises(ValueError, match="dimensions exceed"):
        CompositorSourceLayer(
            _solid((1.0, 0.0, 0.0)),
            np.zeros((1, 1, MAX_COMPOSITOR_DIMENSION + 1), dtype=np.float32),
            "Wide mask",
            0,
            0,
            1,
            1,
        )

    monkeypatch.setattr("dinkster_nodes_image.support.MAX_IMAGE_BYTES", 8)
    with pytest.raises(ValueError, match="image operation limit"):
        _source()
    with pytest.raises(ValueError, match="image operation limit"):
        CompositorSourceLayer(
            np.zeros((1, 1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            "Large mask",
            0,
            0,
            1,
            1,
        )


def test_layer_stack_codec_rejects_unknown_truncated_and_trailing_data() -> None:
    spec = _legacy_registry().spec(LAYERS_TYPE)
    encoded = spec.encode(LayerStack((_source(),)))
    with pytest.raises(ValueError, match="payload is truncated"):
        spec.decode(encoded[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        spec.decode(encoded + b"junk")
    header = _layers_header(encoded)
    header["future"] = True
    with pytest.raises(ValueError, match="requires exactly"):
        spec.decode(_replace_layers_header(encoded, header))


def test_layer_stack_codec_round_trips_maximum_unicode_names() -> None:
    sources = tuple(_source(name="\U0001f642" * 256) for _ in range(MAX_COMPOSITOR_LAYERS))
    spec = _legacy_registry().spec(LAYERS_TYPE)
    encoded = spec.encode(LayerStack(sources))
    decoded = cast("LayerStack", spec.decode(encoded))
    assert [source.name for source in decoded.layers] == [source.name for source in sources]


def test_compositor_names_reject_non_scalar_unicode() -> None:
    with pytest.raises(ValueError, match="surrogate code points"):
        _source(name="bad\ud800name")
    source = _source()
    fingerprint = source_layer_fingerprint(source)
    with pytest.raises(ValueError, match="surrogate code points"):
        _edit(source, 0, fingerprint, name="bad\udc00name")


def test_compositor_recipe_codec_is_canonical_strict_and_bounded() -> None:
    source = _source()
    recipe = _recipe(
        (source,),
        width=32,
        height=24,
        background_color="#A0B0C0",
        background_opacity=0.5,
        background_visible=True,
    )
    spec = _legacy_registry().spec(COMPOSITOR_TYPE)
    encoded = spec.encode(recipe)
    assert b" " not in encoded
    assert spec.encode(spec.decode(encoded)) == encoded
    assert cast("CompositorRecipe", spec.decode(encoded)).background_color == "#a0b0c0"
    assert spec.coerce is not None
    assert spec.coerce(recipe.to_record()) == recipe
    assert spec.meta is not None
    assert spec.meta(recipe) == {"version": 1, "inputs": 1, "canvas": (32, 24)}
    empty = cast("CompositorRecipe", spec.coerce({"version": 1, "inputs": [], "layers": []}))
    assert empty.to_record() == {"version": 1, "inputs": [], "layers": []}
    assert spec.encode(empty) == b'{"inputs":[],"layers":[],"version":1}'

    record = recipe.to_record()
    record["future"] = True
    with pytest.raises(ValueError, match="requires exactly"):
        spec.coerce(record)
    wrong_version = recipe.to_record()
    wrong_version["version"] = True
    with pytest.raises(ValueError, match="version must be 1"):
        spec.coerce(wrong_version)
    nonempty_short = {"version": 1, "inputs": ["a" * 64], "layers": []}
    with pytest.raises(ValueError, match="short compositor recipe form must be exactly empty"):
        spec.coerce(nonempty_short)
    too_wide = recipe.to_record()
    cast("dict[str, object]", too_wide["canvas"])["width"] = MAX_COMPOSITOR_DIMENSION + 1
    with pytest.raises(ValueError, match="canvas width"):
        spec.coerce(too_wide)


def test_compositor_values_enforce_layer_count_and_source_permutation() -> None:
    image = _solid((1.0, 0.0, 0.0), batch=MAX_COMPOSITOR_LAYERS + 1)
    with pytest.raises(ValueError, match="cannot exceed 50 expanded layers"):
        LayerStack(
            (
                CompositorSourceLayer(
                    image=image,
                    mask=None,
                    name="Too many",
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                ),
            )
        )

    first = _source(name="First")
    second = _source(name="Second")
    fingerprints = tuple(source_layer_fingerprint(source) for source in (first, second))
    duplicate = (
        _edit(first, 0, fingerprints[0]),
        _edit(second, 0, fingerprints[1], id="second"),
    )
    with pytest.raises(ValueError, match="permutation"):
        _recipe((first, second), width=1, height=1, layers=duplicate)


def test_add_layer_preserves_native_batch_mask_and_existing_canvas() -> None:
    image = _solid((1.0, 0.0, 0.0), width=2, height=3, batch=2)
    mask = np.zeros((1, 3, 2), dtype=np.float32)
    first = cast(
        "ImageDocument",
        AddLayer.execute(
            image=image,
            mask=mask,
            name="Frame",
            x=4,
            y=5,
            width=8,
            height=9,
            rotation=0.5,
            opacity=0.25,
            blend_mode="multiply",
            flip_horizontal=True,
        )["layers"],
    )
    record = first.to_record()
    assert len(record["layers"]) == 2
    expanded = list(record["layers"].values())
    assert [source["name"] for source in expanded] == ["Frame 1", "Frame 2"]
    assert all(len(source["maskIds"]) == 1 for source in expanded)
    assert [
        (resource["width"], resource["height"]) for resource in record["resources"].values()
    ] == [(2, 3), (2, 3)]
    transform = expanded[0]["transform"]["components"]
    assert tuple(transform[key] for key in ("x", "y", "width", "height")) == (4, 5, 8, 9)
    record["canvas"].update(width=20, height=10)
    first = ImageDocument.from_record(record)
    appended = cast(
        "ImageDocument",
        AddLayer.execute(image=_solid((0.0, 1.0, 0.0)), layers=first)["layers"],
    )
    record = appended.to_record()
    assert len(record["layers"]) == 3
    assert (record["canvas"]["width"], record["canvas"]["height"]) == (20, 10)


def test_layers_from_bounding_boxes_accepts_regions_and_detections_in_order() -> None:
    images = np.concatenate(
        (_solid((1.0, 0.0, 0.0)), _solid((0.0, 1.0, 0.0))),
        axis=0,
    )
    detection_mask = np.full((1, 1), 0.25, dtype=np.float32)
    boxes = (
        Detection("Subject", 0.75, Region(1, 2, 3, 4), detection_mask),
        Region(5, 6, 7, 8),
    )
    stack = cast(
        "ImageDocument",
        LayersFromBoundingBoxes.execute(
            image=images,
            bounding_boxes=boxes,
            canvas_width=16,
            canvas_height=20,
        )["layers"],
    )
    record = stack.to_record()
    layers = list(record["layers"].values())
    assert [layer["name"] for layer in layers] == ["Subject", "Layer 2"]
    assert [
        tuple(layer["transform"]["components"][key] for key in ("x", "y", "width", "height"))
        for layer in layers
    ] == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]
    assert len(layers[0]["maskIds"]) == 1
    mask = record["masks"][layers[0]["maskIds"][0]]
    assert mask["invert"] is True and mask["channel"] == "luminance"
    assert not layers[1]["maskIds"]
    assert (record["canvas"]["width"], record["canvas"]["height"]) == (16, 20)

    with pytest.raises(ValueError, match="bounding box count"):
        LayersFromBoundingBoxes.execute(image=images, bounding_boxes=boxes[:1])


def test_source_fingerprint_covers_pixels_mask_and_native_placement() -> None:
    source = _source(mask=np.zeros((1, 1, 1), dtype=np.float32))
    assert source_layer_fingerprint(source) == source_layer_fingerprint(
        _source(mask=np.zeros((1, 1, 1), dtype=np.float32))
    )
    assert source_layer_fingerprint(source) != source_layer_fingerprint(
        _source(color=(0.0, 1.0, 0.0), mask=np.zeros((1, 1, 1), dtype=np.float32))
    )
    assert source_layer_fingerprint(source) != source_layer_fingerprint(
        _source(mask=np.ones((1, 1, 1), dtype=np.float32))
    )
    assert source_layer_fingerprint(source) != source_layer_fingerprint(
        _source(x=1, mask=np.zeros((1, 1, 1), dtype=np.float32))
    )


def test_native_fallback_composites_bottom_to_top_and_masks_are_transparency() -> None:
    bottom = _source(width=2)
    top = _source(
        (0.0, 0.0, 1.0),
        x=1,
        opacity=0.5,
        mask=np.zeros((1, 1, 1), dtype=np.float32),
    )
    stack = LayerStack((bottom, top))
    result = CreateLayeredImage.execute(layers=stack)
    image = cast("np.ndarray", result["image"])
    transparency = cast("np.ndarray", result["transparency_mask"])
    np.testing.assert_allclose(
        image,
        np.asarray([[[[1.0, 0.0, 0.0], [127 / 255, 0.0, 128 / 255]]]], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(transparency, np.zeros((1, 1, 2), dtype=np.float32))

    blocked = replace(top, mask=np.ones((1, 1, 1), dtype=np.float32))
    masked = CreateLayeredImage.execute(layers=LayerStack((bottom, blocked)))
    np.testing.assert_array_equal(masked["image"], _solid((1.0, 0.0, 0.0), width=2))


def test_empty_recipe_uses_native_fallback_without_reporting_stale_state() -> None:
    source = _source()
    captured: list[Captured] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))

    with use_reporter(capture):
        result = CreateLayeredImage.execute(
            layers=LayerStack((source,)),
            compositor={"version": 1, "inputs": [], "layers": []},
        )

    np.testing.assert_array_equal(result["image"], source.image)
    state = next(data for name, data, _blob in captured if name == "dinkster.compositor.state")
    assert state["stale"] is False
    assert state["version"] == 2
    assert len(cast("dict[str, list[str]]", state["document"])["rootLayerIds"]) == 1


def test_identity_render_uses_canonical_raster_quantization() -> None:
    image = np.asarray([[[[0.123456, 0.345678, 0.567891]]]], dtype=np.float32)
    result = CreateLayeredImage.execute(
        layers=LayerStack((CompositorSourceLayer(image, None, "Layer", 0, 0, 1, 1),))
    )
    np.testing.assert_array_equal(result["image"], np.rint(image * 255) / 255)


def test_exact_recipe_reorders_flips_and_controls_background_visibility() -> None:
    source = CompositorSourceLayer(
        image=np.asarray([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]], dtype=np.float32),
        mask=None,
        name="Strip",
        x=0,
        y=0,
        width=2,
        height=1,
    )
    fingerprint = source_layer_fingerprint(source)
    flipped = _edit(source, 0, fingerprint, flip_horizontal=True)
    recipe = _recipe((source,), width=2, height=1, layers=(flipped,))
    image = CreateLayeredImage.execute(
        layers=LayerStack((source,)),
        compositor=recipe,
    )["image"]
    np.testing.assert_array_equal(
        image,
        np.asarray([[[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]], dtype=np.float32),
    )

    hidden = replace(flipped, visible=False)
    background = _recipe(
        (source,),
        width=2,
        height=1,
        layers=(hidden,),
        background_color="#00ff00",
        background_opacity=0.5,
        background_visible=True,
    )
    result = CreateLayeredImage.execute(layers=LayerStack((source,)), compositor=background)
    np.testing.assert_array_equal(result["image"], _solid((0.0, 1.0, 0.0, 128 / 255), width=2))
    np.testing.assert_array_equal(
        result["transparency_mask"],
        1 - np.full((1, 1, 2), 128 / 255, dtype=np.float32),
    )

    transparent = replace(background, background_visible=False)
    result = CreateLayeredImage.execute(layers=LayerStack((source,)), compositor=transparent)
    np.testing.assert_array_equal(result["image"], _solid((0.0, 0.0, 0.0, 0.0), width=2))
    np.testing.assert_array_equal(
        result["transparency_mask"],
        np.ones((1, 1, 2), dtype=np.float32),
    )


def test_exact_recipe_does_not_validate_unused_native_fallback_canvas() -> None:
    source = _source(x=MAX_COMPOSITOR_DIMENSION + 1)
    fingerprint = source_layer_fingerprint(source)
    edit = _edit(
        source,
        0,
        fingerprint,
        transform=CompositorTransform(0, 0, 1, 1, 0),
    )
    recipe = _recipe((source,), width=1, height=1, layers=(edit,))
    result = CreateLayeredImage.execute(layers=LayerStack((source,)), compositor=recipe)
    np.testing.assert_array_equal(result["image"], _solid((1.0, 0.0, 0.0)))


def test_fingerprint_mismatch_falls_back_and_reports_exact_layer_streams() -> None:
    first = _source(name="First")
    second = _source((0.0, 1.0, 0.0), name="Second", x=1)
    sources = (first, second)
    recipe = _recipe(sources, width=2, height=1)
    stale = replace(recipe, input_fingerprints=tuple(reversed(recipe.input_fingerprints)))
    captured: list[Captured] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))

    stack = LayerStack(sources)
    with use_reporter(capture):
        stale_result = CreateLayeredImage.execute(layers=stack, compositor=stale)
    fallback_result = CreateLayeredImage.execute(layers=stack)
    np.testing.assert_array_equal(stale_result["image"], fallback_result["image"])
    assert [name for name, _, _ in captured] == [
        "preview",
        "preview",
        "dinkster.compositor.state",
    ]
    assert [data for _, data, _ in captured[:2]] == [
        {"mime": "image/png", "width": 2, "height": 1, "stream": "compositor.layer.0"},
        {"mime": "image/png", "width": 2, "height": 1, "stream": "compositor.layer.1"},
    ]
    assert all(blob is not None and blob.startswith(b"\x89PNG") for _, _, blob in captured[:2])
    state = captured[2]
    assert state[2] is None
    assert state[1]["stale"] is True
    assert state[1]["version"] == 2
    assert len(cast("dict[str, dict[str, object]]", state[1]["document"])["layers"]) == 2
    assert state[1]["layerStreams"] == ["compositor.layer.0", "compositor.layer.1"]


def test_perceptual_and_linear_light_compositing_are_distinct_and_exact() -> None:
    black = _source((0.0, 0.0, 0.0))
    white = _source((1.0, 1.0, 1.0), opacity=0.5)
    stack = LayerStack((black, white))
    perceptual = cast(
        "np.ndarray",
        CreateLayeredImage.execute(layers=stack, color_space="perceptual")["image"],
    )
    linear = cast(
        "np.ndarray",
        CreateLayeredImage.execute(layers=stack, color_space="linear")["image"],
    )
    np.testing.assert_allclose(perceptual, 128 / 255, rtol=0, atol=1e-7)
    expected_linear = float(linear_to_srgb(np.asarray(0.5, dtype=np.float32)))
    np.testing.assert_allclose(linear, round(expected_linear * 255) / 255, rtol=0, atol=1e-6)
    assert expected_linear > 0.7
    np.testing.assert_array_equal(
        CreateLayeredImage.execute(layers=stack, color_space="future")["image"], perceptual
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("normal", (0.8, 0.4, 0.2)),
        ("multiply", (0.2, 0.2, 0.15)),
        ("screen", (0.85, 0.7, 0.8)),
        ("overlay", (0.4, 0.4, 0.6)),
        ("linear_dodge", (1.0, 0.9, 0.95)),
        ("linear_burn", (0.05, 0.0, 0.0)),
        ("difference", (0.55, 0.1, 0.55)),
        ("darken", (0.25, 0.4, 0.2)),
        ("lighten", (0.8, 0.5, 0.75)),
    ),
)
def test_representative_blend_mode_values(
    mode: str,
    expected: tuple[float, float, float],
) -> None:
    bottom = np.asarray([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    top = np.asarray([[[0.8, 0.4, 0.2]]], dtype=np.float32)
    np.testing.assert_allclose(
        blend(bottom, top, mode, seed=1),
        np.asarray([[expected]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_every_blend_mode_is_finite_and_dissolve_is_deterministic() -> None:
    colors = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    for mode in COMPOSITOR_BLEND_MODES:
        result = blend(colors, 1.0 - colors, mode, seed=7)
        assert result.shape == colors.shape
        assert np.isfinite(result).all()
        assert np.logical_and(result >= 0.0, result <= 1.0).all()

    stack = LayerStack(
        (
            _source((0.0, 0.0, 0.0), width=16, height=16),
            _source(
                (1.0, 1.0, 1.0),
                width=16,
                height=16,
                opacity=0.5,
                blend_mode="dissolve",
            ),
        )
    )
    first = cast("np.ndarray", CreateLayeredImage.execute(layers=stack)["image"])
    second = cast("np.ndarray", CreateLayeredImage.execute(layers=stack)["image"])
    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)).issubset({0.0, 1.0})
    assert 0.0 in first and 1.0 in first


def test_compositor_refuses_output_memory_budget_before_allocation() -> None:
    source = _source()
    recipe = _recipe(
        (source,),
        width=MAX_COMPOSITOR_DIMENSION,
        height=MAX_COMPOSITOR_DIMENSION,
    )
    with pytest.raises(ValueError, match="canvas facts are unsupported"):
        CreateLayeredImage.execute(
            layers=LayerStack((source,)),
            compositor=recipe,
        )


def test_migrated_legacy_recipe_preserves_stale_detection_and_mask_placement() -> None:
    from dinkster_nodes_image.layer_document import migrate_layers
    from dinkster_nodes_image.layer_recipe import replay_compositor

    source = _source(width=2, height=1, mask=np.array([[[0, 1]]], dtype=np.float32))
    value = migrate_layers(LayerStack((source,), canvas_width=4, canvas_height=1))
    recipe = _recipe((source,), width=4, height=1)
    edit = replace(recipe.layers[0], transform=CompositorTransform(2, 0, 2, 1, 0))
    recipe = replace(recipe, layers=(edit,))
    actual, stale = replay_compositor(value, recipe, value)
    assert not stale
    pixels = actual.render().pixels
    assert pixels[0, :, 3].tolist() == [0, 0, 65535, 0]
    stale_recipe = replace(recipe, input_fingerprints=("0" * 64,))
    unchanged, stale = replay_compositor(value, stale_recipe, value)
    assert stale and unchanged.data == value.data
