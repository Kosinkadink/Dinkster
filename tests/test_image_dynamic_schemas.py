from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import (
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    SlotValue,
    TypeExpr,
)
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy.image import register_image_type
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_image import IMAGE_NODES
from dinkster_nodes_image.adjust import ImageAdjust
from dinkster_nodes_image.batch import (
    ImageBatchCombine,
    ImageBatchEdit,
    ImageListToBatch,
    MaskBatchCombine,
    MaskBatchEdit,
)
from dinkster_nodes_image.composition import ImageComposite
from dinkster_nodes_image.drawing import DrawRegion, MakeImage
from dinkster_nodes_image.filters import ImageFilter
from dinkster_nodes_image.geometry import ImageCrop, ImageResize, ImageTransform
from dinkster_nodes_image.mask import ImageToMask, MakeMask, MaskMorphology, MaskToImage, TextMask
from dinkster_nodes_image.transition import ImageTransition
from dinkster_schema import DynamicEntry, build_node_types, build_schemas
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

NODES = (
    ImageResize,
    ImageTransform,
    ImageCrop,
    ImageComposite,
    ImageAdjust,
    ImageFilter,
    MakeMask,
    TextMask,
    MaskMorphology,
    ImageToMask,
    MaskToImage,
    MakeImage,
    DrawRegion,
    ImageBatchEdit,
    MaskBatchEdit,
    ImageBatchCombine,
    MaskBatchCombine,
    ImageListToBatch,
    ImageTransition,
)


def _combo(entries: Sequence[DynamicEntry], combo_id: str) -> DynamicComboSpec:
    return next(
        entry for entry in entries if isinstance(entry, DynamicComboSpec) and entry.id == combo_id
    )


def _option(combo: DynamicComboSpec, key: str) -> DynamicComboOption:
    option = combo.option(key)
    assert option is not None
    return option


def _ids(option: DynamicComboOption) -> tuple[str, ...]:
    return tuple(entry.id for entry in option.inputs)


def _options(combo: DynamicComboSpec) -> dict[str, tuple[str, ...]]:
    return {option.key: _ids(option) for option in combo.options}


def _construct_paths(entries: Sequence[DynamicEntry], parent: str = "") -> set[str]:
    paths: set[str] = set()
    for entry in entries:
        path = f"{parent}.{entry.id}" if parent else entry.id
        if isinstance(entry, DynamicComboSpec):
            paths.add(path)
            for option in entry.options:
                paths.update(_construct_paths(option.inputs, path))
        elif isinstance(entry, DynamicSlotSpec):
            paths.add(path)
            paths.update(_construct_paths(entry.inputs, path))
            for variant in entry.variants or ():
                paths.update(_construct_paths(variant.inputs, path))
    return paths


def _dynamic_ids(entries: Sequence[DynamicEntry]) -> set[str]:
    ids: set[str] = set()
    for entry in entries:
        ids.add(entry.id)
        if isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                ids.update(_dynamic_ids(option.inputs))
        elif isinstance(entry, DynamicSlotSpec):
            ids.update(_dynamic_ids(entry.inputs))
            for variant in entry.variants or ():
                ids.update(_dynamic_ids(variant.inputs))
    return ids


def test_image_dynamic_schema_inventory() -> None:
    expected_types = {node.define_schema().node_type for node in NODES}
    registered = {node.schema().node_type: node.schema() for node in IMAGE_NODES}
    assert len(expected_types) == 19
    assert expected_types <= registered.keys()
    assert registered["dinkster.image.resize"].version == 4
    assert all(
        schema.version
        == (3 if node_type in {"dinkster.image.composite", "dinkster.mask.to_image"} else 2)
        for node_type, schema in registered.items()
        if node_type in expected_types and node_type != "dinkster.image.resize"
    )

    paths = {
        node_type: _construct_paths((*registered[node_type].combos, *registered[node_type].slots))
        for node_type in expected_types
    }
    assert sum(len(node_paths) for node_paths in paths.values()) == 32
    assert (
        sum(
            path != "mask" or node_type != "dinkster.image.composite"
            for node_type, node_paths in paths.items()
            for path in node_paths
        )
        == 31
    )
    assert (
        sum(
            len(
                _dynamic_ids((*registered[node_type].combos, *registered[node_type].slots))
                - {combo.id for combo in registered[node_type].combos}
            )
            for node_type in expected_types
        )
        == 150
    )


def test_geometry_dynamic_schema_shape() -> None:
    resize = ImageResize.define_schema()
    assert tuple(item.id for item in resize.inputs) == (
        "image",
        "mask",
        "fit_rounding",
        "apply",
        "interpolation",
    )
    resize_type = TypeExpr.variable("input_type", ("dinkster.image", "dinkster.mask"))
    assert resize.inputs[0].type == resize_type
    assert resize.inputs[1].type == TypeExpr.concrete("dinkster.mask")
    assert resize.inputs[1].required is False
    assert resize.inputs[2].advanced is True
    assert resize.inputs[3].advanced is True
    assert resize.inputs[4].advanced is True
    assert resize.outputs[0].type == resize_type
    assert resize.outputs[1].type == TypeExpr.concrete("dinkster.mask")
    assert resize.outputs[1].optional is True
    assert resize.aliases == ()
    target, mode, divisibility = resize.combos
    assert _options(target) == {
        "dimensions": ("width", "height"),
        "width": ("width",),
        "height": ("height",),
        "longest": ("size",),
        "shortest": ("size",),
        "factor": ("factor",),
        "total_pixels": ("megapixels", "resolution_steps"),
        "match": ("reference",),
        "multiple_cover": ("multiple_of",),
    }
    assert _options(mode) == {
        "stretch": (),
        "fit": (),
        "fill": ("mode_anchor",),
        "pad": ("mode_anchor", "mode_padding"),
    }
    mode_padding = _combo(_option(mode, "pad").inputs, "mode_padding")
    assert _options(mode_padding) == {
        "constant": ("pad_value", "pad_color"),
        "edge_average": (),
        "edge_pixel": (),
        "blurred_background": (),
    }
    assert _options(divisibility) == {
        "none": (),
        "nearest": ("multiple_of",),
        "crop": ("multiple_of", "final_anchor"),
        "pad": ("multiple_of", "final_anchor", "final_padding"),
    }
    final_padding = _combo(_option(divisibility, "pad").inputs, "final_padding")
    assert _options(final_padding) == {
        "constant": ("final_pad_value", "final_pad_color"),
        "edge_average": (),
        "edge_pixel": (),
        "blurred_background": (),
    }

    transform = ImageTransform.define_schema()
    assert tuple(item.id for item in transform.inputs) == ("image",)
    assert _options(transform.combos[0]) == {
        "rotate_90": ("steps",),
        "rotate": ("angle", "expand", "interpolation", "fill"),
        "flip_horizontal": (),
        "flip_vertical": (),
        "translate": ("x", "y", "units", "interpolation", "fill"),
        "shear": ("x", "y", "units", "interpolation", "fill"),
        "pad": ("left", "top", "right", "bottom", "feathering"),
    }

    crop = ImageCrop.define_schema()
    assert tuple(item.id for item in crop.inputs) == ("image", "padding", "rounding")
    assert _options(crop.combos[0]) == {
        "coordinates": ("placement", "x", "y", "width", "height"),
        "region": ("region", "mask", "mask_blur"),
        "mask": ("mask", "mask_threshold", "mask_blur"),
    }
    assert _option(crop.combos[0], "region").inputs[0].required is True
    assert _option(crop.combos[0], "region").inputs[1].required is False
    assert _option(crop.combos[0], "mask").inputs[0].required is True
    assert _options(crop.combos[1]) == {"clip": (), "pad": ("fill",), "comfy": ()}


def test_mask_dynamic_schema_shape() -> None:
    make = MakeMask.define_schema()
    assert tuple(item.id for item in make.inputs) == ("width", "height", "batch_size", "invert")
    operation = make.combos[0]
    assert _options(operation) == {
        "solid": ("foreground",),
        "rectangle": (
            "foreground",
            "background",
            "x",
            "y",
            "shape_origin",
            "shape_width",
            "shape_height",
            "grow",
        ),
        "ellipse": (
            "foreground",
            "background",
            "x",
            "y",
            "shape_origin",
            "shape_width",
            "shape_height",
            "grow",
        ),
        "triangle": (
            "foreground",
            "background",
            "x",
            "y",
            "shape_origin",
            "shape_width",
            "shape_height",
            "grow",
        ),
        "polygon": ("foreground", "background", "points_x", "points_y"),
        "region": ("foreground", "background", "region"),
        "linear_gradient": ("foreground", "background", "angle"),
        "radial_gradient": ("foreground", "background", "center_x", "center_y", "radius"),
        "frame_gradient": (),
        "transition": (
            "foreground",
            "background",
            "start_frame",
            "end_frame",
            "transition_type",
            "timing_function",
        ),
        "checkerboard": ("foreground", "background", "tile_size"),
        "noise": ("foreground", "background", "seed", "noise_mode"),
    }
    assert _options(_combo(_option(operation, "noise").inputs, "noise_mode")) == {
        "uniform": (),
        "binary": ("noise_density",),
    }

    text = TextMask.define_schema()
    assert "color" in {item.id for item in text.inputs}
    assert "foreground" not in {item.id for item in text.inputs}
    assert _options(text.combos[0]) == {"foreground": ("foreground",), "color_red": ()}

    morphology = MaskMorphology.define_schema()
    assert tuple(item.id for item in morphology.inputs) == ("mask",)
    morph = morphology.combos[0]
    assert _options(morph) == {
        "threshold": ("threshold",),
        "remove_small_components": ("minimum_area", "threshold", "connectivity"),
        "fill_holes": (),
        "grow_erode": ("radius", "iterations", "tapered_corners", "edge_policy"),
        "grow_blur": (
            "radius",
            "blur_radius",
            "blur_amount",
            "incremental_expandrate",
            "flip_input",
            "lerp_alpha",
            "decay_factor",
            "fill_holes",
            "iterations",
            "tapered_corners",
            "sigma",
            "edge_policy",
        ),
        "open": ("kernel_size", "iterations", "tapered_corners", "edge_policy"),
        "close": ("kernel_size", "iterations", "tapered_corners", "edge_policy"),
        "feather_edges": ("left", "top", "right", "bottom"),
        "blur": ("radius", "sigma", "edge_policy"),
        "offset": ("x", "y", "edge_policy"),
        "remap": ("input_low", "input_high", "output_low", "output_high", "clamp"),
        "round": ("radius", "iterations", "edge_policy"),
        "block": ("kernel_size", "block_mode"),
        "invert": (),
        "crop": ("x", "y", "width", "height"),
    }
    for key in ("grow_erode", "grow_blur", "open", "close", "blur", "offset", "round"):
        assert _options(_combo(_option(morph, key).inputs, "edge_policy")) == {
            "constant": ("edge_value",),
            "replicate": (),
            "reflect": (),
            "wrap": (),
        }

    to_mask = ImageToMask.define_schema()
    assert tuple(item.id for item in to_mask.inputs) == ("image", "invert")
    policy = to_mask.combos[0]
    assert _options(policy) == {
        "channel": ("channel",),
        "exact_color": ("color_source",),
        "tolerance_color": ("color_source", "tolerance", "metric"),
    }
    for key in ("exact_color", "tolerance_color"):
        assert _options(_combo(_option(policy, key).inputs, "color_source")) == {
            "hex": ("color",),
            "integer": ("color_value",),
            "channels": ("red", "green", "blue"),
        }

    to_image = MaskToImage.define_schema()
    assert tuple(item.id for item in to_image.inputs) == ("mask",)
    assert _options(to_image.combos[0]) == {
        "rgb": (),
        "rgba": ("color", "mask_polarity"),
    }


def test_composite_batch_drawing_and_filter_dynamic_schema_shape() -> None:
    composite = ImageComposite.define_schema()
    assert tuple(item.id for item in composite.inputs) == (
        "destination",
        "source",
        "x",
        "y",
        "blend_mode",
        "factor",
        "clamp_output",
        "preserve_destination_alpha",
        "batch_policy",
    )
    assert _options(composite.combos[0]) == {
        "none": (),
        "stretch": ("interpolation",),
        "fill": ("interpolation",),
    }
    (mask_slot,) = composite.slots
    assert mask_slot.id == "mask" and mask_slot.required is False
    assert mask_slot.variants is not None
    assert [
        (variant.key, variant.type, tuple(item.id for item in variant.inputs))
        for variant in mask_slot.variants
    ] == [("mask", mask_slot.variants[0].type, ("mask_polarity",))]

    expected = {
        ImageAdjust: ("operation",),
        ImageFilter: ("operation",),
        MakeImage: ("color_source", "operation"),
        DrawRegion: ("mode",),
        ImageBatchEdit: ("operation",),
        MaskBatchEdit: ("operation",),
        ImageBatchCombine: ("operation", "shape_policy"),
        MaskBatchCombine: ("operation", "shape_policy"),
        ImageListToBatch: ("shape_policy",),
        ImageTransition: ("mode", "shape_policy"),
    }
    for node, combo_ids in expected.items():
        assert tuple(combo.id for combo in node.define_schema().combos) == combo_ids


def test_materialized_dynamic_inputs_preserve_execution() -> None:
    image = np.arange(36, dtype=np.float32).reshape(1, 3, 4, 3) / 35.0
    expected_resize = ImageResize.execute(
        image=image,
        target="width",
        width=2,
        mode="stretch",
        interpolation="bilinear",
    )
    materialized_resize = cast("Any", ImageResize.execute)(
        image=image,
        **{
            "target": "width",
            "target.width": 2,
            "mode": "stretch",
            "interpolation": "bilinear",
            "divisibility": "none",
        },
    )
    np.testing.assert_array_equal(materialized_resize["image"], expected_resize["image"])

    expected_mask = ImageToMask.execute(
        image=image,
        policy="tolerance_color",
        color_source="integer",
        color_value=0x123456,
        tolerance=0.2,
        metric="max_channel",
    )
    materialized_mask = cast("Any", ImageToMask.execute)(
        image=image,
        policy="tolerance_color",
        **{
            "policy.color_source": "integer",
            "policy.color_source.color_value": 0x123456,
            "policy.tolerance": 0.2,
            "policy.metric": "max_channel",
        },
    )
    np.testing.assert_array_equal(materialized_mask["mask"], expected_mask["mask"])

    mask = np.full((1, 3, 4), 0.25, dtype=np.float32)
    expected_composite = ImageComposite.execute(
        destination=image,
        source=1.0 - image,
        mask=mask,
        mask_polarity="transparency",
    )
    materialized_composite = ImageComposite.execute(
        destination=image,
        source=1.0 - image,
        mask=SlotValue("mask", mask, {"mask_polarity": "transparency"}),
    )
    np.testing.assert_array_equal(materialized_composite["image"], expected_composite["image"])

    components = np.array([[[0.75, 0.0, 0.8, 0.9]]], dtype=np.float32)
    cleaned = cast("Any", MaskMorphology.execute)(
        mask=components,
        **{
            "operation": "remove_small_components",
            "operation.minimum_area": 2,
            "operation.threshold": 0.5,
            "operation.connectivity": "4",
        },
    )
    np.testing.assert_array_equal(
        cleaned["mask"], np.array([[[0.0, 0.0, 0.8, 0.9]]], dtype=np.float32)
    )


def test_resize_v4_binds_image_and_mask_outputs_at_worker_boundary() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_image_type(registry, "dinkster.image")
        register_image_type(registry, "dinkster.mask")
        schemas = build_schemas((ImageResize,))
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types((ImageResize,)), registry),
            cache=MemoryLRUCache(),
        )
        choices = {
            "target": "dimensions",
            "mode": "stretch",
            "divisibility": "none",
        }
        inputs = {
            "target.width": 2,
            "target.height": 2,
            "interpolation": "nearest-exact",
        }
        image = np.arange(36, dtype=np.float32).reshape(1, 3, 4, 3) / 35.0
        mask = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 11.0
        graph = Graph(
            {
                "image": GraphNode(
                    "dinkster.image.resize",
                    {"image": TypedLiteral("dinkster.image", image.tolist()), **inputs},
                    slot_variants=choices,
                ),
                "mask": GraphNode(
                    "dinkster.image.resize",
                    {"image": TypedLiteral("dinkster.mask", mask.tolist()), **inputs},
                    slot_variants=choices,
                ),
            }
        )

        result = await engine.run(graph, ("image", "mask"))
        image_output = result.outputs["image"]["image"]
        mask_output = result.outputs["mask"]["image"]
        assert image_output.type_id == "dinkster.image"
        assert mask_output.type_id == "dinkster.mask"
        assert np.asarray(image_output.resolve()).shape == (1, 2, 2, 3)
        assert np.asarray(mask_output.resolve()).shape == (1, 2, 2)

    asyncio.run(scenario())
