from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_image import IMAGE_NODES, MakeRegion, Region
from dinkster_schema import schema_from_wire, schema_to_wire, validate_replacement_references
from dinkster_schema.replace import rule_from_wire, rule_to_wire

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-image" / "comfy-aliases.json"
DYNAMIC_ALIAS_IDS = frozenset(
    {
        "comfy_alias:comfy-core/AdjustBrightness",
        "comfy_alias:comfy-core/AdjustContrast",
        "comfy_alias:comfy-core/BatchImagesNode",
        "comfy_alias:comfy-core/CropMask",
        "comfy_alias:comfy-core/EmptyImage",
        "comfy_alias:comfy-core/FeatherMask",
        "comfy_alias:comfy-core/GrowMask",
        "comfy_alias:comfy-core/ImageBatch",
        "comfy_alias:comfy-core/ImageBlend",
        "comfy_alias:comfy-core/ImageBlur",
        "comfy_alias:comfy-core/ImageColorToMask",
        "comfy_alias:comfy-core/ImageCompositeMasked",
        "comfy_alias:comfy-core/ImageCrop",
        "comfy_alias:comfy-core/ImageCropV2",
        "comfy_alias:comfy-core/ImageFlip",
        "comfy_alias:comfy-core/ImageFromBatch",
        "comfy_alias:comfy-core/ImageInvert",
        "comfy_alias:comfy-core/ImagePadForOutpaint",
        "comfy_alias:comfy-core/ImageQuantize",
        "comfy_alias:comfy-core/ImageRotate",
        "comfy_alias:comfy-core/ImageScale",
        "comfy_alias:comfy-core/ImageScaleBy",
        "comfy_alias:comfy-core/ImageScaleToMaxDimension",
        "comfy_alias:comfy-core/ImageScaleToTotalPixels",
        "comfy_alias:comfy-core/ImageSharpen",
        "comfy_alias:comfy-core/ImageToMask",
        "comfy_alias:comfy-core/InvertMask",
        "comfy_alias:comfy-core/MaskToImage",
        "comfy_alias:comfy-core/Morphology",
        "comfy_alias:comfy-core/NormalizeImages",
        "comfy_alias:comfy-core/RepeatImageBatch",
        "comfy_alias:comfy-core/ResizeImageMaskNode",
        "comfy_alias:comfy-core/ResizeAndPadImage",
        "comfy_alias:comfy-core/SolidMask",
        "comfy_alias:comfy-core/ThresholdMask",
        "comfy_alias:comfyui-kjnodes/BboxVisualize",
        "comfy_alias:comfyui-kjnodes/ColorToMask",
        "comfy_alias:comfyui-kjnodes/CreateShapeMask",
        "comfy_alias:comfyui-kjnodes/CreateTextMask",
        "comfy_alias:comfyui-kjnodes/GetImageSizeAndCount",
        "comfy_alias:comfyui-kjnodes/GrowMaskWithBlur",
        "comfy_alias:comfyui-kjnodes/ImageBatchJoinWithTransition",
        "comfy_alias:comfyui-kjnodes/ImageBatchMulti",
        "comfy_alias:comfyui-kjnodes/ImageBatchRepeatInterleaving",
        "comfy_alias:comfyui-kjnodes/ImageCropByMask",
        "comfy_alias:comfyui-kjnodes/ImageResizeKJ",
        "comfy_alias:comfyui-kjnodes/ImageResizeKJv2",
        "comfy_alias:comfyui-kjnodes/InsertImagesToBatchIndexed",
        "comfy_alias:comfyui-kjnodes/ReplaceImagesInBatch",
        "comfy_alias:comfyui-kjnodes/ReverseImageBatch",
        "comfy_alias:comfyui-kjnodes/ShuffleImageBatch",
        "comfy_alias:comfyui-kjnodes/TransitionImagesInBatch",
        "comfy_alias:comfyui-kjnodes/TransitionImagesMulti",
        "comfy_alias:comfyui_essentials/ImageBatchMultiple+",
        "comfy_alias:comfyui_essentials/ImageCrop+",
        "comfy_alias:comfyui_essentials/ImageExpandBatch+",
        "comfy_alias:comfyui_essentials/ImageFromBatch+",
        "comfy_alias:comfyui_essentials/ImageListToBatch+",
        "comfy_alias:comfyui_essentials/ImageResize+",
        "comfy_alias:comfyui_essentials/MaskBatch+",
        "comfy_alias:comfyui_essentials/MaskBoundingBox+",
        "comfy_alias:comfyui_essentials/MaskExpandBatch+",
        "comfy_alias:comfyui_essentials/MaskFromBatch+",
        "comfy_alias:comfyui_essentials/MaskFromColor+",
        "comfy_alias:comfyui_essentials/TransitionMask+",
    }
)
FANOUT_CASE_EXCESS = {
    "comfy_alias:comfy-core/ResizeAndPadImage": 1,
    "comfy_alias:comfy-core/ImageFlip": 1,
    "comfy_alias:comfy-core/ImageScale": 1,
    "comfy_alias:comfy-core/Morphology": 6,
    "comfy_alias:comfy-core/ResizeImageMaskNode": 22,
    "comfy_alias:comfyui-kjnodes/CreateShapeMask": 2,
    "comfy_alias:comfyui-kjnodes/ImageResizeKJv2": 9,
    "comfy_alias:comfyui-kjnodes/InsertImagesToBatchIndexed": 1,
    "comfy_alias:comfyui_essentials/ImageResize+": 3,
}
NON_TRANSLATABLE_CARRIERS = {
    # Native alpha conversions and polarity annotation have no source node.
    "dinkster.image.alpha.premultiply",
    "dinkster.image.alpha.unpremultiply",
    "dinkster.mask.polarity",
    # First-party shader execution has no trusted ComfyUI node counterpart.
    "dinkster.image.glsl_shader",
    # Document editing, selectors and layered-file persistence are native capabilities.
    "dinkster.layers.edit",
    "dinkster.layers.flatten",
    "dinkster.layers.split",
    "dinkster.layers.load",
    "dinkster.layers.save",
    # Per-frame crop/region reinsertion has no matching contract in the pinned
    # ComfyUI alias registry.
    "dinkster.image.tracked_crop",
    "dinkster.image.tracked_uncrop",
    # Internal tiled-refine nodes lack single-node ComfyUI counterparts; the
    # compat prompt pass composes them inside Ultimate SD Upscale expansion.
    "dinkster.image.tile_refine_plan",
    "dinkster.mask.tile_blend",
    # Its ComfyUI counterpart splits across UpscaleModelLoader and
    # ImageUpscaleWithModel, so the compat prompt pass expands the pair.
    "dinkster.image.upscale_model",
    "dinkster.detection.make",
    "dinkster.detection.info",
    "dinkster.detection.filter",
    "dinkster.detection.sort",
    "dinkster.detection.to_masks",
    # Model-backed capability owner schemas; their comfy alias coverage
    # (Impact detector+SEGS chains, SAM preprocessors, bg-removal packs,
    # core SAM3/RTDETR) is tracked in #677.
    "dinkster.detection.detect",
    "dinkster.detection.segment",
    "dinkster.detection.segment_text",
    "dinkster.detection.track",
    "dinkster.image.matte",
    "dinkster.pose.export_json",
    "dinkster.pose.import_json",
    "dinkster.pose.render",
    "dinkster.preprocess.model_depth",
}


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def test_image_comfy_aliases_use_the_canonical_wire_contract() -> None:
    registry = _registry()
    assert registry["format"] == "dinkster-comfy-alias/1"
    assert set(registry) == {"format", "sourceSchemas", "records"}

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    assert len(source_schemas) == len({schema.node_type for schema in source_schemas})
    assert all(not schema.replacements for schema in source_schemas)
    assert [schema_to_wire(schema) for schema in source_schemas] == registry["sourceSchemas"]

    records = registry["records"]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    assert all(alias_id.startswith("comfy_alias:") for alias_id in ids)
    source_types = {schema.node_type for schema in source_schemas}
    assert {record["source"]["nodeType"] for record in records} == source_types

    for record in records:
        assert record["mappingKind"] == "op"
        assert record["source"]["nodeType"] == record["replacement"]["from"]
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        assert record["carrier"] in {case.to for case in rule.cases}


def test_dynamic_alias_inventory_is_65_records_and_82_source_cases() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    affected = {
        alias_id
        for alias_id, record in records.items()
        if any("slotVariants" in case for case in record["replacement"]["cases"])
    }
    assert affected == DYNAMIC_ALIAS_IDS
    materialized_cases = sum(
        len(records[alias_id]["replacement"]["cases"]) for alias_id in affected
    )
    assert len(affected) == 65
    assert materialized_cases == 128
    assert FANOUT_CASE_EXCESS.keys() <= affected
    assert materialized_cases - sum(FANOUT_CASE_EXCESS.values()) == 82
    assert "comfy_alias:comfyui-kjnodes/ImageResizeKJv2" in records


def test_image_comfy_aliases_preserve_executable_source_domains() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    records = {record["id"]: record for record in registry["records"]}

    assert "comfy.comfyui-kjnodes.CreateGradientMask" not in source_schemas
    assert "comfy_alias:comfyui-kjnodes/CreateGradientMask" not in records

    bbox_schema = source_schemas["comfy.comfyui-kjnodes.BboxVisualize"]
    bbox_input = next(item for item in bbox_schema["interface"] if item["id"] == "bboxes")
    assert bbox_input["type"] == {
        "kind": "union",
        "types": ["comfy.BBOX", "comfy.BOUNDING_BOX"],
    }
    bbox_inputs = records["comfy_alias:comfyui-kjnodes/BboxVisualize"]["replacement"]["cases"][0][
        "inputs"
    ]
    assert bbox_inputs["coordinate_format"] == {"kind": "copy", "input": "bbox_format"}
    assert bbox_inputs["batch_policy"] == {"kind": "constant", "value": "pairwise_truncate"}

    composite_inputs = records["comfy_alias:comfy-core/MaskComposite"]["replacement"]["cases"][0][
        "inputs"
    ]
    assert composite_inputs["batch_policy"] == {"kind": "constant", "value": "source_singleton"}
    crop = records["comfy_alias:comfy-core/CropMask"]
    assert crop["confidence"]["tier"] == "parametric"
    assert "no source overlap" in crop["replacement"]["note"]


def test_image_rotate_alias_uses_literal_dynamic_choices() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    cases = records["comfy_alias:comfy-core/ImageRotate"]["replacement"]["cases"]
    assert len(cases) == 4
    assert "when" not in cases[-1]
    assert all(case["slotVariants"] == {"operation": "rotate_90"} for case in cases)
    assert cases[-1]["inputs"]["operation.steps"] == {"kind": "constant", "value": 3}


def test_mtb_bbox_alias_matches_pinned_static_reference() -> None:
    revision = "b35b5d8a17c0d59e80a8b3627b679c2c1003d04f"
    inputs = {"x": 7, "y": 11, "width": 13, "height": 17}
    expected_bbox = {"x": 7, "y": 11, "width": 13, "height": 17}
    records = {record["id"]: record for record in _registry()["records"]}
    record = records["comfy_alias:comfy-mtb/BBox (mtb)"]
    assert record["source"]["revision"] == revision
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        name: {"kind": "copy", "input": name} for name in ("x", "y", "width", "height")
    }
    region = cast(
        "Region",
        MakeRegion.execute(
            x=inputs["x"],
            y=inputs["y"],
            width=inputs["width"],
            height=inputs["height"],
        )["region"],
    )
    assert region.to_record() == expected_bbox


def test_batch_layout_aliases_declare_consolidated_families() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    expected = {
        "comfy_alias:comfy-core/ImageBatch",
        "comfy_alias:comfy-core/ImageFromBatch",
        "comfy_alias:comfy-core/ImageStitch",
        "comfy_alias:comfy-core/RebatchImages",
        "comfy_alias:comfy-core/RepeatImageBatch",
        "comfy_alias:comfyui-kjnodes/ImageBatchJoinWithTransition",
        "comfy_alias:comfyui-kjnodes/ImageBatchMulti",
        "comfy_alias:comfyui-kjnodes/ImageBatchRepeatInterleaving",
        "comfy_alias:comfyui-kjnodes/ImageGridComposite2x2",
        "comfy_alias:comfyui-kjnodes/ImageGridComposite3x3",
        "comfy_alias:comfyui-kjnodes/ImageGridtoBatch",
        "comfy_alias:comfyui-kjnodes/InsertImagesToBatchIndexed",
        "comfy_alias:comfyui-kjnodes/ReplaceImagesInBatch",
        "comfy_alias:comfyui-kjnodes/ReverseImageBatch",
        "comfy_alias:comfyui-kjnodes/ShuffleImageBatch",
        "comfy_alias:comfyui-kjnodes/TransitionImagesInBatch",
        "comfy_alias:comfyui-kjnodes/TransitionImagesMulti",
        "comfy_alias:comfyui_essentials/ImageBatchMultiple+",
        "comfy_alias:comfyui_essentials/ImageBatchToList+",
        "comfy_alias:comfyui_essentials/ImageExpandBatch+",
        "comfy_alias:comfyui_essentials/ImageFromBatch+",
        "comfy_alias:comfyui_essentials/ImageListToBatch+",
        "comfy_alias:comfyui_essentials/ImageTile+",
        "comfy_alias:comfyui_essentials/ImageUntile+",
        "comfy_alias:comfyui_essentials/MaskBatch+",
        "comfy_alias:comfyui_essentials/MaskExpandBatch+",
        "comfy_alias:comfyui_essentials/MaskFromBatch+",
    }
    assert expected <= records.keys()

    family = records["comfy_alias:comfyui_essentials/ImageBatchMultiple+"]["replacement"]["cases"][
        0
    ]["inputFamilies"]["images"]
    assert family["kind"] == "members"
    assert [member["inputs"]["value"]["input"] for member in family["members"]] == [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]

    transition = records["comfy_alias:comfyui-kjnodes/TransitionImagesMulti"]["replacement"][
        "cases"
    ][0]
    assert transition["slotVariants"]["mode"] == "between_inputs"
    assert transition["inputs"]["transition"]["transform"]["map"]["horizontal slide"] == (
        "horizontal_slide"
    )


def test_batch_aliases_preserve_dynamic_list_mask_and_default_contracts() -> None:
    registry = _registry()
    source_schemas = {
        schema["nodeType"]: schema
        for schema in cast("list[dict[str, Any]]", registry["sourceSchemas"])
    }
    records = {record["id"]: record for record in registry["records"]}

    for node_class in ("ImageBatchMulti", "TransitionImagesMulti"):
        schema = source_schemas[f"comfy.comfyui-kjnodes.{node_class}"]
        family = next(item for item in schema["interface"] if item["role"] == "inputFamily")
        assert (family["memberPrefix"], family["minMembers"], family["maxMembers"]) == (
            "image_",
            1,
            1000,
        )
        case = records[f"comfy_alias:comfyui-kjnodes/{node_class}"]["replacement"]["cases"][0]
        assert case["inputs"]["input_count"] == {"kind": "copy", "input": "inputcount"}
        assert case["inputFamilies"]["images"] == {
            "kind": "copy",
            "sourceFamily": "images",
            "inputs": {"value": {"kind": "copy", "input": "value"}},
        }

    essentials_list_input = source_schemas["comfy.comfyui_essentials.ImageListToBatch+"][
        "interface"
    ][0]["type"]
    essentials_list_output = source_schemas["comfy.comfyui_essentials.ImageBatchToList+"][
        "interface"
    ][1]["type"]
    assert essentials_list_input["kind"] == essentials_list_output["kind"] == "list"
    assert essentials_list_input["element"] == essentials_list_output["element"]

    multiple_schema = source_schemas["comfy.comfyui_essentials.ImageBatchMultiple+"]
    method = next(item for item in multiple_schema["interface"] if item["id"] == "method")
    assert method["default"] == "lanczos"
    within_schema = source_schemas["comfy.comfyui-kjnodes.TransitionImagesInBatch"]
    frames = next(
        item for item in within_schema["interface"] if item["id"] == "transitioning_frames"
    )
    assert frames["default"] == 1
    assert frames["widget"]["min"] == 0

    rebatch = records["comfy_alias:comfy-core/RebatchImages"]["replacement"]["cases"][0]
    assert rebatch["nodes"]["batch_size"] == {
        "type": "std.list.element",
        "values": {"index": 0},
    }
    assert rebatch["inputs"]["batch_size:list"] == {
        "kind": "copy",
        "input": "batch_size",
    }
    assert rebatch["links"] == [{"from": "batch_size:item", "to": "batch_size"}]

    repeat = records["comfy_alias:comfyui-kjnodes/ImageBatchRepeatInterleaving"]["replacement"][
        "cases"
    ][0]
    assert repeat["inputs"]["mask"] == {"kind": "copy", "input": "mask"}
    assert repeat["inputs"]["operation.generate_repeat_marker"] == {
        "kind": "constant",
        "value": True,
    }
    assert repeat["outputs"] == {"image": "image", "mask": "mask"}

    replace_cases = records["comfy_alias:comfyui-kjnodes/ReplaceImagesInBatch"]["replacement"][
        "cases"
    ]
    assert len(replace_cases) == 4
    assert (
        replace_cases[0]["outputs"]
        == replace_cases[1]["outputs"]
        == {
            "image": "image",
            "mask": "mask",
        }
    )
    assert replace_cases[2]["to"] == "dinkster.mask.batch.combine"
    assert replace_cases[2]["nodes"]["image"]["type"] == "dinkster.image.generate"
    assert replace_cases[3]["to"] == "dinkster.image.generate"
    assert replace_cases[3]["nodes"]["mask"]["type"] == "dinkster.mask.make"


def test_geometry_aliases_preserve_modes_outputs_and_refusals() -> None:
    registry = _registry()
    source_schemas = {
        schema["nodeType"]: schema
        for schema in cast("list[dict[str, Any]]", registry["sourceSchemas"])
    }
    records = {record["id"]: record for record in registry["records"]}
    source_contracts = {
        "comfy.comfyui-kjnodes.ImageResizeKJ": (
            [
                "image",
                "width",
                "height",
                "upscale_method",
                "keep_proportion",
                "divisible_by",
                "get_image_size",
                "crop",
            ],
            ["IMAGE", "width", "height"],
        ),
        "comfy.comfyui-kjnodes.ImageResizeKJv2": (
            [
                "image",
                "width",
                "height",
                "upscale_method",
                "keep_proportion",
                "pad_color",
                "crop_position",
                "divisible_by",
                "mask",
                "device",
            ],
            ["IMAGE", "width", "height", "mask"],
        ),
        "comfy.comfyui-kjnodes.GetImageSizeAndCount": (
            ["image"],
            ["image", "width", "height", "count"],
        ),
        "comfy.comfyui-kjnodes.ImageCropByMask": (["image", "mask"], ["image"]),
        "comfy.comfyui_essentials.ImageResize+": (
            ["image", "width", "height", "interpolation", "method", "condition", "multiple_of"],
            ["IMAGE", "width", "height"],
        ),
        "comfy.comfyui_essentials.ImageCrop+": (
            ["image", "width", "height", "position", "x_offset", "y_offset"],
            ["IMAGE", "x", "y"],
        ),
        "comfy.comfyui_essentials.GetImageSize+": (
            ["image"],
            ["width", "height", "count"],
        ),
        "comfy.comfyui_essentials.ImageRemoveAlpha+": (["image"], ["image"]),
        "comfy.comfyui_essentials.MaskBoundingBox+": (
            ["mask", "padding", "blur", "image_optional"],
            ["MASK", "IMAGE", "x", "y", "width", "height"],
        ),
    }
    for node_type, (expected_inputs, expected_outputs) in source_contracts.items():
        interface = source_schemas[node_type]["interface"]
        assert [item["id"] for item in interface if item["role"] == "input"] == expected_inputs
        assert [item["id"] for item in interface if item["role"] == "output"] == expected_outputs
        record = next(
            record for record in records.values() if record["source"]["nodeType"] == node_type
        )
        for case in record["replacement"]["cases"]:
            assert set(case["outputs"].values()) == set(expected_outputs)

    kj_v1 = records["comfy_alias:comfyui-kjnodes/ImageResizeKJ"]["replacement"]["cases"]
    assert len(kj_v1) == 8
    assert kj_v1[0]["when"] == {
        "kind": "all",
        "of": [
            {"kind": "inputConnected", "input": "get_image_size"},
            {"kind": "valueEquals", "input": "crop", "value": "center"},
        ],
    }
    assert kj_v1[1]["when"]["of"][1]["value"] == 0
    assert all(
        case["inputs"]["target.reference"] == {"kind": "copy", "input": "get_image_size"}
        for case in kj_v1[:3]
    )
    assert kj_v1[3]["when"] == {
        "kind": "all",
        "of": [
            {"kind": "valueEquals", "input": "keep_proportion", "value": True},
            {
                "kind": "any",
                "of": [
                    {"kind": "valueEquals", "input": "crop", "value": "center"},
                    {"kind": "valueEquals", "input": "crop", "value": 0},
                ],
            },
        ],
    }
    assert kj_v1[3]["inputs"]["mode.mode_anchor"] == {
        "kind": "constant",
        "value": "center",
    }
    assert kj_v1[4]["when"] == {
        "kind": "valueEquals",
        "input": "keep_proportion",
        "value": True,
    }
    assert [case["slotVariants"]["mode"] for case in kj_v1[3:]] == [
        "fill",
        "fit",
        "fill",
        "fill",
        "stretch",
    ]
    assert all(case["slotVariants"]["divisibility"] == "crop" for case in kj_v1)
    assert (
        "center-fit uses centered fill"
        in records["comfy_alias:comfyui-kjnodes/ImageResizeKJ"]["replacement"]["note"]
    )

    kj_v2_record = records["comfy_alias:comfyui-kjnodes/ImageResizeKJv2"]
    kj_v2 = kj_v2_record["replacement"]["cases"]
    assert [case["slotVariants"]["mode"] for case in kj_v2] == [
        "fit",
        "pad",
        "pad",
        "pad",
        "fill",
        "pad",
        "stretch",
        "stretch",
        "stretch",
        "stretch",
        "stretch",
    ]
    assert len(kj_v2) == 11
    assert [case["slotVariants"].get("mode.mode_padding") for case in kj_v2[:6]] == [
        None,
        "constant",
        "edge_average",
        "edge_pixel",
        None,
        "blurred_background",
    ]
    assert all(
        case["inputs"]["interpolation"]["transform"]["map"]
        == {
            "nearest-exact": "nearest-exact",
            "bilinear": "bilinear",
            "area": "area",
            "bicubic": "bicubic",
            "lanczos": "lanczos",
        }
        for case in kj_v2
    )
    assert all(
        case["inputs"]["mask"] == {"kind": "copy", "input": "mask"}
        and case["outputs"]["mask"] == "mask"
        for case in kj_v2
    )
    assert "mode.mode_padding.pad_color" in kj_v2[1]["inputs"]
    assert all("fallback_mask" not in case["nodes"] for case in kj_v2)
    expected_helper_nodes = (
        {"info", "pixels"},
        {"info", "pixels", "height"},
        {"info", "pixels", "width"},
        {"info", "pixels", "width", "height"},
    )
    for case, expected_nodes in zip(kj_v2[6:10], expected_helper_nodes, strict=True):
        assert case["nodes"]["pixels"] == {
            "type": "dinkster.math.expression",
            "values": {"expression": "a * b / 1048576"},
        }
        assert case["slotVariants"]["target"] == "total_pixels"
        assert case["inputFamilies"]["pixels:values"]["kind"] == "members"
        assert [
            member["suffix"] for member in case["inputFamilies"]["pixels:values"]["members"]
        ] == ["a", "b"]
        assert {"from": "pixels:float", "to": "target.megapixels"} in case["links"]
        assert set(case["nodes"]) == expected_nodes
    assert "No-mask unpadded output is absent" in kj_v2_record["replacement"]["note"]
    assert kj_v2_record["confidence"]["tolerances"] == [
        {"metric": "max_abs", "operator": "<=", "value": 2e-6}
    ]

    essentials = records["comfy_alias:comfyui_essentials/ImageResize+"]["replacement"]["cases"]
    assert [case["slotVariants"]["mode"] for case in essentials] == [
        "fit",
        "fill",
        "pad",
        "stretch",
    ]
    assert all(
        case["inputs"]["apply"]["transform"]["map"]
        == {
            "always": "always",
            "downscale if bigger": "only_if_bigger",
            "upscale if smaller": "only_if_smaller",
            "if bigger area": "only_if_bigger_area",
            "if smaller area": "only_if_smaller_area",
        }
        for case in essentials
    )

    kj_size = records["comfy_alias:comfyui-kjnodes/GetImageSizeAndCount"]["replacement"]["cases"][0]
    assert kj_size["to"] == "dinkster.image.crop"
    assert kj_size["nodes"]["info"]["type"] == "dinkster.image.info"
    assert kj_size["slotVariants"] == {"source": "coordinates", "outside": "clip"}
    assert kj_size["links"] == [{"from": "image", "to": "info:image"}]

    crop = records["comfy_alias:comfyui_essentials/ImageCrop+"]
    assert crop["confidence"]["tier"] == "parametric"
    crop_case = crop["replacement"]["cases"][0]
    assert crop_case["to"] == "dinkster.region.info"
    assert crop_case["nodes"]["crop"]["type"] == "dinkster.image.crop"
    assert crop_case["slotVariants"] == {"crop:source": "coordinates", "crop:outside": "clip"}
    assert crop_case["outputs"] == {
        "crop:image": "IMAGE",
        "integer_x": "x",
        "integer_y": "y",
    }

    assert (
        records["comfy_alias:comfyui_essentials/GetImageSize+"]["replacement"]["cases"][0]["to"]
        == "dinkster.image.info"
    )
    assert (
        records["comfy_alias:comfyui_essentials/ImageRemoveAlpha+"]["replacement"]["cases"][0]["to"]
        == "dinkster.image.channels.split"
    )
    assert records["comfy_alias:comfyui_essentials/ImageRemoveAlpha+"]["replacement"]["cases"][0][
        "inputs"
    ]["single_channel_image"] == {"kind": "constant", "value": "preserve"}

    bbox_cases = records["comfy_alias:comfyui_essentials/MaskBoundingBox+"]["replacement"]["cases"]
    assert bbox_cases[0]["when"] == {"kind": "inputConnected", "input": "image_optional"}
    assert bbox_cases[0]["inputs"]["source.mask_blur"] == {"kind": "copy", "input": "blur"}
    assert bbox_cases[1]["inputs"]["crop:source.mask_blur"] == {
        "kind": "copy",
        "input": "blur",
    }
    assert bbox_cases[1]["nodes"]["image"] == {"type": "dinkster.mask.to_image"}
    assert bbox_cases[1]["slotVariants"]["image:channels"] == "rgb"


def test_resize_image_mask_alias_covers_every_dynamic_selection() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    record = next(
        record
        for record in registry["records"]
        if record["source"]["nodeClass"] == "ResizeImageMaskNode"
    )

    assert record["source"]["revision"] == "b78cec87"
    assert record["confidence"] == {
        "tier": "equivalent",
        "evidence": [
            "tests/test_image_comfy_aliases.py::test_resize_image_mask_alias_covers_every_dynamic_selection",
            "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
        ],
        "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
    }
    source = source_schemas["comfy.ResizeImageMaskNode"]
    assert [item["id"] for item in source["interface"] if item["role"] == "input"] == [
        "input",
        "scale_method",
    ]
    dynamic = next(item for item in source["interface"] if item["role"] == "dynamicCombo")
    assert dynamic["id"] == "resize_type"
    assert [option["key"] for option in dynamic["options"]] == [
        "scale dimensions",
        "scale by multiplier",
        "scale longer dimension",
        "scale shorter dimension",
        "scale width",
        "scale height",
        "scale total pixels",
        "match size",
        "scale to multiple",
    ]
    assert source["interface"][-1]["type"] == {
        "kind": "variable",
        "templateId": "input_type",
        "allowed": ["comfy.IMAGE", "comfy.MASK"],
    }

    cases = record["replacement"]["cases"]
    assert len(cases) == 23
    expected_groups = [
        ("dimensions", "stretch", "target.width", "resize_type.width"),
        ("dimensions", "fill", "target.width", "resize_type.width"),
        ("factor", "stretch", "target.factor", "resize_type.multiplier"),
        ("longest", "stretch", "target.size", "resize_type.longer_size"),
        ("shortest", "stretch", "target.size", "resize_type.shorter_size"),
        ("width", "stretch", "target.width", "resize_type.width"),
        ("height", "stretch", "target.height", "resize_type.height"),
        ("total_pixels", "stretch", "target.megapixels", "resize_type.megapixels"),
        ("match", "stretch", "target.reference", "resize_type.match"),
        ("match", "fill", "target.reference", "resize_type.match"),
        ("multiple_cover", "stretch", "target.multiple_of", "resize_type.multiple"),
    ]
    for group, (target, mode, active_target, active_source) in zip(
        (cases[index : index + 2] for index in range(0, 22, 2)),
        expected_groups,
        strict=True,
    ):
        assert all(case["to"] == "dinkster.image.resize" for case in group)
        assert all(case["slotVariants"]["target"] == target for case in group)
        assert all(case["slotVariants"]["mode"] == mode for case in group)
        assert all(case["inputs"]["image"] == {"kind": "copy", "input": "input"} for case in group)
        assert all(
            case["inputs"][active_target] == {"kind": "copy", "input": active_source}
            for case in group
        )
        assert group[0]["inputs"]["interpolation"] == {
            "kind": "copy",
            "input": "scale_method",
        }
        assert group[1]["inputs"]["interpolation"] == {"kind": "constant", "value": "area"}

    dimensions = cases[:4]
    match = cases[16:20]
    assert all("inputConnected" in json.dumps(case["when"]) for case in dimensions + match)
    assert all(case["outputs"] == {"image": "resized"} for case in cases)
    refusal = cases[-1]
    assert "when" not in refusal
    assert refusal["inputs"]["target.factor"]["transform"] == {
        "kind": "enumRename",
        "map": {},
    }


def test_every_translatable_image_node_carries_an_executable_comfy_alias() -> None:
    registry = _registry()
    native_schemas = {node.schema().node_type: node.schema() for node in IMAGE_NODES}
    records_by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in registry["records"]:
        records_by_carrier[record["carrier"]].append(record)
    assert set(records_by_carrier) == set(native_schemas) - NON_TRANSLATABLE_CARRIERS

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    schemas = {schema.node_type: schema for schema in source_schemas}
    for carrier, schema in native_schemas.items():
        rules = tuple(
            rule_from_wire(record["replacement"]) for record in records_by_carrier[carrier]
        )
        schemas[carrier] = replace(schema, replacements=rules)
    assert validate_replacement_references(schemas) == ()
    assert all(
        not alias.startswith("comfy_alias:")
        for schema in native_schemas.values()
        for alias in schema.aliases
    )


def test_image_comfy_alias_confidence_has_pinned_evidence() -> None:
    registry = _registry()
    allowed_tiers = {"exact", "parametric", "equivalent", "grouped"}
    revisions = {
        "comfy-core": "b78cec87",
        "comfy-mtb": "b35b5d8a17c0d59e80a8b3627b679c2c1003d04f",
        "comfyui_controlnet_aux": "59b1fc411ede8623b2997855b8018f0b3b6cf49f",
        "comfyui-kjnodes": "827fe6ee0ed7348d8daa988ed852bedf1272380c",
        "comfyui_essentials": "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9",
    }
    for record in registry["records"]:
        source = record["source"]
        expected_revision = (
            "f00bfd610cb001381603669e2cc01160ae37aaf3"
            if source["nodeClass"] in {"ImageCompositor", "AddLayer", "LayersFromBoundingBoxes"}
            else revisions[source["pack"]]
        )
        assert source["revision"] == expected_revision
        confidence = record["confidence"]
        assert confidence["tier"] in allowed_tiers
        assert confidence["evidence"]
        for selector in confidence["evidence"]:
            path_text, _, test_name = selector.partition("::")
            path = ROOT / path_text
            assert path.is_file()
            assert test_name and f"def {test_name}(" in path.read_text(encoding="utf-8")
        tolerances = confidence.get("tolerances")
        if confidence["tier"] == "equivalent":
            assert tolerances
        if confidence["tier"] == "exact":
            assert tolerances is None
        for tolerance in tolerances or ():
            assert tolerance["operator"] in {"<=", ">="}
            assert math.isfinite(tolerance["value"])


def test_image_pack_bundles_alias_registry_and_required_helpers() -> None:
    configuration = cast(
        "dict[str, Any]",
        tomllib.loads(
            (ROOT / "packages" / "dinkster-nodes-image" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        ),
    )
    force_include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["comfy-aliases.json"] == ("dinkster_nodes_image_pack/comfy-aliases.json")
    manifest = tomllib.loads(
        (ROOT / "packages" / "dinkster-nodes-image" / "dinkster-pack.toml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["pack"]["dependencies"]["dinkster-nodes-foundation"] == ">=0.0.1,<1"


def test_new_image_aliases_execute() -> None:
    import numpy as np
    from dinkster_api.v1 import ABSENT, annotate_image, media_semantics
    from dinkster_nodes_image import ImageBatchCombine, ImageCompare, ImageResize

    records = {record["source"]["nodeClass"]: record for record in _registry()["records"]}
    rgb = np.full((1, 3, 5, 3), 0.25, np.float32)
    rgba = np.full((2, 6, 10, 4), 0.5, np.float32)
    batch = ImageBatchCombine.execute(
        images={"0.value": rgb, "1.value": rgba},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    assert np.asarray(batch).shape == (3, 3, 5, 4)
    np.testing.assert_array_equal(np.asarray(batch)[0, ..., 3], 1.0)
    family = records["BatchImagesNode"]["replacement"]["cases"][0]["inputFamilies"]["images"]
    assert family["inputs"]["value"] == {"kind": "copy", "input": "image"}
    for case, pad in zip(
        records["ResizeAndPadImage"]["replacement"]["cases"], (0.0, 1.0), strict=True
    ):
        inputs = case["inputs"]
        values = {
            name: value["value"] for name, value in inputs.items() if value["kind"] == "constant"
        }
        values.update(
            {"image": rgb, "target.width": 4, "target.height": 4, "interpolation": "nearest-exact"}
        )
        result = ImageResize.execute(**values, **case["slotVariants"])["image"]
        assert np.asarray(result).shape == (1, 4, 4, 3)
        np.testing.assert_array_equal(np.asarray(result)[:, (0, 3)], pad)
        np.testing.assert_array_equal(np.asarray(result)[:, 1:3], 0.25)
    compare = records["ImageCompare"]
    assert compare["carrier"] == "dinkster.image.compare"
    assert len(compare["replacement"]["cases"]) == 1
    for a, b in ((rgb, rgba), (rgb, None), (None, rgba), (None, None)):
        result = ImageCompare.execute(image_a=a, image_b=b)
        for name, expected in (("image_a", a), ("image_b", b)):
            if expected is None:
                assert result[name] is ABSENT
            else:
                np.testing.assert_array_equal(result[name], expected)
    premultiplied = annotate_image(rgba, alpha="premultiplied")
    preview = ImageCompare.execute(image_a=premultiplied, image_b=np.empty((0, 2, 3, 4)))
    assert media_semantics(preview["image_a"]) == media_semantics(premultiplied)
    np.testing.assert_array_equal(preview["image_a"], premultiplied)
    assert preview["image_b"] is ABSENT


def test_media_alias_records_keep_the_current_wire_contract() -> None:
    from dinkster_schema import comfy_alias_registry_from_wire, comfy_alias_registry_to_wire

    fields = {"id", "mappingKind", "carrier", "source", "replacement", "confidence", "family"}
    for pack in ("dinkster-nodes-image", "dinkster-nodes-media-io"):
        registry = json.loads((ROOT / "packages" / pack / "comfy-aliases.json").read_text())
        decoded = comfy_alias_registry_from_wire(registry)
        assert comfy_alias_registry_to_wire(decoded) == registry
        records = cast("list[dict[str, object]]", registry["records"])
        assert all(set(record) <= fields for record in records)


def test_new_image_aliases_match_pinned_goldens() -> None:
    import numpy as np
    from dinkster_nodes_image import ImageBatchCombine, ImageResize

    fixture = json.loads((ROOT / "tests/goldens/image_aliases_b78cec87.json").read_text())
    assert fixture["baseline"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"

    def array(value: dict[str, Any]) -> np.ndarray:
        return np.array(value["values"], np.float32).reshape(value["shape"])

    batch = ImageBatchCombine.execute(
        images={"1": array(fixture["rgb"]), "2": array(fixture["rgba"])},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    np.testing.assert_allclose(np.asarray(batch), array(fixture["batch"]), rtol=0, atol=1e-7)
    for case in fixture["resize"]:
        result = ImageResize.execute(
            image=array(fixture["rgb"]),
            width=4,
            height=4,
            mode="pad",
            fit_rounding="floor",
            interpolation=case["method"],
            pad_value=float(case["color"] == "white"),
        )["image"]
        if case["method"] in ("nearest-exact", "lanczos"):
            np.testing.assert_array_equal(result, array(case["image"]))
        else:
            np.testing.assert_allclose(np.asarray(result), array(case["image"]), rtol=0, atol=1e-7)
