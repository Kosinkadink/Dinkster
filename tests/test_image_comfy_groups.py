from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_image import IMAGE_NODES
from dinkster_schema import (
    comfy_group_registry_from_wire,
    comfy_group_registry_problems,
    comfy_group_registry_to_wire,
)

from tools.gen_image_comfy_groups import build_registry

ROOT = Path(__file__).parent.parent
GROUP_PATH = ROOT / "packages" / "dinkster-nodes-image" / "comfy-groups.json"


def _registry_wire() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(GROUP_PATH.read_text(encoding="utf-8")))


def test_image_comfy_groups_are_canonical_and_reference_native_schemas() -> None:
    wire = _registry_wire()
    registry = comfy_group_registry_from_wire(wire)
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_NODES}

    assert comfy_group_registry_to_wire(registry) == wire
    assert comfy_group_registry_to_wire(build_registry()) == wire
    assert comfy_group_registry_problems(registry, schemas) == ()
    assert {record.source.revision for record in registry.records} == {"c67885b1"}


def test_image_comfy_groups_are_limited_to_truthful_core_chains() -> None:
    records = _registry_wire()["records"]
    assert {record["id"] for record in records} == {
        "comfy_group:comfy-core/remove-background-birefnet",
        "comfy_group:comfy-core/rtdetr-detect-fp16",
        "comfy_group:comfy-core/sam3-text-detection",
        "comfy_group:comfy-core/sam3-video-track-initial-mask",
    }
    assert {record["source"]["pack"] for record in records} == {"comfy-core"}


def test_rtdetr_groups_preserve_detection_parameters() -> None:
    records = {record["id"]: record for record in _registry_wire()["records"]}
    assert "comfy_group:comfy-core/rtdetr-detect-fp32" not in records
    record = records["comfy_group:comfy-core/rtdetr-detect-fp16"]
    assert record["carrier"] == "dinkster.detection.detect"
    assert record["pattern"]["constants"] == {
        "loader:unet_name": "rt_detr_v4-x-hgnet_fp16.safetensors",
        "loader:weight_dtype": "default",
    }
    cases = record["replacement"]["cases"]
    assert cases[0]["when"] == {
        "kind": "valueEquals",
        "input": "class_name",
        "value": "all",
    }
    assert cases[0]["inputs"]["prompt"] == {"kind": "constant", "value": ""}
    assert cases[1]["inputs"]["prompt"] == {
        "kind": "copy",
        "input": "class_name",
    }
    for case in cases:
        assert case["inputs"]["min_score"] == {"kind": "copy", "input": "threshold"}
        assert case["inputs"]["max_results"] == {
            "kind": "copy",
            "input": "max_detections",
        }
        assert case["inputs"]["result_limit_mode"] == {
            "kind": "constant",
            "value": "slice-stop",
        }
        assert case["inputs"]["provider"] == {
            "kind": "constant",
            "value": "dinkster-vision-rtdetr",
        }
        assert "nodes" not in case
        assert "links" not in case
        assert case["outputs"] == {"detections": "bboxes"}


def test_sam3_text_detection_preserves_supported_inputs() -> None:
    record = next(
        record
        for record in _registry_wire()["records"]
        if record["id"] == "comfy_group:comfy-core/sam3-text-detection"
    )
    assert record["carrier"] == "dinkster.detection.detect"
    assert record["pattern"]["constants"] == {
        "loader:ckpt_name": "sam3.1_multiplex_fp16.safetensors",
        "detect:refine_iterations": 2,
        "detect:individual_masks": False,
    }
    assert record["pattern"]["disconnected"] == [
        "detect:bboxes",
        "detect:positive_coords",
        "detect:negative_coords",
    ]
    case = record["replacement"]["cases"][0]
    assert "when" not in case
    assert case["inputs"] == {
        "image": {"kind": "copy", "input": "image"},
        "prompt": {"kind": "copy", "input": "text"},
        "prompt_mode": {"kind": "constant", "value": "literal"},
        "min_score": {"kind": "copy", "input": "threshold"},
        "max_results": {"kind": "constant", "value": 1},
        "provider": {"kind": "constant", "value": "dinkster-vision-sam31"},
    }
    assert case["outputs"] == {"detections": "bboxes"}


def test_sam3_video_tracking_preserves_initial_masks() -> None:
    record = next(
        record
        for record in _registry_wire()["records"]
        if record["id"] == "comfy_group:comfy-core/sam3-video-track-initial-mask"
    )
    assert record["carrier"] == "dinkster.detection.track"
    assert record["pattern"]["constants"] == {
        "loader:ckpt_name": "sam3.1_multiplex_fp16.safetensors",
        "track:detection_threshold": 0.5,
        "track:max_objects": 4,
        "track:detect_interval": 1,
        "to_mask:object_indices": "",
    }
    assert record["pattern"]["disconnected"] == ["track:conditioning"]
    case = record["replacement"]["cases"][0]
    assert "when" not in case
    assert case["inputs"] == {
        "image": {"kind": "copy", "input": "images"},
        "initial_masks": {"kind": "copy", "input": "initial_mask"},
        "provider": {"kind": "constant", "value": "dinkster-vision-sam31"},
    }
    assert case["outputs"] == {"combined": "masks"}


def test_background_group_selects_birefnet() -> None:
    record = next(
        record
        for record in _registry_wire()["records"]
        if record["id"] == "comfy_group:comfy-core/remove-background-birefnet"
    )
    assert record["carrier"] == "dinkster.image.matte"
    assert record["pattern"]["constants"] == {"loader:bg_removal_name": "birefnet.safetensors"}
    case = record["replacement"]["cases"][0]
    assert case["inputs"]["image"] == {"kind": "copy", "input": "image"}
    assert case["inputs"]["provider"] == {
        "kind": "constant",
        "value": "dinkster-vision-birefnet",
    }
    assert case["outputs"] == {"mask": "mask"}


def test_image_pack_bundles_group_registry() -> None:
    configuration = tomllib.loads(
        (ROOT / "packages" / "dinkster-nodes-image" / "pyproject.toml").read_text(encoding="utf-8")
    )
    force_include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["comfy-groups.json"] == ("dinkster_nodes_image_pack/comfy-groups.json")
