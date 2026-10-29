"""The generation pack's maintained ComfyUI group translations stay canonical."""

from __future__ import annotations

import json
from pathlib import Path

from dinkster_nodes_generation import GENERATION_NODES
from dinkster_schema import (
    comfy_group_registry_from_wire,
    comfy_group_registry_problems,
    comfy_group_registry_to_wire,
)

ROOT = Path(__file__).parent.parent
GROUP_PATH = ROOT / "packages" / "dinkster-nodes-generation" / "comfy-groups.json"
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-generation" / "comfy-aliases.json"
IMPACT_REVISION = "429d0159ad429e64d2b3916e6e7be9c22d025c3c"


def _registry():  # noqa: ANN202
    return comfy_group_registry_from_wire(json.loads(GROUP_PATH.read_text(encoding="utf-8")))


def _ksampler_enum_maps() -> tuple[dict[str, str], dict[str, str]]:
    registry = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
    record = next(
        item for item in registry["records"] if item["id"] == "comfy_alias:comfy-core/KSampler"
    )
    inputs = record["replacement"]["cases"][0]["inputs"]
    return (
        inputs["sampler_name"]["transform"]["map"],
        inputs["scheduler"]["transform"]["map"],
    )


def test_impact_regional_single_region_group_is_valid() -> None:
    raw = json.loads(GROUP_PATH.read_text(encoding="utf-8"))
    registry = comfy_group_registry_from_wire(raw)
    schemas = {node.schema().node_type: node.schema() for node in GENERATION_NODES}

    assert comfy_group_registry_to_wire(registry) == raw
    assert comfy_group_registry_problems(registry, schemas) == ()
    assert len(registry.records) == 1
    record = registry.records[0]
    assert record.id == "comfy_group:comfyui-impact-pack/regional-sampler-single-region"
    assert record.carrier == "dinkster.compat.impact_regional_sampler"
    assert record.source.revision == IMPACT_REVISION
    assert record.confidence.tier == "grouped"
    assert tuple(node.source.node_class for _, node in record.pattern.nodes) == (
        "KSamplerAdvancedProvider",
        "KSamplerAdvancedProvider",
        "RegionalPrompt",
        "RegionalSampler",
    )


def test_impact_regional_group_fixes_the_supported_source_options() -> None:
    pattern = _registry().records[0].pattern
    assert dict(pattern.constants) == {
        "base_provider:sigma_factor": 1.0,
        "region_provider:sigma_factor": 1.0,
        "regional_prompt:variation_seed": 0,
        "regional_prompt:variation_strength": 0.0,
        "regional_prompt:variation_method": "linear",
        "regional_sampler:seed_2nd": 0,
        "regional_sampler:seed_2nd_mode": "ignore",
        "regional_sampler:additional_mode": "DISABLE",
        "regional_sampler:additional_sampler": "AUTO",
        "regional_sampler:additional_sigma_ratio": 0.3,
    }
    assert pattern.disconnected == (
        "base_provider:sampler_opt",
        "base_provider:scheduler_func_opt",
        "region_provider:sampler_opt",
        "region_provider:scheduler_func_opt",
    )
    assert tuple((edge.from_address, edge.to) for edge in pattern.edges) == (
        (
            "base_provider:_0_KSAMPLER_ADVANCED_",
            "regional_sampler:base_sampler",
        ),
        (
            "region_provider:_0_KSAMPLER_ADVANCED_",
            "regional_prompt:advanced_sampler",
        ),
        (
            "regional_prompt:_0_REGIONAL_PROMPTS_",
            "regional_sampler:regional_prompts",
        ),
    )


def test_impact_regional_group_reuses_core_sampler_and_scheduler_transforms() -> None:
    sampler_map, scheduler_map = _ksampler_enum_maps()
    inputs = dict(_registry().records[0].replacement.cases[0].inputs)

    for input_id in ("base_sampler_name", "region_sampler_name"):
        transform = inputs[input_id].transform
        assert transform is not None
        assert dict(transform.map) == sampler_map
    for input_id in ("base_scheduler", "region_scheduler"):
        transform = inputs[input_id].transform
        assert transform is not None
        assert dict(transform.map) == scheduler_map
