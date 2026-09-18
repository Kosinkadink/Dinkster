"""Exact Hunyuan Image component detection and recipe binding."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import FLOAT32, ComponentPlan
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_registry import DetectedComponents
from dinkster_inference.hunyuan_image_text import bind_hunyuan_image_text
from dinkster_inference.qwen_image_text import QWEN_IMAGE_TEXT_CONFIG, qwen_image_text_layout
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.t5_text import (
    BYT5_SMALL_GLYPH_CONFIG,
    T5TextDetectError,
    detect_t5_config,
    t5_layout,
)
from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe
from dinkster_inference.weights import TensorGeometry, WeightEntry


class Header:
    asset_digest = "blake3:" + "1" * 64
    asset_size = 42

    def __init__(self, layout: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.items()}

    def keys(self) -> tuple[str, ...]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.shapes[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}

    def read_float_scalar(self, key: str) -> float:
        raise AssertionError(f"planner must not read model data: {key}")


def test_byt5_small_is_exact_and_does_not_broaden_t5_detection() -> None:
    geometry = {
        key: TensorGeometry(shape, FLOAT32)
        for key, shape in t5_layout(BYT5_SMALL_GLYPH_CONFIG).items()
    }
    assert detect_t5_config(geometry) == BYT5_SMALL_GLYPH_CONFIG
    with pytest.raises(T5TextDetectError):
        detect_t5_config(
            {
                **geometry,
                "shared.weight": TensorGeometry((1511, 1472), FLOAT32),
            }
        )
    with pytest.raises(T5TextDetectError):
        detect_t5_config(
            {
                **geometry,
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (
                    TensorGeometry((32, 7), FLOAT32)
                ),
            }
        )


def test_registry_binds_both_orders_and_retains_every_source() -> None:
    registry = default_component_registry()
    qwen = Header(qwen_image_text_layout())
    byt5 = Header(t5_layout(BYT5_SMALL_GLYPH_CONFIG))
    detected = tuple(
        registry.detect(source, Path(name))
        for source, name in ((qwen, "qwen.safetensors"), (byt5, "byt5.safetensors"))
    )
    binding = resolve_text_recipe(detected, "hunyuan_image")
    reverse = resolve_text_recipe(tuple(reversed(detected)), "dinkster.text_hunyuan_image")
    assert binding.family_id == "dinkster.hunyuan_image"
    assert binding.composition_roles == ("qwen25_vl", "byt5_small")
    assert [(part.source_index, part.role) for part in binding.components] == [
        (0, "qwen25_vl"),
        (1, "byt5_small"),
    ]
    assert [(part.source_index, part.role) for part in reverse.components] == [
        (1, "qwen25_vl"),
        (0, "byt5_small"),
    ]
    qwen_facts = binding.components[0].plan.runtime_facts
    assert any(fact.startswith("qwen_tokenizer_source=https://") for fact in qwen_facts)
    assert sum("qwen_tokenizer_" in fact and "sha256=" in fact for fact in qwen_facts) == 3
    byt5_facts = binding.components[1].plan.runtime_facts
    assert any(fact.startswith("byt5_tokenizer_source=https://") for fact in byt5_facts)
    assert sum(fact.startswith("byt5_tokenizer_sha256=") for fact in byt5_facts) == 3
    refs = (
        WeightSourceRef("blake3:" + "1" * 64, "qwen", 42),
        WeightSourceRef("blake3:" + "2" * 64, "byt5", 42),
    )
    assert (
        binding.recipe(refs, "float32").runtime_identity
        != reverse.recipe(tuple(reversed(refs)), "float32").runtime_identity
    )
    with pytest.raises(UnresolvedTextRecipe, match="discard an input source"):
        bind_hunyuan_image_text((*detected, ()))
    with pytest.raises(UnresolvedTextRecipe, match="one unambiguous byt5_small"):
        bind_hunyuan_image_text(detected[:1])
    with pytest.raises(UnresolvedTextRecipe, match="one unambiguous qwen25_vl"):
        bind_hunyuan_image_text((detected[0], detected[0], detected[1]))


def test_recipe_rejects_wrong_qwen_and_byt5_variants() -> None:
    registry = default_component_registry()
    qwen_descriptor = registry.get("dinkster.hunyuan_image")
    classic_descriptor = registry.get("dinkster.classic_text")
    assert qwen_descriptor is not None and classic_descriptor is not None
    qwen = ComponentPlan("qwen25_vl", Path("qwen"), "wrong", {}, {}, {})
    byt5 = ComponentPlan("byt5_small", Path("byt5"), BYT5_SMALL_GLYPH_CONFIG, {}, {}, {})
    detected = (
        (DetectedComponents(qwen_descriptor, (("qwen25_vl", qwen),)),),
        (DetectedComponents(classic_descriptor, (("byt5_small", byt5),)),),
    )
    with pytest.raises(UnresolvedTextRecipe, match="Qwen2.5-VL-7B"):
        bind_hunyuan_image_text(detected)
    valid_qwen = replace(qwen, config=QWEN_IMAGE_TEXT_CONFIG)
    wrong_byt5 = replace(byt5, config=replace(BYT5_SMALL_GLYPH_CONFIG, vocab_size=1511))
    with pytest.raises(UnresolvedTextRecipe, match="exact ByT5-small"):
        bind_hunyuan_image_text(
            (
                (DetectedComponents(qwen_descriptor, (("qwen25_vl", valid_qwen),)),),
                (DetectedComponents(classic_descriptor, (("byt5_small", wrong_byt5),)),),
            )
        )
