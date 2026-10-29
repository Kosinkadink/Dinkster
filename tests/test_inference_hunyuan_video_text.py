"""Exact Llama admission, quant retention, and ordered Hunyuan identities."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import FLOAT8_E4M3, FLOAT32, ComponentPlan
from dinkster_inference.assembly import AssemblyError
from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG, clip_text_layout
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.hunyuan_video_text import bind_hunyuan_video_text
from dinkster_inference.llama3_text import (
    HUNYUAN_LLAMA3_CONFIG,
    detect_llama3_config,
    plan_llama3_component,
)
from dinkster_inference.qwen_text import (
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_CONFIG,
    QwenTextDetectError,
    detect_qwen_text_config,
    qwen_text_layout,
)
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe
from dinkster_inference.weights import TensorGeometry, WeightEntry


class Header:
    asset_digest = "blake3:" + "1" * 64
    asset_size = 42

    def __init__(
        self,
        layout: dict[str, tuple[int, ...]] | None = None,
        *,
        prefix: str = "model.",
        quantized: bool = False,
    ) -> None:
        selected = qwen_text_layout(HUNYUAN_LLAMA3_CONFIG) if layout is None else layout
        self.shapes = {
            prefix + key: TensorGeometry(shape, FLOAT32) for key, shape in selected.items()
        }
        self.extra: dict[str, str] = {}
        if quantized:
            layer = prefix + "layers.0.self_attn.q_proj"
            self.shapes[layer + ".weight"] = replace(
                self.shapes[layer + ".weight"], dtype=FLOAT8_E4M3
            )
            self.shapes[layer + ".weight_scale"] = TensorGeometry((), FLOAT32)
            self.extra["_quantization_metadata"] = json.dumps(
                {"layers": {layer: {"format": "float8_e4m3fn"}}}
            )

    def keys(self) -> tuple[str, ...]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.shapes[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return self.extra

    def read_float_scalar(self, key: str) -> float:
        raise AssertionError(f"planner must not read model data: {key}")


@pytest.mark.parametrize("prefix", ("", "model.", "llama.transformer.model."))
@pytest.mark.parametrize("quantized", (False, True))
def test_exact_llama_layout_and_component_quantization(prefix: str, quantized: bool) -> None:
    source = Header(prefix=prefix, quantized=quantized)
    plan = plan_llama3_component(source, Path("llama.safetensors"))
    assert plan.config == HUNYUAN_LLAMA3_CONFIG
    assert len(plan.keys) == 290
    assert plan.keys["embed_tokens.weight"] == prefix + "embed_tokens.weight"
    assert plan.identity_facts == (f"asset_digest={source.asset_digest}", "asset_size=42")
    if quantized:
        quant = plan.quant["layers.0.self_attn.q_proj"]
        assert quant.format == "float8_e4m3fn"
        assert quant.weight_scale == prefix + "layers.0.self_attn.q_proj.weight_scale"
    else:
        assert not plan.quant


def test_qwen_and_llama_signatures_cannot_collide() -> None:
    llama = Header(prefix="").shapes
    with pytest.raises(QwenTextDetectError):
        detect_qwen_text_config(llama)
    for config in (KLEIN_QWEN3_8B_CONFIG, MISTRAL3_24B_CONFIG):
        with pytest.raises(AssemblyError, match="exact Hunyuan"):
            detect_llama3_config(
                {
                    key: TensorGeometry(shape, FLOAT32)
                    for key, shape in qwen_text_layout(config).items()
                }
            )
    for key in ("layers.31.mlp.down_proj.weight", "norm.weight"):
        incomplete = dict(llama)
        del incomplete[key]
        with pytest.raises(AssemblyError):
            detect_llama3_config(incomplete)
    extra = {**llama, "layers.0.self_attn.q_norm.weight": TensorGeometry((128,), FLOAT32)}
    with pytest.raises(AssemblyError):
        detect_llama3_config(extra)


def test_production_registries_resolve_both_source_orders_and_reject_extra_sources() -> None:
    registry = default_component_registry()
    descriptor = registry.get("dinkster.hunyuan_video")
    assert descriptor is not None
    assert descriptor.requires_text_recipe
    assert descriptor.text_encoder_roles == ("llama",)
    llama_source = Header(prefix="llama.model.", quantized=True)
    clip_source = Header(clip_text_layout(CLIP_L_TEXT_CONFIG), prefix="clip_l.")
    detected = tuple(
        registry.detect(source, Path(name))
        for source, name in ((llama_source, "llama"), (clip_source, "clip"))
    )
    normal = resolve_text_recipe(detected, "hunyuan_video")
    reverse = resolve_text_recipe(tuple(reversed(detected)), "dinkster.text_hunyuan_video")
    assert normal.id == "dinkster.text_hunyuan_video"
    assert normal.family_id == "dinkster.hunyuan_video"
    assert [(part.source_index, part.role) for part in normal.components] == [
        (0, "llama"),
        (1, "clip_l"),
    ]
    assert [(part.source_index, part.role) for part in reverse.components] == [
        (1, "llama"),
        (0, "clip_l"),
    ]
    refs = (
        WeightSourceRef("blake3:" + "1" * 64, "llama", 42),
        WeightSourceRef("blake3:" + "2" * 64, "clip", 42),
    )
    before = normal.recipe(refs, "float32", embedding_binding_digest="a" * 64)
    after = reverse.recipe(tuple(reversed(refs)), "float32", embedding_binding_digest="a" * 64)
    assert before.runtime_identity != after.runtime_identity
    assert tuple(item.source for item in after.sources) == tuple(reversed(refs))
    assert normal.components[0].plan.quant
    assert normal.components[1].profile is not None
    assert not normal.components[1].profile.projected_pooled
    with pytest.raises(UnresolvedTextRecipe, match="discard an input source"):
        bind_hunyuan_video_text((*detected, ()))
    with pytest.raises(UnresolvedTextRecipe, match="one unambiguous llama"):
        bind_hunyuan_video_text((detected[0], detected[0], detected[1]))


def test_recipe_identity_retains_llama_policy() -> None:
    llama = plan_llama3_component(Header(quantized=True), Path("llama.safetensors"))
    clip = ComponentPlan("clip_l", Path("clip.safetensors"), CLIP_L_TEXT_CONFIG, {}, {}, {})
    registry = default_component_registry()
    llama_descriptor = registry.get("dinkster.hunyuan_video")
    clip_descriptor = registry.get("dinkster.classic_text")
    assert llama_descriptor is not None and clip_descriptor is not None
    from dinkster_inference.component_registry import DetectedComponents

    detected = (
        (DetectedComponents(llama_descriptor, (("llama", llama),)),),
        (DetectedComponents(clip_descriptor, (("clip_l", clip),)),),
    )
    refs = (
        WeightSourceRef("blake3:" + "1" * 64, "llama", 42),
        WeightSourceRef("blake3:" + "2" * 64, "clip", 42),
    )
    original = bind_hunyuan_video_text(detected).recipe(refs, "float32")
    for change in ({"output_hidden_layer": -2}, {"prompt_template": "{}"}, {"min_tokens": 256}):
        modified = replace(llama, config=replace(llama.config, **change))
        changed = (
            (DetectedComponents(llama_descriptor, (("llama", modified),)),),
            detected[1],
        )
        assert bind_hunyuan_video_text(changed).recipe(refs, "float32").runtime_identity != (
            original.runtime_identity
        )
