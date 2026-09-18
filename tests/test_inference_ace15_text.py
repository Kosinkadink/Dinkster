"""Exact ACE geometry and ordered recipe planning, without tensor allocation."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import BFLOAT16, FLOAT8_E4M3, FLOAT32, TensorGeometry
from dinkster_inference.ace15_text import (
    ACE15_TEXT_ROLES,
    bind_ace15_text_recipe,
    detect_ace15_components,
)
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.qwen_text import (
    ACE15_QWEN3_06B_CONFIG,
    ACE15_QWEN3_2B_CONFIG,
    ACE15_QWEN3_4B_CONFIG,
    ANIMA_QWEN3_06B_CONFIG,
    QwenTextConfig,
    QwenTextDetectError,
    detect_qwen_text_config,
    qwen_text_layout,
)
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.text_recipes import (
    UnresolvedTextRecipe,
    default_text_recipe_registry,
    resolve_text_recipe,
)
from dinkster_inference.weights import WeightEntry


def header(config: QwenTextConfig, prefix: str = "model.", digest: str = "1") -> SafetensorsSource:
    entries = {}
    for key, shape in qwen_text_layout(config).items():
        geometry = TensorGeometry(shape, BFLOAT16)
        name = prefix + key
        entries[name] = WeightEntry(name, geometry, 0, geometry.nbytes)
    return SafetensorsSource(Path("ace.safetensors"), entries, {}, "blake3:" + digest * 64, 8)


@pytest.mark.parametrize(
    "config,shape,ffn,layers,heads,maximum",
    [
        (ACE15_QWEN3_06B_CONFIG, (151669, 1024), 3072, 28, 16, 32768),
        (ACE15_QWEN3_2B_CONFIG, (217204, 2048), 6144, 28, 16, 40960),
        (ACE15_QWEN3_4B_CONFIG, (217204, 2560), 9728, 36, 32, 40960),
    ],
)
def test_exact_profiles(config, shape, ffn, layers, heads, maximum) -> None:
    layout = qwen_text_layout(config)
    assert layout["embed_tokens.weight"] == shape
    assert config.intermediate_size == ffn
    assert config.num_hidden_layers == layers
    assert config.num_attention_heads == heads
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.max_position_embeddings == maximum
    assert config.rope_theta == 1_000_000 and config.rms_norm_eps == 1e-6
    assert config.qk_norm and not config.qkv_bias
    geometries = {key: TensorGeometry(value, BFLOAT16) for key, value in layout.items()}
    assert detect_qwen_text_config(geometries) == config
    del geometries["norm.weight"]
    with pytest.raises(QwenTextDetectError, match="missing norm.weight"):
        detect_qwen_text_config(geometries)


@pytest.mark.parametrize(
    "config", [ACE15_QWEN3_06B_CONFIG, ACE15_QWEN3_2B_CONFIG, ACE15_QWEN3_4B_CONFIG]
)
@pytest.mark.parametrize(
    "prefix", ["", "model.", "transformer.model.", "text_encoders.qwen3_2b.transformer.model."]
)
def test_detection_and_quant_retention(config: QwenTextConfig, prefix: str) -> None:
    source = header(config, prefix)
    entries = dict(source.entries)
    layer = prefix + "layers.0.self_attn.q_proj"
    weight = entries[layer + ".weight"]
    entries[weight.key] = replace(weight, geometry=replace(weight.geometry, dtype=FLOAT8_E4M3))
    entries[layer + ".weight_scale"] = WeightEntry(
        layer + ".weight_scale", TensorGeometry((), FLOAT32), 0, 4
    )
    source = replace(
        source,
        entries=entries,
        extra={
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": {layer: {"format": "float8_e4m3fn"}}}
            )
        },
    )
    ((role, plan),) = detect_ace15_components(source, source.path)
    assert role == ACE15_TEXT_ROLES[0 if config.hidden_size == 1024 else 1]
    assert plan.config == config
    assert plan.quant["layers.0.self_attn.q_proj"].weight == layer + ".weight"
    assert plan.quant["layers.0.self_attn.q_proj"].weight_scale == layer + ".weight_scale"
    assert f"asset_digest={source.asset_digest}" in plan.identity_facts
    assert not detect_ace15_components(header(ANIMA_QWEN3_06B_CONFIG), source.path)


@pytest.mark.parametrize("lm_config", [ACE15_QWEN3_2B_CONFIG, ACE15_QWEN3_4B_CONFIG])
def test_order_identity_and_ambiguous_sources(lm_config: QwenTextConfig) -> None:
    registry = default_component_registry()
    descriptor = default_text_recipe_registry().get("dinkster.text_ace15")
    assert descriptor is not None
    assert descriptor.aliases == ("ace", "ace15")
    sources = (header(ACE15_QWEN3_06B_CONFIG), header(lm_config, digest="2"))
    detected = tuple(registry.detect(source, source.path) for source in sources)
    refs = tuple(
        WeightSourceRef(
            cast(str, source.asset_digest), source.path.name, cast(int, source.asset_size)
        )
        for source in sources
    )
    binding = resolve_text_recipe(detected, "ace")
    forward = binding.recipe(refs, "float32")
    reverse_binding = resolve_text_recipe(tuple(reversed(detected)), "ace15")
    reverse = reverse_binding.recipe(tuple(reversed(refs)), "float32")
    assert binding.family_id == "dinkster.ace_step_1_5"
    assert all(
        any(fact.startswith("ace15_tokenizer_vocab_sha256=") for fact in part.plan.runtime_facts)
        and "ace15_composer_generation_bound=min_as_max" in part.plan.runtime_facts
        for part in binding.components
    )
    assert [part.role for part in reverse_binding.components] == list(reversed(ACE15_TEXT_ROLES))
    assert [item.source for item in reverse.sources] == list(reversed(refs))
    assert forward.runtime_identity != reverse.runtime_identity
    assert forward.runtime_identity != binding.recipe(refs, "bfloat16").runtime_identity
    lm_name = "qwen3_2b" if lm_config.hidden_size == 2048 else "qwen3_4b"
    combined_parts = (
        header(
            ACE15_QWEN3_06B_CONFIG,
            "text_encoders.qwen3_06b.transformer.model.",
        ),
        header(
            lm_config,
            f"text_encoders.{lm_name}.transformer.model.",
        ),
    )
    combined = replace(
        combined_parts[0],
        entries={
            part: entry for source in combined_parts for part, entry in source.entries.items()
        },
    )
    combined_binding = resolve_text_recipe((registry.detect(combined, combined.path),), "ace")
    assert [(part.source_index, part.role) for part in combined_binding.components] == [
        (0, ACE15_TEXT_ROLES[0]),
        (0, ACE15_TEXT_ROLES[1]),
    ]
    with pytest.raises(UnresolvedTextRecipe):
        bind_ace15_text_recipe((detected[0],))
    with pytest.raises(UnresolvedTextRecipe):
        bind_ace15_text_recipe((*detected, detected[0]))
