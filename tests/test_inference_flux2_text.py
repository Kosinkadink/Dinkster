"""Flux2 text-encoder tower profiles and prompt policies.

Layout literals are pinned from the official checkpoint headers
(flux2 mistral_3_small_flux2_bf16, qwen_3_4b, qwen_3_8b) and the
config facts from ComfyUI's comfy/text_encoders/llama.py and flux.py
@ b78cec87. The mistral checkpoint also carries vision_tower.*,
multi_modal_projector.*, and tekken_model siblings outside model.*;
those never reach tower detection.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from dinkster_inference import (
    BFLOAT16,
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    Z_IMAGE_QWEN3_4B_CONFIG,
    QwenTextConfig,
    QwenTextDetectError,
    TensorGeometry,
    detect_qwen_text_config,
    qwen_text_layout,
)

KLEIN_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MISTRAL3_TEMPLATE = (
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions."
    " You give structured responses focusing on object relationships, object\n"
    "attribution and actions without speculation.[/SYSTEM_PROMPT][INST]{}[/INST]"
)


def geometries_for(config: QwenTextConfig) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, BFLOAT16) for key, shape in qwen_text_layout(config).items()}


def test_mistral3_pruned_identity_and_layout() -> None:
    config = MISTRAL3_24B_PRUNED_CONFIG
    assert (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (131072, 5120, 32768, 30, 32, 8, 128)
    assert (config.max_position_embeddings, config.rms_norm_eps, config.rope_theta) == (
        8192,
        1e-5,
        1_000_000_000.0,
    )
    assert not config.qk_norm
    assert not config.qkv_bias
    assert not config.final_norm
    assert not config.layer_norm_hidden_state
    assert not config.zero_masked
    assert config.output_hidden_layers == (9, 19, 29)
    assert config.prompt_template == MISTRAL3_TEMPLATE
    assert (config.min_tokens, config.pad_token_id) == (1, 11)
    assert config.slice_marker_id is None and config.slice_marker_suffix_id is None

    layout = qwen_text_layout(config)
    assert len(layout) == 272
    assert layout["embed_tokens.weight"] == (131072, 5120)
    assert layout["norm.weight"] == (5120,)
    assert layout["layers.0.self_attn.q_proj.weight"] == (4096, 5120)
    assert layout["layers.0.self_attn.k_proj.weight"] == (1024, 5120)
    assert layout["layers.0.self_attn.o_proj.weight"] == (5120, 4096)
    assert layout["layers.29.mlp.gate_proj.weight"] == (32768, 5120)
    assert "layers.30.input_layernorm.weight" not in layout
    assert not any("q_norm" in key or "bias" in key for key in layout)
    assert detect_qwen_text_config(geometries_for(config)) is config


def test_mistral3_full_and_pruned_disambiguate_by_layer_count() -> None:
    full = MISTRAL3_24B_CONFIG
    assert full.num_hidden_layers == 40
    assert full.final_norm
    assert full.output_hidden_layers == (9, 19, 29)
    assert len(qwen_text_layout(full)) == 362
    assert detect_qwen_text_config(geometries_for(full)) is full
    assert detect_qwen_text_config(geometries_for(MISTRAL3_24B_PRUNED_CONFIG)) is (
        MISTRAL3_24B_PRUNED_CONFIG
    )

    truncated = geometries_for(full)
    del truncated["layers.39.mlp.down_proj.weight"]
    with pytest.raises(QwenTextDetectError, match="missing layers.39"):
        detect_qwen_text_config(truncated)


def test_klein_qwen3_8b_identity_and_layout() -> None:
    config = KLEIN_QWEN3_8B_CONFIG
    assert (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (151936, 4096, 12288, 36, 32, 8, 128)
    assert config.qk_norm
    assert config.final_norm
    assert not config.layer_norm_hidden_state
    assert not config.zero_masked
    assert config.output_hidden_layers == (8, 17, 26)
    assert config.prompt_template == KLEIN_TEMPLATE
    assert (config.min_tokens, config.pad_token_id) == (512, 151643)

    layout = qwen_text_layout(config)
    assert len(layout) == 398
    assert layout["embed_tokens.weight"] == (151936, 4096)
    assert layout["layers.0.self_attn.q_proj.weight"] == (4096, 4096)
    assert layout["layers.0.self_attn.k_proj.weight"] == (1024, 4096)
    assert layout["layers.35.self_attn.q_norm.weight"] == (128,)
    # lm_head.weight lives outside model.* in the checkpoint; inside the
    # tower scope it must refuse.
    geometries = geometries_for(config)
    assert detect_qwen_text_config(geometries) is config
    geometries["lm_head.weight"] = TensorGeometry((151936, 4096), BFLOAT16)
    with pytest.raises(QwenTextDetectError, match="unexpected key lm_head.weight"):
        detect_qwen_text_config(geometries)


def test_klein_qwen3_4b_shares_z_image_tower() -> None:
    config = KLEIN_QWEN3_4B_CONFIG
    assert qwen_text_layout(config) == qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG)
    # Identical bytes: detection returns the canonical Z-Image tower and
    # the Flux2 assembly substitutes the Klein policy.
    assert detect_qwen_text_config(geometries_for(config)) is Z_IMAGE_QWEN3_4B_CONFIG
    assert config.prompt_template == KLEIN_TEMPLATE
    assert (config.min_tokens, config.pad_token_id) == (512, 151643)
    assert config.output_hidden_layer is None
    assert config.output_hidden_layers == (8, 17, 26)
    assert config.slice_marker_id is None and config.slice_marker_suffix_id is None
    assert not config.zero_masked


def test_unknown_width_lists_accepted_profiles() -> None:
    with pytest.raises(QwenTextDetectError, match="mistral3_24b_pruned \\(131072, 5120\\)"):
        detect_qwen_text_config({"embed_tokens.weight": TensorGeometry((999, 999), BFLOAT16)})


def test_capture_and_marker_validators() -> None:
    base = MISTRAL3_24B_PRUNED_CONFIG
    with pytest.raises(ValueError, match="exclusive"):
        replace(base, output_hidden_layer=-2)
    with pytest.raises(ValueError, match="at least one layer"):
        replace(base, output_hidden_layers=())
    with pytest.raises(ValueError, match="address model layers"):
        replace(base, output_hidden_layers=(9, 19, 30))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(base, output_hidden_layers=(9, 9, 19))
    with pytest.raises(ValueError, match="set together"):
        replace(base, slice_marker_id=11)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        replace(base, slice_marker_id=131072, slice_marker_suffix_id=1)
