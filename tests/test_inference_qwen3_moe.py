"""Qwen3-30B-A3B profile and original-checkpoint layout admission."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    QWEN3_30B_A3B_CONFIG,
    Qwen3MoeDetectError,
    TensorGeometry,
    detect_qwen3_moe_config,
    qwen3_moe_layout,
)


def _geometries() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, BFLOAT16)
        for key, shape in qwen3_moe_layout(QWEN3_30B_A3B_CONFIG).items()
    }


def test_exact_qwen3_30b_a3b_profile_and_layout() -> None:
    config = QWEN3_30B_A3B_CONFIG
    assert (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.moe_intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.num_experts,
        config.num_experts_per_tok,
    ) == (151936, 2048, 6144, 768, 48, 32, 4, 128, 128, 8)
    assert config.max_position_embeddings == 40960
    assert config.decoder_sparse_step == 1
    assert config.mlp_only_layers == ()
    assert config.norm_topk_prob
    assert config.qk_norm and not config.qkv_bias
    assert not config.tie_word_embeddings
    assert config.eos_token_id == 151645

    layout = qwen3_moe_layout(config)
    assert len(layout) == 18867
    assert layout["model.layers.0.self_attn.q_proj.weight"] == (4096, 2048)
    assert layout["model.layers.0.self_attn.k_proj.weight"] == (512, 2048)
    assert layout["model.layers.47.mlp.gate.weight"] == (128, 2048)
    assert layout["model.layers.47.mlp.experts.127.down_proj.weight"] == (2048, 768)
    assert detect_qwen3_moe_config(_geometries()) is config


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "missing model.layers.47.mlp.experts.127.down_proj.weight"),
        ("wrong_shape", r"expected shape \(128, 2048\), found \(127, 2048\)"),
        ("wrong_dtype", "expected dtype bfloat16, found float32"),
        ("unexpected", "unexpected key model.layers.48.mlp.gate.weight"),
    ),
)
def test_detection_rejects_incomplete_or_altered_layout(mutation: str, match: str) -> None:
    geometries = _geometries()
    if mutation == "missing":
        del geometries["model.layers.47.mlp.experts.127.down_proj.weight"]
    elif mutation == "wrong_shape":
        geometries["model.layers.0.mlp.gate.weight"] = TensorGeometry((127, 2048), BFLOAT16)
    elif mutation == "wrong_dtype":
        geometries["model.layers.0.mlp.gate.weight"] = TensorGeometry((128, 2048), FLOAT32)
    else:
        geometries["model.layers.48.mlp.gate.weight"] = TensorGeometry((128, 2048), BFLOAT16)
    with pytest.raises(Qwen3MoeDetectError, match=match):
        detect_qwen3_moe_config(geometries)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"num_experts_per_tok": 129}, "must not exceed"),
        ({"num_key_value_heads": 3}, "must be divisible"),
        ({"head_dim": 127}, "must be even"),
        ({"decoder_sparse_step": 2}, "all-sparse"),
        ({"mlp_only_layers": (47,)}, "all-sparse"),
        ({"mlp_only_layers": (48,)}, "decoder-layer indices"),
        ({"eos_token_id": True}, "within the vocabulary"),
    ),
)
def test_config_rejects_invalid_semantic_bounds(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(QWEN3_30B_A3B_CONFIG, **changes)
