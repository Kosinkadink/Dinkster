from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from dinkster_inference.devices import DType
from dinkster_inference.minimax_h3_conditioner import (
    MINIMAX_H3_CONDITIONER_CONFIG,
    MiniMaxH3ConditionerDetectError,
    detect_minimax_h3_conditioner,
    minimax_h3_conditioner_layout,
)
from dinkster_inference.weights import TensorGeometry


def _geometries() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, DType("bfloat16", 2))
        for key, shape in minimax_h3_conditioner_layout().keys.items()
    }


def test_conditioner_profile_and_exact_902_key_layout_are_frozen() -> None:
    config = MINIMAX_H3_CONDITIONER_CONFIG
    assert (
        config.hidden_size,
        config.intermediate_size,
        config.layers,
        config.attention_heads,
        config.key_value_heads,
        config.rope_dimensions,
        config.max_position_embeddings,
    ) == (5120, 25600, 50, 64, 8, (24, 20, 20), 262144)
    assert config.final_norm is False
    assert (
        config.vision_hidden_size,
        config.vision_intermediate_size,
        config.vision_layers,
        config.deepstack_layers,
    ) == (1152, 4304, 27, (8, 16, 24))
    layout = minimax_h3_conditioner_layout()
    assert len(layout.keys) == 902
    assert layout.keys["model.layers.49.mlp.down_proj.weight"] == (5120, 25600)
    assert layout.keys["visual.patch_embed.proj.weight"] == (1152, 3, 2, 16, 16)
    assert layout.keys["visual.deepstack_merger_list.2.linear_fc2.weight"] == (5120, 4608)
    with pytest.raises(TypeError):
        layout.keys["extra"] = (1,)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.layers = 49  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(config, layers=49)


def test_conditioner_detector_accepts_only_the_complete_exact_layout() -> None:
    geometries = _geometries()
    assert detect_minimax_h3_conditioner(geometries) is MINIMAX_H3_CONDITIONER_CONFIG
    cases = []
    missing = dict(geometries)
    missing.pop("model.layers.49.mlp.down_proj.weight")
    cases.append(missing)
    wrong = dict(geometries)
    wrong["visual.blocks.26.attn.qkv.weight"] = TensorGeometry((3455, 1152), DType("bfloat16", 2))
    cases.append(wrong)
    extra = dict(geometries)
    extra["model.norm.weight"] = TensorGeometry((5120,), DType("bfloat16", 2))
    cases.append(extra)
    for broken in cases:
        with pytest.raises(MiniMaxH3ConditionerDetectError):
            detect_minimax_h3_conditioner(broken)
