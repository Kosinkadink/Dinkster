"""Proving tests for the torch-free SD1/SDXL CLIP text-model layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). The two accepted layouts are pinned against
the EXECUTED reference's own full-size state-dict listings
(goldens/clip_text_goldens.json "layouts", produced by
comfy.clip_model.CLIPTextModel @ the audited baseline on the meta
device), and every documented rejection branch refuses with a
ClipTextDetectError naming what it found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dinkster_inference import (
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    CLIP_TEXT_OPTIONAL_KEYS,
    FLOAT16,
    FLOAT32,
    KNOWN_CLIP_TEXT_CONFIGS,
    ClipTextConfig,
    ClipTextDetectError,
    TensorGeometry,
    clip_text_layout,
    detect_clip_text_config,
)

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "clip_text_goldens.json"
)


def golden_layout(name: str) -> list[tuple[str, list[int]]]:
    payload = json.loads(GOLDENS.read_text())
    return [(key, list(shape)) for key, shape in payload["layouts"][name]]


def geometries_of(
    layout: list[tuple[str, list[int]]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in layout}


# ------------------------------------------------------------ config


def test_known_configs_are_the_reference_geometries() -> None:
    assert CLIP_L_TEXT_CONFIG.hidden_size == 768
    assert CLIP_L_TEXT_CONFIG.num_hidden_layers == 12
    assert CLIP_L_TEXT_CONFIG.num_attention_heads == 12
    assert CLIP_L_TEXT_CONFIG.intermediate_size == 3072
    assert CLIP_L_TEXT_CONFIG.hidden_act == "quick_gelu"
    assert CLIP_G_TEXT_CONFIG.hidden_size == 1280
    assert CLIP_G_TEXT_CONFIG.num_hidden_layers == 32
    assert CLIP_G_TEXT_CONFIG.num_attention_heads == 20
    assert CLIP_G_TEXT_CONFIG.intermediate_size == 5120
    assert CLIP_G_TEXT_CONFIG.hidden_act == "gelu"
    for config in KNOWN_CLIP_TEXT_CONFIGS:
        assert config.vocab_size == 49408
        assert config.max_position_embeddings == 77
        assert config.eos_token_id == 49407


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    ],
)
def test_config_rejects_non_positive_dimensions(field: str) -> None:
    base = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "hidden_act": "gelu",
        "vocab_size": 49408,
        "max_position_embeddings": 77,
    }
    base[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        ClipTextConfig(**base)  # type: ignore[arg-type]


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ClipTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=3,
            intermediate_size=128,
            hidden_act="gelu",
        )


def test_config_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="hidden_act"):
        ClipTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act="gelu_pytorch_tanh",
        )


def test_config_rejects_eos_outside_vocabulary() -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        ClipTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act="gelu",
            vocab_size=1000,
            eos_token_id=1000,
        )


def test_config_rejects_non_positive_eps() -> None:
    with pytest.raises(ValueError, match="layer_norm_eps"):
        ClipTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act="gelu",
            layer_norm_eps=0.0,
        )


# ------------------------------------------------------------ layout


@pytest.mark.parametrize(
    ("name", "config"),
    [("clip_l", CLIP_L_TEXT_CONFIG), ("clip_g", CLIP_G_TEXT_CONFIG)],
)
def test_layout_matches_executed_reference(name: str, config: ClipTextConfig) -> None:
    ours = sorted((key, list(shape)) for key, shape in clip_text_layout(config).items())
    assert ours == golden_layout(name)


def test_optional_keys_exist_in_both_layouts() -> None:
    for config in KNOWN_CLIP_TEXT_CONFIGS:
        layout = clip_text_layout(config)
        for key in CLIP_TEXT_OPTIONAL_KEYS:
            assert key in layout


# --------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("name", "config"),
    [("clip_l", CLIP_L_TEXT_CONFIG), ("clip_g", CLIP_G_TEXT_CONFIG)],
)
def test_detects_reference_layouts(name: str, config: ClipTextConfig) -> None:
    assert detect_clip_text_config(geometries_of(golden_layout(name))) == config


def test_detects_sd1_checkpoint_without_text_projection() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    del geometries["text_projection.weight"]
    assert detect_clip_text_config(geometries) == CLIP_L_TEXT_CONFIG


def test_detection_ignores_dtypes() -> None:
    geometries = {
        key: TensorGeometry(geometry.shape, FLOAT16)
        for key, geometry in geometries_of(golden_layout("clip_g")).items()
    }
    assert detect_clip_text_config(geometries) == CLIP_G_TEXT_CONFIG


def test_rejects_empty_header() -> None:
    with pytest.raises(ClipTextDetectError, match="empty"):
        detect_clip_text_config({})


def test_rejects_openclip_format_by_name() -> None:
    geometries = {
        "transformer.resblocks.0.attn.in_proj_weight": TensorGeometry((3840, 1280), FLOAT32),
    }
    with pytest.raises(ClipTextDetectError, match="OpenCLIP"):
        detect_clip_text_config(geometries)


def test_rejects_header_without_token_embedding() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    del geometries["text_model.embeddings.token_embedding.weight"]
    with pytest.raises(ClipTextDetectError, match="token_embedding"):
        detect_clip_text_config(geometries)


def test_rejects_token_embedding_of_wrong_rank() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    geometries["text_model.embeddings.token_embedding.weight"] = TensorGeometry(
        (49408, 768, 1), FLOAT32
    )
    with pytest.raises(ClipTextDetectError, match="rank"):
        detect_clip_text_config(geometries)


def test_rejects_unknown_geometry() -> None:
    # CLIP-H (1024 hidden) is real but the head count is not
    # derivable from shapes; only L and G are accepted.
    geometries = geometries_of(golden_layout("clip_l"))
    geometries["text_model.embeddings.token_embedding.weight"] = TensorGeometry(
        (49408, 1024), FLOAT32
    )
    with pytest.raises(ClipTextDetectError, match="unknown CLIP text"):
        detect_clip_text_config(geometries)


def test_rejects_missing_required_key() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    del geometries["text_model.encoder.layers.7.mlp.fc1.weight"]
    with pytest.raises(ClipTextDetectError, match="missing"):
        detect_clip_text_config(geometries)


def test_rejects_mismatched_shape() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    geometries["text_model.final_layer_norm.weight"] = TensorGeometry((1024,), FLOAT32)
    with pytest.raises(ClipTextDetectError, match="expected shape"):
        detect_clip_text_config(geometries)


def test_rejects_unexpected_keys() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    geometries["text_model.embeddings.position_ids"] = TensorGeometry((1, 77), FLOAT32)
    with pytest.raises(ClipTextDetectError, match="unexpected key"):
        detect_clip_text_config(geometries)


def test_rejects_long_context_position_table() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    geometries["text_model.embeddings.position_embedding.weight"] = TensorGeometry(
        (248, 768), FLOAT32
    )
    with pytest.raises(ClipTextDetectError, match="position_embedding"):
        detect_clip_text_config(geometries)


def test_rejection_message_caps_problem_listing() -> None:
    geometries = geometries_of(golden_layout("clip_l"))
    for i in range(10):
        del geometries[f"text_model.encoder.layers.{i}.mlp.fc1.weight"]
    with pytest.raises(ClipTextDetectError, match="and 4 more"):
        detect_clip_text_config(geometries)
