"""Proving tests for the torch-free T5 text-encoder layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). Synthetic cases pin every documented
rejection branch; when the real checkpoints are present on this
machine, the T5-XXL header (t5xxl_fp16.safetensors) must detect and
the UMT5 scaled-fp8 architecture must detect after quantization
metadata is separated - grounding acceptance in real files rather
than fabrications.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    T5_TEXT_OPTIONAL_KEYS,
    T5_XXL_CONFIG,
    UMT5_XXL_CONFIG,
    UMT5_XXL_WAN_PROFILE,
    T5Config,
    T5TextDetectError,
    TensorGeometry,
    detect_t5_config,
    load_safetensors_header,
    t5_layout,
)

T5XXL_FP16 = Path("/home/kosin/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors")
UMT5_FP8 = Path("/home/kosin/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")


def geometries_of(layout: dict[str, tuple[int, ...]], dtype=FLOAT32) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}


def xxl_header() -> dict[str, TensorGeometry]:
    """The full checkpoint header shape: module layout plus the
    shipped embed_tokens duplicate."""
    geometries = geometries_of(t5_layout(T5_XXL_CONFIG))
    geometries["encoder.embed_tokens.weight"] = geometries["shared.weight"]
    return geometries


# ------------------------------------------------------------ config


def test_xxl_config_is_the_reference_json() -> None:
    # comfy/text_encoders/t5_config_xxl.json @ 947c2749.
    assert T5_XXL_CONFIG.d_model == 4096
    assert T5_XXL_CONFIG.d_ff == 10240
    assert T5_XXL_CONFIG.d_kv == 64
    assert T5_XXL_CONFIG.num_heads == 64
    assert T5_XXL_CONFIG.num_layers == 24
    assert T5_XXL_CONFIG.vocab_size == 32128
    assert T5_XXL_CONFIG.dense_act_fn == "gelu_pytorch_tanh"
    assert T5_XXL_CONFIG.is_gated_act
    assert T5_XXL_CONFIG.model_type == "t5"
    assert T5_XXL_CONFIG.inner_dim == 4096
    assert T5_XXL_CONFIG.layer_norm_eps == 1e-6


@pytest.mark.parametrize(
    "field",
    ["d_model", "d_ff", "d_kv", "num_heads", "num_layers", "vocab_size"],
)
def test_config_rejects_non_positive_dimensions(field: str) -> None:
    kwargs = {
        "d_model": 8,
        "d_ff": 16,
        "d_kv": 2,
        "num_heads": 4,
        "num_layers": 2,
        "vocab_size": 10,
        "dense_act_fn": "relu",
        "is_gated_act": False,
    }
    kwargs[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        T5Config(**kwargs)


def test_config_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="dense_act_fn"):
        T5Config(
            d_model=8,
            d_ff=16,
            d_kv=2,
            num_heads=4,
            num_layers=2,
            vocab_size=10,
            dense_act_fn="silu",
            is_gated_act=False,
        )


def test_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        T5Config(
            d_model=8,
            d_ff=16,
            d_kv=2,
            num_heads=4,
            num_layers=2,
            vocab_size=10,
            dense_act_fn="relu",
            is_gated_act=False,
            model_type="mt5",
        )


def test_umt5_xxl_config_and_wan_profile_are_exact() -> None:
    assert UMT5_XXL_CONFIG == T5Config(
        d_model=4096,
        d_ff=10240,
        d_kv=64,
        num_heads=64,
        num_layers=24,
        vocab_size=256384,
        dense_act_fn="gelu_pytorch_tanh",
        is_gated_act=True,
        model_type="umt5",
    )
    assert UMT5_XXL_WAN_PROFILE.max_length == 99999999
    assert UMT5_XXL_WAN_PROFILE.start_token is None
    assert UMT5_XXL_WAN_PROFILE.end_token == 1
    assert UMT5_XXL_WAN_PROFILE.pad_token == 0
    assert not UMT5_XXL_WAN_PROFILE.pad_to_max_length
    assert UMT5_XXL_WAN_PROFILE.min_length == 512


# ------------------------------------------------------------ layout


def test_xxl_layout_key_count_matches_real_checkpoint() -> None:
    # t5xxl_fp16.safetensors carries 220 keys: this layout's 219 plus
    # the embed_tokens duplicate.
    layout = t5_layout(T5_XXL_CONFIG)
    assert len(layout) == 219
    assert len(xxl_header()) == 220


def test_xxl_layout_facts() -> None:
    layout = t5_layout(T5_XXL_CONFIG)
    assert layout["shared.weight"] == (32128, 4096)
    assert layout["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] == (
        32,
        64,
    )
    # Bias table in block 0 ONLY (model_type t5, not umt5).
    assert "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight" not in layout
    assert layout["encoder.block.7.layer.1.DenseReluDense.wi_0.weight"] == (
        10240,
        4096,
    )
    assert layout["encoder.block.7.layer.1.DenseReluDense.wo.weight"] == (
        4096,
        10240,
    )
    # RMS norms: weight only, never a bias.
    assert not any(key.endswith(".bias") for key in layout)
    assert not any("embed_tokens" in key for key in layout)


def test_umt5_xxl_layout_owns_relative_bias_in_every_block() -> None:
    layout = t5_layout(UMT5_XXL_CONFIG)
    assert len(layout) == 242
    assert layout["shared.weight"] == (256384, 4096)
    for index in range(24):
        assert layout[
            f"encoder.block.{index}.layer.0.SelfAttention.relative_attention_bias.weight"
        ] == (32, 64)


def test_non_gated_layout_uses_wi() -> None:
    config = T5Config(
        d_model=8,
        d_ff=16,
        d_kv=2,
        num_heads=4,
        num_layers=1,
        vocab_size=10,
        dense_act_fn="relu",
        is_gated_act=False,
    )
    layout = t5_layout(config)
    assert layout["encoder.block.0.layer.1.DenseReluDense.wi.weight"] == (16, 8)
    assert not any("wi_0" in key for key in layout)


# --------------------------------------------------------- detection


def test_detects_xxl_module_layout() -> None:
    assert detect_t5_config(geometries_of(t5_layout(T5_XXL_CONFIG))) is T5_XXL_CONFIG


def test_detects_xxl_checkpoint_with_embed_tokens_duplicate() -> None:
    assert detect_t5_config(xxl_header()) is T5_XXL_CONFIG


def test_detection_ignores_dtypes() -> None:
    header = geometries_of(t5_layout(T5_XXL_CONFIG), dtype=FLOAT16)
    assert detect_t5_config(header) is T5_XXL_CONFIG


def test_rejects_empty_header() -> None:
    with pytest.raises(T5TextDetectError, match="empty"):
        detect_t5_config({})


def test_detects_umt5_xxl_layout() -> None:
    assert detect_t5_config(geometries_of(t5_layout(UMT5_XXL_CONFIG))) is UMT5_XXL_CONFIG


def test_rejects_header_without_shared_embedding() -> None:
    header = xxl_header()
    del header["shared.weight"]
    with pytest.raises(T5TextDetectError, match="shared.weight"):
        detect_t5_config(header)


def test_rejects_unknown_geometry() -> None:
    header = xxl_header()
    header["shared.weight"] = TensorGeometry((32128, 2048), FLOAT32)
    header["encoder.embed_tokens.weight"] = header["shared.weight"]
    with pytest.raises(T5TextDetectError, match="only T5-XXL.*UMT5-XXL"):
        detect_t5_config(header)


def test_rejects_missing_required_key() -> None:
    header = xxl_header()
    del header["encoder.block.11.layer.0.SelfAttention.k.weight"]
    with pytest.raises(T5TextDetectError, match="missing"):
        detect_t5_config(header)


def test_rejects_mismatched_shape() -> None:
    header = xxl_header()
    header["encoder.block.3.layer.1.DenseReluDense.wo.weight"] = TensorGeometry(
        (4096, 5120), FLOAT32
    )
    with pytest.raises(T5TextDetectError, match="expected shape"):
        detect_t5_config(header)


def test_rejects_unexpected_keys() -> None:
    header = xxl_header()
    header["decoder.block.0.layer.0.SelfAttention.q.weight"] = TensorGeometry((4096, 4096), FLOAT32)
    with pytest.raises(T5TextDetectError, match="unexpected key"):
        detect_t5_config(header)


def test_rejects_misshapen_embed_tokens_duplicate() -> None:
    header = xxl_header()
    header["encoder.embed_tokens.weight"] = TensorGeometry((32100, 4096), FLOAT32)
    with pytest.raises(T5TextDetectError, match="embed_tokens"):
        detect_t5_config(header)


def test_optional_keys_are_only_the_embed_tokens_duplicate() -> None:
    assert T5_TEXT_OPTIONAL_KEYS == {"encoder.embed_tokens.weight"}


# ------------------------------------------------ real checkpoints


@pytest.mark.skipif(not T5XXL_FP16.exists(), reason="real checkpoint not present")
def test_real_t5xxl_fp16_header_detects() -> None:
    source = load_safetensors_header(T5XXL_FP16)
    header = {key: source.entry(key).geometry for key in source.keys()}
    assert len(header) == 220
    assert detect_t5_config(header) is T5_XXL_CONFIG


@pytest.mark.skipif(not UMT5_FP8.exists(), reason="real checkpoint not present")
def test_real_umt5_scaled_fp8_architecture_detects() -> None:
    from dinkster_inference.quantization import split_quantization

    source = load_safetensors_header(UMT5_FP8)
    header = {key: source.entry(key).geometry for key in source.keys()}
    architecture = split_quantization(header, source.metadata()).architecture
    # Assembly consumes the embedded spiece_model tokenizer tensor before
    # layout detection; mirror that here.
    architecture = {key: value for key, value in architecture.items() if key != "spiece_model"}
    assert detect_t5_config(architecture) is UMT5_XXL_CONFIG
