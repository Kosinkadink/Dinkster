"""Proving tests for the torch-free SD1/SDXL diffusion UNet layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). The three accepted geometries are pinned
against the EXECUTED reference's own full-size state-dict listings
(goldens/unet_goldens.json "layouts", produced by the reference
UNetModel @ the audited baseline on the meta device), and every
documented rejection branch refuses with a UNetDetectError naming
what it found.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    KNOWN_UNET_CONFIGS,
    SD15_INPAINT_UNET_CONFIG,
    SD15_UNET_CONFIG,
    SDXL_INPAINT_UNET_CONFIG,
    SDXL_REFINER_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    UNET_HEAD_PROFILES,
    TensorGeometry,
    UNetDetectError,
    detect_unet_config,
    unet_layout,
)

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "unet_goldens.json"
)

NAMED = {
    "sd15": SD15_UNET_CONFIG,
    "sd15_inpaint": SD15_INPAINT_UNET_CONFIG,
    "sdxl": SDXL_UNET_CONFIG,
    "sdxl_inpaint": SDXL_INPAINT_UNET_CONFIG,
    "sdxl_refiner": SDXL_REFINER_UNET_CONFIG,
}


def golden_layout(name: str) -> list[tuple[str, list[int]]]:
    payload = json.loads(GOLDENS.read_text())
    return [(key, list(shape)) for key, shape in payload["layouts"][name]]


def geometries_of(
    layout: list[tuple[str, list[int]]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in layout}


# ------------------------------------------------------------ config


def test_known_configs_are_the_reference_geometries() -> None:
    assert SD15_UNET_CONFIG.model_channels == 320
    assert SD15_UNET_CONFIG.context_dim == 768
    assert SD15_UNET_CONFIG.num_heads == 8
    assert not SD15_UNET_CONFIG.use_linear_in_transformer
    assert SD15_UNET_CONFIG.adm_in_channels is None
    assert SDXL_UNET_CONFIG.model_channels == 320
    assert SDXL_UNET_CONFIG.context_dim == 2048
    assert SDXL_UNET_CONFIG.adm_in_channels == 2816
    assert SDXL_UNET_CONFIG.transformer_depth_middle == 10
    assert SDXL_INPAINT_UNET_CONFIG.in_channels == 9
    assert SDXL_INPAINT_UNET_CONFIG.adm_in_channels == 2816
    assert SDXL_REFINER_UNET_CONFIG.model_channels == 384
    assert SDXL_REFINER_UNET_CONFIG.context_dim == 1280
    assert SDXL_REFINER_UNET_CONFIG.adm_in_channels == 2560
    assert SDXL_REFINER_UNET_CONFIG.transformer_depth == (
        0,
        0,
        4,
        4,
        4,
        4,
        0,
        0,
    )
    assert {config.in_channels for config in KNOWN_UNET_CONFIGS} == {4, 9}
    for config in KNOWN_UNET_CONFIGS:
        assert config.out_channels == 4
        assert config.time_embed_dim == config.model_channels * 4
    assert set(KNOWN_UNET_CONFIGS) == set(NAMED.values())


def test_head_profiles_match_the_supported_models_table() -> None:
    assert UNET_HEAD_PROFILES[(768, False)] == (8, -1)
    assert UNET_HEAD_PROFILES[(2048, True)] == (-1, 64)
    assert UNET_HEAD_PROFILES[(1280, True)] == (-1, 64)
    assert (1024, True) not in UNET_HEAD_PROFILES  # SD2.x deferred


def test_config_rejects_odd_model_channels() -> None:
    with pytest.raises(ValueError, match="must be even"):
        replace(SD15_UNET_CONFIG, model_channels=321)


def test_config_rejects_channel_mult_not_starting_at_one() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        replace(SD15_UNET_CONFIG, channel_mult=(2, 2, 4, 4))


def test_config_rejects_level_count_mismatch() -> None:
    with pytest.raises(ValueError, match="levels"):
        replace(SD15_UNET_CONFIG, channel_mult=(1, 2, 4))


def test_config_rejects_wrong_transformer_depth_length() -> None:
    with pytest.raises(ValueError, match="per input res block"):
        replace(SD15_UNET_CONFIG, transformer_depth=(1, 1, 1))
    with pytest.raises(ValueError, match="per output res block"):
        replace(SD15_UNET_CONFIG, transformer_depth_output=(1, 1, 1))


def test_config_rejects_pruned_middle_depths() -> None:
    with pytest.raises(ValueError, match="SSD1B"):
        replace(SD15_UNET_CONFIG, transformer_depth_middle=-1)


def test_config_requires_exactly_one_head_policy() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        replace(SD15_UNET_CONFIG, num_heads=8, num_head_channels=64)
    with pytest.raises(ValueError, match="exactly one"):
        replace(SD15_UNET_CONFIG, num_heads=-1, num_head_channels=-1)


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        replace(SD15_UNET_CONFIG, num_heads=7)
    with pytest.raises(ValueError, match="do not divide"):
        replace(SDXL_UNET_CONFIG, num_heads=-1, num_head_channels=384)


def test_heads_for_resolves_both_policies() -> None:
    assert SD15_UNET_CONFIG.heads_for(640) == (8, 80)
    assert SDXL_UNET_CONFIG.heads_for(640) == (10, 64)
    assert SDXL_REFINER_UNET_CONFIG.heads_for(1536) == (24, 64)


# ------------------------------------------------------------ layout


@pytest.mark.parametrize("name", sorted(NAMED))
def test_layout_matches_executed_reference(name: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in unet_layout(NAMED[name]).items())
    assert predicted == golden_layout(name)


def test_layout_final_bias_follows_out_channels() -> None:
    layout = unet_layout(replace(SD15_UNET_CONFIG, out_channels=8))
    assert layout["out.2.weight"] == (8, 320, 3, 3)
    assert layout["out.2.bias"] == (8,)


# --------------------------------------------------------- detection


@pytest.mark.parametrize("name", sorted(NAMED))
def test_detects_reference_geometry(name: str) -> None:
    assert detect_unet_config(geometries_of(golden_layout(name))) == (NAMED[name])


def test_detects_fp16_checkpoints() -> None:
    geometries = {
        key: TensorGeometry(tuple(shape), FLOAT16) for key, shape in golden_layout("sd15")
    }
    assert detect_unet_config(geometries) == SD15_UNET_CONFIG


def test_detects_inpainting_style_in_channels() -> None:
    """Inpainting/ip2p variants differ only in the first conv's input
    width; detection reads it from the geometry."""
    config = SD15_INPAINT_UNET_CONFIG
    detected = detect_unet_config(
        geometries_of([(key, list(shape)) for key, shape in unet_layout(config).items()])
    )
    assert detected == config


def test_rejects_empty_header() -> None:
    with pytest.raises(UNetDetectError, match="empty"):
        detect_unet_config({})


def test_rejects_non_sd_families() -> None:
    geometries = {
        "joint_blocks.0.context_block.attn.qkv.weight": TensorGeometry((1536, 1536), FLOAT32)
    }
    with pytest.raises(UNetDetectError, match="not an SD-era UNet"):
        detect_unet_config(geometries)


def test_rejects_non_conv_first_layer() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    geometries["input_blocks.0.0.weight"] = TensorGeometry((320, 4), FLOAT32)
    with pytest.raises(UNetDetectError, match="rank 2"):
        detect_unet_config(geometries)


def test_rejects_malformed_output_projection_rank() -> None:
    """Malformed geometry refuses with UNetDetectError, never an
    incidental IndexError."""
    geometries = geometries_of(golden_layout("sd15"))
    geometries["out.2.weight"] = TensorGeometry((4,), FLOAT32)
    with pytest.raises(UNetDetectError, match="out.2.weight has rank 1"):
        detect_unet_config(geometries)


def test_rejects_malformed_adm_projection_rank() -> None:
    geometries = geometries_of(golden_layout("sdxl"))
    geometries["label_emb.0.0.weight"] = TensorGeometry((1280,), FLOAT32)
    with pytest.raises(UNetDetectError, match="label_emb.0.0.weight has rank 1"):
        detect_unet_config(geometries)


def test_rejects_malformed_attention_to_k_rank() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    key = "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
    geometries[key] = TensorGeometry((320, 768, 1, 1), FLOAT32)
    with pytest.raises(UNetDetectError, match="to_k.weight has rank 4"):
        detect_unet_config(geometries)


def test_rejects_malformed_proj_in_rank() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    geometries["input_blocks.1.1.proj_in.weight"] = TensorGeometry((320, 320, 1), FLOAT32)
    with pytest.raises(UNetDetectError, match="proj_in.weight has rank"):
        detect_unet_config(geometries)


def test_rejects_malformed_resblock_widening_rank() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    geometries["input_blocks.1.0.out_layers.3.weight"] = TensorGeometry((320,), FLOAT32)
    with pytest.raises(UNetDetectError, match="out_layers.3.weight has"):
        detect_unet_config(geometries)


def test_rejects_temporal_video_unets() -> None:
    geometries = geometries_of(golden_layout("sdxl"))
    geometries["input_blocks.4.1.time_stack.0.attn1.to_q.weight"] = TensorGeometry(
        (640, 640), FLOAT32
    )
    with pytest.raises(UNetDetectError, match="temporal/video"):
        detect_unet_config(geometries)


def test_rejects_unknown_head_profile() -> None:
    """SD2.x geometry (context 1024, linear projections is closest;
    conv here) has no pinned head profile and must refuse, never
    guess."""
    geometries = geometries_of(golden_layout("sd15"))
    for key in list(geometries):
        if key.endswith(("attn2.to_k.weight", "attn2.to_v.weight")):
            rows = geometries[key].shape[0]
            geometries[key] = TensorGeometry((rows, 1024), FLOAT32)
    with pytest.raises(UNetDetectError, match="unknown head profile"):
        detect_unet_config(geometries)


def test_rejects_middle_block_without_transformer() -> None:
    geometries = {
        key: value
        for key, value in geometries_of(golden_layout("sd15")).items()
        if not key.startswith("middle_block.1.")
    }
    with pytest.raises(UNetDetectError, match="SSD1B"):
        detect_unet_config(geometries)


def test_rejects_missing_middle_block() -> None:
    geometries = {
        key: value
        for key, value in geometries_of(golden_layout("sd15")).items()
        if not key.startswith("middle_block.")
    }
    with pytest.raises(UNetDetectError, match="KOALA"):
        detect_unet_config(geometries)


def test_rejects_transformerless_scan() -> None:
    geometries = {
        "input_blocks.0.0.weight": TensorGeometry((32, 4, 3, 3), FLOAT32),
        "input_blocks.0.0.bias": TensorGeometry((32,), FLOAT32),
        "out.2.weight": TensorGeometry((4, 32, 3, 3), FLOAT32),
        "middle_block.1.proj_in.weight": TensorGeometry((32, 32, 1, 1), FLOAT32),
    }
    with pytest.raises(UNetDetectError, match="no spatial transformer"):
        detect_unet_config(geometries)


def test_rejects_missing_key_against_layout() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    del geometries["output_blocks.11.0.out_layers.3.bias"]
    with pytest.raises(UNetDetectError, match="missing output_blocks.11"):
        detect_unet_config(geometries)


def test_rejects_unexpected_key_against_layout() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    geometries["input_blocks.1.0.stray.weight"] = TensorGeometry((320,), FLOAT32)
    with pytest.raises(UNetDetectError, match="unexpected key"):
        detect_unet_config(geometries)


def test_rejects_drifted_shape_against_layout() -> None:
    geometries = geometries_of(golden_layout("sd15"))
    geometries["time_embed.0.weight"] = TensorGeometry((1280, 640), FLOAT32)
    with pytest.raises(UNetDetectError, match="time_embed.0.weight"):
        detect_unet_config(geometries)
