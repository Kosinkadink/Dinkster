"""Torch-free LTX family detection and catalog tests."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference.catalog import LTXAV, builtin_family_registry
from dinkster_inference.devices import BFLOAT16, FLOAT32
from dinkster_inference.ltx import (
    LTX_SIGMAS,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V23_VAE_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_LATENT,
    LTXAV_VIDEO_CODEC,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    LTXV_2B_V095_CONFIG,
    LTXV_2B_V095_VAE_CONFIG,
    LTXV_CODEC,
    LTXV_LATENT,
    LTXV_SHIFT_BASE,
    LTXV_SHIFT_MAX,
    LTXV_SHIFT_TOKENS_HIGH,
    LTXV_SHIFT_TOKENS_LOW,
    LTXV_VAE_DECODE_NOISE_SCALE,
    LTXV_VAE_DECODE_TIMESTEP,
    LTXAVConfig,
    LTXGeneratedKeyframes,
    LTXVAEBlock,
    LTXVConfig,
    LTXVideoVAEConfig,
    detect_ltxav,
    detect_ltxv,
    ltxav_layout,
    ltxv_dynamic_shift,
    ltxv_layout,
    ltxv_vae_layout,
)
from dinkster_inference.t5_spm import T5_XXL_FLUX_PROFILE, T5_XXL_LTXV_PROFILE
from dinkster_inference.weights import TensorGeometry, WeightEntry

V095_METADATA = {"config": json.dumps({"transformer": {"causal_temporal_positioning": True}})}


class HeaderSource:
    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self._metadata = dict(metadata or {})

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], BFLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return self._metadata


def shapes(profile: str, prefix: str = "") -> dict[str, tuple[int, ...]]:
    av = profile in ("19b", "22b", "22b-v2.5")
    hidden, layers, ffn = (4096, 48, 16384) if av else (2048, 28, 8192)
    adaln = 36864 if profile.startswith("22b") else 24576 if av else 12288
    result = {
        "adaln_single.emb.timestep_embedder.linear_1.bias": (hidden,),
        "patchify_proj.weight": (hidden, 128),
        "proj_out.weight": (128, hidden),
        "adaln_single.linear.weight": (adaln, hidden),
        "transformer_blocks.0.ff.net.0.proj.weight": (ffn, hidden),
    }
    for index in range(layers):
        result[f"transformer_blocks.{index}.attn2.to_k.weight"] = (hidden, hidden)
    if profile != "22b-v2.5":
        result["transformer_blocks.0.ff.net.0.proj.bias"] = (ffn,)
    if av:
        audio_adaln = 18432 if profile.startswith("22b") else 12288
        result.update(
            {
                "audio_patchify_proj.weight": (2048, 128),
                "audio_proj_out.weight": (128, 2048),
                "audio_adaln_single.linear.weight": (audio_adaln, 2048),
            }
        )
    if profile.startswith("22b"):
        result["prompt_adaln_single.emb.timestep_embedder.linear_1.bias"] = (4096,)
        result["audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias"] = (2048,)
    return {prefix + key: value for key, value in result.items()}


@pytest.mark.parametrize("profile", ("2b-v0.9", "2b-v0.9.5", "19b", "22b", "22b-v2.5"))
@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_ltx_profiles(profile: str, prefix: str) -> None:
    metadata = V095_METADATA if profile == "2b-v0.9.5" else None
    source = HeaderSource(shapes(profile, prefix), metadata)
    evidence = (
        detect_ltxav(source) if profile in ("19b", "22b", "22b-v2.5") else detect_ltxv(source)
    )
    assert evidence is not None
    assert evidence.family_id == (
        "dinkster.ltxav" if profile in ("19b", "22b", "22b-v2.5") else "dinkster.ltxv"
    )
    assert evidence.fields["profile"] == profile
    assert evidence.fields["layers"] == (48 if profile in ("19b", "22b", "22b-v2.5") else 28)
    assert evidence.fields["hidden_width"] == (
        4096 if profile in ("19b", "22b", "22b-v2.5") else 2048
    )
    assert evidence.fields["av"] is (profile in ("19b", "22b", "22b-v2.5"))
    assert evidence.fields["prompt_adaln"] is profile.startswith("22b")
    assert evidence.fields["ff_bias"] is (profile != "22b-v2.5")


def test_ltx_detection_fails_closed_and_families_are_disjoint() -> None:
    wrong = shapes("19b")
    wrong["patchify_proj.weight"] = (4095, 128)
    assert detect_ltxav(HeaderSource(wrong)) is None
    video = HeaderSource(shapes("2b-v0.9"))
    assert detect_ltxav(video) is None
    assert detect_ltxv(video) is not None
    ambiguous = shapes("19b")
    ambiguous.update(
        {"model.diffusion_model." + key: value for key, value in shapes("19b").items()}
    )
    assert detect_ltxav(HeaderSource(ambiguous)) is None


def test_ltx_2b_version_split_follows_config_metadata() -> None:
    plain = detect_ltxv(HeaderSource(shapes("2b-v0.9")))
    assert plain is not None and plain.fields["profile"] == "2b-v0.9"
    causal = detect_ltxv(HeaderSource(shapes("2b-v0.9"), V095_METADATA))
    assert causal is not None and causal.fields["profile"] == "2b-v0.9.5"
    explicit_off = {"config": json.dumps({"transformer": {"causal_temporal_positioning": False}})}
    off = detect_ltxv(HeaderSource(shapes("2b-v0.9"), explicit_off))
    assert off is not None and off.fields["profile"] == "2b-v0.9"
    for malformed in (
        {"config": "not json"},
        {"config": json.dumps(["list"])},
        {"config": json.dumps({"transformer": "text"})},
        {"config": json.dumps({"transformer": {"causal_temporal_positioning": 1}})},
    ):
        assert detect_ltxv(HeaderSource(shapes("2b-v0.9"), malformed)) is None
        assert detect_ltxav(HeaderSource(shapes("19b"), malformed)) is None


def test_ltx_catalog_registration_and_multistream_refusal() -> None:
    registry = builtin_family_registry()
    assert {"dinkster.ltxv", "dinkster.ltxav"} <= set(registry.ids())
    result = registry.detect(HeaderSource(shapes("22b")))
    assert result.best is not None and result.best.family_id == "dinkster.ltxav"
    with pytest.raises(ValueError, match="multistream latent"):
        LTXAV.single_stream_latent()


def test_ltx_nvfp4_packed_input_axes_remain_detectable() -> None:
    packed = shapes("19b")
    for key, shape in tuple(packed.items()):
        if key.endswith(".weight") and len(shape) == 2:
            packed[key] = (shape[0], shape[1] // 2)
    assert detect_ltxav(HeaderSource(packed)) is not None


def test_latent_and_flow_facts_match_upstream() -> None:
    assert (
        LTXV_LATENT.channels,
        LTXV_LATENT.dimensions,
        LTXV_LATENT.temporal_causal,
        LTXV_LATENT.temporal_downscale,
        LTXV_LATENT.spatial_downscale,
    ) == (128, 3, True, 8, 32)
    assert LTXV_LATENT.rgb_factors is not None
    assert len(LTXV_LATENT.rgb_factors) == 128
    assert LTXV_LATENT.rgb_factors[0] == (0.011202, -0.00063815, -0.010021)
    assert LTXV_LATENT.rgb_bias == (-0.0571, -0.1657, -0.2512)

    assert tuple(name for name, _ in LTXAV_LATENT.streams) == ("video", "audio")
    streams = dict(LTXAV_LATENT.streams)
    video = streams["video"]
    assert (video.channels, video.dimensions, video.temporal_causal) == (128, 3, True)
    assert video.rgb_factors is not None
    assert len(video.rgb_factors) == 128
    assert video.rgb_factors[0] == (0.001135, -0.010555, -0.004925)
    assert video.rgb_bias == (-0.347892, -0.363814, -0.370287)
    audio = streams["audio"]
    assert (audio.channels, audio.dimensions) == (8, 2)

    assert LTX_SIGMAS.shift == 2.37
    assert LTX_SIGMAS.sigma_max == 1.0
    assert LTX_SIGMAS.sigma_min == pytest.approx(0.00106870286531909, rel=0, abs=1e-17)


def test_ltxv_2b_configs_carry_the_published_geometry() -> None:
    for config in (LTXV_2B_V09_CONFIG, LTXV_2B_V095_CONFIG):
        assert (config.hidden_size, config.ffn_dim, config.num_layers) == (2048, 8192, 28)
    assert LTXV_2B_V09_CONFIG.causal_temporal_positioning is False
    assert LTXV_2B_V095_CONFIG.causal_temporal_positioning is True
    with pytest.raises(ValueError, match="cross_attention_dim"):
        LTXVConfig(cross_attention_dim=1024)


def test_ltxv_layout_matches_the_checkpoint_census() -> None:
    """715 keys per published 2B checkpoint, and every shape the
    detector fingerprints agrees with the detection fixtures."""
    layout = ltxv_layout(LTXV_2B_V09_CONFIG)
    assert len(layout) == 715
    assert layout == ltxv_layout(LTXV_2B_V095_CONFIG)
    for key, shape in shapes("2b-v0.9").items():
        if key in layout:
            assert layout[key] == shape


def test_ltxav_22b_v23_config_matches_the_checkpoint_census() -> None:
    config = LTXAV_22B_V23_CONFIG
    assert (config.hidden_size, config.audio_hidden_size, config.num_layers) == (4096, 2048, 48)
    assert (config.ffn_dim, config.audio_ffn_dim, config.adaln_rows) == (16384, 8192, 9)
    assert config.caption_proj_before_connector is True
    assert config.gated_attention is True
    assert config.video_connector is not None and config.audio_connector is not None
    assert (
        config.video_connector.inner_dim,
        config.audio_connector.inner_dim,
        config.video_connector.num_layers,
        config.audio_connector.num_layers,
    ) == (4096, 2048, 8, 8)

    layout = ltxav_layout(config)
    assert len(layout) == 4444
    assert layout["video_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight"] == (
        4096,
        4096,
    )
    assert layout["audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight"] == (
        2048,
        2048,
    )
    assert layout["transformer_blocks.0.attn1.to_gate_logits.weight"] == (32, 4096)
    assert layout["transformer_blocks.0.audio_attn1.to_gate_logits.weight"] == (32, 2048)
    assert "caption_projection.linear_1.weight" not in layout

    with pytest.raises(ValueError, match="configured together"):
        LTXAVConfig(video_connector=config.video_connector)
    with pytest.raises(ValueError, match="caption_proj_before_connector"):
        LTXAVConfig(
            video_connector=config.video_connector,
            audio_connector=config.audio_connector,
        )


def test_ltxav_22b_v25_removes_only_video_feed_forward_biases() -> None:
    old_layout = ltxav_layout(LTXAV_22B_V23_CONFIG)
    config = LTXAV_22B_V25_CONFIG
    layout = ltxav_layout(config)

    assert config.ff_bias is False
    assert config.audio_ff_bias is True
    assert len(layout) == 4348
    assert set(old_layout) - set(layout) == {
        f"transformer_blocks.{index}.ff.net.{slot}.bias"
        for index in range(48)
        for slot in ("0.proj", "2")
    }


def test_ltxv_vae_block_and_config_validators_fire() -> None:
    with pytest.raises(ValueError, match="stack layers"):
        LTXVAEBlock("compress_all", layers=2)
    with pytest.raises(ValueError, match="inject noise"):
        LTXVAEBlock("compress_all", inject_noise=True)
    with pytest.raises(ValueError, match="residual skip"):
        LTXVAEBlock("res_x", residual=True)
    with pytest.raises(ValueError, match="positive"):
        LTXVAEBlock("res_x", layers=0)
    with pytest.raises(ValueError, match="unknown LTX VAE block kind"):
        LTXVAEBlock(cast(Any, "res_z"))
    with pytest.raises(ValueError, match="change channel width"):
        LTXVAEBlock("res_x", multiplier=2)
    with pytest.raises(ValueError, match="equal input and output"):
        LTXVAEBlock("res_x_y", multiplier=2, inject_noise=True)
    with pytest.raises(ValueError, match="must divide that packing"):
        LTXVAEBlock("compress_time_res", multiplier=4)
    with pytest.raises(ValueError, match="must divide that packing"):
        LTXVAEBlock("compress_time", multiplier=4, residual=True)
    with pytest.raises(ValueError, match="do not resample"):
        _ = LTXVAEBlock("res_x").stride
    with pytest.raises(ValueError, match="decoder options"):
        LTXVideoVAEConfig(
            encoder_blocks=(LTXVAEBlock("res_x", inject_noise=True),),
            decoder_blocks=(LTXVAEBlock("res_x"),),
        )
    with pytest.raises(ValueError, match="encoder blocks"):
        LTXVideoVAEConfig(
            encoder_blocks=(LTXVAEBlock("res_x"),),
            decoder_blocks=(LTXVAEBlock("compress_all_res"),),
        )
    with pytest.raises(ValueError, match="power of two"):
        LTXVideoVAEConfig(
            encoder_blocks=(LTXVAEBlock("res_x"),),
            decoder_blocks=(LTXVAEBlock("res_x"),),
            patch_size=3,
        )
    with pytest.raises(ValueError, match="name encoder and decoder blocks"):
        LTXVideoVAEConfig(encoder_blocks=(), decoder_blocks=(LTXVAEBlock("res_x"),))
    with pytest.raises(ValueError, match="space-to-depth packing"):
        LTXVideoVAEConfig(
            encoder_blocks=(LTXVAEBlock("compress_all_res", multiplier=2),),
            decoder_blocks=(LTXVAEBlock("res_x"),),
            base_channels=2,
        )
    with pytest.raises(ValueError, match="zeros or reflect"):
        LTXVideoVAEConfig(
            encoder_blocks=(LTXVAEBlock("res_x"),),
            decoder_blocks=(LTXVAEBlock("res_x"),),
            encoder_spatial_padding_mode=cast(Any, "replicate"),
        )


def test_ltxv_vae_layout_matches_the_checkpoint_census() -> None:
    """Key counts and fingerprint shapes from the published checkpoint
    headers (ltx-video-2b-v0.9 / v0.9.5, ``vae.`` prefix stripped);
    the checkpoints additionally carry three unconsumed
    per_channel_statistics aggregates the reference never loads."""
    v09 = ltxv_vae_layout(LTXV_2B_V09_VAE_CONFIG)
    assert len(v09) == 190
    assert v09["encoder.conv_in.conv.weight"] == (128, 48, 3, 3, 3)
    assert v09["encoder.conv_out.conv.weight"] == (129, 512, 3, 3, 3)
    assert v09["decoder.conv_in.conv.weight"] == (512, 128, 3, 3, 3)
    assert v09["decoder.up_blocks.2.conv.conv.weight"] == (4096, 512, 3, 3, 3)
    assert v09["decoder.conv_out.conv.weight"] == (48, 128, 3, 3, 3)
    assert v09["per_channel_statistics.std-of-means"] == (128,)
    assert "decoder.timestep_scale_multiplier" not in v09

    v095 = ltxv_vae_layout(LTXV_2B_V095_VAE_CONFIG)
    assert len(v095) == 226
    assert v095["encoder.conv_out.conv.weight"] == (129, 2048, 3, 3, 3)
    assert v095["encoder.down_blocks.1.conv.conv.weight"] == (64, 128, 3, 3, 3)
    assert v095["decoder.conv_in.conv.weight"] == (1024, 128, 3, 3, 3)
    assert v095["decoder.up_blocks.0.res_blocks.0.scale_shift_table"] == (4, 1024)
    assert v095["decoder.timestep_scale_multiplier"] == ()
    assert v095["decoder.last_scale_shift_table"] == (2, 128)

    v23 = ltxv_vae_layout(LTXAV_22B_V23_VAE_CONFIG)
    assert len(v23) == 170
    assert v23["encoder.conv_out.conv.weight"] == (129, 1024, 3, 3, 3)
    assert v23["encoder.down_blocks.7.conv.conv.weight"] == (128, 1024, 3, 3, 3)
    assert v23["decoder.conv_in.conv.weight"] == (1024, 128, 3, 3, 3)
    assert v23["decoder.up_blocks.3.conv.conv.weight"] == (4096, 512, 3, 3, 3)
    assert "decoder.timestep_scale_multiplier" not in v23
    assert (
        LTXAV_22B_V23_VAE_CONFIG.spatial_ratio,
        LTXAV_22B_V23_VAE_CONFIG.temporal_ratio,
        LTXAV_22B_V23_VAE_CONFIG.encoder_spatial_padding_mode,
        LTXAV_22B_V23_VAE_CONFIG.decoder_spatial_padding_mode,
    ) == (32, 8, "zeros", "zeros")


def test_ltxv_vae_ratios_and_decode_constants_match_upstream() -> None:
    for config in (LTXV_2B_V09_VAE_CONFIG, LTXV_2B_V095_VAE_CONFIG):
        assert (config.spatial_ratio, config.temporal_ratio) == (32, 8)
    assert LTXV_2B_V09_VAE_CONFIG.timestep_conditioning is False
    assert LTXV_2B_V095_VAE_CONFIG.timestep_conditioning is True
    assert LTXV_VAE_DECODE_NOISE_SCALE == 0.025
    assert LTXV_VAE_DECODE_TIMESTEP == 0.05


def test_ltxv_codec_descriptor_matches_the_vae_contract() -> None:
    assert LTXV_CODEC.id == "dinkster.ltxv_vae"
    assert LTXV_CODEC.kind == "video"
    assert LTXV_CODEC.latent is LTXV_LATENT
    assert LTXV_CODEC.content_channels == 3
    # The VAE streams internal chunks; the tiling planner never splits it.
    assert LTXV_CODEC.supports_tiling is False
    assert LTXV_CODEC.supported_dtypes == frozenset({BFLOAT16, FLOAT32})


def test_ltxav_codec_declares_comfy_video_tiling_defaults() -> None:
    assert LTXAV_VIDEO_CODEC.supports_tiling is True
    assert LTXAV_VIDEO_CODEC.tiling is not None
    assert LTXAV_VIDEO_CODEC.tiling.decode_tile == (999, 32, 32)
    assert LTXAV_VIDEO_CODEC.tiling.decode_overlap == (1, 8, 8)


def test_ltxv_dynamic_shift_matches_the_reference_ramp() -> None:
    assert (LTXV_SHIFT_BASE, LTXV_SHIFT_MAX) == (0.95, 2.05)
    assert (LTXV_SHIFT_TOKENS_LOW, LTXV_SHIFT_TOKENS_HIGH) == (1024, 4096)
    assert ltxv_dynamic_shift(1024) == pytest.approx(0.95)
    assert ltxv_dynamic_shift(4096) == pytest.approx(2.05)
    assert ltxv_dynamic_shift(2560) == pytest.approx(1.5)
    # The reference line is unclamped on both sides.
    assert ltxv_dynamic_shift(512) == pytest.approx(0.95 - 512 * (1.1 / 3072))
    assert ltxv_dynamic_shift(8192) == pytest.approx(2.05 + 4096 * (1.1 / 3072))
    for tokens in (0, -1, 1.0, True):
        with pytest.raises(ValueError, match="positive integer"):
            ltxv_dynamic_shift(cast(Any, tokens))


def test_ltxv_dynamic_shift_accepts_the_reference_node_endpoints() -> None:
    assert ltxv_dynamic_shift(1024, base_shift=1.25, max_shift=3.5) == pytest.approx(1.25)
    assert ltxv_dynamic_shift(4096, base_shift=1.25, max_shift=3.5) == pytest.approx(3.5)
    assert ltxv_dynamic_shift(2560, base_shift=1.25, max_shift=3.5) == pytest.approx(2.375)
    for name, value in (
        ("base_shift", -0.1),
        ("base_shift", float("nan")),
        ("max_shift", float("inf")),
        ("max_shift", True),
    ):
        with pytest.raises(ValueError, match=f"{name} must be a finite nonnegative number"):
            ltxv_dynamic_shift(4096, **cast("Any", {name: value}))


def test_ltxv_dynamic_shift_matches_pinned_comfyui() -> None:
    golden = json.loads((Path(__file__).parent / "goldens/ltxv_dynamic_shift.json").read_text())
    assert golden["comfyui_commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    for case in golden["cases"]:
        assert ltxv_dynamic_shift(
            case["tokens"],
            max_shift=case["max_shift"],
            base_shift=case["base_shift"],
        ) == pytest.approx(case["shift"])


def test_generated_keyframe_metadata_is_strict_and_frozen() -> None:
    value = LTXGeneratedKeyframes(16, 2, 3)

    assert dataclasses.astuple(value) == (16, 2, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.num_keyframes = 4  # type: ignore[misc]
    for fields in ((16.0, 2, 3), (16, True, 3), (16, 2, None)):
        with pytest.raises(TypeError, match="exact integers"):
            LTXGeneratedKeyframes(*cast("Any", fields))
    for fields in ((0, 2, 3), (16, -1, 3), (16, 2, -1)):
        with pytest.raises(ValueError, match="positive|nonnegative"):
            LTXGeneratedKeyframes(*fields)


def test_ltxv_t5_profile_only_lowers_the_flux_padding_floor() -> None:
    assert T5_XXL_LTXV_PROFILE.min_length == 128
    assert T5_XXL_FLUX_PROFILE.min_length == 256
    assert T5_XXL_LTXV_PROFILE == dataclasses.replace(T5_XXL_FLUX_PROFILE, min_length=128)
