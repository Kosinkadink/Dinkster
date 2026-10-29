"""Torch-free LTX audio VAE and vocoder catalog tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_inference.ltx_audio import (
    LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    LTX_VOCODER_MEL_BINS,
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_VOCODER_CONFIG,
    LTXAV_BWE_VOCODER_CONFIG,
    LTXAudioVAEConfig,
    LTXVocoderBWEConfig,
    LTXVocoderConfig,
    ltx_audio_latents_from_frames,
    ltx_audio_output_sample_rate,
    ltx_audio_vae_layout,
    ltx_vocoder_bwe_layout,
    ltx_vocoder_layout,
)


def test_audio_vae_config_validators_fire() -> None:
    with pytest.raises(ValueError, match="positive"):
        LTXAudioVAEConfig(z_channels=0)
    with pytest.raises(ValueError, match="mono or stereo"):
        LTXAudioVAEConfig(out_channels=3)
    with pytest.raises(ValueError, match="multipliers must be positive"):
        LTXAudioVAEConfig(ch_mult=(1, 0, 4))
    with pytest.raises(ValueError, match="4x latent downsampling"):
        LTXAudioVAEConfig(ch_mult=(1, 2))
    with pytest.raises(ValueError, match="4x-downsampled"):
        LTXAudioVAEConfig(mel_bins=66)
    with pytest.raises(ValueError, match="at least one hop"):
        LTXAudioVAEConfig(n_fft=128, mel_hop_length=160)


def test_vocoder_config_validators_fire() -> None:
    with pytest.raises(ValueError, match="unknown vocoder resblock kind"):
        LTXVocoderConfig(upsample_rates=(2,), upsample_kernel_sizes=(4,), resblock=cast(Any, "3"))
    with pytest.raises(ValueError, match="unknown vocoder activation"):
        LTXVocoderConfig(
            upsample_rates=(2,), upsample_kernel_sizes=(4,), activation=cast(Any, "relu")
        )
    with pytest.raises(ValueError, match="upsample at least once"):
        LTXVocoderConfig(upsample_rates=(), upsample_kernel_sizes=())
    with pytest.raises(ValueError, match="needs a kernel size"):
        LTXVocoderConfig(upsample_rates=(2, 2), upsample_kernel_sizes=(4,))
    with pytest.raises(ValueError, match="needs its dilation walk"):
        LTXVocoderConfig(
            upsample_rates=(2,),
            upsample_kernel_sizes=(4,),
            resblock_kernel_sizes=(3, 5),
            resblock_dilation_sizes=((1, 3, 5),),
        )
    with pytest.raises(ValueError, match="stacks 3 dilated"):
        LTXVocoderConfig(
            upsample_rates=(2,),
            upsample_kernel_sizes=(4,),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3),),
        )
    with pytest.raises(ValueError, match="stacks 2 dilated"):
        LTXVocoderConfig(
            upsample_rates=(2,),
            upsample_kernel_sizes=(4,),
            resblock="2",
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
        )
    with pytest.raises(ValueError, match="halves its width"):
        LTXVocoderConfig(
            upsample_rates=(2, 2), upsample_kernel_sizes=(4, 4), upsample_initial_channel=6
        )


def _tiny_vocoder(**overrides: Any) -> LTXVocoderConfig:
    spec: dict[str, Any] = {
        "upsample_rates": (2, 2),
        "upsample_kernel_sizes": (4, 4),
        "resblock_kernel_sizes": (3,),
        "resblock_dilation_sizes": ((1, 3, 5),),
        "upsample_initial_channel": 32,
        "apply_final_activation": False,
    }
    spec.update(overrides)
    return LTXVocoderConfig(**spec)


def _tiny_bwe(**overrides: Any) -> LTXVocoderBWEConfig:
    spec: dict[str, Any] = {
        "vocoder": _tiny_vocoder(apply_final_activation=True),
        "bwe_generator": _tiny_vocoder(),
        "input_sampling_rate": 8000,
        "output_sampling_rate": 16000,
        "hop_length": 16,
        "n_fft": 64,
        "num_mels": 64,
    }
    spec.update(overrides)
    return LTXVocoderBWEConfig(**spec)


def test_bwe_config_validators_fire() -> None:
    assert _tiny_bwe().resample_ratio == 2
    with pytest.raises(ValueError, match="must be stereo"):
        _tiny_bwe(bwe_generator=_tiny_vocoder(stereo=False))
    with pytest.raises(ValueError, match="without a final activation"):
        _tiny_bwe(bwe_generator=_tiny_vocoder(apply_final_activation=True))
    with pytest.raises(ValueError, match="counts must be positive"):
        _tiny_bwe(hop_length=0)
    with pytest.raises(ValueError, match="half the BWE generator input width"):
        _tiny_bwe(num_mels=32)
    with pytest.raises(ValueError, match="integer ratio"):
        _tiny_bwe(output_sampling_rate=12000)
    with pytest.raises(ValueError, match="integer ratio"):
        _tiny_bwe(output_sampling_rate=8000)
    with pytest.raises(ValueError, match="cover at least one hop"):
        _tiny_bwe(n_fft=8)


def test_19b_configs_carry_the_published_geometry() -> None:
    config = LTXAV_19B_AUDIO_VAE_CONFIG
    assert config.latents_per_second == 25.0
    assert config.latent_frequency_bins == 16
    assert config.statistics_channels == 128
    vocoder = LTXAV_19B_VOCODER_CONFIG
    assert vocoder.upsample_factor == 240
    assert vocoder.in_channels == 2 * LTX_VOCODER_MEL_BINS
    assert vocoder.final_channels == 32
    assert ltx_audio_output_sample_rate(config, vocoder) == 24000
    explicit = LTXVocoderConfig(
        upsample_rates=(2,), upsample_kernel_sizes=(4,), output_sample_rate=48000
    )
    assert ltx_audio_output_sample_rate(config, explicit) == 48000


def test_standalone_bwe_vocoder_carries_the_published_geometry() -> None:
    config = LTXAV_BWE_VOCODER_CONFIG
    assert config.vocoder.upsample_factor == 160
    assert config.bwe_generator.upsample_factor == 240
    assert config.resample_ratio == 3
    assert len(ltx_vocoder_bwe_layout(config)) == 1227


def test_latents_from_frames_matches_the_reference_math() -> None:
    config = LTXAV_19B_AUDIO_VAE_CONFIG
    assert LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR == 4
    assert ltx_audio_latents_from_frames(config, 97, 24) == 101
    assert ltx_audio_latents_from_frames(config, 121, 24) == 126
    assert ltx_audio_latents_from_frames(config, 48, 12) == 100
    assert ltx_audio_latents_from_frames(config, 1, 25) == 1
    # 1 frame at 10 fps spans 2.5 latents; built-in round gives 2 (half to even).
    assert ltx_audio_latents_from_frames(config, 1, 10) == 2


def test_audio_vae_layout_matches_the_checkpoint_census() -> None:
    """Key counts and fingerprint shapes from the published checkpoint
    header (Lightricks/LTX-2 ltx-2-19b-dev, ``audio_vae.`` prefix
    stripped)."""
    layout = ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG)
    assert len(layout) == 102
    assert sum(1 for key in layout if key.startswith("encoder.")) == 44
    assert sum(1 for key in layout if key.startswith("decoder.")) == 56
    assert layout["encoder.conv_in.conv.weight"] == (128, 2, 3, 3)
    assert layout["encoder.conv_out.conv.weight"] == (16, 512, 3, 3)
    assert layout["encoder.down.0.downsample.conv.weight"] == (128, 128, 3, 3)
    assert layout["encoder.down.1.block.0.nin_shortcut.conv.weight"] == (256, 128, 1, 1)
    assert "encoder.down.2.downsample.conv.weight" not in layout
    assert layout["decoder.conv_in.conv.weight"] == (512, 8, 3, 3)
    assert layout["decoder.up.1.upsample.conv.conv.weight"] == (256, 256, 3, 3)
    assert "decoder.up.0.upsample.conv.conv.weight" not in layout
    assert layout["decoder.conv_out.conv.weight"] == (2, 128, 3, 3)
    assert layout["per_channel_statistics.std-of-means"] == (128,)
    assert layout["per_channel_statistics.mean-of-means"] == (128,)


def test_vocoder_layout_matches_the_checkpoint_census() -> None:
    """Key counts and fingerprint shapes from the published checkpoint
    header (``vocoder.`` prefix stripped): conv_pre/post, five
    transposed-conv stages, and 15 residual blocks of 12 tensors."""
    layout = ltx_vocoder_layout(LTXAV_19B_VOCODER_CONFIG)
    assert len(layout) == 194
    assert layout["conv_pre.weight"] == (1024, 128, 7)
    assert layout["ups.1.weight"] == (512, 256, 15)
    assert layout["ups.4.weight"] == (64, 32, 4)
    assert layout["resblocks.0.convs1.0.weight"] == (512, 512, 3)
    assert layout["resblocks.14.convs2.2.weight"] == (32, 32, 11)
    assert layout["conv_post.weight"] == (2, 32, 7)
    assert layout["conv_post.bias"] == (2,)
    assert not any(".acts" in key or key.startswith("act_post") for key in layout)


def test_amp_and_bwe_layouts_carry_their_extra_tensors() -> None:
    amp = ltx_vocoder_layout(
        _tiny_vocoder(resblock="AMP1", activation="snakebeta", apply_final_activation=True)
    )
    assert amp["resblocks.0.acts1.0.act.alpha"] == (16,)
    assert amp["resblocks.0.acts1.0.act.beta"] == (16,)
    assert amp["resblocks.0.acts1.0.upsample.filter"] == (1, 1, 12)
    assert amp["resblocks.0.acts2.2.downsample.lowpass.filter"] == (1, 1, 12)
    assert amp["act_post.act.alpha"] == (8,)

    no_bias = ltx_vocoder_layout(_tiny_vocoder(use_bias_at_final=False))
    assert "conv_post.bias" not in no_bias

    bwe = ltx_vocoder_bwe_layout(_tiny_bwe())
    assert bwe["vocoder.conv_pre.weight"] == (32, 128, 7)
    assert bwe["bwe_generator.conv_pre.weight"] == (32, 128, 7)
    assert bwe["mel_stft.mel_basis"] == (64, 33)
    assert bwe["mel_stft.stft_fn.forward_basis"] == (66, 1, 64)
    assert bwe["mel_stft.stft_fn.inverse_basis"] == (66, 1, 64)
    assert not any(key.startswith("resampler.") for key in bwe)
