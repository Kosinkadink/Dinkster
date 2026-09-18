"""The native LTX audio VAE and vocoder against the executed reference.

Every golden in goldens/ltx_audio_vae_goldens.json was produced by
RUNNING the reference AudioVAE and vocoders (comfy/ldm/lightricks @ the
audited baseline, tools/gen_ltx_audio_vae_goldens.py). Weights come
from the shared deterministic hash (ltx_vae_fill.py) and inputs from
unet_fill's ``hashed_input`` namespace.

The torchaudio-backed front end (windowed-sinc resample, log-mel
spectrogram) is held bit-exact against LIVE torchaudio executed in the
same process (skipped when torchaudio is not installed): Dinkster
reimplements those kernels in plain torch, and any drift there is a
transcription bug, never a tolerance case. The STORED-golden replay of
the mel spectrogram carries a tiny documented tolerance instead,
because the mel filterbank matmul changes float reduction order with
the host's torch thread partitioning, so its output is not
bit-portable across hosts (issue #549). Network outputs replay under
the same tight tolerance as the video VAE goldens.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_VOCODER_CONFIG,
    LTXAudioVAEConfig,
    LTXVocoderBWEConfig,
    LTXVocoderConfig,
    ltx_audio_vae_layout,
    ltx_vocoder_bwe_layout,
    ltx_vocoder_layout,
)
from dinkster_inference_torch import (
    LTXAudioVAE,
    LTXVocoder,
    LTXVocoderWithBWE,
    ltx_audio_resample,
    ltx_audio_waveform_to_mel,
    ltx_vocoder_features,
)
from dinkster_inference_torch.module_residency import enroll_component
from golden_files import load_platform_golden
from ltx_vae_fill import fill_vae_state_dict
from unet_fill import hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "ltx_audio_vae_goldens.json")

AUDIO_VAE_CASES = ("audio_vae_tiny_encode", "audio_vae_tiny_decode")
VOCODER_CASES = tuple(sorted(case for case in GOLDENS["cases"] if case.startswith("vocoder_")))


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str, prefix: str = "") -> list[tuple[str, list[int]]]:
    return [
        (key, list(shape))
        for key, shape in GOLDENS["cases"][case]["state_dict"]
        if key.startswith(prefix)
    ]


def audio_vae_config(case: str) -> LTXAudioVAEConfig:
    spec = GOLDENS["cases"][case]["config"]
    return LTXAudioVAEConfig(
        in_channels=spec["in_channels"],
        out_channels=spec["out_channels"],
        base_channels=spec["base_channels"],
        ch_mult=tuple(spec["ch_mult"]),
        num_res_blocks=spec["num_res_blocks"],
        z_channels=spec["z_channels"],
        mel_bins=spec["mel_bins"],
        sampling_rate=spec["sampling_rate"],
        mel_hop_length=spec["mel_hop_length"],
        n_fft=spec["n_fft"],
    )


def vocoder_config(spec: dict[str, Any]) -> LTXVocoderConfig:
    return LTXVocoderConfig(
        upsample_rates=tuple(spec["upsample_rates"]),
        upsample_kernel_sizes=tuple(spec["upsample_kernel_sizes"]),
        resblock=spec["resblock"],
        stereo=spec["stereo"],
        activation=spec.get("activation", "snake"),
        resblock_kernel_sizes=tuple(spec["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(
            tuple(dilation) for dilation in spec["resblock_dilation_sizes"]
        ),
        upsample_initial_channel=spec["upsample_initial_channel"],
        use_bias_at_final=spec.get("use_bias_at_final", True),
        use_tanh_at_final=spec.get("use_tanh_at_final", True),
        apply_final_activation=spec.get("apply_final_activation", True),
    )


def bwe_config(spec: dict[str, Any]) -> LTXVocoderBWEConfig:
    bwe = spec["bwe"]
    return LTXVocoderBWEConfig(
        vocoder=vocoder_config(spec["vocoder"]),
        bwe_generator=vocoder_config({**bwe, "apply_final_activation": False}),
        input_sampling_rate=bwe["input_sampling_rate"],
        output_sampling_rate=bwe["output_sampling_rate"],
        hop_length=bwe["hop_length"],
        n_fft=bwe["n_fft"],
        num_mels=bwe["num_mels"],
    )


def build_audio_vae(case: str) -> LTXAudioVAE:
    vae = LTXAudioVAE(audio_vae_config(case))
    fill = fill_vae_state_dict(golden_entries(case))
    vae.load_state_dict(
        {
            key.removeprefix("autoencoder."): value
            for key, value in fill.items()
            if key.startswith("autoencoder.")
        },
        strict=True,
    )
    return vae


def build_vocoder(case: str) -> LTXVocoder | LTXVocoderWithBWE:
    spec = GOLDENS["cases"][case]["reference_config"]
    if "bwe" in spec:
        module: LTXVocoder | LTXVocoderWithBWE = LTXVocoderWithBWE(bwe_config(spec))
    else:
        module = LTXVocoder(vocoder_config(spec))
    module.load_state_dict(fill_vae_state_dict(golden_entries(case)), strict=True)
    return module


def test_audio_codec_state_is_admitted_by_component_residency() -> None:
    enroll_component(
        build_audio_vae("audio_vae_tiny_decode"),
        load_device="cpu",
        offload_device="cpu",
    )
    enroll_component(
        build_vocoder("vocoder_bwe"),
        load_device="cpu",
        offload_device="cpu",
    )


# ------------------------------------------------------ key layout


def test_full_size_module_matches_reference_layout() -> None:
    """The real 19B audio VAE + vocoder, constructed on the meta device
    (initless factories never touch the storage), against the reference
    AudioVAE's own full-size listing."""
    with torch.device("meta"):
        vae = LTXAudioVAE(LTXAV_19B_AUDIO_VAE_CONFIG)
        vocoder = LTXVocoder(LTXAV_19B_VOCODER_CONFIG)
    ours = sorted(
        [("autoencoder." + key, list(value.shape)) for key, value in vae.state_dict().items()]
        + [("vocoder." + key, list(value.shape)) for key, value in vocoder.state_dict().items()]
    )
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["ltxav_19b_audio_vae"]]
    assert ours == golden


def test_torch_free_layout_predicts_the_full_size_reference() -> None:
    predicted = sorted(
        [
            ("autoencoder." + key, list(shape))
            for key, shape in ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG).items()
        ]
        + [
            ("vocoder." + key, list(shape))
            for key, shape in ltx_vocoder_layout(LTXAV_19B_VOCODER_CONFIG).items()
        ]
    )
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["ltxav_19b_audio_vae"]]
    assert predicted == golden


@pytest.mark.parametrize("case", AUDIO_VAE_CASES)
def test_audio_vae_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted(
        ("autoencoder." + key, list(value.shape))
        for key, value in build_audio_vae(case).state_dict().items()
    )
    assert ours == golden_entries(case, "autoencoder.")


@pytest.mark.parametrize("case", AUDIO_VAE_CASES)
def test_torch_free_audio_vae_layout_predicts_the_module(case: str) -> None:
    predicted = sorted(
        ("autoencoder." + key, list(shape))
        for key, shape in ltx_audio_vae_layout(audio_vae_config(case)).items()
    )
    assert predicted == golden_entries(case, "autoencoder.")


@pytest.mark.parametrize("case", VOCODER_CASES)
def test_vocoder_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted(
        (key, list(value.shape)) for key, value in build_vocoder(case).state_dict().items()
    )
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", VOCODER_CASES)
def test_torch_free_vocoder_layout_predicts_the_module(case: str) -> None:
    spec = GOLDENS["cases"][case]["reference_config"]
    if "bwe" in spec:
        layout = ltx_vocoder_bwe_layout(bwe_config(spec))
    else:
        layout = ltx_vocoder_layout(vocoder_config(spec))
    predicted = sorted((key, list(shape)) for key, shape in layout.items())
    assert predicted == golden_entries(case)


# ------------------------------------------------- front-end replay


@pytest.mark.parametrize("name", ["resample_24000_to_16000", "resample_44100_to_16000"])
def test_resample_is_bit_equal_to_torchaudio(name: str) -> None:
    spec = GOLDENS["transforms"][name]
    x = hashed_input(f"{name}:x", spec["input_shape"])
    resampled = ltx_audio_resample(x, spec["source_rate"], spec["target_rate"])
    assert torch.equal(resampled, dec(spec["output"]))


# The stored mel goldens are not bit-portable across hosts: MelScale's
# 64x513 @ 513xT filterbank matmul flips float reduction order with the
# host's torch thread partitioning (issue #549; reproduced on a 64-thread
# Threadripper for mel_44100 and on another host for mel_16000, while
# threads 1..32 on the same build match the goldens bit-exactly). The
# executed reference itself reproduces the identical divergence against
# the stored golden, so this is reduction-order drift, never port drift;
# the same-process live-torchaudio test below keeps the transcription
# contract exact. Observed drift: 38/2688 elements, max abs 2.384186e-07,
# max rel 3.122455e-07, max 3 ulp; the tolerance carries roughly 3-4x
# headroom over that bound (4.2x abs, 3.2x rel).
MEL_GOLDEN_RTOL = 1e-6
MEL_GOLDEN_ATOL = 1e-6


@pytest.mark.parametrize("name", ["mel_16000", "mel_44100"])
def test_mel_spectrogram_matches_executed_torchaudio(name: str) -> None:
    spec = GOLDENS["transforms"][name]
    x = hashed_input(f"{name}:x", spec["input_shape"])
    mel = ltx_audio_waveform_to_mel(x, spec["source_rate"], LTXAV_19B_AUDIO_VAE_CONFIG)
    torch.testing.assert_close(mel, dec(spec["output"]), rtol=MEL_GOLDEN_RTOL, atol=MEL_GOLDEN_ATOL)


@pytest.mark.parametrize(
    "name",
    ["resample_24000_to_16000", "resample_44100_to_16000", "mel_16000", "mel_44100"],
)
def test_front_end_is_bit_equal_to_live_torchaudio(name: str) -> None:
    """The exact transcription contract: in the same process the plain-torch
    front end is bit-identical to torchaudio (the reference's backend),
    on any host. Runs only where torchaudio is installed."""
    torchaudio = pytest.importorskip("torchaudio")
    spec = GOLDENS["transforms"][name]
    x = hashed_input(f"{name}:x", spec["input_shape"])
    config = LTXAV_19B_AUDIO_VAE_CONFIG
    with torch.no_grad():
        if name.startswith("resample_"):
            actual = ltx_audio_resample(x, spec["source_rate"], spec["target_rate"])
            expected = torchaudio.functional.resample(x, spec["source_rate"], spec["target_rate"])
        else:
            actual = ltx_audio_waveform_to_mel(x, spec["source_rate"], config)
            waveform = x
            if spec["source_rate"] != config.sampling_rate:
                waveform = torchaudio.functional.resample(
                    x, spec["source_rate"], config.sampling_rate
                )
            # The reference AudioPreprocessor.waveform_to_mel's exact
            # construction (comfy/ldm/lightricks/vae/audio_vae.py @ the
            # audited baseline).
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sampling_rate,
                n_fft=config.n_fft,
                win_length=config.n_fft,
                hop_length=config.mel_hop_length,
                f_min=0.0,
                f_max=config.sampling_rate / 2.0,
                n_mels=config.mel_bins,
                window_fn=torch.hann_window,
                center=True,
                pad_mode="reflect",
                power=1.0,
                mel_scale="slaney",
                norm="slaney",
            )
            mel = torch.log(torch.clamp(transform(waveform), min=1e-5))
            expected = mel.permute(0, 1, 3, 2).contiguous()
    assert torch.equal(actual, expected)


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", AUDIO_VAE_CASES)
def test_encode_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    vae = build_audio_vae(case)
    x = hashed_input(f"{case}:x", spec["input_shape"])
    with torch.no_grad():
        latent = vae.encode(x, sample_rate=spec["input_sample_rate"])
    torch.testing.assert_close(latent, dec(spec["encode"]), rtol=1e-4, atol=1e-5)


def test_encode_expands_mono_input() -> None:
    spec = GOLDENS["cases"]["audio_vae_tiny_encode"]
    vae = build_audio_vae("audio_vae_tiny_encode")
    x = hashed_input("audio_vae_tiny_encode:x_mono", (1, 1, spec["input_shape"][2]))
    with torch.no_grad():
        latent = vae.encode(x, sample_rate=spec["input_sample_rate"])
    torch.testing.assert_close(latent, dec(spec["encode_mono"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", AUDIO_VAE_CASES)
def test_decode_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    vae = build_audio_vae(case)
    latent = hashed_input(f"{case}:latent", spec["latent_shape"])
    with torch.no_grad():
        mel = vae.decode(latent)
    torch.testing.assert_close(mel, dec(spec["decode_mel"]), rtol=1e-4, atol=1e-5)


def test_decode_waveform_matches_executed_reference() -> None:
    """The full decode chain: latents to mel through the audio VAE, then
    the wrapper's vocoder handoff into the tiny vocoder."""
    spec = GOLDENS["cases"]["audio_vae_tiny_decode"]
    vae = build_audio_vae("audio_vae_tiny_decode")
    vocoder = LTXVocoder(vocoder_config(spec["vocoder_reference_config"]))
    fill = fill_vae_state_dict(golden_entries("audio_vae_tiny_decode"))
    vocoder.load_state_dict(
        {
            key.removeprefix("vocoder."): value
            for key, value in fill.items()
            if key.startswith("vocoder.")
        },
        strict=True,
    )
    latent = hashed_input("audio_vae_tiny_decode:latent", spec["latent_shape"])
    with torch.no_grad():
        mel = vae.decode(latent)
        waveform = vocoder(ltx_vocoder_features(mel, 2))
    torch.testing.assert_close(waveform, dec(spec["decode_waveform"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", VOCODER_CASES)
def test_vocoder_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    module = build_vocoder(case)
    x = hashed_input(f"{case}:x", spec["input_shape"])
    with torch.no_grad():
        waveform = module(x)
    torch.testing.assert_close(waveform, dec(spec["output"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", VOCODER_CASES)
def test_constructed_filters_are_bit_equal_to_the_reference(case: str) -> None:
    """Anti-aliasing filters the modules compute at construction (AMP
    activations, the BWE skip resampler), pinned bit-exactly against the
    reference's fresh-module buffers."""
    spec = GOLDENS["cases"][case]
    if not spec["computed_filters"]:
        pytest.skip("no constructed filters in this geometry")
    config = spec["reference_config"]
    if "bwe" in config:
        module: LTXVocoder | LTXVocoderWithBWE = LTXVocoderWithBWE(bwe_config(config))
    else:
        module = LTXVocoder(vocoder_config(config))
    # named_buffers also yields non-persistent buffers (the BWE skip
    # resampler's filter, which never appears in a state dict).
    buffers = dict(module.named_buffers())
    for key, golden in spec["computed_filters"].items():
        assert torch.equal(buffers[key], dec(golden)), key
