"""Torch-free LTX-2 audio VAE and vocoder checkpoint facts.

Geometry follows ``comfy/ldm/lightricks/vae/audio_vae.py``,
``comfy/ldm/lightricks/vae/causal_audio_autoencoder.py``, and
``comfy/ldm/lightricks/vocoders/vocoder.py`` at ComfyUI ``b78cec87``.
Checkpoints store the autoencoder under ``audio_vae.`` and the vocoder
under ``vocoder.``; the layout functions here list keys without those
prefixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

#: The reference wrapper hardcodes 4x latent downsampling everywhere it
#: does time or frequency math: the statistics patch width, the decode
#: target length, and the latents-per-second rate (audio_vae.py
#: LATENT_DOWNSAMPLE_FACTOR @ b78cec87).
LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4

#: The reference vocoder hardcodes its mel input width: 64 bins per
#: audio channel, concatenated to 128 for stereo (Vocoder @ b78cec87).
LTX_VOCODER_MEL_BINS = 64


@dataclass(frozen=True, slots=True)
class LTXAudioVAEConfig:
    """Exact geometry of one LTX causal audio VAE.

    Field names follow the reference ``ddconfig`` dictionary and the
    surrounding checkpoint config (CausalAudioAutoencoder @ b78cec87).
    Aspects every supported checkpoint shares are pinned rather than
    configurable: pixel normalization, no attention, height (time)
    causality, doubled latent channels (the wrapper always takes the
    posterior mode, halving the encoder output), zero dropout, and
    strided-convolution resampling.
    """

    in_channels: int = 2
    out_channels: int = 2
    base_channels: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    z_channels: int = 8
    mel_bins: int = 64
    sampling_rate: int = 16000
    mel_hop_length: int = 160
    n_fft: int = 1024

    def __post_init__(self) -> None:
        counts = (
            self.in_channels,
            self.out_channels,
            self.base_channels,
            self.num_res_blocks,
            self.z_channels,
            self.mel_bins,
            self.sampling_rate,
            self.mel_hop_length,
            self.n_fft,
        )
        if any(count <= 0 for count in counts):
            raise ValueError("LTX audio VAE counts must be positive")
        if self.out_channels not in (1, 2):
            raise ValueError("the vocoder handoff supports mono or stereo output only")
        if not self.ch_mult or any(mult <= 0 for mult in self.ch_mult):
            raise ValueError("LTX audio VAE channel multipliers must be positive")
        if 2 ** (len(self.ch_mult) - 1) != LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR:
            raise ValueError(
                "the reference wrapper hardcodes 4x latent downsampling; the encoder"
                f" level count {len(self.ch_mult)} must halve time and frequency twice"
            )
        if self.mel_bins % LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR:
            raise ValueError("LTX audio VAE mel bins must divide into 4x-downsampled latents")
        if self.n_fft < self.mel_hop_length:
            raise ValueError("the mel window must cover at least one hop")

    @property
    def latent_frequency_bins(self) -> int:
        return self.mel_bins // LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR

    @property
    def statistics_channels(self) -> int:
        """Width of the per-channel statistics: latents are patched to
        (batch, time, channels * frequency) before normalization."""
        return self.z_channels * self.latent_frequency_bins

    @property
    def latents_per_second(self) -> float:
        return self.sampling_rate / self.mel_hop_length / LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR


def ltx_audio_latents_from_frames(config: LTXAudioVAEConfig, frames: int, frame_rate: float) -> int:
    """The reference's ``num_of_latents_from_frames``: latent steps
    nearest the video clip's duration (built-in ``round``, so exact
    halves round to even like the reference)."""
    return round((float(frames) / frame_rate) * config.latents_per_second)


def _audio_causal_conv(
    key: str, out_channels: int, in_channels: int, kernel: int = 3
) -> dict[str, tuple[int, ...]]:
    return {
        key + ".conv.weight": (out_channels, in_channels, kernel, kernel),
        key + ".conv.bias": (out_channels,),
    }


def _audio_res_block(key: str, in_channels: int, out_channels: int) -> dict[str, tuple[int, ...]]:
    layout = _audio_causal_conv(key + ".conv1", out_channels, in_channels)
    layout.update(_audio_causal_conv(key + ".conv2", out_channels, out_channels))
    if in_channels != out_channels:
        layout.update(_audio_causal_conv(key + ".nin_shortcut", out_channels, in_channels, 1))
    return layout


def ltx_audio_vae_layout(config: LTXAudioVAEConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the LTX audio VAE, without the
    checkpoint's ``audio_vae.`` prefix."""
    ch = config.base_channels
    in_ch_mult = (1,) + tuple(config.ch_mult)
    layout = _audio_causal_conv("encoder.conv_in", ch, config.in_channels)
    for level, mult in enumerate(config.ch_mult):
        into = ch * in_ch_mult[level]
        out = ch * mult
        for block in range(config.num_res_blocks):
            layout.update(_audio_res_block(f"encoder.down.{level}.block.{block}", into, out))
            into = out
        if level != len(config.ch_mult) - 1:
            # The encoder downsample is a bare strided conv, not a causal wrapper.
            layout[f"encoder.down.{level}.downsample.conv.weight"] = (out, out, 3, 3)
            layout[f"encoder.down.{level}.downsample.conv.bias"] = (out,)
    deepest = ch * config.ch_mult[-1]
    layout.update(_audio_res_block("encoder.mid.block_1", deepest, deepest))
    layout.update(_audio_res_block("encoder.mid.block_2", deepest, deepest))
    layout.update(_audio_causal_conv("encoder.conv_out", 2 * config.z_channels, deepest))

    layout.update(_audio_causal_conv("decoder.conv_in", deepest, config.z_channels))
    layout.update(_audio_res_block("decoder.mid.block_1", deepest, deepest))
    layout.update(_audio_res_block("decoder.mid.block_2", deepest, deepest))
    into = deepest
    for level in reversed(range(len(config.ch_mult))):
        out = ch * config.ch_mult[level]
        for block in range(config.num_res_blocks + 1):
            layout.update(_audio_res_block(f"decoder.up.{level}.block.{block}", into, out))
            into = out
        if level != 0:
            layout.update(_audio_causal_conv(f"decoder.up.{level}.upsample.conv", into, into))
    layout.update(_audio_causal_conv("decoder.conv_out", config.out_channels, into))

    layout["per_channel_statistics.std-of-means"] = (config.statistics_channels,)
    layout["per_channel_statistics.mean-of-means"] = (config.statistics_channels,)
    return layout


LTXVocoderResblockKind = Literal["1", "2", "AMP1"]
LTXVocoderActivation = Literal["snake", "snakebeta"]

#: Kernel width of the anti-aliasing filters inside every AMP block
#: activation (Activation1d defaults @ b78cec87).
_AMP_FILTER_TAPS = 12


@dataclass(frozen=True, slots=True)
class LTXVocoderConfig:
    """Exact geometry of one LTX vocoder (Vocoder @ b78cec87).

    ``resblock`` "1" and "2" are the HiFi-GAN residual blocks; "AMP1" is
    the BigVGAN anti-aliased block whose ``activation`` picks Snake or
    SnakeBeta. The mel input width is hardcoded by the reference: 64
    bins per audio channel."""

    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock: LTXVocoderResblockKind = "1"
    stereo: bool = True
    activation: LTXVocoderActivation = "snake"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_initial_channel: int = 1024
    use_bias_at_final: bool = True
    use_tanh_at_final: bool = True
    apply_final_activation: bool = True
    output_sample_rate: int | None = None

    def __post_init__(self) -> None:
        if self.resblock not in ("1", "2", "AMP1"):
            raise ValueError(f"unknown vocoder resblock kind {self.resblock!r}")
        if self.activation not in ("snake", "snakebeta"):
            raise ValueError(f"unknown vocoder activation {self.activation!r}")
        if not self.upsample_rates:
            raise ValueError("LTX vocoders must upsample at least once")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("every vocoder upsample rate needs a kernel size")
        if not self.resblock_kernel_sizes:
            raise ValueError("LTX vocoders must carry at least one residual block per stage")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("every vocoder resblock kernel needs its dilation walk")
        stacked = 2 if self.resblock == "2" else 3
        for dilation in self.resblock_dilation_sizes:
            if len(dilation) != stacked:
                raise ValueError(
                    f"resblock kind {self.resblock!r} stacks {stacked} dilated convolutions"
                )
            if any(step <= 0 for step in dilation):
                raise ValueError("vocoder dilations must be positive")
        counts = (
            self.upsample_rates
            + self.upsample_kernel_sizes
            + self.resblock_kernel_sizes
            + (self.upsample_initial_channel,)
        )
        if any(count <= 0 for count in counts):
            raise ValueError("LTX vocoder counts must be positive")
        if self.upsample_initial_channel % 2 ** len(self.upsample_rates):
            raise ValueError("the vocoder halves its width at every upsampling stage")
        if self.output_sample_rate is not None and self.output_sample_rate <= 0:
            raise ValueError("the vocoder output sample rate must be positive")

    @property
    def audio_channels(self) -> int:
        return 2 if self.stereo else 1

    @property
    def in_channels(self) -> int:
        return LTX_VOCODER_MEL_BINS * self.audio_channels

    @property
    def final_channels(self) -> int:
        return self.upsample_initial_channel // 2 ** len(self.upsample_rates)

    @property
    def upsample_factor(self) -> int:
        return math.prod(self.upsample_rates)


def ltx_audio_output_sample_rate(config: LTXAudioVAEConfig, vocoder: LTXVocoderConfig) -> int:
    """The reference wrapper's output rate: the vocoder's explicit rate
    when present, otherwise the mel frame rate times the vocoder's
    upsampling."""
    if vocoder.output_sample_rate is not None:
        return vocoder.output_sample_rate
    return int(config.sampling_rate * vocoder.upsample_factor / config.mel_hop_length)


def _vocoder_conv1d(
    key: str, out_channels: int, in_channels: int, kernel: int
) -> dict[str, tuple[int, ...]]:
    return {
        key + ".weight": (out_channels, in_channels, kernel),
        key + ".bias": (out_channels,),
    }


def _vocoder_amp_activation(
    key: str, channels: int, activation: LTXVocoderActivation
) -> dict[str, tuple[int, ...]]:
    layout: dict[str, tuple[int, ...]] = {key + ".act.alpha": (channels,)}
    if activation == "snakebeta":
        layout[key + ".act.beta"] = (channels,)
    layout[key + ".upsample.filter"] = (1, 1, _AMP_FILTER_TAPS)
    layout[key + ".downsample.lowpass.filter"] = (1, 1, _AMP_FILTER_TAPS)
    return layout


def ltx_vocoder_layout(config: LTXVocoderConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the LTX vocoder, without the
    checkpoint's ``vocoder.`` prefix. AMP anti-aliasing filters are
    persistent buffers the reference computes at construction, so they
    appear here even though checkpoints may overwrite them."""
    layout = _vocoder_conv1d("conv_pre", config.upsample_initial_channel, config.in_channels, 7)
    for index, kernel in enumerate(config.upsample_kernel_sizes):
        into = config.upsample_initial_channel // 2**index
        out = config.upsample_initial_channel // 2 ** (index + 1)
        layout[f"ups.{index}.weight"] = (into, out, kernel)
        layout[f"ups.{index}.bias"] = (out,)
    block = 0
    for index in range(len(config.upsample_rates)):
        channels = config.upsample_initial_channel // 2 ** (index + 1)
        for kernel in config.resblock_kernel_sizes:
            key = f"resblocks.{block}"
            if config.resblock == "2":
                for conv in range(2):
                    layout.update(
                        _vocoder_conv1d(f"{key}.convs.{conv}", channels, channels, kernel)
                    )
            else:
                for conv in range(3):
                    layout.update(
                        _vocoder_conv1d(f"{key}.convs1.{conv}", channels, channels, kernel)
                    )
                    layout.update(
                        _vocoder_conv1d(f"{key}.convs2.{conv}", channels, channels, kernel)
                    )
                if config.resblock == "AMP1":
                    for conv in range(3):
                        layout.update(
                            _vocoder_amp_activation(
                                f"{key}.acts1.{conv}", channels, config.activation
                            )
                        )
                        layout.update(
                            _vocoder_amp_activation(
                                f"{key}.acts2.{conv}", channels, config.activation
                            )
                        )
            block += 1
    if config.resblock == "AMP1":
        layout.update(_vocoder_amp_activation("act_post", config.final_channels, config.activation))
    layout["conv_post.weight"] = (config.audio_channels, config.final_channels, 7)
    if config.use_bias_at_final:
        layout["conv_post.bias"] = (config.audio_channels,)
    return layout


@dataclass(frozen=True, slots=True)
class LTXVocoderBWEConfig:
    """Exact geometry of one bandwidth-extended LTX vocoder
    (VocoderWithBWE @ b78cec87): a base vocoder, a residual generator
    fed by a causal mel spectrogram of the low-rate waveform, and a
    windowed-sinc skip resampler."""

    vocoder: LTXVocoderConfig
    bwe_generator: LTXVocoderConfig
    input_sampling_rate: int
    output_sampling_rate: int
    hop_length: int
    n_fft: int
    num_mels: int

    def __post_init__(self) -> None:
        if not self.vocoder.stereo or not self.bwe_generator.stereo:
            raise ValueError(
                "bandwidth extension recomputes a stereo mel; both stages must be stereo"
            )
        if self.bwe_generator.apply_final_activation:
            raise ValueError("the reference builds the BWE generator without a final activation")
        counts = (
            self.input_sampling_rate,
            self.output_sampling_rate,
            self.hop_length,
            self.n_fft,
            self.num_mels,
        )
        if any(count <= 0 for count in counts):
            raise ValueError("LTX BWE vocoder counts must be positive")
        if self.num_mels * 2 != self.bwe_generator.in_channels:
            raise ValueError(
                "the stereo mel handoff needs num_mels to be half the BWE generator input width"
            )
        if (
            self.output_sampling_rate <= self.input_sampling_rate
            or self.output_sampling_rate % self.input_sampling_rate
        ):
            raise ValueError("the BWE skip path upsamples by an integer ratio")
        if self.n_fft < self.hop_length:
            raise ValueError("the BWE mel window must cover at least one hop")

    @property
    def resample_ratio(self) -> int:
        return self.output_sampling_rate // self.input_sampling_rate


def ltx_vocoder_bwe_layout(config: LTXVocoderBWEConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the bandwidth-extended vocoder.
    The skip resampler's filter is a non-persistent buffer the reference
    computes at construction; it never appears in checkpoints."""
    layout = {"vocoder." + key: shape for key, shape in ltx_vocoder_layout(config.vocoder).items()}
    layout.update(
        ("bwe_generator." + key, shape)
        for key, shape in ltx_vocoder_layout(config.bwe_generator).items()
    )
    n_freqs = config.n_fft // 2 + 1
    layout["mel_stft.mel_basis"] = (config.num_mels, n_freqs)
    layout["mel_stft.stft_fn.forward_basis"] = (n_freqs * 2, 1, config.n_fft)
    layout["mel_stft.stft_fn.inverse_basis"] = (n_freqs * 2, 1, config.n_fft)
    return layout


# Both 19B geometries are transcribed from the config JSON embedded in
# the published checkpoint's safetensors metadata (Lightricks/LTX-2
# ltx-2-19b-dev.safetensors, "audio_vae" and "vocoder"); the reference
# requires that embedded config (comfy/sd.py VAE.__init__ @ b78cec87).
LTXAV_19B_AUDIO_VAE_CONFIG = LTXAudioVAEConfig(
    in_channels=2,
    out_channels=2,
    base_channels=128,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    z_channels=8,
    mel_bins=64,
    sampling_rate=16000,
    mel_hop_length=160,
    n_fft=1024,
)

LTXAV_19B_VOCODER_CONFIG = LTXVocoderConfig(
    upsample_rates=(6, 5, 2, 2, 2),
    upsample_kernel_sizes=(16, 15, 8, 4, 4),
    resblock="1",
    stereo=True,
    resblock_kernel_sizes=(3, 7, 11),
    resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    upsample_initial_channel=1024,
)

LTXAV_BWE_VOCODER_CONFIG = LTXVocoderBWEConfig(
    vocoder=LTXVocoderConfig(
        upsample_rates=(5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(11, 4, 4, 4, 4, 4),
        resblock="AMP1",
        stereo=True,
        activation="snakebeta",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_initial_channel=1536,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    ),
    bwe_generator=LTXVocoderConfig(
        upsample_rates=(6, 5, 2, 2, 2),
        upsample_kernel_sizes=(12, 11, 4, 4, 4),
        resblock="AMP1",
        stereo=True,
        activation="snakebeta",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_initial_channel=512,
        use_bias_at_final=False,
        use_tanh_at_final=False,
        apply_final_activation=False,
    ),
    input_sampling_rate=16000,
    output_sampling_rate=48000,
    hop_length=80,
    n_fft=512,
    num_mels=64,
)
