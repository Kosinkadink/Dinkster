"""Native LTX-2 audio VAE and vocoder.

Faithful standalone port of ``comfy/ldm/lightricks/vae/audio_vae.py``,
``causal_audio_autoencoder.py``, and ``vocoders/vocoder.py`` at ComfyUI
commit b78cec87. State-dict names match the reference.

The reference front-end runs on torchaudio (``functional.resample`` and
``transforms.MelSpectrogram``); Dinkster carries no torchaudio dependency,
so :func:`ltx_audio_resample` and :func:`ltx_audio_waveform_to_mel`
reimplement those kernels in plain torch, transcribing torchaudio's
windowed-sinc resampler (lowpass width 6, rolloff 0.99) and
slaney-scale, slaney-normalized mel filterbank so the numerics match
the reference bit for bit.

:class:`LTXAudioVAE` covers the wrapper's mel/latent math: ``encode``
takes a waveform to normalized latents (always the posterior mode, as
the reference wrapper does) and ``decode`` takes latents back to a mel
spectrogram. Waveform synthesis is the vocoder's job: feed
``decode``'s output through :func:`ltx_vocoder_features` into
:class:`LTXVocoder` or :class:`LTXVocoderWithBWE`.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    LTXAudioVAEConfig,
    LTXVocoderBWEConfig,
    LTXVocoderConfig,
)

from .operations import INITLESS, Operations, ResidencyRouted

__all__ = [
    "LTXAudioDecoder",
    "LTXAudioEncoder",
    "LTXAudioVAE",
    "LTXVocoder",
    "LTXVocoderWithBWE",
    "ltx_audio_resample",
    "ltx_audio_waveform_to_mel",
    "ltx_vocoder_features",
]

_LOG_MEL_FLOOR = 1e-5


def _hz_to_mel_slaney(freq: float) -> float:
    """torchaudio's ``_hz_to_mel`` with ``mel_scale="slaney"``."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    if freq >= min_log_hz:
        return min_log_hz / f_sp + math.log(freq / min_log_hz) / (math.log(6.4) / 27.0)
    return freq / f_sp


def _mel_to_hz_slaney(mels: torch.Tensor) -> torch.Tensor:
    """torchaudio's ``_mel_to_hz`` with ``mel_scale="slaney"``."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    freqs = f_sp * mels
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp((math.log(6.4) / 27.0) * (mels[log_t] - min_log_mel))
    return freqs


def _mel_filterbank(n_freqs: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """torchaudio's ``melscale_fbanks`` pinned to the reference's
    arguments: ``f_min`` 0, ``f_max`` half the sample rate, slaney scale,
    slaney area normalization. Returns ``(n_freqs, n_mels)`` float32."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel_slaney(0.0)
    m_max = _hz_to_mel_slaney(sample_rate / 2.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz_slaney(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(torch.zeros(1), torch.min(down_slopes, up_slopes))

    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    return fb * enorm.unsqueeze(0)


def ltx_audio_resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """torchaudio's ``functional.resample`` with its default kernel (sinc
    interpolation, hann window, lowpass width 6, rolloff 0.99), which is
    what the reference encode path calls."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("resampling rates must be positive")
    gcd = math.gcd(orig_freq, new_freq)
    orig_freq //= gcd
    new_freq //= gcd
    if orig_freq == new_freq:
        return waveform

    lowpass_width = 6
    rolloff = 0.99
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=waveform.dtype, device=waveform.device)
    t = (
        torch.arange(0, -new_freq, -1, dtype=waveform.dtype, device=waveform.device)[:, None, None]
        / new_freq
        + idx[None, None] / orig_freq
    )
    t *= base_freq
    t = t.clamp_(-lowpass_width, lowpass_width)
    window = torch.cos(t * math.pi / lowpass_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base_freq / orig_freq)

    shape = waveform.size()
    length = shape[-1]
    flat = waveform.view(-1, length)
    flat = F.pad(flat, (width, width + orig_freq))
    resampled = F.conv1d(flat[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(flat.shape[0], -1)
    resampled = resampled[..., : math.ceil(new_freq * length / orig_freq)]
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def ltx_audio_waveform_to_mel(
    waveform: torch.Tensor, sample_rate: int, config: LTXAudioVAEConfig
) -> torch.Tensor:
    """The reference wrapper's ``waveform_to_mel``: resample to the VAE's
    rate, magnitude STFT (hann window, centered with reflect padding),
    slaney mel projection, log with a 1e-5 floor, then swap to the
    encoder's (batch, channels, time, mel) order."""
    if sample_rate != config.sampling_rate:
        waveform = ltx_audio_resample(waveform, sample_rate, config.sampling_rate)

    window = torch.hann_window(config.n_fft, device=waveform.device)
    shape = waveform.size()
    spec = torch.stft(
        waveform.reshape(-1, shape[-1]),
        n_fft=config.n_fft,
        hop_length=config.mel_hop_length,
        win_length=config.n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = spec.reshape(shape[:-1] + spec.shape[-2:]).abs()

    fb = _mel_filterbank(config.n_fft // 2 + 1, config.mel_bins, config.sampling_rate)
    fb = fb.to(device=magnitude.device, dtype=magnitude.dtype)
    mel = torch.matmul(magnitude.transpose(-1, -2), fb).transpose(-1, -2)
    mel = torch.log(torch.clamp(mel, min=_LOG_MEL_FLOOR))
    return mel.permute(0, 1, 3, 2).contiguous()


def _pixel_norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-6)


class _CausalConv2d(torch.nn.Module):
    """Square conv padded fully to the past on the time (height) axis and
    symmetrically on the frequency axis."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, *, operations: Operations
    ) -> None:
        super().__init__()
        pad = kernel - 1
        self._padding = (pad // 2, pad - pad // 2, pad, 0)
        self.conv = operations.conv2d(in_channels, out_channels, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self._padding))


class _ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv1 = _CausalConv2d(in_channels, out_channels, 3, operations=operations)
        self.conv2 = _CausalConv2d(out_channels, out_channels, 3, operations=operations)
        self.nin_shortcut = (
            _CausalConv2d(in_channels, out_channels, 1, operations=operations)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(_pixel_norm(x)))
        h = self.conv2(F.silu(_pixel_norm(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class _Downsample(torch.nn.Module):
    """Stride-2 conv halving time and frequency; time is padded fully to
    the past, frequency picks up one trailing column."""

    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = operations.conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 2, 0)))


class _Upsample(torch.nn.Module):
    """Nearest 2x upsample and causal conv; the first upsampled time row
    only repeats the past, so it is dropped."""

    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = _CausalConv2d(channels, channels, 3, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)[:, :, 1:, :]


class _EncoderLevel(torch.nn.Module):
    def __init__(self, blocks: list[_ResnetBlock], downsample: _Downsample | None) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList(blocks)
        self.downsample = downsample


class _DecoderLevel(torch.nn.Module):
    def __init__(self, blocks: list[_ResnetBlock], upsample: _Upsample | None) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList(blocks)
        self.upsample = upsample


class _MidBlocks(torch.nn.Module):
    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.block_1 = _ResnetBlock(channels, channels, operations=operations)
        self.block_2 = _ResnetBlock(channels, channels, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_2(self.block_1(x))


class LTXAudioEncoder(torch.nn.Module):
    """Mel spectrogram (batch, channels, time, mel) to doubled latents
    (batch, 2 * z, time / 4, mel / 4)."""

    def __init__(self, config: LTXAudioVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        ch = config.base_channels
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.conv_in = _CausalConv2d(config.in_channels, ch, 3, operations=operations)
        levels: list[_EncoderLevel] = []
        for level, mult in enumerate(config.ch_mult):
            into = ch * in_ch_mult[level]
            out = ch * mult
            blocks: list[_ResnetBlock] = []
            for _ in range(config.num_res_blocks):
                blocks.append(_ResnetBlock(into, out, operations=operations))
                into = out
            last = level == len(config.ch_mult) - 1
            downsample = None if last else _Downsample(out, operations=operations)
            levels.append(_EncoderLevel(blocks, downsample))
        self.down = torch.nn.ModuleList(levels)
        deepest = ch * config.ch_mult[-1]
        self.mid = _MidBlocks(deepest, operations=operations)
        self.conv_out = _CausalConv2d(deepest, 2 * config.z_channels, 3, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for module in self.down:
            level = cast(_EncoderLevel, module)
            for block in level.block:
                h = block(h)
            if level.downsample is not None:
                h = level.downsample(h)
        h = self.mid(h)
        return self.conv_out(F.silu(_pixel_norm(h)))


def _adjust_output_shape(
    decoded: torch.Tensor, target_shape: tuple[int, int, int, int]
) -> torch.Tensor:
    """The reference decoder's crop-then-pad to the requested
    (channels, time, frequency)."""
    _, channels, time, freq = target_shape
    decoded = decoded[:, :channels, : min(decoded.shape[2], time), : min(decoded.shape[3], freq)]
    time_pad = time - decoded.shape[2]
    freq_pad = freq - decoded.shape[3]
    if time_pad > 0 or freq_pad > 0:
        decoded = F.pad(decoded, (0, max(freq_pad, 0), 0, max(time_pad, 0)))
    return decoded[:, :channels, :time, :freq]


class LTXAudioDecoder(torch.nn.Module):
    """Latents (batch, z, time, mel / 4) back to a mel spectrogram of the
    requested target shape."""

    def __init__(self, config: LTXAudioVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        ch = config.base_channels
        into = ch * config.ch_mult[-1]
        self.conv_in = _CausalConv2d(config.z_channels, into, 3, operations=operations)
        self.mid = _MidBlocks(into, operations=operations)
        levels: list[_DecoderLevel | None] = [None] * len(config.ch_mult)
        for level in reversed(range(len(config.ch_mult))):
            out = ch * config.ch_mult[level]
            blocks: list[_ResnetBlock] = []
            for _ in range(config.num_res_blocks + 1):
                blocks.append(_ResnetBlock(into, out, operations=operations))
                into = out
            upsample = _Upsample(into, operations=operations) if level != 0 else None
            levels[level] = _DecoderLevel(blocks, upsample)
        self.up = torch.nn.ModuleList(cast(list[_DecoderLevel], levels))
        self.conv_out = _CausalConv2d(into, config.out_channels, 3, operations=operations)

    def forward(self, z: torch.Tensor, target_shape: tuple[int, int, int, int]) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        for index in reversed(range(len(self.up))):
            level = cast(_DecoderLevel, self.up[index])
            for block in level.block:
                h = block(h)
            if level.upsample is not None:
                h = level.upsample(h)
        h = self.conv_out(F.silu(_pixel_norm(h)))
        return _adjust_output_shape(h, target_shape)


class _PerChannelStatistics(ResidencyRouted, torch.nn.Module):
    """Checkpoint-loaded latent statistics; the dashes in the buffer
    names are the reference's, so access goes through ``get_buffer``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(channels))
        self.register_buffer("mean-of-means", torch.empty(channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").to(x)
        std = self.get_buffer("std-of-means").to(x)
        return (x - mean) / std

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").to(x)
        std = self.get_buffer("std-of-means").to(x)
        return x * std + mean


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    """(batch, channels, time, freq) -> (batch, time, channels * freq)."""
    batch, channels, time, freq = latents.shape
    return latents.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)


def _unpatchify(latents: torch.Tensor, channels: int, freq: int) -> torch.Tensor:
    batch, time, _ = latents.shape
    return latents.reshape(batch, time, channels, freq).permute(0, 2, 1, 3)


def _decode_target_shape(
    latents_shape: torch.Size, config: LTXAudioVAEConfig
) -> tuple[int, int, int, int]:
    """The reference wrapper's causal decode target: 4x the latent steps
    minus the three rows the causal upsamples never produce."""
    scale = LTX_AUDIO_LATENT_DOWNSAMPLE_FACTOR
    time = latents_shape[2] * scale - (scale - 1)
    return (latents_shape[0], config.out_channels, time, config.mel_bins)


class LTXAudioVAE(torch.nn.Module):
    """The reference ``AudioVAE`` wrapper minus the vocoder handoff:
    waveforms to normalized latents and latents back to mel
    spectrograms."""

    def __init__(self, config: LTXAudioVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        self.encoder = LTXAudioEncoder(config, operations=operations)
        self.decoder = LTXAudioDecoder(config, operations=operations)
        self.per_channel_statistics = _PerChannelStatistics(config.statistics_channels)

    def encode(self, waveform: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
        if waveform.shape[1] != self.config.in_channels:
            if waveform.shape[1] != 1:
                raise ValueError(
                    f"expected {self.config.in_channels} audio channels or mono,"
                    f" got {waveform.shape[1]}"
                )
            waveform = waveform.expand(-1, self.config.in_channels, *waveform.shape[2:])
        mel = ltx_audio_waveform_to_mel(waveform, sample_rate, self.config)
        latents = self.encoder(mel)
        # The wrapper always takes the posterior mode: the first half of
        # the doubled encoder output.
        means = torch.chunk(latents, 2, dim=1)[0]
        normalized = self.per_channel_statistics.normalize(_patchify(means))
        return _unpatchify(normalized, self.config.z_channels, self.config.latent_frequency_bins)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        denormalized = self.per_channel_statistics.un_normalize(_patchify(latents))
        denormalized = _unpatchify(
            denormalized, self.config.z_channels, self.config.latent_frequency_bins
        )
        return self.decoder(denormalized, _decode_target_shape(latents.shape, self.config))


def ltx_vocoder_features(mel_spec: torch.Tensor, audio_channels: int) -> torch.Tensor:
    """The reference wrapper's vocoder handoff: swap decoded mel output
    to (batch, channels, mel, time) and drop the channel axis for
    mono."""
    features = mel_spec.transpose(2, 3)
    if audio_channels == 1:
        return features.squeeze(1)
    if audio_channels != 2:
        raise ValueError(f"the vocoder handoff supports mono or stereo, got {audio_channels}")
    return features


def _get_padding(kernel: int, dilation: int) -> int:
    return (kernel * dilation - dilation) // 2


def _kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """BigVGAN's kaiser-windowed sinc lowpass, shaped (1, 1, taps)."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    scaled = 2 * cutoff * time
    sinc = torch.where(
        scaled == 0, torch.tensor(1.0), torch.sin(math.pi * scaled) / math.pi / scaled
    )
    filt = 2 * cutoff * window * sinc
    filt /= filt.sum()
    return filt.view(1, 1, kernel_size)


class _LowPassFilter1d(ResidencyRouted, torch.nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int) -> None:
        super().__init__()
        self.stride = stride
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.register_buffer("filter", _kaiser_sinc_filter(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        kernel = self.get_buffer("filter").expand(channels, -1, -1).to(x)
        return F.conv1d(x, kernel, stride=self.stride, groups=channels)


class _UpSample1d(ResidencyRouted, torch.nn.Module):
    """Anti-aliased integer upsampling: replicate-padded grouped
    transposed convolution with a windowed-sinc filter. The kaiser
    window is the BigVGAN AMP default; the hann window replicates
    torchaudio's resample kernel and drives the bandwidth-extension skip
    path (its filter is never stored in checkpoints)."""

    def __init__(
        self,
        ratio: int,
        *,
        kernel_size: int | None = None,
        window_type: Literal["kaiser", "hann"] = "kaiser",
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        if window_type == "hann":
            rolloff = 0.99
            width = math.ceil(6 / rolloff)
            kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = kernel_size - ratio
            t = (torch.arange(kernel_size) / ratio - width) * rolloff
            window = torch.cos(t.clamp(-6, 6) * math.pi / 6 / 2) ** 2
            filt = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        else:
            if kernel_size is None:
                kernel_size = int(6 * ratio // 2) * 2
            self.pad = kernel_size // ratio - 1
            self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
            self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
            filt = _kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size)
        self.register_buffer("filter", filt, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.get_buffer("filter").expand(channels, -1, -1).to(x)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, kernel, stride=self.ratio, groups=channels)
        return x[..., self.pad_left : -self.pad_right]


class _DownSample1d(torch.nn.Module):
    def __init__(self, ratio: int, *, kernel_size: int | None = None) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = int(6 * ratio // 2) * 2
        self.lowpass = _LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class _Activation1d(torch.nn.Module):
    """BigVGAN's alias-free activation: upsample 2x, activate, downsample
    2x, both with 12-tap kaiser filters."""

    def __init__(self, act: torch.nn.Module) -> None:
        super().__init__()
        self.act = act
        self.upsample = _UpSample1d(2, kernel_size=12)
        self.downsample = _DownSample1d(2, kernel_size=12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class _Snake(ResidencyRouted, torch.nn.Module):
    """BigVGAN Snake with log-scale magnitude (the reference default)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.view(1, -1, 1).to(device=x.device, dtype=x.dtype))
        return x + (1.0 / (alpha + 1e-9)) * torch.sin(x * alpha).pow(2)


class _SnakeBeta(ResidencyRouted, torch.nn.Module):
    """BigVGAN SnakeBeta with log-scale magnitudes (the reference
    default): alpha sets the frequency, beta the magnitude."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(channels))
        self.beta = torch.nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.view(1, -1, 1).to(device=x.device, dtype=x.dtype))
        beta = torch.exp(self.beta.view(1, -1, 1).to(device=x.device, dtype=x.dtype))
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


def _snake_activation(channels: int, activation: str) -> torch.nn.Module:
    return _SnakeBeta(channels) if activation == "snakebeta" else _Snake(channels)


class _AMPBlock1(torch.nn.Module):
    """BigVGAN AMP residual block: three dilated convs and three plain
    convs, each preceded by an alias-free Snake activation."""

    def __init__(
        self,
        channels: int,
        kernel: int,
        dilation: tuple[int, ...],
        activation: str,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            operations.conv1d(
                channels, channels, kernel, dilation=step, padding=_get_padding(kernel, step)
            )
            for step in dilation
        )
        self.convs2 = torch.nn.ModuleList(
            operations.conv1d(channels, channels, kernel, padding=_get_padding(kernel, 1))
            for _ in dilation
        )
        self.acts1 = torch.nn.ModuleList(
            _Activation1d(_snake_activation(channels, activation)) for _ in dilation
        )
        self.acts2 = torch.nn.ModuleList(
            _Activation1d(_snake_activation(channels, activation)) for _ in dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2, strict=True
        ):
            xt = conv1(act1(x))
            xt = conv2(act2(xt))
            x = x + xt
        return x


_LRELU_SLOPE = 0.1


class _ResBlock1(torch.nn.Module):
    """HiFi-GAN residual block kind "1": three dilated convs interleaved
    with three undilated ones."""

    def __init__(
        self,
        channels: int,
        kernel: int,
        dilation: tuple[int, ...],
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            operations.conv1d(
                channels, channels, kernel, dilation=step, padding=_get_padding(kernel, step)
            )
            for step in dilation
        )
        self.convs2 = torch.nn.ModuleList(
            operations.conv1d(channels, channels, kernel, padding=_get_padding(kernel, 1))
            for _ in dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = conv1(F.leaky_relu(x, _LRELU_SLOPE))
            xt = conv2(F.leaky_relu(xt, _LRELU_SLOPE))
            x = xt + x
        return x


class _ResBlock2(torch.nn.Module):
    """HiFi-GAN residual block kind "2": two dilated convs."""

    def __init__(
        self,
        channels: int,
        kernel: int,
        dilation: tuple[int, ...],
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList(
            operations.conv1d(
                channels, channels, kernel, dilation=step, padding=_get_padding(kernel, step)
            )
            for step in dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = conv(F.leaky_relu(x, _LRELU_SLOPE))
            x = xt + x
        return x


class LTXVocoder(torch.nn.Module):
    """HiFi-GAN / BigVGAN vocoder: mel features (stereo channels
    concatenated on the mel axis) to a waveform."""

    def __init__(self, config: LTXVocoderConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        initial = config.upsample_initial_channel
        self.conv_pre = operations.conv1d(config.in_channels, initial, 7, padding=3)
        self.ups = torch.nn.ModuleList(
            operations.conv_transpose1d(
                initial // 2**index,
                initial // 2 ** (index + 1),
                kernel,
                stride=rate,
                padding=(kernel - rate) // 2,
            )
            for index, (rate, kernel) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)
            )
        )
        blocks: list[torch.nn.Module] = []
        for index in range(len(config.upsample_rates)):
            channels = initial // 2 ** (index + 1)
            for kernel, dilation in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes, strict=True
            ):
                if config.resblock == "1":
                    blocks.append(_ResBlock1(channels, kernel, dilation, operations=operations))
                elif config.resblock == "2":
                    blocks.append(_ResBlock2(channels, kernel, dilation, operations=operations))
                else:
                    blocks.append(
                        _AMPBlock1(
                            channels, kernel, dilation, config.activation, operations=operations
                        )
                    )
        self.resblocks = torch.nn.ModuleList(blocks)
        self.act_post = (
            _Activation1d(_snake_activation(config.final_channels, config.activation))
            if config.resblock == "AMP1"
            else None
        )
        self.conv_post = operations.conv1d(
            config.final_channels,
            config.audio_channels,
            7,
            padding=3,
            bias=config.use_bias_at_final,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[1] != 2:
                raise ValueError(
                    f"stereo mel input must carry two audio channels, got {x.shape[1]}"
                )
            x = torch.cat((x[:, 0], x[:, 1]), dim=1)
        x = self.conv_pre(x)
        kernels = len(self.config.resblock_kernel_sizes)
        for index, up in enumerate(self.ups):
            if self.config.resblock != "AMP1":
                x = F.leaky_relu(x, _LRELU_SLOPE)
            x = up(x)
            first = index * kernels
            total = self.resblocks[first](x)
            for offset in range(1, kernels):
                total += self.resblocks[first + offset](x)
            x = total / kernels
        if self.act_post is not None:
            x = self.act_post(x)
        else:
            # The reference's post activation is a stock LeakyReLU, not
            # the 0.1-slope one used between stages.
            x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        if self.config.apply_final_activation:
            x = torch.tanh(x) if self.config.use_tanh_at_final else torch.clamp(x, -1.0, 1.0)
        return x


class _STFTFn(ResidencyRouted, torch.nn.Module):
    """Checkpoint-loaded DFT-basis convolution STFT with causal left
    padding. The inverse basis is unused at inference but kept for
    checkpoint state-dict parity."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.empty(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.empty(n_freqs * 2, 1, filter_length))

    def magnitude(self, y: torch.Tensor) -> torch.Tensor:
        y = F.pad(y.unsqueeze(1), (max(0, self.win_length - self.hop_length), 0))
        basis = self.get_buffer("forward_basis").to(y)
        spec = F.conv1d(y, basis, stride=self.hop_length)
        n_freqs = spec.shape[1] // 2
        real = spec[:, :n_freqs]
        imag = spec[:, n_freqs:]
        return torch.sqrt(real**2 + imag**2)


class _MelSTFT(ResidencyRouted, torch.nn.Module):
    """Checkpoint-loaded mel projection over :class:`_STFTFn`
    magnitudes."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int, n_mels: int) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        self.register_buffer("mel_basis", torch.empty(n_mels, filter_length // 2 + 1))

    def log_mel(self, y: torch.Tensor) -> torch.Tensor:
        magnitude = self.stft_fn.magnitude(y)
        mel = torch.matmul(self.get_buffer("mel_basis").to(magnitude), magnitude)
        return torch.log(torch.clamp(mel, min=_LOG_MEL_FLOOR))


class LTXVocoderWithBWE(torch.nn.Module):
    """Bandwidth-extended vocoder: the base vocoder's low-rate waveform
    plus a residual predicted from its own causal mel spectrogram,
    upsampled through a windowed-sinc skip path."""

    def __init__(self, config: LTXVocoderBWEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        self.vocoder = LTXVocoder(config.vocoder, operations=operations)
        self.bwe_generator = LTXVocoder(config.bwe_generator, operations=operations)
        self.mel_stft = _MelSTFT(config.n_fft, config.hop_length, config.n_fft, config.num_mels)
        self.resampler = _UpSample1d(config.resample_ratio, window_type="hann", persistent=False)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = self.vocoder(mel_spec)
        out_length = (
            x.shape[-1] * self.config.output_sampling_rate // self.config.input_sampling_rate
        )
        remainder = x.shape[-1] % self.config.hop_length
        if remainder:
            x = F.pad(x, (0, self.config.hop_length - remainder))
        batch, channels, _ = x.shape
        mel = self.mel_stft.log_mel(x.reshape(batch * channels, -1))
        mel = mel.reshape(batch, channels, mel.shape[1], mel.shape[2])
        residual = self.bwe_generator(mel)
        skip = self.resampler(x)
        return torch.clamp(residual + skip, -1.0, 1.0)[..., :out_length]
