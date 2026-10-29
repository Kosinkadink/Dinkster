"""Torch decoders for the preview provider specs in dinkster_inference.preview.

Each builder takes a latent descriptor and returns a decoder callable:
sampler state tensor in, small CPU :class:`PreviewFrame` out. Decoders run
on the state tensor's device and downscale there, so only the final
uint8 frame crosses to the CPU.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from dinkster_inference import (
    LATENT2RGB_ANIMATION_PROVIDER,
    LATENT2RGB_PROVIDER,
    LATENT2RGB_WEBP_PROVIDER,
    LATENT2WAVEFORM_PROVIDER,
    MEBIBYTE,
    EncodedPreviewAnimation,
    LatentDescriptor,
    PreviewClip,
    PreviewFrame,
    preview_fps,
)
from dinkster_inference.taehv import TAEHVConfig
from dinkster_inference.taesd import TAESDConfig

from .operations import bound_compute_device
from .taehv import TAEHVDecoder
from .taesd import TAESDDecoder
from .triposplat_decoder import OctreeGaussianDecoder, SplatTensors

PREVIEW_MAX_EDGE = 512
PREVIEW_FRAME_BUDGET = 3
PREVIEW_ENCODED_FRAME_CAP = 24
WAVEFORM_WIDTH = 512
WAVEFORM_HEIGHT = 128

# The reference TripoSplatSamplingPreview node defaults plus the fixed
# arguments its preview module hardcodes (comfy_extras/nodes_triposplat.py
# and comfy/ldm/triposplat/preview.py @ 36408117).
PREVIEW_SPLAT_SIZE = 320
PREVIEW_SPLAT_GAUSSIANS = 16384
PREVIEW_SPLAT_LEVEL = 5
PREVIEW_SPLAT_YAW = 90.0
PREVIEW_SPLAT_PITCH = 15.0
PREVIEW_SPLAT_MAX_RADIUS = 3
PREVIEW_SPLAT_MIN_OPACITY = 0.01
_SPLAT_SH_C0 = 0.28209479177387814
_SPLAT_FOV_DEGREES = 35.0
_SPLAT_CAMERA_DISTANCE = 2.2

PreviewDecoder = Callable[[torch.Tensor], PreviewFrame | PreviewClip | EncodedPreviewAnimation]
PreviewDecoderBuilder = Callable[[LatentDescriptor], PreviewDecoder]


def _frame_slice(state: torch.Tensor, channels: int) -> torch.Tensor:
    """The single [C, H, W] frame a still preview shows: batch 0, and for
    video latents ([B, C, T, H, W]) the first temporal frame."""
    if state.ndim == 5 and state.shape[1] == channels:
        return state[0, :, 0]
    if state.ndim == 4 and state.shape[1] == channels:
        return state[0]
    raise ValueError(
        f"preview state must be [B,{channels},H,W] or [B,{channels},T,H,W],"
        f" got shape {tuple(state.shape)}"
    )


def _downscale(chw: torch.Tensor, max_edge: int) -> torch.Tensor:
    height, width = int(chw.shape[-2]), int(chw.shape[-1])
    longest = max(height, width)
    if longest <= max_edge:
        return chw
    scale = max_edge / longest
    return torch.nn.functional.interpolate(
        chw.unsqueeze(0),
        size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="area",
    ).squeeze(0)


def latent2rgb_decoder(descriptor: LatentDescriptor) -> PreviewDecoder:
    """Constant-cost projection preview: the descriptor's per-channel RGB
    factors applied to the raw sampler state, matching the reference's
    Latent2RGBPreviewer value convention (result in [-1, 1], mapped to
    [0, 255]; latent_preview.py @ b78cec87)."""
    if descriptor.rgb_factors is None:
        raise ValueError("latent2rgb preview requires descriptor rgb_factors")
    weight = torch.tensor(descriptor.rgb_factors, dtype=torch.float32).transpose(0, 1)
    bias = (
        torch.tensor(descriptor.rgb_bias, dtype=torch.float32)
        if descriptor.rgb_bias is not None
        else None
    )
    channels = descriptor.channels

    def decode(state: torch.Tensor) -> PreviewFrame:
        frame = _frame_slice(state, channels).to(dtype=torch.float32)
        local_weight = weight.to(frame.device)
        local_bias = bias.to(frame.device) if bias is not None else None
        rgb = torch.nn.functional.linear(frame.permute(1, 2, 0), local_weight, local_bias)
        return _uint8_frame(_downscale(rgb.permute(2, 0, 1), PREVIEW_MAX_EDGE))

    return decode


def latent2waveform_decoder(descriptor: LatentDescriptor) -> PreviewDecoder:
    """Constant-cost waveform preview for 1D audio latents: the per-timestep
    RMS envelope across latent channels, peak-normalized and rendered as a
    symmetric waveform around the midline. No codec runs - the image shows
    the latent's temporal energy taking shape as denoising settles."""
    channels = descriptor.channels

    def decode(state: torch.Tensor) -> PreviewFrame:
        # H3 audio latents carry a stereo axis between channels and time
        # ([B,C,S,T]); any such interior axes fold into the RMS alongside
        # the channels, leaving the trailing time axis as the envelope.
        if state.ndim < 3 or state.shape[1] != channels:
            raise ValueError(
                f"waveform preview state must be [B,{channels},...,T], "
                f"got shape {tuple(state.shape)}"
            )
        sample = state[0].to(dtype=torch.float32)
        envelope = sample.pow(2).mean(dim=tuple(range(sample.ndim - 1))).sqrt()
        resampled = torch.nn.functional.interpolate(
            envelope.view(1, 1, -1), size=WAVEFORM_WIDTH, mode="linear", align_corners=False
        ).view(-1)
        peak = resampled.max()
        if peak > 0:
            resampled = resampled / peak
        half_heights = resampled * ((WAVEFORM_HEIGHT - 2) / 2.0)
        rows = torch.arange(WAVEFORM_HEIGHT, dtype=torch.float32, device=half_heights.device)
        center = (WAVEFORM_HEIGHT - 1) / 2.0
        band = (rows.view(-1, 1) - center).abs() <= (half_heights.view(1, -1) + 0.5)
        pixels = (
            (band.to(torch.uint8) * 230).unsqueeze(-1).expand(-1, -1, 3).contiguous().cpu().numpy()
        )
        return PreviewFrame(rgb=pixels, width=WAVEFORM_WIDTH, height=WAVEFORM_HEIGHT)

    return decode


class FramePacer:
    """A rotating contiguous window over a temporal frame ring.

    Each call yields the next window of latent timesteps to decode,
    budgeted to sustain the display rate since the previous call (always
    at least one frame, never more than ``cap``). Without a rate, every
    call spends the full cap."""

    def __init__(
        self,
        fps: float | None,
        *,
        cap: int = PREVIEW_FRAME_BUDGET,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cap < 1:
            raise ValueError("frame budget cap must be at least 1")
        self._fps = fps
        self._cap = cap
        self._clock = clock
        self._cursor = 0
        self._last: float | None = None

    def window(self, frame_count: int) -> tuple[int, int]:
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1")
        now = self._clock()
        if self._last is None or self._fps is None:
            budget = self._cap
        else:
            budget = min(self._cap, max(1, math.ceil((now - self._last) * self._fps)))
        self._last = now
        start = self._cursor if self._cursor < frame_count else 0
        stop = min(start + budget, frame_count)
        self._cursor = stop % frame_count
        return start, stop


def _clip_window(state: torch.Tensor, channels: int, pacer: FramePacer) -> tuple[int, int, int]:
    """The pacer's next (start, stop, frame_count) for a video latent."""
    if state.ndim != 5 or state.shape[1] != channels:
        raise ValueError(
            f"animated preview state must be [B,{channels},T,H,W], got shape {tuple(state.shape)}"
        )
    frame_count = int(state.shape[2])
    start, stop = pacer.window(frame_count)
    return start, stop, frame_count


def latent2rgb_animation_decoder(descriptor: LatentDescriptor) -> PreviewDecoder:
    """Rotating-window latent2rgb preview for video latents: the same
    per-channel projection as :func:`latent2rgb_decoder`, applied to the
    pacer's window of latent timesteps, one frame per timestep."""
    if descriptor.rgb_factors is None:
        raise ValueError("latent2rgb preview requires descriptor rgb_factors")
    weight = torch.tensor(descriptor.rgb_factors, dtype=torch.float32).transpose(0, 1)
    bias = (
        torch.tensor(descriptor.rgb_bias, dtype=torch.float32)
        if descriptor.rgb_bias is not None
        else None
    )
    channels = descriptor.channels
    fps = preview_fps(descriptor)
    pacer = FramePacer(fps)

    def decode(state: torch.Tensor) -> PreviewClip:
        start, stop, frame_count = _clip_window(state, channels, pacer)
        window = state[0, :, start:stop].to(dtype=torch.float32)
        local_weight = weight.to(window.device)
        local_bias = bias.to(window.device) if bias is not None else None
        rgb = torch.nn.functional.linear(window.permute(1, 2, 3, 0), local_weight, local_bias)
        frames = tuple(
            _uint8_frame(_downscale(frame.permute(2, 0, 1), PREVIEW_MAX_EDGE)) for frame in rgb
        )
        return PreviewClip(
            frames=frames,
            frame_indices=tuple(range(start, stop)),
            frame_count=frame_count,
            fps=fps,
        )

    return decode


def latent2rgb_encoded_frames_decoder(descriptor: LatentDescriptor) -> PreviewDecoder:
    """Whole-clip latent2rgb frames for the encoded-animation transport.

    Unlike the ring decoders, every call re-decodes the full temporal span
    (the emitter re-encodes one self-contained animation per emit, so a
    rotating window would produce partial animations). Clips longer than
    ``PREVIEW_ENCODED_FRAME_CAP`` latent timesteps are stride-sampled to
    the cap with the display rate scaled to keep wall-clock duration; the
    returned clip is self-consistent (frame_count == len(frames))."""
    if descriptor.rgb_factors is None:
        raise ValueError("latent2rgb preview requires descriptor rgb_factors")
    weight = torch.tensor(descriptor.rgb_factors, dtype=torch.float32).transpose(0, 1)
    bias = (
        torch.tensor(descriptor.rgb_bias, dtype=torch.float32)
        if descriptor.rgb_bias is not None
        else None
    )
    channels = descriptor.channels
    fps = preview_fps(descriptor)

    def decode(state: torch.Tensor) -> PreviewClip:
        if state.ndim != 5 or state.shape[1] != channels:
            raise ValueError(
                f"encoded animation preview state must be [B,{channels},T,H,W],"
                f" got shape {tuple(state.shape)}"
            )
        frame_count = int(state.shape[2])
        stride = max(1, math.ceil(frame_count / PREVIEW_ENCODED_FRAME_CAP))
        window = state[0, :, ::stride].to(dtype=torch.float32)
        local_weight = weight.to(window.device)
        local_bias = bias.to(window.device) if bias is not None else None
        rgb = torch.nn.functional.linear(window.permute(1, 2, 3, 0), local_weight, local_bias)
        frames = tuple(
            _uint8_frame(_downscale(frame.permute(2, 0, 1), PREVIEW_MAX_EDGE)) for frame in rgb
        )
        return PreviewClip(
            frames=frames,
            frame_indices=tuple(range(len(frames))),
            frame_count=len(frames),
            fps=fps / stride if fps is not None else None,
        )

    return decode


def taehv_preview_decoder(
    decoder: TAEHVDecoder, config: TAEHVConfig, descriptor: LatentDescriptor
) -> PreviewDecoder:
    """Rotating-window TAEHV decode preview for video latents.

    The pacer's window rides through the decoder as one causal clip
    (matching the reference's value convention: output already in [0, 1],
    clamped and truncated to uint8 with no rescale; latent_preview.py @
    783545f6). Each latent timestep contributes its final content frame,
    so clip slots stay one-per-latent-timestep like the cheap path."""
    channels = config.latent_channels
    fps = preview_fps(descriptor)
    pacer = FramePacer(fps)
    upscale = config.temporal_upscale

    def decode(state: torch.Tensor) -> PreviewClip:
        start, stop, frame_count = _clip_window(state, channels, pacer)
        parameter = next(decoder.parameters())
        device = _routed_device(decoder) or parameter.device
        window = state[:1, :, start:stop].to(device=device, dtype=parameter.dtype)
        with torch.inference_mode():
            content = decoder.decode(window)[0, ::upscale]
        frames = tuple(
            _uint8_unit_frame(_downscale(frame.to(dtype=torch.float32), PREVIEW_MAX_EDGE))
            for frame in content
        )
        return PreviewClip(
            frames=frames,
            frame_indices=tuple(range(start, stop)),
            frame_count=frame_count,
            fps=fps,
        )

    return decode


def _routed_device(module: torch.nn.Module) -> torch.device | None:
    """The residency mechanism's load device when the module is enrolled.

    Routed layers lease their weights onto that device at forward time
    even while the stored parameters sit offloaded on the CPU, so an
    enrolled module's compute device is the mechanism's, never the
    parameters'."""
    for submodule in module.modules():
        device = bound_compute_device(submodule)
        if device is not None:
            return device
    return None


def taesd_preview_decoder(decoder: TAESDDecoder, config: TAESDConfig) -> PreviewDecoder:
    """Full TAESD decode preview matching the reference's TAESDPreviewerImpl
    value convention (decode the first frame in latent space, result in
    [-1, 1]; latent_preview.py @ b78cec87). The latent moves to the
    decoder's compute device at call time (the residency load device for
    an enrolled decoder, the parameter device otherwise), so the caller
    controls placement through residency staging."""
    scale = config.vae_scale
    assert scale is not None
    shift = config.vae_shift
    channels = config.latent_channels

    def decode(state: torch.Tensor) -> PreviewFrame:
        frame = _frame_slice(state, channels)
        parameter = next(decoder.parameters())
        device = _routed_device(decoder) or parameter.device
        latent = frame.unsqueeze(0).to(device=device, dtype=parameter.dtype)
        with torch.inference_mode():
            rgb = decoder.decode(latent.sub(shift).mul(scale))[0]
        return _uint8_frame(_downscale(rgb.to(dtype=torch.float32), PREVIEW_MAX_EDGE))

    return decode


def triposplat_preview_decoder(decoder: OctreeGaussianDecoder) -> PreviewDecoder:
    """Coarse gaussian-splat preview matching the reference's
    TripoSplatSamplingPreview behavior (decode_x0_to_image @ 36408117):
    the x0 estimate decodes at a reduced octree level into few gaussians,
    and the splat software-renders to one small RGB frame on the CPU.
    The latent moves to the decoder's compute device at call time, so the
    caller controls placement through residency staging."""
    channels = decoder.config.latent_channels
    level = min(PREVIEW_SPLAT_LEVEL, decoder.config.max_voxel_level)

    def decode(state: torch.Tensor) -> PreviewFrame:
        if state.ndim != 3 or state.shape[-1] != channels:
            raise ValueError(
                f"preview state must be [B,tokens,{channels}], got shape {tuple(state.shape)}"
            )
        parameter = next(decoder.parameters())
        device = _routed_device(decoder) or parameter.device
        latent = state[:1].to(device=device, dtype=parameter.dtype)
        # A fixed seed keeps the octree point sampling stable between
        # steps, so consecutive previews show denoising progress rather
        # than resampling noise.
        generator = torch.Generator().manual_seed(0)
        with torch.inference_mode():
            splat = decoder.decode(
                latent,
                num_gaussians=PREVIEW_SPLAT_GAUSSIANS,
                generator=generator,
                level=level,
            )[0]
        return _splat_frame(splat)

    return decode


def _splat_view_matrix(yaw_degrees: float, pitch_degrees: float) -> Any:
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    around_y = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        np.float32,
    )
    around_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        np.float32,
    )
    return around_x @ around_y


def _splat_frame(splat: SplatTensors) -> PreviewFrame:
    """One splat rasterized with the reference's software renderer
    (render_splat @ 36408117): perspective projection, filled disks
    bucketed by integer radius, nearest-wins z-buffer. SplatTensors are
    already viewer-Y-up, so the reference's object-to-viewer transform
    is skipped here."""
    size = PREVIEW_SPLAT_SIZE
    positions = splat.positions.float().cpu().numpy()
    colors = splat.sh.float().cpu().numpy()[:, 0, :] * _SPLAT_SH_C0 + 0.5
    scale = splat.scales.float().cpu().numpy().max(axis=1)
    opacity = splat.opacities.float().cpu().numpy()[:, 0]

    view = positions @ _splat_view_matrix(PREVIEW_SPLAT_YAW, PREVIEW_SPLAT_PITCH).T
    depth = view[:, 2] + _SPLAT_CAMERA_DISTANCE
    keep = (depth > 1e-2) & (opacity > PREVIEW_SPLAT_MIN_OPACITY)
    view, depth, scale = view[keep], depth[keep], scale[keep]
    color = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)[keep]
    if view.shape[0] == 0:
        blank = np.zeros((size, size, 3), np.uint8)
        return PreviewFrame(rgb=blank, width=size, height=size)
    focal = (size / 2) / math.tan(math.radians(_SPLAT_FOV_DEGREES) / 2)
    center_x = size / 2 + focal * view[:, 0] / depth
    center_y = size / 2 + focal * view[:, 1] / depth
    radius = np.clip(np.round(focal * scale / depth), 1, PREVIEW_SPLAT_MAX_RADIUS).astype(np.int32)

    # Expand each splat to its disk pixels, bucketed by integer radius so
    # the expansion stays vectorized.
    columns: list[Any] = []
    rows: list[Any] = []
    depths: list[Any] = []
    shades: list[Any] = []
    for r in range(int(radius.min()), int(radius.max()) + 1):
        mask = radius == r
        if not mask.any():
            continue
        offset_y, offset_x = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (offset_x * offset_x + offset_y * offset_y) <= r * r
        ox, oy = offset_x[disk], offset_y[disk]
        columns.append((center_x[mask, None] + ox).ravel())
        rows.append((center_y[mask, None] + oy).ravel())
        depths.append(np.repeat(depth[mask], ox.size))
        shades.append(np.repeat(color[mask], ox.size, axis=0))
    column = np.clip(np.concatenate(columns), 0, size - 1).astype(np.int64)
    row = np.clip(np.concatenate(rows), 0, size - 1).astype(np.int64)
    pixel_depth = np.concatenate(depths)
    pixel_color = np.concatenate(shades)

    # Nearest-wins z-buffer: pack (quantized depth, source index), take the
    # per-pixel minimum, then decode the winning index back to its color.
    pixel = row * size + column
    quantized = np.clip((pixel_depth * 1024.0).astype(np.int64), 0, MEBIBYTE - 1)
    key = (quantized << 32) | np.arange(pixel.size, dtype=np.int64)
    buffer = np.full(size * size, 1 << 62, np.int64)
    np.minimum.at(buffer, pixel, key)
    image = np.zeros((size * size, 3), np.uint8)
    hit = buffer < (1 << 62)
    image[hit] = pixel_color[buffer[hit] & 0xFFFFFFFF]
    return PreviewFrame(rgb=image.reshape(size, size, 3), width=size, height=size)


def _uint8_frame(rgb: torch.Tensor) -> PreviewFrame:
    """One CHW [-1, 1] float frame as a CPU HWC uint8 PreviewFrame.

    Truncating (not rounding) to uint8 matches the reference's
    preview_to_image exactly (latent_preview.py @ b78cec87)."""
    pixels = (
        rgb.add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return PreviewFrame(rgb=pixels, width=width, height=height)


def _uint8_unit_frame(rgb: torch.Tensor) -> PreviewFrame:
    """One CHW [0, 1] float frame as a CPU HWC uint8 PreviewFrame.

    The light video TAEs already produce [0, 1] content; clamping and
    truncating without rescale matches the reference's
    preview_to_image(do_scale=False) exactly (latent_preview.py @ 783545f6)."""
    pixels = (
        rgb.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    )
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return PreviewFrame(rgb=pixels, width=width, height=height)


def preview_decoder_builders() -> Mapping[str, PreviewDecoderBuilder]:
    """Torch decoder builders keyed by preview provider spec id.

    Only descriptor-sufficient decoders appear here; model-cost decoders
    that need loaded weights (TAESD, TAEHV) are constructed by the
    sampling arm, which owns asset resolution and residency."""
    return {
        LATENT2RGB_PROVIDER.id: latent2rgb_decoder,
        LATENT2RGB_ANIMATION_PROVIDER.id: latent2rgb_animation_decoder,
        LATENT2RGB_WEBP_PROVIDER.id: latent2rgb_encoded_frames_decoder,
        LATENT2WAVEFORM_PROVIDER.id: latent2waveform_decoder,
    }


__all__ = [
    "PREVIEW_ENCODED_FRAME_CAP",
    "PREVIEW_FRAME_BUDGET",
    "PREVIEW_MAX_EDGE",
    "PREVIEW_SPLAT_GAUSSIANS",
    "PREVIEW_SPLAT_LEVEL",
    "PREVIEW_SPLAT_MAX_RADIUS",
    "PREVIEW_SPLAT_MIN_OPACITY",
    "PREVIEW_SPLAT_PITCH",
    "PREVIEW_SPLAT_SIZE",
    "PREVIEW_SPLAT_YAW",
    "WAVEFORM_HEIGHT",
    "WAVEFORM_WIDTH",
    "FramePacer",
    "PreviewDecoder",
    "PreviewDecoderBuilder",
    "latent2rgb_animation_decoder",
    "latent2rgb_decoder",
    "latent2rgb_encoded_frames_decoder",
    "latent2waveform_decoder",
    "preview_decoder_builders",
    "taehv_preview_decoder",
    "taesd_preview_decoder",
    "triposplat_preview_decoder",
]
