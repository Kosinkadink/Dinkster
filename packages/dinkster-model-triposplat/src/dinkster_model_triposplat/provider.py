"""TripoSplat node execution over the public inference resource seams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from dinkster_api.v1 import AssetRef
from dinkster_inference import (
    BFLOAT16,
    FLUX2_SHARED_COMPONENT_FAMILY_ID,
    TRIPOSPLAT_CONFIG,
    ComponentBinding,
    InferenceComponentHandle,
    MultiStreamLatent,
    TripoSplatComponentRole,
    bind_component_conditioning,
    load_safetensors_header,
    plan_triposplat_split_component,
    require_inference_codec_handle,
    require_inference_component_handle,
    triposplat_component_runtime_identity,
)
from dinkster_inference_torch import (
    ComponentPublisher,
    TripoSplatConditioning,
    component_publisher,
    kl_codec_plugin,
    load_triposplat_component,
    triposplat_conditioning_to_carrier,
)

_VISION_ROLE: TripoSplatComponentRole = "dinov3-vision-conditioner"
_DECODER_ROLE: TripoSplatComponentRole = "gaussian-decoder"
_MAX_SEED = 0xFFFFFFFFFFFFFFFF
_MIN_GAUSSIANS = 32768
_MAX_GAUSSIANS = 1048576


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _image(value: object, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError(f"{name} must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if (
        tensor.ndim != 4
        or tensor.shape[-1] not in (1, 3, 4)
        or any(size <= 0 for size in tensor.shape)
    ):
        raise ValueError(f"{name} must be nonempty HWC or BHWC with 1, 3, or 4 channels")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError(f"{name} values must be finite and in [0, 1]")
    if tensor.shape[-1] == 1:
        return tensor.expand(*tensor.shape[:-1], 3).contiguous()
    return tensor[..., :3].contiguous()


def _mask(value: object, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError(f"{name} must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must be nonempty HW or BHW")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError(f"{name} values must be finite and in [0, 1]")
    return tensor


def _asset(value: object, name: str) -> AssetRef:
    if type(value) is not AssetRef:
        raise TypeError(f"{name} must be a resolved asset reference")
    return value


def _load_component(
    asset: AssetRef,
    role: TripoSplatComponentRole,
    *,
    publisher: ComponentPublisher | None = None,
) -> object:
    path = asset.local_path()
    source = load_safetensors_header(path, asset_digest=asset.digest, asset_size=asset.size)
    planned = plan_triposplat_split_component(source, role=role, path=path)
    identity = triposplat_component_runtime_identity(planned, BFLOAT16)
    loaded = load_triposplat_component(
        path,
        asset=asset,
        expected_role=role,
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )
    active_publisher = component_publisher() if publisher is None else publisher
    return active_publisher.publish(loaded.module, resource_identity=identity)


def execute_load_triposplat_vision_encoder(
    *,
    vision_encoder: object,
    publisher: ComponentPublisher | None = None,
) -> Mapping[str, object]:
    asset = _asset(vision_encoder, "vision_encoder")
    return {"vision": _load_component(asset, _VISION_ROLE, publisher=publisher)}


def execute_load_triposplat_decoder(
    *,
    decoder: object,
    publisher: ComponentPublisher | None = None,
) -> Mapping[str, object]:
    asset = _asset(decoder, "decoder")
    return {"decoder": _load_component(asset, _DECODER_ROLE, publisher=publisher)}


def _triposplat_component(
    value: object, name: str, role: TripoSplatComponentRole
) -> InferenceComponentHandle:
    handle = require_inference_component_handle(value, name)
    try:
        ComponentBinding(role, TRIPOSPLAT_CONFIG.family_id, handle.resource_identity)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a native TripoSplat {role} component") from error
    return handle


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] > batch:
        return tensor.narrow(0, 0, batch)
    if tensor.shape[0] < batch:
        repeats = -(-batch // tensor.shape[0])
        return tensor.repeat(repeats, *([1] * (tensor.ndim - 1))).narrow(0, 0, batch)
    return tensor


def _lanczos_resize(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """The reference's PIL Lanczos roundtrip (comfy.utils.lanczos @ 36408117)."""
    from PIL import Image

    arrays = samples.movedim(1, -1).cpu().float().numpy()
    resized: list[torch.Tensor] = []
    for array in arrays:
        image = Image.fromarray(np.clip(255.0 * array, 0, 255).astype(np.uint8))
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        resized.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0))
    return torch.stack(resized).to(samples.device, samples.dtype)


def _preprocess_item(
    image: torch.Tensor, mask: torch.Tensor, erode_radius: int, size: int
) -> torch.Tensor:
    """One image's subject crop (TripoSplatPreprocessImage @ 36408117):
    mask the subject over black, erode the matte, crop the square
    bounding box with 1.2x margin, and resize to the target square."""
    rgb = image[..., :3].clamp(0.0, 1.0).movedim(-1, 0)
    alpha = mask.clamp(0.0, 1.0)[None]
    rgba = torch.cat([rgb, alpha], dim=0)[None]
    scale = size / min(rgba.shape[2], rgba.shape[3])
    rgba = _lanczos_resize(rgba, round(rgba.shape[3] * scale), round(rgba.shape[2] * scale)).clamp(
        0.0, 1.0
    )
    if erode_radius > 0:
        eroded = -functional.max_pool2d(
            -rgba[:, 3:4], kernel_size=2 * erode_radius + 1, stride=1, padding=erode_radius
        )
        rgba = torch.cat([rgba[:, :3], eroded], dim=1)
    foreground = torch.nonzero(rgba[0, 3] > 0)
    if foreground.numel() == 0:
        raise ValueError("mask is empty (no foreground pixels)")
    minimum = foreground.min(dim=0).values
    maximum = foreground.max(dim=0).values
    center_y = (minimum[0].item() + maximum[0].item()) / 2
    center_x = (minimum[1].item() + maximum[1].item()) / 2
    half = max(maximum[0].item() - minimum[0].item(), maximum[1].item() - minimum[1].item()) / 2
    half = half * 1.2
    top = int(center_y - half)
    bottom = int(center_y + half)
    left = int(center_x - half)
    right = int(center_x + half)
    crop = rgba.new_zeros((1, 4, bottom - top, right - left))
    source_top = max(top, 0)
    source_bottom = min(bottom, rgba.shape[2])
    source_left = max(left, 0)
    source_right = min(right, rgba.shape[3])
    crop[
        :,
        :,
        source_top - top : source_bottom - top,
        source_left - left : source_right - left,
    ] = rgba[:, :, source_top:source_bottom, source_left:source_right]
    crop = _lanczos_resize(crop, size, size).clamp(0.0, 1.0)
    composited = crop[:, :3] * crop[:, 3:4]
    return composited[0].movedim(0, -1).unsqueeze(0)


def execute_triposplat_preprocess_image(
    *, image: object, mask: object, erode_radius: int, size: int
) -> Mapping[str, object]:
    frames = _image(image, "image")
    matte = _mask(mask, "mask")
    erode = _integer(erode_radius, "erode_radius", minimum=0, maximum=16)
    target = _integer(size, "size", minimum=256, maximum=4096)
    target = max(16, (target // 16) * 16)
    if matte.shape[0] != frames.shape[0]:
        matte = _repeat_to_batch(matte, frames.shape[0])
    if tuple(matte.shape[1:]) != tuple(frames.shape[1:3]):
        matte = functional.interpolate(
            matte[:, None],
            size=(frames.shape[1], frames.shape[2]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    prepared = torch.cat(
        [_preprocess_item(frames[i], matte[i], erode, target) for i in range(frames.shape[0])],
        dim=0,
    )
    return {"image": prepared}


def _encoded_reference(vae: object, pixel: torch.Tensor) -> torch.Tensor:
    if isinstance(vae, InferenceComponentHandle):
        handle = vae
        try:
            ComponentBinding("vae", FLUX2_SHARED_COMPONENT_FAMILY_ID, handle.resource_identity)
        except (TypeError, ValueError) as error:
            raise TypeError("vae must be a native Flux2 VAE component") from error
        with handle.stage():
            with torch.inference_mode():
                module = cast("Any", handle.component)
                plugin = kl_codec_plugin(module)
                plugin = replace(plugin, compute_dtype=next(module.parameters()).dtype)
                if plugin.descriptor.latent.channels != TRIPOSPLAT_CONFIG.cond2_channels:
                    raise ValueError(
                        "vae must encode "
                        f"{TRIPOSPLAT_CONFIG.cond2_channels}-channel reference latents"
                    )
                encoded = plugin.encode(pixel.to(device=cast("Any", handle.load_device)))
    else:
        codec = require_inference_codec_handle(vae, "vae")
        if codec.descriptor.latent.channels != TRIPOSPLAT_CONFIG.cond2_channels:
            raise ValueError(
                f"vae must encode {TRIPOSPLAT_CONFIG.cond2_channels}-channel reference latents"
            )
        with codec.stage():
            with torch.inference_mode():
                encoded = cast("Any", codec).encode_content(
                    pixel.to(device=cast("Any", codec.load_device))
                )
    if type(encoded) is not torch.Tensor:
        raise TypeError("vae encode must return an exact torch.Tensor")
    return encoded.detach().to(device="cpu", dtype=torch.float32)


def execute_triposplat_conditioning(
    *, vision: object, vae: object, image: object
) -> Mapping[str, object]:
    frames = _image(image, "image")
    pixel = frames.permute(0, 3, 1, 2).contiguous()
    vision_handle = _triposplat_component(vision, "vision", _VISION_ROLE)
    with vision_handle.stage():
        with torch.inference_mode():
            module = cast("Any", vision_handle.component)
            config = module.config
            pixel_values = pixel.to(device=cast("Any", vision_handle.load_device)).float()
            mean = pixel_values.new_tensor(config.image_mean).view(1, 3, 1, 1)
            std = pixel_values.new_tensor(config.image_std).view(1, 3, 1, 1)
            normalized = ((pixel_values - mean) / std).to(dtype=next(module.parameters()).dtype)
            sequence = module(normalized)
            features = functional.layer_norm(sequence.float(), sequence.shape[-1:]).cpu()
    reference = _encoded_reference(vae, pixel)
    if reference.ndim != 4 or reference.shape[0] != features.shape[0]:
        raise ValueError("vae reference latent batch must match the image batch")
    binding = ComponentBinding(
        _VISION_ROLE, TRIPOSPLAT_CONFIG.family_id, vision_handle.resource_identity
    )
    positive = bind_component_conditioning(
        triposplat_conditioning_to_carrier(TripoSplatConditioning(features, reference)),
        binding,
    )
    negative = bind_component_conditioning(
        triposplat_conditioning_to_carrier(
            TripoSplatConditioning(torch.zeros_like(features), torch.zeros_like(reference))
        ),
        binding,
    )
    batch = features.shape[0]
    latent = MultiStreamLatent.from_pairs(
        (
            (
                "latent",
                torch.zeros(
                    (batch, TRIPOSPLAT_CONFIG.q_token_length, TRIPOSPLAT_CONFIG.latent_channels),
                    dtype=torch.float32,
                ),
            ),
            (
                "camera",
                torch.zeros((batch, 1, TRIPOSPLAT_CONFIG.cam_channels), dtype=torch.float32),
            ),
        )
    )
    return {"positive": positive, "negative": negative, "latent": {"samples": latent}}


def execute_triposplat_decode(
    *, samples: object, decoder: object, num_gaussians: int, seed: int
) -> Mapping[str, object]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a latent mapping containing 'samples'")
    streams = cast("Mapping[object, object]", samples).get("samples")
    if type(streams) is not MultiStreamLatent:
        raise TypeError("samples must contain a multi-stream TripoSplat latent")
    streams_value = cast("MultiStreamLatent[Any]", streams)
    if streams_value.roles != ("latent", "camera"):
        raise ValueError("samples must contain exactly the ('latent', 'camera') streams")
    latent = streams_value.by_role("latent")
    if type(latent) is not torch.Tensor or latent.ndim != 3:
        raise TypeError("the TripoSplat latent stream must be a rank-3 torch.Tensor")
    handle = _triposplat_component(decoder, "decoder", _DECODER_ROLE)
    count = _integer(num_gaussians, "num_gaussians", minimum=_MIN_GAUSSIANS, maximum=_MAX_GAUSSIANS)
    seed_value = _integer(seed, "seed", minimum=0, maximum=_MAX_SEED)
    with handle.stage():
        with torch.inference_mode():
            module = cast("Any", handle.component)
            per_point = int(module.gaussians_per_point)
            if count % per_point:
                count = round(count / per_point) * per_point
            parameter = next(module.parameters())
            moved = latent.detach().to(device=parameter.device)
            generator = torch.Generator(device="cpu").manual_seed(seed_value)
            parts = module.decode(moved, num_gaussians=count, generator=generator)
            splat = {
                key: torch.stack([getattr(part, key) for part in parts]).cpu()
                for key in ("positions", "scales", "rotations", "opacities", "sh")
            }
    return {"splat": splat}
