"""Native LTXV latent guide attachment and attention coordinates."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)

from .conditioning import conditioning_set_values
from .ltx_video_conditioning import get_noise_mask


def conditioning_get_any_value(
    conditioning: Sequence[Sequence[Any]], key: str, default: Any = None
) -> Any:
    return next((entry[1][key] for entry in conditioning if key in entry[1]), default)


def get_keyframe_idxs(
    conditioning: Sequence[Sequence[Any]], latent_shape: Any = None
) -> tuple[Any, int]:
    indices = conditioning_get_any_value(conditioning, "keyframe_idxs")
    if indices is None:
        return None, 0
    if latent_shape is not None and len(latent_shape) == 5:
        return indices, indices.shape[2] // (latent_shape[-2] * latent_shape[-1])
    entries = conditioning_get_any_value(conditioning, "guide_attention_entries")
    if entries:
        return indices, sum(entry["latent_shape"][0] for entry in entries)
    torch = importlib.import_module("torch")
    return indices, torch.unique(indices[:, 0, :, 0]).shape[0]


def append_guide_attention_entry(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
    pre_filter_count: int,
    latent_shape: list[int],
    strength: float = 1.0,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None
        else None,
        "latent_shape": latent_shape,
    }
    results: list[Any] = []
    for conditioning in (positive, negative):
        existing: Sequence[Any] = []
        for item in conditioning:
            if item[1].get("guide_attention_entries") is not None:
                existing = item[1]["guide_attention_entries"]
                break
        results.append(
            conditioning_set_values(conditioning, {"guide_attention_entries": [*existing, entry]})
        )
    return results[0], results[1]


def attach_guide_latent(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
    latent_image: Any,
    noise_mask: Any,
    guide_latent: Any,
    frame_idx: int,
    strength: float,
    scale_factors: Any,
    latent_downscale_factor: int = 1,
    causal_fix: bool | None = None,
    attention_mask: Any = None,
) -> tuple[Any, Any, Any, Any]:
    initialize_inference_process()
    torch = importlib.import_module("torch")
    patchifier_module = importlib.import_module(
        "dinkster_inference.ldm.lightricks.symmetric_patchifier"
    )
    guide_shape = list(guide_latent.shape[2:])
    guide_mask = None
    if latent_downscale_factor > 1:
        scale = int(latent_downscale_factor)
        dilated = torch.zeros(
            (*guide_latent.shape[:3], guide_latent.shape[3] * scale, guide_latent.shape[4] * scale),
            device=guide_latent.device,
            dtype=guide_latent.dtype,
        )
        dilated[..., ::scale, ::scale] = guide_latent
        guide_latent = dilated
        guide_mask = torch.full(
            (dilated.shape[0], 1, *dilated.shape[2:]),
            -1.0,
            device=dilated.device,
            dtype=dilated.dtype,
        )
        guide_mask[..., ::scale, ::scale] = 1.0
    if latent_image.shape[1] != 128 or guide_latent.shape[1] != 128:
        raise ValueError("Adding guide to a combined AV latent is not supported.")
    if causal_fix is None:
        causal_fix = frame_idx == 0 or guide_latent.shape[2] == 1
    _, coords = patchifier_module.SymmetricPatchifier(1, start_end=True).patchify(guide_latent)
    pixel_coords = patchifier_module.latent_to_pixel_coords(
        coords, scale_factors, causal_fix=causal_fix
    )
    pixel_coords[:, 0] += frame_idx
    offset = (latent_downscale_factor - 1) * torch.tensor(
        scale_factors[1:],
        device=pixel_coords.device,
    ).view(1, -1, 1, 1)
    pixel_coords[:, 1:, :, 1:] += offset.to(pixel_coords.dtype)
    results: list[Any] = []
    for conditioning in (positive, negative):
        existing, _ = get_keyframe_idxs(conditioning)
        indices = (
            pixel_coords.clone() if existing is None else torch.cat((existing, pixel_coords), dim=2)
        )
        results.append(conditioning_set_values(conditioning, {"keyframe_idxs": indices}))
    positive, negative = results
    if guide_mask is not None:
        target_height = max(noise_mask.shape[3], guide_mask.shape[3])
        target_width = max(noise_mask.shape[4], guide_mask.shape[4])
        if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
            noise_mask = noise_mask.expand(-1, -1, -1, target_height, target_width)
        if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
            guide_mask = guide_mask.expand(-1, -1, -1, target_height, target_width)
        mask = guide_mask - strength
    else:
        mask = torch.full(
            (
                noise_mask.shape[0],
                1,
                guide_latent.shape[2],
                noise_mask.shape[3],
                noise_mask.shape[4],
            ),
            max(0.0, 1.0 - strength),
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )
    latent_image = torch.cat((latent_image, guide_latent), dim=2)
    noise_mask = torch.cat((noise_mask, mask), dim=2)
    positive, negative = append_guide_attention_entry(
        positive,
        negative,
        guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4],
        guide_shape,
        strength,
        attention_mask,
    )
    return positive, negative, latent_image, noise_mask


class LTXVAddLatentGuide(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        latent = TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="LTXVAddLatentGuide",
            display_name="LTXV Add Latent Guide",
            category="model/conditioning/ltxv",
            search_terms=("latent guide", "add latent guide", "guide latent"),
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("latent", latent),
                InputSpec("guiding_latent", latent),
                InputSpec(
                    "latent_idx",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=-9999, max=9999),
                ),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec("attention_mask", TypeExpr.concrete(MASK_TYPE), required=False),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("latent", latent),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        latent: object,
        guiding_latent: object,
        latent_idx: int,
        strength: float,
        attention_mask: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        source = materialize_latent(latent)
        samples = source["samples"]
        guide = materialize_latent(guiding_latent)["samples"]
        for name, value in (("latent", samples), ("guiding_latent", guide)):
            if value.ndim != 5:
                raise ValueError(
                    f"{name} must be a 5D video latent (batch, channels, frames, height, width), "
                    f"got shape {list(value.shape)}."
                )
        if latent_idx + guide.shape[2] > samples.shape[2]:
            raise ValueError(
                f"Guide of {guide.shape[2]} latent frame(s) at latent_idx {latent_idx} "
                f"runs past the end of the {samples.shape[2]}-frame latent. Negative values "
                "are allowed and place the guide before the start of the latent."
            )
        if samples.shape[3] % guide.shape[3] != 0 or samples.shape[4] % guide.shape[4] != 0:
            raise ValueError(
                f"Guiding latent spatial size {guide.shape[3]}x{guide.shape[4]} must divide "
                f"the target size {samples.shape[3]}x{samples.shape[4]} by a whole number."
            )
        height_scale, width_scale = (
            samples.shape[3] // guide.shape[3],
            samples.shape[4] // guide.shape[4],
        )
        if height_scale != width_scale:
            raise ValueError(
                f"Guiding latent spatial ratio must be square, got height x{height_scale} and "
                f"width x{width_scale} ({guide.shape[3]}x{guide.shape[4]} -> "
                f"{samples.shape[3]}x{samples.shape[4]})."
            )
        temporal_scale = vae.downscale_index_formula[0]
        frame_idx = (
            latent_idx * temporal_scale
            if latent_idx <= 0
            else 1 + (latent_idx - 1) * temporal_scale
        )
        positive, negative, samples, mask = attach_guide_latent(
            positive,
            negative,
            samples,
            get_noise_mask(source, torch),
            guide,
            frame_idx,
            strength,
            vae.downscale_index_formula,
            width_scale,
            attention_mask=materialize_image(attention_mask)
            if attention_mask is not None
            else None,
        )
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={**source, "samples": samples, "noise_mask": mask},
        )
