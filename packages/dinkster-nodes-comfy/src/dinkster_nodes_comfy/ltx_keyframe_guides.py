"""Pin generated LTXV keyframes as image or pre-encoded guides."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .conditioning import conditioning_set_values
from .ltx_guides import (
    add_image_guide,
    attach_guide_latent,
    conditioning_get_any_value,
    get_keyframe_idxs,
)
from .ltx_video_conditioning import get_noise_mask


def parse_frame_index_list(
    value: str,
    field: str,
    expected_count: int | None,
    first: int,
    last: int,
    expected_desc: str,
    empty_hint: str,
) -> list[int]:
    parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    if not parts:
        raise ValueError(
            f"{field} is empty. Provide at least one index, or leave {field} empty {empty_hint}."
        )
    try:
        indices = [int(part) for part in parts]
    except ValueError:
        bad = ", ".join(repr(part) for part in parts if not re.fullmatch(r"-?\d+", part))
        raise ValueError(
            f"{field} must be a comma-separated list of integers, but could not parse {bad}."
        ) from None
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(
            f"{field} lists {len(indices)} index/indices but {expected_desc}. Provide exactly "
            f"{expected_count}, or leave {field} empty {empty_hint}."
        )
    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(
            f"{field} must not place two keyframes on the same pixel frame, but "
            f"{', '.join(str(index) for index in duplicates)} appear(s) more than once."
        )
    outside = [index for index in indices if not first <= index <= last]
    if outside:
        raise ValueError(
            f"{field} must lie between {first} and {last}, but got "
            f"{', '.join(str(index) for index in outside)}."
        )
    return indices


def scale_frame_indices(indices: list[int], old_num_frames: int, new_num_frames: int) -> list[int]:
    if old_num_frames <= 1:
        raise ValueError(f"Cannot scale keyframe indices from a {old_num_frames}-frame canvas.")
    if new_num_frames <= 1:
        raise ValueError(f"Cannot scale keyframe indices onto a {new_num_frames}-frame canvas.")
    scale = (new_num_frames - 1) / (old_num_frames - 1)
    remapped = [int(round(index * scale)) for index in indices]
    duplicates = sorted({index for index in remapped if remapped.count(index) > 1})
    if duplicates:
        raise ValueError(
            "Scaling keyframe indices onto the new canvas collapsed "
            f"{', '.join(str(index) for index in duplicates)} onto the same pixel frame."
        )
    return remapped


class LTXVGeneratedKeyframesToGuides(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        latent = TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="LTXVGeneratedKeyframesToGuides",
            display_name="LTXV Generated Keyframes to Guides",
            category="model/conditioning/ltxv",
            search_terms=("detailing", "dfr", "keyframe guides"),
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("latent", latent),
                InputSpec("keyframes", latent),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "override_frame_indices",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default="",
                    widget=StringWidget(multiline=False),
                ),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("latent", latent),
            ),
        )

    @classmethod
    def decode_single_frames(cls, vae: Any, keyframe_samples: Any) -> Any:
        if keyframe_samples.shape[2] != 1:
            batch, channels, count, height, width = keyframe_samples.shape
            keyframe_samples = keyframe_samples.movedim(2, 1).reshape(
                batch * count,
                channels,
                1,
                height,
                width,
            )
        images = vae.decode(keyframe_samples)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        latent: object,
        keyframes: object,
        strength: float,
        override_frame_indices: str = "",
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        source, keyframe_source = materialize_latent(latent), materialize_latent(keyframes)
        indices = keyframe_source.get("generated_keyframe_indices")
        if indices is None:
            raise ValueError(
                "This latent does not carry generated keyframe positions. "
                "Connect the keyframes output of LTXV Separate Generated Keyframes."
            )
        if conditioning_get_any_value(positive, "generated_keyframes") is not None:
            raise ValueError(
                "This conditioning still carries generated keyframes. Connect the positive and "
                "negative outputs of LTXV Separate Generated Keyframes."
            )
        samples = source["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframe guides must be added to a plain video latent. Add them before "
                "merging the video and audio latents with Concat AV Latent."
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Only a batch size of 1 is supported, got {samples.shape[0]}. "
                "Each guide is encoded from one image, so it cannot differ across batch elements."
            )
        keyframe_samples = keyframe_source["samples"]
        if keyframe_samples.shape[2] != 1:
            batch, channels, count, height, width = keyframe_samples.shape
            keyframe_samples = keyframe_samples.movedim(2, 1).reshape(
                batch * count,
                channels,
                1,
                height,
                width,
            )
        resize = keyframe_samples.shape[3:] != samples.shape[3:]
        guides = cls.decode_single_frames(vae, keyframe_samples) if resize else keyframe_samples
        if guides.shape[0] != len(indices):
            raise ValueError(
                f"Got {guides.shape[0]} keyframes for {len(indices)} recorded positions."
            )
        _, guide_frames = get_keyframe_idxs(positive, samples.shape)
        num_pixel_frames = (samples.shape[2] - guide_frames - 1) * vae.downscale_index_formula[
            0
        ] + 1
        if override_frame_indices and str(override_frame_indices).strip():
            indices = parse_frame_index_list(
                override_frame_indices,
                "override_frame_indices",
                len(indices),
                1,
                num_pixel_frames - 1,
                f"the keyframes latent carries {len(indices)}",
                "to reuse the recorded positions",
            )
        else:
            old_len = keyframe_source.get("generated_keyframe_num_frames")
            if old_len is not None and int(old_len) != num_pixel_frames:
                indices = scale_frame_indices(list(indices), int(old_len), num_pixel_frames)
        if indices and max(indices) >= num_pixel_frames:
            raise ValueError(
                f"Keyframe position {max(indices)} is outside this latent's "
                f"{num_pixel_frames} frames. "
                "The target was resized temporally after the keyframes were generated."
            )
        for index, frame_idx in enumerate(indices):
            if resize:
                positive, negative, source = add_image_guide(
                    positive,
                    negative,
                    vae,
                    source,
                    guides[index].unsqueeze(0),
                    int(frame_idx),
                    strength,
                )
            else:
                positive, negative, extended, mask = attach_guide_latent(
                    positive,
                    negative,
                    source["samples"],
                    get_noise_mask(source, torch),
                    guides[index : index + 1],
                    int(frame_idx),
                    strength,
                    vae.downscale_index_formula,
                )
                source = {**source, "samples": extended, "noise_mask": mask}
        return cls.outputs(positive=positive, negative=negative, latent=source)


class LTXVSeparateGeneratedKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning, latent = TypeExpr.concrete(CONDITIONING_TYPE), TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="LTXVSeparateGeneratedKeyframes",
            display_name="LTXV Separate Generated Keyframes",
            category="model/conditioning/ltxv",
            search_terms=("detailing", "dfr", "peel keyframes"),
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec("latent", latent),
                InputSpec(
                    "keyframes_to_batch",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=False,
                ),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("latent", latent),
                OutputSpec("keyframes", latent),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        latent: object,
        keyframes_to_batch: bool = False,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        record = conditioning_get_any_value(positive, "generated_keyframes")
        if record is None:
            raise ValueError(
                "This latent has no generated keyframes. "
                "Add them with LTXV Add Generated Keyframes first."
            )
        source = materialize_latent(latent)
        samples = source["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframes must be separated from a plain video latent. "
                "Split the video and audio latents with Separate AV Latent first."
            )
        tokens_per_frame = samples.shape[3] * samples.shape[4]
        if record["tokens_per_frame"] != tokens_per_frame:
            raise ValueError(
                f"The generated keyframes were added at {record['tokens_per_frame']} tokens per "
                f"latent frame but this latent has {tokens_per_frame}. "
                "The latent was rescaled after they were added, "
                "so the keyframes no longer line up. "
                "Separate them before upscaling the latent."
            )
        first, count = record["first_latent_frame"], record["num_keyframes"]
        end = first + count
        if end > samples.shape[2]:
            raise ValueError(
                f"The generated keyframes span latent frames [{first}, {end}) but the latent "
                f"only has {samples.shape[2]}. It was recorded against a different latent."
            )
        keyframe_samples = samples[:, :, first:end].clone()
        if keyframes_to_batch:
            batch, channels, _, height, width = keyframe_samples.shape
            keyframe_samples = keyframe_samples.movedim(2, 1).reshape(
                batch * count,
                channels,
                1,
                height,
                width,
            )
        video_samples = torch.cat((samples[:, :, :first], samples[:, :, end:]), dim=2)
        mask = get_noise_mask(source, torch)
        video_mask = torch.cat((mask[:, :, :first], mask[:, :, end:]), dim=2)
        _, guide_frames = get_keyframe_idxs(positive, samples.shape)
        first_token = (first - (samples.shape[2] - guide_frames)) * tokens_per_frame
        entry_index = record["guide_entry_index"]
        conditions: list[Any] = []
        for conditioning in (positive, negative):
            indices = conditioning_get_any_value(conditioning, "keyframe_idxs")
            if indices is not None:
                indices = torch.cat(
                    (
                        indices[:, :, :first_token],
                        indices[:, :, first_token + count * tokens_per_frame :],
                    ),
                    dim=2,
                )
                if indices.shape[2] == 0:
                    indices = None
            entries = conditioning_get_any_value(conditioning, "guide_attention_entries")
            if entries:
                if entry_index >= len(entries):
                    raise ValueError(
                        f"The generated keyframes recorded guide entry {entry_index} "
                        f"but the conditioning only has {len(entries)}. "
                        "The conditioning was rebuilt after they were added."
                    )
                entries = entries[:entry_index] + entries[entry_index + 1 :]
            conditions.append(
                conditioning_set_values(
                    conditioning,
                    {
                        "keyframe_idxs": indices,
                        "guide_attention_entries": entries or None,
                        "generated_keyframes": None,
                    },
                )
            )
        pixel_frames = record.get("num_pixel_frames")
        if pixel_frames is None:
            pixel_frames = (first - 1) * 8 + 1
        return cls.outputs(
            positive=conditions[0],
            negative=conditions[1],
            latent={**source, "samples": video_samples, "noise_mask": video_mask},
            keyframes={
                "samples": keyframe_samples,
                "generated_keyframe_indices": list(record["frame_indices"]),
                "generated_keyframe_num_frames": int(pixel_frames),
            },
        )
