"""Pin generated LTXV keyframes as image or pre-encoded guides."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
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
    append_guide_attention_entry,
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


def _as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (int, float)):
        return {int(round(value))}
    if isinstance(value, (set, frozenset)):
        return {int(round(item)) for item in cast("set[Any] | frozenset[Any]", value)}
    if isinstance(value, (list, tuple)):
        result: set[int] = set()
        for item in cast("Sequence[Any]", value):
            result |= _as_int_set(item)
        return result
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _as_int_set(value.tolist())
    return {int(round(float(value)))}


class LTXVAddGeneratedKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning, latent = TypeExpr.concrete(CONDITIONING_TYPE), TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="LTXVAddGeneratedKeyframes",
            display_name="LTXV Add Generated Keyframes",
            category="model/conditioning/ltxv",
            search_terms=("detailing", "dfr", "generated keyframes"),
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("latent", latent),
                InputSpec(
                    "interval_frames",
                    TypeExpr.concrete(CORE_INT),
                    required=False,
                    default=24,
                    widget=NumberWidget(min=1, max=1024),
                ),
                InputSpec("keyframes", latent, required=False),
                InputSpec(
                    "frame_indices",
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
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        latent: object,
        interval_frames: int = 24,
        keyframes: object = None,
        frame_indices: str = "",
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        patchifier = importlib.import_module(
            "dinkster_inference.ldm.lightricks.symmetric_patchifier"
        )
        source = materialize_latent(latent)
        samples = source["samples"]
        if not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError(
                "Generated keyframes must be added to a plain video latent. Add them before "
                "merging the video and audio latents with Concat AV Latent."
            )
        existing = conditioning_get_any_value(positive, "generated_keyframes")
        area = samples.shape[3] * samples.shape[4]
        if existing is not None:
            if existing["tokens_per_frame"] != area:
                raise ValueError(
                    "The existing generated keyframes were added at "
                    f"{existing['tokens_per_frame']} "
                    f"tokens per latent frame but this latent has {area}. The latent was rescaled "
                    "after they were added, so more keyframes cannot be appended to them."
                )
            end = existing["first_latent_frame"] + existing["num_keyframes"]
            if end != samples.shape[2]:
                raise ValueError(
                    f"The existing generated keyframes end at latent frame {end} but the "
                    f"latent has {samples.shape[2]}. Something was appended after them, so more "
                    "keyframes would not be contiguous with the existing block. "
                    "Add all the keyframes before any guides."
                )
        scales = vae.downscale_index_formula
        temporal_scale = scales[0]
        keyframe_idxs, guide_frames = get_keyframe_idxs(positive, samples.shape)
        target_frames = samples.shape[2] - guide_frames
        pixel_frames = (target_frames - 1) * temporal_scale + 1
        scan_frames = min(max(int(target_frames), 0), samples.shape[2])
        occupied: set[int] = set()
        mask: Any = source.get("noise_mask")
        use_mask = mask is not None and getattr(mask, "ndim", 0) >= 3
        for index in range(min(cast(Any, mask).shape[2], scan_frames) if use_mask else scan_frames):
            is_occupied = (
                torch.any(cast(Any, mask)[:, :, index : index + 1] < 1.0 - 1e-4)
                if use_mask
                else torch.any(samples[:, :, index : index + 1] != 0)
            )
            if is_occupied:
                pixel = (
                    0
                    if index <= 0
                    else pixel_frames - 1
                    if index >= scan_frames - 1
                    else index * temporal_scale
                )
                occupied.add(min(max(pixel, 0), pixel_frames - 1))
        if keyframe_idxs is not None:
            occupied |= _as_int_set(
                keyframe_idxs[:, 0, :, 0] if keyframe_idxs.ndim >= 4 else keyframe_idxs[:, 0]
            )
        previous_indices = list(existing["frame_indices"]) if existing is not None else []
        occupied |= set(previous_indices)
        if frame_indices and str(frame_indices).strip():
            if pixel_frames - 1 < 1:
                raise ValueError(
                    f"A {pixel_frames}-frame target has no pixel frames to place keyframes on."
                )
            indices = parse_frame_index_list(
                frame_indices,
                "frame_indices",
                None,
                1,
                pixel_frames - 1,
                "the list sets the count",
                "to place them from interval_frames",
            )
        else:
            interval = float(interval_frames)
            if interval <= 0:
                raise ValueError(f"interval_frames must be > 0, got {interval}")
            if pixel_frames <= 1:
                raise ValueError(
                    f"A {pixel_frames}-frame target has no pixel frames to place keyframes on."
                )
            count = max(1, round((pixel_frames - 1) / interval))
            positions = (
                torch.linspace(0, pixel_frames - 1, count + 1).round().to(torch.int64).tolist()[1:]
            )
            positions = [index for index in positions if index != 0]
            if not positions:
                raise ValueError(
                    f"A {pixel_frames}-frame target has no pixel frames to place keyframes on."
                )
            indices = [index for index in positions if index not in occupied]
            if not indices:
                raise ValueError(
                    "Every candidate detailing-keyframe pixel already has an image keyframe "
                    "or a guide. Leave at least one unoccupied frame, or pass frame_indices."
                )
        clashes = sorted(set(indices) & occupied)
        if clashes:
            raise ValueError(
                f"frame_indices reuses pixel frame(s) {', '.join(str(i) for i in clashes)}, "
                "which already hold an image keyframe, a guide, or a generated keyframe. "
                "Each keyframe needs its own pixel frame."
            )
        count = len(indices)
        if keyframes is not None:
            supplied = materialize_latent(keyframes)
            frames = supplied["samples"]
            if frames.ndim != 5:
                raise ValueError(
                    f"The keyframes latent must be 5 dimensional, got {list(frames.shape)}."
                )
            if frames.shape[2] == 1 and frames.shape[0] != samples.shape[0]:
                if frames.shape[0] % samples.shape[0] != 0:
                    raise ValueError(
                        f"The keyframes latent batch ({frames.shape[0]}) is not a "
                        f"multiple of the video latent batch ({samples.shape[0]}), so it "
                        "cannot be reshaped into per-video keyframes."
                    )
                frames = frames.reshape(
                    samples.shape[0],
                    frames.shape[0] // samples.shape[0],
                    frames.shape[1],
                    *frames.shape[3:],
                ).movedim(1, 2)
            if supplied.get("generated_keyframe_indices") is None and frames.shape[2] > count:
                scale = int(temporal_scale)
                if not torch.is_tensor(frames) or frames.ndim != 5:
                    raise ValueError(
                        "Initializing keyframes from a video needs a plain 5D video latent. "
                        "Split audio with Separate AV Latent first, and peel generated "
                        "keyframes before copying from the video."
                    )
                if scale < 1:
                    raise ValueError(f"temporal_scale must be >= 1, got {scale}")
                if frames.shape[2] < 1:
                    raise ValueError("The video latent has no frames to copy from.")
                selected = [
                    min(max(round(index / scale), 0), frames.shape[2] - 1) for index in indices
                ]
                frames = torch.cat([frames[:, :, index : index + 1] for index in selected], dim=2)
            else:
                frames = frames.to(samples)
                expected = (samples.shape[0], samples.shape[1], count, *samples.shape[3:])
                if frames.shape[:2] != expected[:2] or frames.shape[3:] != expected[3:]:
                    raise ValueError(
                        "The keyframes latent must hold whole latent frames "
                        "at this latent's shape, "
                        f"expected {list(expected)} but got {list(frames.shape)}. "
                        "Resize it to this stage's resolution first."
                    )
                have = frames.shape[2]
                if have > count:
                    raise ValueError(
                        f"The keyframes latent holds {have} frame(s) but only {count} free slot(s) "
                        "are available on this canvas. Pass fewer keyframes, or set frame_indices."
                    )
                if have < count:
                    padding = torch.zeros(
                        (samples.shape[0], samples.shape[1], count - have, *samples.shape[3:]),
                        dtype=frames.dtype,
                        device=frames.device,
                    )
                    frames = torch.cat((frames, padding), dim=2)
            frames = frames.to(samples)
        else:
            frames = torch.zeros(
                (samples.shape[0], samples.shape[1], count, *samples.shape[3:]),
                dtype=samples.dtype,
                device=samples.device,
            )
        _, coords = patchifier.SymmetricPatchifier(1, start_end=True).patchify(samples[:, :, :1])
        pixel_coords = patchifier.latent_to_pixel_coords(coords, scales, causal_fix=True)
        generated: list[Any] = []
        for index in indices:
            positioned = pixel_coords.clone()
            positioned[:, 0] += index
            generated.append(positioned)
        generated_coords = torch.cat(generated, dim=2)
        if keyframe_idxs is not None:
            generated_coords = torch.cat((keyframe_idxs, generated_coords.to(keyframe_idxs)), dim=2)
        entries: Sequence[Any] = (
            conditioning_get_any_value(positive, "guide_attention_entries") or []
        )
        record = dict(
            first_latent_frame=existing["first_latent_frame"] if existing else samples.shape[2],
            num_keyframes=existing["num_keyframes"] + count if existing else count,
            frame_indices=previous_indices + list(indices),
            num_pixel_frames=pixel_frames,
            guide_entry_index=existing["guide_entry_index"] if existing else len(entries),
            tokens_per_frame=area,
        )
        values = {"keyframe_idxs": generated_coords, "generated_keyframes": record}
        positive, negative = (
            conditioning_set_values(conditioning, values) for conditioning in (positive, negative)
        )
        if existing is not None:
            grown_conditions: list[Any] = []
            entry_index = existing["guide_entry_index"]
            for conditioning in (positive, negative):
                old_entries: Sequence[Mapping[str, Any]] = []
                for item in conditioning:
                    found = item[1].get("guide_attention_entries")
                    if found is not None:
                        old_entries = cast(Sequence[Mapping[str, Any]], found)
                        break
                if entry_index >= len(old_entries):
                    raise ValueError(
                        f"The generated keyframes recorded guide entry {entry_index} "
                        f"but the conditioning only has {len(old_entries)}. "
                        "The conditioning was rebuilt after they were added."
                    )
                grown_entries: list[Mapping[str, Any]] = list(old_entries)
                grown = dict(cast(Mapping[str, Any], grown_entries[entry_index]))
                grown["pre_filter_count"] += count * area
                shape = list(cast(Sequence[int], grown["latent_shape"]))
                shape[0] += count
                grown["latent_shape"] = shape
                grown_entries[entry_index] = grown
                grown_conditions.append(
                    conditioning_set_values(
                        conditioning, {"guide_attention_entries": grown_entries}
                    )
                )
            positive, negative = grown_conditions
        else:
            positive, negative = append_guide_attention_entry(
                positive,
                negative,
                count * area,
                [count, samples.shape[3], samples.shape[4]],
            )
        noise_mask = get_noise_mask(source, torch)
        added_mask = torch.ones(
            (noise_mask.shape[0], 1, count, *noise_mask.shape[3:]),
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={
                **source,
                "samples": torch.cat((samples, frames), dim=2),
                "noise_mask": torch.cat((noise_mask, added_mask), dim=2),
            },
        )


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
