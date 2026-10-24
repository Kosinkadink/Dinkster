"""Native ComfyUI resize, batching, and color-transfer nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    ComboOption,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
    report_progress,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_IMAGE_OR_MASK = TypeExpr.union(IMAGE_TYPE, MASK_TYPE)
_RESIZE_INPUT = TypeExpr.variable("input_type", (IMAGE_TYPE, MASK_TYPE))
_INT = TypeExpr.concrete(CORE_INT)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_MAX_RESOLUTION = 16384
_SCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
_CROP_METHODS = ("disabled", "center")


def _number(
    name: str,
    value_type: TypeExpr,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    step: float | int | None = None,
    *,
    doc: str = "",
) -> InputSpec:
    return InputSpec(
        name,
        value_type,
        default=default,
        doc=doc,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo(
    name: str, options: tuple[str, ...], default: str | None = None, *, doc: str = ""
) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        doc=doc,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _combo_values(
    combo_id: str, selected: str | Mapping[str, Any], values: Mapping[str, Any]
) -> Mapping[str, Any]:
    if isinstance(selected, Mapping):
        return selected
    prefix = f"{combo_id}."
    return {
        combo_id: selected,
        **{input_id.removeprefix(prefix): value for input_id, value in values.items()},
    }


def _image_or_mask(value: Any) -> Any:
    return materialize_image(value)


def _as_channels(value: Any) -> tuple[Any, bool]:
    is_image = value.ndim == 4
    return (value.movedim(-1, 1) if is_image else value.unsqueeze(1)), is_image


def _from_channels(value: Any, is_image: bool) -> Any:
    return value.movedim(1, -1) if is_image else value.squeeze(1)


def _common_upscale(value: Any, width: int, height: int, method: str, crop: str) -> Any:
    return importlib.import_module("dinkster_inference.utils").common_upscale(
        value, width, height, method, crop
    )


def _scale_by(value: Any, multiplier: float, method: str) -> Any:
    value, is_image = _as_channels(value)
    result = _common_upscale(
        value,
        round(value.shape[-1] * multiplier),
        round(value.shape[-2] * multiplier),
        method,
        "disabled",
    )
    return _from_channels(result, is_image)


def _scale_dimensions(
    value: Any, width: int, height: int, method: str, crop: str = "disabled"
) -> Any:
    if width == 0 and height == 0:
        return value
    value, is_image = _as_channels(value)
    if width == 0:
        width = max(1, round(value.shape[-1] * height / value.shape[-2]))
    elif height == 0:
        height = max(1, round(value.shape[-2] * width / value.shape[-1]))
    return _from_channels(_common_upscale(value, width, height, method, crop), is_image)


def _scale_edge(value: Any, size: int, method: str, *, longer: bool) -> Any:
    value, is_image = _as_channels(value)
    width, height = value.shape[-1], value.shape[-2]
    if longer and height > width or not longer and height < width:
        width, height = round(width / height * size), size
    elif longer and width > height or not longer and width < height:
        width, height = size, round(height / width * size)
    else:
        width = height = size
    return _from_channels(_common_upscale(value, width, height, method, "disabled"), is_image)


def _scale_total_pixels(value: Any, megapixels: float, method: str) -> Any:
    value, is_image = _as_channels(value)
    scale = math.sqrt(int(megapixels * 1024.0 * 1024.0) / (value.shape[-1] * value.shape[-2]))
    result = _common_upscale(
        value, round(value.shape[-1] * scale), round(value.shape[-2] * scale), method, "disabled"
    )
    return _from_channels(result, is_image)


def _scale_match_size(value: Any, match: Any, method: str, crop: str) -> Any:
    value, is_image = _as_channels(value)
    match, _ = _as_channels(match)
    return _from_channels(
        _common_upscale(value, match.shape[-1], match.shape[-2], method, crop), is_image
    )


def _scale_to_multiple(value: Any, multiple: int, method: str) -> Any:
    if multiple <= 1:
        return value
    is_image = value.ndim == 4
    height, width = value.shape[1:3]
    target_width, target_height = (width // multiple) * multiple, (height // multiple) * multiple
    if not target_width or not target_height or (target_width == width and target_height == height):
        return value
    scale_width, scale_height = target_width / width, target_height / height
    if scale_width >= scale_height:
        scaled_width, scaled_height = (
            target_width,
            max(target_height, math.ceil(height * scale_width)),
        )
    else:
        scaled_height, scaled_width = (
            target_height,
            max(target_width, math.ceil(width * scale_height)),
        )
    channels, _ = _as_channels(value)
    resized = _from_channels(
        _common_upscale(channels, scaled_width, scaled_height, method, "disabled"), is_image
    )
    x0, y0 = (scaled_width - target_width) // 2, (scaled_height - target_height) // 2
    if is_image:
        return resized[:, y0 : y0 + target_height, x0 : x0 + target_width, :]
    return resized[:, y0 : y0 + target_height, x0 : x0 + target_width]


class ResizeImageMaskNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        crop = _combo(
            "crop",
            _CROP_METHODS,
            "center",
            doc=(
                "How to handle aspect ratio mismatch: 'disabled' stretches to fit, "
                "'center' crops to maintain aspect ratio."
            ),
        )
        dimensions = (
            _number("width", _INT, 512, 0, _MAX_RESOLUTION, 1),
            _number("height", _INT, 512, 0, _MAX_RESOLUTION, 1),
            crop,
        )
        return NodeSchema(
            node_type="ResizeImageMaskNode",
            display_name="Resize Image/Mask",
            description="Resize an image or mask using various scaling methods.",
            category="image/transform",
            inputs=(
                InputSpec("input", _RESIZE_INPUT),
                _combo(
                    "scale_method",
                    _SCALE_METHODS,
                    "area",
                    doc=(
                        "Interpolation algorithm. 'area' is best for downscaling, 'lanczos' "
                        "for upscaling, 'nearest-exact' for pixel art."
                    ),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "resize_type",
                    (
                        DynamicComboOption("scale dimensions", dimensions),
                        DynamicComboOption(
                            "scale by multiplier",
                            (_number("multiplier", _FLOAT, 1.0, 0.01, 8.0, 0.01),),
                        ),
                        DynamicComboOption(
                            "scale longer dimension",
                            (_number("longer_size", _INT, 512, 0, _MAX_RESOLUTION, 1),),
                        ),
                        DynamicComboOption(
                            "scale shorter dimension",
                            (_number("shorter_size", _INT, 512, 0, _MAX_RESOLUTION, 1),),
                        ),
                        DynamicComboOption(
                            "scale width", (_number("width", _INT, 512, 0, _MAX_RESOLUTION, 1),)
                        ),
                        DynamicComboOption(
                            "scale height", (_number("height", _INT, 512, 0, _MAX_RESOLUTION, 1),)
                        ),
                        DynamicComboOption(
                            "scale total pixels",
                            (_number("megapixels", _FLOAT, 1.0, 0.01, 16.0, 0.01),),
                        ),
                        DynamicComboOption(
                            "match size", (InputSpec("match", _IMAGE_OR_MASK), crop)
                        ),
                        DynamicComboOption(
                            "scale to multiple",
                            (_number("multiple", _INT, 8, 1, _MAX_RESOLUTION, 1),),
                        ),
                    ),
                    doc=(
                        "Select how to resize: by exact dimensions, scale factor, matching another "
                        "image, etc."
                    ),
                ),
            ),
            outputs=(OutputSpec("resized", _RESIZE_INPUT),),
            input_order=("input", "resize_type", "scale_method"),
        )

    @classmethod
    def execute(
        cls, *, input: Any, scale_method: str, resize_type: str | Mapping[str, Any], **values: Any
    ) -> Mapping[str, object]:
        value = _image_or_mask(input)
        resize_type = _combo_values("resize_type", resize_type, values)
        selected = resize_type["resize_type"]
        if selected == "scale by multiplier":
            result = _scale_by(value, resize_type["multiplier"], scale_method)
        elif selected == "scale dimensions":
            result = _scale_dimensions(
                value,
                resize_type["width"],
                resize_type["height"],
                scale_method,
                resize_type["crop"],
            )
        elif selected == "scale longer dimension":
            result = _scale_edge(value, resize_type["longer_size"], scale_method, longer=True)
        elif selected == "scale shorter dimension":
            result = _scale_edge(value, resize_type["shorter_size"], scale_method, longer=False)
        elif selected == "scale width":
            result = _scale_dimensions(value, resize_type["width"], 0, scale_method)
        elif selected == "scale height":
            result = _scale_dimensions(value, 0, resize_type["height"], scale_method)
        elif selected == "scale total pixels":
            result = _scale_total_pixels(value, resize_type["megapixels"], scale_method)
        elif selected == "match size":
            result = _scale_match_size(
                value, _image_or_mask(resize_type["match"]), scale_method, resize_type["crop"]
            )
        elif selected == "scale to multiple":
            result = _scale_to_multiple(value, resize_type["multiple"], scale_method)
        else:
            raise ValueError(f"Unsupported resize type: {selected}")
        return cls.outputs(resized=result)


def _batch_images(images: list[Any]) -> Any | None:
    if not images:
        return None
    torch = importlib.import_module("torch")
    images = [materialize_image(image) for image in images]
    channels = max(image.shape[-1] for image in images)
    padded = [
        torch.nn.functional.pad(image, (0, 1), mode="constant", value=1.0)
        if image.shape[-1] < channels
        else image
        for image in images
    ]
    first = padded[0]
    resized = [
        _common_upscale(
            image.movedim(-1, 1), first.shape[2], first.shape[1], "bilinear", "center"
        ).movedim(1, -1)
        if image.shape[1:] != first.shape[1:]
        else image
        for image in padded
    ]
    return torch.cat(resized, dim=0)


def _batch_masks(masks: list[Any]) -> Any | None:
    if not masks:
        return None
    torch = importlib.import_module("torch")
    masks = [_image_or_mask(mask) for mask in masks]
    first = masks[0]
    resized = [
        _common_upscale(
            mask.unsqueeze(1), first.shape[2], first.shape[1], "bilinear", "center"
        ).squeeze(1)
        if mask.shape[1:] != first.shape[1:]
        else mask
        for mask in masks
    ]
    return torch.cat(resized, dim=0)


def _batch_latents(latents: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not latents:
        return None
    torch = importlib.import_module("torch")
    latents = [materialize_latent(latent) for latent in latents]
    result = latents[0].copy()
    first = latents[0]["samples"]
    tensors: list[Any] = []
    result_batch_indices: list[int] = []
    result["batch_index"] = result_batch_indices
    for latent in latents:
        samples = latent["samples"]
        if not hasattr(samples, "detach"):
            samples = torch.as_tensor(samples)
        if samples.shape[1:] != first.shape[1:]:
            samples = _common_upscale(
                samples, first.shape[-1], first.shape[-2], "bilinear", "center"
            )
        tensors.append(samples)
        batch_indices = cast("list[int]", latent.get("batch_index", list(range(samples.shape[0]))))
        result_batch_indices.extend(batch_indices)
    result["samples"] = torch.cat(tensors, dim=0)
    return result


class BatchImagesNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BatchImagesNode",
            display_name="Batch Images",
            category="image/batch",
            input_families=(
                InputFamilySpec(
                    "images", (InputSpec("image", _IMAGE),), 1, 50, member_prefix="image"
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, images: Mapping[str, Any]) -> Mapping[str, object]:
        return cls.outputs(IMAGE=_batch_images(list(images.values())))


class BatchMasksNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BatchMasksNode",
            display_name="Batch Masks",
            category="image/mask",
            input_families=(
                InputFamilySpec("masks", (InputSpec("mask", _MASK),), 1, 50, member_prefix="mask"),
            ),
            outputs=(OutputSpec("MASK", _MASK),),
        )

    @classmethod
    def execute(cls, *, masks: Mapping[str, Any]) -> Mapping[str, object]:
        return cls.outputs(MASK=_batch_masks(list(masks.values())))


class BatchLatentsNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BatchLatentsNode",
            display_name="Batch Latents",
            category="model/latent/batch",
            input_families=(
                InputFamilySpec(
                    "latents", (InputSpec("latent", _LATENT),), 1, 50, member_prefix="latent"
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, latents: Mapping[str, dict[str, Any]]) -> Mapping[str, object]:
        return cls.outputs(LATENT=_batch_latents(list(latents.values())))


class ColorTransfer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ColorTransfer",
            display_name="Transfer Color",
            description="Match the colors of one image to another using various algorithms.",
            category="image/filters",
            inputs=(
                InputSpec("image_target", _IMAGE, doc="Image(s) to apply the color transform to."),
                InputSpec("image_ref", _IMAGE, doc="Reference image(s) to match colors to."),
                _combo("method", ("reinhard_lab", "mkl_lab", "histogram"), "reinhard_lab"),
                _number("strength", _FLOAT, 1.0, 0.0, 10.0, 0.01),
            ),
            combos=(
                DynamicComboSpec(
                    "source_stats",
                    (
                        DynamicComboOption("per_frame"),
                        DynamicComboOption("uniform"),
                        DynamicComboOption(
                            "target_frame", (_number("target_index", _INT, 0, 0, 10000),)
                        ),
                    ),
                    doc=(
                        "per_frame: each frame matched to image_ref individually. uniform: pool "
                        "stats across all source frames as baseline, match to image_ref. "
                        "target_frame: use one chosen frame as the source baseline for the "
                        "transform to image_ref, applied uniformly to all frames (preserves "
                        "relative differences)"
                    ),
                ),
            ),
            outputs=(OutputSpec("image", _IMAGE),),
            input_order=("image_target", "image_ref", "method", "source_stats", "strength"),
        )

    @staticmethod
    def _to_lab(images: Any, index: int, device: Any) -> Any:
        torch = importlib.import_module("torch")
        color = importlib.import_module("kornia.color")
        return color.rgb_to_lab(
            images[index : index + 1].to(device, dtype=torch.float32).permute(0, 3, 1, 2)
        )

    @classmethod
    def _pool_stats(
        cls, images: Any, device: Any, reinhard: bool, epsilon: float
    ) -> tuple[Any, Any]:
        torch = importlib.import_module("torch")
        frames, channels = images.shape[0], images.shape[3]
        pixels = images.shape[1] * images.shape[2]
        mean = torch.zeros(channels, 1, device=device, dtype=torch.float32)
        for index in range(frames):
            mean += cls._to_lab(images, index, device).view(channels, -1).mean(dim=-1, keepdim=True)
        mean /= frames
        accumulator = torch.zeros(
            channels, 1 if reinhard else channels, device=device, dtype=torch.float32
        )
        for index in range(frames):
            centered = cls._to_lab(images, index, device).view(channels, -1) - mean
            accumulator += (
                (centered * centered).mean(dim=-1, keepdim=True)
                if reinhard
                else centered @ centered.T / pixels
            )
        return (
            (mean, torch.sqrt(accumulator / frames).clamp_min_(epsilon))
            if reinhard
            else (mean, accumulator / frames)
        )

    @staticmethod
    def _frame_stats(lab: Any, pixels: int, reinhard: bool, epsilon: float) -> tuple[Any, Any]:
        mean = lab.mean(dim=-1, keepdim=True)
        if reinhard:
            return mean, lab.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(epsilon)
        centered = lab - mean
        return mean, centered @ centered.T / pixels

    @staticmethod
    def _mkl_matrix(source: Any, reference: Any, epsilon: float) -> Any:
        torch = importlib.import_module("torch")
        source_values, source_vectors = torch.linalg.eigh(source)
        source_root = torch.sqrt(source_values.clamp_min(0)).clamp_min_(epsilon)
        scaled_vectors = source_vectors * source_root.unsqueeze(0)
        middle_values, middle_vectors = torch.linalg.eigh(
            scaled_vectors.T @ reference @ scaled_vectors
        )
        middle_root = torch.sqrt(middle_values.clamp_min(0))
        inverse_scaled = source_vectors * (1.0 / source_root).unsqueeze(0)
        return (
            inverse_scaled
            @ ((middle_vectors * middle_root.unsqueeze(0)) @ middle_vectors.T)
            @ inverse_scaled.T
        )

    @staticmethod
    def _histogram_lut(source: Any, reference: Any, bins: int = 256) -> Any:
        torch = importlib.import_module("torch")
        source_bins = (source * (bins - 1)).long().clamp(0, bins - 1)
        reference_bins = (reference * (bins - 1)).long().clamp(0, bins - 1)
        source_histogram = torch.zeros(
            source.shape[0], bins, device=source.device, dtype=source.dtype
        )
        reference_histogram = torch.zeros(
            reference.shape[0], bins, device=reference.device, dtype=reference.dtype
        )
        source_histogram.scatter_add_(1, source_bins, torch.ones_like(source))
        reference_histogram.scatter_add_(1, reference_bins, torch.ones_like(reference))
        source_cdf = source_histogram.cumsum(1)
        reference_cdf = reference_histogram.cumsum(1)
        return torch.searchsorted(
            reference_cdf / reference_cdf[:, -1:], source_cdf / source_cdf[:, -1:]
        ).clamp_max_(bins - 1).float() / (bins - 1)

    @classmethod
    def _pooled_cdf(cls, images: Any, device: Any, bins: int = 256) -> Any:
        torch = importlib.import_module("torch")
        histogram = torch.zeros(images.shape[3], bins, device=device, dtype=torch.float32)
        for index in range(images.shape[0]):
            frame = (
                images[index]
                .to(device, dtype=torch.float32)
                .permute(2, 0, 1)
                .reshape(images.shape[3], -1)
            )
            histogram.scatter_add_(
                1, (frame * (bins - 1)).long().clamp(0, bins - 1), torch.ones_like(frame)
            )
        cdf = histogram.cumsum(1)
        return cdf / cdf[:, -1:]

    @classmethod
    def _histogram_transform(
        cls, target: Any, reference: Any, device: Any, stats: str, target_index: int
    ) -> Any | None:
        if stats == "per_frame":
            return None
        reference_cdf = cls._pooled_cdf(reference, device)
        index = min(target_index, target.shape[0] - 1)
        source = target[index : index + 1] if stats == "target_frame" else target
        return (
            importlib.import_module("torch")
            .searchsorted(reference_cdf, cls._pooled_cdf(source, device))
            .clamp_max_(255)
            .float()
            / 255.0
        )

    @classmethod
    def _lab_transform(
        cls, target: Any, reference: Any, device: Any, stats: str, target_index: int, reinhard: bool
    ) -> Callable[[Any, int], Any]:
        epsilon = 1e-6
        frames, height, width, channels = target.shape
        reference_frames = reference.shape[0]
        pixels, reference_pixels = height * width, reference.shape[1] * reference.shape[2]
        reference_mean: Any = None
        reference_scale: Any = None
        if reference_frames == 1 or stats in ("uniform", "target_frame"):
            reference_mean, reference_scale = cls._pool_stats(reference, device, reinhard, epsilon)
        if stats in ("uniform", "target_frame"):
            if stats == "target_frame":
                source_lab = cls._to_lab(target, min(target_index, frames - 1), device).view(
                    channels, -1
                )
                source_mean, source_scale = cls._frame_stats(source_lab, pixels, reinhard, epsilon)
            else:
                source_mean, source_scale = cls._pool_stats(target, device, reinhard, epsilon)
            if reinhard:
                scale, offset = (
                    reference_scale / source_scale,
                    reference_mean - reference_scale / source_scale * source_mean,
                )
                return lambda source, _: source * scale + offset
            transform = cls._mkl_matrix(source_scale, reference_scale, epsilon)
            offset = reference_mean - transform @ source_mean
            return lambda source, _: transform @ source + offset

        def per_frame(source: Any, index: int) -> Any:
            source_mean, source_scale = cls._frame_stats(source, pixels, reinhard, epsilon)
            if reference_frames == 1:
                ref_mean, ref_scale = reference_mean, reference_scale
            else:
                ref_lab = cls._to_lab(reference, min(index, reference_frames - 1), device).view(
                    channels, -1
                )
                ref_mean, ref_scale = cls._frame_stats(ref_lab, reference_pixels, reinhard, epsilon)
            centered = source - source_mean
            if reinhard:
                return centered * (ref_scale / source_scale) + ref_mean
            return (
                cls._mkl_matrix(centered @ centered.T / pixels, ref_scale, epsilon) @ centered
                + ref_mean
            )

        return per_frame

    @classmethod
    def execute(
        cls,
        *,
        image_target: Any,
        image_ref: Any,
        method: str,
        source_stats: str | Mapping[str, Any],
        strength: float = 1.0,
        **values: Any,
    ) -> Mapping[str, object]:
        target = materialize_image(image_target)
        if strength == 0 or image_ref is None:
            return cls.outputs(image=target)
        torch = importlib.import_module("torch")
        color = importlib.import_module("kornia.color")
        reference = materialize_image(image_ref)
        management = importlib.import_module("dinkster_inference.model_management")
        device, output_device, output_dtype = (
            management.get_torch_device(),
            management.intermediate_device(),
            management.intermediate_dtype(),
        )
        frames, height, width, channels = target.shape
        output = torch.empty(
            frames, height, width, channels, device=output_device, dtype=output_dtype
        )
        source_stats = _combo_values("source_stats", source_stats, values)
        stats, target_index = source_stats["source_stats"], source_stats.get("target_index", 0)
        if method == "histogram":
            uniform = cls._histogram_transform(target, reference, device, stats, target_index)
            for index in range(frames):
                source = target[index].to(device, dtype=torch.float32).permute(2, 0, 1)
                flat = source.reshape(channels, -1)
                lut = uniform
                if lut is None:
                    ref = (
                        reference[min(index, reference.shape[0] - 1)]
                        .to(device, dtype=torch.float32)
                        .permute(2, 0, 1)
                        .reshape(channels, -1)
                    )
                    lut = cls._histogram_lut(flat, ref)
                matched = lut.gather(1, (flat * 255).long().clamp(0, 255)).view(
                    channels, height, width
                )
                output[index] = (
                    (matched if strength == 1.0 else torch.lerp(source, matched, strength))
                    .permute(1, 2, 0)
                    .clamp_(0, 1)
                    .to(device=output_device, dtype=output_dtype)
                )
                report_progress(index + 1, frames)
        else:
            transform = cls._lab_transform(
                target, reference, device, stats, target_index, method == "reinhard_lab"
            )
            for index in range(frames):
                source = cls._to_lab(target, index, device)
                corrected = transform(source.view(channels, -1), index)
                result = (
                    color.lab_to_rgb(corrected.view(1, channels, height, width))
                    if strength == 1.0
                    else color.lab_to_rgb(
                        torch.lerp(source, corrected.view(1, channels, height, width), strength)
                    )
                )
                output[index] = (
                    result.squeeze(0)
                    .permute(1, 2, 0)
                    .clamp_(0, 1)
                    .to(device=output_device, dtype=output_dtype)
                )
                report_progress(index + 1, frames)
        return cls.outputs(image=output)


POST_PROCESSING_NODES = (
    ResizeImageMaskNode,
    BatchImagesNode,
    BatchMasksNode,
    BatchLatentsNode,
    ColorTransfer,
)
