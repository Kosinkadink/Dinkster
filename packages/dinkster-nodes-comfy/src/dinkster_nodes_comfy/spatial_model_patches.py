"""Spatial attention and block patches for diffusion models."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_CATEGORY = "model/patch/unet"
_UPSCALE_METHODS = ("bicubic", "nearest-exact", "bilinear", "area", "bislerp")


def _nothing(value: Any, mode: str | None = None) -> Any:
    return value


def _gather(input_value: Any, dim: int, index: Any) -> Any:
    torch = importlib.import_module("torch")
    if input_value.shape[-1] == 1:
        return torch.gather(
            input_value.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1),
        ).squeeze(-1)
    return torch.gather(input_value, dim, index)


def bipartite_soft_matching_random2d(
    metric: Any, width: int, height: int, stride_x: int, stride_y: int, remove: int, no_rand: bool
) -> tuple[Callable[[Any, str], Any], Callable[[Any], Any]]:
    """Build the random two-dimensional token merge and inverse callbacks."""
    torch = importlib.import_module("torch")
    batch, tokens, _ = metric.shape
    if remove <= 0 or width == 1 or height == 1:
        return _nothing, _nothing

    gather = _gather if metric.device.type == "mps" else torch.gather
    with torch.no_grad():
        height_regions, width_regions = height // stride_y, width // stride_x
        if no_rand:
            random_index = torch.zeros(
                height_regions, width_regions, 1, device=metric.device, dtype=torch.int64
            )
        else:
            random_index = torch.randint(
                stride_y * stride_x,
                size=(height_regions, width_regions, 1),
                device=metric.device,
            )
        index_view = torch.zeros(
            height_regions,
            width_regions,
            stride_y * stride_x,
            device=metric.device,
            dtype=torch.int64,
        )
        index_view.scatter_(
            dim=2,
            index=random_index,
            src=-torch.ones_like(random_index, dtype=random_index.dtype),
        )
        index_view = (
            index_view.view(height_regions, width_regions, stride_y, stride_x)
            .transpose(1, 2)
            .reshape(height_regions * stride_y, width_regions * stride_x)
        )
        if (height_regions * stride_y) < height or (width_regions * stride_x) < width:
            index_buffer = torch.zeros(height, width, device=metric.device, dtype=torch.int64)
            index_buffer[: height_regions * stride_y, : width_regions * stride_x] = index_view
        else:
            index_buffer = index_view
        random_index = index_buffer.reshape(1, -1, 1).argsort(dim=1)
        destination_count = height_regions * width_regions
        source_index = random_index[:, destination_count:, :]
        destination_index = random_index[:, :destination_count, :]

        def split(value: Any) -> tuple[Any, Any]:
            channels = value.shape[-1]
            source = gather(
                value, dim=1, index=source_index.expand(batch, tokens - destination_count, channels)
            )
            destination = gather(
                value, dim=1, index=destination_index.expand(batch, destination_count, channels)
            )
            return source, destination

        normalized_metric = metric / metric.norm(dim=-1, keepdim=True)
        source_metric, destination_metric = split(normalized_metric)
        scores = source_metric @ destination_metric.transpose(-1, -2)
        remove = min(source_metric.shape[1], remove)
        node_max, node_index = scores.max(dim=-1)
        edge_index = node_max.argsort(dim=-1, descending=True)[..., None]
        unmerged_index = edge_index[..., remove:, :]
        source_merge_index = edge_index[..., :remove, :]
        destination_merge_index = gather(node_index[..., None], dim=-2, index=source_merge_index)

    def merge(value: Any, mode: str = "mean") -> Any:
        source, destination = split(value)
        count, source_tokens, channels = source.shape
        unmerged = gather(
            source,
            dim=-2,
            index=unmerged_index.expand(count, source_tokens - remove, channels),
        )
        source = gather(source, dim=-2, index=source_merge_index.expand(count, remove, channels))
        destination = destination.scatter_reduce(
            -2, destination_merge_index.expand(count, remove, channels), source, reduce=mode
        )
        return torch.cat([unmerged, destination], dim=1)

    def unmerge(value: Any) -> Any:
        unmerged_length = unmerged_index.shape[1]
        unmerged, destination = value[..., :unmerged_length, :], value[..., unmerged_length:, :]
        _, _, channels = unmerged.shape
        source = gather(
            destination, dim=-2, index=destination_merge_index.expand(batch, remove, channels)
        )
        output = torch.zeros(batch, tokens, channels, device=value.device, dtype=value.dtype)
        output.scatter_(
            dim=-2,
            index=destination_index.expand(batch, destination_count, channels),
            src=destination,
        )
        output.scatter_(
            dim=-2,
            index=gather(
                source_index.expand(batch, source_index.shape[1], 1), dim=1, index=unmerged_index
            ).expand(batch, unmerged_length, channels),
            src=unmerged,
        )
        output.scatter_(
            dim=-2,
            index=gather(
                source_index.expand(batch, source_index.shape[1], 1),
                dim=1,
                index=source_merge_index,
            ).expand(batch, remove, channels),
            src=source,
        )
        return output

    return merge, unmerge


def _tome_functions(
    value: Any, ratio: float, original_shape: Any
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    _, _, original_height, original_width = cast(tuple[int, int, int, int], original_shape)
    original_tokens = original_height * original_width
    downsample = int(math.ceil(math.sqrt(original_tokens // value.shape[1])))
    if downsample <= 1:
        width = int(math.ceil(original_width / downsample))
        height = int(math.ceil(original_height / downsample))
        return bipartite_soft_matching_random2d(
            value, width, height, 2, 2, int(value.shape[1] * ratio), False
        )
    return _nothing, _nothing


class TomePatchModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TomePatchModel",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "ratio",
                    _FLOAT,
                    default=0.3,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, ratio: float) -> Mapping[str, object]:
        unmerge: Callable[[Any], Any] | None = None

        def tome_input(
            query: Any, key: Any, value: Any, extra_options: Mapping[str, Any]
        ) -> tuple[Any, Any, Any]:
            nonlocal unmerge
            merge, unmerge = _tome_functions(query, ratio, extra_options["original_shape"])
            return merge(query), key, value

        def tome_output(value: Any, extra_options: Mapping[str, Any]) -> Any:
            del extra_options
            assert unmerge is not None
            return unmerge(value)

        patched = model.clone()
        patched.set_model_attn1_patch(tome_input)
        patched.set_model_attn1_output_patch(tome_output)
        return cls.outputs(MODEL=patched)


def _random_divisor(value: int, minimum: int, maximum_options: int = 1) -> int:
    torch = importlib.import_module("torch")
    minimum = min(minimum, value)
    divisors = [candidate for candidate in range(minimum, value + 1) if value % candidate == 0]
    options = [value // candidate for candidate in divisors[:maximum_options]]
    if len(options) - 1 > 0:
        index = torch.randint(low=0, high=len(options) - 1, size=(1,)).item()
    else:
        index = 0
    return options[index]


class HyperTile(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HyperTile",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "tile_size",
                    _INT,
                    default=256,
                    advanced=True,
                    widget=NumberWidget(min=1, max=2048),
                ),
                InputSpec(
                    "swap_size", _INT, default=2, advanced=True, widget=NumberWidget(min=1, max=128)
                ),
                InputSpec(
                    "max_depth", _INT, default=0, advanced=True, widget=NumberWidget(min=0, max=10)
                ),
                InputSpec("scale_depth", _BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, tile_size: int, swap_size: int, max_depth: int, scale_depth: bool
    ) -> Mapping[str, object]:
        latent_tile_size = max(32, tile_size) // 8
        temporary: tuple[int, int, int, int] | None = None

        def hypertile_input(
            query: Any, key: Any, value: Any, extra_options: Mapping[str, Any]
        ) -> tuple[Any, Any, Any]:
            nonlocal temporary
            model_channels = query.shape[-2]
            original_shape = extra_options["original_shape"]
            apply_to = [
                (original_shape[-2] / (2**depth)) * (original_shape[-1] / (2**depth))
                for depth in range(max_depth + 1)
            ]
            if model_channels not in apply_to:
                return query, key, value
            aspect_ratio = original_shape[-1] / original_shape[-2]
            tokens = query.size(1)
            height, width = (
                round(math.sqrt(tokens * aspect_ratio)),
                round(math.sqrt(tokens / aspect_ratio)),
            )
            factor = 2 ** apply_to.index(model_channels) if scale_depth else 1
            tiles_height = _random_divisor(height, latent_tile_size * factor, swap_size)
            tiles_width = _random_divisor(width, latent_tile_size * factor, swap_size)
            if tiles_height * tiles_width > 1:
                batch, _, channels = query.shape
                query = (
                    query.reshape(
                        batch,
                        tiles_height,
                        height // tiles_height,
                        tiles_width,
                        width // tiles_width,
                        channels,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(
                        batch * tiles_height * tiles_width,
                        (height // tiles_height) * (width // tiles_width),
                        channels,
                    )
                )
                temporary = (tiles_height, tiles_width, height, width)
            return query, key, value

        def hypertile_output(output: Any, extra_options: Mapping[str, Any]) -> Any:
            nonlocal temporary
            del extra_options
            if temporary is not None:
                tiles_height, tiles_width, height, width = temporary
                temporary = None
                batches, tokens, channels = output.shape
                batch = batches // (tiles_height * tiles_width)
                output = (
                    output.reshape(batch, tiles_height, tiles_width, tokens, channels)
                    .reshape(
                        batch,
                        tiles_height,
                        tiles_width,
                        height // tiles_height,
                        width // tiles_width,
                        channels,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(batch, height * width, channels)
                )
            return output

        patched = model.clone()
        patched.set_model_attn1_patch(hypertile_input)
        patched.set_model_attn1_output_patch(hypertile_output)
        return cls.outputs(MODEL=patched)


class PatchModelAddDownscale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        method_widget = ComboWidget(tuple(ComboOption(method) for method in _UPSCALE_METHODS))
        return NodeSchema(
            node_type="PatchModelAddDownscale",
            display_name="PatchModelAddDownscale (Kohya Deep Shrink)",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "block_number",
                    _INT,
                    default=3,
                    advanced=True,
                    widget=NumberWidget(min=1, max=32, step=1),
                ),
                InputSpec(
                    "downscale_factor",
                    _FLOAT,
                    default=2.0,
                    widget=NumberWidget(min=0.1, max=9.0, step=0.001),
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=0.35,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("downscale_after_skip", _BOOLEAN, default=True, advanced=True),
                InputSpec("downscale_method", _COMBO, widget=method_widget),
                InputSpec("upscale_method", _COMBO, widget=method_widget),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        block_number: int,
        downscale_factor: float,
        start_percent: float,
        end_percent: float,
        downscale_after_skip: bool,
        downscale_method: str,
        upscale_method: str,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        sampling = model.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def input_block_patch(hidden: Any, transformer_options: Mapping[str, Any]) -> Any:
            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                if sigma <= sigma_start and sigma >= sigma_end:
                    upscale = importlib.import_module("dinkster_inference.utils").common_upscale
                    hidden = upscale(
                        hidden,
                        round(hidden.shape[-1] * (1.0 / downscale_factor)),
                        round(hidden.shape[-2] * (1.0 / downscale_factor)),
                        downscale_method,
                        "disabled",
                    )
            return hidden

        def output_block_patch(
            hidden: Any, skip: Any, transformer_options: Mapping[str, Any]
        ) -> tuple[Any, Any]:
            del transformer_options
            if hidden.shape[2] != skip.shape[2]:
                hidden = importlib.import_module("dinkster_inference.utils").common_upscale(
                    hidden, skip.shape[-1], skip.shape[-2], upscale_method, "disabled"
                )
            return hidden, skip

        patched = model.clone()
        if downscale_after_skip:
            patched.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            patched.set_model_input_block_patch(input_block_patch)
        patched.set_model_output_block_patch(output_block_patch)
        return cls.outputs(MODEL=patched)


SPATIAL_MODEL_PATCH_NODES = (TomePatchModel, HyperTile, PatchModelAddDownscale)
