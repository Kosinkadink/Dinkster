"""EasyCache and LazyCache model sampling patches."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)


def _inference_modules() -> tuple[Any, Any]:
    initialize_inference_process()
    model_patcher: Any = importlib.import_module("dinkster_inference.model_patcher")
    patcher_extension: Any = importlib.import_module("dinkster_inference.patcher_extension")
    return model_patcher, patcher_extension


def _extract_tensor(data: Any, output_channels: int) -> tuple[Any, Any | None]:
    if isinstance(data, list):
        video = cast(Any, data[0])
        audio = cast(Any, data[1])
        return video[:, :output_channels], audio[:, :output_channels]
    return data[:, :output_channels], None


class _CacheHolder:
    name = "Cache"
    per_conditioning = False

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        subsample_factor: int,
        offload_cache_diff: bool,
        verbose: bool,
        output_channels: int,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.offload_cache_diff = offload_cache_diff
        self.verbose = verbose
        self.output_channels = output_channels
        self.start_t = 0.0
        self.end_t = 0.0
        self.reset()
        self.relative_transformation_rate = None

    def prepare_timesteps(self, model_sampling: Any) -> _CacheHolder:
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def is_past_end_timestep(self, timestep: Any) -> bool:
        return not (timestep[0] > self.end_t).item()

    def should_do_easycache(self, timestep: Any) -> bool:
        return (timestep[0] <= self.start_t).item()

    def has_x_prev_subsampled(self) -> bool:
        return self.x_prev_subsampled is not None

    def has_output_prev_norm(self) -> bool:
        return self.output_prev_norm is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def check_metadata(self, x: Any) -> bool:
        shape = x.shape[1:] if self.per_conditioning else x.shape
        metadata = (x.device, x.dtype, shape)
        if self.state_metadata is None:
            self.state_metadata = metadata
            return True
        if metadata == self.state_metadata:
            return True
        logging.warning("%s - Tensor shape, dtype or device changed, resetting state", self.name)
        self.reset()
        return False

    def reset(self) -> _CacheHolder:
        self.relative_transformation_rate: Any = 0.0
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.skip_current_step = False
        self.first_cond_uuid: Any | None = None
        self.x_prev_subsampled: Any | None = None
        self.output_prev_subsampled: Any | None = None
        self.output_prev_norm: Any | None = None
        self.cache_diff: Any | None = None
        self.uuid_cache_diffs: dict[Any, Any] = {}
        self.uuid_cache_diffs_audio: dict[Any, Any] = {}
        self.output_change_rates: list[float] = []
        self.approx_output_change_rates: list[float] = []
        self.total_steps_skipped = 0
        self.state_metadata: tuple[Any, Any, Any] | None = None
        return self

    def clone(self) -> _CacheHolder:
        return type(self)(
            self.reuse_threshold,
            self.start_percent,
            self.end_percent,
            self.subsample_factor,
            self.offload_cache_diff,
            self.verbose,
            self.output_channels,
        )


class EasyCacheHolder(_CacheHolder):
    name = "EasyCache"
    per_conditioning = True

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        subsample_factor: int,
        offload_cache_diff: bool,
        verbose: bool,
        output_channels: int,
    ) -> None:
        self.allow_mismatch = True
        self.cut_from_start = True
        super().__init__(
            reuse_threshold,
            start_percent,
            end_percent,
            subsample_factor,
            offload_cache_diff,
            verbose,
            output_channels,
        )

    def reset(self) -> EasyCacheHolder:
        approx_output_change_rates = getattr(self, "approx_output_change_rates", [])
        super().reset()
        self.approx_output_change_rates = approx_output_change_rates
        return self

    def subsample(self, x: Any, uuids: list[Any], clone: bool = True) -> Any:
        batch_offset = x.shape[0] // len(uuids)
        uuid_index = uuids.index(self.first_cond_uuid)
        result = x[uuid_index * batch_offset : (uuid_index + 1) * batch_offset]
        if self.subsample_factor > 1:
            result = result[..., :: self.subsample_factor, :: self.subsample_factor]
        return result.clone() if clone else result

    def can_apply_cache_diff(self, uuids: list[Any]) -> bool:
        return all(uuid in self.uuid_cache_diffs for uuid in uuids)

    def apply_cache_diff(self, x: Any, uuids: list[Any], is_audio: bool = False) -> Any:
        if self.first_cond_uuid in uuids and not is_audio:
            self.total_steps_skipped += 1
        diffs = self.uuid_cache_diffs_audio if is_audio else self.uuid_cache_diffs
        batch_offset = x.shape[0] // len(uuids)
        for index, uuid in enumerate(uuids):
            cache_diff = diffs[uuid]
            batch_slice: list[slice] = [slice(index * batch_offset, (index + 1) * batch_offset)]
            if x.shape[1:] != cache_diff.shape[1:]:
                if not self.allow_mismatch:
                    raise ValueError(
                        f"Cached dims {cache_diff.shape} do not match x dims {x.shape}."
                    )
                for cached, current in zip(cache_diff.shape[1:], x.shape[1:], strict=True):
                    batch_slice.append(
                        slice(current - cached, None)
                        if self.cut_from_start and cached != current
                        else slice(None, cached)
                        if cached != current
                        else slice(None)
                    )
            x[tuple(batch_slice)] += cache_diff.to(x.device)
        return x

    def update_cache_diff(
        self, output: Any, x: Any, uuids: list[Any], is_audio: bool = False
    ) -> None:
        diffs = self.uuid_cache_diffs_audio if is_audio else self.uuid_cache_diffs
        if output.shape[1:] != x.shape[1:]:
            if not self.allow_mismatch:
                raise ValueError(f"Output dims {output.shape} do not match x dims {x.shape}.")
            slices = [slice(None)]
            for output_dim, input_dim in zip(output.shape[1:], x.shape[1:], strict=True):
                slices.append(
                    slice(input_dim - output_dim, None)
                    if self.cut_from_start and output_dim != input_dim
                    else slice(None, output_dim)
                    if output_dim != input_dim
                    else slice(None)
                )
            x = x[tuple(slices)]
        diff = output - x
        batch_offset = diff.shape[0] // len(uuids)
        for index, uuid in enumerate(uuids):
            diffs[uuid] = diff[index * batch_offset : (index + 1) * batch_offset]

    def has_first_cond_uuid(self, uuids: list[Any]) -> bool:
        return self.first_cond_uuid in uuids


class LazyCacheHolder(_CacheHolder):
    name = "LazyCache"

    def subsample(self, x: Any, clone: bool = True) -> Any:
        result = (
            x[..., :: self.subsample_factor, :: self.subsample_factor]
            if self.subsample_factor > 1
            else x
        )
        return result.clone() if clone else result

    def apply_cache_diff(self, x: Any) -> Any:
        assert self.cache_diff is not None
        self.total_steps_skipped += 1
        return x + self.cache_diff.to(x.device)

    def update_cache_diff(self, output: Any, x: Any) -> None:
        self.cache_diff = output - x


def easycache_forward_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
    transformer_options = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options") or args[-2]
    transformer_options = cast(dict[str, Any], transformer_options)
    holder = cast(EasyCacheHolder, transformer_options["easycache"])
    x, audio_x = _extract_tensor(args[0], holder.output_channels)
    sigmas = transformer_options["sigmas"]
    uuids = cast(list[Any], transformer_options["uuids"])
    if sigmas is not None and holder.is_past_end_timestep(sigmas):
        return executor(*args, **kwargs)
    has_first_cond_uuid = holder.has_first_cond_uuid(uuids)
    next_x_prev = x
    input_change = None
    if holder.should_do_easycache(sigmas):
        holder.check_metadata(x)
        can_apply_cache_diff = holder.can_apply_cache_diff(uuids)
        if holder.skip_current_step and can_apply_cache_diff:
            if holder.verbose:
                logging.info(
                    "EasyCache [verbose] - was marked to skip this step by %s. Present uuids: %s",
                    holder.first_cond_uuid,
                    uuids,
                )
            output = holder.apply_cache_diff(x, uuids)
            return (
                [output, holder.apply_cache_diff(audio_x, uuids, True)]
                if audio_x is not None
                else output
            )
        if holder.initial_step:
            holder.first_cond_uuid = uuids[0]
            has_first_cond_uuid = holder.has_first_cond_uuid(uuids)
            holder.initial_step = False
        if has_first_cond_uuid:
            if holder.has_x_prev_subsampled():
                input_change = (
                    (holder.subsample(x, uuids, False) - holder.x_prev_subsampled)
                    .flatten()
                    .abs()
                    .mean()
                )
            if holder.has_output_prev_norm() and holder.has_relative_transformation_rate():
                assert holder.relative_transformation_rate is not None
                assert holder.output_prev_norm is not None
                holder.cumulative_change_rate += (
                    holder.relative_transformation_rate * input_change / holder.output_prev_norm
                )
                if holder.cumulative_change_rate < holder.reuse_threshold and can_apply_cache_diff:
                    if holder.verbose:
                        logging.info(
                            "EasyCache [verbose] - skipping step; cumulative_change_rate: %s, "
                            "reuse_threshold: %s",
                            holder.cumulative_change_rate,
                            holder.reuse_threshold,
                        )
                    holder.skip_current_step = True
                    output = holder.apply_cache_diff(x, uuids)
                    return (
                        [output, holder.apply_cache_diff(audio_x, uuids, True)]
                        if audio_x is not None
                        else output
                    )
                else:
                    if holder.verbose:
                        logging.info(
                            "EasyCache [verbose] - NOT skipping step; cumulative_change_rate: %s, "
                            "reuse_threshold: %s",
                            holder.cumulative_change_rate,
                            holder.reuse_threshold,
                        )
                    holder.cumulative_change_rate = 0.0
    full_output = executor(*args, **kwargs)
    output, audio_output = _extract_tensor(full_output, holder.output_channels)
    if has_first_cond_uuid and holder.has_output_prev_norm():
        assert holder.output_prev_subsampled is not None
        assert holder.output_prev_norm is not None
        output_change = (
            (holder.subsample(output, uuids, False) - holder.output_prev_subsampled)
            .flatten()
            .abs()
            .mean()
        )
        output_change_rate = output_change / holder.output_prev_norm
        if holder.verbose:
            holder.output_change_rates.append(output_change_rate.item())
        if holder.has_relative_transformation_rate():
            assert holder.relative_transformation_rate is not None
            approximation = (
                holder.relative_transformation_rate * input_change
            ) / holder.output_prev_norm
            holder.approx_output_change_rates.append(approximation.item())
            if holder.verbose:
                logging.info("EasyCache [verbose] - approx_output_change_rate: %s", approximation)
        if input_change is not None:
            holder.relative_transformation_rate = output_change / input_change
        if holder.verbose:
            logging.info("EasyCache [verbose] - output_change_rate: %s", output_change_rate)
    holder.update_cache_diff(output, next_x_prev, uuids)
    if audio_output is not None:
        holder.update_cache_diff(audio_output, audio_x, uuids, True)
    if has_first_cond_uuid:
        holder.x_prev_subsampled = holder.subsample(next_x_prev, uuids)
        holder.output_prev_subsampled = holder.subsample(output, uuids)
        holder.output_prev_norm = output.flatten().abs().mean()
        if holder.verbose:
            assert holder.x_prev_subsampled is not None
            logging.info(
                "EasyCache [verbose] - x_prev_subsampled: %s", holder.x_prev_subsampled.shape
            )
    return full_output


def lazycache_predict_noise_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
    timestep = args[1]
    model_options = cast(dict[str, Any], args[2])
    holder = cast(LazyCacheHolder, model_options["transformer_options"]["easycache"])
    if holder.is_past_end_timestep(timestep):
        return executor(*args, **kwargs)
    x = args[0][:, : holder.output_channels]
    next_x_prev = x
    input_change = None
    if holder.should_do_easycache(timestep):
        holder.check_metadata(x)
        if holder.has_x_prev_subsampled():
            input_change = (
                (holder.subsample(x, False) - holder.x_prev_subsampled).flatten().abs().mean()
            )
            if holder.has_output_prev_norm() and holder.has_relative_transformation_rate():
                holder.cumulative_change_rate += (
                    holder.relative_transformation_rate * input_change / holder.output_prev_norm
                )
                if holder.cumulative_change_rate < holder.reuse_threshold:
                    if holder.verbose:
                        logging.info(
                            "LazyCache [verbose] - skipping step; cumulative_change_rate: %s, "
                            "reuse_threshold: %s",
                            holder.cumulative_change_rate,
                            holder.reuse_threshold,
                        )
                    holder.skip_current_step = True
                    return holder.apply_cache_diff(x)
                else:
                    if holder.verbose:
                        logging.info(
                            "LazyCache [verbose] - NOT skipping step; cumulative_change_rate: %s, "
                            "reuse_threshold: %s",
                            holder.cumulative_change_rate,
                            holder.reuse_threshold,
                        )
                    holder.cumulative_change_rate = 0.0
    output = executor(*args, **kwargs)
    if holder.has_output_prev_norm():
        assert holder.output_prev_subsampled is not None
        assert holder.output_prev_norm is not None
        output_change = (
            (holder.subsample(output, False) - holder.output_prev_subsampled).flatten().abs().mean()
        )
        output_change_rate = output_change / holder.output_prev_norm
        if holder.verbose:
            holder.output_change_rates.append(output_change_rate.item())
        if holder.has_relative_transformation_rate():
            assert holder.relative_transformation_rate is not None
            approximation = (
                holder.relative_transformation_rate * input_change
            ) / holder.output_prev_norm
            holder.approx_output_change_rates.append(approximation.item())
            if holder.verbose:
                logging.info("LazyCache [verbose] - approx_output_change_rate: %s", approximation)
        if input_change is not None:
            holder.relative_transformation_rate = output_change / input_change
        if holder.verbose:
            logging.info("LazyCache [verbose] - output_change_rate: %s", output_change_rate)
    holder.update_cache_diff(output, next_x_prev)
    holder.x_prev_subsampled = holder.subsample(next_x_prev)
    holder.output_prev_subsampled = holder.subsample(output)
    holder.output_prev_norm = output.flatten().abs().mean()
    if holder.verbose:
        assert holder.x_prev_subsampled is not None
        logging.info("LazyCache [verbose] - x_prev_subsampled: %s", holder.x_prev_subsampled.shape)
    return output


def easycache_calc_cond_batch_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
    args[-1]["transformer_options"]["easycache"].skip_current_step = False
    return executor(*args, **kwargs)


def easycache_sample_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
    guider = executor.class_obj
    original_options = guider.model_options
    try:
        model_patcher, _ = _inference_modules()
        guider.model_options = model_patcher.create_model_options_clone(original_options)
        holder = guider.model_options["transformer_options"]["easycache"].clone()
        guider.model_options["transformer_options"]["easycache"] = holder.prepare_timesteps(
            guider.model_patcher.model.model_sampling
        )
        logging.info(
            "%s enabled - threshold: %s, start_percent: %s, end_percent: %s",
            holder.name,
            holder.reuse_threshold,
            holder.start_percent,
            holder.end_percent,
        )
        return executor(*args, **kwargs)
    finally:
        holder = guider.model_options["transformer_options"]["easycache"]
        if holder.verbose:
            logging.info(
                "%s [verbose] - output_change_rates %d: %s",
                holder.name,
                len(holder.output_change_rates),
                holder.output_change_rates,
            )
            logging.info(
                "%s [verbose] - approx_output_change_rates %d: %s",
                holder.name,
                len(holder.approx_output_change_rates),
                holder.approx_output_change_rates,
            )
        total_steps = len(args[3]) - 1
        try:
            speedup = total_steps / (total_steps - holder.total_steps_skipped)
        except ZeroDivisionError:
            speedup = 1.0
        logging.info(
            "%s - skipped %d/%d steps (%.2fx speedup).",
            holder.name,
            holder.total_steps_skipped,
            total_steps,
            speedup,
        )
        holder.reset()
        guider.model_options = original_options


class _CacheNode(Node):
    holder_type: type[_CacheHolder]
    wrapper_names: tuple[str, ...]
    description: str

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=cls.__name__,
            display_name=cls.__name__,
            category="advanced/debug",
            description=cls.description,
            inputs=(
                InputSpec("model", _MODEL, doc=f"The model to add {cls.__name__} to."),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_FLOAT),
                        default=default,
                        doc=(
                            "The threshold for reusing cached steps."
                            if name == "reuse_threshold"
                            else "The relative sampling step to "
                            f"{'begin' if name == 'start_percent' else 'end'} "
                            f"use of {cls.__name__}."
                        ),
                        advanced=True,
                        widget=NumberWidget(min=0.0, max=maximum, step=0.01),
                    )
                    for name, default, maximum in (
                        ("reuse_threshold", 0.2, 3.0),
                        ("start_percent", 0.15, 1.0),
                        ("end_percent", 0.95, 1.0),
                    )
                ),
                InputSpec(
                    "verbose",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=False,
                    doc="Whether to log verbose information.",
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL, doc=f"The model with {cls.__name__}."),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        verbose: bool,
    ) -> Mapping[str, object]:
        _, patcher_extension = _inference_modules()
        patched = model.clone()
        patched.model_options["transformer_options"]["easycache"] = cls.holder_type(
            reuse_threshold,
            start_percent,
            end_percent,
            8,
            False,
            verbose,
            patched.model.latent_format.latent_channels,
        )
        patched.add_wrapper_with_key(
            patcher_extension.WrappersMP.OUTER_SAMPLE,
            cls.wrapper_names[0],
            easycache_sample_wrapper,
        )
        for name in cls.wrapper_names[1:]:
            wrapper = (
                easycache_calc_cond_batch_wrapper
                if name == "easycache"
                else lazycache_predict_noise_wrapper
            )
            point = (
                patcher_extension.WrappersMP.CALC_COND_BATCH
                if name == "easycache"
                else patcher_extension.WrappersMP.PREDICT_NOISE
            )
            patched.add_wrapper_with_key(point, name, wrapper)
        if cls is EasyCache:
            patched.add_wrapper_with_key(
                patcher_extension.WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper
            )
        return cls.outputs(MODEL=patched)


class EasyCache(_CacheNode):
    holder_type = EasyCacheHolder
    wrapper_names = ("easycache", "easycache")
    description = "Native EasyCache implementation."


class LazyCache(_CacheNode):
    holder_type = LazyCacheHolder
    wrapper_names = ("lazycache", "lazycache")
    description = (
        "A homebrew version of EasyCache - even 'easier' version of EasyCache to implement. "
        "Overall works worse than EasyCache, but better in some rare cases AND universal "
        "compatibility with everything in ComfyUI."
    )


CACHE_NODES = (EasyCache, LazyCache)
