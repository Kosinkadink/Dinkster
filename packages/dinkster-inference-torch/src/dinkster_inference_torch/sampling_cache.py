from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from dinkster_inference import (
    AutoregressiveDenoiser,
    Denoiser,
    GuidanceEvaluationRequest,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    SamplerInfo,
    UncondDenoiser,
)

if TYPE_CHECKING:
    from .guidance import CompiledConditioningPlan


class SamplingCacheError(RuntimeError):
    pass


GuidanceEvaluator = Callable[
    [GuidanceEvaluationRequest[torch.Tensor]], GuidancePredictions[torch.Tensor]
]


class GuidanceEvaluationCache(Protocol):
    def evaluate(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
        plan: CompiledConditioningPlan,
        evaluate: GuidanceEvaluator,
    ) -> GuidancePredictions[torch.Tensor]: ...


_ACTIVE_GUIDANCE_CACHE: ContextVar[GuidanceEvaluationCache | None] = ContextVar(
    "dinkster_active_guidance_cache", default=None
)


def active_guidance_evaluation_cache() -> GuidanceEvaluationCache | None:
    return _ACTIVE_GUIDANCE_CACHE.get()


@contextmanager
def _use_guidance_evaluation_cache(
    cache: GuidanceEvaluationCache,
) -> Generator[None, None, None]:
    active = _ACTIVE_GUIDANCE_CACHE.get()
    if active is not None:
        raise SamplingCacheError("sampling caches cannot be nested")
    token = _ACTIVE_GUIDANCE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_GUIDANCE_CACHE.reset(token)


@dataclass(frozen=True, slots=True)
class LazyCacheConfig:
    """Run-scoped approximate caching of complete guided denoiser results."""

    reuse_threshold: float = 0.2
    start_percent: float = 0.15
    end_percent: float = 0.95
    subsample_factor: int = 8

    def __post_init__(self) -> None:
        for name, value in (
            ("reuse_threshold", self.reuse_threshold),
            ("start_percent", self.start_percent),
            ("end_percent", self.end_percent),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
        if not 0.0 <= self.reuse_threshold <= 3.0:
            raise ValueError("reuse_threshold must be in [0.0, 3.0]")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError(
                "cache percentages must satisfy 0 <= start_percent <= end_percent <= 1"
            )
        if type(self.subsample_factor) is not int or self.subsample_factor < 1:
            raise ValueError("subsample_factor must be a positive integer")

    def wrap_denoiser(
        self,
        denoiser: Denoiser[torch.Tensor],
        sigmas: Sequence[float],
        info: SamplerInfo,
    ) -> Denoiser[torch.Tensor]:
        del sigmas
        if info.percent_to_sigma is None:
            raise SamplingCacheError("LazyCache requires the model's percent-to-sigma mapping")
        return _LazyCacheDenoiser(
            denoiser,
            reuse_threshold=self.reuse_threshold,
            start_sigma=info.percent_to_sigma(self.start_percent),
            end_sigma=info.percent_to_sigma(self.end_percent),
            subsample_factor=self.subsample_factor,
        )


class _LazyCacheDenoiser:
    def __init__(
        self,
        inner: Denoiser[torch.Tensor],
        *,
        reuse_threshold: float,
        start_sigma: float,
        end_sigma: float,
        subsample_factor: int,
    ) -> None:
        self._inner = inner
        self._reuse_threshold = reuse_threshold
        self._start_sigma = start_sigma
        self._end_sigma = end_sigma
        self._subsample_factor = subsample_factor
        self._metadata: tuple[torch.Size, torch.dtype, torch.device] | None = None
        self._x_previous: torch.Tensor | None = None
        self._output_previous: torch.Tensor | None = None
        self._output_previous_norm: torch.Tensor | None = None
        self._relative_change_rate: torch.Tensor | None = None
        self._cumulative_change: torch.Tensor | float = 0.0
        self._cache_diff: torch.Tensor | None = None
        self._uncond_cache_diff: torch.Tensor | None = None

    def _subsample(self, value: torch.Tensor) -> torch.Tensor:
        factor = self._subsample_factor
        if factor == 1:
            return value.clone()
        return value[..., ::factor, ::factor].clone()

    def _reset(self) -> None:
        self._x_previous = None
        self._output_previous = None
        self._output_previous_norm = None
        self._relative_change_rate = None
        self._cumulative_change = 0.0
        self._cache_diff = None
        self._uncond_cache_diff = None

    def _prepare(
        self,
        x: torch.Tensor,
        sigma: float,
        *,
        require_uncond: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        if sigma > self._start_sigma:
            return False, None
        metadata = (x.shape, x.dtype, x.device)
        if self._metadata is None:
            self._metadata = metadata
        elif self._metadata != metadata:
            self._reset()
            self._metadata = metadata
            return False, None
        if self._x_previous is None:
            return False, None
        input_change = (self._subsample(x) - self._x_previous).flatten().abs().mean()
        if self._output_previous_norm is None or self._relative_change_rate is None:
            return False, input_change
        estimated_change = self._relative_change_rate * input_change / self._output_previous_norm
        self._cumulative_change += estimated_change
        has_cached_output = self._cache_diff is not None and (
            not require_uncond or self._uncond_cache_diff is not None
        )
        if has_cached_output and bool(self._cumulative_change < self._reuse_threshold):
            return True, input_change
        self._cumulative_change = 0.0
        return False, input_change

    def _record(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        input_change: torch.Tensor | None,
        *,
        uncond: torch.Tensor | None,
    ) -> None:
        if self._output_previous_norm is not None and self._output_previous is not None:
            output_change = (self._subsample(output) - self._output_previous).flatten().abs().mean()
            if input_change is not None:
                self._relative_change_rate = output_change / input_change
        self._cache_diff = output - x
        self._uncond_cache_diff = None if uncond is None else uncond - x
        self._x_previous = self._subsample(x)
        self._output_previous = self._subsample(output)
        self._output_previous_norm = output.flatten().abs().mean()

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= self._end_sigma:
            return self._inner(x, sigma)
        cached, input_change = self._prepare(x, sigma, require_uncond=False)
        if cached:
            assert self._cache_diff is not None
            return x + self._cache_diff.to(x.device)
        output = self._inner(x, sigma)
        self._record(x, output, input_change, uncond=None)
        return output

    def call_with_uncond(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self._inner, UncondDenoiser):
            raise SamplingCacheError("LazyCache received a denoiser without unconditional output")
        if sigma <= self._end_sigma:
            return self._inner.call_with_uncond(x, sigma)
        cached, input_change = self._prepare(x, sigma, require_uncond=True)
        if cached:
            assert self._cache_diff is not None
            assert self._uncond_cache_diff is not None
            return (
                x + self._cache_diff.to(x.device),
                x + self._uncond_cache_diff.to(x.device),
            )
        output, uncond = self._inner.call_with_uncond(x, sigma)
        self._record(x, output, input_change, uncond=uncond)
        return output, uncond


@dataclass(frozen=True, slots=True)
class EasyCacheConfig:
    """Run-scoped approximate caching of lane-keyed model residuals."""

    reuse_threshold: float = 0.2
    start_percent: float = 0.15
    end_percent: float = 0.95
    subsample_factor: int = 8

    def __post_init__(self) -> None:
        for name, value in (
            ("reuse_threshold", self.reuse_threshold),
            ("start_percent", self.start_percent),
            ("end_percent", self.end_percent),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
        if not 0.0 <= self.reuse_threshold <= 3.0:
            raise ValueError("reuse_threshold must be in [0.0, 3.0]")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError(
                "cache percentages must satisfy 0 <= start_percent <= end_percent <= 1"
            )
        if type(self.subsample_factor) is not int or self.subsample_factor < 1:
            raise ValueError("subsample_factor must be a positive integer")

    def wrap_denoiser(
        self,
        denoiser: Denoiser[torch.Tensor],
        sigmas: Sequence[float],
        info: SamplerInfo,
    ) -> Denoiser[torch.Tensor]:
        del sigmas
        if info.percent_to_sigma is None:
            raise SamplingCacheError("EasyCache requires the model's percent-to-sigma mapping")
        if isinstance(denoiser, AutoregressiveDenoiser):
            raise SamplingCacheError("EasyCache does not support autoregressive denoisers")
        return _EasyCacheDenoiser(
            denoiser,
            _EasyCacheState(
                reuse_threshold=self.reuse_threshold,
                start_sigma=info.percent_to_sigma(self.start_percent),
                end_sigma=info.percent_to_sigma(self.end_percent),
                subsample_factor=self.subsample_factor,
            ),
        )


_TensorMetadata = tuple[
    torch.Size,
    torch.dtype,
    torch.device,
    torch.layout,
    tuple[int, ...],
]


class _EasyCacheState:
    def __init__(
        self,
        *,
        reuse_threshold: float,
        start_sigma: float,
        end_sigma: float,
        subsample_factor: int,
    ) -> None:
        self._reuse_threshold = reuse_threshold
        self._start_sigma = start_sigma
        self._end_sigma = end_sigma
        self._subsample_factor = subsample_factor
        self._metadata: _TensorMetadata | None = None
        self._plan_digest: str | None = None
        self._driver_lane_id: str | None = None
        self._lane_sources: dict[str, object | None] = {}
        self._x_previous: torch.Tensor | None = None
        self._output_previous: torch.Tensor | None = None
        self._output_previous_norm: torch.Tensor | None = None
        self._relative_change_rate: torch.Tensor | None = None
        self._cumulative_change: torch.Tensor | float = 0.0
        self._residuals: dict[str, torch.Tensor] = {}
        self.claims = 0
        self.hits = 0
        self.misses = 0

    def _subsample(self, value: torch.Tensor) -> torch.Tensor:
        factor = self._subsample_factor
        if factor == 1:
            return value.clone()
        return value[..., ::factor, ::factor].clone()

    def _reset(self) -> None:
        self._plan_digest = None
        self._driver_lane_id = None
        self._lane_sources.clear()
        self._x_previous = None
        self._output_previous = None
        self._output_previous_norm = None
        self._relative_change_rate = None
        self._cumulative_change = 0.0
        self._residuals.clear()

    @staticmethod
    def _tensor_metadata(value: torch.Tensor) -> _TensorMetadata:
        return (
            value.shape,
            value.dtype,
            value.device,
            value.layout,
            value.stride(),
        )

    def _matches_plan(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
        plan: CompiledConditioningPlan,
    ) -> bool:
        if self._plan_digest != plan.plan_digest:
            return False
        ids = tuple(lane.id for lane in request.plan.lanes)
        if tuple(self._lane_sources) != ids:
            return False
        return all(self._lane_sources[lane.id] is lane.conditioning for lane in request.plan.lanes)

    def _bind_plan(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
        plan: CompiledConditioningPlan,
    ) -> None:
        requested = tuple((lane.id, lane.role) for lane in request.plan.lanes)
        compiled = tuple((lane.lane_id, lane.role) for lane in plan.lanes)
        if compiled != requested:
            raise SamplingCacheError("EasyCache received a mismatched conditioning plan")
        primary = next(lane for lane in request.plan.lanes if lane.id == request.plan.primary_id)
        if primary.conditioning is None:
            raise SamplingCacheError("EasyCache requires a model-evaluated primary lane")
        if not plan.calls or not plan.calls[0].lane_ids:
            raise SamplingCacheError("EasyCache requires at least one model-evaluated lane")
        self._plan_digest = plan.plan_digest
        self._driver_lane_id = plan.calls[0].lane_ids[0]
        self._lane_sources = {lane.id: lane.conditioning for lane in request.plan.lanes}

    def _prepare(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> tuple[bool, torch.Tensor | None]:
        x = request.input
        if self._x_previous is None:
            return False, None
        input_change = (self._subsample(x) - self._x_previous).flatten().abs().mean()
        if self._output_previous_norm is None or self._relative_change_rate is None:
            return False, input_change
        estimated_change = self._relative_change_rate * input_change / self._output_previous_norm
        self._cumulative_change += estimated_change
        model_lane_ids = {lane.id for lane in request.plan.lanes if lane.conditioning is not None}
        if model_lane_ids == set(self._residuals) and bool(
            self._cumulative_change < self._reuse_threshold
        ):
            return True, input_change
        self._cumulative_change = 0.0
        return False, input_change

    def _record(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
        predictions: GuidancePredictions[torch.Tensor],
        input_change: torch.Tensor | None,
    ) -> None:
        by_id = {item.lane_id: item for item in predictions.items}
        if self._driver_lane_id is None:
            raise SamplingCacheError("EasyCache has no decision-driving conditioning lane")
        driver = by_id[self._driver_lane_id].value
        if self._output_previous_norm is not None and self._output_previous is not None:
            output_change = (self._subsample(driver) - self._output_previous).flatten().abs().mean()
            if input_change is not None:
                self._relative_change_rate = output_change / input_change
        self._residuals = {
            lane.id: by_id[lane.id].value - request.input
            for lane in request.plan.lanes
            if lane.conditioning is not None
        }
        self._x_previous = self._subsample(request.input)
        self._output_previous = self._subsample(driver)
        self._output_previous_norm = driver.flatten().abs().mean()

    def _cached_predictions(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    (
                        torch.zeros_like(request.input)
                        if lane.conditioning is None
                        else request.input + self._residuals[lane.id]
                    ),
                    (
                        GuidancePredictionSource.SYNTHETIC_ZERO
                        if lane.conditioning is None
                        else GuidancePredictionSource.MODEL
                    ),
                )
                for lane in request.plan.lanes
            )
        )

    def evaluate(
        self,
        request: GuidanceEvaluationRequest[torch.Tensor],
        plan: CompiledConditioningPlan,
        evaluate: GuidanceEvaluator,
    ) -> GuidancePredictions[torch.Tensor]:
        self.claims += 1
        metadata = self._tensor_metadata(request.input)
        if self._metadata is None:
            self._metadata = metadata
        elif self._metadata != metadata:
            self._reset()
            self._metadata = metadata
        sigma = request.execution.current_sigma
        if sigma > self._start_sigma or sigma <= self._end_sigma:
            self._reset()
            return evaluate(request)
        if not self._matches_plan(request, plan):
            self._reset()
            self._bind_plan(request, plan)
        cached, input_change = self._prepare(request)
        if cached:
            self.hits += 1
            return self._cached_predictions(request)
        self.misses += 1
        predictions = evaluate(request)
        self._record(request, predictions, input_change)
        return predictions


class _EasyCacheDenoiser:
    def __init__(self, inner: Denoiser[torch.Tensor], state: _EasyCacheState) -> None:
        self._inner = inner
        self._state = state

    def _invoke(self, call: Callable[[], torch.Tensor]) -> torch.Tensor:
        claims = self._state.claims
        with _use_guidance_evaluation_cache(self._state):
            output = call()
        if self._state.claims == claims:
            raise SamplingCacheError(
                "EasyCache requires a denoiser that uses the shared guidance evaluation seam"
            )
        return output

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return self._invoke(lambda: self._inner(x, sigma))

    def call_with_uncond(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self._inner, UncondDenoiser):
            raise SamplingCacheError("EasyCache received a denoiser without unconditional output")
        claims = self._state.claims
        with _use_guidance_evaluation_cache(self._state):
            output = self._inner.call_with_uncond(x, sigma)
        if self._state.claims == claims:
            raise SamplingCacheError(
                "EasyCache requires a denoiser that uses the shared guidance evaluation seam"
            )
        return output


__all__ = ["EasyCacheConfig", "LazyCacheConfig", "SamplingCacheError"]
