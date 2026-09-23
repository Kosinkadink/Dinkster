"""Windowed conditioning evaluation over a temporal latent axis.

:func:`windowed_conditioning_evaluation` wraps one family-declared
:class:`~dinkster_inference_torch.guidance.ConditioningEvaluation` so every
model application runs per context window and the per-window outputs
fuse back into the full-length result beneath the guidance combine,
matching the reference behavior of ComfyUI's IndexListContextHandler
(comfy/context_windows.py @ b78cec87) around calc_cond_batch. Window
schedules, fuse weights, and the FreeNoise shuffle plan come from the
torch-free planner in :mod:`dinkster_inference.context_windows`; this
module realizes them on tensors.

The wrapper never consumes attention-kind guidance contributions: its
``evaluate_batch_attention`` is None, so guided execution refuses
attention-level guidance before sampling instead of silently applying
it unwindowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

import torch
from dinkster_inference.context_windows import (
    ContextFuseMethod,
    ContextWindowsError,
    ContextWindowsSpec,
    context_window_plan,
    freenoise_plan,
    plan_windows,
    relative_bias,
    step_index_for_sigma,
)

from .guidance import ConditioningEvaluation

PreparedCondition = TypeVar("PreparedCondition")

_WindowRun = Callable[[torch.Tensor, tuple[int, ...]], tuple[torch.Tensor, ...]]

__all__ = [
    "apply_freenoise",
    "windowed_conditioning_evaluation",
]


def apply_freenoise(
    noise: torch.Tensor, dim: int, length: int, overlap: int, seed: int
) -> torch.Tensor:
    """FreeNoise initial-noise shuffle: a new tensor whose frames past the
    first window reuse seeded permutations of earlier frames.

    Spans come from :func:`~dinkster_inference.context_windows.freenoise_plan`
    and are applied sequentially onto one buffer, so a later span's source
    reads any earlier span's already-shuffled frames exactly as the
    reference's in-place loop does. The input tensor is not modified."""

    if not 0 <= dim < noise.ndim:
        raise ContextWindowsError(f"dim {dim} is out of range for a {noise.ndim}-d noise tensor")
    result = noise.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for source_start, target_start, count in freenoise_plan(noise.size(dim), length, overlap):
        order = torch.randperm(count, generator=generator, device="cpu") + source_start
        source: list[slice | torch.Tensor] = [slice(None)] * result.ndim
        source[dim] = order.to(result.device)
        target: list[slice | torch.Tensor] = [slice(None)] * result.ndim
        target[dim] = slice(target_start, target_start + count)
        result[tuple(target)] = result[tuple(source)]
    return result


class _StepTracker:
    """Maps each evaluation sigma to its schedule step, keeping the prior
    step for off-schedule sigmas (multi-stage solver substeps)."""

    def __init__(self, sigmas: tuple[float, ...]) -> None:
        self._sigmas = sigmas
        self._step = 0

    def step_for(self, sigma: float) -> int:
        index = step_index_for_sigma(self._sigmas, sigma)
        if index is not None:
            self._step = index
        return self._step


def _window_input(
    x: torch.Tensor, spec: ContextWindowsSpec, indices: tuple[int, ...]
) -> tuple[torch.Tensor, bool]:
    """The model input for one window and whether an anchor frame was
    prepended (the frame before a non-initial causal window).

    Latent retain indices overwrite the window input positionally after
    any anchor prepend: local position ``r`` becomes full-tensor index
    ``r``, the reference's retain_index_list semantics."""

    anchored = spec.causal_anchor and indices[0] > 0
    input_indices = (indices[0] - 1, *indices) if anchored else indices
    index_tensor = torch.tensor(input_indices, dtype=torch.long, device=x.device)
    window = x.index_select(spec.dim, index_tensor)
    if spec.latent_retain_indices:
        limit = len(input_indices)
        for position in spec.latent_retain_indices:
            if position >= limit:
                raise ContextWindowsError(
                    f"latent retain index {position} exceeds a {limit}-frame window input"
                )
        retain_tensor = torch.tensor(spec.latent_retain_indices, dtype=torch.long, device=x.device)
        window.index_copy_(spec.dim, retain_tensor, x.index_select(spec.dim, retain_tensor))
    return window, anchored


def _strip_anchor(outputs: tuple[torch.Tensor, ...], dim: int) -> tuple[torch.Tensor, ...]:
    return tuple(value.narrow(dim, 1, value.size(dim) - 1) for value in outputs)


def _full_length_zeros(window_output: torch.Tensor, dim: int, num_frames: int) -> torch.Tensor:
    shape = list(window_output.shape)
    shape[dim] = num_frames
    return torch.zeros(shape, dtype=window_output.dtype, device=window_output.device)


def _accumulate_fused(
    x: torch.Tensor,
    spec: ContextWindowsSpec,
    step: int,
    run: _WindowRun,
) -> tuple[torch.Tensor, ...]:
    """Accumulate-normalize fusing: out[i] = sum(w * window_out[i]) / sum(w)
    over every window occurrence of frame i, the reference's non-relative
    combine realized through the compiled composite window plan."""

    dim = spec.dim
    num_frames = x.size(dim)
    plan = context_window_plan(spec, num_frames, step)
    accumulators: list[torch.Tensor] | None = None
    counts: list[torch.Tensor] | None = None
    for window in plan.joint_windows:
        indices = window.axis_indices[0][1]
        window_input, anchored = _window_input(x, spec, indices)
        input_indices = (indices[0] - 1, *indices) if anchored else indices
        outputs = run(window_input, input_indices)
        if anchored:
            outputs = _strip_anchor(outputs, dim)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=x.device)
        weights = [occurrence.weight for occurrence in window.occurrences]
        if accumulators is None or counts is None:
            accumulators = [_full_length_zeros(value, dim, num_frames) for value in outputs]
            counts = [
                torch.zeros(
                    [1] * dim + [num_frames] + [1] * (value.ndim - dim - 1),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value in outputs
            ]
        for lane, value in enumerate(outputs):
            weight = torch.tensor(weights, dtype=value.dtype, device=value.device)
            broadcast = weight.reshape([1] * dim + [len(weights)] + [1] * (value.ndim - dim - 1))
            accumulators[lane].index_add_(dim, index_tensor, value * broadcast)
            counts[lane].index_add_(dim, index_tensor, broadcast)
    if accumulators is None or counts is None:
        raise ContextWindowsError("context windows produced no windows to evaluate")
    return tuple(
        accumulator / count for accumulator, count in zip(accumulators, counts, strict=True)
    )


def _relative_fused(
    x: torch.Tensor,
    spec: ContextWindowsSpec,
    step: int,
    run: _WindowRun,
) -> tuple[torch.Tensor, ...]:
    """Relative fusing: windows blend sequentially per frame index, each
    weighted by its bias against the biases already blended there. The
    first blend of an index has zero prior bias, so the zero-initialized
    accumulator never contributes."""

    dim = spec.dim
    num_frames = x.size(dim)
    windows = plan_windows(spec, num_frames, step)
    accumulators: list[torch.Tensor] | None = None
    bias_totals = [0.0] * num_frames
    for indices in windows:
        window_input, anchored = _window_input(x, spec, indices)
        input_indices = (indices[0] - 1, *indices) if anchored else indices
        outputs = run(window_input, input_indices)
        if anchored:
            outputs = _strip_anchor(outputs, dim)
        if accumulators is None:
            accumulators = [_full_length_zeros(value, dim, num_frames) for value in outputs]
        for position, index in enumerate(indices):
            bias = relative_bias(index, indices[0], indices[-1])
            total = bias_totals[index]
            previous_weight = total / (total + bias)
            new_weight = bias / (total + bias)
            index_slice = (slice(None),) * dim + (index,)
            position_slice = (slice(None),) * dim + (position,)
            for lane, value in enumerate(outputs):
                accumulators[lane][index_slice] = (
                    accumulators[lane][index_slice] * previous_weight
                    + value[position_slice] * new_weight
                )
            bias_totals[index] = total + bias
    if accumulators is None:
        raise ContextWindowsError("context windows produced no windows to evaluate")
    return tuple(accumulators)


def windowed_conditioning_evaluation(
    inner: ConditioningEvaluation[PreparedCondition],
    spec: ContextWindowsSpec,
    sigmas: tuple[float, ...],
) -> ConditioningEvaluation[PreparedCondition]:
    """Wrap ``inner`` so every model application evaluates per context
    window and fuses back to full length.

    Both the single-condition and the batched evaluation paths window, so
    evaluate-only families are covered. Uniform schedules re-plan per
    schedule step; an off-schedule sigma keeps the prior step's windows.
    The wrapper forwards preparation, identity, layout, and batching
    declarations unchanged and declares no attention-level evaluation."""

    tracker = _StepTracker(sigmas)

    def fused(x: torch.Tensor, sigma: float, run: _WindowRun) -> tuple[torch.Tensor, ...]:
        if spec.dim >= x.ndim:
            raise ContextWindowsError(
                f"window dim {spec.dim} is out of range for a {x.ndim}-d latent"
            )
        step = tracker.step_for(sigma)
        if spec.fuse_method is ContextFuseMethod.RELATIVE:
            return _relative_fused(x, spec, step, run)
        return _accumulate_fused(x, spec, step, run)

    def evaluate(x: torch.Tensor, sigma: float, condition: PreparedCondition) -> torch.Tensor:
        shape = tuple(x.shape)

        def run(window: torch.Tensor, indices: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
            prepared = (
                condition
                if inner.window_conditioning is None
                else inner.window_conditioning(condition, spec.dim, indices, shape)
            )
            return (inner.evaluate(window, sigma, prepared),)

        (value,) = fused(x, sigma, run)
        return value

    inner_batch = inner.evaluate_batch
    evaluate_batch: (
        Callable[[torch.Tensor, float, tuple[PreparedCondition, ...]], tuple[torch.Tensor, ...]]
        | None
    ) = None
    if inner_batch is not None:

        def windowed_batch(
            x: torch.Tensor,
            sigma: float,
            conditions: tuple[PreparedCondition, ...],
        ) -> tuple[torch.Tensor, ...]:
            shape = tuple(x.shape)

            def run(window: torch.Tensor, indices: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
                prepared = (
                    conditions
                    if inner.window_conditioning is None
                    else tuple(
                        inner.window_conditioning(condition, spec.dim, indices, shape)
                        for condition in conditions
                    )
                )
                return inner_batch(window, sigma, prepared)

            return fused(x, sigma, run)

        evaluate_batch = windowed_batch

    return replace(
        inner,
        evaluate=evaluate,
        evaluate_batch=evaluate_batch,
        evaluate_batch_attention=None,
    )
