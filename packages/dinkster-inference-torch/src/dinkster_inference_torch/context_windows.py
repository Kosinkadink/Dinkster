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
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeVar

import torch
from dinkster_inference import LatentPackLayout, LatentStream, MultiStreamLatent
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
from .latent_streams import pack_latent_streams, unpack_latent_streams

PreparedCondition = TypeVar("PreparedCondition")

_WindowRun = Callable[[torch.Tensor, tuple[int, ...]], tuple[torch.Tensor, ...]]

__all__ = [
    "PackedContextWindowAxis",
    "PackedContextWindowScale",
    "PackedContextWindowStream",
    "PackedContextWindows",
    "apply_freenoise",
    "windowed_conditioning_evaluation",
]


class PackedContextWindowScale(Enum):
    """How one packed stream axis follows the primary window axis."""

    IDENTITY = "identity"
    PROPORTIONAL = "proportional"


@dataclass(frozen=True, slots=True)
class PackedContextWindowStream:
    """One role and tensor axis sliced by a logical context-window axis."""

    role: str
    tensor_dim: int
    scale: PackedContextWindowScale = PackedContextWindowScale.IDENTITY

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role:
            raise ValueError("packed context-window roles must be non-empty")
        if type(self.tensor_dim) is not int or self.tensor_dim < 1:
            raise ValueError("packed context-window tensor dims must be positive integers")
        if type(self.scale) is not PackedContextWindowScale:
            raise TypeError("packed context-window scales must be exact enum values")


@dataclass(frozen=True, slots=True)
class PackedContextWindowAxis:
    """A logical window axis and the packed stream axes that follow it.

    The first stream is the primary axis whose extent is supplied to the
    shared planner. Roles omitted from ``streams`` remain whole in every
    window and are averaged across overlapping evaluations.
    """

    dim: int
    streams: tuple[PackedContextWindowStream, ...]

    def __post_init__(self) -> None:
        if type(self.dim) is not int or self.dim < 0:
            raise ValueError("packed context-window logical dims must be nonnegative integers")
        if type(self.streams) is not tuple or not self.streams:
            raise ValueError("packed context-window axes require at least one stream")
        if any(type(stream) is not PackedContextWindowStream for stream in self.streams):
            raise TypeError("packed context-window streams must be exact declarations")
        roles = tuple(stream.role for stream in self.streams)
        if len(set(roles)) != len(roles):
            raise ValueError("a packed role may appear only once per logical axis")
        if self.streams[0].scale is not PackedContextWindowScale.IDENTITY:
            raise ValueError("the primary packed context-window stream must use identity scale")


@dataclass(frozen=True, slots=True)
class _PackedWindowSelection:
    packed: torch.Tensor
    layout: LatentPackLayout
    indices: tuple[tuple[str, int, tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class PackedContextWindows:
    """Family-declared packed layout consumed by the shared window engine."""

    layout: LatentPackLayout
    axes: tuple[PackedContextWindowAxis, ...]

    def __post_init__(self) -> None:
        if type(self.layout) is not LatentPackLayout:
            raise TypeError("packed context windows require an exact LatentPackLayout")
        if type(self.axes) is not tuple or not self.axes:
            raise ValueError("packed context windows require at least one logical axis")
        if any(type(axis) is not PackedContextWindowAxis for axis in self.axes):
            raise TypeError("packed context-window axes must be exact declarations")
        dims = tuple(axis.dim for axis in self.axes)
        if len(set(dims)) != len(dims):
            raise ValueError("packed context-window logical dims must be unique")
        for axis in self.axes:
            for stream in axis.streams:
                layout = self.layout.by_role(stream.role)
                if stream.tensor_dim >= len(layout.shape):
                    raise ValueError(
                        f"packed context-window dim {stream.tensor_dim} is out of range "
                        f"for role {stream.role!r} shape {layout.shape}"
                    )
            primary = axis.streams[0]
            primary_extent = self.layout.by_role(primary.role).shape[primary.tensor_dim]
            for stream in axis.streams[1:]:
                extent = self.layout.by_role(stream.role).shape[stream.tensor_dim]
                if (
                    stream.scale is PackedContextWindowScale.IDENTITY
                    and extent != primary_extent
                ):
                    raise ValueError("identity-scaled packed window axes must have equal extents")
                if (
                    stream.scale is PackedContextWindowScale.PROPORTIONAL
                    and extent < primary_extent
                ):
                    raise ValueError(
                        "proportional packed window axes must not be shorter than the primary axis"
                    )

    def axis(self, dim: int) -> PackedContextWindowAxis:
        for axis in self.axes:
            if axis.dim == dim:
                return axis
        raise ContextWindowsError(f"packed latent does not declare logical window dim {dim}")

    def axis_size(self, dim: int) -> int:
        primary = self.axis(dim).streams[0]
        return self.layout.by_role(primary.role).shape[primary.tensor_dim]

    @staticmethod
    def _mapped_indices(
        source_indices: tuple[int, ...],
        source_extent: int,
        target_extent: int,
        scale: PackedContextWindowScale,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if scale is PackedContextWindowScale.IDENTITY:
            return source_indices, tuple(range(len(source_indices)))
        target: list[int] = []
        parents: list[int] = []
        for position, index in enumerate(source_indices):
            start = (index * target_extent + source_extent // 2) // source_extent
            stop = ((index + 1) * target_extent + source_extent // 2) // source_extent
            target.extend(range(start, stop))
            parents.extend((position,) * (stop - start))
        return tuple(target), tuple(parents)

    def select(
        self, packed: torch.Tensor, dim: int, source_indices: tuple[int, ...]
    ) -> _PackedWindowSelection:
        if tuple(packed.shape) != self.layout.packed_shape:
            raise ContextWindowsError(
                f"packed context-window input shape must be {self.layout.packed_shape}"
            )
        axis = self.axis(dim)
        primary = axis.streams[0]
        source_extent = self.layout.by_role(primary.role).shape[primary.tensor_dim]
        declarations = {stream.role: stream for stream in axis.streams}
        selected: list[LatentStream[torch.Tensor]] = []
        mappings: list[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = []
        for stream in unpack_latent_streams(packed, self.layout).streams:
            declaration = declarations.get(stream.role)
            if declaration is None:
                selected.append(stream)
                continue
            target_extent = stream.payload.shape[declaration.tensor_dim]
            indices, parents = self._mapped_indices(
                source_indices,
                source_extent,
                target_extent,
                declaration.scale,
            )
            if not indices:
                raise ContextWindowsError(
                    f"window maps to no indices on packed role {stream.role!r}"
                )
            index = torch.tensor(indices, dtype=torch.long, device=packed.device)
            selected.append(
                LatentStream(
                    stream.role,
                    stream.payload.index_select(declaration.tensor_dim, index),
                )
            )
            mappings.append((stream.role, declaration.tensor_dim, indices, parents))
        window, layout = pack_latent_streams(MultiStreamLatent(tuple(selected)))
        return _PackedWindowSelection(window, layout, tuple(mappings))


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


def _packed_fused(
    x: torch.Tensor,
    spec: ContextWindowsSpec,
    step: int,
    run: _WindowRun,
    packed: PackedContextWindows,
) -> tuple[torch.Tensor, ...]:
    if spec.causal_anchor or spec.latent_retain_indices:
        raise ContextWindowsError(
            "packed context windows do not support causal anchors or latent retain indices"
        )
    num_frames = packed.axis_size(spec.dim)
    if spec.fuse_method is ContextFuseMethod.RELATIVE:
        weighted_windows = tuple(
            (
                indices,
                tuple(relative_bias(index, indices[0], indices[-1]) for index in indices),
            )
            for indices in plan_windows(spec, num_frames, step)
        )
    else:
        plan = context_window_plan(spec, num_frames, step)
        weighted_windows = tuple(
            (
                window.axis_indices[0][1],
                tuple(occurrence.weight for occurrence in window.occurrences),
            )
            for window in plan.joint_windows
        )
    accumulators: list[dict[str, torch.Tensor]] | None = None
    counts: list[dict[str, torch.Tensor]] | None = None
    for indices, source_weights in weighted_windows:
        selection = packed.select(x, spec.dim, indices)
        outputs = run(selection.packed, indices)
        mappings = {
            role: (tensor_dim, target_indices, parents)
            for role, tensor_dim, target_indices, parents in selection.indices
        }
        unpacked = tuple(unpack_latent_streams(output, selection.layout) for output in outputs)
        if accumulators is None or counts is None:
            accumulators = [
                {
                    stream.role: torch.zeros(
                        packed.layout.by_role(stream.role).shape,
                        dtype=stream.payload.dtype,
                        device=stream.payload.device,
                    )
                    for stream in output.streams
                }
                for output in unpacked
            ]
            counts = [
                {
                    stream.role: torch.zeros(
                        [1] * len(packed.layout.by_role(stream.role).shape),
                        dtype=stream.payload.dtype,
                        device=stream.payload.device,
                    )
                    for stream in output.streams
                }
                for output in unpacked
            ]
        for lane, output in enumerate(unpacked):
            for stream in output.streams:
                mapping = mappings.get(stream.role)
                if mapping is None:
                    weight = torch.tensor(
                        sum(source_weights) / len(source_weights),
                        dtype=stream.payload.dtype,
                        device=stream.payload.device,
                    )
                    accumulators[lane][stream.role].add_(stream.payload * weight)
                    counts[lane][stream.role].add_(weight)
                    continue
                tensor_dim, target_indices, parents = mapping
                target = torch.tensor(
                    target_indices, dtype=torch.long, device=stream.payload.device
                )
                weights = torch.tensor(
                    [source_weights[parent] for parent in parents],
                    dtype=stream.payload.dtype,
                    device=stream.payload.device,
                )
                shape = [1] * stream.payload.ndim
                shape[tensor_dim] = len(weights)
                broadcast = weights.reshape(shape)
                accumulators[lane][stream.role].index_add_(
                    tensor_dim, target, stream.payload * broadcast
                )
                count = counts[lane][stream.role]
                if count.shape[tensor_dim] == 1:
                    count_shape = list(count.shape)
                    count_shape[tensor_dim] = packed.layout.by_role(stream.role).shape[tensor_dim]
                    count = torch.zeros(
                        count_shape,
                        dtype=stream.payload.dtype,
                        device=stream.payload.device,
                    )
                    counts[lane][stream.role] = count
                count.index_add_(tensor_dim, target, broadcast)
    if accumulators is None or counts is None:
        raise ContextWindowsError("context windows produced no windows to evaluate")
    fused: list[torch.Tensor] = []
    for lane_accumulators, lane_counts in zip(accumulators, counts, strict=True):
        streams: list[LatentStream[torch.Tensor]] = []
        for layout in packed.layout.streams:
            count = lane_counts[layout.role]
            if bool(torch.any(count == 0)):
                raise ContextWindowsError(
                    f"context windows did not cover packed role {layout.role!r}"
                )
            streams.append(
                LatentStream(layout.role, lane_accumulators[layout.role] / count)
            )
        value, layout = pack_latent_streams(MultiStreamLatent(tuple(streams)))
        if layout != packed.layout:
            raise ContextWindowsError("fused packed context-window layout changed")
        fused.append(value)
    return tuple(fused)


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
    packed: PackedContextWindows | None = None,
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
        if packed is not None:
            if tuple(x.shape) != packed.layout.packed_shape:
                raise ContextWindowsError(
                    f"packed context-window input shape must be {packed.layout.packed_shape}"
                )
            return _packed_fused(x, spec, tracker.step_for(sigma), run, packed)
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
