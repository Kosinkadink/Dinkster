"""Declared Flux spatial-window geometry and its tensor realization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar, cast

import torch
from dinkster_inference import (
    AccumulationDType,
    CompositeWindowPlan,
    IntegerAffineIndexMap,
    JointWindow,
    ModelTokenLayout,
    ModelTokenSegment,
    TokenGridTransform,
    TokenLayoutError,
    map_transforms,
)

from ._conditioning_layout import (
    DeclaredConditioning,
    conditioning_layout,
    conditioning_token_transforms,
    validate_flux_layout,
)
from .guidance import evaluate_conditioning_batch as _engine_evaluate_conditioning_batch

PreparedConditionT = TypeVar("PreparedConditionT")


class FluxWindowInnerEvaluator(Protocol[PreparedConditionT]):
    def prepare_conditioning(self, conditioning: object) -> PreparedConditionT: ...

    def batchable(self, conditions: tuple[PreparedConditionT, ...]) -> bool: ...

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: PreparedConditionT,
    ) -> torch.Tensor: ...

    evaluate_conditioning_batch: Callable[
        [torch.Tensor, float, tuple[PreparedConditionT, ...], object | None],
        tuple[torch.Tensor, ...],
    ]


class FluxWindowError(ValueError):
    pass


def _refuse(code: str, detail: str) -> FluxWindowError:
    return FluxWindowError(f"{code}: {detail}")


def derive_flux_window_layout(
    *,
    text_token_count: int,
    latent_height: int,
    latent_width: int,
    patch_size: int,
    height_indices: tuple[int, ...],
    width_indices: tuple[int, ...],
) -> tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]]:
    """Derive one inner-call declaration without inspecting tensors."""

    if type(text_token_count) is not int or text_token_count < 1:
        raise TokenLayoutError("Flux text token count must be an exact int >= 1")
    if type(patch_size) is not int or patch_size != 2:
        raise TokenLayoutError("Flux window layout requires the declared 2x2 latent pack")
    if (
        type(latent_height) is not int
        or type(latent_width) is not int
        or latent_height < 1
        or latent_width < 1
        or latent_height % patch_size
        or latent_width % patch_size
    ):
        raise TokenLayoutError("Flux window layout requires integral latent packing boundaries")
    for name, indices, extent in (
        ("height", height_indices, latent_height // patch_size),
        ("width", width_indices, latent_width // patch_size),
    ):
        if (
            type(indices) is not tuple
            or not indices
            or any(type(index) is not int or not 0 <= index < extent for index in indices)
        ):
            raise TokenLayoutError(
                f"Flux window {name} indices must be declared packed-grid coordinates"
            )
    image_grid = (len(height_indices), len(width_indices))
    image_tokens = image_grid[0] * image_grid[1]
    layout = ModelTokenLayout(
        (
            ModelTokenSegment(
                "text",
                "text",
                "context",
                0,
                text_token_count,
                (text_token_count,),
            ),
            ModelTokenSegment(
                "latent_image",
                "image",
                "latent",
                text_token_count,
                text_token_count + image_tokens,
                image_grid,
            ),
        ),
        0,
    )
    return (
        layout,
        (
            TokenGridTransform(
                "flux.text-context-identity.v1",
                "text",
                "text",
                (text_token_count,),
                None,
            ),
            TokenGridTransform(
                "flux.latent-window-crop.v1",
                "image",
                "latent_image",
                (latent_height, latent_width),
                None,
            ),
        ),
    )


def bind_flux_window_layout(
    conditioning: object,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: int,
    height_indices: tuple[int, ...],
    width_indices: tuple[int, ...],
) -> object:
    """Bind an encoder-declared text payload to one declared window."""

    if type(conditioning) is not DeclaredConditioning:
        return conditioning
    layout, transforms = derive_flux_window_layout(
        text_token_count=conditioning.source_token_count,
        latent_height=latent_height,
        latent_width=latent_width,
        patch_size=patch_size,
        height_indices=height_indices,
        width_indices=width_indices,
    )
    return replace(
        conditioning,
        model_token_layout=layout,
        token_transforms=transforms,
    )


@dataclass(frozen=True, slots=True)
class PreparedFluxWindow:
    declaration: JointWindow
    height_indices: tuple[int, ...]
    width_indices: tuple[int, ...]
    occurrence_axis_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedFluxWindowPlan:
    declaration: CompositeWindowPlan
    latent_height: int
    latent_width: int
    patch_size: int
    windows: tuple[PreparedFluxWindow, ...]


def prepare_flux_window_plan(
    plan: CompositeWindowPlan,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: int,
) -> PreparedFluxWindowPlan:
    """Validate one compiled plan against Flux's declared packed geometry."""

    if type(plan) is not CompositeWindowPlan:
        raise _refuse("invalid-window-plan", "expected an exact CompositeWindowPlan")
    if (
        type(patch_size) is not int
        or patch_size != 2
        or latent_height % patch_size
        or latent_width % patch_size
    ):
        raise _refuse(
            "non-integral-latent-packing-boundary",
            "Flux windows require the declared 2x2 pack to divide the latent geometry",
        )
    packed_extents = {
        "height": latent_height // patch_size,
        "width": latent_width // patch_size,
    }
    axis_names = tuple(axis.name for axis in plan.axes)
    if not axis_names or any(axis not in packed_extents for axis in axis_names):
        raise _refuse(
            "axis-unmappable-to-flux-packed-image-grid",
            f"Flux window axes must be a non-empty subset of {tuple(packed_extents)!r}",
        )
    for axis in plan.axes:
        if axis.extent != packed_extents[axis.name]:
            raise _refuse(
                "plan-latent-geometry-mismatch",
                f"axis {axis.name!r} extent {axis.extent} does not match"
                f" packed latent extent {packed_extents[axis.name]}",
            )
    kinds = {kind.name: kind for kind in plan.kinds}
    if set(kinds) != {"latent_image", "text"}:
        raise _refuse(
            "axis-unmappable-to-flux-packed-image-grid",
            "Flux plans require exactly latent_image and text kind declarations",
        )
    text_kind = kinds["text"]
    if text_kind.axis_maps or text_kind.invariant_axes != axis_names:
        raise _refuse(
            "axis-unmappable-to-flux-packed-image-grid",
            "the text kind must be invariant on every sliced axis",
        )
    image_kind = kinds["latent_image"]
    image_maps = {mapping.axis: mapping for mapping in image_kind.axis_maps}
    if image_kind.invariant_axes or set(image_maps) != set(axis_names):
        raise _refuse(
            "axis-unmappable-to-flux-packed-image-grid",
            "latent_image must map every sliced axis",
        )
    for axis in axis_names:
        mapping = image_maps[axis]
        if (
            mapping.extent != packed_extents[axis]
            or type(mapping.profile) is not IntegerAffineIndexMap
            or mapping.profile.scale != 1
            or mapping.profile.offset != 0
        ):
            raise _refuse(
                "axis-unmappable-to-flux-packed-image-grid",
                f"latent_image axis {axis!r} must use the identity packed-grid mapping",
            )
    if plan.passthrough_coordinates:
        raise _refuse(
            "unsupported-structural-window-rows",
            "classic Flux windows do not carry passthrough rows",
        )

    occurrence_axis_order = tuple(axis for layer in plan.layers for axis in layer.axes)
    full_indices = {axis: tuple(range(extent)) for axis, extent in packed_extents.items()}
    windows: list[PreparedFluxWindow] = []
    for window in plan.joint_windows:
        mapped_kinds = {indices.kind: dict(indices.axes) for indices in window.kind_indices}
        image_indices = mapped_kinds.get("latent_image")
        if image_indices is None or set(image_indices) != set(axis_names):
            raise _refuse(
                "axis-unmappable-to-flux-packed-image-grid",
                f"joint window {window.index} lacks the declared latent_image mapping",
            )
        declared_axes = dict(window.axis_indices)
        if any(image_indices[axis] != declared_axes[axis] for axis in axis_names):
            raise _refuse(
                "axis-unmappable-to-flux-packed-image-grid",
                f"joint window {window.index} does not preserve packed-grid coordinates",
            )
        windows.append(
            PreparedFluxWindow(
                window,
                image_indices.get("height", full_indices["height"]),
                image_indices.get("width", full_indices["width"]),
                occurrence_axis_order,
            )
        )
    return PreparedFluxWindowPlan(
        plan,
        latent_height,
        latent_width,
        patch_size,
        tuple(windows),
    )


def crop_flux_window(
    value: torch.Tensor,
    plan: PreparedFluxWindowPlan,
    window_index: int,
) -> torch.Tensor:
    """Gather one window in the exact declared packed-grid order."""

    if value.ndim != 4 or tuple(value.shape[-2:]) != (
        plan.latent_height,
        plan.latent_width,
    ):
        raise _refuse(
            "window-tensor-declaration-mismatch",
            "the materialized latent does not match the admitted window geometry",
        )
    window = plan.windows[window_index]
    patch = plan.patch_size
    height = tuple(
        index * patch + offset for index in window.height_indices for offset in range(patch)
    )
    width = tuple(
        index * patch + offset for index in window.width_indices for offset in range(patch)
    )
    return value.index_select(2, torch.tensor(height, device=value.device)).index_select(
        3, torch.tensor(width, device=value.device)
    )


def flux_window_position_ids(
    *,
    height_indices: tuple[int, ...],
    width_indices: tuple[int, ...],
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the declared global row-major Flux image coordinates."""

    coordinates = tuple(
        (0.0, float(height), float(width)) for height in height_indices for width in width_indices
    )
    return (
        torch.tensor(coordinates, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch, -1, -1)
    )


def _window_slices(
    plan: PreparedFluxWindowPlan,
    window: PreparedFluxWindow,
    local_positions: tuple[int, ...],
    coordinate: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    patch = plan.patch_size
    local_by_axis = dict(zip(window.occurrence_axis_order, local_positions, strict=True))
    coordinate_by_axis = dict(
        zip((axis.name for axis in plan.declaration.axes), coordinate, strict=True)
    )
    source = [slice(None), slice(None)]
    target = [slice(None), slice(None)]
    for axis, full_extent in (
        ("height", plan.latent_height),
        ("width", plan.latent_width),
    ):
        if axis in local_by_axis:
            source_start = local_by_axis[axis] * patch
            target_start = coordinate_by_axis[axis] * patch
            source.append(slice(source_start, source_start + patch))
            target.append(slice(target_start, target_start + patch))
        else:
            source.append(slice(0, full_extent))
            target.append(slice(0, full_extent))
    return tuple(source), tuple(target)


def _coordinate_target_slices(
    plan: PreparedFluxWindowPlan,
    coordinate: tuple[int, ...],
) -> tuple[slice, ...]:
    patch = plan.patch_size
    coordinate_by_axis = dict(
        zip((axis.name for axis in plan.declaration.axes), coordinate, strict=True)
    )
    target = [slice(None), slice(None)]
    for axis, full_extent in (
        ("height", plan.latent_height),
        ("width", plan.latent_width),
    ):
        if axis in coordinate_by_axis:
            start = coordinate_by_axis[axis] * patch
            target.append(slice(start, start + patch))
        else:
            target.append(slice(0, full_extent))
    return tuple(target)


def merge_flux_window_outputs(
    plan: PreparedFluxWindowPlan,
    outputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Tensor realization of the plan's accumulate-normalize merge."""

    if type(outputs) is not tuple or len(outputs) != len(plan.windows):
        raise FluxWindowError("window outputs must contain one tensor per joint window")
    first = outputs[0]
    if type(first) is not torch.Tensor:
        raise FluxWindowError("window outputs must be exact torch.Tensor values")
    accumulation_dtype = {
        AccumulationDType.FLOAT32: torch.float32,
        AccumulationDType.FLOAT64: torch.float64,
    }[plan.declaration.merge.accumulation_dtype]
    accumulator = torch.zeros(
        (first.shape[0], first.shape[1], plan.latent_height, plan.latent_width),
        dtype=accumulation_dtype,
        device=first.device,
    )
    denominator = torch.zeros(
        (1, 1, plan.latent_height, plan.latent_width),
        dtype=accumulation_dtype,
        device=first.device,
    )
    expected_device = first.device
    expected_dtype = first.dtype
    for window, output in zip(plan.windows, outputs, strict=True):
        expected_shape = (
            first.shape[0],
            first.shape[1],
            len(window.height_indices) * plan.patch_size,
            len(window.width_indices) * plan.patch_size,
        )
        if (
            type(output) is not torch.Tensor
            or tuple(output.shape) != expected_shape
            or output.dtype != expected_dtype
            or output.device != expected_device
        ):
            raise FluxWindowError(
                f"joint window {window.declaration.index} returned an incompatible tensor"
            )
        for occurrence in window.declaration.occurrences:
            source, target = _window_slices(
                plan,
                window,
                occurrence.local_positions,
                occurrence.coordinate,
            )
            accumulator[target].add_(output[source].to(accumulation_dtype) * occurrence.weight)
    for coordinate, weight in zip(
        plan.declaration.coordinates,
        plan.declaration.total_weights,
        strict=True,
    ):
        target = _coordinate_target_slices(plan, coordinate)
        denominator[target] = weight
    return (accumulator / denominator).to(expected_dtype)


@dataclass(frozen=True, slots=True)
class PreparedFluxWindowConditioning(Generic[PreparedConditionT]):
    full_layout: ModelTokenLayout | None
    windows: tuple[PreparedConditionT, ...]
    window_layouts: tuple[ModelTokenLayout | None, ...]
    window_transforms: tuple[tuple[TokenGridTransform, ...], ...]


class FluxWindowConditioningEvaluation(Generic[PreparedConditionT]):
    """Multiply family model calls below one full-geometry guidance execution."""

    def __init__(
        self,
        prepared_plan: PreparedFluxWindowPlan,
        evaluators: tuple[FluxWindowInnerEvaluator[PreparedConditionT], ...],
    ) -> None:
        if len(evaluators) != len(prepared_plan.windows):
            raise FluxWindowError("each joint window requires one conditioning evaluator")
        self.prepared_plan = prepared_plan
        self.evaluators = evaluators

    def prepare_conditioning(
        self,
        conditioning: object,
    ) -> PreparedFluxWindowConditioning[PreparedConditionT]:
        full_layout = conditioning_layout(conditioning)
        if full_layout is None:
            raise _refuse(
                "window-layout-declaration-required",
                "windowed execution requires declared conditioning with a bound model token"
                " layout; pre-encoded conditioning executes only without a window plan",
            )
        prepared: list[PreparedConditionT] = []
        layouts: list[ModelTokenLayout | None] = []
        transforms: list[tuple[TokenGridTransform, ...]] = []
        for window, evaluator in zip(
            self.prepared_plan.windows,
            self.evaluators,
            strict=True,
        ):
            bound = bind_flux_window_layout(
                conditioning,
                latent_height=self.prepared_plan.latent_height,
                latent_width=self.prepared_plan.latent_width,
                patch_size=self.prepared_plan.patch_size,
                height_indices=window.height_indices,
                width_indices=window.width_indices,
            )
            condition = evaluator.prepare_conditioning(bound)
            layout = conditioning_layout(bound)
            declared_transforms = conditioning_token_transforms(bound)
            if layout is None:
                if declared_transforms:
                    raise TokenLayoutError("Flux window transforms require a declared layout")
            else:
                declared_transforms = tuple(map_transforms(layout, declared_transforms).values())
            prepared.append(condition)
            layouts.append(layout)
            transforms.append(declared_transforms)
        return PreparedFluxWindowConditioning(
            full_layout,
            tuple(prepared),
            tuple(layouts),
            tuple(transforms),
        )

    def validate_layout(
        self,
        conditioning: PreparedFluxWindowConditioning[PreparedConditionT],
        layout: ModelTokenLayout,
    ) -> None:
        if conditioning.full_layout != layout:
            raise TokenLayoutError("Flux full-call layout changed during window preparation")
        validate_flux_layout(
            cast("tuple[torch.Tensor, torch.Tensor | None]", conditioning.windows[0]),
            layout,
            latent_height=self.prepared_plan.latent_height,
            latent_width=self.prepared_plan.latent_width,
            patch_size=self.prepared_plan.patch_size,
        )
        for window, condition, window_layout in zip(
            self.prepared_plan.windows,
            conditioning.windows,
            conditioning.window_layouts,
            strict=True,
        ):
            if window_layout is None:
                raise TokenLayoutError("Flux declared windows require per-window layouts")
            validate_flux_layout(
                cast("tuple[torch.Tensor, torch.Tensor | None]", condition),
                window_layout,
                latent_height=len(window.height_indices) * self.prepared_plan.patch_size,
                latent_width=len(window.width_indices) * self.prepared_plan.patch_size,
                patch_size=self.prepared_plan.patch_size,
            )

    @staticmethod
    def inner_calls(
        conditioning: PreparedFluxWindowConditioning[PreparedConditionT],
    ) -> tuple[tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]], ...]:
        if any(layout is None for layout in conditioning.window_layouts):
            raise TokenLayoutError("Flux declared windows require per-window layouts")
        return tuple(
            (cast("ModelTokenLayout", layout), transforms)
            for layout, transforms in zip(
                conditioning.window_layouts,
                conditioning.window_transforms,
                strict=True,
            )
        )

    def batchable(
        self,
        conditions: tuple[PreparedFluxWindowConditioning[PreparedConditionT], ...],
    ) -> bool:
        return bool(conditions) and all(
            evaluator.batchable(tuple(condition.windows[index] for condition in conditions))
            for index, evaluator in enumerate(self.evaluators)
        )

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: PreparedFluxWindowConditioning[PreparedConditionT],
    ) -> torch.Tensor:
        outputs = tuple(
            evaluator.evaluate_conditioning(
                crop_flux_window(x, self.prepared_plan, index),
                sigma,
                condition.windows[index],
            )
            for index, evaluator in enumerate(self.evaluators)
        )
        return merge_flux_window_outputs(self.prepared_plan, outputs)

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[PreparedFluxWindowConditioning[PreparedConditionT], ...],
    ) -> tuple[torch.Tensor, ...]:
        by_condition: list[list[torch.Tensor]] = [[] for _ in conditions]
        for index, evaluator in enumerate(self.evaluators):
            outputs = evaluator.evaluate_conditioning_batch(
                crop_flux_window(x, self.prepared_plan, index),
                sigma,
                tuple(condition.windows[index] for condition in conditions),
                None,
            )
            if len(outputs) != len(conditions):
                raise FluxWindowError(
                    f"joint window {index} returned the wrong number of lane predictions"
                )
            for condition_outputs, output in zip(by_condition, outputs, strict=True):
                condition_outputs.append(output)
        return tuple(
            merge_flux_window_outputs(self.prepared_plan, tuple(outputs))
            for outputs in by_condition
        )


__all__ = [
    "FluxWindowError",
    "FluxWindowConditioningEvaluation",
    "FluxWindowInnerEvaluator",
    "PreparedFluxWindow",
    "PreparedFluxWindowConditioning",
    "PreparedFluxWindowPlan",
    "bind_flux_window_layout",
    "crop_flux_window",
    "derive_flux_window_layout",
    "flux_window_position_ids",
    "merge_flux_window_outputs",
    "prepare_flux_window_plan",
]
