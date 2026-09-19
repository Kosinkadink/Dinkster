"""Flux realization of compiled spatial window plans."""

from __future__ import annotations

import hashlib

import pytest
import torch
from dinkster_inference import (
    AccumulationDType,
    Conditioning,
    FluxConfig,
    IntegerAffineIndexMap,
    KindAxisMap,
    LayerWindow,
    MediaAxis,
    MergeDeclaration,
    WindowIndexList,
    WindowKind,
    WindowPlanBinding,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    build_windowed_evaluation_slot,
    compile_window_plan,
    merge_window_outputs,
)
from dinkster_inference_torch import Flux
from dinkster_inference_torch._conditioning_layout import (
    bind_flux_layout,
    declare_text_conditioning,
)
from dinkster_inference_torch.flux_window import (
    FluxWindowConditioningEvaluation,
    FluxWindowError,
    derive_flux_window_layout,
    flux_window_position_ids,
    merge_flux_window_outputs,
    prepare_flux_window_plan,
)
from dinkster_inference_torch.guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)


class _IdentityEvaluator:
    @staticmethod
    def prepare_conditioning(
        conditioning: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(conditioning, Conditioning)
        return conditioning.embeddings, conditioning.pooled

    @staticmethod
    def batchable(
        conditions: tuple[tuple[torch.Tensor, torch.Tensor | None], ...],
    ) -> bool:
        return bool(conditions) and all(
            condition[0].shape[1] == conditions[0][0].shape[1] for condition in conditions
        )

    @staticmethod
    def evaluate_conditioning(
        x: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        del sigma, condition
        return x

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    @staticmethod
    def _evaluate_conditioning_batch(
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[tuple[torch.Tensor, torch.Tensor | None], ...],
    ) -> tuple[torch.Tensor, ...]:
        del sigma
        return tuple(x for _ in conditions)


def _kind_declarations(
    axes: tuple[MediaAxis, ...],
) -> tuple[WindowKind, WindowKind]:
    names = tuple(axis.name for axis in axes)
    return (
        WindowKind(
            "latent_image",
            tuple(KindAxisMap(axis.name, axis.extent, IntegerAffineIndexMap(1)) for axis in axes),
        ),
        WindowKind("text", invariant_axes=names),
    )


def _plan(
    accumulation_dtype: AccumulationDType = AccumulationDType.FLOAT64,
    *,
    weighted: bool = False,
):
    axes = (
        MediaAxis("height", 2, wrappable=True),
        MediaAxis("width", 3, wrappable=True),
    )
    layer = WindowPlanLayer(
        ("height", "width"),
        (
            LayerWindow(
                (
                    WindowIndexList((0, 1)),
                    WindowIndexList((2, 3, 5), modular=True),
                )
            ),
            LayerWindow((WindowIndexList((0, 1)), WindowIndexList((1,)))),
        ),
        (
            WindowWeightProfile(WindowWeightKind.FLAT),
            WindowWeightProfile(
                WindowWeightKind.OVERLAP_LINEAR if weighted else WindowWeightKind.FLAT,
                overlap=1 if weighted else 0,
            ),
        ),
        MergeDeclaration(accumulation_dtype),
    )
    return compile_window_plan(
        axes=axes,
        kinds=_kind_declarations(axes),
        layers=(layer,),
    )


def test_window_layout_is_declared_from_global_geometry_and_local_extents() -> None:
    layout, transforms = derive_flux_window_layout(
        text_token_count=5,
        latent_height=4,
        latent_width=6,
        patch_size=2,
        height_indices=(1, 0),
        width_indices=(2, 0),
    )

    assert tuple(segment.identity for segment in layout.segments) == (
        "text",
        "latent_image",
    )
    assert layout.segments[0].grid == (5,)
    assert layout.segments[1].grid == (2, 2)
    assert transforms[1].transform == "flux.latent-window-crop.v1"
    assert transforms[1].source_geometry == (4, 6)


def test_plan_accepts_independently_wrapped_noncontiguous_packed_axes() -> None:
    prepared = prepare_flux_window_plan(
        _plan(),
        latent_height=4,
        latent_width=6,
        patch_size=2,
    )

    assert prepared.windows[0].height_indices == (0, 1)
    assert prepared.windows[0].width_indices == (2, 0, 2)
    assert prepared.windows[1].width_indices == (1,)


def test_wrapped_duplicate_gather_and_merge_executes_per_occurrence() -> None:
    prepared_plan = prepare_flux_window_plan(
        _plan(),
        latent_height=4,
        latent_width=6,
        patch_size=2,
    )
    evaluation = FluxWindowConditioningEvaluation(
        prepared_plan,
        tuple(_IdentityEvaluator() for _ in prepared_plan.windows),
    )
    source = declare_text_conditioning(Conditioning(torch.zeros(1, 5, 3)), 5)
    condition = evaluation.prepare_conditioning(
        bind_flux_layout(source, latent_height=4, latent_width=6, patch_size=2)
    )
    value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)

    actual = evaluation.evaluate_conditioning(value, 1.0, condition)

    assert torch.equal(actual, value)


def test_flux_accepts_only_exact_declared_global_position_ids() -> None:
    model = Flux(
        FluxConfig(
            in_channels=16,
            out_channels=16,
            vec_in_dim=8,
            context_in_dim=8,
            hidden_size=32,
            depth=1,
            depth_single_blocks=1,
            num_heads=2,
            axes_dim=(4, 6, 6),
        )
    )
    value = torch.zeros(2, 16, 4, 4)
    position_ids = flux_window_position_ids(
        height_indices=(1, 0),
        width_indices=(2, 0),
        batch=2,
        device=value.device,
    )

    _, accepted = model._patchify(value, position_ids)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(accepted, position_ids)
    assert accepted.tolist() == [
        [[0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
    ]

    wrong_axis = position_ids.clone()
    wrong_axis[0, 1, 1] = 0.0
    wrong_batch = position_ids.clone()
    wrong_batch[1, :, 2] += 1.0
    for invalid in (
        position_ids.to(torch.float64),
        position_ids[:, :-1],
        position_ids.index_fill(2, torch.tensor([0]), 1.0),
        position_ids.index_fill(2, torch.tensor([2]), torch.inf),
        wrong_axis,
        wrong_batch,
    ):
        with pytest.raises(ValueError, match="image_position_ids"):
            model._patchify(value, invalid)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "accumulation_dtype",
    (AccumulationDType.FLOAT32, AccumulationDType.FLOAT64),
)
def test_tensor_merge_is_element_exact_with_scalar_reference(
    accumulation_dtype: AccumulationDType,
) -> None:
    plan = _plan(accumulation_dtype, weighted=True)
    assert any(
        occurrence.weight != 1.0
        for window in plan.joint_windows
        for occurrence in window.occurrences
    )
    prepared = prepare_flux_window_plan(
        plan,
        latent_height=4,
        latent_width=6,
        patch_size=2,
    )
    outputs = tuple(
        torch.arange(
            len(window.height_indices) * 2 * len(window.width_indices) * 2,
            dtype=torch.float32,
        ).reshape(1, 1, len(window.height_indices) * 2, len(window.width_indices) * 2)
        / (window.declaration.index + 3)
        for window in prepared.windows
    )

    actual = merge_flux_window_outputs(prepared, outputs)
    expected = torch.empty_like(actual)
    axis_order = prepared.windows[0].occurrence_axis_order
    for height_offset in range(2):
        for width_offset in range(2):
            scalar_outputs: list[tuple[float, ...]] = []
            for window, output in zip(prepared.windows, outputs, strict=True):
                values = []
                for occurrence in window.declaration.occurrences:
                    local = dict(zip(axis_order, occurrence.local_positions, strict=True))
                    values.append(
                        float(
                            output[
                                0,
                                0,
                                local["height"] * 2 + height_offset,
                                local["width"] * 2 + width_offset,
                            ]
                        )
                    )
                scalar_outputs.append(tuple(values))
            merged = merge_window_outputs(plan, tuple(scalar_outputs))
            for coordinate, value in zip(plan.coordinates, merged, strict=True):
                height, width = coordinate
                expected[0, 0, height * 2 + height_offset, width * 2 + width_offset] = value
    assert torch.equal(actual, expected)


def test_manifest_binds_digests_from_the_execution_layout_deriver() -> None:
    plan = _plan()
    prepared_plan = prepare_flux_window_plan(
        plan,
        latent_height=4,
        latent_width=6,
        patch_size=2,
    )
    evaluation = FluxWindowConditioningEvaluation(
        prepared_plan,
        tuple(_IdentityEvaluator() for _ in prepared_plan.windows),
    )
    source = declare_text_conditioning(
        Conditioning(torch.zeros(1, 5, 3)),
        5,
    )
    full = bind_flux_layout(
        source,
        latent_height=4,
        latent_width=6,
        patch_size=2,
    )
    executed = evaluation.prepare_conditioning(full)
    assert executed.full_layout is not None
    evaluation.validate_layout(executed, executed.full_layout)
    assert all(layout is not None for layout in executed.window_layouts)
    assert all(
        tuple(transform.transform for transform in transforms)
        == ("flux.text-context-identity.v1", "flux.latent-window-crop.v1")
        for transforms in executed.window_transforms
    )
    executed_layout_digests = tuple(
        layout.digest for layout in executed.window_layouts if layout is not None
    )
    manifest_layout_digests = tuple(
        derive_flux_window_layout(
            text_token_count=5,
            latent_height=4,
            latent_width=6,
            patch_size=2,
            height_indices=window.height_indices,
            width_indices=window.width_indices,
        )[0].digest
        for window in prepared_plan.windows
    )
    assert executed_layout_digests == manifest_layout_digests
    placeholder = tuple(
        hashlib.sha256(f"placeholder:{index}".encode()).hexdigest()
        for index in range(len(plan.joint_windows))
    )
    slot = build_windowed_evaluation_slot(
        derivation_identity="static-flux-window-plan.v1",
        derivation_facts_digest=hashlib.sha256(plan.digest.encode()).hexdigest(),
        plan_bindings=(WindowPlanBinding(plan, manifest_layout_digests, placeholder, placeholder),),
    )

    for index, digest in enumerate(manifest_layout_digests):
        assert f"plan[0].window[{index}].token_layout={digest}" in slot.facts


@pytest.mark.parametrize(
    ("plan", "height", "width", "message"),
    (
        (_plan(), 5, 6, "non-integral-latent-packing-boundary"),
        (_plan(), 4, 8, "plan-latent-geometry-mismatch"),
    ),
)
def test_plan_geometry_refuses_before_execution(
    plan: object,
    height: int,
    width: int,
    message: str,
) -> None:
    with pytest.raises(FluxWindowError, match=message):
        prepare_flux_window_plan(
            plan,  # type: ignore[arg-type]
            latent_height=height,
            latent_width=width,
            patch_size=2,
        )


def test_plan_refuses_unmappable_axes_and_missing_kind_declarations() -> None:
    temporal = compile_window_plan(
        axes=(MediaAxis("temporal", 2),),
        layers=(
            WindowPlanLayer(
                ("temporal",),
                (LayerWindow((WindowIndexList((0, 1)),)),),
                (WindowWeightProfile(WindowWeightKind.FLAT),),
                MergeDeclaration(),
            ),
        ),
    )
    missing_kinds = compile_window_plan(
        axes=(MediaAxis("height", 2),),
        layers=(
            WindowPlanLayer(
                ("height",),
                (LayerWindow((WindowIndexList((0, 1)),)),),
                (WindowWeightProfile(WindowWeightKind.FLAT),),
                MergeDeclaration(),
            ),
        ),
    )

    for plan in (temporal, missing_kinds):
        with pytest.raises(
            FluxWindowError,
            match="axis-unmappable-to-flux-packed-image-grid",
        ):
            prepare_flux_window_plan(
                plan,
                latent_height=4,
                latent_width=6,
                patch_size=2,
            )
