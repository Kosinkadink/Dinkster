"""Grouped regional and prepared condition-scale proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any, Never, cast

import pytest
import torch
from dinkster_inference import (
    EMPTY_RANGE,
    FLUX_DEV,
    FLUX_SCHNELL,
    SD15,
    SDXL,
    SDXL_REFINER,
    AdapterPatch,
    Conditioning,
    ConditioningRange,
    DiffPatch,
    FlowSigmas,
    GuidanceRole,
    Parameterization,
    PatchEntry,
    PatchSet,
    PercentRange,
)
from dinkster_inference_torch import (
    INITLESS,
    CastOperations,
    DeviceMemory,
    LoRAAdapter,
    MaterializedRegion,
    PreparedScaledPatches,
    RegionalConditioningError,
    enroll_component,
    evaluate_grouped_regions,
    flux_grouped_region_evaluator,
    prepare_grouped_patches,
    prepare_scaled_patches,
    regional_working_memory,
    sd_grouped_region_evaluator,
)
from dinkster_inference_torch import scaled_patches as scaled_module
from test_denoise import tiny_cond as flux_cond
from test_denoise import tiny_flux
from test_denoise import tiny_latent as flux_latent
from test_sd_denoise import SPACE as SD15_SIGMAS
from test_sd_denoise import TINY_ADM, tiny_adm, tiny_unet
from test_sd_denoise import tiny_cond as sd_cond
from test_sd_denoise import tiny_latent as sd_latent

FLOW = FlowSigmas()
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
ALL = PercentRange(0.0, 1.0)


def _region(
    *,
    scale: float | None = None,
    digest: str | None = None,
    schedule: ConditioningRange = ALL,
    tokens: int = 3,
) -> MaterializedRegion:
    return MaterializedRegion(
        Conditioning(torch.zeros((1, tokens, 2))),
        None,
        1.0,
        None,
        1.0,
        schedule,
        (),
        (2, 2),
        None if scale is None else torch.tensor([scale]),
        digest,
    )


def _patch(value: float = 1.0, size: int = 1, *, digest: str = DIGEST) -> PatchSet[torch.Tensor]:
    return PatchSet(
        {"weight": (PatchEntry(DiffPatch(torch.full((size, size), value, dtype=torch.float32))),)},
        structural_digest=digest,
    )


def _model() -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    return model


def _model_zero_patch(model: torch.nn.Module) -> PatchSet[torch.Tensor]:
    for name, module in model.named_modules():
        if type(module).__name__ in ("Linear", "_InitlessLinear", "_CastLinear"):
            linear = cast(torch.nn.Linear, module)
            return PatchSet(
                {f"{name}.weight": (PatchEntry(DiffPatch(torch.zeros_like(linear.weight))),)},
                structural_digest=DIGEST,
            )
    raise AssertionError("tiny production model has no admitted Linear")


def _evaluate(
    model: torch.nn.Module,
    calls: list[tuple[tuple[GuidanceRole, ...], int]],
):
    def evaluate(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        roles: tuple[GuidanceRole, ...],
        batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        del sigma, conditioning, batch_sizes
        calls.append((roles, len(regions)))
        value = x.movedim(1, -1)
        return model(value).movedim(-1, 1)

    return evaluate


def test_grouped_condition_scale_reverse_one_forward_accumulation_and_dynamic_uncond() -> None:
    model = _model()
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    result = evaluate_grouped_regions(
        (_region(scale=1.0, digest=DIGEST), _region(scale=3.0, digest=DIGEST)),
        (_region(digest=None, schedule=EMPTY_RANGE),),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(10**12, 0),
    )
    assert torch.equal(result.conditional, torch.full((1, 1, 2, 2), 4.0))
    assert torch.equal(result.unconditional, torch.zeros((1, 1, 2, 2)))
    assert calls == [((GuidanceRole.CONDITIONAL, GuidanceRole.CONDITIONAL), 2)]
    assert result.model_calls == 1
    assert result.subgroup_sizes == (2,)
    assert result.staged_bytes == 4
    assert not model._forward_hooks


def test_mixed_roles_share_one_reversed_physical_forward() -> None:
    model = _model()
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    result = evaluate_grouped_regions(
        (_region(scale=1.0, digest=DIGEST),),
        (_region(scale=3.0, digest=DIGEST),),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(10**12, 0),
    )
    assert calls == [((GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL), 2)]
    assert result.model_calls == 1
    assert torch.equal(result.conditional, torch.full_like(result.conditional, 3.0))
    assert torch.equal(result.unconditional, torch.full_like(result.unconditional, 5.0))


def test_real_flux_and_sd_grouped_helpers_execute_one_reversed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flux_model = tiny_flux()
    flux_conditional = flux_cond("grouped-cond")
    flux_unconditional = flux_cond("grouped-uncond")
    flux_cond_region = replace(
        _region(scale=1.0, digest=DIGEST),
        conditioning=flux_conditional,
        latent_shape=(8, 8),
    )
    flux_uncond_region = replace(
        _region(scale=3.0, digest=DIGEST),
        conditioning=flux_unconditional,
        latent_shape=(8, 8),
    )
    flux_x = flux_latent()
    assert flux_unconditional.pooled is not None
    assert flux_conditional.pooled is not None
    flux_grouped = Conditioning(
        torch.cat((flux_unconditional.embeddings, flux_conditional.embeddings)),
        torch.cat((flux_unconditional.pooled, flux_conditional.pooled)),
    )
    flux_evaluator = flux_grouped_region_evaluator(flux_model, compute_dtype=torch.float32)
    flux_patch = _model_zero_patch(flux_model)
    with prepare_scaled_patches(
        flux_model, flux_patch, "cpu", torch.float32, lambda: False
    ) as prepared:
        with prepared.activate(torch.tensor([3.0, 1.0])):
            expected_flux = flux_evaluator(
                (flux_uncond_region, flux_cond_region),
                torch.cat((flux_x, flux_x)),
                0.5,
                flux_grouped,
                (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
                (1, 1),
            )
    flux_calls: list[int] = []
    flux_forward = flux_model.forward

    def observe_flux(*args: Any, **kwargs: Any) -> torch.Tensor:
        flux_calls.append(args[0].shape[0])
        return flux_forward(*args, **kwargs)

    monkeypatch.setattr(flux_model, "forward", observe_flux)
    flux_roles: list[tuple[GuidanceRole, ...]] = []

    def observe_flux_evaluator(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        roles: tuple[GuidanceRole, ...],
        batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        flux_roles.append(roles)
        return flux_evaluator(regions, x, sigma, conditioning, roles, batch_sizes)

    flux_activations: list[torch.Tensor] = []
    original_activate = PreparedScaledPatches.activate

    def observe_flux_activate(owner: PreparedScaledPatches, scale_vector: torch.Tensor) -> Any:
        flux_activations.append(scale_vector.detach().clone())
        return original_activate(owner, scale_vector)

    with monkeypatch.context() as activation_patch:
        activation_patch.setattr(PreparedScaledPatches, "activate", observe_flux_activate)
        actual_flux = evaluate_grouped_regions(
            (flux_cond_region,),
            (flux_uncond_region,),
            flux_x,
            0.5,
            FLOW,
            FLUX_DEV,
            flux_model,
            observe_flux_evaluator,
            {DIGEST: flux_patch},
            lambda: False,
            compute_dtype=torch.float32,
            free_memory=lambda _device: DeviceMemory(10**12, 0),
        )
    assert actual_flux.model_calls == 1
    assert flux_calls == [2]
    assert flux_roles == [(GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL)]
    assert len(flux_activations) == 1
    assert torch.equal(flux_activations[0], torch.tensor([3.0, 1.0]))
    assert torch.equal(actual_flux.unconditional, expected_flux[:1])
    assert torch.equal(actual_flux.conditional, expected_flux[1:])

    sd_model = tiny_unet(TINY_ADM)
    sd_conditional = sd_cond("grouped-cond", features=TINY_ADM.context_dim)
    sd_unconditional = sd_cond("grouped-uncond", features=TINY_ADM.context_dim)
    sd_cond_region = replace(
        _region(scale=2.0, digest=DIGEST),
        conditioning=sd_conditional,
        latent_shape=(8, 8),
    )
    sd_uncond_region = replace(
        _region(scale=4.0, digest=DIGEST),
        conditioning=sd_unconditional,
        latent_shape=(8, 8),
    )
    sd_x = sd_latent()
    cond_adm = tiny_adm("grouped-cond")
    uncond_adm = tiny_adm("grouped-uncond")

    def adm(_region: MaterializedRegion, role: GuidanceRole) -> torch.Tensor:
        return cond_adm if role is GuidanceRole.CONDITIONAL else uncond_adm

    sd_grouped = Conditioning(torch.cat((sd_unconditional.embeddings, sd_conditional.embeddings)))
    sd_evaluator = sd_grouped_region_evaluator(
        sd_model,
        SD15_SIGMAS,
        parameterization=Parameterization.EPS,
        adm=adm,
        compute_dtype=torch.float32,
    )
    sd_patch = _model_zero_patch(sd_model)
    with prepare_scaled_patches(
        sd_model, sd_patch, "cpu", torch.float32, lambda: False
    ) as prepared:
        with prepared.activate(torch.tensor([4.0, 2.0])):
            expected_sd = sd_evaluator(
                (sd_uncond_region, sd_cond_region),
                torch.cat((sd_x, sd_x)),
                0.5,
                sd_grouped,
                (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
                (1, 1),
            )
    sd_calls: list[tuple[int, torch.Tensor | None]] = []
    sd_forward = sd_model.forward

    def observe_sd(*args: Any, **kwargs: Any) -> torch.Tensor:
        sd_calls.append((args[0].shape[0], kwargs.get("y")))
        return sd_forward(*args, **kwargs)

    monkeypatch.setattr(sd_model, "forward", observe_sd)
    sd_roles: list[tuple[GuidanceRole, ...]] = []

    def observe_sd_evaluator(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        roles: tuple[GuidanceRole, ...],
        batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        sd_roles.append(roles)
        return sd_evaluator(regions, x, sigma, conditioning, roles, batch_sizes)

    sd_activations: list[torch.Tensor] = []

    def observe_sd_activate(owner: PreparedScaledPatches, scale_vector: torch.Tensor) -> Any:
        sd_activations.append(scale_vector.detach().clone())
        return original_activate(owner, scale_vector)

    with monkeypatch.context() as activation_patch:
        activation_patch.setattr(PreparedScaledPatches, "activate", observe_sd_activate)
        actual_sd = evaluate_grouped_regions(
            (sd_cond_region,),
            (sd_uncond_region,),
            sd_x,
            0.5,
            SD15_SIGMAS,
            SD15,
            sd_model,
            observe_sd_evaluator,
            {DIGEST: sd_patch},
            lambda: False,
            compute_dtype=torch.float32,
            free_memory=lambda _device: DeviceMemory(10**12, 0),
        )
    assert actual_sd.model_calls == 1
    assert len(sd_calls) == 1 and sd_calls[0][0] == 2
    assert sd_calls[0][1] is not None
    assert sd_roles == [(GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL)]
    assert len(sd_activations) == 1
    assert torch.equal(sd_activations[0], torch.tensor([4.0, 2.0]))
    assert torch.equal(sd_calls[0][1], torch.cat((uncond_adm, cond_adm)))
    assert torch.equal(actual_sd.unconditional, expected_sd[:1])
    assert torch.equal(actual_sd.conditional, expected_sd[1:])


def test_memory_reciprocal_floor_patch_isolation_and_mapping_refuse_before_model() -> None:
    model = _model()
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    regions = tuple(_region(scale=float(index), digest=DIGEST) for index in range(4))
    two_required = regional_working_memory(SD15, batch=2, height=2, width=2, dtype=torch.float32)
    result = evaluate_grouped_regions(
        regions,
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(int(two_required * 1.6), 0),
    )
    assert result.subgroup_sizes == (2, 2)
    assert result.model_calls == 2

    boundary = evaluate_grouped_regions(
        regions[:2],
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(int(two_required * 1.5), 0),
    )
    assert boundary.subgroup_sizes == (1, 1)

    base = regional_working_memory(SD15, batch=3, height=12, width=12, dtype=torch.float32)
    assert base == 18119393.28
    assert (
        regional_working_memory(SDXL, batch=3, height=12, width=12, dtype=torch.float32)
        == base * 0.8
    )
    assert (
        regional_working_memory(SDXL_REFINER, batch=3, height=12, width=12, dtype=torch.float32)
        == base
    )
    assert (
        regional_working_memory(FLUX_DEV, batch=3, height=12, width=12, dtype=torch.float32)
        == base * 3.1
    )
    assert (
        regional_working_memory(FLUX_SCHNELL, batch=3, height=12, width=12, dtype=torch.float32)
        == base * 3.1
    )
    provided_family = replace(
        SD15,
        engine=replace(SD15.engine, regional_memory_factor=2.5),
    )
    assert (
        regional_working_memory(provided_family, batch=3, height=12, width=12, dtype=torch.float32)
        == base * 2.5
    )

    six_regions = tuple(_region(scale=float(index), digest=DIGEST) for index in range(6))
    three_required = regional_working_memory(SD15, batch=3, height=2, width=2, dtype=torch.float32)
    memory_reads: list[torch.device] = []

    def boundary_memory(device: torch.device) -> DeviceMemory:
        memory_reads.append(device)
        return DeviceMemory(int((two_required * 1.5 + three_required * 1.5) / 2), 0)

    reciprocal_boundary = evaluate_grouped_regions(
        six_regions,
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=boundary_memory,
    )
    assert reciprocal_boundary.subgroup_sizes == (2, 2, 2)
    assert memory_reads == [torch.device("cpu")] * 3

    isolated = evaluate_grouped_regions(
        (
            _region(scale=1.0, digest=DIGEST),
            _region(scale=1.0, digest=OTHER_DIGEST),
        ),
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch(1.0), OTHER_DIGEST: _patch(2.0, digest=OTHER_DIGEST)},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(10**12, 0),
    )
    assert isolated.subgroup_sizes == (1, 1)
    assert isolated.model_calls == 2
    assert isolated.staged_bytes == 8
    assert torch.equal(isolated.conditional, torch.full_like(isolated.conditional, 3.5))

    calls.clear()
    with pytest.raises(RegionalConditioningError, match="patch-mapping"):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert calls == []

    with pytest.raises(RegionalConditioningError, match="patch-mapping"):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: _patch(digest=OTHER_DIGEST)},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert calls == []

    absent_digest = PatchSet(
        {"weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),)},
        structural_digest=None,
    )
    with pytest.raises(RegionalConditioningError, match="patch-mapping"):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: absent_digest},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert calls == []

    mismatched = PatchSet(
        {"missing.weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),)},
        structural_digest=DIGEST,
    )
    with pytest.raises(RegionalConditioningError, match="patch-preparation.*target-resolution"):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: mismatched},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert calls == []

    with pytest.raises(RegionalConditioningError, match="patch-mapping"):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: _patch(), OTHER_DIGEST: _patch(digest=OTHER_DIGEST)},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("override", "code"),
    (
        ({"conditional": []}, "invalid-conditional"),
        ({"unconditional": (object(),)}, "invalid-unconditional"),
        ({"x": object()}, "latent-shape"),
        ({"x": torch.ones((1, 1, 2))}, "latent-shape"),
        ({"x": torch.ones((1, 1, 2, 2), dtype=torch.float64)}, "latent-dtype"),
        (
            {
                "x": torch.sparse_coo_tensor(
                    torch.zeros((4, 1), dtype=torch.int64),
                    torch.ones(1),
                    (1, 1, 2, 2),
                )
            },
            "latent-layout",
        ),
        ({"model": object()}, "model"),
        ({"patch_sets": []}, "patch-mapping-type"),
        ({"patch_sets": {DIGEST: object()}}, "patch-mapping-type"),
        (
            {"patch_sets": cast(Any, {DIGEST: _patch(), 1: _patch()})},
            "patch-mapping-type",
        ),
        ({"compute_dtype": torch.int64}, "compute-dtype"),
        ({"space": object()}, "sigma-space"),
        ({"sigma": float("nan")}, "sigma"),
        ({"cancel": lambda: None}, "callback"),
        (
            {
                "conditional": (
                    replace(
                        _region(scale=1.0, digest=DIGEST),
                        scale_vector=torch.tensor([float("nan")]),
                    ),
                )
            },
            "scale-vector",
        ),
        (
            {
                "conditional": (
                    replace(
                        _region(scale=1.0, digest=DIGEST),
                        scale_vector=torch.sparse_coo_tensor(
                            torch.tensor([[0]]),
                            torch.tensor([1.0]),
                            (1,),
                            check_invariants=False,
                        ),
                    ),
                )
            },
            "scale-vector",
        ),
        (
            {
                "conditional": (
                    replace(
                        _region(scale=1.0, digest=DIGEST),
                        scale_vector=torch.ones(2),
                    ),
                )
            },
            "scale-batch",
        ),
        (
            {"conditional": (replace(_region(), mask=torch.empty((0, 2, 2))),)},
            "mask-shape",
        ),
        (
            {"conditional": (replace(_region(), mask=torch.ones((2, 2))),)},
            "mask-shape",
        ),
        (
            {
                "conditional": (
                    replace(
                        _region(),
                        conditioning=Conditioning(torch.zeros((2, 3, 2))),
                    ),
                )
            },
            "conditioning-shape",
        ),
    ),
)
def test_grouped_public_preflight_refuses_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    code: str,
) -> None:
    model_calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    memory_calls: list[torch.device] = []
    stage_calls = 0
    original_stage = scaled_module._stage_targets

    def observe_stage(*args: Any, **kwargs: Any):
        nonlocal stage_calls
        stage_calls += 1
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(scaled_module, "_stage_targets", observe_stage)

    def free_memory(device: torch.device) -> DeviceMemory:
        memory_calls.append(device)
        return DeviceMemory(10**12, 0)

    arguments: dict[str, object] = {
        "conditional": (_region(scale=1.0, digest=DIGEST),),
        "unconditional": (),
        "x": torch.ones((1, 1, 2, 2)),
        "sigma": 0.5,
        "space": FLOW,
        "family": SD15,
        "model": _model(),
        "evaluate": _evaluate(_model(), model_calls),
        "patch_sets": {DIGEST: _patch()},
        "cancel": lambda: False,
        "compute_dtype": torch.float32,
        "free_memory": free_memory,
    }
    arguments.update(override)
    with pytest.raises(RegionalConditioningError, match=code):
        evaluate_grouped_regions(**cast(Any, arguments))
    assert stage_calls == 0
    assert model_calls == []
    assert memory_calls == []


def test_grouped_region_preflight_precedes_mapping_copy() -> None:
    class UnreadableMapping(Mapping[str, PatchSet[torch.Tensor]]):
        def __getitem__(self, key: str) -> PatchSet[torch.Tensor]:
            raise AssertionError(f"mapping read before preflight: {key}")

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("mapping iterated before preflight")

        def __len__(self) -> int:
            return 1

    invalid = replace(
        _region(scale=1.0, digest=DIGEST),
        scale_vector=torch.tensor([float("inf")]),
    )
    with pytest.raises(RegionalConditioningError, match="scale-vector"):
        evaluate_grouped_regions(
            (invalid,),
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            _model(),
            _evaluate(_model(), []),
            UnreadableMapping(),
            lambda: False,
            compute_dtype=torch.float32,
        )


def test_grouped_mapping_exception_is_normalized_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisingMapping(Mapping[str, PatchSet[torch.Tensor]]):
        def __getitem__(self, key: str) -> PatchSet[torch.Tensor]:
            raise RuntimeError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("mapping-body")

        def __len__(self) -> int:
            return 1

    stage_calls = 0

    def observe_stage(*_args: Any, **_kwargs: Any) -> None:
        nonlocal stage_calls
        stage_calls += 1

    monkeypatch.setattr(scaled_module, "_stage_targets", observe_stage)
    with pytest.raises(RegionalConditioningError, match="patch-mapping-type"):
        evaluate_grouped_regions(
            (_region(),),
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            _model(),
            _evaluate(_model(), []),
            RaisingMapping(),
            lambda: False,
            compute_dtype=torch.float32,
            free_memory=lambda _device: (_ for _ in ()).throw(AssertionError("free-memory called")),
        )
    assert stage_calls == 0


def test_pairwise_cross_attention_cap_allows_aggregate_lcm_repeat() -> None:
    model = _model()
    observed: list[tuple[tuple[float, ...], int]] = []

    def evaluate(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        _sigma: float,
        conditioning: Conditioning[torch.Tensor],
        _roles: tuple[GuidanceRole, ...],
        _batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        scales: list[float] = []
        for region in regions:
            assert region.scale_vector is not None
            scales.append(float(region.scale_vector.item()))
        observed.append(
            (
                tuple(scales),
                conditioning.embeddings.shape[1],
            )
        )
        return model(x.movedim(1, -1)).movedim(-1, 1)

    result = evaluate_grouped_regions(
        (
            _region(scale=1.0, digest=DIGEST, tokens=6),
            _region(scale=2.0, digest=DIGEST, tokens=4),
            _region(scale=3.0, digest=DIGEST, tokens=9),
        ),
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        evaluate,
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
        free_memory=lambda _device: DeviceMemory(10**12, 0),
    )
    assert result.subgroup_sizes == (3,)
    assert observed == [((3.0, 2.0, 1.0), 36)]


def test_scale_batch_unit_rows_and_cancellation_cleanup() -> None:
    model = _model()
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    x = torch.ones((2, 1, 2, 2))
    region = replace(
        _region(digest=DIGEST),
        scale_vector=torch.tensor([1.0, 3.0]),
    )
    result = evaluate_grouped_regions(
        (region,),
        (),
        x,
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
    )
    assert torch.equal(result.conditional[:, 0, 0, 0], torch.tensor([3.0, 5.0]))

    unit = evaluate_grouped_regions(
        (_region(digest=DIGEST),),
        (),
        torch.ones((1, 1, 2, 2)),
        0.5,
        FLOW,
        SD15,
        model,
        _evaluate(model, calls),
        {DIGEST: _patch()},
        lambda: False,
        compute_dtype=torch.float32,
    )
    assert torch.equal(unit.conditional, torch.full_like(unit.conditional, 3.0))

    bad = replace(region, scale_vector=torch.ones(3))
    with pytest.raises(RegionalConditioningError, match="scale-batch"):
        evaluate_grouped_regions(
            (bad,),
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: _patch()},
            lambda: False,
            compute_dtype=torch.float32,
        )
    assert not model._forward_hooks

    cancelled = False

    def free_memory(_device: torch.device) -> DeviceMemory:
        nonlocal cancelled
        cancelled = True
        return DeviceMemory(10**12, 0)

    with pytest.raises(RegionalConditioningError, match="cancelled"):
        evaluate_grouped_regions(
            (_region(scale=1.0, digest=DIGEST), _region(scale=1.0, digest=DIGEST)),
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            {DIGEST: _patch()},
            lambda: cancelled,
            compute_dtype=torch.float32,
            free_memory=free_memory,
        )
    assert not model._forward_hooks


def test_grouped_model_error_is_preserved_and_all_patch_owners_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    calls = 0
    closes: list[int] = []
    staged: list[weakref.ReferenceType[torch.Tensor]] = []
    original_close = PreparedScaledPatches.close

    def observe_close(owner: PreparedScaledPatches) -> None:
        closes.append(id(owner))
        for target in owner._staged:
            for entry in target.entries:
                staged.append(weakref.ref(entry.first))
                if entry.second is not None:
                    staged.append(weakref.ref(entry.second))
        original_close(owner)

    monkeypatch.setattr(PreparedScaledPatches, "close", observe_close)

    def fail(
        _regions: tuple[MaterializedRegion, ...],
        _x: torch.Tensor,
        _sigma: float,
        _conditioning: Conditioning[torch.Tensor],
        _roles: tuple[GuidanceRole, ...],
        _batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise RuntimeError("first-model-error")

    with pytest.raises(RuntimeError, match="first-model-error"):
        evaluate_grouped_regions(
            (
                _region(scale=1.0, digest=DIGEST),
                _region(scale=1.0, digest=OTHER_DIGEST),
            ),
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            fail,
            {DIGEST: _patch(), OTHER_DIGEST: _patch(2.0, digest=OTHER_DIGEST)},
            lambda: False,
            compute_dtype=torch.float32,
            free_memory=lambda _device: DeviceMemory(10**12, 0),
        )
    assert calls == 1
    assert not model._forward_hooks
    assert len(closes) == 2 and len(set(closes)) == 2
    gc.collect()
    assert staged and all(reference() is None for reference in staged)
    with prepare_scaled_patches(model, _patch(), "cpu", torch.float32, lambda: False) as owner:
        with owner.activate(torch.ones(1)):
            model(torch.ones((1, 1)))


def test_execution_owner_stages_once_across_two_sigma_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    patch = _patch()

    def cancel() -> bool:
        return False

    stage_calls = 0
    close_calls = 0
    original_stage = scaled_module._stage_targets
    original_close = PreparedScaledPatches.close

    def observe_stage(*args: Any, **kwargs: Any):
        nonlocal stage_calls
        stage_calls += 1
        return original_stage(*args, **kwargs)

    def observe_close(owner: PreparedScaledPatches) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(owner)

    monkeypatch.setattr(scaled_module, "_stage_targets", observe_stage)
    monkeypatch.setattr(PreparedScaledPatches, "close", observe_close)

    class CountingMapping(Mapping[str, PatchSet[torch.Tensor]]):
        def __init__(self) -> None:
            self.data = {DIGEST: patch}
            self.iterations = 0

        def __getitem__(self, key: str) -> PatchSet[torch.Tensor]:
            return self.data[key]

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            return iter(self.data)

        def __len__(self) -> int:
            return len(self.data)

    supplied = CountingMapping()
    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))
    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        supplied,
        "cpu",
        torch.float32,
        cancel,
    )
    assert supplied.iterations == 1
    supplied.data.clear()
    assert owner.digests == frozenset((DIGEST,))
    assert owner._prepared[DIGEST].patch_set is patch
    staged_ids = tuple(
        id(entry.first) for target in owner._prepared[DIGEST]._staged for entry in target.entries
    )
    staged_refs = tuple(
        weakref.ref(entry.first)
        for target in owner._prepared[DIGEST]._staged
        for entry in target.entries
    )
    assert stage_calls == 1 and staged_ids
    for sigma in (0.75, 0.25):
        result = evaluate_grouped_regions(
            regions,
            (),
            x,
            sigma,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda _device: DeviceMemory(10**12, 0),
        )
        assert result.staged_bytes == 4
        assert staged_ids == tuple(
            id(entry.first)
            for target in owner._prepared[DIGEST]._staged
            for entry in target.entries
        )
    assert stage_calls == 1 and close_calls == 0 and supplied.iterations == 1
    with owner.evaluation(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        torch.device("cpu"),
        torch.float32,
        cancel,
    ):
        with pytest.raises(ValueError, match="prepared-active"):
            owner.close()
    assert owner._prepared
    owner.close()
    owner.close()
    assert close_calls == 1 and owner._prepared == {}
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS
    gc.collect()
    assert all(reference() is None for reference in staged_refs)


def test_execution_owner_refuses_mutated_scale_before_evaluation_side_effects() -> None:
    model = _model()
    scale = torch.ones(1)
    regions = (replace(_region(digest=DIGEST), scale_vector=scale),)
    x = torch.ones((1, 1, 2, 2))
    model_calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    memory_calls: list[torch.device] = []

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )
    scale.fill_(2.0)
    with pytest.raises(RegionalConditioningError, match="prepared-sample"):
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, model_calls),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda device: memory_calls.append(device) or DeviceMemory(10**12, 0),
        )
    assert model_calls == []
    assert memory_calls == []
    owner.close()
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS


def test_execution_owner_refuses_same_set_digest_reordering() -> None:
    model = _model()
    regions = (
        _region(scale=1.0, digest=DIGEST),
        _region(scale=1.0, digest=OTHER_DIGEST),
    )
    x = torch.ones((1, 1, 2, 2))

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch(), OTHER_DIGEST: _patch(digest=OTHER_DIGEST)},
        "cpu",
        torch.float32,
        cancel,
    )
    object.__setattr__(regions[0], "patch_digest", OTHER_DIGEST)
    object.__setattr__(regions[1], "patch_digest", DIGEST)
    with pytest.raises(RegionalConditioningError, match="prepared-sample"):
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda _device: (_ for _ in ()).throw(AssertionError("free-memory called")),
        )
    owner.close()
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS


@pytest.mark.parametrize(
    "drift",
    (
        "space",
        "latent-descriptor",
        "region-tuple",
        "tensor-version",
        "tensor-identity",
        "mask-version",
    ),
)
def test_execution_owner_refuses_execution_descriptor_drift(drift: str) -> None:
    model = _model()
    region = _region(scale=1.0, digest=DIGEST)
    if drift == "mask-version":
        region = replace(region, mask=torch.ones((1, 2, 2)))
    regions = (region,)
    x = torch.ones((1, 1, 2, 2))
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )
    evaluated_regions = regions
    evaluated_x = x
    evaluated_space = FLOW
    if drift == "space":
        evaluated_space = FlowSigmas()
    elif drift == "latent-descriptor":
        evaluated_x = torch.empty_strided(x.shape, (4, 4, 1, 2))
    elif drift == "region-tuple":
        evaluated_regions = tuple([region])
    elif drift == "tensor-version":
        region.conditioning.embeddings.add_(1.0)
    elif drift == "tensor-identity":
        object.__setattr__(
            region,
            "conditioning",
            Conditioning(
                region.conditioning.embeddings.clone(),
                region.conditioning.pooled,
            ),
        )
    else:
        assert region.mask is not None
        region.mask.add_(0.5)
    with pytest.raises(RegionalConditioningError, match="prepared-sample"):
        evaluate_grouped_regions(
            evaluated_regions,
            (),
            evaluated_x,
            0.5,
            evaluated_space,
            SD15,
            model,
            _evaluate(model, calls),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda _device: (_ for _ in ()).throw(AssertionError("free-memory called")),
        )
    assert calls == []
    owner.close()
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS


def test_execution_owner_preflights_target_and_cancel_drift_before_callbacks() -> None:
    model = _model()
    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )

    def foreign_hook(*_args: Any) -> None:
        return None

    handle = model.register_forward_hook(foreign_hook)
    memory_calls: list[torch.device] = []
    with pytest.raises(RegionalConditioningError, match="prepared-patch"):
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda device: memory_calls.append(device) or DeviceMemory(10**12, 0),
        )
    assert memory_calls == []
    handle.remove()

    replacement_calls = 0

    def replacement_cancel() -> bool:
        nonlocal replacement_calls
        replacement_calls += 1
        return False

    with pytest.raises(RegionalConditioningError, match="prepared-cancel"):
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            owner,
            replacement_cancel,
            compute_dtype=torch.float32,
        )
    assert replacement_calls == 0
    owner.close()
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS


def test_execution_owner_normalizes_finite_payload_version_drift() -> None:
    model = _model()
    payload = torch.ones((1, 1))
    patch = PatchSet(
        {"weight": (PatchEntry(DiffPatch(payload)),)},
        structural_digest=DIGEST,
    )
    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: patch},
        "cpu",
        torch.float32,
        cancel,
    )
    payload.add_(0.5)
    with pytest.raises(
        RegionalConditioningError,
        match="prepared-patch.*payload-drift",
    ):
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda _device: (_ for _ in ()).throw(AssertionError("free-memory called")),
        )
    owner.close()
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS


def test_grouped_inference_tensor_and_memory_result_refuse_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.inference_mode():
        inference_embeddings = torch.ones((1, 3, 2))
    region = replace(
        _region(scale=1.0, digest=DIGEST),
        conditioning=Conditioning(inference_embeddings),
    )
    stage_calls = 0

    def reject_stage(*_args: Any, **_kwargs: Any) -> Never:
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("staging must not run")

    monkeypatch.setattr(scaled_module, "_stage_targets", reject_stage)
    with pytest.raises(RegionalConditioningError, match="inference-tensor"):
        prepare_grouped_patches(
            (region,),
            (),
            torch.ones((1, 1, 2, 2)),
            "dinkster.sd15",
            FLOW,
            _model(),
            {DIGEST: _patch()},
            "cpu",
            torch.float32,
            lambda: False,
        )
    assert stage_calls == 0

    with pytest.raises(RegionalConditioningError, match="memory-result"):
        evaluate_grouped_regions(
            (_region(),),
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            _model(),
            _evaluate(_model(), []),
            {},
            lambda: False,
            compute_dtype=torch.float32,
            free_memory=lambda _device: cast(Any, None),
        )


@pytest.mark.parametrize("kind", ("diff", "lora-up", "lora-down"))
def test_grouped_finite_payload_overflow_refuses_transactionally(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    model = _model().to(dtype=torch.float16)
    overflow = torch.tensor(1e10, dtype=torch.float32)
    assert torch.isfinite(overflow)
    if kind == "diff":
        patch_value: DiffPatch[torch.Tensor] | AdapterPatch[torch.Tensor] = DiffPatch(
            overflow.reshape(1, 1)
        )
    else:
        up = torch.ones((1, 1))
        down = torch.ones((1, 1))
        if kind == "lora-up":
            up = overflow.reshape(1, 1)
        else:
            down = overflow.reshape(1, 1)
        patch_value = AdapterPatch(LoRAAdapter(up, down))
    patch = PatchSet(
        {"weight": (PatchEntry(patch_value),)},
        structural_digest=DIGEST,
    )
    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_finite = scaled_module._require_finite

    def observe(value: torch.Tensor, code: str) -> None:
        if code == "staged-payload-finite":
            staged_refs.append(weakref.ref(value))
        original_finite(value, code)

    monkeypatch.setattr(scaled_module, "_require_finite", observe)
    with pytest.raises(
        RegionalConditioningError,
        match="patch-preparation.*staged-payload-finite",
    ):
        prepare_grouped_patches(
            (_region(scale=1.0, digest=DIGEST),),
            (),
            torch.ones((1, 1, 2, 2)),
            "dinkster.sd15",
            FLOW,
            model,
            {DIGEST: patch},
            "cpu",
            torch.float16,
            lambda: False,
        )
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS
    gc.collect()
    assert staged_refs and all(reference() is None for reference in staged_refs)


def test_grouped_finite_scale_overflow_refuses_transactionally() -> None:
    model = _model().to(dtype=torch.float16)
    scale = torch.tensor([1e10], dtype=torch.float32)
    assert torch.isfinite(scale).all()
    regions = (replace(_region(digest=DIGEST), scale_vector=scale),)

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        torch.ones((1, 1, 2, 2)),
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float16,
        cancel,
    )
    calls: list[tuple[tuple[GuidanceRole, ...], int]] = []
    with pytest.raises(
        RegionalConditioningError,
        match="staged-scale-finite",
    ):
        evaluate_grouped_regions(
            regions,
            (),
            torch.ones((1, 1, 2, 2)),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, calls),
            owner,
            cancel,
            compute_dtype=torch.float16,
        )
    assert calls == []
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS
    owner.close()


@pytest.mark.parametrize(
    ("override", "code"),
    (
        ({"model": object()}, "model"),
        ({"conditional": []}, "invalid-conditional"),
        ({"x": torch.ones((1, 1, 2))}, "latent-shape"),
        ({"x": torch.ones((1, 1, 2, 2), dtype=torch.float64)}, "latent-dtype"),
        ({"patch_sets": []}, "patch-mapping-type"),
        ({"patch_sets": {DIGEST: object()}}, "patch-mapping-type"),
        ({"device": "not-a-device"}, "execution-device"),
        ({"dtype": torch.int64}, "compute-dtype"),
        ({"cancel": object()}, "callback"),
        ({"cancel": lambda: None}, "callback"),
        (
            {
                "conditional": (
                    replace(
                        _region(scale=1.0, digest=DIGEST),
                        scale_vector=torch.ones(2),
                    ),
                )
            },
            "scale-batch",
        ),
        (
            {
                "conditional": (
                    replace(
                        _region(scale=1.0, digest=DIGEST),
                        scale_vector=torch.tensor([float("nan")]),
                    ),
                )
            },
            "scale-vector",
        ),
    ),
)
def test_prepare_grouped_patches_public_preflight_is_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    code: str,
) -> None:
    stage_calls = 0
    original_stage = scaled_module._stage_targets

    def observe_stage(*args: Any, **kwargs: Any):
        nonlocal stage_calls
        stage_calls += 1
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(scaled_module, "_stage_targets", observe_stage)
    arguments: dict[str, object] = {
        "conditional": (_region(scale=1.0, digest=DIGEST),),
        "unconditional": (),
        "x": torch.ones((1, 1, 2, 2)),
        "family_id": "dinkster.sd15",
        "space": FLOW,
        "model": _model(),
        "patch_sets": {DIGEST: _patch()},
        "device": "cpu",
        "dtype": torch.float32,
        "cancel": lambda: False,
    }
    arguments.update(override)
    with pytest.raises(RegionalConditioningError, match=code):
        prepare_grouped_patches(**cast(Any, arguments))
    assert stage_calls == 0


def test_prepare_grouped_patches_rejects_regions_before_mapping_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingMapping(Mapping[str, PatchSet[torch.Tensor]]):
        iterations = 0

        def __getitem__(self, key: str) -> PatchSet[torch.Tensor]:
            if key != DIGEST:
                raise KeyError(key)
            return _patch()

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            return iter((DIGEST,))

        def __len__(self) -> int:
            return 1

    stage_calls = 0

    def reject_stage(*_args: Any, **_kwargs: Any) -> Never:
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("staging must not run")

    monkeypatch.setattr(scaled_module, "_stage_targets", reject_stage)
    supplied = CountingMapping()
    invalid = replace(
        _region(scale=1.0, digest=DIGEST),
        scale_vector=torch.tensor([float("inf")]),
    )
    with pytest.raises(RegionalConditioningError, match="scale-vector"):
        prepare_grouped_patches(
            (invalid,),
            (),
            torch.ones((1, 1, 2, 2)),
            "dinkster.sd15",
            FLOW,
            _model(),
            supplied,
            "cpu",
            torch.float32,
            lambda: False,
        )
    assert supplied.iterations == 0
    assert stage_calls == 0


def test_prepare_grouped_patches_polls_cancel_before_mapping_copy() -> None:
    class UnreadableMapping(Mapping[str, PatchSet[torch.Tensor]]):
        def __getitem__(self, key: str) -> PatchSet[torch.Tensor]:
            raise AssertionError(key)

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("mapping iterated")

        def __len__(self) -> int:
            return 1

    with pytest.raises(RegionalConditioningError, match="cancelled"):
        prepare_grouped_patches(
            (_region(scale=1.0, digest=DIGEST),),
            (),
            torch.ones((1, 1, 2, 2)),
            "dinkster.sd15",
            FLOW,
            _model(),
            UnreadableMapping(),
            "cpu",
            torch.float32,
            lambda: True,
        )


def test_prepare_grouped_patches_preflights_all_digests_before_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_calls = 0
    original_stage = scaled_module._stage_targets

    def observe_stage(*args: Any, **kwargs: Any):
        nonlocal stage_calls
        stage_calls += 1
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(scaled_module, "_stage_targets", observe_stage)
    invalid = PatchSet(
        {"missing.weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),)},
        structural_digest=OTHER_DIGEST,
    )
    with pytest.raises(RegionalConditioningError, match="patch-preparation.*target-resolution"):
        prepare_grouped_patches(
            (
                _region(scale=1.0, digest=DIGEST),
                _region(scale=1.0, digest=OTHER_DIGEST),
            ),
            (),
            torch.ones((1, 1, 2, 2)),
            "dinkster.sd15",
            FLOW,
            _model(),
            {DIGEST: _patch(), OTHER_DIGEST: invalid},
            "cpu",
            torch.float32,
            lambda: False,
        )
    assert stage_calls == 0


@pytest.mark.parametrize("failure", ("cancel", "error"))
def test_execution_owner_failure_close_releases_all_state(failure: str) -> None:
    model = _model()
    cancelled = False

    def cancel() -> bool:
        return cancelled

    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))
    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )
    staged_refs = tuple(
        weakref.ref(entry.first)
        for target in owner._prepared[DIGEST]._staged
        for entry in target.entries
    )

    def fail(
        _regions: tuple[MaterializedRegion, ...],
        _x: torch.Tensor,
        _sigma: float,
        _conditioning: Conditioning[torch.Tensor],
        _roles: tuple[GuidanceRole, ...],
        _batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        raise RuntimeError("model-error")

    if failure == "cancel":
        cancelled = True
        expected = pytest.raises(RegionalConditioningError, match="cancelled")
    else:
        expected = pytest.raises(RuntimeError, match="model-error")
    with expected:
        evaluate_grouped_regions(
            regions,
            (),
            x,
            0.5,
            FLOW,
            SD15,
            model,
            fail if failure == "error" else _evaluate(model, []),
            owner,
            cancel,
            compute_dtype=torch.float32,
            free_memory=lambda _device: DeviceMemory(10**12, 0),
        )
    owner.close()
    assert owner._prepared == {}
    assert not model._forward_hooks and model not in scaled_module._ACTIVE_MODELS
    gc.collect()
    assert all(reference() is None for reference in staged_refs)


def test_execution_owner_refuses_binding_drift_and_context_reentry() -> None:
    model = _model()
    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )
    with pytest.raises(ValueError, match="prepared-execution"):
        owner.validate_execution(
            regions,
            (),
            x,
            "dinkster.sd15",
            FLOW,
            _model(),
            torch.device("cpu"),
            torch.float32,
            cancel,
        )
    with pytest.raises(ValueError, match="prepared-execution"):
        owner.validate_execution(
            regions,
            (),
            x,
            "dinkster.sd15",
            FLOW,
            model,
            torch.device("cpu"),
            torch.float64,
            cancel,
        )
    with pytest.raises(ValueError, match="prepared-cancel"):
        owner.validate_execution(
            regions,
            (),
            x,
            "dinkster.sd15",
            FLOW,
            model,
            torch.device("cpu"),
            torch.float32,
            lambda: False,
        )
    original_model = owner._model
    owner._model = _model()
    with pytest.raises(ValueError, match="prepared-binding"):
        owner.validate_execution(
            regions,
            (),
            x,
            "dinkster.sd15",
            FLOW,
            owner._model,
            torch.device("cpu"),
            torch.float32,
            cancel,
        )
    owner._model = original_model
    owner.__enter__()
    with pytest.raises(ValueError, match="prepared-closed"):
        owner.__enter__()
    owner.__exit__(None, None, None)
    assert owner._closed and owner._prepared == {}


def test_execution_owner_active_exit_is_transactional() -> None:
    model = _model()
    regions = (_region(scale=1.0, digest=DIGEST),)
    x = torch.ones((1, 1, 2, 2))

    def cancel() -> bool:
        return False

    owner = prepare_grouped_patches(
        regions,
        (),
        x,
        "dinkster.sd15",
        FLOW,
        model,
        {DIGEST: _patch()},
        "cpu",
        torch.float32,
        cancel,
    )
    owner.__enter__()
    prepared = dict(owner._prepared)
    with pytest.raises(RuntimeError, match="body"):
        with owner.evaluation(
            regions,
            (),
            x,
            "dinkster.sd15",
            FLOW,
            model,
            torch.device("cpu"),
            torch.float32,
            cancel,
        ):
            with pytest.raises(ValueError, match="prepared-active"):
                owner.__enter__()
            try:
                raise RuntimeError("body")
            except RuntimeError:
                with pytest.raises(ValueError, match="prepared-active"):
                    owner.__exit__(None, None, None)
                assert owner._entered
                assert owner._prepared == prepared
                assert not owner._closed
                raise
    owner.__exit__(None, None, None)
    assert owner._closed and owner._prepared == {}


def test_prepared_activation_refuses_stale_owner_paths_and_weight_aliases() -> None:
    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = _model()

    patch = PatchSet(
        {"target.weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),)},
        structural_digest=DIGEST,
    )
    model = Container()
    prepared = prepare_scaled_patches(model, patch, "cpu", torch.float32, lambda: False)
    model.alias = model.target
    with pytest.raises(ValueError, match="shared-module"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not model.target._forward_hooks
    del model.alias
    prepared.close()

    model = Container()
    prepared = prepare_scaled_patches(model, patch, "cpu", torch.float32, lambda: False)
    model.alias = _model()
    model.alias.weight = torch.nn.Parameter(model.target.weight.detach())
    with pytest.raises(ValueError, match="shared-weight"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not model.target._forward_hooks
    prepared.close()

    model = Container()
    prepared = prepare_scaled_patches(model, patch, "cpu", torch.float32, lambda: False)
    original = model.target
    model.moved = original
    model.target = _model()
    with pytest.raises(ValueError, match="target-resolution"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not original._forward_hooks
    prepared.close()


def test_prepared_owner_admits_production_linears_accounts_and_cleans() -> None:
    models = (
        INITLESS.linear(1, 1, bias=False),
        CastOperations(torch.float32).linear(1, 1, bias=False),
    )
    for model in models:
        with torch.no_grad():
            model.weight.fill_(2.0)
        mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
        mechanism.partially_load(None)
        mechanism.partially_unload(mechanism.loaded_bytes())
        assert mechanism.loaded_bytes() == 0
        prepared = prepare_scaled_patches(model, _patch(), "cpu", torch.float32, lambda: False)
        assert prepared.staged_bytes == 4
        with prepared.activate(torch.tensor([2.0])):
            assert torch.equal(model(torch.ones((1, 1))), torch.tensor([[4.0]]))
        prepared.close()
        assert not model._forward_hooks
        assert torch.equal(model(torch.ones((1, 1))), torch.tensor([[2.0]]))

    with pytest.raises(ValueError, match="execution-placement"):
        prepare_scaled_patches(_model(), _patch(), "cpu", torch.float16, lambda: False)


def test_prepared_error_poison_and_close_are_transactional() -> None:
    model = _model()
    prepared = prepare_scaled_patches(model, _patch(), "cpu", torch.float32, lambda: False)
    with pytest.raises(RuntimeError, match="body"):
        with prepared.activate(torch.ones(1)):
            raise RuntimeError("body")
    assert not model._forward_hooks
    with pytest.raises(ValueError, match="prepared-poisoned"):
        prepared.activate(torch.ones(1)).__enter__()
    prepared.close()
    prepared.close()

    prepared = prepare_scaled_patches(model, _patch(), "cpu", torch.float32, lambda: False)
    foreign_hook = model.register_forward_hook(lambda _module, _args, output: output)
    with pytest.raises(ValueError, match="target-hooked"):
        prepared.activate(torch.ones(1)).__enter__()
    foreign_hook.remove()
    assert not model._forward_hooks
    prepared.close()

    cast_model = CastOperations(torch.float32).linear(1, 1, bias=False)
    prepared = prepare_scaled_patches(cast_model, _patch(), "cpu", torch.float32, lambda: False)
    cast(Any, cast_model).fp8_matmul = True
    with pytest.raises(ValueError, match="fp8-matmul"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not cast_model._forward_hooks
    prepared.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_grouped_condition_scale_cuda_production_linear() -> None:
    device = torch.device("cuda:0")
    for model in (
        INITLESS.linear(8, 8, bias=False),
        CastOperations(torch.float32).linear(8, 8, bias=False),
    ):
        with torch.no_grad():
            model.weight.fill_(2.0)
        mechanism = enroll_component(model, load_device=device, offload_device="cpu")
        mechanism.partially_load(None)
        mechanism.partially_unload(mechanism.loaded_bytes())
        assert mechanism.loaded_bytes() == 0
        assert model.weight.device.type == "cpu"
        region = replace(
            _region(scale=2.0, digest=DIGEST),
            conditioning=Conditioning(torch.zeros((1, 3, 2), device=device)),
            scale_vector=torch.tensor([2.0], device=device),
        )
        result = evaluate_grouped_regions(
            (region,),
            (),
            torch.ones((1, 8, 2, 2), device=device),
            0.5,
            FLOW,
            SD15,
            model,
            _evaluate(model, []),
            {DIGEST: _patch(size=8)},
            lambda: False,
            compute_dtype=torch.float32,
        )
        assert torch.equal(result.conditional, torch.full_like(result.conditional, 32.0))
        assert result.staged_bytes == 256
        assert mechanism.loaded_bytes() == 0
        assert not model._forward_hooks


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_grouped_condition_scale_same_process_two_gpu_isolation() -> None:
    outputs = []
    for index in (0, 1):
        device = torch.device(f"cuda:{index}")
        model = _model().to(device)
        region = replace(
            _region(scale=2.0, digest=DIGEST),
            conditioning=Conditioning(torch.zeros((1, 3, 2), device=device)),
            scale_vector=torch.tensor([2.0], device=device),
        )
        for _ in range(2):
            result = evaluate_grouped_regions(
                (region,),
                (),
                torch.ones((1, 1, 2, 2), device=device),
                0.5,
                FLOW,
                SD15,
                model,
                _evaluate(model, []),
                {DIGEST: _patch()},
                lambda: False,
                compute_dtype=torch.float32,
            )
            outputs.append(result.conditional.cpu())
            assert result.staged_bytes == 4
            assert not model._forward_hooks
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
