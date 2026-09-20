"""Scheduled runtime authority, sampling, and cleanup proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLUX_DEV,
    FLUX_SCHNELL,
    SD15,
    SDXL_REFINER,
    AdapterPatch,
    Conditioning,
    CustomSamplingRequest,
    DiffPatch,
    DiscreteSigmas,
    FluxFlowSigmas,
    GuidanceContribution,
    GuidancePostCFGDescriptor,
    GuidanceRole,
    KeyedContribution,
    PatchEntry,
    PatchSet,
    PatchTargetComponent,
    PercentRange,
    SamplerDescriptor,
    SamplerInfo,
    SamplingCancelled,
    SamplingGuidance,
    StepEvent,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    encode_conditioning_carrier,
)
from dinkster_inference_torch import (
    INFERENCE_PATCH_PROVIDERS_SURFACE,
    FluxRuntime,
    LoRAAdapter,
    PatchProviderSnapshot,
    PreparedGroupedPatches,
    ScheduledPatchResolution,
    ScheduledPatchResolutionRequest,
    ScheduledSamplingError,
    ScheduledSamplingOptions,
    SDRuntime,
    torch_sampler_registry,
    torch_scheduler_registry,
)
from dinkster_inference_torch import scheduled_sampling as scheduled_module
from dinkster_inference_torch.cfg import cfg_combine
from dinkster_inference_torch.denoise import DenoiseError
from dinkster_inference_torch.scheduled import _carrier
from dinkster_inference_torch.scheduled_sampling import (
    ScheduledConditioningDenoiser,
    _resolve_patch_sets,
)
from dinkster_inference_torch.wiring import WiringError
from test_denoise import tiny_cond as flux_cond
from test_denoise import tiny_flux
from test_denoise import tiny_latent as flux_latent
from test_regional import _carrier as regional_carrier
from test_sd_denoise import TINY_ADM, tiny_adm, tiny_unet
from test_sd_denoise import tiny_cond as sd_cond
from test_sd_denoise import tiny_latent as sd_latent

FULL = PercentRange(0.0, 1.0)
OVERLAY = "a" * 64
STACK = hashlib.sha256(json.dumps([OVERLAY], separators=(",", ":")).encode()).hexdigest()
GOLDEN = json.loads((Path(__file__).parent / "goldens/scheduled_sampling_goldens.json").read_text())
PERFORMANCE = json.loads(
    (Path(__file__).parent / "performance/scheduled_runtime_blackwell.json").read_text()
)
HISTORICAL_MEASURED_COMMIT = "0afd90dc7498fd922765eee9b3cf4739cad8f47c"
HISTORICAL_MEASURED_SOURCE_SHA256 = (
    "sha256:b6637772d36bf37e00c43e7422ba2c68aa3d82b9d34be6a0ba90b49bf5ad9b71"
)


def _metadata(family_id: str) -> tuple[tuple[str, object], ...]:
    effective = hashlib.sha256(
        json.dumps(
            {
                "version": 1,
                "target": family_id,
                "text": None,
                "diffusion": STACK,
                "transforms": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return (
        ("dinkster.inference/version", 1),
        ("dinkster.inference/target", family_id),
        ("dinkster.inference/text-overlay-digests", ()),
        ("dinkster.inference/diffusion-overlay-digests", (OVERLAY,)),
        ("dinkster.inference/transform-ids", ()),
        ("dinkster.inference/transform-digests", ()),
        ("dinkster.inference/effective-patch-state", effective),
        ("dinkster.inference/diffusion-overlay-stack-digest", STACK),
    )


def _conditioning_carrier(
    family_id: str,
    conditioning: Conditioning[torch.Tensor],
    *,
    patched: bool = False,
):
    if family_id == FLUX_DEV.id:
        streams = ("clip_l", "t5")
        segments = (TokenSegmentDescriptor("t5", "t5", 0, conditioning.embeddings.shape[1]),)
    else:
        streams = ("clip_l",)
        segments = (
            TokenSegmentDescriptor("clip_l", "clip_l", 0, conditioning.embeddings.shape[1]),
        )
    layout = TokenLayoutDescriptor(family_id, 1, streams, segments)
    return _carrier(
        ((conditioning, FULL, layout, cast(Any, _metadata(family_id) if patched else ())),)
    )


def _runtime(kind: str):
    if kind == "flux":
        runtime = object.__new__(FluxRuntime)
        runtime.assembled = cast(Any, SimpleNamespace(family=FLUX_DEV, diffusion=tiny_flux()))
        runtime._runtime_identity = "native:test:scheduled-flux"
        runtime._space = FluxFlowSigmas(shift=FLUX_DEV.sampling.shift)
        runtime._samplers = torch_sampler_registry()
        runtime._schedulers = torch_scheduler_registry()
        runtime._guidance = None
        return runtime
    runtime = object.__new__(SDRuntime)
    model = tiny_unet()
    runtime.assembled = cast(Any, SimpleNamespace(family=SD15, diffusion=model))
    runtime._runtime_identity = "native:test:scheduled-sd"
    runtime._samplers = torch_sampler_registry()
    runtime._schedulers = torch_scheduler_registry()
    runtime._guidance = None
    runtime.sampling = SD15.sampling
    runtime._space = DiscreteSigmas.linear_beta(zsnr=False)
    runtime._percent_to_sigma = runtime._space.percent_to_sigma
    return runtime


def _zero_patch(model: torch.nn.Module) -> PatchSet[torch.Tensor]:
    for name, module in model.named_modules():
        if type(module).__name__ in ("Linear", "_InitlessLinear", "_CastLinear"):
            linear = cast(torch.nn.Linear, module)
            key = f"{name}.weight" if name else "weight"
            return PatchSet(
                {key: (PatchEntry(DiffPatch(torch.zeros_like(linear.weight))),)},
                structural_digest=STACK,
            )
    raise AssertionError("test model has no plain Linear")


def _nonzero_patch(model: torch.nn.Module) -> PatchSet[torch.Tensor]:
    for name, module in model.named_modules():
        if type(module).__name__ in ("Linear", "_InitlessLinear", "_CastLinear"):
            linear = cast(torch.nn.Linear, module)
            key = f"{name}.weight" if name else "weight"
            return PatchSet(
                {key: (PatchEntry(DiffPatch(torch.ones_like(linear.weight) * 0.125)),)},
                structural_digest=STACK,
            )
    raise AssertionError("test model has no plain Linear")


def _resolver(model: torch.nn.Module, calls: list[object]):
    patch_set = _zero_patch(model)
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "test.provider")
    snapshot = PatchProviderSnapshot((declaration,))

    def resolve(
        requests: tuple[ScheduledPatchResolutionRequest, ...],
        cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        calls.append((requests, cancel))
        return tuple(
            ScheduledPatchResolution(
                request,
                "worker-generation-1",
                snapshot,
                declaration.id,
                declaration,
                patch_set,
            )
            for request in requests
        )

    return resolve


def _patch_resolver(model: torch.nn.Module, calls: list[object]):
    patch_set = _nonzero_patch(model)
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "test.provider")
    snapshot = PatchProviderSnapshot((declaration,))

    def resolve(
        requests: tuple[ScheduledPatchResolutionRequest, ...],
        cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        calls.append((requests, cancel))
        return tuple(
            ScheduledPatchResolution(
                request,
                "worker-generation-1",
                snapshot,
                declaration.id,
                declaration,
                patch_set,
            )
            for request in requests
        )

    return resolve


def _conv_lora_resolver(model: torch.nn.Module, calls: list[object]):
    for name, module in model.named_modules():
        if type(module).__name__ in ("Conv2d", "_InitlessConv2d", "_CastConv2d"):
            conv = cast(torch.nn.Conv2d, module)
            key = f"{name}.weight" if name else "weight"
            up = torch.full((conv.out_channels, 1, 1, 1), 0.05)
            down = torch.full((1, conv.in_channels, *conv.kernel_size), -0.025)
            patch_set = PatchSet(
                {key: (PatchEntry(AdapterPatch(LoRAAdapter(up, down))),)},
                structural_digest=STACK,
            )
            break
    else:
        raise AssertionError("test model has no plain Conv2d")
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "test.provider")
    snapshot = PatchProviderSnapshot((declaration,))

    def resolve(
        requests: tuple[ScheduledPatchResolutionRequest, ...],
        cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        calls.append((requests, cancel))
        return tuple(
            ScheduledPatchResolution(
                request,
                "worker-generation-1",
                snapshot,
                declaration.id,
                declaration,
                patch_set,
            )
            for request in requests
        )

    return resolve


def test_public_frozen_request_and_resolution_bind_every_authority_identity() -> None:
    request = ScheduledPatchResolutionRequest(
        "b" * 64,
        "runtime",
        PatchTargetComponent.DIFFUSION,
        (OVERLAY,),
        STACK,
    )
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "provider")
    patch_set = PatchSet({}, structural_digest=STACK)
    resolution = ScheduledPatchResolution(
        request,
        "generation",
        PatchProviderSnapshot((declaration,)),
        "provider",
        declaration,
        patch_set,
    )
    assert resolution.patch_set is patch_set
    with pytest.raises((TypeError, ValueError)):
        replace(request, target=PatchTargetComponent.TEXT)
    with pytest.raises((TypeError, ValueError)):
        replace(resolution, generation_key="")
    with pytest.raises((TypeError, ValueError)):
        replace(resolution, provider_id="other")


def test_pinned_comfy_cfg_facts_match_scheduled_combination_exactly() -> None:
    assert GOLDEN["reference"]["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    cond = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.float32)
    uncond = torch.tensor([[[[-0.5, 0.25], [1.5, -2.0]]]], dtype=torch.float32)
    for case in GOLDEN["cases"]:
        actual = cfg_combine(cond, uncond, case["scale"])
        assert tuple(actual.shape) == tuple(case["shape"])
        assert actual.dtype is torch.float32
        assert actual.reshape(-1).tolist() == case["data"]


def test_historical_performance_record_pins_measured_source_and_all_checks() -> None:
    assert PERFORMANCE["implementation_source_sha256"] == HISTORICAL_MEASURED_SOURCE_SHA256, (
        f"schema-1 record must remain bound to measured commit {HISTORICAL_MEASURED_COMMIT}"
    )
    assert PERFORMANCE["overall_pass"] is True
    assert PERFORMANCE["checks"]
    assert all(value is True for value in PERFORMANCE["checks"].values())


@pytest.mark.parametrize("bad", ([], {}, None, (object(),)))
def test_resolver_refuses_non_tuple_and_wrong_result_types_without_patch_authority(
    bad: object,
) -> None:
    request = ScheduledPatchResolutionRequest(
        "b" * 64, "runtime", PatchTargetComponent.DIFFUSION, (OVERLAY,), STACK
    )
    with pytest.raises(ScheduledSamplingError, match="resolution-type"):
        _resolve_patch_sets(
            (request,),
            cast(Any, lambda _requests, _cancel: bad),  # pyright: ignore[reportUnknownLambdaType]
            lambda: False,
        )


def test_resolver_refuses_missing_duplicate_generation_snapshot_and_patch_identity() -> None:
    first = ScheduledPatchResolutionRequest(
        "b" * 64, "runtime", PatchTargetComponent.DIFFUSION, (OVERLAY,), STACK
    )
    second = replace(first, carrier_sha256="c" * 64)
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "provider")
    snapshot = PatchProviderSnapshot((declaration,))
    patch = PatchSet({}, structural_digest=STACK)

    def item(
        request: ScheduledPatchResolutionRequest,
        generation: str = "generation",
        patch_set: PatchSet[torch.Tensor] = patch,
    ) -> ScheduledPatchResolution:
        return ScheduledPatchResolution(
            request, generation, snapshot, "provider", declaration, patch_set
        )

    with pytest.raises(ScheduledSamplingError, match="resolution-set"):
        _resolve_patch_sets(
            (first,),
            cast(Any, lambda _requests, _cancel: ()),  # pyright: ignore[reportUnknownLambdaType]
            lambda: False,
        )
    with pytest.raises(ScheduledSamplingError, match="resolution-duplicate"):
        _resolve_patch_sets(
            (first,),
            cast(Any, lambda _requests, _cancel: (item(first), item(first))),  # pyright: ignore[reportUnknownLambdaType]
            lambda: False,
        )
    with pytest.raises(ScheduledSamplingError, match="resolution-generation"):
        _resolve_patch_sets(
            (first, second),
            cast(Any, lambda _requests, _cancel: (item(first), item(second, "other"))),  # pyright: ignore[reportUnknownLambdaType]
            lambda: False,
        )
    with pytest.raises(ScheduledSamplingError, match="resolution-patch-identity"):
        _resolve_patch_sets(
            (first, second),
            cast(
                Any,
                lambda _requests, _cancel: (  # pyright: ignore[reportUnknownLambdaType]
                    item(first),
                    item(second, patch_set=PatchSet({}, structural_digest=STACK)),
                ),
            ),
            lambda: False,
        )


def test_resolver_requires_exact_echo_but_accepts_equal_snapshot_and_declaration() -> None:
    first = ScheduledPatchResolutionRequest(
        "b" * 64, "runtime", PatchTargetComponent.DIFFUSION, (OVERLAY,), STACK
    )
    second = replace(first, carrier_sha256="c" * 64)
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "provider")
    patch = PatchSet({}, structural_digest=STACK)

    def replacement(
        _requests: tuple[ScheduledPatchResolutionRequest, ...],
        _cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        equal_request = replace(first)
        return (
            ScheduledPatchResolution(
                equal_request,
                "generation",
                PatchProviderSnapshot((declaration,)),
                "provider",
                declaration,
                patch,
            ),
        )

    with pytest.raises(ScheduledSamplingError, match="resolution-request"):
        _resolve_patch_sets((first,), cast(Any, replacement), lambda: False)

    def equal_authority(
        _requests: tuple[ScheduledPatchResolutionRequest, ...],
        _cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        return tuple(
            ScheduledPatchResolution(
                request,
                "generation",
                PatchProviderSnapshot((replace(declaration),)),
                "provider",
                replace(declaration),
                patch,
            )
            for request in (first, second)
        )

    resolved = _resolve_patch_sets((first, second), cast(Any, equal_authority), lambda: False)
    assert resolved[STACK] is patch


def test_resolver_allows_distinct_stacks_to_use_distinct_snapshot_providers() -> None:
    other_overlay = "d" * 64
    other_stack = hashlib.sha256(
        json.dumps([other_overlay], separators=(",", ":")).encode()
    ).hexdigest()
    first = ScheduledPatchResolutionRequest(
        "b" * 64, "runtime", PatchTargetComponent.DIFFUSION, (OVERLAY,), STACK
    )
    second = ScheduledPatchResolutionRequest(
        "c" * 64,
        "runtime",
        PatchTargetComponent.DIFFUSION,
        (other_overlay,),
        other_stack,
    )
    first_declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "first")
    second_declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "second")
    snapshot = PatchProviderSnapshot((first_declaration, second_declaration))
    first_patch = PatchSet({}, structural_digest=STACK)
    second_patch = PatchSet({}, structural_digest=other_stack)

    def resolve(
        _requests: tuple[ScheduledPatchResolutionRequest, ...],
        _cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        return (
            ScheduledPatchResolution(
                first,
                "generation",
                snapshot,
                first_declaration.id,
                first_declaration,
                first_patch,
            ),
            ScheduledPatchResolution(
                second,
                "generation",
                snapshot,
                second_declaration.id,
                second_declaration,
                second_patch,
            ),
        )

    resolved = _resolve_patch_sets((first, second), cast(Any, resolve), lambda: False)
    assert resolved[STACK] is first_patch
    assert resolved[other_stack] is second_patch


def test_resolver_refuses_distinct_authorities_for_the_same_stack() -> None:
    first = ScheduledPatchResolutionRequest(
        "b" * 64, "runtime", PatchTargetComponent.DIFFUSION, (OVERLAY,), STACK
    )
    second = replace(first, carrier_sha256="c" * 64)
    first_declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "first")
    second_declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "second")
    snapshot = PatchProviderSnapshot((first_declaration, second_declaration))
    patch = PatchSet({}, structural_digest=STACK)

    with pytest.raises(ScheduledSamplingError, match="resolution-provider"):
        _resolve_patch_sets(
            (first, second),
            cast(
                Any,
                lambda _requests, _cancel: (  # pyright: ignore[reportUnknownLambdaType]
                    ScheduledPatchResolution(
                        first,
                        "generation",
                        snapshot,
                        first_declaration.id,
                        first_declaration,
                        patch,
                    ),
                    ScheduledPatchResolution(
                        second,
                        "generation",
                        snapshot,
                        second_declaration.id,
                        second_declaration,
                        patch,
                    ),
                ),
            ),
            lambda: False,
        )


def test_flux_and_sd_scheduled_ordinary_carriers_are_byte_exact_to_ordinary_sample() -> None:
    flux = _runtime("flux")
    latent = flux_latent()
    cond = flux_cond("cond")
    uncond = flux_cond("uncond")
    ordinary = flux.sample(
        latent,
        cond=cond,
        cfg=SamplingGuidance(uncond, 2.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=7,
        compute_dtype=torch.float32,
    )
    scheduled = flux.sample_scheduled(
        latent,
        cond=_conditioning_carrier(FLUX_DEV.id, cond),
        cfg=SamplingGuidance(_conditioning_carrier(FLUX_DEV.id, uncond), 2.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=7,
        compute_dtype=torch.float32,
    )
    assert torch.equal(scheduled, ordinary)

    sd = _runtime("sd")
    sd_input = sd_latent()
    sd_positive = sd_cond("cond")
    sd_negative = sd_cond("uncond")
    ordinary_sd = sd.sample(
        sd_input,
        cond=sd_positive,
        cfg=SamplingGuidance(sd_negative, 2.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=11,
        compute_dtype=torch.float32,
    )
    scheduled_sd = sd.sample_scheduled(
        sd_input,
        cond=_conditioning_carrier(SD15.id, sd_positive),
        cfg=SamplingGuidance(_conditioning_carrier(SD15.id, sd_negative), 2.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=11,
        compute_dtype=torch.float32,
    )
    assert torch.equal(scheduled_sd, ordinary_sd)


@pytest.mark.parametrize("kind", ("flux", "sd"))
def test_scheduled_facade_executes_through_sample_custom(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(kind)
    latent = flux_latent() if kind == "flux" else sd_latent()
    family_id = FLUX_DEV.id if kind == "flux" else SD15.id
    conditioning = flux_cond("facade") if kind == "flux" else sd_cond("facade")
    carrier = _conditioning_carrier(family_id, conditioning)
    original = runtime.sample_custom
    calls: list[dict[str, object]] = []
    states: list[object] = []

    def on_state(event: object) -> None:
        states.append(event)

    def capture(latent_value: torch.Tensor, **kwargs: object):
        calls.append(kwargs)
        return original(latent_value, **cast(Any, kwargs))

    monkeypatch.setattr(runtime, "sample_custom", capture)
    result = runtime.sample_scheduled(
        latent,
        cond=carrier,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        seed=19,
        compute_dtype=torch.float32,
        on_state=on_state,
    )

    assert result.shape == latent.shape
    assert len(calls) == 1
    assert calls[0]["cond"] is carrier
    assert calls[0]["on_state"] is on_state
    assert type(calls[0]["scheduled"]) is ScheduledSamplingOptions
    assert states


@pytest.mark.parametrize("kind", ("flux", "sd"))
def test_missing_uncond_preserves_ordinary_cfg_behavior(kind: str) -> None:
    runtime = _runtime(kind)
    if kind == "flux":
        latent = flux_latent()
        conditioning = flux_cond("cond-only")
        family_id = FLUX_DEV.id
    else:
        latent = sd_latent()
        conditioning = sd_cond("cond-only")
        family_id = SD15.id
    ordinary = runtime.sample(
        latent,
        cond=conditioning,
        cfg=SamplingGuidance(scale=2.5),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=13,
        compute_dtype=torch.float32,
    )
    scheduled = runtime.sample_scheduled(
        latent,
        cond=_conditioning_carrier(family_id, conditioning),
        cfg=SamplingGuidance(scale=2.5),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=13,
        compute_dtype=torch.float32,
    )
    assert torch.equal(scheduled, ordinary)


def test_patch_resolution_is_eager_once_and_zero_patch_preserves_output() -> None:
    runtime = _runtime("flux")
    latent = flux_latent()
    cond = flux_cond("cond")
    carrier = _conditioning_carrier(FLUX_DEV.id, cond, patched=True)
    calls: list[object] = []
    result = runtime.sample_scheduled(
        latent,
        cond=carrier,
        resolver=_resolver(runtime.assembled.diffusion, calls),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=3,
        compute_dtype=torch.float32,
    )
    baseline = runtime.sample_scheduled(
        latent,
        cond=_conditioning_carrier(FLUX_DEV.id, cond),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=3,
        compute_dtype=torch.float32,
    )
    assert torch.equal(result, baseline)
    assert len(calls) == 1
    requests, cancel = cast(Any, calls[0])
    assert (
        requests[0].carrier_sha256
        == hashlib.sha256(encode_conditioning_carrier(carrier)).hexdigest()
    )
    assert requests[0].runtime_identity == runtime.runtime_identity
    assert requests[0].overlay_digests == (OVERLAY,)
    assert requests[0].stack_digest == STACK
    assert cancel is not None
    assert not runtime.assembled.diffusion._forward_hooks


def test_sd_scheduled_sampling_evaluates_convolution_lora_and_cleans_hooks() -> None:
    runtime = _runtime("sd")
    model = runtime.assembled.diffusion
    latent = sd_latent()
    conditioning = sd_cond("conv-lora")
    calls: list[object] = []
    arguments: dict[str, object] = {
        "cond": _conditioning_carrier(SD15.id, conditioning, patched=True),
        "resolver": _conv_lora_resolver(model, calls),
        "sampler_id": "dinkster.euler",
        "scheduler_id": "dinkster.normal",
        "steps": 2,
        "seed": 29,
        "compute_dtype": torch.float32,
    }
    first = runtime.sample_scheduled(latent, **cast(Any, arguments))
    second = runtime.sample_scheduled(latent, **cast(Any, arguments))
    baseline = runtime.sample_scheduled(
        latent,
        cond=_conditioning_carrier(SD15.id, conditioning),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=2,
        seed=29,
        compute_dtype=torch.float32,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, baseline)
    assert len(calls) == 2
    assert not any(module._forward_hooks for module in model.modules())


def test_sample_scheduled_condition_scale_dynamic_uncond_and_stage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    cond_value = flux_cond("scaled-cond")
    uncond_value = flux_cond("scaled-uncond")
    metadata = _metadata(FLUX_DEV.id)
    conditional = regional_carrier(
        (
            {
                "text": cond_value.embeddings,
                "pooled": cond_value.pooled,
                "scale": torch.tensor([2.0], dtype=torch.float32),
                "schedule": PercentRange(0.0, 1.0),
                "extension_metadata": metadata,
            },
        ),
        family=FLUX_DEV.id,
        tokens=cond_value.embeddings.shape[1],
        features=cond_value.embeddings.shape[2],
    )
    unconditional = regional_carrier(
        (
            {
                "text": uncond_value.embeddings,
                "pooled": uncond_value.pooled,
                "scale": torch.tensor([3.0], dtype=torch.float32),
                "schedule": PercentRange(0.0, 0.5),
                "extension_metadata": metadata,
            },
        ),
        family=FLUX_DEV.id,
        tokens=uncond_value.embeddings.shape[1],
        features=uncond_value.embeddings.shape[2],
    )
    from dinkster_inference_torch.wiring import _flux_sigma_space

    space = _flux_sigma_space(FLUX_DEV)
    active_sigma = space.percent_to_sigma(0.25)
    inactive_sigma = space.percent_to_sigma(0.75)
    lanes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def make(_options: Mapping[str, Any]):
        def solve(
            denoiser: Any,
            x: torch.Tensor,
            _sigmas: Sequence[float],
            _info: SamplerInfo,
            *,
            noise: Any = None,
            on_step: Any = None,
            on_step_begin: Any = None,
        ) -> torch.Tensor:
            del noise, on_step, on_step_begin
            lanes.append(denoiser.call_with_uncond(x, active_sigma))
            lanes.append(denoiser.call_with_uncond(x, inactive_sigma))
            return x

        return solve

    runtime._samplers = torch_sampler_registry(
        (
            SamplerDescriptor(
                "test.condition-scale", "Condition scale", cast(Any, make), needs_uncond=True
            ),
        )
    )
    prepare_calls: list[object] = []
    prepare = scheduled_module.prepare_grouped_patches

    def counted_prepare(*args: Any, **kwargs: Any):
        owner = prepare(*args, **kwargs)
        prepare_calls.append(owner)
        return owner

    monkeypatch.setattr(scheduled_module, "prepare_grouped_patches", counted_prepare)
    resolver_calls: list[object] = []
    result = runtime.sample_scheduled(
        flux_latent(),
        cond=conditional,
        cfg=SamplingGuidance(unconditional, 2.0),
        resolver=_patch_resolver(model, resolver_calls),
        sampler_id="test.condition-scale",
        scheduler_id="dinkster.normal",
        steps=2,
        compute_dtype=torch.float32,
    )
    assert result.shape == flux_latent().shape
    assert len(resolver_calls) == 1
    assert len(prepare_calls) == 1
    assert cast(Any, prepare_calls[0]).staged_bytes > 0
    assert len(lanes) == 2
    assert torch.count_nonzero(lanes[0][1]) > 0
    assert torch.equal(lanes[1][1], torch.zeros_like(lanes[1][1]))
    assert not any(module._forward_hooks for module in model.modules())


def test_cfg1_skips_uncond_materialization_but_cfgpp_forces_it() -> None:
    runtime = _runtime("flux")
    cond = _conditioning_carrier(FLUX_DEV.id, flux_cond("cond"))
    # This invalid object proves the cfg==1 lane is not materialized.
    result = runtime.sample_scheduled(
        flux_latent(),
        cond=cond,
        cfg=SamplingGuidance(cast(Any, object()), 1.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        compute_dtype=torch.float32,
    )
    assert result.shape == flux_latent().shape
    with pytest.raises(ScheduledSamplingError, match="carrier"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=cond,
            cfg=SamplingGuidance(cast(Any, object()), 1.0),
            sampler_id="dinkster.euler_cfg_pp",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )


def test_cfgpp_executes_call_with_uncond_and_cleans_sample_owner() -> None:
    runtime = _runtime("sd")
    result = runtime.sample_scheduled(
        sd_latent(),
        cond=_conditioning_carrier(SD15.id, sd_cond("cond")),
        cfg=SamplingGuidance(_conditioning_carrier(SD15.id, sd_cond("uncond")), 1.0),
        sampler_id="dinkster.euler_cfg_pp",
        scheduler_id="dinkster.normal",
        steps=2,
        compute_dtype=torch.float32,
    )
    assert result.shape == sd_latent().shape
    assert not runtime.assembled.diffusion._forward_hooks


def test_sample_scheduled_adm_family_preserves_cond_uncond_polarity() -> None:
    runtime = _runtime("sd")
    cast(Any, runtime).assembled.family = SDXL_REFINER
    cast(Any, runtime).assembled.diffusion = tiny_unet(TINY_ADM)
    cast(Any, runtime).sampling = SDXL_REFINER.sampling
    calls: list[bool] = []

    def resolve_adm(
        _conditioning: Conditioning[torch.Tensor],
        _latent: torch.Tensor,
        *,
        negative: bool,
    ) -> torch.Tensor:
        calls.append(negative)
        return tiny_adm("negative" if negative else "positive")

    cast(Any, runtime)._adm = resolve_adm
    cond_value = sd_cond("adm-cond", features=TINY_ADM.context_dim)
    uncond_value = sd_cond("adm-uncond", features=TINY_ADM.context_dim)
    cond = regional_carrier(
        ({"text": cond_value.embeddings, "pooled": torch.zeros((1, 8))},),
        family=SDXL_REFINER.id,
        tokens=cond_value.embeddings.shape[1],
        features=cond_value.embeddings.shape[2],
    )
    uncond = regional_carrier(
        ({"text": uncond_value.embeddings, "pooled": torch.ones((1, 8))},),
        family=SDXL_REFINER.id,
        tokens=uncond_value.embeddings.shape[1],
        features=uncond_value.embeddings.shape[2],
    )
    result = runtime.sample_scheduled(
        sd_latent(),
        cond=cond,
        cfg=SamplingGuidance(uncond, 2.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        compute_dtype=torch.float32,
    )
    assert result.shape == sd_latent().shape
    assert calls == [True, False]


def test_sample_scheduled_sd15_installs_no_adm_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("sd")
    observed: list[object] = []
    grouped = scheduled_module.sd_grouped_region_evaluator

    def inspect(*args: Any, **kwargs: Any):
        observed.append(kwargs.get("adm"))
        return grouped(*args, **kwargs)

    monkeypatch.setattr(scheduled_module, "sd_grouped_region_evaluator", inspect)
    runtime.sample_scheduled(
        sd_latent(),
        cond=_conditioning_carrier(SD15.id, sd_cond("cond")),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        compute_dtype=torch.float32,
    )
    assert observed == [None]


def test_empty_schedule_materializes_and_resolves_but_never_stages_or_models() -> None:
    runtime = _runtime("flux")
    latent = flux_latent()
    calls: list[object] = []
    forwards: list[object] = []
    handle = runtime.assembled.diffusion.register_forward_pre_hook(
        lambda _module, args: forwards.append(args)
    )
    try:
        result = runtime.sample_scheduled(
            latent,
            cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond"), patched=True),
            resolver=_resolver(runtime.assembled.diffusion, calls),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            denoise=0.0,
            compute_dtype=torch.float32,
        )
    finally:
        handle.remove()
    assert result is latent
    assert len(calls) == 1
    assert forwards == []


def test_cancellation_is_exact_bool_and_progress_checks_before_and_after() -> None:
    runtime = _runtime("flux")
    carrier = _conditioning_carrier(FLUX_DEV.id, flux_cond("cond"))
    with pytest.raises(ScheduledSamplingError, match="cancel-callback"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=carrier,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            cancelled=cast(Any, lambda: 0),
            compute_dtype=torch.float32,
        )
    state = {"cancel": False, "progress": 0}

    def cancelled() -> bool:
        return state["cancel"] is True

    def progress(_event: StepEvent) -> None:
        state["progress"] += 1
        state["cancel"] = True

    with pytest.raises(SamplingCancelled, match="cancelled"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=carrier,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            cancelled=cancelled,
            on_step=progress,
            compute_dtype=torch.float32,
        )
    assert state["progress"] == 1
    assert not runtime.assembled.diffusion._forward_hooks


def test_model_error_preserves_primary_over_close_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    original = model.forward
    original_close = PreparedGroupedPatches.close

    def close_then_fail(owner: PreparedGroupedPatches) -> None:
        original_close(owner)
        raise RuntimeError("close-secondary")

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("model-primary")

    monkeypatch.setattr(PreparedGroupedPatches, "close", close_then_fail)
    cast(Any, model).forward = fail
    try:
        with pytest.raises(RuntimeError, match="model-primary"):
            runtime.sample_scheduled(
                flux_latent(),
                cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond"), patched=True),
                resolver=_resolver(model, []),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float32,
            )
    finally:
        cast(Any, model).forward = original
    assert not any(module._forward_hooks for module in model.modules())


def test_progress_primary_wins_close_error_and_fresh_retry_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    carrier = _conditioning_carrier(FLUX_DEV.id, flux_cond("cond"), patched=True)
    original_close = PreparedGroupedPatches.close
    close_calls: list[object] = []

    def close_then_fail(owner: PreparedGroupedPatches) -> None:
        close_calls.append(owner)
        original_close(owner)
        raise RuntimeError("close-secondary")

    monkeypatch.setattr(PreparedGroupedPatches, "close", close_then_fail)

    def progress_failure(_event: StepEvent) -> None:
        raise RuntimeError("progress-primary")

    with pytest.raises(RuntimeError, match="progress-primary"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=carrier,
            resolver=_patch_resolver(model, []),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            on_step=progress_failure,
            compute_dtype=torch.float32,
        )
    assert len(close_calls) == 1
    assert not any(module._forward_hooks for module in model.modules())

    monkeypatch.undo()
    retried = runtime.sample_scheduled(
        flux_latent(),
        cond=carrier,
        resolver=_patch_resolver(model, []),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        compute_dtype=torch.float32,
    )
    assert retried.shape == flux_latent().shape
    assert not any(module._forward_hooks for module in model.modules())


def test_resolver_error_precedes_staging_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    prepare_calls: list[object] = []
    model_calls: list[object] = []
    prepare = scheduled_module.prepare_grouped_patches

    def observe_prepare(*args: Any, **kwargs: Any):
        prepare_calls.append(args)
        return prepare(*args, **kwargs)

    handle = model.register_forward_pre_hook(lambda _module, args: model_calls.append(args))
    monkeypatch.setattr(scheduled_module, "prepare_grouped_patches", observe_prepare)

    def fail_resolver(
        _requests: tuple[ScheduledPatchResolutionRequest, ...],
        _cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        raise RuntimeError("resolver-primary")

    try:
        with pytest.raises(RuntimeError, match="resolver-primary"):
            runtime.sample_scheduled(
                flux_latent(),
                cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond"), patched=True),
                resolver=cast(Any, fail_resolver),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float32,
            )
    finally:
        handle.remove()
    assert prepare_calls == []
    assert model_calls == []


def test_cleanup_only_error_surfaces_after_resources_are_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    original_close = PreparedGroupedPatches.close

    def close_then_fail(owner: PreparedGroupedPatches) -> None:
        original_close(owner)
        raise RuntimeError("cleanup-only")

    monkeypatch.setattr(PreparedGroupedPatches, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="cleanup-only"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond"), patched=True),
            resolver=_patch_resolver(model, []),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )
    assert not any(module._forward_hooks for module in model.modules())


def test_context_solver_uses_same_cancel_object_and_cancels_after_progress() -> None:
    runtime = _runtime("flux")
    observed: list[object] = []

    def make_context(_options: Mapping[str, Any]):
        def solve(
            denoiser: Any,
            x: torch.Tensor,
            context: Any,
            _info: SamplerInfo,
            *,
            noise: Any = None,
        ) -> torch.Tensor:
            del noise
            observed.append(context.cancellation.cancelled)
            prediction = denoiser(x, context.sigma_schedule[0], outer_step=0)
            context.progress.report(StepEvent(0, 1, context.sigma_schedule[0]))
            return prediction.value

        return solve

    runtime._samplers = torch_sampler_registry(
        (
            SamplerDescriptor(
                "test.context",
                "Context test",
                cast(Any, make_context),
                context_aware=True,
            ),
        )
    )
    state = {"cancel": False}

    def cancelled() -> bool:
        return state["cancel"]

    def progress(_event: StepEvent) -> None:
        state["cancel"] = True

    with pytest.raises(SamplingCancelled, match="cancelled"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond")),
            sampler_id="test.context",
            scheduler_id="dinkster.normal",
            steps=1,
            cancelled=cancelled,
            on_step=progress,
            compute_dtype=torch.float32,
        )
    assert observed == [cancelled]
    assert not runtime.assembled.diffusion._forward_hooks


@pytest.mark.parametrize("kind", ("sd", "flux"))
def test_scheduled_sampling_uses_shared_distributed_evaluator(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module

    runtime = _runtime(kind)
    latent = sd_latent() if kind == "sd" else flux_latent()
    cond = sd_cond("cond") if kind == "sd" else flux_cond("cond")
    family = SD15 if kind == "sd" else FLUX_DEV

    def sample() -> torch.Tensor:
        return runtime.sample_scheduled(
            latent,
            cond=_conditioning_carrier(family.id, cond),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )

    baseline = sample()
    config = distributed_module.DistributedSamplingConfig(
        0, 2, "guidance", "file:///group", "1" * 32
    )
    monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: config)
    monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: config)
    calls: list[object] = []

    def evaluate(evaluator: Any, x: Any, sigma: Any, request: Any) -> Any:
        calls.append(request)
        return evaluator.evaluate(x, sigma, request)

    monkeypatch.setattr(
        distributed_module.DistributedGuidanceEvaluator, "evaluate_request", evaluate
    )
    assert torch.equal(sample(), baseline)
    assert calls


def test_active_guidance_refuses_before_invalid_carrier_materialization() -> None:
    runtime = _runtime("flux")
    cast(Any, runtime)._guidance = SimpleNamespace(registry=SimpleNamespace(active=True))
    with pytest.raises(ScheduledSamplingError, match="guidance-extensions"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=cast(Any, object()),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )


def test_invalid_unconditional_carrier_refuses_before_brownian_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime("flux")
    latent = flux_latent()
    sampler = runtime._samplers.get("dinkster.euler")
    assert sampler is not None
    request = CustomSamplingRequest(sampler, (), (1.0, 0.0))
    calls: list[object] = []

    def allocate(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(scheduled_module, "brownian_step_noise", allocate)
    with pytest.raises(ScheduledSamplingError, match="guidance-carrier"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("cond")),
            cfg=SamplingGuidance(cast(Any, object()), 2.0),
            request=request,
            compute_dtype=torch.float32,
        )
    assert calls == []


def test_empty_non_tuple_ipadapter_input_refuses_in_scheduled_engine() -> None:
    runtime = cast(SDRuntime, _runtime("sd"))
    latent = sd_latent()
    sampler = runtime._samplers.get("dinkster.euler")
    assert sampler is not None

    with pytest.raises(ScheduledSamplingError, match="attention-contributions"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=_conditioning_carrier(SD15.id, sd_cond("cond")),
            cfg=None,
            request=CustomSamplingRequest(sampler, (), (1.0, 0.0)),
            sd15_attention_contributions=cast(Any, []),
            compute_dtype=torch.float32,
        )


def test_per_run_guidance_transforms_refuse_at_both_scheduled_entrypoints() -> None:
    contribution: GuidanceContribution[torch.Tensor] = GuidanceContribution(
        post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
    )
    cfg = SamplingGuidance(scale=1.0, transforms=(("run", contribution),))
    for family in ("flux", "sd"):
        runtime = _runtime(family)
        latent = flux_latent() if family == "flux" else sd_latent()
        with pytest.raises(ScheduledSamplingError, match="guidance-extensions"):
            runtime.sample_scheduled(
                latent,
                cond=cast(Any, object()),
                cfg=cfg,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
            )


def test_attention_guidance_refuses_at_both_scheduled_entrypoints() -> None:
    from dinkster_inference_torch import guidance_transforms

    cfg = SamplingGuidance(
        scale=1.0,
        transforms=(("dinkster.nag", guidance_transforms.nag(5.0, 0.5, 1.5)),),
    )
    for family in ("flux", "sd"):
        runtime = _runtime(family)
        latent = flux_latent() if family == "flux" else sd_latent()
        with pytest.raises(ScheduledSamplingError, match="guidance-extensions"):
            runtime.sample_scheduled(
                latent,
                cond=cast(Any, object()),
                cfg=cfg,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
            )


def test_schnell_guidance_refuses_before_carrier_or_resolver_work() -> None:
    runtime = _runtime("flux")
    cast(Any, runtime).assembled.family = FLUX_SCHNELL
    cast(Any, runtime).assembled.diffusion.guidance_in = None
    calls: list[object] = []

    with pytest.raises(DenoiseError, match="schnell"):
        runtime.sample_scheduled(
            flux_latent(),
            cond=cast(Any, object()),
            resolver=cast(
                Any,
                lambda requests, cancel: calls.append((requests, cancel)),  # pyright: ignore[reportUnknownLambdaType]
            ),
            guidance=2.0,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )
    assert calls == []


def test_sd_scheduled_inpaint_and_guidance_refuse_before_materialization() -> None:
    runtime = _runtime("sd")
    invalid = cast(Any, object())
    with pytest.raises(WiringError, match="distilled-guidance"):
        runtime.sample_scheduled(
            sd_latent(),
            cond=invalid,
            guidance=1.0,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )
    with pytest.raises(ScheduledSamplingError, match="scheduled-sd-inpaint"):
        runtime.sample_scheduled(
            sd_latent(),
            cond=invalid,
            denoise_mask=torch.ones((1, 8, 8)),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )
    cast(Any, runtime).assembled.diffusion.config = replace(
        runtime.assembled.diffusion.config, in_channels=9
    )
    with pytest.raises(ScheduledSamplingError, match="scheduled-sd-inpaint"):
        runtime.sample_scheduled(
            sd_latent(),
            cond=invalid,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )


def test_private_denoiser_prepares_once_and_refuses_stride_drift_with_cleanup() -> None:
    runtime = _runtime("flux")
    cond = flux_cond("cond")
    carrier = _conditioning_carrier(FLUX_DEV.id, cond)
    from dinkster_inference_torch.regional import materialize_regions
    from dinkster_inference_torch.wiring import _flux_sigma_space

    regions = materialize_regions(carrier, FLUX_DEV.id, 8, 8, "cpu")

    def zero_evaluate(
        _regions: tuple[object, ...],
        x: torch.Tensor,
        _sigma: float,
        *_rest: object,
    ) -> torch.Tensor:
        return torch.zeros_like(x)

    denoiser = ScheduledConditioningDenoiser(
        regions,
        (),
        family_id=FLUX_DEV.id,
        space=_flux_sigma_space(FLUX_DEV),
        model=runtime.assembled.diffusion,
        evaluate=cast(Any, zero_evaluate),
        patch_sets=cast(Any, {}),
        compute_dtype=torch.float32,
        device=torch.device("cpu"),
        cancel=lambda: False,
    )
    x = flux_latent()
    prepared = denoiser.prepare_conditioning(regions, GuidanceRole.CONDITIONAL)
    denoiser.evaluate_conditioning(x, 1.0, prepared)
    assert denoiser.prepare_count == 1
    denoiser.evaluate_conditioning(x, 0.5, prepared)
    assert denoiser.prepare_count == 1
    drifted = torch.empty_strided(x.shape, tuple(reversed(x.stride())))
    with pytest.raises(ScheduledSamplingError, match="evaluation: prepared-sample"):
        denoiser.evaluate_conditioning(drifted, 0.25, prepared)
    denoiser.close()
    denoiser.close()
    assert not runtime.assembled.diffusion._forward_hooks


def test_public_api_does_not_widen_family_runtime_protocol() -> None:
    from dinkster_inference.runtime import FamilyRuntime

    assert hasattr(FluxRuntime, "sample_scheduled")
    assert hasattr(SDRuntime, "sample_scheduled")
    assert "sample_scheduled" not in FamilyRuntime.__dict__


def test_scheduled_cuda_single_visible_device_checkout_source() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda:0")
    runtime = _runtime("flux")
    runtime.assembled.diffusion.to(device)
    output = runtime.sample_scheduled(
        flux_latent().to(device),
        cond=_conditioning_carrier(FLUX_DEV.id, flux_cond("single-device")),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        seed=9,
        compute_dtype=torch.float32,
        device=device,
    )
    assert output.device == device
    assert bool(torch.isfinite(output).all())
    assert not runtime.assembled.diffusion._forward_hooks


def test_scheduled_cuda_two_device_isolation() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    outputs: list[torch.Tensor] = []
    for index in (0, 1):
        device = torch.device(f"cuda:{index}")
        runtime = _runtime("flux")
        runtime.assembled.diffusion.to(device)
        latent = flux_latent().to(device)
        carrier = _conditioning_carrier(FLUX_DEV.id, flux_cond("cond"))
        output = runtime.sample_scheduled(
            latent,
            cond=carrier,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=5,
            compute_dtype=torch.float32,
            device=device,
        )
        outputs.append(output.cpu())
        assert output.device == device
        assert not runtime.assembled.diffusion._forward_hooks
    assert torch.equal(outputs[0], outputs[1])
