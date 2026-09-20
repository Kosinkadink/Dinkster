# pyright: basic
"""The generic sampling composition seam (sampling_execution.py).

These are bit-equality pins, not tolerance checks: each helper must
reproduce the family runtimes' composition behavior exactly. The
guided path in particular must replay plain classifier-free guidance
to the bit with an empty registry - the cfg==1 optimization, the
fused uncond-then-cond batch order, the unfused two-forward order,
and the CFG++ synthetic-zero uncond all included.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLUX_DEV,
    AttentionGuidanceDescriptor,
    CancellationToken,
    Conditioning,
    ConditioningBatching,
    CustomSamplingResult,
    CustomSamplingRuntime,
    DiscreteSigmas,
    DualSamplingGuidance,
    FlowSigmas,
    FluxConfig,
    GuidanceContractError,
    GuidanceContribution,
    GuidancePostCFGDescriptor,
    GuidanceRole,
    MultiStreamLatent,
    NoiseKind,
    Parameterization,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    ProgressScope,
    Registry,
    SamplerDescriptor,
    SamplerInfo,
    SamplingExecutionContext,
    SamplingGuidance,
    SamplingSegment,
    SchedulerDescriptor,
    SparseLatent,
    UNetConfig,
    builtin_samplers,
    offset_first_sigma_for_snr,
    sampling_sigmas,
    use_sampling_environment,
)
from dinkster_inference.solvers import (
    DINKSTER_DPM_2,
    DINKSTER_DPMPP_2M_SDE,
    DINKSTER_ER_SDE,
    DINKSTER_EULER,
    DINKSTER_EULER_CFG_PP,
    DINKSTER_UNI_PC,
)
from dinkster_inference_torch import (
    BrownianTreeNoise,
    Flux,
    FluxDenoiser,
    GuidanceExecutor,
    GuidanceRegistry,
    SDDenoiser,
    UNetModel,
    guidance_transforms,
)
from dinkster_inference_torch.cfg import cfg_combine
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.guidance import (
    ConditioningEvaluation,
    ConditioningValidationPath,
    GuidedDenoiser,
)
from dinkster_inference_torch.sampling_execution import (
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingExecutionRegistration,
    SamplingSchedule,
    SingleStreamLatentAdapter,
    brownian_step_noise,
    build_custom_sampling_schedule,
    build_sampling_schedule,
    compile_guidance_plan,
    guided_denoiser,
    narrow_single_stream_custom_sampling,
    resolve_sampling,
    run_ksampler_as_custom,
    sampling_execution,
    slice_sampling_schedule,
)
from dinkster_inference_torch.schedules import (
    _exact_table_space,
    beta_schedule,
    custom_beta_sigmas,
    discrete_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.sparse import make_sparse_support, pack_sparse_latent
from dinkster_inference_torch.unet import AttentionGuidanceContext
from dinkster_inference_torch.wiring import _flux_sigma_space
from unet_fill import fill_state_dict, hashed_input

TINY = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=12,
    context_in_dim=24,
    hidden_size=32,
    depth=2,
    depth_single_blocks=2,
    num_heads=2,
    axes_dim=(4, 6, 6),
)


def tiny_flux() -> Flux:
    model = Flux(TINY)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def tiny_cond(name: str, tokens: int = 3) -> Conditioning[torch.Tensor]:
    pooled_width = TINY.vec_in_dim
    assert pooled_width is not None
    return Conditioning(
        embeddings=hashed_input(f"{name}:ctx", (1, tokens, TINY.context_in_dim)),
        pooled=hashed_input(f"{name}:y", (1, pooled_width)),
    )


def tiny_latent() -> torch.Tensor:
    return hashed_input("latent", (1, TINY.in_channels, 8, 8))


def execution() -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((1.0, 0.5, 0.0), 0, 0, 1.0, 7, token, ProgressScope(token), {})


def guided(
    evaluator: FluxDenoiser,
    cond: Conditioning[torch.Tensor],
    uncond: Conditioning[torch.Tensor] | None,
    cfg_scale: float,
    sampler: SamplerDescriptor = DINKSTER_EULER,
) -> GuidedDenoiser:
    return guided_denoiser(
        ConditioningEvaluation(
            lambda value, _role: evaluator.prepare_conditioning(value),
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            standard_activation_memory_factor=1.0,
        ),
        input=tiny_latent(),
        executor=None,
        plan=compile_guidance_plan(
            cond,
            SamplingGuidance(uncond, cfg_scale),
            sampler,
            None,
        ),
        execution=execution(),
    )


class _FamilyError(Exception):
    pass


class TestNarrowSingleStreamCustomSampling:
    """The single-stream admission arm of the one custom-sampling seam."""

    @staticmethod
    def _narrow(**overrides: object) -> tuple[object, ...]:
        values: dict[str, Any] = {
            "latent": torch.zeros(1, 4, 8, 8),
            "noise": torch.zeros(1, 4, 8, 8),
            "cond": Conditioning(torch.zeros(1, 3, 16)),
            "cfg": None,
            "denoise_mask": None,
        }
        values.update(overrides)
        return narrow_single_stream_custom_sampling("dinkster.test", error=_FamilyError, **values)

    def test_passes_the_single_stream_shape_through_unchanged(self) -> None:
        latent = torch.zeros(1, 4, 8, 8)
        noise = torch.ones(1, 4, 8, 8)
        cond = Conditioning(torch.zeros(1, 3, 16))
        cfg = SamplingGuidance(Conditioning(torch.zeros(1, 3, 16)), 3.5)
        mask = torch.ones(1, 1, 8, 8)
        narrowed = self._narrow(latent=latent, noise=noise, cond=cond, cfg=cfg, denoise_mask=mask)
        assert narrowed == (latent, noise, cond, cfg, mask)

    def test_refuses_a_multistream_latent(self) -> None:
        streams = MultiStreamLatent.from_pairs((("video", torch.zeros(1, 4, 8, 8)),))
        with pytest.raises(_FamilyError, match="single-stream tensor latent"):
            self._narrow(latent=streams)

    def test_refuses_a_sparse_latent(self) -> None:
        sparse = pack_sparse_latent(
            make_sparse_support(
                torch.tensor(((0, 1, 2, 3),), dtype=torch.int32),
                (1,),
                8,
                (-0.5, -0.5, -0.5),
                (0.125, 0.125, 0.125),
            ),
            torch.zeros(1, 4),
        )
        with pytest.raises(_FamilyError, match="single-stream tensor latent"):
            self._narrow(latent=sparse)
        with pytest.raises(_FamilyError, match="single-stream tensor noise"):
            self._narrow(noise=sparse)
        with pytest.raises(_FamilyError, match="single-stream denoise mask"):
            self._narrow(denoise_mask=sparse)

    def test_refuses_multistream_noise(self) -> None:
        streams = MultiStreamLatent.from_pairs((("video", torch.zeros(1, 4, 8, 8)),))
        with pytest.raises(_FamilyError, match="single-stream tensor noise"):
            self._narrow(noise=streams)

    def test_refuses_prepared_multistream_conditioning(self) -> None:
        prepared = PreparedMultiStreamConditioning("native:test", object())
        with pytest.raises(_FamilyError, match="not a prepared multi-stream payload"):
            self._narrow(cond=prepared)

    def test_admits_dual_guidance(self) -> None:
        latent = torch.zeros(1, 4, 8, 8)
        noise = torch.ones(1, 4, 8, 8)
        cond = Conditioning(torch.zeros(1, 3, 16))
        dual = DualSamplingGuidance(cond, cond, 3.0, 1.5)
        narrowed = self._narrow(latent=latent, noise=noise, cond=cond, cfg=dual)
        assert narrowed == (latent, noise, cond, dual, None)

    def test_refuses_prepared_dual_guidance_lanes(self) -> None:
        cond = Conditioning(torch.zeros(1, 3, 16))
        prepared = PreparedMultiStreamConditioning("native:test", object())
        with pytest.raises(_FamilyError, match="guidance requires a Conditioning payload"):
            self._narrow(cfg=DualSamplingGuidance(cast("Any", prepared), cond, 3.0, 1.5))

    def test_refuses_prepared_guidance_uncond(self) -> None:
        prepared = PreparedMultiStreamConditioning("native:test", object())
        with pytest.raises(_FamilyError, match="guidance requires a Conditioning payload"):
            self._narrow(cfg=SamplingGuidance(prepared, 3.0))

    def test_refuses_perp_neg_guidance_unless_admitted(self) -> None:
        cond = Conditioning(torch.zeros(1, 3, 16))
        perp = PerpNegSamplingGuidance(cond, cond, 3.0, 1.0)
        with pytest.raises(_FamilyError, match="does not support PerpNegSamplingGuidance"):
            self._narrow(cfg=perp)

    def test_admits_perp_neg_guidance_when_opted_in(self) -> None:
        latent = torch.zeros(1, 4, 8, 8)
        noise = torch.ones(1, 4, 8, 8)
        cond = Conditioning(torch.zeros(1, 3, 16))
        perp = PerpNegSamplingGuidance(
            Conditioning(torch.zeros(1, 3, 16)),
            Conditioning(torch.zeros(1, 3, 16)),
            3.0,
            1.0,
        )
        narrowed = self._narrow(
            latent=latent, noise=noise, cond=cond, cfg=perp, admit_perp_neg=True
        )
        assert narrowed == (latent, noise, cond, perp, None)

    def test_refuses_prepared_perp_neg_payloads_even_when_admitted(self) -> None:
        prepared = PreparedMultiStreamConditioning("native:test", object())
        cond = Conditioning(torch.zeros(1, 3, 16))
        with pytest.raises(_FamilyError, match="guidance requires a Conditioning payload"):
            self._narrow(
                cfg=PerpNegSamplingGuidance(cast("Any", prepared), cond, 3.0, 1.0),
                admit_perp_neg=True,
            )
        with pytest.raises(_FamilyError, match="guidance requires a Conditioning payload"):
            self._narrow(
                cfg=PerpNegSamplingGuidance(cond, cast("Any", prepared), 3.0, 1.0),
                admit_perp_neg=True,
            )

    def test_refuses_a_multistream_denoise_mask(self) -> None:
        streams = MultiStreamLatent.from_pairs((("video", torch.zeros(1, 1, 8, 8)),))
        with pytest.raises(_FamilyError, match="single-stream denoise mask"):
            self._narrow(denoise_mask=streams)


class _CaptureSparseRuntime:
    def __init__(self) -> None:
        self.noise: object | None = None
        self.kwargs: dict[str, object] = {}

    def check_custom_sampling(self, request: object, **kwargs: object) -> None:
        pass

    def sample_custom(self, latent: object, **kwargs: object) -> CustomSamplingResult[object]:
        self.noise = kwargs["noise"]
        self.kwargs = dict(kwargs)
        return CustomSamplingResult(latent, None)


def test_torch_registry_binds_every_backend_agnostic_sampler() -> None:
    registry = torch_sampler_registry()
    descriptors = builtin_samplers()

    assert registry.ids() == tuple(descriptor.id for descriptor in descriptors)
    assert all(descriptor.make is not None for descriptor in registry)
    assert tuple(replace(descriptor, make=None) for descriptor in registry) == descriptors


def test_ksampler_sparse_noise_samples_only_features_and_preserves_support() -> None:
    support = make_sparse_support(
        torch.tensor(((0, 1, 2, 3), (0, 3, 2, 1)), dtype=torch.int32),
        (2,),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    features = torch.zeros(2, 4)
    latent = pack_sparse_latent(support, features)
    runtime = _CaptureSparseRuntime()
    result = run_ksampler_as_custom(
        cast("CustomSamplingRuntime[torch.Tensor]", runtime),
        latent,
        samplers=torch_sampler_registry(),
        schedulers=torch_scheduler_registry(),
        space=FlowSigmas(shift=1.0),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=11,
        cond=Conditioning(torch.zeros(1, 1, 4)),
        error=_FamilyError,
    )
    assert result.output is latent
    assert type(runtime.noise) is SparseLatent
    assert runtime.noise.support is support
    assert torch.equal(runtime.noise.features, prepare_noise(features, 11))


def test_ksampler_sparse_noise_indexes_batches_with_variable_point_counts() -> None:
    support = make_sparse_support(
        torch.tensor(
            ((0, 1, 2, 3), (0, 3, 2, 1), (1, 4, 5, 6)),
            dtype=torch.int32,
        ),
        (2, 1),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    features = torch.zeros(3, 4)
    runtime = _CaptureSparseRuntime()
    run_ksampler_as_custom(
        cast("CustomSamplingRuntime[torch.Tensor]", runtime),
        pack_sparse_latent(support, features),
        samplers=torch_sampler_registry(),
        schedulers=torch_scheduler_registry(),
        space=FlowSigmas(shift=1.0),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=11,
        cond=Conditioning(torch.zeros(1, 1, 4)),
        noise_inds=(1, 1),
        error=_FamilyError,
    )
    assert type(runtime.noise) is SparseLatent
    expected_batches = prepare_noise(torch.empty(2, 2, 4), 11, (1, 1))
    assert torch.equal(
        runtime.noise.features,
        torch.cat((expected_batches[0], expected_batches[1, :1])),
    )


def test_ksampler_sparse_no_noise_segment_preserves_support_with_zero_features() -> None:
    support = make_sparse_support(
        torch.tensor(((0, 1, 2, 3), (0, 3, 2, 1)), dtype=torch.int32),
        (2,),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    runtime = _CaptureSparseRuntime()
    run_ksampler_as_custom(
        cast("CustomSamplingRuntime[torch.Tensor]", runtime),
        pack_sparse_latent(support, torch.ones(2, 4)),
        samplers=torch_sampler_registry(),
        schedulers=torch_scheduler_registry(),
        space=FlowSigmas(shift=1.0),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=11,
        cond=Conditioning(torch.zeros(1, 1, 4)),
        segment=SamplingSegment(2, 0, 2, False, False),
        error=_FamilyError,
    )
    assert type(runtime.noise) is SparseLatent
    assert runtime.noise.support is support
    assert torch.equal(runtime.noise.features, torch.zeros(2, 4))


@pytest.mark.parametrize("reserved", ["latent", "seed", "context_windows"])
def test_ksampler_forwards_family_runtime_options_without_overriding_composition(
    reserved: str,
) -> None:
    latent = torch.zeros(1, 4)
    runtime = _CaptureSparseRuntime()
    run_ksampler_as_custom(
        cast("CustomSamplingRuntime[torch.Tensor]", runtime),
        latent,
        samplers=torch_sampler_registry(),
        schedulers=torch_scheduler_registry(),
        space=FlowSigmas(shift=1.0),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        cond=Conditioning(torch.zeros(1, 1, 4)),
        sample_custom_kwargs={"sampling_shift": 1.15, "compute_dtype": torch.float32},
        error=_FamilyError,
    )
    assert runtime.kwargs["sampling_shift"] == 1.15
    assert runtime.kwargs["compute_dtype"] is torch.float32
    with pytest.raises(_FamilyError, match=f"cannot override KSampler composition: {reserved}"):
        run_ksampler_as_custom(
            cast("CustomSamplingRuntime[torch.Tensor]", runtime),
            latent,
            samplers=torch_sampler_registry(),
            schedulers=torch_scheduler_registry(),
            space=FlowSigmas(shift=1.0),
            flow=True,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=2,
            cond=Conditioning(torch.zeros(1, 1, 4)),
            sample_custom_kwargs={reserved: 19},
            error=_FamilyError,
        )


class TestResolveSampling:
    def test_returns_both_descriptors(self) -> None:
        sampler, scheduler = resolve_sampling(
            torch_sampler_registry(),
            torch_scheduler_registry(),
            "dinkster.euler",
            "dinkster.simple",
            error=ValueError,
        )
        assert sampler.id == "dinkster.euler"
        assert scheduler.id == "dinkster.simple"

    def test_er_sde_descriptor_modes_execute_on_torch_tensors(self) -> None:
        descriptor = torch_sampler_registry().get("dinkster.er_sde")
        assert descriptor is not None
        assert descriptor.options == DINKSTER_ER_SDE.options
        x = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float64)
        sigmas = (4.0, 2.0, 1.0, 0.4, 0.0)

        def denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
            return x / (1.0 + sigma) + x.square() * (0.05 / (1.0 + sigma))

        calls: list[tuple[float, float]] = []

        def noise(sigma_from: float, sigma_to: float) -> torch.Tensor:
            calls.append((sigma_from, sigma_to))
            return torch.ones_like(x)

        defaults = descriptor.build()(
            denoiser, x, sigmas, SamplerInfo(Parameterization.EPS), noise=noise
        )
        explicit = descriptor.build(solver_type="ER-SDE", max_stage=3, eta=1.0, s_noise=1.0)(
            denoiser, x, sigmas, SamplerInfo(Parameterization.EPS), noise=noise
        )
        assert torch.equal(defaults, explicit)

        calls.clear()
        eta_zero = descriptor.build(
            solver_type="Reverse-time SDE", max_stage=3, eta=0.0, s_noise=100.0
        )(denoiser, x, sigmas, SamplerInfo(Parameterization.EPS), noise=noise)
        ode = descriptor.build(solver_type="ODE", max_stage=3, eta=2.0, s_noise=1.0)(
            denoiser,
            x,
            sigmas,
            SamplerInfo(Parameterization.EPS),
            noise=noise,
        )
        assert torch.equal(eta_zero, ode)
        assert calls == []

    def test_unknown_sampler_raises_the_family_error(self) -> None:
        class FamilyError(ValueError):
            pass

        with pytest.raises(FamilyError, match="unknown sampler 'nope'"):
            resolve_sampling(
                torch_sampler_registry(),
                torch_scheduler_registry(),
                "nope",
                "dinkster.simple",
                error=FamilyError,
            )

    def test_unknown_scheduler_raises_the_family_error(self) -> None:
        class FamilyError(ValueError):
            pass

        with pytest.raises(FamilyError, match="unknown scheduler 'nope'"):
            resolve_sampling(
                torch_sampler_registry(),
                torch_scheduler_registry(),
                "dinkster.euler",
                "nope",
                error=FamilyError,
            )

    def test_scheduler_lookup_never_runs_on_unknown_sampler(self) -> None:
        """Sampler resolution refuses before any scheduler lookup runs."""

        class Exploding(Registry[SchedulerDescriptor]):
            def get(self, key: str) -> SchedulerDescriptor | None:
                raise AssertionError("scheduler registry consulted")

        with pytest.raises(ValueError, match="unknown sampler"):
            resolve_sampling(
                torch_sampler_registry(),
                Exploding(),
                "nope",
                "dinkster.simple",
                error=ValueError,
            )


class TestBuildSamplingSchedule:
    def test_discrete_exact_table_view_uses_the_reference_percent_kernel(self) -> None:
        space = DiscreteSigmas.linear_beta()
        percent = 0.37

        exact = _exact_table_space(space).percent_to_sigma(percent)

        assert exact == discrete_percent_to_sigma(space, percent)
        assert exact == 2.5423052310943604
        assert exact != space.percent_to_sigma(percent)

    def test_discrete_beta_uses_the_reference_float32_table(self) -> None:
        space = DiscreteSigmas.linear_beta()
        scheduler = torch_scheduler_registry().get("dinkster.beta")
        assert scheduler is not None

        catalog = scheduler.make_sigmas(4, space)

        assert catalog == custom_beta_sigmas(space, 4, 0.6, 0.6)
        assert catalog == (
            14.614641189575195,
            5.686591625213623,
            1.6182788610458374,
            0.5153906345367432,
            0.0,
        )
        assert beta_schedule(space=space, steps=13, alpha=2.0, beta=5.0) == (
            14.614641189575195,
            1.8396111726760864,
            1.4082000255584717,
            1.1682159900665283,
            1.007049322128296,
            0.8800130486488342,
            0.7789232730865479,
            0.6906159520149231,
            0.6102519631385803,
            0.5362927913665771,
            0.46285301446914673,
            0.3865652084350586,
            0.2982633113861084,
            0.0,
        )

    def test_discrete_ddim_uniform_uses_the_reference_float32_table(self) -> None:
        scheduler = torch_scheduler_registry().get("dinkster.ddim_uniform")
        assert scheduler is not None

        sigmas = scheduler.make_sigmas(4, DiscreteSigmas.linear_beta())

        assert sigmas == (
            4.116696357727051,
            1.6236920356750488,
            0.6983981132507324,
            0.04131441190838814,
            0.0,
        )

    def test_simple_unipc_uses_the_reference_float32_flow_table(self) -> None:
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None

        schedule = build_sampling_schedule(
            scheduler,
            FlowSigmas(shift=5.0),
            DINKSTER_UNI_PC,
            4,
            denoise=None,
            flow=True,
        )

        assert schedule.sigmas == (
            1.0,
            0.9523810148239136,
            0.882352888584137,
            0.7692307829856873,
            0.0,
        )

    def test_matches_the_hand_rolled_flux_composition(self) -> None:
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        schedule = build_sampling_schedule(
            scheduler, space, DINKSTER_DPMPP_2M_SDE, 4, denoise=None, flow=True
        )
        expected_pre = sampling_sigmas(
            scheduler,
            space,
            4,
            denoise=None,
            discard_penultimate=DINKSTER_DPMPP_2M_SDE.discard_penultimate,
        )
        assert schedule.pre_offset == expected_pre
        assert schedule.sigmas == offset_first_sigma_for_snr(expected_pre, space, flow=True)
        assert schedule.sigmas != schedule.pre_offset
        assert schedule.initial_sigma == 1.0

    def test_no_offset_without_flow(self) -> None:
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        schedule = build_sampling_schedule(
            scheduler, space, DINKSTER_DPMPP_2M_SDE, 4, denoise=None, flow=False
        )
        assert schedule.sigmas == schedule.pre_offset

    def test_no_offset_without_the_sampler_declaration(self) -> None:
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        schedule = build_sampling_schedule(
            scheduler, space, DINKSTER_EULER, 4, denoise=None, flow=True
        )
        assert schedule.sigmas == schedule.pre_offset

    def test_custom_schedule_preserves_the_exact_pre_offset_sequence(self) -> None:
        sigmas = (1.0, 0.8, 0.3, 0.0)
        schedule = build_custom_sampling_schedule(
            sigmas,
            _flux_sigma_space(FLUX_DEV),
            DINKSTER_DPMPP_2M_SDE,
            flow=True,
        )
        assert schedule.pre_offset is sigmas
        assert schedule.sigmas == offset_first_sigma_for_snr(
            sigmas,
            _flux_sigma_space(FLUX_DEV),
            flow=True,
        )

    def test_custom_schedule_never_applies_k_sampler_discard(self) -> None:
        sigmas = (1.0, 0.7, 0.4, 0.0)
        schedule = build_custom_sampling_schedule(
            sigmas,
            _flux_sigma_space(FLUX_DEV),
            DINKSTER_DPM_2,
            flow=True,
        )
        assert DINKSTER_DPM_2.discard_penultimate
        assert schedule.pre_offset == sigmas
        assert schedule.sigmas == sigmas

    def test_partial_denoise_below_sigma_one_is_untouched(self) -> None:
        """offset_first_sigma_for_snr is the identity when the schedule
        already starts below 1.0 - the img2img case."""
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        schedule = build_sampling_schedule(
            scheduler, space, DINKSTER_DPMPP_2M_SDE, 4, denoise=0.5, flow=True
        )
        assert schedule.pre_offset[0] < 1.0
        assert schedule.sigmas == schedule.pre_offset
        assert schedule.initial_sigma == schedule.sigmas[0]

    def test_segment_slices_the_full_schedule_in_reference_order(self) -> None:
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        full = build_sampling_schedule(
            scheduler, space, DINKSTER_EULER, 6, denoise=1.0, flow=True
        ).pre_offset
        leftover = SamplingSegment(6, 2, 5, False, True)
        forced = replace(leftover, return_with_leftover_noise=False)
        assert (
            build_sampling_schedule(
                scheduler,
                space,
                DINKSTER_EULER,
                6,
                denoise=1.0,
                flow=True,
                segment=leftover,
            ).pre_offset
            == full[:6][2:]
        )
        assert build_sampling_schedule(
            scheduler,
            space,
            DINKSTER_EULER,
            6,
            denoise=1.0,
            flow=True,
            segment=forced,
        ).pre_offset == (*full[2:5], 0.0)

    def test_segment_refuses_step_and_denoise_mismatches(self) -> None:
        space = _flux_sigma_space(FLUX_DEV)
        scheduler = torch_scheduler_registry().get("dinkster.simple")
        assert scheduler is not None
        segment = SamplingSegment(6, 0, 3, True, True)
        with pytest.raises(ValueError, match="built for 6 steps"):
            build_sampling_schedule(
                scheduler,
                space,
                DINKSTER_EULER,
                7,
                denoise=1.0,
                flow=True,
                segment=segment,
            )
        with pytest.raises(ValueError, match="denoise=1.0 or None"):
            build_sampling_schedule(
                scheduler,
                space,
                DINKSTER_EULER,
                6,
                denoise=0.5,
                flow=True,
                segment=segment,
            )

    def test_start_beyond_effective_discarded_schedule_returns_empty(self) -> None:
        segment = SamplingSegment(6, 5, 6, True, False)
        assert (
            slice_sampling_schedule(
                (1.0, 0.8, 0.6, 0.4, 0.0),
                segment,
                steps=6,
                denoise=1.0,
            )
            == ()
        )


class TestBrownianStepNoise:
    def test_none_for_non_brownian_solvers(self) -> None:
        schedule = SamplingSchedule((1.0, 0.5, 0.0), (0.99, 0.5, 0.0))
        assert brownian_step_noise(DINKSTER_EULER, schedule, tiny_latent(), seed=7) is None

    def test_none_when_the_offset_did_not_move_the_schedule(self) -> None:
        """The engine's own construction from the executed sigmas is
        bit-identical then (same bounds, same seed, and BrownianTreeNoise
        reads only shape/dtype/device from ``like``)."""
        schedule = SamplingSchedule((0.8, 0.5, 0.0), (0.8, 0.5, 0.0))
        assert brownian_step_noise(DINKSTER_DPMPP_2M_SDE, schedule, tiny_latent(), seed=7) is None

    def test_moved_schedule_builds_the_pre_offset_tree(self) -> None:
        pre = (1.0, 0.5, 0.0)
        schedule = SamplingSchedule(pre, (0.9999, 0.5, 0.0))
        like = tiny_latent()
        noise = brownian_step_noise(DINKSTER_DPMPP_2M_SDE, schedule, like, seed=7)
        assert noise is not None
        expected = BrownianTreeNoise(
            like.to(dtype=torch.float32),
            0.5,
            1.0,
            seed=7,
            cpu=DINKSTER_DPMPP_2M_SDE.noise is NoiseKind.BROWNIAN,
        )
        assert torch.equal(noise(0.9, 0.6), expected(0.9, 0.6))

    def test_none_without_positive_pre_offset_sigmas(self) -> None:
        schedule = SamplingSchedule((0.0,), (0.5,))
        assert brownian_step_noise(DINKSTER_DPMPP_2M_SDE, schedule, tiny_latent(), seed=7) is None


class TestGuidedDenoiserParity:
    """Shared guidance preserves Flux evaluation order and values."""

    def test_cfg_above_one_matches_the_fused_plain_path(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        expected = cfg_combine(positive, negative, 3.0)
        wrapped = guided(evaluator, cond, uncond, 3.0)
        assert torch.equal(wrapped(x, 0.7), expected)
        assert wrapped.conditioning_plan is not None
        assert all(
            lane.validation is ConditioningValidationPath.LEGACY_SHAPE_PREDICATE
            for lane in wrapped.conditioning_plan.lanes
        )

    def test_cfg_one_evaluates_only_one_condition(self) -> None:
        model = tiny_flux()
        cond = tiny_cond("p")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        calls: list[str] = []
        contract = ConditioningEvaluation(
            lambda value, role: (
                calls.append(role.value),
                evaluator.prepare_conditioning(value),
            )[1],
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
        )
        x = tiny_latent()
        out = guided_denoiser(
            contract,
            input=x,
            executor=None,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(cast(Any, object()), 1.0),
                DINKSTER_EULER,
                None,
            ),
            execution=execution(),
        )(x, 0.7)
        expected = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        assert torch.equal(out, expected)
        assert calls == [GuidanceRole.CONDITIONAL.value]

    def test_disabled_cfg_one_optimization_evaluates_both_conditions(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        roles: list[GuidanceRole] = []

        def prepare(value: object, role: GuidanceRole) -> object:
            roles.append(role)
            return evaluator.prepare_conditioning(cast("Any", value))

        x = tiny_latent()
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                prepare,
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=x,
            executor=None,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, 1.0, disable_cfg1_optimization=True),
                DINKSTER_EULER,
                None,
            ),
            execution=execution(),
        )
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(wrapped(x, 0.7), cfg_combine(positive, negative, 1.0))
        assert roles == [GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL]

    def test_no_uncond_matches_the_plain_path(self) -> None:
        model = tiny_flux()
        cond = tiny_cond("p")
        evaluator = FluxDenoiser(model, cond, compute_dtype=torch.float32)
        x = tiny_latent()
        assert torch.equal(guided(evaluator, cond, None, 3.0)(x, 0.7), evaluator(x, 0.7))

    def test_unfused_token_lengths_match_the_two_forward_order(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p", tokens=3), tiny_cond("n", tokens=5)
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        positive = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        negative = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(uncond))
        assert torch.equal(
            guided(evaluator, cond, uncond, 3.0)(x, 0.7),
            cfg_combine(positive, negative, 3.0),
        )

    def test_cfg_pp_returns_the_planned_unconditional_prediction(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        wrapped = guided(evaluator, cond, uncond, 3.0, sampler=DINKSTER_EULER_CFG_PP)
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(got, cfg_combine(positive, negative, 3.0))
        assert torch.equal(got_uncond, negative)

    def test_cfg_pp_without_uncond_matches_the_synthetic_zero(self) -> None:
        model = tiny_flux()
        cond = tiny_cond("p")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        wrapped = guided(evaluator, cond, None, 3.0, sampler=DINKSTER_EULER_CFG_PP)
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        positive = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        assert torch.equal(got, cfg_combine(positive, torch.zeros_like(positive), 3.0))
        assert torch.equal(got_uncond, torch.zeros_like(got))

    def test_cfg_pp_forces_uncond_at_cfg_one(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        wrapped = guided(evaluator, cond, uncond, 1.0, sampler=DINKSTER_EULER_CFG_PP)
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(got, cfg_combine(positive, negative, 1.0))
        assert torch.equal(got_uncond, negative)

    def test_inactive_caller_executor_is_replaced_by_a_builtin(self) -> None:
        """An executor whose registry declares nothing never reaches
        execution: guided_denoiser substitutes a fresh exact builtin,
        so caller-reachable instance state cannot run where plain
        classifier-free guidance is meant."""
        model = tiny_flux()
        cond = tiny_cond("p")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        executor = GuidanceExecutor(GuidanceRegistry())
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
            ),
            input=tiny_latent(),
            executor=executor,
            plan=compile_guidance_plan(cond, SamplingGuidance(None, 1.0), DINKSTER_EULER, executor),
            execution=execution(),
        )
        resolved = wrapped._executor  # noqa: SLF001
        assert resolved is not executor
        assert type(resolved) is GuidanceExecutor
        assert type(resolved.registry) is GuidanceRegistry
        assert not resolved.registry.active

    def test_inactive_executor_with_shadowed_execute_cannot_alter_output(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        executor = GuidanceExecutor(GuidanceRegistry())

        def hijacked(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("an inactive caller executor must never run")

        executor.execute = hijacked  # type: ignore[method-assign]
        x = tiny_latent()
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=x,
            executor=executor,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, 3.0),
                DINKSTER_EULER,
                executor,
            ),
            execution=execution(),
        )
        assert torch.equal(wrapped(x, 0.7), guided(evaluator, cond, uncond, 3.0)(x, 0.7))

    def test_active_caller_executor_is_used(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        executor = GuidanceExecutor(GuidanceRegistry((("ext", contribution),)))
        x = tiny_latent()
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=x,
            executor=executor,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, 3.0),
                DINKSTER_EULER,
                executor,
            ),
            execution=execution(),
        )
        assert wrapped._executor is executor  # noqa: SLF001
        assert torch.equal(wrapped(x, 0.7), guided(evaluator, cond, uncond, 3.0)(x, 0.7) + 1.0)

    def test_per_run_transforms_match_the_load_time_registry(self) -> None:
        """A contribution carried on SamplingGuidance.transforms executes
        bit-identically to the same contribution registered load-time."""
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        x = tiny_latent()

        def wrap(
            executor: GuidanceExecutor | None,
            cfg: SamplingGuidance[Conditioning[torch.Tensor]],
        ) -> GuidedDenoiser:
            return guided_denoiser(
                ConditioningEvaluation(
                    lambda value, _role: evaluator.prepare_conditioning(value),
                    evaluator.evaluate_conditioning,
                    evaluator.batchable,
                    evaluator.evaluate_conditioning_batch,
                    standard_activation_memory_factor=1.0,
                ),
                input=x,
                executor=executor,
                plan=compile_guidance_plan(cond, cfg, DINKSTER_EULER, executor),
                execution=execution(),
            )

        load_time = wrap(
            GuidanceExecutor(GuidanceRegistry((("ext", contribution),))),
            SamplingGuidance(uncond, 3.0),
        )
        per_run = wrap(None, SamplingGuidance(uncond, 3.0, (("ext", contribution),)))
        assert torch.equal(per_run(x, 0.7), load_time(x, 0.7))
        assert torch.equal(per_run(x, 0.7), guided(evaluator, cond, uncond, 3.0)(x, 0.7) + 1.0)

    def test_per_run_transforms_merge_after_the_load_time_registry(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        load_time = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("a.plus", lambda context: context.reduced + 1.0),)
        )
        per_run = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("b.times", lambda context: context.reduced * 2.0),)
        )
        executor = GuidanceExecutor(GuidanceRegistry((("load", load_time),)))
        x = tiny_latent()
        cfg = SamplingGuidance(uncond, 3.0, (("run", per_run),))
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=x,
            executor=executor,
            plan=compile_guidance_plan(cond, cfg, DINKSTER_EULER, executor),
            execution=execution(),
        )
        # Post transforms apply in (order, id) order: +1 then *2.
        assert torch.equal(
            wrapped(x, 0.7), (guided(evaluator, cond, uncond, 3.0)(x, 0.7) + 1.0) * 2.0
        )

    def test_duplicate_per_run_owner_id_is_refused(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        executor = GuidanceExecutor(GuidanceRegistry((("ext", contribution),)))
        with pytest.raises(GuidanceContractError, match="'ext' is already registered"):
            compile_guidance_plan(
                tiny_cond("p"),
                SamplingGuidance(tiny_cond("n"), 3.0, (("ext", contribution),)),
                DINKSTER_EULER,
                executor,
            )

    def test_duplicate_owner_id_within_transforms_is_refused(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        with pytest.raises(GuidanceContractError, match="'ext' is already registered"):
            compile_guidance_plan(
                tiny_cond("p"),
                SamplingGuidance(
                    tiny_cond("n"), 3.0, (("ext", contribution), ("ext", contribution))
                ),
                DINKSTER_EULER,
                None,
            )

    def test_per_run_requires_uncond_forces_the_uncond_lane(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor(
                    "x.plus", lambda context: context.reduced, requires_uncond=True
                ),
            )
        )
        uncond = tiny_cond("n")
        plan = compile_guidance_plan(
            tiny_cond("p"),
            SamplingGuidance(uncond, 1.0, (("ext", contribution),)),
            DINKSTER_EULER,
            None,
        )
        assert plan.force_uncond
        assert not plan.has_strategy
        assert plan.contributions == (("ext", contribution),)
        carried = plan.with_conditioning(tiny_cond("p2"), uncond)
        assert carried.contributions == plan.contributions
        assert carried.has_strategy == plan.has_strategy

    def test_transforms_validation_refuses_malformed_entries(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced),)
        )
        with pytest.raises(TypeError, match="owner id must be a non-empty string"):
            SamplingGuidance(None, 1.0, (("", contribution),))
        with pytest.raises(TypeError, match="must be a GuidanceContribution"):
            SamplingGuidance(None, 1.0, (("ext", cast("Any", object())),))
        with pytest.raises(TypeError, match="pairs"):
            SamplingGuidance(None, 1.0, cast("Any", (("ext",),)))

    @pytest.mark.parametrize(
        "guidance",
        (
            SamplingGuidance(None, 1.0),
            DualSamplingGuidance(None, None, 1.0, 1.0),
            PerpNegSamplingGuidance(None, None, 1.0, 1.0),
        ),
    )
    def test_cfg_one_optimization_setting_requires_an_exact_bool(self, guidance: object) -> None:
        with pytest.raises(TypeError, match="disable_cfg1_optimization must be a bool"):
            replace(cast("Any", guidance), disable_cfg1_optimization=cast("Any", 1))

    def test_dual_guidance_retains_positional_batching_contract(self) -> None:
        batching = ConditioningBatching()
        guidance = DualSamplingGuidance(None, None, 2.0, 1.5, False, batching, True)

        assert guidance.batching is batching
        assert guidance.disable_cfg1_optimization is True
        assert guidance.transforms == ()


class TestDualCFGGuidance:
    @staticmethod
    def _run(guidance: DualSamplingGuidance[Conditioning[torch.Tensor]]) -> torch.Tensor:
        evaluator = FluxDenoiser(tiny_flux(), compute_dtype=torch.float32)
        latent = tiny_latent()
        return guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=latent,
            executor=None,
            plan=compile_guidance_plan(tiny_cond("p"), guidance, DINKSTER_EULER, None),
            execution=execution(),
        )(latent, 0.7)

    @pytest.mark.parametrize("nested", (False, True))
    def test_reproduces_three_lane_combination(self, nested: bool) -> None:
        evaluator = FluxDenoiser(tiny_flux(), compute_dtype=torch.float32)
        latent = tiny_latent()
        positive, middle, negative = tiny_cond("p"), tiny_cond("m"), tiny_cond("n")
        positive_out, middle_out, negative_out = evaluator.evaluate_conditioning_batch(
            latent,
            0.7,
            tuple(evaluator.prepare_conditioning(value) for value in (positive, middle, negative)),
        )
        guidance = DualSamplingGuidance(middle, negative, 2.5, 1.75, nested)
        expected = (
            negative_out + 1.75 * (cfg_combine(positive_out, middle_out, 2.5) - negative_out)
            if nested
            else cfg_combine(middle_out, negative_out, 1.75) + (positive_out - middle_out) * 2.5
        )

        assert torch.equal(self._run(guidance), expected)

    def test_model_guidance_transform_composes_after_dual_reduction(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor("test.plus-one", lambda context: context.reduced + 1.0),
            )
        )
        guidance = DualSamplingGuidance(
            tiny_cond("m"),
            tiny_cond("n"),
            2.5,
            1.75,
            transforms=(("test.model-guidance", contribution),),
        )
        without_transform = replace(guidance, transforms=())

        assert torch.equal(self._run(guidance), self._run(without_transform) + 1.0)


class TestPerpNegGuidance:
    """PerpNegSamplingGuidance replays comfy_extras/nodes_perpneg.py
    Guider_PerpNeg @ b78cec87 to the bit: the fused three-lane batch,
    the reference lane dropping, and the perpendicular rejection."""

    @staticmethod
    def _wrap(
        evaluator: FluxDenoiser,
        cond: Conditioning[torch.Tensor],
        guidance: PerpNegSamplingGuidance[Conditioning[torch.Tensor]],
        sampler: SamplerDescriptor[Any] = DINKSTER_EULER,
    ) -> GuidedDenoiser:
        return guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=tiny_latent(),
            executor=None,
            plan=compile_guidance_plan(cond, guidance, sampler, None),
            execution=execution(),
        )

    def test_compiles_three_lanes_and_the_strategy(self) -> None:
        guidance = PerpNegSamplingGuidance(tiny_cond("n"), tiny_cond("e"), 3.5, 0.5)
        plan = compile_guidance_plan(tiny_cond("p"), guidance, DINKSTER_EULER, None)
        assert [(lane.id, lane.role) for lane in plan.conditions] == [
            ("positive", GuidanceRole.CONDITIONAL),
            ("negative", GuidanceRole.UNCONDITIONAL),
            ("empty", GuidanceRole.AUXILIARY),
        ]
        assert plan.conditions[1].conditioning is guidance.uncond
        assert plan.conditions[2].conditioning is guidance.empty
        assert plan.cfg_scale == 3.5
        assert plan.has_strategy
        assert not plan.force_uncond
        assert plan.contributions[-1][0] == "dinkster.perp-neg"

    def test_cfg_pp_sampler_forces_the_negative_lane(self) -> None:
        guidance = PerpNegSamplingGuidance(tiny_cond("n"), tiny_cond("e"), 3.0, 0.0)
        plan = compile_guidance_plan(tiny_cond("p"), guidance, DINKSTER_EULER_CFG_PP, None)
        assert plan.force_uncond

    def test_duplicate_perp_neg_owner_id_is_refused(self) -> None:
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        guidance = PerpNegSamplingGuidance(
            tiny_cond("n"), tiny_cond("e"), 3.0, 1.0, (("dinkster.perp-neg", contribution),)
        )
        with pytest.raises(
            GuidanceContractError, match="'dinkster.perp-neg' is already registered"
        ):
            compile_guidance_plan(tiny_cond("p"), guidance, DINKSTER_EULER, None)

    def test_a_load_time_strategy_conflicts_loudly(self) -> None:
        executor = GuidanceExecutor(
            GuidanceRegistry((("load.strategy", guidance_transforms.perp_neg(0.5)),))
        )
        guidance = PerpNegSamplingGuidance(tiny_cond("n"), tiny_cond("e"), 3.0, 1.0)
        with pytest.raises(GuidanceContractError, match="multiple guidance strategies"):
            compile_guidance_plan(tiny_cond("p"), guidance, DINKSTER_EULER, executor)

    def test_reproduces_the_reference_three_lane_combination(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        wrapped = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 3.0, 0.75))
        got = wrapped(x, 0.7)
        empty_p, neg_p, pos_p = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(empty),
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        pos = pos_p - empty_p
        neg = neg_p - empty_p
        perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
        assert torch.equal(got, empty_p + 3.0 * (pos - perp * 0.75))

    def test_neg_scale_zero_drops_the_negative_lane(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        got = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 3.0, 0.0))(x, 0.7)
        # The dropped lane reduces against calc_cond_batch's untouched
        # zeros, exactly as the reference cfg1 optimization does.
        empty_p, pos_p = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (evaluator.prepare_conditioning(empty), evaluator.prepare_conditioning(cond)),
        )
        pos = pos_p - empty_p
        neg = torch.zeros_like(x) - empty_p
        perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
        assert torch.equal(got, empty_p + 3.0 * (pos - perp * 0.0))

    def test_neg_scale_zero_matches_plain_guidance_against_the_empty_prompt(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        got = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 3.0, 0.0))(x, 0.7)
        assert torch.equal(got, guided(evaluator, cond, empty, 3.0)(x, 0.7))

    def test_neg_scale_zero_at_cfg_one_evaluates_only_the_positive_lane(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        got = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 1.0, 0.0))(x, 0.7)
        positive = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        assert torch.equal(got, positive)

    def test_cfg_one_with_active_neg_scale_keeps_every_lane(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        got = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 1.0, 1.0))(x, 0.7)
        empty_p, neg_p, pos_p = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(empty),
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        pos = pos_p - empty_p
        neg = neg_p - empty_p
        perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
        assert torch.equal(got, empty_p + 1.0 * (pos - perp * 1.0))

    def test_cfg_pp_returns_the_negative_lane_prediction(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        wrapped = self._wrap(
            evaluator,
            cond,
            PerpNegSamplingGuidance(uncond, empty, 3.0, 0.5),
            sampler=DINKSTER_EULER_CFG_PP,
        )
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        empty_p, neg_p, pos_p = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(empty),
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        pos = pos_p - empty_p
        neg = neg_p - empty_p
        perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
        assert torch.equal(got, empty_p + 3.0 * (pos - perp * 0.5))
        assert torch.equal(got_uncond, neg_p)

    def test_per_run_transforms_compose_after_the_combination(self) -> None:
        model = tiny_flux()
        cond, uncond, empty = tiny_cond("p"), tiny_cond("n"), tiny_cond("e")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced + 1.0),)
        )
        x = tiny_latent()
        base = self._wrap(evaluator, cond, PerpNegSamplingGuidance(uncond, empty, 3.0, 0.75))
        composed = self._wrap(
            evaluator,
            cond,
            PerpNegSamplingGuidance(uncond, empty, 3.0, 0.75, (("ext", contribution),)),
        )
        assert torch.equal(composed(x, 0.7), base(x, 0.7) + 1.0)

    def test_transforms_validation_matches_sampling_guidance(self) -> None:
        cond = tiny_cond("n")
        contribution = GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.plus", lambda context: context.reduced),)
        )
        with pytest.raises(TypeError, match="owner id must be a non-empty string"):
            PerpNegSamplingGuidance(cond, cond, 1.0, 1.0, (("", contribution),))
        with pytest.raises(TypeError, match="must be a GuidanceContribution"):
            PerpNegSamplingGuidance(cond, cond, 1.0, 1.0, (("ext", cast("Any", object())),))
        with pytest.raises(TypeError, match="pairs"):
            PerpNegSamplingGuidance(cond, cond, 1.0, 1.0, cast("Any", (("ext",),)))


TINY_SD1 = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(1, 1),
    transformer_depth_output=(1, 1, 1, 1),
    transformer_depth_middle=1,
    context_dim=16,
    use_linear_in_transformer=False,
    num_heads=8,
)
TINY_ADM = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(0, 2),
    transformer_depth_output=(0, 0, 2, 2),
    transformer_depth_middle=2,
    context_dim=24,
    use_linear_in_transformer=True,
    adm_in_channels=12,
    num_head_channels=16,
)
SD_SPACE = DiscreteSigmas.linear_beta()


def tiny_unet(config: UNetConfig = TINY_SD1) -> UNetModel:
    model = UNetModel(config)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def sd_cond(
    name: str, tokens: int = 3, features: int = TINY_SD1.context_dim
) -> Conditioning[torch.Tensor]:
    return Conditioning(embeddings=hashed_input(f"{name}:ctx", (1, tokens, features)))


def sd_latent() -> torch.Tensor:
    return hashed_input("sd:latent", (1, 4, 8, 8))


def sd_guided(
    evaluator: SDDenoiser,
    cond: Conditioning[torch.Tensor],
    uncond: Conditioning[torch.Tensor] | None,
    cfg_scale: float,
    sampler: SamplerDescriptor = DINKSTER_EULER,
    adm: Callable[[Conditioning[torch.Tensor], GuidanceRole], torch.Tensor | None] | None = None,
) -> GuidedDenoiser:
    def prepare(value: object, role: GuidanceRole):
        assert isinstance(value, Conditioning)
        return evaluator.prepare_conditioning(
            value,
            adm=None if adm is None else adm(value, role),
        )

    return guided_denoiser(
        ConditioningEvaluation(
            prepare,
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            standard_activation_memory_factor=1.0,
        ),
        input=sd_latent(),
        executor=None,
        plan=compile_guidance_plan(
            cond,
            SamplingGuidance(uncond, cfg_scale),
            sampler,
            None,
        ),
        execution=execution(),
    )


class TestSDGuidedDenoiserParity:
    """Shared guidance preserves SD evaluation order and values."""

    def test_cfg_above_one_matches_the_fused_plain_path(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(
            sd_guided(evaluator, cond, uncond, 3.0)(x, 0.7),
            cfg_combine(positive, negative, 3.0),
        )

    def test_cfg_one_matches_the_single_forward_optimization(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, cond, compute_dtype=torch.float32)
        x = sd_latent()
        out = sd_guided(evaluator, cond, uncond, 1.0)(x, 0.7)
        assert torch.equal(out, evaluator(x, 0.7))

    def test_no_uncond_matches_the_plain_path(self) -> None:
        model = tiny_unet()
        cond = sd_cond("p")
        evaluator = SDDenoiser(model, SD_SPACE, cond, compute_dtype=torch.float32)
        x = sd_latent()
        assert torch.equal(sd_guided(evaluator, cond, None, 3.0)(x, 0.7), evaluator(x, 0.7))

    def test_unfused_token_lengths_match_the_two_forward_order(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p", tokens=3), sd_cond("n", tokens=17)
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        positive = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        negative = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(uncond))
        assert torch.equal(
            sd_guided(evaluator, cond, uncond, 3.0)(x, 0.7),
            cfg_combine(positive, negative, 3.0),
        )

    def test_cfg_pp_call_with_uncond_matches(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        wrapped = sd_guided(evaluator, cond, uncond, 3.0, sampler=DINKSTER_EULER_CFG_PP)
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(got, cfg_combine(positive, negative, 3.0))
        assert torch.equal(got_uncond, negative)

    def test_cfg_pp_without_uncond_matches_the_synthetic_zero(self) -> None:
        model = tiny_unet()
        cond = sd_cond("p")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        wrapped = sd_guided(evaluator, cond, None, 3.0, sampler=DINKSTER_EULER_CFG_PP)
        got, got_uncond = wrapped.call_with_uncond(x, 0.7)
        positive = evaluator.evaluate_conditioning(x, 0.7, evaluator.prepare_conditioning(cond))
        assert torch.equal(got, cfg_combine(positive, torch.zeros_like(positive), 3.0))
        assert torch.equal(got_uncond, torch.zeros_like(got))

    def test_adm_resolver_lanes_match_the_plain_path(self) -> None:
        model = tiny_unet(TINY_ADM)
        cond = sd_cond("p", features=TINY_ADM.context_dim)
        uncond = sd_cond("n", features=TINY_ADM.context_dim)
        adm_cond = hashed_input("adm:p", (1, 12))
        adm_uncond = hashed_input("adm:n", (1, 12))

        def resolver(conditioning: Conditioning[torch.Tensor], role: GuidanceRole) -> torch.Tensor:
            return adm_uncond if role is GuidanceRole.UNCONDITIONAL else adm_cond

        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond, adm=adm_uncond),
                evaluator.prepare_conditioning(cond, adm=adm_cond),
            ),
        )
        assert torch.equal(
            sd_guided(evaluator, cond, uncond, 3.0, adm=resolver)(x, 0.7),
            cfg_combine(positive, negative, 3.0),
        )

    def test_inpaint_input_composition_matches_the_plain_path(self) -> None:
        model = tiny_unet(replace(TINY_SD1, in_channels=9))
        cond, uncond = sd_cond("p"), sd_cond("n")
        mask = hashed_input("inpaint:mask", (1, 1, 8, 8)).clamp(0, 1)
        masked_image = hashed_input("inpaint:image", (1, 4, 8, 8))
        evaluator = SDDenoiser(
            model,
            SD_SPACE,
            inpaint_mask=mask,
            inpaint_masked_image=masked_image,
            compute_dtype=torch.float32,
        )
        x = sd_latent()
        negative, positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        assert torch.equal(
            sd_guided(evaluator, cond, uncond, 3.0)(x, 0.7),
            cfg_combine(positive, negative, 3.0),
        )


class TestAttentionGuidance:
    """Attention-kind contributions rewrite attn1 outputs inside the SD
    fused forward and refuse everywhere they cannot execute."""

    @staticmethod
    def _evaluation(evaluator: SDDenoiser, opt_in: bool = True) -> ConditioningEvaluation[Any]:
        return ConditioningEvaluation(
            lambda value, _role: evaluator.prepare_conditioning(cast("Any", value)),
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            evaluate_batch_attention=(
                evaluator.evaluate_conditioning_batch_attention if opt_in else None
            ),
            standard_activation_memory_factor=1.0,
        )

    def _wrap(
        self,
        evaluator: SDDenoiser,
        cond: Conditioning[torch.Tensor],
        uncond: Conditioning[torch.Tensor] | None,
        cfg_scale: float,
        transforms: tuple[tuple[str, GuidanceContribution[torch.Tensor]], ...],
        opt_in: bool = True,
    ) -> GuidedDenoiser:
        return guided_denoiser(
            self._evaluation(evaluator, opt_in),
            input=sd_latent(),
            executor=None,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, cfg_scale, transforms),
                DINKSTER_EULER,
                None,
            ),
            execution=execution(),
        )

    def test_nag_matches_the_manually_rewritten_fused_forward(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        contribution = guidance_transforms.nag(5.0, 0.5, 1.5)
        descriptor = contribution.attention
        assert descriptor is not None
        negative, positive = evaluator.evaluate_conditioning_batch_attention(
            x,
            0.7,
            (evaluator.prepare_conditioning(uncond), evaluator.prepare_conditioning(cond)),
            (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
            (descriptor,),
        )
        wrapped = self._wrap(evaluator, cond, uncond, 3.0, (("run", contribution),))
        assert torch.equal(wrapped(x, 0.7), cfg_combine(positive, negative, 3.0))
        plain_negative, plain_positive = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (evaluator.prepare_conditioning(uncond), evaluator.prepare_conditioning(cond)),
        )
        assert torch.equal(negative, plain_negative)
        assert not torch.equal(positive, plain_positive)

    def test_lane_rows_follow_roles_not_batch_positions(self) -> None:
        """With the conditional lane in either batch position, the
        unconditional rows stay bit-identical to the plain fused
        forward in the same order and the conditional rows move: the
        row mapping is derived from roles, not batch positions.
        Cross-order bit-equality is not asserted because batch position
        legitimately changes float accumulation order."""
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        x = sd_latent()
        descriptor = guidance_transforms.nag(5.0, 0.5, 1.5).attention
        assert descriptor is not None
        positive_first, negative_first = evaluator.evaluate_conditioning_batch_attention(
            x,
            0.7,
            (evaluator.prepare_conditioning(cond), evaluator.prepare_conditioning(uncond)),
            (GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL),
            (descriptor,),
        )
        plain_positive_first, plain_negative_first = evaluator.evaluate_conditioning_batch(
            x,
            0.7,
            (evaluator.prepare_conditioning(cond), evaluator.prepare_conditioning(uncond)),
        )
        assert torch.equal(negative_first, plain_negative_first)
        assert not torch.equal(positive_first, plain_positive_first)

    def test_requires_uncond_evaluates_the_negative_lane_at_cfg_one(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        prepared_roles: list[GuidanceRole] = []

        def prepare(value: object, role: GuidanceRole) -> object:
            prepared_roles.append(role)
            return evaluator.prepare_conditioning(cast("Any", value))

        x = sd_latent()
        wrapped = guided_denoiser(
            ConditioningEvaluation(
                prepare,
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                evaluate_batch_attention=evaluator.evaluate_conditioning_batch_attention,
                standard_activation_memory_factor=1.0,
            ),
            input=x,
            executor=None,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, 1.0, (("run", guidance_transforms.nag(5.0, 0.5, 1.5)),)),
                DINKSTER_EULER,
                None,
            ),
            execution=execution(),
        )
        descriptor = guidance_transforms.nag(5.0, 0.5, 1.5).attention
        assert descriptor is not None
        negative, positive = evaluator.evaluate_conditioning_batch_attention(
            x,
            0.7,
            (evaluator.prepare_conditioning(uncond), evaluator.prepare_conditioning(cond)),
            (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
            (descriptor,),
        )
        assert torch.equal(wrapped(x, 0.7), cfg_combine(positive, negative, 1.0))
        assert GuidanceRole.UNCONDITIONAL in prepared_roles

    def test_refuses_without_family_opt_in(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        with pytest.raises(GuidanceContractError, match="has not opted in"):
            self._wrap(
                evaluator,
                cond,
                uncond,
                3.0,
                (("run", guidance_transforms.nag(5.0, 0.5, 1.5)),),
                opt_in=False,
            )

    def test_refuses_a_missing_unconditional_lane(self) -> None:
        model = tiny_unet()
        cond = sd_cond("p")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        wrapped = self._wrap(
            evaluator, cond, None, 3.0, (("run", guidance_transforms.nag(5.0, 0.5, 1.5)),)
        )
        with pytest.raises(GuidanceContractError, match="unconditional"):
            wrapped(sd_latent(), 0.7)

    def test_refuses_unfused_lanes(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p", tokens=3), sd_cond("n", tokens=17)
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        wrapped = self._wrap(
            evaluator, cond, uncond, 3.0, (("run", guidance_transforms.nag(5.0, 0.5, 1.5)),)
        )
        with pytest.raises(GuidanceContractError, match="one fused forward"):
            wrapped(sd_latent(), 0.7)

    def test_refuses_replica_evaluation(self) -> None:
        model = tiny_unet()
        cond, uncond = sd_cond("p"), sd_cond("n")
        evaluator = SDDenoiser(model, SD_SPACE, compute_dtype=torch.float32)
        with pytest.raises(GuidanceContractError, match="replica-evaluated"):
            guided_denoiser(
                self._evaluation(evaluator),
                input=sd_latent(),
                executor=None,
                plan=compile_guidance_plan(
                    cond,
                    SamplingGuidance(
                        uncond, 3.0, (("run", guidance_transforms.nag(5.0, 0.5, 1.5)),)
                    ),
                    DINKSTER_EULER,
                    None,
                ),
                execution=execution(),
                replica_evaluator_factory=cast("Any", lambda inner: inner),
            )

    def test_nag_transform_math(self) -> None:
        positive = hashed_input("nag:pos", (2, 3, 4))
        negative = hashed_input("nag:neg", (2, 3, 4))
        no_clamp = guidance_transforms.nag(2.0, 1.0, 1e9).attention
        assert no_clamp is not None
        guided = positive * 2.0 - negative * (2.0 - 1.0)
        assert torch.equal(no_clamp.transform(positive, negative), guided)
        identity = guidance_transforms.nag(2.0, 0.0, 1e9).attention
        assert identity is not None
        assert torch.equal(identity.transform(positive, negative), positive)
        clamped = guidance_transforms.nag(2.0, 1.0, 1.0).attention
        assert clamped is not None
        out = clamped.transform(positive, negative)
        norm_pos = torch.norm(positive, p=1, dim=-1, keepdim=True).clamp_min(1e-6)
        norm_out = torch.norm(out, p=1, dim=-1, keepdim=True)
        assert torch.all(norm_out <= norm_pos * (1.0 + 1e-5))

    def test_context_applies_descriptors_in_order_and_leaves_negative_rows_untouched(
        self,
    ) -> None:
        out = hashed_input("ctx:out", (4, 2, 3))
        original = out.clone()
        first = AttentionGuidanceDescriptor[torch.Tensor](
            "x.plus", lambda positive, negative: positive + negative
        )
        second = AttentionGuidanceDescriptor[torch.Tensor](
            "x.double", lambda positive, negative: positive * 2.0
        )
        context = AttentionGuidanceContext((first, second), (0, 2), (2, 4))
        result = context.apply(out)
        assert result is out
        assert torch.equal(result[2:4], original[2:4])
        assert torch.equal(result[0:2], (original[0:2] + original[2:4]) * 2.0)


@pytest.mark.parametrize("distributed_mode", (None, "auto", "guidance", "sequence", "window"))
@pytest.mark.parametrize("world_size", (2, 3))
@pytest.mark.parametrize("cfg_scale", (1.0, 2.0))
def test_sampling_execution_owns_masks_denoise_range_cancellation_previews_and_distribution(
    monkeypatch: pytest.MonkeyPatch,
    distributed_mode: str | None,
    world_size: int,
    cfg_scale: float,
) -> None:
    from dinkster_inference import SamplingCancelled, SigmaSpace
    from dinkster_inference_torch import distributed
    from dinkster_inference_torch.sampling_runtime import SingleStreamSamplingRuntime

    initialized: list[object] = []
    transported: list[object] = []

    def process_group() -> Any:
        initialized.append(config)
        return config

    def transport(evaluator: Any, x: Any, sigma: Any, request: Any) -> Any:
        transported.append(request)
        return evaluator.evaluate(x, sigma, request)

    config = (
        None
        if distributed_mode is None
        else distributed.DistributedSamplingConfig(
            rank=0,
            world_size=world_size,
            mode=distributed_mode,
            rendezvous="test",
            token="test",
        )
    )
    monkeypatch.setattr(distributed, "distributed_sampling_config", lambda: config)
    monkeypatch.setattr(distributed, "ensure_process_group", process_group)
    monkeypatch.setattr(distributed.DistributedGuidanceEvaluator, "evaluate_request", transport)

    class DenoiserAdapter:
        evaluator_identity = "test.synthetic.conditioning.v1"

        @staticmethod
        def prepare_conditioning(value: object, _role: GuidanceRole) -> object:
            return value

        @staticmethod
        def evaluate_conditioning(
            value: torch.Tensor, _sigma: float, context: object
        ) -> torch.Tensor:
            condition = cast("Conditioning[torch.Tensor]", context)
            return value * 0.5 + condition.embeddings.mean()

        @staticmethod
        def batchable(_conditions: tuple[object, ...]) -> bool:
            return True

        def evaluate_batch(
            self,
            value: torch.Tensor,
            sigma: float,
            conditions: tuple[object, ...],
            _context: object | None = None,
        ) -> tuple[torch.Tensor, ...]:
            return tuple(self.evaluate_conditioning(value, sigma, item) for item in conditions)

        evaluate_conditioning_batch = evaluate_batch

    def denoiser(
        _runtime: object,
        _dtype: torch.dtype,
        _context: SamplingAdapterContext,
    ) -> SamplingDenoiserAdapter:
        return cast("SamplingDenoiserAdapter", DenoiserAdapter())

    class SeamRuntime(SingleStreamSamplingRuntime):
        supports_denoised_capture = True
        sampling_execution_registration = SamplingExecutionRegistration(
            latent=SingleStreamLatentAdapter(lambda _latent: None),
            denoiser=denoiser,
            device=lambda _runtime: torch.device("cpu"),
            compute_dtype=lambda _runtime: torch.float32,
            flow=True,
        )

        def __init__(self) -> None:
            self._samplers = torch_sampler_registry()
            self._schedulers = torch_scheduler_registry()
            self._guidance = None

        @property
        def family(self) -> Any:
            return replace(
                FLUX_DEV,
                id="test.synthetic-family",
                latent=replace(FLUX_DEV.single_stream_latent(), scale_factor=1.0, shift_factor=0.0),
            )

        def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
            return FlowSigmas()

        sample_custom = sampling_execution

    runtime = SeamRuntime()
    steps: list[object] = []
    states: list[object] = []
    latent = torch.zeros((1, 16, 2, 2))
    mask = torch.ones_like(latent)
    mask[..., 0] = 0

    def sample() -> torch.Tensor:
        return runtime.sample(
            latent,
            cond=Conditioning(torch.ones((1, 2, 8)), None),
            cfg=SamplingGuidance(Conditioning(torch.zeros((1, 2, 8)), None), cfg_scale),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=2,
            denoise_mask=mask,
            on_step=steps.append,
            on_state=states.append,
        )

    output = sample()
    assert torch.equal(output[..., 0], latent[..., 0])
    assert torch.count_nonzero(output[..., 1])
    assert len(steps) == 2
    assert len(states) == 2
    unmasked_full_denoise = runtime.sample(
        latent,
        cond=Conditioning(torch.ones((1, 2, 8)), None),
        cfg=SamplingGuidance(Conditioning(torch.zeros((1, 2, 8)), None), cfg_scale),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
    )
    partial_denoise = runtime.sample(
        latent,
        cond=Conditioning(torch.ones((1, 2, 8)), None),
        cfg=SamplingGuidance(Conditioning(torch.zeros((1, 2, 8)), None), cfg_scale),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=0.5,
    )
    assert not torch.equal(partial_denoise, unmasked_full_denoise)

    cancelled = False

    def cancellation_requested() -> bool:
        return cancelled

    def cancel_after_first_step(_event: object) -> None:
        nonlocal cancelled
        cancelled = True

    with use_sampling_environment((), cancellation_requested):
        with pytest.raises(SamplingCancelled, match="cancelled"):
            runtime.sample(
                latent,
                cond=Conditioning(torch.ones((1, 2, 8)), None),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=2,
                on_step=cancel_after_first_step,
            )
    if distributed_mode is None:
        assert not initialized and not transported
    else:
        assert initialized and transported
        config = None
        assert torch.equal(output, sample())
