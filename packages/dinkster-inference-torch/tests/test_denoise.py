# pyright: basic
"""Stage 5: the classic-Flux denoise/runtime bridge.

FluxDenoiser and run_denoise are bridges over already-pinned parts
(the native Flux transformer is golden-pinned in test_flux.py, the
prediction/CFG/step math in the torch-free suites), so these tests
pin the BRIDGE semantics against the reference behaviors they port:
prepare_noise's exact draw, the CPU seed bump in the step-noise
sampler, distilled guidance vs external CFG, the batched-CFG
equivalence, the empty-latent process-in guard, and the float32
sampling-state contract.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest
import torch
from dinkster_inference import (
    FLUX_DEV,
    SD15,
    SDXL,
    CancellationToken,
    Conditioning,
    CustomSamplingRequest,
    Denoiser,
    FluxConfig,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePostCFGDescriptor,
    GuidancePreCFGDescriptor,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    MultiStreamLatent,
    NoiseKind,
    Parameterization,
    ProgressScope,
    SamplerInfo,
    SamplingCancelled,
    SamplingDescriptor,
    SamplingExecutionContext,
    SamplingStateEvent,
    SamplingTimelineSchedule,
    StepCallback,
    StepEvent,
    UncondDenoiser,
)
from dinkster_inference_torch import (
    FLUX_GUIDANCE_DEFAULT,
    BrownianTreeNoise,
    DenoiseError,
    EasyCacheConfig,
    Flux,
    FluxDenoiser,
    GaussianNoise,
    GuidanceExecutor,
    GuidanceRegistry,
    LazyCacheConfig,
    latent_process_in,
    latent_process_out,
    prepare_multistream_noise,
    prepare_noise,
    run_denoise,
    run_sampler_engine,
)
from dinkster_inference_torch import distributed as distributed_module
from dinkster_inference_torch._portable_solvers import (
    DINKSTER_DDIM,
    DINKSTER_DPMPP_2M_SDE,
    DINKSTER_EULER,
)
from dinkster_inference_torch.cfg import cfg_combine
from dinkster_inference_torch.denoise import (
    _InpaintDenoiser,
    _noise_scaling,
    prepare_denoise_mask,
    to_batch,
)
from dinkster_inference_torch.guidance import ConditioningEvaluation, GuidedDenoiser
from dinkster_inference_torch.parameterizations import calculate_denoised, noise_scaling
from dinkster_protocol import GuidancePhaseParticipation
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


def tiny_flux(*, guidance_embed: bool = True) -> Flux:
    config = TINY if guidance_embed else replace(TINY, guidance_embed=False)
    model = Flux(config)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def tiny_vector_free_flux() -> Flux:
    config = replace(
        TINY,
        vec_in_dim=None,
        guidance_embed=False,
        txt_norm=True,
        yak_mlp=True,
        txt_ids_dims=(1, 2),
    )
    model = Flux(config)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True, assign=True)
    return model


def tiny_cond(name: str, batch: int = 1, tokens: int = 3) -> Conditioning[torch.Tensor]:
    pooled_width = TINY.vec_in_dim
    assert pooled_width is not None
    return Conditioning(
        embeddings=hashed_input(f"{name}:ctx", (batch, tokens, TINY.context_in_dim)),
        pooled=hashed_input(f"{name}:y", (batch, pooled_width)),
    )


def tiny_latent(batch: int = 1) -> torch.Tensor:
    return hashed_input("latent", (batch, TINY.in_channels, 8, 8))


def guidance_execution() -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((1.0, 0.0), 0, 0, 1.0, 7, token, ProgressScope(token), {})


class TestPrepareNoise:
    def test_matches_reference_draw(self) -> None:
        """Identical values to the reference's global-seed draw
        (comfy/sample.py prepare_noise_inner @ 947c2749: float32 CPU
        randn, then cast)."""
        latent = torch.zeros(2, 4, 8, 8)
        state = torch.random.get_rng_state()
        torch.manual_seed(1234)
        reference = torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            device="cpu",
        )
        torch.random.set_rng_state(state)
        assert torch.equal(prepare_noise(latent, 1234), reference)

    def test_leaves_global_rng_untouched(self) -> None:
        state = torch.random.get_rng_state()
        prepare_noise(torch.zeros(1, 4, 4, 4), 7)
        assert torch.equal(torch.random.get_rng_state(), state)

    def test_casts_to_latent_dtype(self) -> None:
        noise = prepare_noise(torch.zeros(1, 4, 4, 4, dtype=torch.float16), 7)
        assert noise.dtype == torch.float16
        f32 = prepare_noise(torch.zeros(1, 4, 4, 4), 7)
        assert torch.equal(noise, f32.to(torch.float16))


class TestPrepareMultistreamNoise:
    def test_one_generator_draws_streams_in_role_order(self) -> None:
        """One CPU generator seeded once, per-stream draws in role
        order (comfy/sample.py prepare_noise @ b78cec87 over a nested
        latent), matching the multistream family runtimes' internal
        draw."""
        latent = MultiStreamLatent.from_pairs(
            (("video", torch.zeros(1, 4, 2, 8, 8)), ("audio", torch.zeros(1, 8, 16)))
        )
        noise = prepare_multistream_noise(latent, 1234)
        assert noise.roles == ("video", "audio")
        generator = torch.Generator("cpu")
        generator.manual_seed(1234)
        for role, shape in (("video", (1, 4, 2, 8, 8)), ("audio", (1, 8, 16))):
            expected = torch.randn(
                shape,
                dtype=torch.float32,
                layout=torch.strided,
                generator=generator,
                device="cpu",
            )
            assert torch.equal(noise.by_role(role), expected)

    def test_second_stream_continues_the_first_draw(self) -> None:
        latent = MultiStreamLatent.from_pairs(
            (("a", torch.zeros(1, 4, 4, 4)), ("b", torch.zeros(1, 4, 4, 4)))
        )
        noise = prepare_multistream_noise(latent, 7)
        assert torch.equal(noise.by_role("a"), prepare_noise(torch.zeros(1, 4, 4, 4), 7))
        assert not torch.equal(noise.by_role("b"), noise.by_role("a"))

    def test_noise_inds_apply_to_every_stream(self) -> None:
        latent = MultiStreamLatent.from_pairs(
            (("a", torch.zeros(2, 4, 4, 4)), ("b", torch.zeros(2, 8, 2)))
        )
        noise = prepare_multistream_noise(latent, 11, (0, 0))
        for role in ("a", "b"):
            drawn = noise.by_role(role)
            assert torch.equal(drawn[0], drawn[1])

    def test_leaves_global_rng_untouched(self) -> None:
        state = torch.random.get_rng_state()
        prepare_multistream_noise(
            MultiStreamLatent.from_pairs((("only", torch.zeros(1, 4, 4, 4)),)), 7
        )
        assert torch.equal(torch.random.get_rng_state(), state)

    def test_low_precision_streams_never_round_the_draw(self) -> None:
        """Draws happen against float32 views of the streams, so
        half-precision latents get the same float32 noise as float32
        latents of the same shape."""
        low = MultiStreamLatent.from_pairs(
            (
                ("video", torch.zeros(1, 4, 2, 8, 8, dtype=torch.float16)),
                ("audio", torch.zeros(1, 8, 16, dtype=torch.bfloat16)),
            )
        )
        full = MultiStreamLatent.from_pairs(
            (("video", torch.zeros(1, 4, 2, 8, 8)), ("audio", torch.zeros(1, 8, 16)))
        )
        noise = prepare_multistream_noise(low, 1234)
        expected = prepare_multistream_noise(full, 1234)
        for role in ("video", "audio"):
            drawn = noise.by_role(role)
            assert drawn.dtype is torch.float32
            assert torch.equal(drawn, expected.by_role(role))


class TestGaussianNoise:
    def test_cpu_seed_bump_matches_reference(self) -> None:
        """The reference bumps the seed by one on CPU so the step
        stream never replays prepare_noise's draw
        (default_noise_sampler @ 947c2749)."""
        like = torch.zeros(1, 4, 8, 8)
        sampler = GaussianNoise(like, seed=41)
        generator = torch.Generator("cpu")
        generator.manual_seed(42)
        expected = torch.randn(like.shape, generator=generator)
        assert torch.equal(sampler(1.0, 0.5), expected)

    def test_deterministic_and_fresh_per_step(self) -> None:
        like = torch.zeros(2, 4, 4, 4)
        a = GaussianNoise(like, seed=9)
        b = GaussianNoise(like, seed=9)
        first_a = a(1.0, 0.5)
        first_b = b(1.0, 0.5)
        assert torch.equal(first_a, first_b)
        assert not torch.equal(a(0.5, 0.2), first_a)


class TestLatentAffine:
    def test_roundtrip_flux_descriptor(self) -> None:
        latent = tiny_latent()
        descriptor = FLUX_DEV.single_stream_latent()
        processed = latent_process_in(latent, descriptor)
        expected = (latent - descriptor.shift_factor) * descriptor.scale_factor
        assert torch.equal(processed, expected)
        back = latent_process_out(processed, descriptor)
        assert torch.allclose(back, latent, atol=1e-6)


class TestFluxDenoiser:
    def test_real_provider_guidance_phases_match_baseline_and_fuse(self) -> None:
        model = tiny_flux()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        cfg, x, sigma = 4.0, tiny_latent(), 0.6
        denoiser = FluxDenoiser(model, compute_dtype=torch.float32)
        conditioning = ConditioningEvaluation(
            lambda value, _role: denoiser.prepare_conditioning(value),
            denoiser.evaluate_conditioning,
            denoiser.batchable,
            denoiser.evaluate_conditioning_batch,
            standard_activation_memory_factor=1.0,
        )
        baseline_uncond, baseline_cond = denoiser.evaluate_conditioning_batch(
            x,
            sigma,
            (
                denoiser.prepare_conditioning(uncond),
                denoiser.prepare_conditioning(cond),
            ),
        )
        baseline = cfg_combine(baseline_cond, baseline_uncond, cfg), baseline_uncond
        trace: list[str] = []

        def wrapper(request, next):
            trace.append("wrapper-enter")
            result = next(request)
            trace.append("wrapper-exit")
            return result

        contribution = GuidanceContribution(
            evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("test.wrapper", wrapper),),
            pre_cfg=(
                GuidancePreCFGDescriptor(
                    "test.pre",
                    lambda context: trace.append("pre") or context.predictions,
                ),
            ),
            strategy=GuidanceStrategyDescriptor(
                "test.strategy",
                lambda context: GuidanceEvaluationPlan(context.conditions, "cond", "uncond"),
                lambda context: (
                    trace.append("reducer")
                    or cfg_combine(
                        context.predictions.items[0].value,
                        context.predictions.items[1].value,
                        context.cfg_scale,
                    )
                ),
                participation=GuidancePhaseParticipation.COMPOSE,
            ),
            post_cfg=(
                GuidancePostCFGDescriptor(
                    "test.post", lambda context: trace.append("post") or context.reduced
                ),
            ),
        )
        guided = GuidedDenoiser(
            conditioning,
            GuidanceExecutor(GuidanceRegistry((("test", contribution),))),
            (
                GuidanceCondition("cond", GuidanceRole.CONDITIONAL, cond),
                GuidanceCondition("uncond", GuidanceRole.UNCONDITIONAL, uncond),
            ),
            cfg_scale=cfg,
            force_uncond=True,
            input=x,
            execution=guidance_execution(),
        )
        seen: list[torch.Tensor] = []
        handle = model.register_forward_pre_hook(
            lambda _module, args: seen.append(args[2].detach().clone())
        )
        try:
            actual = guided.call_with_uncond(x, sigma)
        finally:
            handle.remove()
        assert trace == ["wrapper-enter", "wrapper-exit", "pre", "reducer", "post"]
        assert torch.equal(actual[0], baseline[0])
        assert torch.equal(actual[1], baseline[1])
        assert len(seen) == 1
        first, second = seen[0].chunk(2)
        assert torch.equal(first, uncond.embeddings)
        assert torch.equal(second, cond.embeddings)

    def test_real_provider_guidance_missing_uncond_is_exact_zero(self) -> None:
        cond = tiny_cond("p")
        denoiser = FluxDenoiser(tiny_flux(), compute_dtype=torch.float32)
        conditioning = ConditioningEvaluation(
            lambda value, _role: denoiser.prepare_conditioning(value),
            denoiser.evaluate_conditioning,
        )
        x, sigma = tiny_latent(), 0.5
        guided = GuidedDenoiser(
            conditioning,
            GuidanceExecutor(GuidanceRegistry()),
            (
                GuidanceCondition("cond", GuidanceRole.CONDITIONAL, cond),
                GuidanceCondition("uncond", GuidanceRole.UNCONDITIONAL, None),
            ),
            cfg_scale=4.0,
            force_uncond=True,
            input=x,
            execution=guidance_execution(),
        )
        actual = guided.call_with_uncond(x, sigma)
        conditional = denoiser.evaluate_conditioning(x, sigma, denoiser.prepare_conditioning(cond))
        baseline = (
            cfg_combine(conditional, torch.zeros_like(conditional), 4.0),
            torch.zeros_like(conditional),
        )
        assert torch.equal(actual[0], baseline[0])
        assert torch.equal(actual[1], baseline[1])
        assert torch.equal(actual[1], torch.zeros_like(actual[1]))

    def test_is_calculate_denoised_over_the_model(self) -> None:
        """cfg=1: exactly the reference _apply_model composition for
        the FLOW path - identity input scaling, sigma-as-timestep,
        x - out * sigma."""
        model = tiny_flux()
        cond = tiny_cond("p")
        denoiser = FluxDenoiser(model, cond, compute_dtype=torch.float32)
        x = tiny_latent()
        sigma = 0.7
        out = denoiser(x, sigma)
        assert cond.pooled is not None
        raw = model(
            x,
            torch.full((1,), sigma),
            cond.embeddings,
            cond.pooled,
            torch.full((1,), FLUX_GUIDANCE_DEFAULT),
        )
        expected = calculate_denoised(Parameterization.FLOW, sigma, raw.float(), x)
        assert torch.allclose(out, expected, atol=1e-6)
        assert out.dtype == torch.float32

    def test_guidance_defaults_to_reference_value(self) -> None:
        denoiser = FluxDenoiser(tiny_flux(), tiny_cond("p"))
        assert denoiser.guidance == FLUX_GUIDANCE_DEFAULT
        assert FLUX_GUIDANCE_DEFAULT == 3.5

    def test_schnell_takes_no_guidance(self) -> None:
        schnell = tiny_flux(guidance_embed=False)
        denoiser = FluxDenoiser(schnell, tiny_cond("p"), compute_dtype=torch.float32)
        assert denoiser.guidance is None
        denoiser(tiny_latent(), 0.5)  # runs without a guidance tensor
        with pytest.raises(DenoiseError, match="guidance"):
            FluxDenoiser(schnell, tiny_cond("p"), guidance=4.0)

    def test_vector_free_flux_executes_without_pooled_conditioning(self) -> None:
        model = tiny_vector_free_flux()
        cond = Conditioning(embeddings=hashed_input("vector-free:ctx", (1, 3, TINY.context_in_dim)))
        denoiser = FluxDenoiser(model, cond, compute_dtype=torch.float32)
        assert denoiser.guidance is None
        output = denoiser(tiny_latent(), 0.5)
        assert output.shape == tiny_latent().shape
        assert bool(torch.isfinite(output).all())

    def test_vector_free_flux_does_not_forward_incidental_pooled_value(self) -> None:
        model = tiny_vector_free_flux()
        embeddings = hashed_input("vector-free:ctx", (1, 3, TINY.context_in_dim))
        without = FluxDenoiser(
            model,
            Conditioning(embeddings=embeddings),
            compute_dtype=torch.float32,
        )
        with_incidental = FluxDenoiser(
            model,
            Conditioning(
                embeddings=embeddings,
                pooled=torch.full((1, 1), float("nan")),
            ),
            compute_dtype=torch.float32,
        )
        x = tiny_latent()
        assert torch.equal(without(x, 0.5), with_incidental(x, 0.5))

    def test_compatible_conditions_run_as_one_batch(self) -> None:
        model = tiny_flux()
        cond = tiny_cond("p")
        uncond = tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        x = tiny_latent()
        sigma = 0.6
        conditions = (
            evaluator.prepare_conditioning(uncond),
            evaluator.prepare_conditioning(cond),
        )
        assert evaluator.batchable(conditions)
        actual_uncond, actual_cond = evaluator.evaluate_conditioning_batch(x, sigma, conditions)
        assert torch.allclose(
            actual_cond,
            evaluator.evaluate_conditioning(x, sigma, conditions[1]),
            atol=1e-5,
        )
        assert torch.allclose(
            actual_uncond,
            evaluator.evaluate_conditioning(x, sigma, conditions[0]),
            atol=1e-5,
        )

    def test_fused_batch_runs_uncond_first(self) -> None:
        """calc_cond_batch @ 947c2749 pops a REVERSED candidate list,
        so the physical fused batch is [uncond, cond]; batch position
        is bit-visible under float accumulation, so the order itself
        is the parity surface, not just the combined output."""
        model = tiny_flux()
        cond = tiny_cond("p")
        uncond = tiny_cond("n")
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        seen: list[torch.Tensor] = []
        handle = model.register_forward_pre_hook(lambda _module, args: seen.append(args[2]))
        try:
            evaluator.evaluate_conditioning_batch(
                tiny_latent(),
                0.6,
                (
                    evaluator.prepare_conditioning(uncond),
                    evaluator.prepare_conditioning(cond),
                ),
            )
        finally:
            handle.remove()
        assert len(seen) == 1
        first, second = seen[0].chunk(2)
        assert torch.equal(first, uncond.embeddings)
        assert torch.equal(second, cond.embeddings)

    def test_unequal_token_counts_repeat_to_lcm_and_match_separate_forwards(self) -> None:
        """ComfyUI repeats 2/3-token lanes to six before one model call.

        Independent calls over those normalized contexts prove lane isolation;
        hooks pin the one-versus-two call-count improvement and physical order.
        """
        model = tiny_flux()
        cond = tiny_cond("p", tokens=2)
        uncond = tiny_cond("n", tokens=3)
        evaluator = FluxDenoiser(model, compute_dtype=torch.float32)
        conditions = (
            evaluator.prepare_conditioning(uncond),
            evaluator.prepare_conditioning(cond),
        )
        original = tuple(
            evaluator.evaluate_conditioning(tiny_latent(), 0.6, condition)
            for condition in conditions
        )
        seen: list[torch.Tensor] = []
        handle = model.register_forward_pre_hook(lambda _module, args: seen.append(args[2]))
        try:
            assert evaluator.batchable(conditions)
            actual = evaluator.evaluate_conditioning_batch(tiny_latent(), 0.6, conditions)
            repeated_conditions = (
                (conditions[0][0].repeat(1, 2, 1), conditions[0][1]),
                (conditions[1][0].repeat(1, 3, 1), conditions[1][1]),
            )
            expected = tuple(
                evaluator.evaluate_conditioning(tiny_latent(), 0.6, condition)
                for condition in repeated_conditions
            )
        finally:
            handle.remove()
        assert len(seen) == 3
        assert seen[0].shape[1] == 6
        repeated_uncond, repeated_cond = seen[0].chunk(2)
        assert torch.equal(repeated_uncond, uncond.embeddings.repeat(1, 2, 1))
        assert torch.equal(repeated_cond, cond.embeddings.repeat(1, 3, 1))
        assert torch.allclose(actual[0], expected[0], atol=1e-5)
        assert torch.allclose(actual[1], expected[1], atol=1e-5)
        assert not torch.allclose(actual[0], original[0], atol=1e-5)
        assert not torch.allclose(actual[1], original[1], atol=1e-5)

    def test_token_repeat_factor_beyond_limit_is_not_batchable(self) -> None:
        evaluator = FluxDenoiser(tiny_flux(), compute_dtype=torch.float32)
        at_limit = tuple(
            evaluator.prepare_conditioning(tiny_cond(name, tokens=tokens))
            for name, tokens in (("limit-p", 1), ("limit-n", 4))
        )
        assert evaluator.batchable(at_limit)
        conditions = tuple(
            evaluator.prepare_conditioning(tiny_cond(name, tokens=tokens))
            for name, tokens in (("p", 1), ("n", 5))
        )
        assert not evaluator.batchable(conditions)
        with pytest.raises(DenoiseError, match="incompatible"):
            evaluator.evaluate_conditioning_batch(tiny_latent(), 0.6, conditions)

    def test_reference_latent_shapes_still_control_batching(self) -> None:
        evaluator = FluxDenoiser(tiny_flux(), compute_dtype=torch.float32)
        cond = evaluator.prepare_conditioning(tiny_cond("p", tokens=2))
        uncond = evaluator.prepare_conditioning(tiny_cond("n", tokens=3))
        assert not evaluator.batchable(
            (
                (cond[0], cond[1], (torch.zeros(1, 4, 2, 2),)),
                (uncond[0], uncond[1], (torch.zeros(1, 4, 3, 2),)),
            )
        )

    def test_singleton_conditioning_broadcasts(self) -> None:
        model = tiny_flux()
        denoiser = FluxDenoiser(model, tiny_cond("p", batch=1), compute_dtype=torch.float32)
        out = denoiser(tiny_latent(batch=2), 0.5)
        assert out.shape[0] == 2

    def test_undersized_conditioning_repeats_cyclically(self) -> None:
        """repeat_to_batch_size @ 947c2749: batch 2 -> 3 rides rows
        [0, 1, 0], never an error (the reference resizes ANY source
        batch to the latent batch)."""
        model = tiny_flux()
        cond2 = tiny_cond("p", batch=2)
        assert cond2.pooled is not None
        resized = Conditioning(
            embeddings=cond2.embeddings[torch.tensor([0, 1, 0])],
            pooled=cond2.pooled[torch.tensor([0, 1, 0])],
        )
        x = tiny_latent(batch=3)
        out = FluxDenoiser(model, cond2, compute_dtype=torch.float32)(x, 0.5)
        expected = FluxDenoiser(model, resized, compute_dtype=torch.float32)(x, 0.5)
        assert torch.equal(out, expected)

    def test_oversized_conditioning_truncates(self) -> None:
        """repeat_to_batch_size @ 947c2749: batch 3 -> 2 keeps the
        first two rows."""
        model = tiny_flux()
        cond3 = tiny_cond("p", batch=3)
        assert cond3.pooled is not None
        truncated = Conditioning(embeddings=cond3.embeddings[:2], pooled=cond3.pooled[:2])
        x = tiny_latent(batch=2)
        out = FluxDenoiser(model, cond3, compute_dtype=torch.float32)(x, 0.5)
        expected = FluxDenoiser(model, truncated, compute_dtype=torch.float32)(x, 0.5)
        assert torch.equal(out, expected)

    def test_rejects_malformed_conditioning(self) -> None:
        model = tiny_flux()
        pooled_width = TINY.vec_in_dim
        assert pooled_width is not None
        with pytest.raises(DenoiseError, match="pooled"):
            FluxDenoiser(model, Conditioning(embeddings=torch.zeros(1, 3, TINY.context_in_dim)))
        with pytest.raises(DenoiseError, match="batch x tokens"):
            FluxDenoiser(
                model,
                Conditioning(
                    embeddings=torch.zeros(3, TINY.context_in_dim),
                    pooled=torch.zeros(1, pooled_width),
                ),
            )

    def test_autograd_flows(self) -> None:
        """No hidden inference_mode/no_grad: gradients must flow
        through a denoiser call (training seams stay open)."""
        denoiser = FluxDenoiser(tiny_flux(), tiny_cond("p"), compute_dtype=torch.float32)
        x = tiny_latent().requires_grad_(True)
        out = denoiser(x, 0.5)
        assert out.grad_fn is not None
        out.sum().backward()
        assert x.grad is not None


class TestToBatch:
    """to_batch is repeat_to_batch_size (comfy/utils.py @ 947c2749,
    dim 0): cyclic whole-tensor repeat then truncate. Pinned directly
    because every denoiser family shares this helper."""

    def test_exact_match_is_identity(self) -> None:
        t = torch.arange(6.0).reshape(2, 3)
        assert to_batch(t, 2) is t

    def test_two_to_three_repeats_cyclically(self) -> None:
        t = torch.arange(6.0).reshape(2, 3)
        assert torch.equal(to_batch(t, 3), t[torch.tensor([0, 1, 0])])

    def test_two_to_five_repeats_cyclically(self) -> None:
        t = torch.arange(6.0).reshape(2, 3)
        assert torch.equal(to_batch(t, 5), t[torch.tensor([0, 1, 0, 1, 0])])

    def test_oversized_truncates_to_first_rows(self) -> None:
        t = torch.arange(12.0).reshape(4, 3)
        assert torch.equal(to_batch(t, 2), t[:2])

    def test_higher_rank_repeats_along_batch_only(self) -> None:
        t = torch.arange(24.0).reshape(2, 3, 4)
        out = to_batch(t, 3)
        assert out.shape == (3, 3, 4)
        assert torch.equal(out, t[torch.tensor([0, 1, 0])])


class _CapturingSolver:
    """A SolverFn stub recording its inputs and returning a fixed x."""

    def __init__(self, result: torch.Tensor) -> None:
        self.result = result
        self.x: torch.Tensor | None = None
        self.sigmas: Sequence[float] | None = None
        self.info: SamplerInfo | None = None
        self.noise: object = "unset"

    def __call__(
        self,
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: object = None,
        on_step: StepCallback | None = None,
    ) -> torch.Tensor:
        del denoiser, on_step
        self.x = x
        self.sigmas = sigmas
        self.info = info
        self.noise = noise
        return self.result


class _IdentityDenoiser:
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x


class TestRunDenoise:
    def test_legacy_solver_receives_only_the_public_solver_contract(self) -> None:
        latent = torch.zeros(1, 4, 2, 2)
        solver = _CapturingSolver(latent)
        result = run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(1.0, 0.0),
            family=SD15,
        )
        assert torch.equal(result, latent)

    def test_step_begin_refuses_solver_without_internal_capability(self) -> None:
        latent = torch.zeros(1, 4, 2, 2)
        with pytest.raises(DenoiseError, match="declares the internal on_step_begin capability"):
            run_denoise(
                _IdentityDenoiser(),
                _CapturingSolver(latent),
                latent=latent,
                noise=torch.zeros_like(latent),
                sigmas=(1.0, 0.0),
                family=SD15,
                on_step_begin=lambda _index: None,
            )

    def test_executed_max_denoise_noise_scaling_is_byte_exact_to_comfyui(
        self,
    ) -> None:
        sigma = 4518.763671875
        latent = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
        noise = prepare_noise(latent, 685468484323813)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        expected = noise * torch.sqrt(1.0 + sigma_tensor**2.0)
        expected += latent

        actual = _noise_scaling(
            Parameterization.V_PREDICTION,
            sigma,
            noise,
            latent,
            max_denoise=True,
        )
        portable = noise_scaling(
            Parameterization.V_PREDICTION,
            sigma,
            noise,
            latent,
            max_denoise=True,
        )

        assert torch.equal(actual, expected)
        # Regression-sensitive at the executed zsnr maximum: the portable
        # Python-float square/root rounds the scale differently.
        assert not torch.equal(portable, expected)

    @pytest.mark.parametrize("sigma", [1.3382458686828613, 0.029167160391807556])
    def test_executed_partial_denoise_noise_scaling_is_byte_exact_to_comfyui(
        self, sigma: float
    ) -> None:
        noise = torch.linspace(-4.0, 4.0, 257, dtype=torch.float32)
        latent = torch.linspace(2.0, -2.0, 257, dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        expected = noise * sigma_tensor
        expected += latent

        actual = _noise_scaling(Parameterization.V_PREDICTION, sigma, noise, latent)

        assert torch.equal(actual, expected)

    def test_executed_noise_scaling_leaves_flow_on_portable_path(self) -> None:
        noise = torch.linspace(-1.0, 1.0, 17, dtype=torch.float32)
        latent = torch.linspace(1.0, -1.0, 17, dtype=torch.float32)
        expected = noise_scaling(Parameterization.FLOW, 0.4, noise, latent)
        assert torch.equal(_noise_scaling(Parameterization.FLOW, 0.4, noise, latent), expected)

    def test_image_to_image_flow_uses_only_latent_initialization_and_final_inversion(self) -> None:
        latent = torch.full((1, 4, 1, 1), 2.0)
        noise = torch.full_like(latent, 99.0)
        processed = latent + 3.0
        solver = _CapturingSolver(processed)

        result = run_sampler_engine(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=noise,
            sigmas=(1.0, 0.0),
            parameterization=Parameterization.IMAGE_TO_IMAGE_FLOW,
            sigma_max=1.0,
            process_in=lambda value: value + 3.0,
            process_out=lambda value: value + 7.0,
        )

        assert solver.x is not None
        assert torch.equal(solver.x, processed)
        assert solver.info is not None
        assert solver.info.parameterization is Parameterization.IMAGE_TO_IMAGE_FLOW
        assert torch.equal(result, torch.full_like(latent, 3.0))

    def test_pre_offset_initial_sigma_does_not_change_solver_schedule(self) -> None:
        latent = torch.zeros(1, 4, 2, 2)
        noise = torch.linspace(-1.0, 1.0, latent.numel()).reshape_as(latent)
        solver = _CapturingSolver(latent)
        sigmas = (0.9999857130611196, 0.5, 0.0)

        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=noise,
            sigmas=sigmas,
            initial_sigma=1.0,
            family=FLUX_DEV,
        )

        assert solver.x is not None
        assert torch.equal(solver.x, noise)
        assert solver.sigmas == sigmas
        assert not torch.equal(solver.x, noise * sigmas[0])

    def test_inpaint_mask_blends_each_evaluation_without_rounding(self) -> None:
        latent = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
        noise = torch.full_like(latent, 2.0)
        raw_mask = torch.tensor([[[[0.25, 0.75]]]])
        inner_inputs: list[torch.Tensor] = []
        solver_outputs: list[torch.Tensor] = []

        class Inner:
            def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
                inner_inputs.append(x)
                return torch.full_like(x, 7.0)

        class Solver:
            def __call__(
                self,
                denoiser: Denoiser[torch.Tensor],
                x: torch.Tensor,
                sigmas: Sequence[float],
                info: SamplerInfo,
                *,
                noise: object = None,
                on_step: StepCallback | None = None,
                on_step_begin: Callable[[int], None] | None = None,
            ) -> torch.Tensor:
                del sigmas, info, noise, on_step, on_step_begin
                output = denoiser(x, 0.5)
                solver_outputs.append(output)
                return output

        run_denoise(
            Inner(),
            Solver(),
            latent=latent,
            noise=noise,
            sigmas=(1.0, 0.0),
            family=SD15,
            denoise_mask=raw_mask,
        )
        mask = raw_mask.repeat(1, 4, 2, 1)
        latent_in = latent_process_in(latent, SD15.single_stream_latent())
        initial = noise + latent_in
        source = noise * 0.5 + latent_in
        assert torch.equal(inner_inputs[0], initial * mask + source * (1.0 - mask))
        assert torch.equal(
            solver_outputs[0],
            torch.full_like(latent, 7.0) * mask + latent_in * (1.0 - mask),
        )

    def test_inpaint_fixed_latent_is_opt_in(self) -> None:
        latent = torch.tensor([[[[2.0, 3.0]]]])
        noise = torch.tensor([[[[5.0, 7.0]]]])
        mask = torch.tensor([[[[1.0, 0.0]]]])
        x = torch.zeros_like(latent)
        seen: list[torch.Tensor] = []

        class Inner:
            def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
                del sigma
                seen.append(x)
                return x

        ordinary = _InpaintDenoiser(
            Inner(),
            mask=mask,
            latent=latent,
            noise=noise,
            parameterization=Parameterization.EPS,
        )
        fixed = _InpaintDenoiser(
            Inner(),
            mask=mask,
            latent=latent,
            noise=noise,
            parameterization=Parameterization.EPS,
            fixed_latent=True,
        )

        ordinary(x, 0.5)
        fixed(x, 0.5)

        keep = 1.0 - mask
        assert torch.equal(seen[0], (noise * 0.5 + latent) * keep)
        assert torch.equal(seen[1], latent * keep)

    def test_ddim_random_inpaint_noise_matches_direct_ksampler_reference(
        self,
    ) -> None:
        """DDIM is Euler whose KSamplerX0Inpaint source uses seed+1 noise."""
        latent = torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3) / 10
        initial_noise = prepare_noise(latent, 31)
        random_inpaint_noise = prepare_noise(latent, 32)
        raw_mask = torch.tensor([[[[0.0, 0.5]]]], dtype=torch.float32)
        sigmas = (1.0, 0.6, 0.2, 0.0)

        class Inner:
            def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
                return x * 0.25 + sigma

        actual = run_denoise(
            Inner(),
            DINKSTER_DDIM.build(),
            latent=latent,
            noise=initial_noise,
            inpaint_noise=random_inpaint_noise,
            sigmas=sigmas,
            family=SD15,
            denoise_mask=raw_mask,
        )

        latent_in = latent_process_in(latent, SD15.single_stream_latent())
        mask = torch.nn.functional.interpolate(
            raw_mask,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask = mask.repeat(1, latent.shape[1], 1, 1)
        mask = to_batch(mask, latent.shape[0])

        class DirectKSamplerX0Inpaint:
            def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
                keep = 1.0 - mask
                source = _noise_scaling(
                    Parameterization.EPS,
                    sigma,
                    random_inpaint_noise,
                    latent_in,
                )
                masked = x * mask + source * keep
                return Inner()(masked, sigma) * mask + latent_in * keep

        x = _noise_scaling(
            Parameterization.EPS,
            sigmas[0],
            initial_noise,
            latent_in,
        )
        expected = DINKSTER_EULER.build()(
            DirectKSamplerX0Inpaint(),
            x,
            sigmas,
            SamplerInfo(Parameterization.EPS),
        )
        expected = latent_process_out(expected, SD15.single_stream_latent())
        assert torch.equal(actual, expected)

        ordinary = run_denoise(
            Inner(),
            DINKSTER_DDIM.build(),
            latent=latent,
            noise=initial_noise,
            sigmas=sigmas,
            family=SD15,
            denoise_mask=raw_mask,
        )
        assert not torch.equal(actual, ordinary)

    def test_inpaint_mask_preserves_cfgpp_unconditional_channel(self) -> None:
        latent = torch.ones(1, 4, 2, 2)
        mask = torch.zeros(1, 1, 2, 2)
        seen: list[tuple[torch.Tensor, torch.Tensor]] = []

        class Inner:
            def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
                return x

            def call_with_uncond(
                self, x: torch.Tensor, sigma: float
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.full_like(x, 3.0), torch.full_like(x, 4.0)

        class Solver:
            def __call__(
                self,
                denoiser: Denoiser[torch.Tensor],
                x: torch.Tensor,
                sigmas: Sequence[float],
                info: SamplerInfo,
                *,
                noise: object = None,
                on_step: StepCallback | None = None,
                on_step_begin: Callable[[int], None] | None = None,
            ) -> torch.Tensor:
                del sigmas, info, noise, on_step, on_step_begin
                assert isinstance(denoiser, UncondDenoiser)
                seen.append(denoiser.call_with_uncond(x, 0.5))
                return seen[0][0]

        run_denoise(
            Inner(),
            Solver(),
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(1.0, 0.0),
            family=SD15,
            denoise_mask=mask,
        )
        latent_in = latent_process_in(latent, SD15.single_stream_latent())
        assert torch.equal(seen[0][0], latent_in)
        assert torch.equal(seen[0][1], latent_in)

    def test_empty_sigmas_returns_latent_untouched(self) -> None:
        latent = tiny_latent()
        out = run_denoise(
            _IdentityDenoiser(),
            _CapturingSolver(torch.zeros(1)),
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(),
            family=FLUX_DEV,
        )
        assert out is latent

    def test_shape_mismatch_refuses(self) -> None:
        latent = tiny_latent()
        with pytest.raises(DenoiseError, match="shape"):
            run_denoise(
                _IdentityDenoiser(),
                _CapturingSolver(latent),
                latent=latent,
                noise=torch.zeros(1, 1, 2, 2),
                sigmas=(1.0, 0.0),
                family=FLUX_DEV,
            )

    def test_brownian_kind_builds_reference_bounded_sampler(self) -> None:
        """NoiseKind.BROWNIAN constructs BrownianTreeNoise with the
        reference SDE solvers' bounds (sigmas[sigmas > 0].min(),
        sigmas.max() @ 947c2749) and the run seed: the same tree the
        reference would build, proven by an identically-seeded
        stand-alone sampler producing the identical step draw."""
        latent = torch.zeros(1, 16, 8, 8)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(0.9999, 0.5, 0.02, 0.0),
            family=FLUX_DEV,
            noise_kind=NoiseKind.BROWNIAN,
            seed=11,
        )
        assert isinstance(solver.noise, BrownianTreeNoise)
        twin = BrownianTreeNoise(latent, 0.02, 0.9999, seed=11)
        assert torch.equal(solver.noise(0.9999, 0.5), twin(0.9999, 0.5))

    def test_brownian_one_step_schedule_builds_zero_width_tree(self) -> None:
        """A one-step (sigma, 0) schedule is legal: the reference
        solver constructs BrownianTreeNoiseSampler(x, s, s) - equal
        bounds, a zero-width tree - and never queries it (the
        sigma-to-zero step takes the pure denoising branch)."""
        latent = torch.zeros(1, 16, 8, 8)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(1.0, 0.0),
            family=FLUX_DEV,
            noise_kind=NoiseKind.BROWNIAN,
            seed=5,
        )
        assert isinstance(solver.noise, BrownianTreeNoise)

    def test_brownian_single_entry_schedule_skips_construction(self) -> None:
        """len(sigmas) == 1 mirrors the reference solvers'
        len(sigmas) <= 1 early return: zero steps run, no tree is
        built (the reference returns before constructing the
        sampler), so even an all-zero single-entry schedule does not
        refuse."""
        latent = torch.zeros(1, 16, 8, 8)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(0.0,),
            family=FLUX_DEV,
            noise_kind=NoiseKind.BROWNIAN,
        )
        assert solver.noise is None

    def test_brownian_without_positive_sigma_refuses(self) -> None:
        latent = tiny_latent()
        with pytest.raises(DenoiseError, match="positive sigma"):
            run_denoise(
                _IdentityDenoiser(),
                _CapturingSolver(latent),
                latent=latent,
                noise=torch.zeros_like(latent),
                sigmas=(0.0, 0.0),
                family=FLUX_DEV,
                noise_kind=NoiseKind.BROWNIAN,
            )

    def test_injected_noise_sampler_wins(self) -> None:
        """The noise_sampler override (the reference solvers'
        same-named parameter) bypasses default construction - the
        pre-offset-bounds escape hatch for SNR-offset flow
        schedules."""
        latent = torch.zeros(1, 16, 8, 8)
        injected = BrownianTreeNoise(latent, 0.02, 1.0, seed=11)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(0.9999, 0.5, 0.02, 0.0),
            family=FLUX_DEV,
            noise_kind=NoiseKind.BROWNIAN,
            noise_sampler=injected,
            seed=11,
        )
        assert solver.noise is injected

    def test_zero_latent_skips_process_in(self) -> None:
        """The reference's empty-latent guard: an all-zero latent is
        NOT affine-shifted before noising, so the solver's start
        state is exactly sigma * noise for flow."""
        latent = torch.zeros(1, 16, 8, 8)
        noise = prepare_noise(latent, 3)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=noise,
            sigmas=(1.0, 0.5, 0.0),
            family=FLUX_DEV,
        )
        assert solver.x is not None
        assert torch.allclose(solver.x, noise * 1.0, atol=1e-7)

    def test_nonzero_latent_is_processed_and_noised(self) -> None:
        latent = tiny_latent()
        noise = prepare_noise(latent, 3)
        sigma0 = 0.75
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=noise,
            sigmas=(sigma0, 0.5, 0.0),
            family=FLUX_DEV,
        )
        expected = noise_scaling(
            Parameterization.FLOW,
            sigma0,
            noise,
            latent_process_in(latent, FLUX_DEV.single_stream_latent()),
            max_denoise=False,
        )
        assert solver.x is not None
        assert torch.allclose(solver.x, expected, atol=1e-6)
        assert solver.info is not None
        assert solver.info.parameterization is Parameterization.FLOW

    def test_variant_sampling_controls_parameterization_and_partial_denoise(self) -> None:
        latent = tiny_latent()[:, :4]
        noise = prepare_noise(latent, 3)
        sigma0 = 20.0
        sampling = SamplingDescriptor(
            Parameterization.V_PREDICTION,
            sigma_min=0.029,
            sigma_max=4518.0,
            zsnr=True,
        )
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=noise,
            sigmas=(sigma0, 0.0),
            family=SDXL,
            sampling=sampling,
        )
        expected = noise_scaling(
            Parameterization.V_PREDICTION,
            sigma0,
            noise,
            latent_process_in(latent, SDXL.single_stream_latent()),
            max_denoise=False,
        )
        assert solver.x is not None
        assert torch.allclose(solver.x, expected, atol=1e-6)
        assert solver.info is not None
        assert solver.info.parameterization is Parameterization.V_PREDICTION

    def test_output_is_processed_out(self) -> None:
        """The solver result is inverse-scaled at the terminal sigma
        (identity at 0.0 for flow) and mapped out of model space."""
        latent = torch.zeros(1, 16, 8, 8)
        result = tiny_latent()
        solver = _CapturingSolver(result.clone())
        out = run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(1.0, 0.0),
            family=FLUX_DEV,
        )
        expected_out = latent_process_out(result, FLUX_DEV.single_stream_latent())
        assert torch.allclose(out, expected_out, atol=1e-6)
        assert out.dtype == torch.float32

    def test_gaussian_kind_builds_noise_sampler(self) -> None:
        latent = torch.zeros(1, 16, 8, 8)
        solver = _CapturingSolver(latent.clone())
        run_denoise(
            _IdentityDenoiser(),
            solver,
            latent=latent,
            noise=torch.zeros_like(latent),
            sigmas=(1.0, 0.0),
            family=FLUX_DEV,
            noise_kind=NoiseKind.GAUSSIAN,
            seed=11,
        )
        assert isinstance(solver.noise, GaussianNoise)

    def test_end_to_end_tiny_flux_euler(self) -> None:
        """A real (tiny) txt2img run: FluxDenoiser + euler over a
        3-sigma schedule produces a finite float32 latent-space
        image batch."""
        model = tiny_flux()
        denoiser = FluxDenoiser(
            model,
            tiny_cond("p"),
            compute_dtype=torch.float32,
        )
        latent = torch.zeros(1, 16, 8, 8)
        noise = prepare_noise(latent, 5)
        out = run_denoise(
            denoiser,
            DINKSTER_EULER.build(),
            latent=latent,
            noise=noise,
            sigmas=(1.0, 0.6, 0.3, 0.0),
            family=FLUX_DEV,
            seed=5,
        )
        assert out.shape == latent.shape
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        # Deterministic: the same run repeats bit-identically.
        again = run_denoise(
            denoiser,
            DINKSTER_EULER.build(),
            latent=latent,
            noise=prepare_noise(latent, 5),
            sigmas=(1.0, 0.6, 0.3, 0.0),
            family=FLUX_DEV,
            seed=5,
        )
        assert torch.equal(out, again)

    def test_end_to_end_tiny_flux_dpmpp_2m_sde(self) -> None:
        """A real (tiny) brownian-solver run: FluxDenoiser +
        dpmpp_2m_sde over an SNR-offset flow schedule produces a
        finite float32 batch and replays bit-identically (the
        brownian tree is deterministic in the seed). The step-noise
        stream itself is golden-pinned in test_brownian.py."""
        model = tiny_flux()
        denoiser = FluxDenoiser(
            model,
            tiny_cond("p"),
            compute_dtype=torch.float32,
        )
        latent = torch.zeros(1, 16, 8, 8)
        sigmas = (0.9999, 0.6, 0.3, 0.1, 0.0)

        def run() -> torch.Tensor:
            return run_denoise(
                denoiser,
                DINKSTER_DPMPP_2M_SDE.build(),
                latent=latent,
                noise=prepare_noise(latent, 5),
                sigmas=sigmas,
                family=FLUX_DEV,
                seed=5,
                noise_kind=DINKSTER_DPMPP_2M_SDE.noise,
            )

        out = run()
        assert out.shape == latent.shape
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert torch.equal(out, run())


@pytest.mark.parametrize("batch", (1, 2, 3))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_image_mask_normalization_preserves_reference_layout(
    batch: int, dtype: torch.dtype
) -> None:
    latent = torch.zeros((batch, 4, 3, 5))
    mask = torch.arange(24, dtype=dtype).reshape(2, 3, 2, 2) / 24
    expected = torch.nn.functional.interpolate(
        mask.reshape(-1, 1, 2, 2).float(), size=(3, 5), mode="bilinear", align_corners=False
    ).repeat(1, 4, 1, 1)
    actual = prepare_denoise_mask(mask, latent)
    assert actual is not None
    assert torch.equal(actual, to_batch(expected, batch))


@pytest.mark.parametrize("dimensions", (1, 3))
def test_dense_audio_and_video_masks_resize_repeat_and_preserve_zero_regions(
    dimensions: int,
) -> None:
    shape = (3, 4, *((4,) * dimensions))
    latent = torch.zeros(shape)
    mask = torch.zeros((1, 1, *((2,) * dimensions)))
    mask[..., -1] = 1
    mode = "linear" if dimensions == 1 else "trilinear"
    expected = torch.nn.functional.interpolate(mask, size=shape[2:], mode=mode)
    expected = expected.repeat((3, 4) + (1,) * dimensions)
    actual = prepare_denoise_mask(mask, latent)
    assert actual is not None
    assert torch.equal(actual, expected)
    assert prepare_denoise_mask(None, latent) is None


def test_sampling_timeline_routes_distributed_execution_through_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = distributed_module.DistributedSamplingConfig(
        0, 2, "guidance", "file:///group", "1" * 32
    )
    calls: list[tuple[torch.Size, object]] = []
    monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: config)

    def route(
        _action: Callable[[], torch.Tensor],
        template: torch.Tensor,
        received: object,
    ) -> torch.Tensor:
        calls.append((template.shape, received))
        return torch.full_like(template, 9.0)

    monkeypatch.setattr(distributed_module, "run_rank_zero_sampling", route)
    request = CustomSamplingRequest(
        DINKSTER_EULER,
        (),
        (1.0, 0.0),
        timeline=SamplingTimelineSchedule("sage", 0.0, 1.0),
    )
    latent = torch.zeros(1, 1, 1, 1)

    output = run_sampler_engine(
        _IdentityDenoiser(),
        request.build_solver(),
        latent=latent,
        noise=latent,
        sigmas=request.sigmas,
        parameterization=Parameterization.EPS,
        sigma_max=1.0,
        process_in=lambda value: value,
        process_out=lambda value: value,
    )

    assert torch.equal(output, torch.full_like(latent, 9.0))
    assert calls == [(latent.shape, config)]
    assert "scheduled sampling timeline executes on rank 0" in caplog.text


@pytest.mark.parametrize("cancel", (False, True))
def test_rank_zero_timeline_executes_callback_and_cancellation_on_local_engine(
    cancel: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = CustomSamplingRequest(
        DINKSTER_EULER,
        (),
        (1.0, 0.0),
        timeline=SamplingTimelineSchedule("sage", 0.0, 1.0),
    )
    latent = torch.zeros(1, 1, 1, 1)
    config = distributed_module.DistributedSamplingConfig(
        0, 2, "guidance", "file:///group", "1" * 32
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_sampling_config",
        lambda: None if distributed_module.rank_zero_sampling_active() else config,
    )
    monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: config)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda _tensor, src, group=None: None,
    )
    steps: list[int] = []
    begins: list[int] = []
    states: list[int] = []

    def on_step(event: StepEvent) -> None:
        steps.append(event.step)
        if cancel:
            raise SamplingCancelled("cancelled by rank-zero callback")

    def on_state(event: SamplingStateEvent[object]) -> None:
        states.append(event.step)

    def run() -> torch.Tensor:
        return run_sampler_engine(
            _IdentityDenoiser(),
            request.build_solver(),
            latent=latent,
            noise=latent,
            sigmas=request.sigmas,
            parameterization=Parameterization.EPS,
            sigma_max=1.0,
            process_in=lambda value: value,
            process_out=lambda value: value,
            on_step=on_step,
            on_step_begin=begins.append,
            on_state=on_state,
        )

    if cancel:
        with pytest.raises(SamplingCancelled, match="rank-zero callback"):
            run()
    else:
        assert torch.equal(run(), latent)
    assert steps == [0]
    assert begins == [0]
    assert states == [0]


def test_autoregressive_sampling_routes_distributed_execution_through_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Autoregressive(_IdentityDenoiser):
        def prepare_autoregressive(
            self,
            x: torch.Tensor,
            sigmas: Sequence[float],
            *,
            num_frame_per_block: int,
        ) -> object:
            raise AssertionError("distributed admission must precede model execution")

    config = distributed_module.DistributedSamplingConfig(
        0, 2, "guidance", "file:///group", "1" * 32
    )
    monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: config)
    monkeypatch.setattr(
        distributed_module,
        "run_rank_zero_sampling",
        lambda _action, template, _config: torch.full_like(template, 11.0),
    )
    latent = torch.zeros(1, 1, 1, 1)
    request = CustomSamplingRequest(DINKSTER_EULER, (), (1.0, 0.0))
    output = run_sampler_engine(
        Autoregressive(),
        request.build_solver(),
        latent=latent,
        noise=latent,
        sigmas=request.sigmas,
        parameterization=Parameterization.FLOW,
        sigma_max=1.0,
        process_in=lambda value: value,
        process_out=lambda value: value,
    )

    assert torch.equal(output, torch.full_like(latent, 11.0))
    assert "autoregressive block sampling executes on rank 0" in caplog.text


@pytest.mark.parametrize("cache", (LazyCacheConfig(), EasyCacheConfig()))
def test_sampling_cache_routes_distributed_execution_through_rank_zero(
    cache: object,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = distributed_module.DistributedSamplingConfig(
        0, 2, "guidance", "file:///group", "1" * 32
    )
    monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: config)
    monkeypatch.setattr(
        distributed_module,
        "run_rank_zero_sampling",
        lambda _action, template, _config: torch.full_like(template, 13.0),
    )
    latent = torch.zeros(1, 1, 1, 1)
    request = CustomSamplingRequest(DINKSTER_EULER, (), (1.0, 0.0), cache=cache)  # type: ignore[arg-type]

    output = run_sampler_engine(
        _IdentityDenoiser(),
        request.build_solver(),
        latent=latent,
        noise=latent,
        sigmas=request.sigmas,
        parameterization=Parameterization.EPS,
        sigma_max=1.0,
        process_in=lambda value: value,
        process_out=lambda value: value,
    )

    assert torch.equal(output, torch.full_like(latent, 13.0))
    assert f"{type(cache).__name__} executes on rank 0" in caplog.text
