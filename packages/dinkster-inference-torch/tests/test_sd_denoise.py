# pyright: basic
"""Stage 5: the SD1/SDXL denoise bridge.

SDDenoiser composes already-pinned parts (the golden-pinned UNet in
test_unet.py, the torch-free EPS/CFG/discrete-sigma math), so these
tests pin the BRIDGE semantics against the reference behaviors it
ports: the _apply_model EPS composition (input preconditioning,
nearest-log-sigma timestep lookup, float32 lift), the SDXL/refiner
encode_adm vector layout, CONDCrossAttn's repeat-to-lcm batching
(equivalence within the factor-4 limit, two forwards beyond it), and
the ADM-iff-adm_in_channels validation.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    ContinuousEDMSigmas,
    DiscreteSigmas,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePostCFGDescriptor,
    GuidancePreCFGDescriptor,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    Parameterization,
    ProgressScope,
    SamplingExecutionContext,
    UNetConfig,
    calculate_denoised,
    calculate_input,
    cfg_combine,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    CROSS_ATTN_REPEAT_LIMIT,
    SDXL_AESTHETIC_DEFAULT,
    SDXL_NEGATIVE_AESTHETIC_DEFAULT,
    DenoiseError,
    GuidanceExecutor,
    GuidanceRegistry,
    SDDenoiser,
    UNetModel,
    cross_attn_repeat,
    encode_sdxl_adm,
    encode_sdxl_refiner_adm,
    timestep_embedding,
    torch_scheduler_registry,
)
from dinkster_inference_torch._conditioning_layout import repeat_cross_attn
from dinkster_inference_torch.guidance import ConditioningEvaluation, GuidedDenoiser
from dinkster_inference_torch.schedules import continuous_edm_percent_to_sigma
from dinkster_inference_torch.sd_denoise import (
    _calculate_denoised,
    _calculate_input,
    inpaint_model_input,
)
from dinkster_protocol import GuidancePhaseParticipation
from golden_files import (
    assert_reference_schedule,
    load_platform_golden,
    reference_validation_enabled,
)
from unet_fill import fill_state_dict, hashed_input

VPRED_GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "sdxl_vpred_goldens.json")
EDM_VPRED_GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "sdxl_edm_vpred_goldens.json"
)

# The golden suite's tiny shapes (test_unet.py): an SD1-style conv
# UNet without ADM and an SDXL-style linear UNet with a small ADM
# vector - SDDenoiser only reads adm_in_channels, so the tiny width
# exercises the same codepath as 2816/2560.
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

SPACE = DiscreteSigmas.linear_beta()
EDM_SPACE = ContinuousEDMSigmas(min_sigma=0.125, max_sigma=42.5)
COSXL_SIGMA_MIN = 0.0020000000949949026
COSXL_SIGMA_MAX = 120.0
COSXL_EDM_SPACE = ContinuousEDMSigmas(min_sigma=COSXL_SIGMA_MIN, max_sigma=COSXL_SIGMA_MAX)
_LINUX_COSXL_EFFECTIVE_SIGMAS = (
    120.00000762939453,
    67.25115203857422,
    37.68928909301758,
    21.1220645904541,
    11.837356567382812,
    6.633963108062744,
    3.717846393585205,
    2.083578109741211,
    1.1676915884017944,
    0.654404878616333,
    0.3667454421520233,
    0.2055337131023407,
    0.11518645286560059,
    0.06455349177122116,
    0.03617746755480766,
    0.020274797454476357,
    0.011362524703145027,
    0.00636785663664341,
    0.003568712156265974,
    0.001999999862164259,
)
if sys.platform.startswith("linux"):
    COSXL_EFFECTIVE_SIGMAS = _LINUX_COSXL_EFFECTIVE_SIGMAS
else:
    COSXL_EFFECTIVE_SIGMAS = tuple(EDM_VPRED_GOLDENS["cosxl"]["normal"][:-1])
INPAINT_GOLDEN = load_platform_golden(Path(__file__).parent / "goldens/sd15_inpaint_goldens.json")


def tiny_unet(config: UNetConfig = TINY_SD1) -> UNetModel:
    model = UNetModel(config)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def tiny_cond(
    name: str,
    *,
    batch: int = 1,
    tokens: int = 3,
    features: int = TINY_SD1.context_dim,
    pooled: int | None = None,
) -> Conditioning[torch.Tensor]:
    return Conditioning(
        embeddings=hashed_input(f"{name}:ctx", (batch, tokens, features)),
        pooled=(None if pooled is None else hashed_input(f"{name}:pooled", (batch, pooled))),
    )


def tiny_adm(name: str, batch: int = 1) -> torch.Tensor:
    channels = TINY_ADM.adm_in_channels
    assert channels is not None
    return hashed_input(f"{name}:adm", (batch, channels))


def tiny_latent(batch: int = 1) -> torch.Tensor:
    return hashed_input("latent", (batch, TINY_SD1.in_channels, 8, 8))


def golden_tensor(value: dict[str, object]) -> torch.Tensor:
    data = cast(list[float], value["data"])
    shape = cast(list[int], value["shape"])
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


@pytest.mark.parametrize(
    "case_name",
    (
        "default",
        "resize_round_and_batch",
        "center_crop",
        "resize_batch_distribution",
    ),
)
def test_inpaint_model_input_matches_executed_reference(case_name: str) -> None:
    case = INPAINT_GOLDEN["cases"][case_name]
    noise = golden_tensor(case["noise"])
    mask = None if case["mask"] is None else golden_tensor(case["mask"])
    latent = golden_tensor(case["latent"]) * 0.18215
    actual = inpaint_model_input(
        noise,
        denoise_mask=mask,
        masked_image=latent,
    )
    assert torch.equal(actual, golden_tensor(case["output"]))


@pytest.mark.parametrize("scheduler_name", tuple(EDM_VPRED_GOLDENS["schedules_20"]))
def test_edm_vpred_schedules_match_executed_reference_exactly(
    scheduler_name: str,
) -> None:
    bounds = EDM_VPRED_GOLDENS["bounds"]
    space = ContinuousEDMSigmas(min_sigma=bounds["sigma_min"], max_sigma=bounds["sigma_max"])
    scheduler = torch_scheduler_registry().get(f"dinkster.{scheduler_name}")
    assert scheduler is not None
    assert_reference_schedule(
        sampling_sigmas(scheduler, space, 20),
        EDM_VPRED_GOLDENS["schedules_20"][scheduler_name],
    )


@pytest.mark.parametrize("scheduler_name", tuple(EDM_VPRED_GOLDENS["close_bounds"]["schedules_20"]))
def test_edm_vpred_close_bounds_table_schedules_match_reference(
    scheduler_name: str,
) -> None:
    golden = EDM_VPRED_GOLDENS["close_bounds"]
    space = ContinuousEDMSigmas(min_sigma=golden["sigma_min"], max_sigma=golden["sigma_max"])
    scheduler = torch_scheduler_registry().get(f"dinkster.{scheduler_name}")
    assert scheduler is not None
    assert sampling_sigmas(scheduler, space, 20) == tuple(golden["schedules_20"][scheduler_name])


def test_edm_vpred_percent_to_sigma_matches_reference_exactly() -> None:
    for percent, expected in EDM_VPRED_GOLDENS["percent_to_sigma"].items():
        assert continuous_edm_percent_to_sigma(EDM_SPACE, float(percent)) == expected


def guidance_execution() -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((1.0, 0.0), 0, 0, 1.0, 7, token, ProgressScope(token), {})


# --- encode_adm ------------------------------------------------------------


class TestEncodeAdm:
    def test_sdxl_layout_and_order(self) -> None:
        """SDXL.encode_adm @ 947c2749: pooled then six Timestep(256)
        embeddings in reference order (height, width, crop_h, crop_w,
        target_height, target_width)."""
        pooled = hashed_input("adm:pooled", (1, 1280))
        out = encode_sdxl_adm(
            pooled,
            width=832,
            height=1216,
            crop_w=8,
            crop_h=16,
            target_width=1024,
            target_height=1024,
        )
        assert out.shape == (1, 1280 + 6 * 256)
        expected = torch.cat(
            [
                timestep_embedding(torch.tensor([float(v)]), 256)
                for v in (1216, 832, 16, 8, 1024, 1024)
            ],
            dim=1,
        )
        assert torch.equal(out[:, :1280], pooled.float())
        assert torch.equal(out[:, 1280:], expected)

    def test_sdxl_targets_default_to_the_size(self) -> None:
        explicit = encode_sdxl_adm(
            torch.zeros(1, 8),
            width=640,
            height=480,
            target_width=640,
            target_height=480,
        )
        defaulted = encode_sdxl_adm(torch.zeros(1, 8), width=640, height=480)
        assert torch.equal(explicit, defaulted)

    def test_refiner_layout_and_aesthetic(self) -> None:
        """SDXLRefiner.encode_adm @ 947c2749: five embeddings, the
        last the aesthetic score (6 positive / 2.5 negative)."""
        pooled = hashed_input("ref:pooled", (1, 1280))
        out = encode_sdxl_refiner_adm(pooled, width=512, height=768, aesthetic_score=6.0)
        assert out.shape == (1, 1280 + 5 * 256)
        expected = torch.cat(
            [timestep_embedding(torch.tensor([float(v)]), 256) for v in (768, 512, 0, 0, 6.0)],
            dim=1,
        )
        assert torch.equal(out[:, 1280:], expected)
        assert SDXL_AESTHETIC_DEFAULT == 6.0
        assert SDXL_NEGATIVE_AESTHETIC_DEFAULT == 2.5

    def test_batch_rides_the_pooled(self) -> None:
        out = encode_sdxl_adm(torch.zeros(3, 8))
        assert out.shape == (3, 8 + 6 * 256)

    def test_rejects_non_2d_pooled(self) -> None:
        with pytest.raises(DenoiseError, match="batch x features"):
            encode_sdxl_adm(torch.zeros(8))


# --- cross_attn_repeat ------------------------------------------------------


class TestCrossAttnRepeat:
    def test_equal_lengths_need_no_repeat(self) -> None:
        assert cross_attn_repeat([77, 77]) == [1, 1]

    def test_multiples_repeat_to_the_lcm(self) -> None:
        assert cross_attn_repeat([77, 154]) == [2, 1]
        assert cross_attn_repeat([2, 3]) == [3, 2]
        assert cross_attn_repeat([1, 4]) == [4, 1]

    def test_repeat_tiles_only_the_token_axis(self) -> None:
        context = torch.arange(6).reshape(2, 1, 3)
        assert torch.equal(
            repeat_cross_attn(context, 4),
            torch.cat((context, context, context, context), dim=1),
        )

    def test_factor_beyond_the_limit_refuses(self) -> None:
        """comfy/conds.py can_concat @ 947c2749: lcm // min > 4 falls
        back to separate forwards."""
        assert CROSS_ATTN_REPEAT_LIMIT == 4
        assert cross_attn_repeat([3, 5]) is None
        assert cross_attn_repeat([1, 5]) is None


# --- SDDenoiser: the _apply_model EPS composition ---------------------------


class TestSDDenoiser:
    def test_cosxl_normal_schedule_binds_effective_float32_endpoints(self) -> None:
        """ContinuousEDM properties expose the float32 table endpoints,
        not the raw checkpoint scalars, and normal scheduling consumes those
        effective values (ComfyUI f4b99bc model_sampling.py:199-220 and
        samplers.py:628-650)."""
        reference_table = torch.linspace(
            math.log(COSXL_SIGMA_MIN), math.log(COSXL_SIGMA_MAX), 1000
        ).exp()
        assert float(reference_table[0]) == COSXL_EFFECTIVE_SIGMAS[-1]
        assert float(reference_table[-1]) == COSXL_EFFECTIVE_SIGMAS[0]
        assert float(reference_table[0]) != COSXL_SIGMA_MIN
        assert float(reference_table[-1]) != COSXL_SIGMA_MAX

        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert scheduler is not None
        assert sampling_sigmas(scheduler, COSXL_EDM_SPACE, 20) == (
            *COSXL_EFFECTIVE_SIGMAS,
            0.0,
        )

    def test_cosxl_edm_vprediction_input_is_byte_exact_to_comfyui(self) -> None:
        noise = torch.linspace(-4.0, 4.0, 4 * 64 * 64, dtype=torch.float32).reshape(1, 4, 64, 64)
        portable_differences = 0
        for sigma in COSXL_EFFECTIVE_SIGMAS:
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
            expected = noise / (sigma_tensor**2 + 1.0**2) ** 0.5
            actual = _calculate_input(Parameterization.V_PREDICTION, sigma, noise)
            portable = calculate_input(Parameterization.V_PREDICTION, sigma, noise)
            assert torch.equal(actual, expected)
            portable_differences += not torch.equal(portable, expected)
        assert portable_differences > 0

    @pytest.mark.parametrize(
        "sigma",
        [
            COSXL_EFFECTIVE_SIGMAS[0],
            COSXL_EFFECTIVE_SIGMAS[9],
            COSXL_EFFECTIVE_SIGMAS[-1],
        ],
    )
    def test_cosxl_edm_vprediction_denoised_is_byte_exact_to_comfyui(self, sigma: float) -> None:
        generator = torch.Generator("cpu").manual_seed(685468484323813)
        model_input = torch.randn((1, 4, 8, 8), generator=generator)
        model_output = torch.randn((1, 4, 8, 8), generator=generator)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        sigma_data = 1.0
        expected = (
            model_input * sigma_data**2 / (sigma_tensor**2 + sigma_data**2)
            - model_output * sigma_tensor * sigma_data / (sigma_tensor**2 + sigma_data**2) ** 0.5
        )
        actual = _calculate_denoised(
            Parameterization.V_PREDICTION, sigma, model_output, model_input
        )
        assert torch.equal(actual, expected)

    def test_cosxl_edm_vprediction_direct_cfg_and_timestep_are_byte_exact(self) -> None:
        sigma = COSXL_EFFECTIVE_SIGMAS[9]
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        x = torch.linspace(-2.0, 2.0, 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
        cond_output = torch.full_like(x, 0.25)
        uncond_output = torch.full_like(x, -0.5)
        cond = tiny_cond("p")
        uncond = tiny_cond("n")

        def reference(output: torch.Tensor) -> torch.Tensor:
            return (
                x / (sigma_tensor**2 + 1.0) - output * sigma_tensor / (sigma_tensor**2 + 1.0) ** 0.5
            )

        class FixedOutputUNet(torch.nn.Module):
            config = TINY_SD1

            def forward(
                self,
                input: torch.Tensor,
                timesteps: torch.Tensor,
                *,
                context: torch.Tensor,
                y: torch.Tensor | None = None,
                attention_guidance: object | None = None,
            ) -> torch.Tensor:
                expected_input = x / (sigma_tensor**2 + 1.0) ** 0.5
                copies = input.shape[0]
                assert torch.equal(input, expected_input.expand(copies, -1, -1, -1))
                expected_timestep = (0.25 * sigma_tensor.log()).expand(copies)
                assert torch.equal(timesteps, expected_timestep)
                assert y is None
                if copies == 1:
                    return cond_output
                assert copies == 2
                return torch.cat((uncond_output, cond_output))

        model = cast(UNetModel, FixedOutputUNet())
        direct = SDDenoiser(
            model,
            COSXL_EDM_SPACE,
            cond,
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        assert torch.equal(direct(x, sigma), reference(cond_output))

        evaluator = SDDenoiser(
            model,
            COSXL_EDM_SPACE,
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        uncond_denoised, cond_denoised = evaluator.evaluate_conditioning_batch(
            x,
            sigma,
            (
                evaluator.prepare_conditioning(uncond),
                evaluator.prepare_conditioning(cond),
            ),
        )
        expected_cfg = cfg_combine(reference(cond_output), reference(uncond_output), 8.0)
        assert torch.equal(cfg_combine(cond_denoised, uncond_denoised, 8.0), expected_cfg)
        assert torch.equal(uncond_denoised, reference(uncond_output))

    def test_executed_vprediction_input_is_byte_exact_to_comfyui(self) -> None:
        """The portable Python-float coefficient is intentionally retained, so
        this test also proves the executed override remains regression-sensitive."""
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert scheduler is not None
        sigmas = sampling_sigmas(
            scheduler,
            DiscreteSigmas.linear_beta(zsnr=True),
            20,
        )[:-1]
        expected_sigmas = (
            4518.763671875,
            38.195377349853516,
            17.054519653320312,
            10.207263946533203,
            6.909849643707275,
            5.015027046203613,
            3.8084592819213867,
            2.9853084087371826,
            2.3943428993225098,
            1.9524929523468018,
            1.6106818914413452,
            1.3382458686828613,
            1.115142822265625,
            0.9277320504188538,
            0.7663238048553467,
            0.6236164569854736,
            0.49348947405815125,
            0.3695031702518463,
            0.24073567986488342,
            0.029167160391807556,
        )
        if sys.platform == "darwin":
            # ComfyUI executed on Apple Silicon produces one-float32-ulp
            # different values at these two steps; Dinkster matches the
            # reference bit-exactly on each platform.
            overrides = {7: 2.9853081703186035, 17: 0.3695032000541687}
            expected_sigmas = tuple(
                overrides.get(index, value) for index, value in enumerate(expected_sigmas)
            )
        assert len(sigmas) == len(expected_sigmas)
        assert all(math.isfinite(sigma) and sigma > 0 for sigma in sigmas)
        assert all(left > right for left, right in zip(sigmas, sigmas[1:], strict=False))
        if reference_validation_enabled():
            assert sigmas == expected_sigmas
        assert {
            4518.763671875,
            1.3382458686828613,
            0.029167160391807556,
        }.issubset(sigmas)
        noise = torch.linspace(
            -4.0,
            4.0,
            4 * 128 * 128,
            dtype=torch.float32,
        ).reshape(1, 4, 128, 128)

        for sigma in sigmas:
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
            expected = noise / (sigma_tensor**2 + 1.0**2) ** 0.5
            actual = _calculate_input(Parameterization.V_PREDICTION, sigma, noise)
            portable = calculate_input(Parameterization.V_PREDICTION, sigma, noise)

            assert torch.equal(actual, expected)
            assert not torch.equal(portable, expected)

    @pytest.mark.parametrize(
        "sigma",
        [4518.763671875, 1.3382458686828613, 0.029167160391807556],
    )
    def test_executed_vprediction_is_byte_exact_to_comfyui(self, sigma: float) -> None:
        generator = torch.Generator("cpu").manual_seed(685468484323813)
        model_input = torch.randn((1, 4, 8, 8), generator=generator)
        model_output = torch.randn((1, 4, 8, 8), generator=generator)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        sigma_data = 1.0
        expected = (
            model_input * sigma_data**2 / (sigma_tensor**2 + sigma_data**2)
            - model_output * sigma_tensor * sigma_data / (sigma_tensor**2 + sigma_data**2) ** 0.5
        )

        actual = _calculate_denoised(
            Parameterization.V_PREDICTION, sigma, model_output, model_input
        )

        assert torch.equal(actual, expected)

    def test_executed_vprediction_remains_exact_through_cfg(self) -> None:
        generator = torch.Generator("cpu").manual_seed(685468484323813)
        model_input = torch.randn((1, 4, 8, 8), generator=generator)
        cond_output = torch.randn((1, 4, 8, 8), generator=generator)
        uncond_output = torch.randn((1, 4, 8, 8), generator=generator)
        sigma = 1.3382458686828613
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

        def reference(output: torch.Tensor) -> torch.Tensor:
            sigma_data = 1.0
            return (
                model_input * sigma_data**2 / (sigma_tensor**2 + sigma_data**2)
                - output * sigma_tensor * sigma_data / (sigma_tensor**2 + sigma_data**2) ** 0.5
            )

        cond = _calculate_denoised(Parameterization.V_PREDICTION, sigma, cond_output, model_input)
        uncond = _calculate_denoised(
            Parameterization.V_PREDICTION, sigma, uncond_output, model_input
        )

        assert torch.equal(
            cfg_combine(cond, uncond, 8.0),
            cfg_combine(reference(cond_output), reference(uncond_output), 8.0),
        )

    @pytest.mark.parametrize(
        ("parameterization", "space"),
        (
            (Parameterization.EPS, SPACE),
            (Parameterization.V_PREDICTION, SPACE),
            (Parameterization.V_PREDICTION, COSXL_EDM_SPACE),
        ),
    )
    def test_real_provider_guidance_phases_match_baseline_and_fuse(
        self, parameterization: Parameterization, space: DiscreteSigmas | ContinuousEDMSigmas
    ) -> None:
        model = tiny_unet()
        cond, uncond = tiny_cond("p"), tiny_cond("n")
        cfg, x = 4.0, tiny_latent()
        sigma = COSXL_EFFECTIVE_SIGMAS[9] if space is COSXL_EDM_SPACE else 0.6
        denoiser = SDDenoiser(
            model,
            space,
            parameterization=parameterization,
            compute_dtype=torch.float32,
        )
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
            lambda _module, _args, kwargs: seen.append(kwargs["context"].detach().clone()),
            with_kwargs=True,
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

    @pytest.mark.parametrize(
        ("parameterization", "space"),
        (
            (Parameterization.EPS, SPACE),
            (Parameterization.V_PREDICTION, SPACE),
            (Parameterization.V_PREDICTION, COSXL_EDM_SPACE),
        ),
    )
    def test_real_provider_guidance_missing_uncond_is_exact_zero(
        self, parameterization: Parameterization, space: DiscreteSigmas | ContinuousEDMSigmas
    ) -> None:
        cond = tiny_cond("p")
        denoiser = SDDenoiser(
            tiny_unet(), space, parameterization=parameterization, compute_dtype=torch.float32
        )
        conditioning = ConditioningEvaluation(
            lambda value, _role: denoiser.prepare_conditioning(value),
            denoiser.evaluate_conditioning,
        )
        x = tiny_latent()
        sigma = COSXL_EFFECTIVE_SIGMAS[9] if space is COSXL_EDM_SPACE else 0.5
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

    def test_is_the_eps_apply_model_composition(self) -> None:
        """cfg=1: precondition the input, look the sigma up in the
        discrete table, forward, lift to float32, eps -> denoised -
        BaseModel._apply_model @ 947c2749 piece by piece."""
        model = tiny_unet()
        cond = tiny_cond("p")
        denoiser = SDDenoiser(model, SPACE, cond, compute_dtype=torch.float32)
        x = tiny_latent()
        sigma = 0.7
        out = denoiser(x, sigma)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        xc = x / (sigma_tensor**2 + 1.0**2) ** 0.5
        timesteps = torch.full((1,), SPACE.timestep(sigma))
        raw = model(xc, timesteps, context=cond.embeddings)
        expected = calculate_denoised(Parameterization.EPS, sigma, raw.float(), x)
        assert torch.allclose(out, expected, atol=1e-6)
        assert out.dtype == torch.float32

    def test_eps_preconditioning_matches_device_float32_reference_kernel(self) -> None:
        expected_input = torch.tensor([-0.8828125], dtype=torch.float16).reshape(1, 1, 1, 1)

        class CaptureUNet(torch.nn.Module):
            config = TINY_SD1

            def forward(
                self,
                input: torch.Tensor,
                timesteps: torch.Tensor,
                *,
                context: torch.Tensor,
                y: torch.Tensor | None = None,
                attention_guidance: object | None = None,
            ) -> torch.Tensor:
                assert input.flatten()[10055 % input.numel()] == expected_input.flatten()[0]
                return torch.zeros_like(input)

        x = torch.full((1, 4, 64, 64), -12.92857837677002, dtype=torch.float32)
        denoiser = SDDenoiser(
            cast(UNetModel, CaptureUNet()),
            SPACE,
            tiny_cond("p"),
            compute_dtype=torch.float16,
        )
        denoiser(x, 14.614640235900879)

    def test_vprediction_apply_model_composition(self) -> None:
        """The checkpoint-selected v-prediction meaning reaches both
        preconditioning and raw-output conversion without changing the
        discrete timestep or SDXL conditioning path."""
        model = tiny_unet()
        cond = tiny_cond("p")
        denoiser = SDDenoiser(
            model,
            SPACE,
            cond,
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        x = tiny_latent()
        sigma = 0.7
        out = denoiser(x, sigma)
        xc = _calculate_input(Parameterization.V_PREDICTION, sigma, x)
        timesteps = torch.full((1,), SPACE.timestep(sigma))
        raw = model(xc, timesteps, context=cond.embeddings)
        expected = _calculate_denoised(Parameterization.V_PREDICTION, sigma, raw.float(), x)
        assert torch.equal(out, expected)
        eps = calculate_denoised(Parameterization.EPS, sigma, raw.float(), x)
        assert not torch.allclose(out, eps)

    def test_vprediction_component_matches_executed_comfyui_golden(self) -> None:
        golden = VPRED_GOLDENS["denoiser"]
        x = torch.tensor(golden["model_input"]).reshape(1, 4, 1, 1)
        model_output = torch.tensor(golden["model_output"]).reshape_as(x)
        expected_input = torch.tensor(golden["calculate_input"]).reshape_as(x)
        expected = torch.tensor(golden["calculate_denoised"]).reshape_as(x)
        sigma = golden["sigma"]

        class FixedOutputUNet(torch.nn.Module):
            config = TINY_SD1

            def forward(
                self,
                input: torch.Tensor,
                timesteps: torch.Tensor,
                *,
                context: torch.Tensor,
                y: torch.Tensor | None = None,
                attention_guidance: object | None = None,
            ) -> torch.Tensor:
                assert torch.equal(input, expected_input)
                assert timesteps.shape == (1,)
                assert context.shape[0] == 1
                assert y is None
                return model_output

        denoiser = SDDenoiser(
            cast(UNetModel, FixedOutputUNet()),
            SPACE,
            tiny_cond("p"),
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        tolerance = VPRED_GOLDENS["_meta"]["tolerance"]
        assert torch.allclose(denoiser(x, sigma), expected, **tolerance)

    def test_edm_vprediction_component_matches_executed_comfyui_golden(self) -> None:
        golden = EDM_VPRED_GOLDENS["denoiser"]
        bounds = EDM_VPRED_GOLDENS["bounds"]
        space = ContinuousEDMSigmas(min_sigma=bounds["sigma_min"], max_sigma=bounds["sigma_max"])
        x = torch.tensor(golden["model_input"]).reshape(1, 4, 1, 1)
        model_output = torch.tensor(golden["model_output"]).reshape_as(x)
        expected_input = torch.tensor(golden["calculate_input"]).reshape_as(x)
        expected = torch.tensor(golden["calculate_denoised"]).reshape_as(x)
        sigma = golden["sigma"]

        class FixedOutputUNet(torch.nn.Module):
            config = TINY_SD1

            def forward(
                self,
                input: torch.Tensor,
                timesteps: torch.Tensor,
                *,
                context: torch.Tensor,
                y: torch.Tensor | None = None,
                attention_guidance: object | None = None,
            ) -> torch.Tensor:
                assert torch.equal(input, expected_input)
                assert torch.equal(
                    timesteps, torch.tensor([golden["timestep"]], dtype=torch.float32)
                )
                assert context.shape[0] == 1
                assert y is None
                return model_output

        denoiser = SDDenoiser(
            cast(UNetModel, FixedOutputUNet()),
            space,
            tiny_cond("p"),
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        tolerance = EDM_VPRED_GOLDENS["_meta"]["tolerance"]
        assert torch.allclose(denoiser(x, sigma), expected, **tolerance)

    def test_timestep_is_the_nearest_table_index(self) -> None:
        """The bridge must use ModelSamplingDiscrete.timestep, not
        sigma-as-timestep: an exact table entry maps to its index."""
        sigma = SPACE.entries[500]
        assert SPACE.timestep(sigma) == 500.0

    def test_adm_model_forwards_y(self) -> None:
        model = tiny_unet(TINY_ADM)
        cond = tiny_cond("p", features=TINY_ADM.context_dim)
        adm = tiny_adm("p")
        denoiser = SDDenoiser(model, SPACE, cond, adm_cond=adm, compute_dtype=torch.float32)
        x = tiny_latent()
        sigma = 0.5
        out = denoiser(x, sigma)
        xc = _calculate_input(Parameterization.EPS, sigma, x)
        timesteps = torch.full((1,), SPACE.timestep(sigma))
        raw = model(xc, timesteps, context=cond.embeddings, y=adm)
        expected = _calculate_denoised(Parameterization.EPS, sigma, raw.float(), x)
        assert torch.equal(out, expected)

    def test_compatible_conditions_run_as_one_batch(self) -> None:
        model = tiny_unet(TINY_ADM)
        cond = tiny_cond("p", features=TINY_ADM.context_dim)
        uncond = tiny_cond("n", features=TINY_ADM.context_dim)
        adm_cond, adm_uncond = tiny_adm("p"), tiny_adm("n")
        evaluator = SDDenoiser(model, SPACE, compute_dtype=torch.float32)
        x = tiny_latent()
        sigma = 0.6
        conditions = (
            evaluator.prepare_conditioning(uncond, adm=adm_uncond),
            evaluator.prepare_conditioning(cond, adm=adm_cond),
        )
        assert evaluator.batchable(conditions)
        actual_uncond, actual_cond = evaluator.evaluate_conditioning_batch(x, sigma, conditions)
        assert torch.allclose(
            actual_cond,
            evaluator.evaluate_conditioning(x, sigma, conditions[1]),
            atol=1e-4,
        )
        assert torch.allclose(
            actual_uncond,
            evaluator.evaluate_conditioning(x, sigma, conditions[0]),
            atol=1e-4,
        )

    @pytest.mark.parametrize(
        "parameterization", [Parameterization.EPS, Parameterization.V_PREDICTION]
    )
    def test_unequal_token_counts_batch_via_lcm_repeat(
        self, parameterization: Parameterization
    ) -> None:
        """CONDCrossAttn.concat @ 947c2749: tokens 2 vs 3 repeat to 6
        (factors 3 and 2, within the limit) and batch as ONE forward;
        repeat-padding is exact for cross-attention, so the result
        matches the separate forwards."""
        assert cross_attn_repeat([2, 3]) == [3, 2]
        model = tiny_unet()
        cond = tiny_cond("p", tokens=2)
        uncond = tiny_cond("n", tokens=3)
        evaluator = SDDenoiser(
            model,
            SPACE,
            parameterization=parameterization,
            compute_dtype=torch.float32,
        )
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
            atol=1e-4,
        )
        assert torch.allclose(
            actual_uncond,
            evaluator.evaluate_conditioning(x, sigma, conditions[0]),
            atol=1e-4,
        )

    def test_repeat_beyond_the_limit_is_not_batchable(self) -> None:
        assert cross_attn_repeat([1, 5]) is None
        cond = tiny_cond("p", tokens=1)
        uncond = tiny_cond("n", tokens=5)
        evaluator = SDDenoiser(tiny_unet(), SPACE, compute_dtype=torch.float32)
        conditions = (
            evaluator.prepare_conditioning(cond),
            evaluator.prepare_conditioning(uncond),
        )
        assert not evaluator.batchable(conditions)
        with pytest.raises(DenoiseError, match="incompatible"):
            evaluator.evaluate_conditioning_batch(tiny_latent(), 0.6, conditions)

    def test_singleton_conditioning_broadcasts(self) -> None:
        model = tiny_unet()
        denoiser = SDDenoiser(model, SPACE, tiny_cond("p", batch=1), compute_dtype=torch.float32)
        out = denoiser(tiny_latent(batch=2), 0.5)
        assert out.shape[0] == 2

    def test_undersized_conditioning_repeats_cyclically(self) -> None:
        """repeat_to_batch_size @ 947c2749: batch 2 -> 3 rides rows
        [0, 1, 0], never an error (the reference resizes ANY source
        batch to the latent batch)."""
        model = tiny_unet()
        cond2 = tiny_cond("p", batch=2)
        resized = Conditioning(embeddings=cond2.embeddings[torch.tensor([0, 1, 0])])
        x = tiny_latent(batch=3)
        out = SDDenoiser(model, SPACE, cond2, compute_dtype=torch.float32)(x, 0.5)
        expected = SDDenoiser(model, SPACE, resized, compute_dtype=torch.float32)(x, 0.5)
        assert torch.equal(out, expected)

    def test_oversized_conditioning_truncates(self) -> None:
        """repeat_to_batch_size @ 947c2749: batch 3 -> 2 keeps the
        first two rows."""
        model = tiny_unet()
        cond3 = tiny_cond("p", batch=3)
        truncated = Conditioning(embeddings=cond3.embeddings[:2])
        x = tiny_latent(batch=2)
        out = SDDenoiser(model, SPACE, cond3, compute_dtype=torch.float32)(x, 0.5)
        expected = SDDenoiser(model, SPACE, truncated, compute_dtype=torch.float32)(x, 0.5)
        assert torch.equal(out, expected)

    def test_fused_batch_runs_uncond_first(self) -> None:
        """calc_cond_batch @ 947c2749 pops a REVERSED candidate list,
        so the physical fused batch is [uncond, cond] for context AND
        y; batch position is bit-visible under float accumulation, so
        the order itself is the parity surface."""
        model = tiny_unet(TINY_ADM)
        cond = tiny_cond("p", features=TINY_ADM.context_dim)
        uncond = tiny_cond("n", features=TINY_ADM.context_dim)
        adm_cond, adm_uncond = tiny_adm("p"), tiny_adm("n")
        evaluator = SDDenoiser(model, SPACE, compute_dtype=torch.float32)
        seen: list[tuple[torch.Tensor, torch.Tensor]] = []
        handle = model.register_forward_pre_hook(
            lambda _module, args, kwargs: seen.append((kwargs["context"], kwargs["y"])),
            with_kwargs=True,
        )
        try:
            evaluator.evaluate_conditioning_batch(
                tiny_latent(),
                0.6,
                (
                    evaluator.prepare_conditioning(uncond, adm=adm_uncond),
                    evaluator.prepare_conditioning(cond, adm=adm_cond),
                ),
            )
        finally:
            handle.remove()
        assert len(seen) == 1
        context, y = seen[0]
        ctx_first, ctx_second = context.chunk(2)
        assert torch.equal(ctx_first, uncond.embeddings)
        assert torch.equal(ctx_second, cond.embeddings)
        y_first, y_second = y.chunk(2)
        assert torch.equal(y_first, adm_uncond)
        assert torch.equal(y_second, adm_cond)

    def test_replacement_conditioning_uses_its_prepared_adm(self) -> None:
        model = tiny_unet(TINY_ADM)
        replacement = tiny_cond("replacement", features=TINY_ADM.context_dim)
        replacement_adm = tiny_adm("replacement")
        evaluator = SDDenoiser(model, SPACE, compute_dtype=torch.float32)
        seen: list[torch.Tensor] = []
        handle = model.register_forward_pre_hook(
            lambda _module, args, kwargs: seen.append(kwargs["y"]),
            with_kwargs=True,
        )
        try:
            evaluator.evaluate_conditioning(
                tiny_latent(),
                0.6,
                evaluator.prepare_conditioning(replacement, adm=replacement_adm),
            )
        finally:
            handle.remove()
        assert len(seen) == 1
        assert torch.equal(seen[0], replacement_adm)

    def test_fused_lcm_repeat_stays_aligned_uncond_first(self) -> None:
        """With unequal token counts the lcm repeat factors must ride
        the SAME uncond-first order as the contexts: tokens 3 (uncond)
        and 2 (cond) repeat by [2, 3] to 6 tokens each."""
        model = tiny_unet()
        cond = tiny_cond("p", tokens=2)
        uncond = tiny_cond("n", tokens=3)
        evaluator = SDDenoiser(model, SPACE, compute_dtype=torch.float32)
        seen: list[torch.Tensor] = []
        handle = model.register_forward_pre_hook(
            lambda _module, args, kwargs: seen.append(kwargs["context"]),
            with_kwargs=True,
        )
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
        assert torch.equal(first, uncond.embeddings.repeat(1, 2, 1))
        assert torch.equal(second, cond.embeddings.repeat(1, 3, 1))

    def test_autograd_flows(self) -> None:
        denoiser = SDDenoiser(tiny_unet(), SPACE, tiny_cond("p"), compute_dtype=torch.float32)
        x = tiny_latent().requires_grad_(True)
        out = denoiser(x, 0.5)
        assert out.grad_fn is not None
        out.sum().backward()
        assert x.grad is not None


# --- ADM validation ---------------------------------------------------------


class TestAdmValidation:
    def test_sd1_refuses_an_adm(self) -> None:
        with pytest.raises(DenoiseError, match="not"):
            SDDenoiser(tiny_unet(), SPACE, tiny_cond("p"), adm_cond=torch.zeros(1, 12))

    def test_adm_model_requires_adm_cond(self) -> None:
        with pytest.raises(DenoiseError, match="ADM conditioning"):
            SDDenoiser(
                tiny_unet(TINY_ADM),
                SPACE,
                tiny_cond("p", features=TINY_ADM.context_dim),
            )

    def test_every_adm_condition_requires_its_vector(self) -> None:
        evaluator = SDDenoiser(tiny_unet(TINY_ADM), SPACE)
        with pytest.raises(DenoiseError, match="ADM conditioning"):
            evaluator.prepare_conditioning(tiny_cond("n", features=TINY_ADM.context_dim))

    def test_no_uncond_needs_no_adm_uncond(self) -> None:
        SDDenoiser(
            tiny_unet(TINY_ADM),
            SPACE,
            tiny_cond("p", features=TINY_ADM.context_dim),
            adm_cond=tiny_adm("p"),
        )

    def test_wrong_adm_width_refuses(self) -> None:
        with pytest.raises(DenoiseError, match="ADM conditioning"):
            SDDenoiser(
                tiny_unet(TINY_ADM),
                SPACE,
                tiny_cond("p", features=TINY_ADM.context_dim),
                adm_cond=torch.zeros(1, 5),
            )

    def test_malformed_conditioning_refuses(self) -> None:
        with pytest.raises(DenoiseError, match="batch x tokens"):
            SDDenoiser(
                tiny_unet(),
                SPACE,
                Conditioning(embeddings=torch.zeros(3, TINY_SD1.context_dim)),
            )
