"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .families.ltx import (
    _ltxav_guidance_runtime,
)
from .native_arm_core import (
    _DISABLE_CFG1_OPTIMIZATION,
    ALIGN_YOUR_STEPS_NOISE_LEVELS,
    GITS_NOISE_LEVELS,
    OPTIMAL_STEPS_NOISE_LEVELS,
    Any,
    KSampler,
    Mapping,
    Node,
    NodeSchema,
    _CustomGuiderValue,
    _CustomNoiseValue,
    _CustomSamplerValue,
    _CustomSigmasValue,
    _DualCFGGuiderValue,
    _DualModelGuiderValue,
    _inference_registries,
    _LTXAVDualGuiderValue,
    _PerpNegGuiderValue,
    _sampler_registry,
    _torch,
    align_your_steps_sigmas,
    cast,
    current_execution_context,
    gits_sigmas,
    importlib,
    math,
    optimal_steps_sigmas,
    re,
)
from .native_arm_latent_utils import _check_bounds
from .native_arm_runtime import (
    _application_chain_model,
    _bind_model_sampling_options,
    _native_model,
    _native_model_h3_control,
    _native_model_sampling_cache,
    _native_model_sampling_space,
    _native_model_sampling_timeline,
    _NativeModelOverlay,
)
from .native_arm_scheduling import (
    _catalog_id,
)
from .nodes_guidance import (
    _guidance_transform_factory,
)
from .nodes_provider import (
    _bind_sampling_shift,
    _generation_provider_schema,
    _require_base_custom_sampling_runtime,
    _require_custom_sampling_runtime,
)
from .nodes_sampling_runtime import (
    NativeKSampler,
    NativeKSamplerAdvanced,
    _batch_index_noise_inds,
)

if TYPE_CHECKING:
    from dinkster_inference import SamplingSegment


class GenerationKSampler(NativeKSampler):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler")

    @classmethod
    @_bind_model_sampling_options
    def execute(
        cls,
        *,
        model: object,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        denoise: float,
        segment: SamplingSegment | None = None,
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        return NativeKSampler.execute(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            segment=segment,
            conditioning_batching=_conditioning_batching_value(
                conditioning_batching, max_fused_lanes
            ),
        )


class GenerationKSamplerAdvanced(NativeKSamplerAdvanced):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler_advanced")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("noise_seed", noise_seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("start_at_step", start_at_step, 0, cls.MAX_STEPS),
            ("end_at_step", end_at_step, 0, cls.MAX_STEPS),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if add_noise not in ("enable", "disable"):
            raise ValueError("add_noise must be 'enable' or 'disable'")
        if return_with_leftover_noise not in ("disable", "enable"):
            raise ValueError("return_with_leftover_noise must be 'disable' or 'enable'")
        effective_end = min(end_at_step, steps)
        if start_at_step >= effective_end:
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            output = dict(cast("Mapping[object, object]", latent_image))
            output.pop("downscale_ratio_spacial", None)
            output.pop("downscale_ratio_temporal", None)
            return cls.outputs(latent=output)
        inference = importlib.import_module("dinkster_inference")
        segment = inference.SamplingSegment(
            steps=steps,
            start_step=start_at_step,
            end_step=effective_end,
            add_noise=add_noise == "enable",
            return_with_leftover_noise=return_with_leftover_noise == "enable",
        )
        result = GenerationKSampler.execute(
            model=model,
            seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            segment=segment,
            conditioning_batching=conditioning_batching,
            max_fused_lanes=max_fused_lanes,
        )
        return cls.outputs(latent=result["latent"])


def _custom_sampler_value(
    sampler_name: str,
    options: Mapping[str, object] | None = None,
) -> _CustomSamplerValue:
    inference = importlib.import_module("dinkster_inference")
    context = current_execution_context()
    snapshot_digest = None if context is None else context.extension_snapshot_digest
    if context is None:
        registry, extension_ids = inference.builtin_registries().samplers, ()
    else:
        registry, extension_ids, _behavior_hash = _sampler_registry(inference, snapshot_digest)
    sampler_id = _catalog_id(registry, sampler_name, "sampler")
    descriptor = registry.get(sampler_id)
    if descriptor is None:
        raise ValueError(f"sampler {sampler_id!r} disappeared from its registry snapshot")
    resolved = inference.resolve_options(descriptor.options, options or {})
    return _CustomSamplerValue(
        descriptor,
        tuple((spec.name, resolved[spec.name]) for spec in descriptor.options),
        extension_snapshot_digest=snapshot_digest,
        extension_ids=extension_ids,
    )


def _sde_sampler_id(noise_device: str, cpu_id: str, gpu_id: str) -> str:
    if noise_device == "cpu":
        return cpu_id
    if noise_device == "gpu":
        return gpu_id
    raise ValueError("noise_device must be 'gpu' or 'cpu'")


class GenerationKSamplerSelect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler_select")

    @classmethod
    def execute(cls, *, sampler_name: str) -> Mapping[str, object]:
        return cls.outputs(sampler=_custom_sampler_value(sampler_name))


class GenerationSamplerDPMPP3MSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_3m_sde")

    @classmethod
    def execute(
        cls,
        *,
        eta: float,
        s_noise: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_3m_sde",
            "dinkster.dpmpp_3m_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(sampler_id, {"eta": eta, "s_noise": s_noise})
        )


class GenerationSamplerDPMPP2MSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_2m_sde")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        eta: float,
        s_noise: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_2m_sde",
            "dinkster.dpmpp_2m_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                sampler_id,
                {"solver_type": solver_type, "eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerDPMPPSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_sde")

    @classmethod
    def execute(
        cls,
        *,
        eta: float,
        s_noise: float,
        r: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_sde",
            "dinkster.dpmpp_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                sampler_id,
                {"eta": eta, "s_noise": s_noise, "r": r},
            )
        )


class GenerationSamplerDPMPP2SAncestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_2s_ancestral")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.dpmpp_2s_ancestral",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerEulerAncestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_euler_ancestral")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.euler_ancestral",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerEulerAncestralCFGPP(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_euler_ancestral_cfg_pp")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.euler_ancestral_cfg_pp",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerLMS(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_lms")

    @classmethod
    def execute(cls, *, order: int) -> Mapping[str, object]:
        return cls.outputs(sampler=_custom_sampler_value("dinkster.lms", {"order": order}))


class GenerationSamplerDPMAdaptative(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpm_adaptative")

    @classmethod
    def execute(
        cls,
        *,
        order: int,
        rtol: float,
        atol: float,
        h_init: float,
        pcoeff: float,
        icoeff: float,
        dcoeff: float,
        accept_safety: float,
        eta: float,
        s_noise: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.dpm_adaptive",
                {
                    "order": order,
                    "rtol": rtol,
                    "atol": atol,
                    "h_init": h_init,
                    "pcoeff": pcoeff,
                    "icoeff": icoeff,
                    "dcoeff": dcoeff,
                    "accept_safety": accept_safety,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            )
        )


class GenerationSamplerERSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_er_sde")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        max_stage: int,
        eta: float,
        s_noise: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.er_sde",
                {
                    "solver_type": solver_type,
                    "max_stage": max_stage,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            )
        )


class GenerationSamplerSEEDS2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_seeds_2")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        eta: float,
        s_noise: float,
        r: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.seeds_2",
                {"solver_type": solver_type, "eta": eta, "s_noise": s_noise, "r": r},
            )
        )


class GenerationSamplerSASolver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_sa_solver")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        eta: float,
        sde_start_percent: float,
        sde_end_percent: float,
        s_noise: float,
        predictor_order: int,
        corrector_order: int,
        use_pece: bool,
        simple_order_2: bool,
    ) -> Mapping[str, object]:
        runtime, sampling_shift, _device = _require_base_custom_sampling_runtime(
            model, "SamplerSASolver"
        )
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma,
            sampling_shift,
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.configured_sa_solver",
                {
                    "eta": eta,
                    "sde_start_sigma": percent_to_sigma(
                        sde_start_percent, return_actual_sigma=False
                    ),
                    "sde_end_sigma": percent_to_sigma(sde_end_percent, return_actual_sigma=False),
                    "s_noise": s_noise,
                    "predictor_order": predictor_order,
                    "corrector_order": corrector_order,
                    "use_pece": use_pece,
                    "simple_order_2": simple_order_2,
                },
            )
        )


class GenerationBasicScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.basic_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        scheduler: str,
        steps: int,
        denoise: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(model, "BasicScheduler")
        inference = importlib.import_module("dinkster_inference")
        scheduler_id = _catalog_id(
            _inference_registries(inference).schedulers, scheduler, "scheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sigmas, sampling_shift)
        return cls.outputs(
            sigmas=_CustomSigmasValue(
                build_sigmas(scheduler_id, steps, denoise, device=device),
                source_scheduler_id=scheduler_id,
            )
        )


class GenerationBetaSamplingScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.beta_sampling_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(
            model, "BetaSamplingScheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_beta_sigmas, sampling_shift)
        return cls.outputs(
            sigmas=_CustomSigmasValue(build_sigmas(steps, alpha, beta, device=device))
        )


class GenerationSDTurboScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sd_turbo_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        steps: int,
        denoise: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= 10:
            raise ValueError(f"steps must be in [1, 10], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(
            model, "SDTurboScheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sd_turbo_sigmas, sampling_shift)
        return cls.outputs(sigmas=_CustomSigmasValue(build_sigmas(steps, denoise, device=device)))


def _custom_sigmas_tensor(value: object) -> tuple[Any, Any]:
    if type(value) is not _CustomSigmasValue:
        raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
    torch = _torch()
    return torch, torch.FloatTensor(value.values)


def _custom_sigmas_value(tensor: Any) -> _CustomSigmasValue:
    return _CustomSigmasValue(tuple(float(value) for value in tensor.tolist()))


class GenerationKarrasScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.karras_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationExponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.exponential_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps).exp()
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationPolyexponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.polyexponential_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        ramp = torch.linspace(1, 0, steps) ** rho
        sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationLaplaceScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.laplace_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        mu: float,
        beta: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        values = torch.linspace(0, 1, steps)
        transformed = mu - beta * torch.sign(0.5 - values) * torch.log(
            1 - 2 * torch.abs(0.5 - values) + 1e-5
        )
        sigmas = torch.clamp(torch.exp(transformed), min=sigma_min, max=sigma_max)
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationVPScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vp_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        beta_d: float,
        beta_min: float,
        eps_s: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        values = torch.linspace(1, eps_s, steps)
        sigmas = torch.sqrt(torch.special.expm1(beta_d * values**2 / 2 + beta_min * values))
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationAlignYourStepsScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.align_your_steps_scheduler")

    @classmethod
    def execute(cls, *, model_type: str, steps: int, denoise: float) -> Mapping[str, object]:
        if model_type not in ALIGN_YOUR_STEPS_NOISE_LEVELS:
            valid = ", ".join(sorted(ALIGN_YOUR_STEPS_NOISE_LEVELS))
            raise ValueError(f"model_type must be one of {valid}, got {model_type!r}")
        if not 1 <= steps <= 10_000:
            raise ValueError(f"steps must be in [1, 10000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(
            sigmas=_CustomSigmasValue(align_your_steps_sigmas(model_type, steps, denoise))
        )


class GenerationGITSScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.gits_scheduler")

    @classmethod
    def execute(cls, *, coeff: float, steps: int, denoise: float) -> Mapping[str, object]:
        if not 0.80 <= coeff <= 1.50 or round(coeff, 2) not in GITS_NOISE_LEVELS:
            valid = ", ".join(f"{key:.2f}" for key in sorted(GITS_NOISE_LEVELS))
            raise ValueError(f"coeff must round to one of {valid}, got {coeff}")
        if not 2 <= steps <= 1000:
            raise ValueError(f"steps must be in [2, 1000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(sigmas=_CustomSigmasValue(gits_sigmas(coeff, steps, denoise)))


class GenerationOptimalStepsScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.optimal_steps_scheduler")

    @classmethod
    def execute(cls, *, model_type: str, steps: int, denoise: float) -> Mapping[str, object]:
        if model_type not in OPTIMAL_STEPS_NOISE_LEVELS:
            valid = ", ".join(sorted(OPTIMAL_STEPS_NOISE_LEVELS))
            raise ValueError(f"model_type must be one of {valid}, got {model_type!r}")
        if not 3 <= steps <= 1000:
            raise ValueError(f"steps must be in [3, 1000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(
            sigmas=_CustomSigmasValue(optimal_steps_sigmas(model_type, steps, denoise))
        )


class GenerationFlux2Scheduler(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384  # ComfyUI's MAX_RESOLUTION
    MAX_STEPS = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux2_scheduler")

    @classmethod
    def execute(cls, *, steps: int, width: int, height: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("steps", steps, 1, cls.MAX_STEPS),
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        mu = inference.flux2_empirical_mu(round(width * height / 256), steps)
        # The reference's generalized time SNR shift at sigma exponent 1.0;
        # the final timestep 0 yields an exact trailing 0.0 sigma.
        timesteps = torch.linspace(1, 0, steps + 1)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationIdeogram4Scheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ideogram4_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        width: int,
        height: int,
        mu: float,
        std: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("steps", steps, 1, 200),
            ("width", width, 256, 8192),
            ("height", height, 256, 8192),
            ("mu", mu, -10.0, 10.0),
            ("std", std, 0.1, 5.0),
        )
        sigmas = importlib.import_module("dinkster_inference_torch").ideogram4_sigmas(
            steps, width, height, mu, std
        )
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationManualSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.manual_sigmas")

    @classmethod
    def execute(cls, *, sigmas: str) -> Mapping[str, object]:
        values = re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)
        tensor = _torch().FloatTensor([float(value) for value in values])
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationSplitSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.split_sigmas")

    @classmethod
    def execute(cls, *, sigmas: object, step: int) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        return cls.outputs(
            high_sigmas=_custom_sigmas_value(tensor[: step + 1]),
            low_sigmas=_custom_sigmas_value(tensor[step:]),
        )


class GenerationSplitSigmasDenoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.split_sigmas_denoise")

    @classmethod
    def execute(cls, *, sigmas: object, denoise: float) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        steps = max(tensor.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        return cls.outputs(
            high_sigmas=_custom_sigmas_value(tensor[:-total_steps]),
            low_sigmas=_custom_sigmas_value(tensor[-(total_steps + 1) :]),
        )


class GenerationFlipSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flip_sigmas")

    @classmethod
    def execute(cls, *, sigmas: object) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        if len(tensor) == 0:
            return cls.outputs(sigmas=_custom_sigmas_value(tensor))
        tensor = tensor.flip(0)
        if tensor[0] == 0:
            tensor[0] = 0.0001
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationSetFirstSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.set_first_sigma")

    @classmethod
    def execute(cls, *, sigmas: object, sigma: float) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        tensor[0] = sigma
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationExtendIntermediateSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.extend_intermediate_sigmas")

    @classmethod
    def execute(
        cls,
        *,
        sigmas: object,
        steps: int,
        start_at_sigma: float,
        end_at_sigma: float,
        spacing: str,
    ) -> Mapping[str, object]:
        torch, tensor = _custom_sigmas_tensor(sigmas)
        if start_at_sigma < 0:
            start_at_sigma = float("inf")
        if spacing not in ("linear", "cosine", "sine"):
            raise KeyError(spacing)
        values = torch.linspace(0, 1, steps + 1, device=tensor.device)[1:-1]
        if spacing == "cosine":
            computed_spacing = torch.sin(values * math.pi / 2)
        elif spacing == "sine":
            computed_spacing = 1 - torch.cos(values * math.pi / 2)
        else:
            computed_spacing = values
        extended_sigmas: list[Any] = []
        for index in range(len(tensor) - 1):
            sigma_current = tensor[index]
            sigma_next = tensor[index + 1]
            extended_sigmas.append(sigma_current)
            if end_at_sigma <= sigma_current <= start_at_sigma:
                interpolated = computed_spacing * (sigma_next - sigma_current) + sigma_current
                extended_sigmas.extend(interpolated.tolist())
        if len(tensor) > 0:
            extended_sigmas.append(tensor[-1])
        return cls.outputs(sigmas=_custom_sigmas_value(torch.FloatTensor(extended_sigmas)))


class GenerationSamplingPercentToSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampling_percent_to_sigma")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        sampling_percent: float,
        return_actual_sigma: bool,
    ) -> Mapping[str, object]:
        if not 0.0 <= sampling_percent <= 1.0:
            raise ValueError(f"sampling_percent must be in [0.0, 1.0], got {sampling_percent}")
        runtime, sampling_shift, _device = _require_custom_sampling_runtime(
            model, "SamplingPercentToSigma"
        )
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma,
            sampling_shift,
        )
        return cls.outputs(
            sigma_value=percent_to_sigma(
                sampling_percent,
                return_actual_sigma=return_actual_sigma,
            )
        )


def _conditioning_batching_value(mode: object, max_fused_lanes: int) -> object:
    if type(mode) is not str:
        raise TypeError("conditioning_batching must be a string")
    if type(max_fused_lanes) is not int or max_fused_lanes < 1:
        raise ValueError("max_fused_lanes must be a positive integer")
    inference = importlib.import_module("dinkster_inference")
    try:
        selected = inference.ConditioningBatchingMode(mode)
    except ValueError:
        raise ValueError(f"unknown conditioning batching mode {mode!r}") from None
    return inference.ConditioningBatching(
        selected,
        max_fused_lanes=(
            max_fused_lanes
            if selected is inference.ConditioningBatchingMode.MAX_FUSED_LANES
            else None
        ),
    )


class GenerationBasicGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.basic_guider")

    @classmethod
    def execute(cls, *, model: object, conditioning: object) -> Mapping[str, object]:
        return cls.outputs(guider=_CustomGuiderValue(model, conditioning, None, 1.0))


class GenerationCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        cfg: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        return cls.outputs(
            guider=_CustomGuiderValue(
                model,
                positive,
                negative,
                cfg,
                batching=_conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.dual_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        cond1: object,
        cond2: object,
        negative: object,
        cfg_conds: float,
        cfg_cond2_negative: float,
        style: str,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("cfg_conds", cfg_conds, 0.0, KSampler.MAX_CFG),
            ("cfg_cond2_negative", cfg_cond2_negative, 0.0, KSampler.MAX_CFG),
        )
        if style not in ("regular", "nested"):
            raise ValueError("style must be 'regular' or 'nested'")
        return cls.outputs(
            guider=_DualCFGGuiderValue(
                model,
                cond1,
                cond2,
                negative,
                cfg_conds,
                cfg_cond2_negative,
                style == "nested",
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDualModelGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.dual_model_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        cfg: float,
        model_negative: object | None = None,
        negative: object | None = None,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        return cls.outputs(
            guider=_DualModelGuiderValue(
                model,
                model_negative,
                positive,
                negative,
                cfg,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationScheduledCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.scheduled_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        sigmas: object,
        from_cfg: float,
        to_cfg: float,
        schedule: str,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(("from_cfg", from_cfg, 0.0, KSampler.MAX_CFG))
        _check_bounds(("to_cfg", to_cfg, 0.0, KSampler.MAX_CFG))
        if schedule not in ("linear", "log", "exp", "cos"):
            raise ValueError(f"unknown scheduled CFG interpolation {schedule!r}")
        if type(sigmas) is not _CustomSigmasValue:
            raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
        contribution = _guidance_transform_factory(
            "scheduled_cfg", sigmas.values, from_cfg, to_cfg, schedule
        )
        admission_cfg = from_cfg if not math.isclose(from_cfg, 1.0) else to_cfg
        return cls.outputs(
            guider=_CustomGuiderValue(
                model,
                positive,
                negative,
                admission_cfg,
                (("dinkster.scheduled_cfg_guider", contribution),),
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            ),
            sigmas=sigmas,
        )


class GenerationLTXVDualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_dual_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        video_cfg: float,
        audio_cfg: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("video_cfg", video_cfg, 0.0, KSampler.MAX_CFG),
            ("audio_cfg", audio_cfg, 0.0, KSampler.MAX_CFG),
        )
        _ltxav_guidance_runtime(model)
        return cls.outputs(
            guider=_LTXAVDualGuiderValue(
                model,
                positive,
                negative,
                video_cfg,
                audio_cfg,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationPerpNegGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.perp_neg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        empty_conditioning: object,
        cfg: float,
        neg_scale: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        if not 0.0 <= neg_scale <= 100.0:
            raise ValueError(f"neg_scale must be in [0.0, 100.0], got {neg_scale}")
        return cls.outputs(
            guider=_PerpNegGuiderValue(
                model,
                positive,
                negative,
                empty_conditioning,
                cfg,
                neg_scale,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDisableCFG1Optimization(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.disable_cfg1_optimization")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        original_model = model
        model, applications = _application_chain_model(model, "model")
        handle, overlays, resolvers, control, shift, transforms, windows, chroma_options = (
            _native_model(model, "model")
        )
        if any(contribution is _DISABLE_CFG1_OPTIMIZATION for _, contribution in transforms):
            return cls.outputs(model=original_model)
        disabled = _NativeModelOverlay(
            handle,
            overlays,
            resolvers,
            control,
            shift,
            (*transforms, ("dinkster.disable_cfg1_optimization", _DISABLE_CFG1_OPTIMIZATION)),
            windows,
            chroma_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
            minimax_h3_control=_native_model_h3_control(model),
        )
        if applications:
            inference = importlib.import_module("dinkster_inference")
            disabled = inference.ApplicationChain(disabled, applications)
        return cls.outputs(model=disabled)


class GenerationDisableNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.disable_noise")

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(noise=_CustomNoiseValue(None))


class GenerationRandomNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.random_noise")

    @classmethod
    def execute(cls, *, noise_seed: int) -> Mapping[str, object]:
        if not 0 <= noise_seed <= KSampler.MAX_SEED:
            raise ValueError(f"noise_seed must be in [0, {KSampler.MAX_SEED}], got {noise_seed}")
        return cls.outputs(noise=_CustomNoiseValue(noise_seed))


class GenerationAddNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.add_noise")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        noise: object,
        sigmas: object,
        latent_image: object,
    ) -> Mapping[str, object]:
        if type(noise) is not _CustomNoiseValue:
            raise TypeError("noise must come from RandomNoise or DisableNoise")
        if type(sigmas) is not _CustomSigmasValue:
            raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
        typed_noise = noise
        typed_sigmas = sigmas
        if not isinstance(latent_image, Mapping):
            raise TypeError("latent_image must be a mapping containing 'samples'")
        sigma_values = tuple(typed_sigmas.values)
        if not sigma_values:
            return cls.outputs(latent=cast("Mapping[object, object]", latent_image))
        latent = cast("Mapping[object, object]", latent_image)
        torch = _torch()
        samples = latent.get("samples")
        if type(samples) is not torch.Tensor:
            raise TypeError("AddNoise requires latent_image['samples'] to be an exact torch.Tensor")
        samples = cast("Any", samples)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        seed = 0 if typed_noise.seed is None else typed_noise.seed
        generated = (
            torch.zeros_like(samples, device="cpu")
            if typed_noise.seed is None
            else inference_torch.prepare_noise(samples, seed, _batch_index_noise_inds(latent))
        )
        runtime, sampling_shift, _device = _require_base_custom_sampling_runtime(model, "AddNoise")
        custom_sampling_add_noise = getattr(runtime, "custom_sampling_add_noise", None)
        if not callable(custom_sampling_add_noise):
            raise TypeError("model runtime does not support AddNoise")
        add_noise = _bind_sampling_shift(custom_sampling_add_noise, sampling_shift)
        first_sigma = sigma_values[0]
        scale = abs(first_sigma - sigma_values[-1]) if len(sigma_values) > 1 else first_sigma
        output = dict(latent)
        output["samples"] = add_noise(samples, generated, scale)
        return cls.outputs(latent=output)
