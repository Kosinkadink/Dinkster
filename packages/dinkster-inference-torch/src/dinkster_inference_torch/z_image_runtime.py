"""Native text encoding, flow denoising, and codec wiring for Z-Image."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Any

import torch
from dinkster_inference import (
    CodecDescriptor,
    CompositeWindowPlan,
    Conditioning,
    ContextWindowsSpec,
    ContributionGain,
    CustomSamplingRequest,
    CustomSamplingResult,
    DirectGainTableCurve,
    FlowSigmas,
    GuidanceRole,
    InpaintConditioning,
    ModelFamily,
    Parameterization,
    RealizedGainRow,
    Registry,
    SamplerDescriptor,
    SamplingStateCallback,
    SchedulerDescriptor,
    StepCallback,
    calculate_denoised,
    contribution_gain_slot_facts,
    executed_sampling_timeline,
    realize_gain_table,
    realize_sampling_timeline,
    require_realized_sampling_step,
    sampling_execution_context,
)
from dinkster_inference.z_image import Z_IMAGE_CONTROL_RESIDUAL_SITES

from .assemble import AssembledZImage
from .autoencoder_kl import kl_codec_plugin
from .brownian import BrownianTreeNoise
from .codecs import CodecPlugin
from .denoise import run_denoise, to_batch
from .guidance import ConditioningEvaluation, GuidanceExecutor
from .qwen_text import ZImageTextEncoder
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    brownian_step_noise,
    build_custom_sampling_schedule,
    compile_guidance_plan,
    custom_denoised_callback,
    guided_denoiser,
    narrow_single_stream_custom_sampling,
    resolve_custom_sampling_request,
)
from .sampling_runtime import SingleStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry
from .z_image import ZImage, ZImagePixelCodec, ZImagePixelSpace
from .z_image_control import (
    ZImageControlConditioning,
    snapshot_z_image_control_conditioning,
    validate_z_image_control_resource,
)


class ZImageRuntimeError(ValueError):
    """A Z-Image runtime request violates its native contract."""


def _z_image_control_gain_row(
    row: RealizedGainRow,
    lane_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    site_gains = dict(row.site_gains)
    lane_gains = dict(row.lane_gains)
    timeline_global = row.timeline_gain * row.global_gain
    values = tuple(
        tuple(
            (timeline_global * site_gains.get(site_id, 1.0)) * lane_gains.get(lane_id, 1.0)
            for lane_id in lane_ids
        )
        for site_id in Z_IMAGE_CONTROL_RESIDUAL_SITES
    )
    if any(not math.isfinite(value) for site in values for value in site):
        raise ZImageRuntimeError("gain_domain_mismatch: Z-Image control gain must be finite")
    return values


class ZImageDenoiser:
    """Single-conditioning FLOW evaluator over the native Z-Image DiT."""

    def __init__(
        self,
        model: ZImage | ZImagePixelSpace,
        *,
        control: ZImageControlConditioning | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.control = None if control is None else snapshot_z_image_control_conditioning(control)
        self._control_lane_ids: tuple[str, ...] = ("positive",)
        self._control_values: tuple[tuple[float, ...], ...] = tuple(
            (1.0,) for _ in Z_IMAGE_CONTROL_RESIDUAL_SITES
        )
        self.compute_dtype = compute_dtype

    def set_control_gain_row(
        self,
        lane_ids: tuple[str, ...],
        values: tuple[tuple[float, ...], ...],
    ) -> None:
        if len(values) != len(Z_IMAGE_CONTROL_RESIDUAL_SITES) or any(
            len(site) != len(lane_ids) for site in values
        ):
            raise ZImageRuntimeError("Z-Image control gain row has incompatible dimensions")
        self._control_lane_ids = lane_ids
        self._control_values = values

    def prepare_conditioning(
        self, value: object, *, lane_id: str = "positive"
    ) -> tuple[torch.Tensor, str]:
        if not isinstance(value, Conditioning):
            raise ZImageRuntimeError("Z-Image conditioning must be a Conditioning value")
        if value.pooled is not None:
            raise ZImageRuntimeError("Z-Image conditioning does not accept a pooled vector")
        context = value.embeddings
        if context.ndim != 3 or context.shape[0] < 1 or context.shape[2] != 2560:
            raise ZImageRuntimeError("Z-Image context must have shape [batch,tokens,2560]")
        if lane_id not in ("positive", "negative"):
            raise ZImageRuntimeError("Z-Image guidance lane id is not supported")
        return context, lane_id

    @staticmethod
    def batchable(conditions: tuple[tuple[torch.Tensor, str], ...]) -> bool:
        return bool(conditions) and all(
            condition[0].shape[1:] == conditions[0][0].shape[1:] for condition in conditions[1:]
        )

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: tuple[torch.Tensor, str]
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    def evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[tuple[torch.Tensor, str], ...],
    ) -> tuple[torch.Tensor, ...]:
        if not self.batchable(conditions):
            raise ZImageRuntimeError("Z-Image conditioning batch is empty or incompatible")
        batch = x.shape[0]
        if any(condition[0].shape[0] not in (1, batch) for condition in conditions):
            raise ZImageRuntimeError("Z-Image conditioning batch must be one or match the latent")
        model_input = x.to(dtype=self.compute_dtype)
        if len(conditions) > 1:
            model_input = torch.cat([model_input] * len(conditions), dim=0)
        context = torch.cat(
            [
                condition[0].to(device=x.device, dtype=self.compute_dtype).expand(batch, -1, -1)
                for condition in conditions
            ],
            dim=0,
        )
        timestep = torch.full((model_input.shape[0],), sigma, dtype=torch.float32, device=x.device)
        control = self.control
        control_latent = None if control is None else control.hint
        if control_latent is not None:
            control_latent = control_latent.to(device=x.device, dtype=self.compute_dtype)
            control_latent = to_batch(control_latent, x.shape[0])
            control_latent = torch.cat([control_latent] * len(conditions), dim=0)
        if control is None:
            output = self.model(model_input, timestep, context).float()
        else:
            condition_lane_ids = tuple(condition[1] for condition in conditions)
            if any(lane_id not in self._control_lane_ids for lane_id in condition_lane_ids):
                raise ZImageRuntimeError("Z-Image control gain does not cover a guidance lane")
            active = tuple(
                any(
                    site[self._control_lane_ids.index(lane_id)] != 0.0
                    for site in self._control_values
                )
                for lane_id in condition_lane_ids
            )
            if not any(active):
                output = self.model(model_input, timestep, context).float()
            else:
                validate_z_image_control_resource(control.model, control.model_digest)
                active_indexes = tuple(index for index, enabled in enumerate(active) if enabled)
                input_chunks = model_input.chunk(len(conditions))
                context_chunks = context.chunk(len(conditions))
                timestep_chunks = timestep.chunk(len(conditions))
                assert control_latent is not None
                control_chunks = control_latent.chunk(len(conditions))
                active_lanes = tuple(condition_lane_ids[index] for index in active_indexes)
                active_output = self.model(
                    torch.cat(tuple(input_chunks[index] for index in active_indexes)),
                    torch.cat(tuple(timestep_chunks[index] for index in active_indexes)),
                    torch.cat(tuple(context_chunks[index] for index in active_indexes)),
                    control=control.model,
                    control_latent=torch.cat(
                        tuple(control_chunks[index] for index in active_indexes)
                    ),
                    control_gains=tuple(
                        torch.cat(
                            tuple(
                                torch.full(
                                    (batch, 1, 1),
                                    site[self._control_lane_ids.index(lane_id)],
                                    device=x.device,
                                    dtype=self.compute_dtype,
                                )
                                for lane_id in active_lanes
                            )
                        )
                        for site in self._control_values
                    ),
                ).float()
                if all(active):
                    output = active_output
                else:
                    inactive_indexes = tuple(
                        index for index, enabled in enumerate(active) if not enabled
                    )
                    inactive_output = self.model(
                        torch.cat(tuple(input_chunks[index] for index in inactive_indexes)),
                        torch.cat(tuple(timestep_chunks[index] for index in inactive_indexes)),
                        torch.cat(tuple(context_chunks[index] for index in inactive_indexes)),
                    ).float()
                    active_chunks = iter(active_output.chunk(len(active_indexes)))
                    inactive_chunks = iter(inactive_output.chunk(len(inactive_indexes)))
                    output = torch.cat(
                        tuple(
                            next(active_chunks) if enabled else next(inactive_chunks)
                            for enabled in active
                        )
                    )
        flow_input = x if len(conditions) == 1 else torch.cat([x] * len(conditions), dim=0)
        denoised = calculate_denoised(Parameterization.FLOW, sigma, output, flow_input)
        return tuple(denoised.chunk(len(conditions)))


def _exact_scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


class ZImageRuntime(SingleStreamSamplingRuntime):
    """Family runtime for latent or pixel-space non-Omni Z-Image."""

    sampling_error = ZImageRuntimeError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True
    accepts_z_image_control = True

    def __init__(
        self,
        assembled: AssembledZImage,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._runtime_identity = runtime_identity
        if isinstance(assembled.vae, ZImagePixelCodec):
            self.codec = CodecPlugin(
                descriptor=CodecDescriptor(
                    id="dinkster.z_image_pixel_identity",
                    display_name="Z-Image Pixel Identity",
                    kind="image",
                    latent=assembled.family.single_stream_latent(),
                    supported_dtypes=assembled.family.supported_dtypes,
                    supports_tiling=False,
                ),
                encoder=assembled.vae,
                decoder=assembled.vae,
            )
        else:
            self.codec = replace(
                kl_codec_plugin(assembled.vae),
                compute_dtype=assembled.compute_dtype("vae"),
            )
        self._text_encoder = ZImageTextEncoder(assembled.qwen3_4b)
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _exact_scheduler_registry(scheduler_registry)
        self._guidance = guidance_executor

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        return self._text_encoder.encode(text)

    @staticmethod
    def _sigma_space() -> FlowSigmas:
        return FlowSigmas(shift=3.0)

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return self._sigma_space()

    def sample_custom(
        self,
        latent: CustomSamplingLatentValue,
        *,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue = None,
        request: CustomSamplingRequest[torch.Tensor],
        seed: int = 0,
        guidance: float | None = None,
        denoise_mask: CustomSamplingLatentValue | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        context_windows: ContextWindowsSpec | CompositeWindowPlan | None = None,
        control: ZImageControlConditioning | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        capture_denoised: bool = True,
    ) -> CustomSamplingResult[torch.Tensor]:
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=ZImageRuntimeError,
        )
        channels = self.family.single_stream_latent().channels
        if latent.ndim != 4 or latent.shape[1] != channels:
            raise ZImageRuntimeError(
                f"Z-Image input must have shape [batch,{channels},height,width]"
            )
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        if control is not None and self.family.id != "dinkster.z_image":
            raise ZImageRuntimeError("Z-Image control patches require latent Z-Image")
        if control is not None and type(control) is not ZImageControlConditioning:
            raise TypeError("control must be an exact ZImageControlConditioning")
        admitted_control = control
        if admitted_control is not None and (
            admitted_control.hint.shape[0] not in (1, latent.shape[0])
            or admitted_control.hint.shape[-2:] != latent.shape[-2:]
        ):
            raise ZImageRuntimeError(
                "Z-Image control hint batch must be one or match the sample, and spatial"
                " geometry must match the sample"
            )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=ZImageRuntimeError
        )
        space = self._sigma_space()
        schedule = build_custom_sampling_schedule(request.sigmas, space, sampler, flow=True)
        noise_sampler: BrownianTreeNoise | None = brownian_step_noise(
            sampler, schedule, latent, seed=seed, device=device
        )
        realized_timeline = (
            None
            if request.timeline is None
            else realize_sampling_timeline(
                request.timeline,
                tuple(float(sigma) for sigma in schedule.sigmas),
            )
        )
        plan = compile_guidance_plan(cond, cfg, sampler, self._guidance)
        include_uncond = plan.conditions[1].conditioning is not None and (
            plan.needs_unconditional or plan.has_strategy
        )
        control_lane_ids = ("positive", "negative") if include_uncond else ("positive",)
        control_table = None
        control_facts: tuple[str, ...] = ()
        off_grid_control = admitted_control is not None and sampler.id in {
            "dinkster.dpm_fast",
            "dinkster.dpm_adaptive",
        }
        if admitted_control is not None and len(schedule.sigmas) > 1:
            gain = admitted_control.gain
            if gain is None:
                start_sigma = space.percent_to_sigma(
                    admitted_control.application.window.start_percent
                )
                end_sigma = space.percent_to_sigma(admitted_control.application.window.end_percent)
                gain = ContributionGain(
                    DirectGainTableCurve(
                        tuple(
                            admitted_control.application.strength
                            if end_sigma <= sigma <= start_sigma
                            else 0.0
                            for sigma in schedule.sigmas[:-1]
                        )
                    ),
                    1.0,
                )
            site_keys = tuple(key for key, _ in gain.site_gains)
            if site_keys and site_keys != Z_IMAGE_CONTROL_RESIDUAL_SITES:
                raise ZImageRuntimeError(
                    "operator_site_mismatch: Z-Image control site gains must cover exactly "
                    + ", ".join(Z_IMAGE_CONTROL_RESIDUAL_SITES)
                )
            lane_keys = tuple(key for key, _ in gain.lane_gains)
            if lane_keys and lane_keys != tuple(sorted(control_lane_ids)):
                raise ZImageRuntimeError(
                    "invalid_gain_schedule: Z-Image control lane gains must cover exactly the"
                    f" active guidance lanes {tuple(sorted(control_lane_ids))!r}"
                )
            if gain.effect_mask_digests:
                raise ZImageRuntimeError("Z-Image control effect masks are not implemented")
            timeline = (
                realized_timeline.executed
                if realized_timeline is not None
                else executed_sampling_timeline(tuple(float(sigma) for sigma in schedule.sigmas))
            )
            control_table = realize_gain_table(gain, timeline)
            rows = tuple(
                _z_image_control_gain_row(row, control_lane_ids) for row in control_table.rows
            )
            full_window = (
                admitted_control.application.window.start_percent == 0.0
                and admitted_control.application.window.end_percent == 1.0
            )
            if off_grid_control and (not full_window or any(row != rows[0] for row in rows[1:])):
                raise ZImageRuntimeError(
                    f"sampler {sampler.id} supports only constant Z-Image control gain over the"
                    " full application window because its internal evaluation timeline does not"
                    " map to executed sigma rows"
                )
            control_facts = (
                f"control.child={admitted_control.application.child_id}",
                f"control.model={admitted_control.model_digest}",
                f"control.hint={admitted_control.hint_digest}",
                *(f"control.{fact}" for fact in contribution_gain_slot_facts(gain, control_table)),
            )
        if compute_dtype is None:
            compute_dtype = self.assembled.compute_dtype("diffusion") or torch.bfloat16
        evaluator = ZImageDenoiser(
            self.assembled.diffusion,
            control=admitted_control,
            compute_dtype=compute_dtype,
        )
        if control_table is not None and off_grid_control:
            evaluator.set_control_gain_row(
                control_lane_ids,
                _z_image_control_gain_row(control_table.rows[0], control_lane_ids),
            )
        evaluator_identity = "dinkster.z-image.conditioning.v1"
        if control_facts:
            digest = hashlib.sha256("\n".join((*control_facts, "")).encode()).hexdigest()
            evaluator_identity += f":intervention-plan={digest}"
        report_state: SamplingStateCallback | None
        captured: list[torch.Tensor]
        if capture_denoised:
            report_state, captured = custom_denoised_callback(self.family, on_state)
        else:
            report_state, captured = on_state, []
        denoiser = guided_denoiser(
            ConditioningEvaluation(
                lambda value, role: evaluator.prepare_conditioning(
                    value,
                    lane_id=("negative" if role is GuidanceRole.UNCONDITIONAL else "positive"),
                ),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                evaluator_identity=lambda _role: evaluator_identity,
                standard_activation_memory_factor=self.family.memory_factor,
            ),
            input=latent,
            executor=self._guidance,
            plan=plan,
            execution=sampling_execution_context(
                sigmas=schedule.sigmas, seed=seed, on_step=on_step, on_state=report_state
            ),
        )

        def apply_control_step(index: int) -> None:
            assert control_table is not None
            row = control_table.rows[index]
            require_realized_sampling_step(index, row.sigma, row.progress)
            evaluator.set_control_gain_row(
                control_lane_ids,
                _z_image_control_gain_row(row, control_lane_ids),
            )

        output = run_denoise(
            denoiser,
            request.build_solver(realized_timeline=realized_timeline),
            latent=latent,
            noise=noise,
            sigmas=schedule.sigmas,
            initial_sigma=schedule.initial_sigma,
            family=self.family,
            seed=seed,
            noise_kind=sampler.noise,
            noise_sampler=noise_sampler,
            percent_to_sigma=space.percent_to_sigma,
            device=device,
            on_step=on_step,
            on_step_begin=(
                None if control_table is None or off_grid_control else apply_control_step
            ),
            on_state=report_state,
            denoise_mask=denoise_mask,
        )
        return CustomSamplingResult(output, captured[-1] if captured else None)

    def _ksampler_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        extra = super()._ksampler_kwargs(kwargs)
        if "window_plan" in extra:
            extra["context_windows"] = extra.pop("window_plan")
        return extra

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)


__all__ = [
    "ZImageDenoiser",
    "ZImageRuntime",
    "ZImageRuntimeError",
]
