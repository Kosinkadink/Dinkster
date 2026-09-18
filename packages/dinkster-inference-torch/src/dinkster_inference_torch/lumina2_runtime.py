"""Native Lumina Image 2.0 text encoding and custom sampling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch
from dinkster_inference import (
    GEMMA2_LUMINA_2B_CONFIG,
    LUMINA2,
    LUMINA2_CONFIG,
    Conditioning,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    FlowSigmas,
    FluxFlowSigmas,
    GuidanceRole,
    InpaintConditioning,
    ModelFamily,
    Parameterization,
    Registry,
    SamplerDescriptor,
    SamplingStateCallback,
    SchedulerDescriptor,
    StepCallback,
    calculate_denoised,
    sampling_execution_context,
    tokenize_lumina2_prompt,
)

from .assemble import AssembledLumina2
from .autoencoder_kl import kl_codec_plugin
from .brownian import BrownianTreeNoise
from .denoise import run_denoise
from .gemma_text import GemmaTextModel
from .gemma_tokenizer import LUMINA2_TOKENIZER_ATTRIBUTE, GemmaSentencePieceTokenizer
from .guidance import ConditioningEvaluation, GuidanceExecutor
from .operations import module_compute_device
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
from .sampling_runtime import FlowSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry
from .z_image import ZImage


class Lumina2RuntimeError(ValueError):
    """A Lumina2 runtime request violates its native contract."""


def _sigma_space(sampling_shift: float | None = None) -> FlowSigmas:
    shift = LUMINA2_CONFIG.sampling_shift if sampling_shift is None else sampling_shift
    if type(shift) is not float or not math.isfinite(shift) or shift <= 0.0:
        raise Lumina2RuntimeError("sampling_shift must be a positive finite float")
    return FlowSigmas(shift=shift)


def _exact_scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


class Lumina2TextRuntime:
    """Text-only Lumina2 encoder over an independently resident Gemma component."""

    def __init__(self, text: GemmaTextModel) -> None:
        if text.config is not GEMMA2_LUMINA_2B_CONFIG:
            raise Lumina2RuntimeError("Lumina2 text runtime requires the Gemma 2 2B profile")
        model = text.__dict__.get(LUMINA2_TOKENIZER_ATTRIBUTE)
        if type(model) is not bytes:
            raise Lumina2RuntimeError("Lumina2 text component has no SentencePiece model")
        self.text = text
        self.tokenizer = GemmaSentencePieceTokenizer(
            model, expected_vocab_size=GEMMA2_LUMINA_2B_CONFIG.vocab_size
        )

    def _encode(self, ids: Sequence[int], attention: Sequence[int]) -> torch.Tensor:
        device = module_compute_device(self.text)
        token_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(attention, dtype=torch.long, device=device).unsqueeze(0)
        return self.text(token_ids, attention_mask)[:, -2]

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_lumina2_prompt(text, encode=self.tokenizer.encode)
        hidden = self._encode(tokens.ids, tokens.attention_mask)
        if any(weight != 1.0 for weight in tokens.weights):
            empty_ids = (GEMMA2_LUMINA_2B_CONFIG.bos_token_id,) + (
                GEMMA2_LUMINA_2B_CONFIG.pad_token_id,
            ) * (len(tokens.ids) - 1)
            empty_attention = (1,) + (0,) * (len(tokens.ids) - 1)
            baseline = self._encode(empty_ids, empty_attention)
            weights = torch.tensor(
                tokens.weights, dtype=hidden.dtype, device=hidden.device
            ).reshape(1, -1, 1)
            hidden = torch.where(
                weights == 1.0,
                hidden,
                (hidden - baseline) * weights + baseline,
            )
        return Conditioning(hidden.float(), None)


class Lumina2Denoiser:
    """Single-conditioning flow evaluator over the native Lumina2 DiT."""

    def __init__(
        self,
        model: ZImage,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def prepare_conditioning(
        self, value: object, *, lane_id: str = "positive"
    ) -> tuple[torch.Tensor, str]:
        if not isinstance(value, Conditioning):
            raise Lumina2RuntimeError("Lumina2 conditioning must be a Conditioning value")
        if value.pooled is not None:
            raise Lumina2RuntimeError("Lumina2 conditioning does not accept a pooled vector")
        context = value.embeddings
        if context.ndim != 3 or context.shape[0] < 1 or context.shape[2] != 2304:
            raise Lumina2RuntimeError("Lumina2 context must have shape [batch,tokens,2304]")
        if lane_id not in ("positive", "negative"):
            raise Lumina2RuntimeError("Lumina2 guidance lane id is not supported")
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
            raise Lumina2RuntimeError("Lumina2 conditioning batch is empty or incompatible")
        rank_five = x.ndim == 5
        if x.ndim not in (4, 5) or (rank_five and x.shape[2] != 1):
            raise Lumina2RuntimeError("Lumina2 accepts image or single-frame video latents")
        batch = x.shape[0]
        if any(condition[0].shape[0] not in (1, batch) for condition in conditions):
            raise Lumina2RuntimeError("Lumina2 conditioning batch must be one or match the latent")
        image = x[:, :, 0] if rank_five else x
        model_input = image.to(dtype=self.compute_dtype)
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
        output = self.model(model_input, timestep, context).float()
        if rank_five:
            output = output.unsqueeze(2)
        flow_input = x if len(conditions) == 1 else torch.cat([x] * len(conditions), dim=0)
        denoised = calculate_denoised(Parameterization.FLOW, sigma, output, flow_input)
        return tuple(denoised.chunk(len(conditions)))


@dataclass(frozen=True)
class _Lumina2DiffusionAssembly:
    diffusion: ZImage
    family: ModelFamily = LUMINA2
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


class Lumina2DiffusionRuntime(FlowSamplingRuntime):
    """Diffusion-only Lumina2 runtime implementing the custom-sampling seam."""

    streamed_residency_components = frozenset()
    sampling_error = Lumina2RuntimeError
    supports_sampling_shift = True
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True

    def __init__(
        self,
        diffusion: ZImage,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        self.assembled: _Lumina2DiffusionAssembly | AssembledLumina2 = _Lumina2DiffusionAssembly(
            diffusion, compute=compute_dtype
        )
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _exact_scheduler_registry(scheduler_registry)
        self._guidance: GuidanceExecutor | None = None

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas | FluxFlowSigmas:
        if self._sampling_space_override is not None:
            if sampling_shift is not None:
                raise Lumina2RuntimeError("cannot combine a sampling shift with an explicit space")
            return self._sampling_space_override
        return _sigma_space(sampling_shift)

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
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        capture_denoised: bool = True,
        sampling_shift: float | None = None,
    ) -> CustomSamplingResult[torch.Tensor]:
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=Lumina2RuntimeError,
        )
        channels = LUMINA2_CONFIG.latent_channels
        if (
            latent.ndim not in (4, 5)
            or latent.shape[1] != channels
            or (latent.ndim == 5 and latent.shape[2] != 1)
        ):
            raise Lumina2RuntimeError(
                f"Lumina2 latent must be [batch,{channels},height,width] or single-frame rank 5"
            )
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=Lumina2RuntimeError
        )
        if compute_dtype is None:
            compute_dtype = self.assembled.compute_dtype("diffusion") or torch.bfloat16
        if device is None:
            device = module_compute_device(self.assembled.diffusion)
        space = self.sampling_sigma_space(sampling_shift)
        schedule = build_custom_sampling_schedule(request.sigmas, space, sampler, flow=True)
        noise_sampler: BrownianTreeNoise | None = brownian_step_noise(
            sampler, schedule, latent, seed=seed, device=device
        )
        plan = compile_guidance_plan(cond, cfg, sampler, self._guidance)
        evaluator = Lumina2Denoiser(self.assembled.diffusion, compute_dtype=compute_dtype)
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
                evaluator_identity=lambda _role: "dinkster.lumina2.conditioning.v1",
                standard_activation_memory_factor=self.family.memory_factor,
            ),
            input=latent,
            executor=self._guidance,
            plan=plan,
            execution=sampling_execution_context(
                sigmas=schedule.sigmas, seed=seed, on_step=on_step, on_state=report_state
            ),
        )
        output = run_denoise(
            denoiser,
            request.build_solver(),
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
            on_state=report_state,
            denoise_mask=denoise_mask,
        )
        return CustomSamplingResult(output, captured[-1] if captured else None)

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise Lumina2RuntimeError("Lumina2 diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise Lumina2RuntimeError("Lumina2 diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise Lumina2RuntimeError("Lumina2 diffusion component carries no VAE codec")


class Lumina2Runtime(Lumina2DiffusionRuntime):
    """Complete checkpoint runtime using the same Lumina2 custom-sampling engine."""

    def __init__(
        self,
        assembled: AssembledLumina2,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        super().__init__(
            assembled.diffusion,
            runtime_identity=runtime_identity,
            compute_dtype=assembled.compute_dtype("diffusion") or torch.bfloat16,
            sampler_registry=sampler_registry,
            scheduler_registry=scheduler_registry,
        )
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._guidance = guidance_executor
        self._text_encoder = Lumina2TextRuntime(assembled.gemma2_2b)
        self.codec = replace(
            kl_codec_plugin(assembled.vae), compute_dtype=assembled.compute_dtype("vae")
        )

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        return self._text_encoder.encode_text(text)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)


__all__ = [
    "Lumina2Denoiser",
    "Lumina2DiffusionRuntime",
    "Lumina2Runtime",
    "Lumina2RuntimeError",
    "Lumina2TextRuntime",
]
