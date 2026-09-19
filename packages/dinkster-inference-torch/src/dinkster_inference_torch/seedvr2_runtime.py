"""Native SeedVR2 conditioning, sampling, and causal video codec runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from dinkster_inference import (
    SEEDVR2,
    SEEDVR2_CODEC,
    SEEDVR2_SIGMAS,
    Conditioning,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    InpaintConditioning,
    ModelFamily,
    Registry,
    SamplerDescriptor,
    SamplingStateCallback,
    SchedulerDescriptor,
    SigmaSpace,
    StepCallback,
    sampling_execution_context,
)

if TYPE_CHECKING:
    from .checkpoint_runtime import ComponentAssembly

from .brownian import BrownianTreeNoise
from .denoise import run_denoise
from .guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .operations import bound_compute_dtype, module_compute_device
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
from .seedvr2_dit import NaDiT
from .seedvr2_vae import VideoAutoencoderKLWrapper
from .solvers import torch_sampler_registry


def _module_compute_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for owner in module.modules():
        dtype = bound_compute_dtype(owner)
        if dtype is not None:
            return dtype
    return next(module.parameters()).dtype


class SeedVR2RuntimeError(ValueError):
    """A SeedVR2 runtime request violates its native contract."""


@dataclass(frozen=True)
class SeedVR2Conditioning(Conditioning[torch.Tensor]):
    """One SeedVR2 restoration latent bound to a built-in context branch."""

    branch: str = "positive"
    component_identity: str = ""

    def __post_init__(self) -> None:
        if self.pooled is not None:
            raise SeedVR2RuntimeError("SeedVR2 conditioning does not accept a pooled vector")
        if (
            self.embeddings.ndim != 5
            or self.embeddings.shape[0] < 1
            or self.embeddings.shape[1] != 17
        ):
            raise SeedVR2RuntimeError(
                "SeedVR2 conditioning must have shape [batch,17,time,height,width]"
            )
        if self.branch not in ("positive", "negative"):
            raise SeedVR2RuntimeError("SeedVR2 conditioning branch must be positive or negative")
        if type(self.component_identity) is not str or not self.component_identity:
            raise SeedVR2RuntimeError("SeedVR2 conditioning requires a component identity")


def seedvr2_conditioning(
    latent: torch.Tensor,
    *,
    component_identity: str,
) -> tuple[SeedVR2Conditioning, SeedVR2Conditioning]:
    """Build the positive and negative restoration lanes from one VAE latent."""

    if latent.ndim != 5 or latent.shape[1] != 16:
        raise SeedVR2RuntimeError(
            "SeedVR2 conditioning requires a [batch,16,time,height,width] VAE latent"
        )
    condition = torch.cat((latent, latent.new_ones((latent.shape[0], 1, *latent.shape[2:]))), dim=1)
    return (
        SeedVR2Conditioning(condition, branch="positive", component_identity=component_identity),
        SeedVR2Conditioning(condition, branch="negative", component_identity=component_identity),
    )


@dataclass(frozen=True, slots=True)
class _PreparedSeedVR2Conditioning:
    condition: torch.Tensor
    branch: str


class SeedVR2Denoiser:
    """FLOW evaluator over one native SeedVR2 diffusion component."""

    def __init__(
        self,
        model: NaDiT,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.runtime_identity = runtime_identity
        self.compute_dtype = compute_dtype

    def prepare_conditioning(self, value: object) -> _PreparedSeedVR2Conditioning:
        if type(value) is not SeedVR2Conditioning:
            raise SeedVR2RuntimeError("SeedVR2 sampling requires exact SeedVR2Conditioning values")
        typed = value
        if typed.component_identity != self.runtime_identity:
            raise SeedVR2RuntimeError(
                "SeedVR2 conditioning was built for a different diffusion component"
            )
        return _PreparedSeedVR2Conditioning(typed.embeddings, typed.branch)

    @staticmethod
    def batchable(conditions: tuple[_PreparedSeedVR2Conditioning, ...]) -> bool:
        return bool(conditions) and all(
            condition.branch == conditions[0].branch
            and condition.condition.shape[1:] == conditions[0].condition.shape[1:]
            for condition in conditions[1:]
        )

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: _PreparedSeedVR2Conditioning,
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[_PreparedSeedVR2Conditioning, ...],
    ) -> None:
        if not self.batchable(conditions):
            raise SeedVR2RuntimeError("SeedVR2 conditioning batch is empty or incompatible")
        batch = x.shape[0]
        if any(condition.condition.shape[0] not in (1, batch) for condition in conditions):
            raise SeedVR2RuntimeError("SeedVR2 conditioning batch must be one or match the latent")

    @staticmethod
    def _conditioning_timestep(sigma: float) -> float:
        return SEEDVR2_SIGMAS.timestep(sigma)

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[_PreparedSeedVR2Conditioning],
    ) -> torch.Tensor:
        condition_latent = torch.cat(
            tuple(
                condition.condition.to(device=batch.latent.device, dtype=self.compute_dtype).expand(
                    batch.batch_size, -1, -1, -1, -1
                )
                for condition in batch.conditions
            ),
            dim=0,
        )
        branch = batch.conditions[0].branch
        with self.model.materialized_text_conditioning(
            branch, device=batch.latent.device, dtype=self.compute_dtype
        ) as model_context:
            context = model_context.unsqueeze(0).expand(batch.model_input.shape[0], -1, -1)
            return self.model(
                batch.model_input,
                batch.timestep,
                context,
                condition=condition_latent,
                transformer_options={"cond_or_uncond": [branch] * batch.model_input.shape[0]},
            ).float()


def _exact_scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


@dataclass(frozen=True)
class _SeedVR2DiffusionAssembly:
    diffusion: NaDiT
    family: ModelFamily = SEEDVR2
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


class SeedVR2DiffusionRuntime(SingleStreamSamplingRuntime):
    """Diffusion-only SeedVR2 custom sampling runtime."""

    streamed_residency_components = frozenset()
    sampling_error = SeedVR2RuntimeError
    supports_denoised_capture = True

    def __init__(
        self,
        diffusion: NaDiT,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = _SeedVR2DiffusionAssembly(diffusion, compute=compute_dtype)
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _exact_scheduler_registry(scheduler_registry)
        self._guidance = guidance_executor

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return SEEDVR2_SIGMAS

    def sampling_memory_requirements(self, latent_shape: Sequence[int]) -> tuple[int, int]:
        if len(latent_shape) != 5 or any(
            type(value) is not int or value < 1 for value in latent_shape
        ):
            raise ValueError("SeedVR2 sampling memory requires a positive rank-5 latent shape")
        compute_dtype = self.assembled.compute_dtype("diffusion")
        if compute_dtype in (torch.float16, torch.bfloat16):
            element_size = 2
        elif compute_dtype == torch.float32:
            element_size = 4
        else:
            raise TypeError("SeedVR2 sampling memory requires a supported compute dtype")
        batch_area = latent_shape[0] * latent_shape[2] * latent_shape[3] * latent_shape[4]
        minimum = int(batch_area * element_size * 0.01 * 2.0 * 1024 * 1024)
        return minimum * 2, minimum

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
    ) -> CustomSamplingResult[torch.Tensor]:
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=SeedVR2RuntimeError,
        )
        if latent.ndim != 5 or latent.shape[1] != 16 or latent.shape[2] < 1:
            raise SeedVR2RuntimeError("SeedVR2 latent must have shape [batch,16,time,height,width]")
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=SeedVR2RuntimeError
        )
        if compute_dtype is None:
            compute_dtype = self.assembled.compute_dtype("diffusion") or torch.bfloat16
        if device is None:
            device = module_compute_device(self.assembled.diffusion)
        schedule = build_custom_sampling_schedule(
            request.sigmas, SEEDVR2_SIGMAS, sampler, flow=True
        )
        noise_sampler: BrownianTreeNoise | None = brownian_step_noise(
            sampler, schedule, latent, seed=seed, device=device
        )
        plan = compile_guidance_plan(cond, cfg, sampler, self._guidance)
        evaluator = SeedVR2Denoiser(
            self.assembled.diffusion,
            runtime_identity=self.runtime_identity,
            compute_dtype=compute_dtype,
        )
        report_state: SamplingStateCallback | None
        captured: list[torch.Tensor]
        if capture_denoised:
            report_state, captured = custom_denoised_callback(self.family, on_state)
        else:
            report_state, captured = on_state, []
        denoiser = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                evaluator_identity=lambda role: f"dinkster.seedvr2.{role.value}.v1",
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
            percent_to_sigma=SEEDVR2_SIGMAS.percent_to_sigma,
            device=device,
            on_step=on_step,
            on_state=report_state,
            denoise_mask=denoise_mask,
        )
        return CustomSamplingResult(output, captured[-1] if captured else None)

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise SeedVR2RuntimeError("SeedVR2 uses built-in conditioning, not a text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise SeedVR2RuntimeError("SeedVR2 diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise SeedVR2RuntimeError("SeedVR2 diffusion component carries no VAE codec")


class _SeedVR2Codec:
    descriptor = SEEDVR2_CODEC
    accepts_batched_video = True
    accepts_image_batch_latent = True
    manages_input_device = True

    def __init__(self, vae: VideoAutoencoderKLWrapper, compute_dtype: torch.dtype | None) -> None:
        self.vae = vae
        self.compute_dtype = compute_dtype

    def _cast(self, value: torch.Tensor) -> torch.Tensor:
        value = value if self.compute_dtype is None else value.to(dtype=self.compute_dtype)
        return value.to(module_compute_device(self.vae))

    def _video_value(
        self,
        value: torch.Tensor,
        *,
        channels: int,
        name: str,
    ) -> tuple[torch.Tensor, bool]:
        rank = self.descriptor.latent.dimensions + 2
        image_batch = value.ndim == rank - 1
        if image_batch:
            value = value.unsqueeze(2)
        if value.ndim != rank or value.shape[1] != channels or value.shape[2] < 1:
            raise SeedVR2RuntimeError(
                f"SeedVR2 {name} must have {channels} channels and rank {rank - 1} or {rank}"
            )
        return value, image_batch

    def _content(self, value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        return self._video_value(
            value,
            channels=self.descriptor.content_channels,
            name="content",
        )

    def _latent(self, value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        return self._video_value(
            value,
            channels=self.descriptor.latent.channels,
            name="latent",
        )

    def _restore_image_batch(
        self, value: torch.Tensor, image_batch: bool, name: str
    ) -> torch.Tensor:
        if not image_batch:
            return value
        rank = self.descriptor.latent.dimensions + 2
        if value.ndim != rank or value.shape[2] != 1:
            raise SeedVR2RuntimeError(
                f"SeedVR2 {name} for an image batch must have one temporal element"
            )
        return value.squeeze(2)

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        content, image_batch = self._content(content)
        content = self._cast(content * 2.0 - 1.0)
        encoded = self.vae.comfy_format_encoded(self.vae.encode(content).float())
        encoded = self._restore_image_batch(encoded, image_batch, "latent")
        return encoded.to("cpu")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent, image_batch = self._latent(latent)
        decoded = self.vae.decode(self._cast(latent))
        decoded = ((decoded.float() + 1.0) / 2.0).clamp_(0.0, 1.0)
        decoded = self._restore_image_batch(decoded, image_batch, "content")
        return decoded.to("cpu")

    @staticmethod
    def _tiling(
        tile: tuple[int, ...] | None,
        overlap: tuple[int, ...] | None,
        *,
        decode: bool,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        defaults = SEEDVR2_CODEC.tiling
        assert defaults is not None
        selected_tile = (
            (defaults.decode_tile if decode else defaults.encode_tile) if tile is None else tile
        )
        selected_overlap = (
            (defaults.decode_overlap if decode else defaults.encode_overlap)
            if overlap is None
            else overlap
        )
        if len(selected_tile) != 3 or len(selected_overlap) != 3:
            raise SeedVR2RuntimeError("SeedVR2 tiled codec geometry must be temporal,height,width")
        if any(value < 1 for value in selected_tile) or any(
            value < 0 for value in selected_overlap
        ):
            raise SeedVR2RuntimeError(
                "SeedVR2 tile sizes must be positive and overlaps non-negative"
            )
        return selected_tile[1:], selected_overlap[1:]

    def encode_tiled(
        self,
        content: torch.Tensor,
        *,
        tile: tuple[int, ...] | None = None,
        overlap: tuple[int, ...] | None = None,
        output_device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        on_tile: Any | None = None,
    ) -> torch.Tensor:
        if on_tile is not None:
            raise SeedVR2RuntimeError("SeedVR2 codec does not expose per-tile callbacks")
        tile_size, tile_overlap = self._tiling(tile, overlap, decode=False)
        content, image_batch = self._content(content)
        content = self._cast(content * 2.0 - 1.0)
        if tile_overlap[0] != tile_overlap[1]:
            from .seedvr2_vae import tiled_vae

            encoded = tiled_vae(
                content,
                self.vae,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                temporal_size=None,
                temporal_overlap=None,
                encode=True,
            )
        else:
            encoded = self.vae.encode_tiled(
                content,
                tile_x=tile_size[1],
                tile_y=tile_size[0],
                overlap=tile_overlap[0],
            )
        encoded = self.vae.comfy_format_encoded(encoded.float())
        encoded = self._restore_image_batch(encoded, image_batch, "latent")
        return encoded.to(device=output_device, dtype=dtype)

    def decode_tiled(
        self,
        latent: torch.Tensor,
        *,
        tile: tuple[int, ...] | None = None,
        overlap: tuple[int, ...] | None = None,
        output_device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        on_tile: Any | None = None,
    ) -> torch.Tensor:
        if on_tile is not None:
            raise SeedVR2RuntimeError("SeedVR2 codec does not expose per-tile callbacks")
        tile_size, tile_overlap = self._tiling(tile, overlap, decode=True)
        scale = SEEDVR2_CODEC.latent.spatial_downscale
        latent, image_batch = self._latent(latent)
        decoded = self.vae.decode(
            self._cast(latent),
            seedvr2_tiling={
                "enable_tiling": True,
                "tile_size": (tile_size[0] * scale, tile_size[1] * scale),
                "tile_overlap": (tile_overlap[0] * scale, tile_overlap[1] * scale),
                "temporal_size": None,
                "temporal_overlap": None,
            },
        )
        decoded = ((decoded.float() + 1.0) / 2.0).clamp_(0.0, 1.0)
        decoded = self._restore_image_batch(decoded, image_batch, "content")
        return decoded.to(device=output_device, dtype=dtype)


class SeedVR2CodecRuntime:
    """Codec facade over one independently resident SeedVR2 VAE."""

    def __init__(
        self,
        vae: VideoAutoencoderKLWrapper,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.vae = vae
        self.codec = _SeedVR2Codec(vae, compute_dtype)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)

    def encode_content_tiled(
        self,
        content: torch.Tensor,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> torch.Tensor:
        return self.codec.encode_tiled(content, tile=tile, overlap=overlap)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def decode_latent_tiled(
        self,
        latent: torch.Tensor,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> torch.Tensor:
        return self.codec.decode_tiled(latent, tile=tile, overlap=overlap)


def checkpoint_codec(
    assembled: ComponentAssembly,
) -> _SeedVR2Codec | None:
    vae = assembled.components.get("vae")
    if vae is None:
        return None
    return _SeedVR2Codec(cast(VideoAutoencoderKLWrapper, vae), _module_compute_dtype(vae))


__all__ = [
    "SeedVR2CodecRuntime",
    "SeedVR2Conditioning",
    "SeedVR2Denoiser",
    "SeedVR2DiffusionRuntime",
    "SeedVR2RuntimeError",
    "seedvr2_conditioning",
]
