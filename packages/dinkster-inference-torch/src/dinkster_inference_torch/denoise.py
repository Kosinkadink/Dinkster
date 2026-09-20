"""One native denoise run: the bridge from an assembled classic-Flux
model to the torch-free sampling substrate.

FluxDenoiser is the executing half of the model evaluation contract:
it owns one conditioning evaluation, compatible batch evaluation,
dtype casts, and the sigma -> timestep lift, mirroring
comfy/model_base.py BaseModel._apply_model plus calc_cond_batch's
model-call batching @ b78cec87 for the classic Flux path.
run_denoise is KSAMPLER.sample + CFGGuider.inner_sample reduced to
the pieces that exist natively today: latent process-in (with the
reference's empty-latent guard), noise scaling, solver drive,
inverse noise scaling, latent process-out. Sampling state stays
float32 throughout, exactly like the reference (outer_sample casts
noise/latent to fp32; the model computes at its own dtype and its
output is lifted back).

Deviations from the reference, all deliberate:

- prepare_noise seeds a private CPU generator instead of the global
  torch.manual_seed (identical draw sequence, no global RNG
  mutation), and omits the noise_inds batch-skip feature (ROADMAP:
  Native inference).
- Compatible conditioning lanes repeat their token sequences to a
  bounded common multiple before batching, matching comfy/conds.py
  CONDCrossAttn.concat @ b78cec87.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch
from dinkster_inference import (
    Conditioning,
    Denoiser,
    LatentDescriptor,
    ModelFamily,
    MultiStreamLatent,
    NoiseKind,
    NoiseSampler,
    Parameterization,
    SamplerInfo,
    SamplingDescriptor,
    SamplingStateCallback,
    SamplingStateEvent,
    SolverFn,
    SolverStateEvent,
    StepCallback,
    StepEvent,
    UncondDenoiser,
    is_flow_parameterization,
    max_denoise,
)
from dinkster_inference.sampling import (
    AutoregressiveDenoiser,
    run_step_begin_solver,
    solver_has_sampling_timeline,
    solver_sampling_cache,
    solver_supports_step_begin,
)

from ._conditioning_layout import cross_attn_repeat, declared_token_count, repeat_cross_attn
from .brownian import BrownianTreeNoise
from .flux import Flux
from .flux_window import flux_window_position_ids
from .guidance import (
    ConditioningBatch,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .latent_streams import reshape_latent_mask
from .parameterizations import calculate_input, inverse_noise_scaling, noise_scaling

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The reference's distilled-guidance default for guidance-embedded
#: Flux models (comfy/model_base.py Flux.extra_conds @ b78cec87).
FLUX_GUIDANCE_DEFAULT = 3.5
FLUX_GUIDANCE_DISABLED = "disabled"
_LOGGER = logging.getLogger(__name__)


class DenoiseError(ValueError):
    """A denoise run this runtime cannot execute: conditioning that
    does not fit the model, guidance against a non-distilled model,
    mismatched noise/latent shapes, or a brownian schedule with no
    positive sigma to bound the tree."""


def _noise_scaling(
    parameterization: Parameterization,
    sigma: float,
    noise: torch.Tensor,
    latent: torch.Tensor,
    *,
    max_denoise: bool = False,
) -> torch.Tensor:
    """Executed non-flow noise scaling on ComfyUI's float32 tensor kernels."""
    if is_flow_parameterization(parameterization):
        return noise_scaling(
            parameterization,
            sigma,
            noise,
            latent,
            max_denoise=max_denoise,
        )
    sigma_tensor = torch.tensor(sigma, device=noise.device, dtype=torch.float32)
    if max_denoise:
        noise = noise * torch.sqrt(1.0 + sigma_tensor**2.0)
    else:
        noise = noise * sigma_tensor
    noise += latent
    return noise


def _checked_flux_cond(
    cond: Conditioning[torch.Tensor], what: str, vec_in_dim: int | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A Flux conditioning's (context, y) pair, validated for the model.

    Classic Flux consumes compose_flux_conditioning's T5 sequence plus
    CLIP-L pooled vector. The accepted vector-free Ovis architecture consumes
    only its Qwen sequence; any incidental pooled value is deliberately not
    forwarded because the reference constructs no vector embedder and skips y.
    """
    if cond.embeddings.ndim != 3:
        raise DenoiseError(
            f"{what} embeddings must be [batch x tokens x features],"
            f" got shape {tuple(cond.embeddings.shape)}"
        )
    token_count = declared_token_count(cond)
    if token_count is not None and cond.embeddings.shape[1] != token_count:
        raise DenoiseError(
            f"{what} embeddings have {cond.embeddings.shape[1]} token rows;"
            f" the family encoder declared {token_count}"
        )
    if vec_in_dim is None:
        return cond.embeddings, None
    if cond.pooled is None:
        raise DenoiseError(
            f"{what} needs the pooled vector (compose_flux_conditioning);"
            " got Conditioning.pooled=None"
        )
    if cond.pooled.ndim != 2:
        raise DenoiseError(
            f"{what} pooled must be [batch x features], got shape {tuple(cond.pooled.shape)}"
        )
    if cond.embeddings.shape[0] != cond.pooled.shape[0]:
        raise DenoiseError(
            f"{what} embeddings batch {cond.embeddings.shape[0]} and"
            f" pooled batch {cond.pooled.shape[0]} disagree"
        )
    return cond.embeddings, cond.pooled


def to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    """Resize a conditioning tensor over the latent batch: the
    reference's repeat_to_batch_size (comfy/utils.py @ b78cec87, as
    called by CONDRegular.process_cond). Oversized sources truncate
    to the first ``batch`` rows; undersized sources repeat cyclically
    (whole-tensor repeat, then truncate), so e.g. batch 2 -> 3 yields
    rows [0, 1, 0]. Never refuses, matching the reference."""
    src = tensor.shape[0]
    if src == batch:
        return tensor
    if src > batch:
        return tensor.narrow(0, 0, batch)
    reps = -(batch // -src)  # ceil(batch / src), like the reference
    return tensor.repeat(reps, *([1] * (tensor.dim() - 1))).narrow(0, 0, batch)


FluxGuidance: TypeAlias = float | Literal["disabled"] | None
FluxCondition: TypeAlias = (
    tuple[torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]]
)


def _flux_condition_parts(
    condition: FluxCondition,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]]:
    if len(condition) == 2:
        return condition[0], condition[1], ()
    return condition


class FluxDenoiser:
    """Single-conditioning evaluator over an assembled classic-Flux DiT.

    One call is BaseModel._apply_model @ b78cec87 for the FLOW/Flux
    path: precondition the input (identity for flow), lift the scalar
    sigma to the [batch] timestep tensor (ModelSamplingFlux.timestep
    is the identity), cast input and conditioning to the model's
    compute dtype (timesteps stay float32 like the reference's
    ``t.float()``), run the transformer, lift the output back to
    float32, and convert to the denoised prediction. The shared
    guidance layer may request one compatible multi-conditioning
    batch, preserving calc_cond_batch's model-call optimization.

    Classic conditioning comes from compose_flux_conditioning: the T5
    sequence rides in ``embeddings`` (the transformer's context) and the
    CLIP-L pooled vector in ``pooled`` (the modulation input y). The
    vector-free Ovis variant takes Qwen sequence conditioning with no pooled
    vector. ``guidance`` is the distilled-guidance embedding input, NOT
    external CFG: it defaults to the reference's 3.5 exactly when the model is
    guidance-embedded (dev) and must be None for models without the embedder
    (schnell). Model placement is the caller's business (residency seams);
    conditioning is moved to the input's device per call.
    """

    def __init__(
        self,
        model: Flux,
        conditioning: Conditioning[torch.Tensor] | None = None,
        *,
        guidance: FluxGuidance = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        image_grid_indices: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> None:
        distilled = model.guidance_in is not None
        if guidance == FLUX_GUIDANCE_DISABLED:
            guidance = None
        elif distilled and guidance is None:
            guidance = FLUX_GUIDANCE_DEFAULT
        if not distilled and guidance is not None:
            raise DenoiseError(
                "guidance was given but this Flux model has no guidance"
                " embedder (schnell); pass guidance=None"
            )
        self.model = model
        self.guidance = guidance
        self.compute_dtype = compute_dtype
        if image_grid_indices is not None and (
            type(image_grid_indices) is not tuple
            or len(image_grid_indices) != 2
            or any(
                type(indices) is not tuple
                or not indices
                or any(type(index) is not int or index < 0 for index in indices)
                for indices in image_grid_indices
            )
        ):
            raise DenoiseError(
                "image_grid_indices must contain declared non-negative height and width tuples"
            )
        self.image_grid_indices = image_grid_indices
        self._conditioning = (
            None if conditioning is None else self.prepare_conditioning(conditioning)
        )

    def prepare_conditioning(self, conditioning: object) -> FluxCondition:
        if not isinstance(conditioning, Conditioning):
            raise DenoiseError("conditioning must be a Conditioning value")
        return _checked_flux_cond(conditioning, "conditioning", self.model.config.vec_in_dim)

    @staticmethod
    def batchable(conditions: tuple[FluxCondition, ...]) -> bool:
        if not conditions:
            return False
        _, _, references = _flux_condition_parts(conditions[0])
        reference_shapes = tuple(reference.shape[1:] for reference in references)
        return cross_attn_repeat(
            [condition[0].shape[1] for condition in map(_flux_condition_parts, conditions)]
        ) is not None and all(
            tuple(reference.shape[1:] for reference in candidate_references) == reference_shapes
            for _, _, candidate_references in map(_flux_condition_parts, conditions[1:])
        )

    def _forward(
        self,
        xc: torch.Tensor,
        sigma: float,
        conds: Sequence[FluxCondition],
        repeats: list[int],
    ) -> torch.Tensor:
        """One transformer call over the engine-stacked conditioning lanes."""
        batch = xc.shape[0] // len(conds)
        total = xc.shape[0]
        device = xc.device
        timesteps = torch.full((total,), sigma, device=device, dtype=torch.float32)
        contexts = []
        for index, (context, _, _) in enumerate(map(_flux_condition_parts, conds)):
            context = to_batch(context, batch)
            context = repeat_cross_attn(context, repeats[index])
            contexts.append(context.to(device=device, dtype=self.compute_dtype))
        context = torch.cat(contexts, dim=0)
        y = None
        if self.model.vector_in is not None:
            pooled_values = []
            for _, pooled, _ in map(_flux_condition_parts, conds):
                assert pooled is not None
                pooled_values.append(
                    to_batch(pooled, batch).to(device=device, dtype=self.compute_dtype)
                )
            y = torch.cat(pooled_values, dim=0)
        reference_groups = tuple(_flux_condition_parts(condition)[2] for condition in conds)
        references = tuple(
            torch.cat(
                tuple(
                    to_batch(group[index], batch).to(device=device, dtype=self.compute_dtype)
                    for group in reference_groups
                ),
                dim=0,
            )
            for index in range(len(reference_groups[0]))
        )
        guidance = None
        if self.guidance is not None:
            guidance = torch.full((total,), self.guidance, device=device, dtype=self.compute_dtype)
        if self.image_grid_indices is None:
            if not references:
                return self.model(xc, timesteps, context, y, guidance).float()
            return self.model(
                xc,
                timesteps,
                context,
                y,
                guidance,
                ref_latents=references,
            ).float()
        if references:
            raise DenoiseError("Flux reference latents cannot be combined with context windows")
        height_indices, width_indices = self.image_grid_indices
        image_position_ids = flux_window_position_ids(
            height_indices=height_indices,
            width_indices=width_indices,
            batch=total,
            device=device,
        )
        return self.model(
            xc,
            timesteps,
            context,
            y,
            guidance,
            image_position_ids=image_position_ids,
        ).float()

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if self._conditioning is None:
            raise DenoiseError("no conditioning is bound to this evaluator")
        return self.evaluate_conditioning(x, sigma, self._conditioning)

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: FluxCondition
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[FluxCondition, ...],
    ) -> None:
        if not conditions or not self.batchable(conditions):
            raise DenoiseError("Flux conditioning batch is empty or incompatible")

    def _conditioning_model_input(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return calculate_input(Parameterization.FLOW, sigma, x).to(self.compute_dtype)

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[FluxCondition],
    ) -> torch.Tensor:
        repeats = cross_attn_repeat(
            [condition[0].shape[1] for condition in map(_flux_condition_parts, batch.conditions)]
        )
        assert repeats is not None
        return self._forward(
            batch.model_input,
            batch.sigma,
            batch.conditions,
            repeats,
        )

    def _grouped_conditioning_forward(
        self,
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
    ) -> torch.Tensor:
        """One dormant B2.3a forward over already-grouped conditioning."""

        return self.evaluate_conditioning(x, sigma, self.prepare_conditioning(conditioning))


def prepare_noise_from_generator(
    latent: torch.Tensor,
    generator: torch.Generator,
    noise_inds: Sequence[int] | None = None,
) -> torch.Tensor:
    """Draw one CPU noise tensor while advancing a caller-owned generator."""

    if noise_inds is None:
        noise = torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        )
        return noise.to(dtype=latent.dtype)
    indices = tuple(noise_inds)
    if len(indices) != latent.shape[0] or any(
        type(index) is not int or index < 0 for index in indices
    ):
        raise ValueError("noise indices must be nonnegative integers matching the latent batch")
    selected = set(indices)
    generated: dict[int, torch.Tensor] = {}
    for index in range(max(indices) + 1):
        noise = torch.randn(
            (1, *latent.shape[1:]),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(dtype=latent.dtype)
        if index in selected:
            generated[index] = noise
    return torch.cat(tuple(generated[index] for index in indices), dim=0)


def prepare_noise(
    latent: torch.Tensor, seed: int, noise_inds: Sequence[int] | None = None
) -> torch.Tensor:
    """The reference's float32 CPU draw, cast to the latent dtype.

    ``noise_inds`` reproduces ComfyUI's skipped and shared batch draws.
    A private generator preserves the draw sequence without changing
    global RNG state.
    """
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    return prepare_noise_from_generator(latent, generator, noise_inds)


def prepare_multistream_noise(
    latent: MultiStreamLatent[torch.Tensor],
    seed: int,
    noise_inds: Sequence[int] | None = None,
) -> MultiStreamLatent[torch.Tensor]:
    """One CPU generator seeded once draws each stream in role order,
    reproducing the reference's nested-latent noise (comfy/sample.py
    prepare_noise @ b78cec87), exactly as the multistream family
    runtimes draw internally: against float32 views of the streams,
    so low-precision latents never round the draw."""
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    return latent.map(
        lambda stream: prepare_noise_from_generator(
            stream.to(dtype=torch.float32), generator, noise_inds
        )
    )


class GaussianNoise:
    """``NoiseSampler`` drawing independent unit gaussians shaped like
    the latent (k_diffusion default_noise_sampler @ b78cec87): one
    device-local generator seeded at construction, a fresh draw per
    step. On CPU the seed is bumped by one exactly like the reference,
    so the step-noise stream never replays prepare_noise's CPU draw
    for the same seed."""

    def __init__(self, like: torch.Tensor, *, seed: int | None = None) -> None:
        self._shape = tuple(like.shape)
        self._dtype = like.dtype
        self._layout = like.layout
        self._device = like.device
        self._generator: torch.Generator | None = None
        if seed is not None:
            if like.device == torch.device("cpu"):
                seed += 1
            self._generator = torch.Generator(device=like.device)
            self._generator.manual_seed(seed)

    def __call__(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        return torch.randn(
            self._shape,
            dtype=self._dtype,
            layout=self._layout,
            device=self._device,
            generator=self._generator,
        )


class _StandardizedGaussianStream:
    """One RES4LYF noise stream: a seeded device-local generator whose
    every draw is a fresh float64 unit gaussian standardized globally,
    ``(n - n.mean()) / n.std()`` (GaussianNoiseGenerator,
    beta/noise_classes.py @ 26036f64)."""

    def __init__(self, like: torch.Tensor, *, seed: int) -> None:
        self._shape = tuple(like.shape)
        self._layout = like.layout
        self._device = like.device
        self._generator = torch.Generator(device=like.device)
        self._generator.manual_seed(seed)

    def __call__(self) -> torch.Tensor:
        draw = torch.randn(
            self._shape,
            dtype=torch.float64,
            layout=self._layout,
            device=self._device,
            generator=self._generator,
        )
        return (draw - draw.mean()) / draw.std()


class RES4LYFTwoStreamNoise:
    """The RES4LYF RK engine's two seeded gaussian noise streams
    (rk_noise_sampler_beta.py @ 26036f64): ``outer`` feeds step-level
    noise swaps, ``substep`` feeds substep swaps. Seeding follows the
    reference's own seeded path - beta/samplers.py rewrites noise_seed
    -1 to the workflow seed + 1, and rk_sampler_beta.py derives the
    substep seed as noise_seed + MAX_STEPS (10000) - so for run seed
    ``s`` the outer stream is seeded ``s + 1`` and the substep stream
    ``s + 10001``. Unlike :class:`GaussianNoise` there is no CPU seed
    bump: the reference RK generators seed exactly as given. The
    reference's UNSEEDED direct-wrapper default
    (``torch.initial_seed() + 1``, process-global state) is
    unreproducible by construction and out of parity scope.

    Each draw returns the reference swap noise: the stream's
    standardized float64 draw z-scored per channel over the spatial
    dims (normalize_zscore(channelwise=True, inplace=True)
    @ 26036f64, whose divisor is the std of the already-centered
    tensor), cast to float32 for the float32 solver state.

    The reference applies scale_av_noise (rk_noise_sampler_beta.py:165-184
    @ 26036f64) between the z-score and the sigma_up multiply; this port
    omits it because it is inert at the pins. The audio-column rescale
    mutates bytes only when its ratio != 1.0, which requires either the
    extra_options string "av_audio_noise_scale" (unreachable through the
    catalog sampler names - the named beta wrappers pass no extra_options,
    so the "" default applies) or av_shift_audio, which is set only for a
    diffusion model exposing sigma_shift_audio AND
    model_sampling.audio_scale == 1.0 (rk_noise_sampler_beta.py:136-153).
    LTX-2 lacks the attribute entirely and MiniMax H3 has audio_scale
    12.0/3.0 = 4.0 (supported_models.py @ b78cec87), so every packed
    audio-video invocation reaches the ``if ratio == 1.0: return noise``
    early return byte-exactly. Re-verify (and port with executed AV
    goldens if live) before admitting a family that has sigma_shift_audio
    with default audio_scale, porting RES4LYF extra_options surfaces, or
    moving either pin.

    Satisfies the torch-free RKNoiseSampler protocol. The plain
    NoiseSampler ``__call__`` refuses loudly: the RK engine never
    draws single-stream step noise.
    """

    def __init__(self, like: torch.Tensor, *, seed: int) -> None:
        self.outer: Callable[[], torch.Tensor] = _StandardizedGaussianStream(like, seed=seed + 1)
        self.substep: Callable[[], torch.Tensor] = _StandardizedGaussianStream(
            like, seed=seed + 10001
        )

    @staticmethod
    def _swap_noise(draw: torch.Tensor) -> torch.Tensor:
        centered = draw - draw.mean(dim=(-2, -1), keepdim=True)
        return (centered / centered.std(dim=(-2, -1), keepdim=True)).to(torch.float32)

    def step_noise(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        return self._swap_noise(self.outer())

    def substep_noise(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        return self._swap_noise(self.substep())

    def __call__(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        raise TypeError(
            "the RES4LYF two-stream noise sampler draws via"
            " step_noise/substep_noise, never as a plain NoiseSampler"
        )


def latent_process_in(latent: torch.Tensor, descriptor: LatentDescriptor) -> torch.Tensor:
    """The descriptor's affine normalization into model space
    (comfy/latent_formats.py process_in @ b78cec87):
    ``(x - shift) * scale``."""
    return (latent - descriptor.shift_factor) * descriptor.scale_factor


def latent_process_out(latent: torch.Tensor, descriptor: LatentDescriptor) -> torch.Tensor:
    """The inverse of :func:`latent_process_in`
    (comfy/latent_formats.py process_out @ b78cec87):
    ``x / scale + shift``."""
    return latent / descriptor.scale_factor + descriptor.shift_factor


class _InpaintDenoiser:
    """ComfyUI KSamplerX0Inpaint's per-evaluation sampler masking."""

    def __init__(
        self,
        inner: Denoiser[torch.Tensor],
        *,
        mask: torch.Tensor,
        latent: torch.Tensor,
        noise: torch.Tensor,
        parameterization: Parameterization,
        fixed_latent: bool = False,
    ) -> None:
        self.inner = inner
        self.mask = mask
        self.latent = latent
        self.noise = noise
        self.parameterization = parameterization
        self.fixed_latent = fixed_latent

    def _input(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        keep = 1.0 - self.mask
        source = (
            self.latent
            if self.fixed_latent
            else _noise_scaling(self.parameterization, sigma, self.noise, self.latent)
        )
        return x * self.mask + source * keep, keep

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        masked, keep = self._input(x, sigma)
        return self.inner(masked, sigma) * self.mask + self.latent * keep

    def call_with_uncond(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.inner, UncondDenoiser):
            raise DenoiseError("inpaint sampler requires an unconditional denoiser")
        masked, keep = self._input(x, sigma)
        combined, uncond = self.inner.call_with_uncond(masked, sigma)
        source = self.latent * keep
        return combined * self.mask + source, uncond * self.mask + source


def prepare_denoise_mask(
    mask: torch.Tensor | None,
    latent: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Normalize dense image, video or audio masks for the sampling engine."""
    if mask is None:
        return None
    if latent.ndim == 4:
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    mask = mask.to(device=device if device is not None else latent.device, dtype=torch.float32)
    return reshape_latent_mask(mask, tuple(latent.shape))


def run_denoise(
    denoiser: Denoiser[torch.Tensor],
    solver: SolverFn[torch.Tensor],
    *,
    latent: torch.Tensor,
    noise: torch.Tensor,
    inpaint_noise: torch.Tensor | None = None,
    sigmas: Sequence[float],
    initial_sigma: float | None = None,
    family: ModelFamily,
    sampling: SamplingDescriptor | None = None,
    process_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
    process_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
    noise_kind: NoiseKind = NoiseKind.NONE,
    noise_sampler: NoiseSampler[torch.Tensor] | None = None,
    percent_to_sigma: Callable[[float], float] | None = None,
    device: torch.device | str | None = None,
    on_step: StepCallback | None = None,
    on_step_begin: Callable[[int], None] | None = None,
    on_state: SamplingStateCallback | None = None,
    unpack_state: Callable[[torch.Tensor], object] | None = None,
    denoise_mask: torch.Tensor | None = None,
    denoise_mask_prepared: bool = False,
    fixed_inpaint_latent: bool = False,
) -> torch.Tensor:
    """Drive one full denoise: the native KSAMPLER.sample +
    CFGGuider.inner_sample pipeline (@ b78cec87).

    ``latent`` is the raw (unprocessed) latent - all-zero for
    txt2img, a VAE encode for img2img; ``noise`` comes from
    :func:`prepare_noise`. ``inpaint_noise`` optionally selects a
    distinct masked-latent source stream; when omitted it is exactly
    ``noise``. These tensors are cast to float32 on ``device``
    (default: the latent's). The latent is processed into model space
    unless it is entirely zero (the reference's "don't shift the
    empty latent image" guard), combined with noise at ``initial_sigma``
    (or ``sigmas[0]`` when omitted) per the active sampling descriptor
    (or the family's canonical descriptor when omitted), driven through the solver,
    un-scaled at the final sigma, and processed back out of model
    space. Empty ``sigmas`` returns the latent untouched (a
    denoise-0 run).

    ``noise_kind`` is the solver descriptor's declared NoiseKind and
    picks the default step-noise construction: GaussianNoise for
    gaussian, BrownianTreeNoise for brownian with the reference
    solvers' bounds (the min positive sigma and the max sigma of
    ``sigmas`` - sample_dpmpp_sde/_2m_sde/_3m_sde @ b78cec87).
    ``noise_sampler`` overrides it, porting the reference solvers'
    same-named parameter. The override matters on SNR-OFFSET FLOW
    schedules: the reference builds the brownian tree from the
    schedule BEFORE offset_first_sigma_for_snr nudges sigmas[0], so
    a caller that offsets the schedule (as
    schedules.offset_first_sigma_for_snr documents) must pass a
    BrownianTreeNoise built from the PRE-offset bounds to reproduce
    the reference noise stream; the sigmas seen here can no longer
    tell.
    """
    if len(sigmas) == 0:
        return latent
    effective_sampling = sampling or family.sampling
    prepared_mask = (
        None
        if denoise_mask is None
        else denoise_mask.to(device=device)
        if denoise_mask_prepared
        else prepare_denoise_mask(denoise_mask, latent, device=device)
    )
    latent_descriptor = None
    if process_in is None or process_out is None:
        latent_descriptor = family.single_stream_latent()
    if process_in is None:
        assert latent_descriptor is not None

        def default_process_in(value: torch.Tensor) -> torch.Tensor:
            return latent_process_in(value, latent_descriptor)

        process_in = default_process_in
    if process_out is None:
        assert latent_descriptor is not None

        def default_process_out(value: torch.Tensor) -> torch.Tensor:
            return latent_process_out(value, latent_descriptor)

        process_out = default_process_out
    if unpack_state is None:
        unpack_state = process_out
    return run_sampler_engine(
        denoiser,
        solver,
        latent=latent,
        noise=noise,
        inpaint_noise=inpaint_noise,
        sigmas=sigmas,
        initial_sigma=initial_sigma,
        parameterization=effective_sampling.parameterization,
        sigma_min=effective_sampling.sigma_min,
        sigma_max=effective_sampling.sigma_max,
        process_in=process_in,
        process_out=process_out,
        seed=seed,
        noise_kind=noise_kind,
        noise_sampler=noise_sampler,
        percent_to_sigma=percent_to_sigma,
        device=device,
        on_step=on_step,
        on_step_begin=on_step_begin,
        on_state=on_state,
        unpack_state=unpack_state,
        denoise_mask=prepared_mask,
        fixed_inpaint_latent=fixed_inpaint_latent,
    )


def run_sampler_engine(
    denoiser: Denoiser[torch.Tensor],
    solver: SolverFn[torch.Tensor],
    *,
    latent: torch.Tensor,
    noise: torch.Tensor,
    sigmas: Sequence[float],
    initial_sigma: float | None = None,
    parameterization: Parameterization,
    sigma_max: float,
    sigma_min: float | None = None,
    process_in: Callable[[torch.Tensor], torch.Tensor],
    process_out: Callable[[torch.Tensor], torch.Tensor],
    inpaint_noise: torch.Tensor | None = None,
    seed: int = 0,
    noise_kind: NoiseKind = NoiseKind.NONE,
    noise_sampler: NoiseSampler[torch.Tensor] | None = None,
    percent_to_sigma: Callable[[float], float] | None = None,
    device: torch.device | str | None = None,
    on_step: StepCallback | None = None,
    on_step_begin: Callable[[int], None] | None = None,
    on_state: SamplingStateCallback | None = None,
    unpack_state: Callable[[torch.Tensor], object] | None = None,
    denoise_mask: torch.Tensor | None = None,
    fixed_inpaint_latent: bool = False,
) -> torch.Tensor:
    """Run the shared packed-state sampling engine for any latent topology.

    ``initial_sigma`` may preserve a pre-offset initial state while the
    solver walks ``sigmas``.
    """

    if len(sigmas) == 0:
        return latent
    target = torch.device(device) if device is not None else latent.device
    from .distributed import (
        distributed_sampling_config,
        run_rank_zero_sampling,
        synchronized_sampling_call,
    )

    config = distributed_sampling_config()
    cache = solver_sampling_cache(solver)
    rank_zero_reasons = (
        *(
            ("autoregressive block sampling",)
            if isinstance(denoiser, AutoregressiveDenoiser)
            else ()
        ),
        *(("scheduled sampling timeline",) if solver_has_sampling_timeline(solver) else ()),
        *((type(cache).__name__,) if cache is not None else ()),
    )
    if config is not None and rank_zero_reasons:
        if config.rank == 0:
            for reason in rank_zero_reasons:
                _LOGGER.warning(
                    "Distributed %s executes on rank 0 and broadcasts its final result", reason
                )
        template = torch.empty_like(latent, device=target, dtype=torch.float32)
        return run_rank_zero_sampling(
            lambda: run_sampler_engine(
                denoiser,
                solver,
                latent=latent,
                noise=noise,
                sigmas=sigmas,
                initial_sigma=initial_sigma,
                parameterization=parameterization,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                process_in=process_in,
                process_out=process_out,
                inpaint_noise=inpaint_noise,
                seed=seed,
                noise_kind=noise_kind,
                noise_sampler=noise_sampler,
                percent_to_sigma=percent_to_sigma,
                device=target,
                on_step=on_step,
                on_step_begin=on_step_begin,
                on_state=on_state,
                unpack_state=unpack_state,
                denoise_mask=denoise_mask,
                fixed_inpaint_latent=fixed_inpaint_latent,
            ),
            template,
            config,
        )

    if config is not None and torch.distributed.is_initialized():
        callbacks = torch.tensor(
            (on_step is not None, on_state is not None, on_step_begin is not None),
            dtype=torch.int32,
            device=target,
        )
        torch.distributed.all_reduce(callbacks, op=torch.distributed.ReduceOp.MAX)
        step_callback, state_callback, begin_callback = on_step, on_state, on_step_begin

        def distributed_step(event: StepEvent) -> None:
            synchronized_sampling_call(
                lambda: None if step_callback is None else step_callback(event),
                target,
                "step callback",
            )

        def distributed_state(event: SamplingStateEvent[object]) -> None:
            synchronized_sampling_call(
                lambda: None if state_callback is None else state_callback(event),
                target,
                "state callback",
            )

        def distributed_begin(index: int) -> None:
            synchronized_sampling_call(
                lambda: None if begin_callback is None else begin_callback(index),
                target,
                "step-begin callback",
            )

        on_step = distributed_step if callbacks[0].item() else None
        on_state = distributed_state if callbacks[1].item() else None
        on_step_begin = distributed_begin if callbacks[2].item() else None
    latent32 = latent.to(device=target, dtype=torch.float32)
    noise32 = noise.to(device=target, dtype=torch.float32)
    if noise32.shape != latent32.shape:
        raise DenoiseError(
            f"noise shape {tuple(noise32.shape)} does not match latent"
            f" shape {tuple(latent32.shape)}"
        )
    latent_in = process_in(latent32) if torch.count_nonzero(latent32) > 0 else latent32
    start_sigma = float(sigmas[0]) if initial_sigma is None else float(initial_sigma)
    x = _noise_scaling(
        parameterization,
        start_sigma,
        noise32,
        latent_in,
        max_denoise=max_denoise(sigma_max, (start_sigma,)),
    )
    if denoise_mask is not None:
        mask = denoise_mask.to(device=target, dtype=torch.float32)
        if mask.shape != latent32.shape:
            raise DenoiseError(
                f"prepared denoise mask shape {tuple(mask.shape)} does not match latent"
                f" shape {tuple(latent32.shape)}"
            )
        selected_inpaint_noise = noise32
        if inpaint_noise is not None:
            selected_inpaint_noise = inpaint_noise.to(device=target, dtype=torch.float32)
            if selected_inpaint_noise.shape != latent32.shape:
                raise DenoiseError(
                    "inpaint noise shape"
                    f" {tuple(selected_inpaint_noise.shape)} does not match"
                    f" latent shape {tuple(latent32.shape)}"
                )
        denoiser = _InpaintDenoiser(
            denoiser,
            mask=mask,
            latent=latent_in,
            noise=selected_inpaint_noise,
            parameterization=parameterization,
            fixed_latent=fixed_inpaint_latent,
        )
    del latent32, latent_in, noise32

    def report_state(event: SolverStateEvent[object]) -> None:
        if on_state is None:
            return
        if type(event.current) is not torch.Tensor:
            raise TypeError("torch sampler state must contain an exact torch.Tensor")
        current = event.current.detach().clone()
        denoised = event.denoised
        if denoised is not None:
            if type(denoised) is not torch.Tensor:
                raise TypeError("torch denoised state must contain an exact torch.Tensor")
            denoised = denoised.detach().clone()
        if unpack_state is None:
            current_state: object = current
            denoised_state: object | None = denoised
        else:
            current_state = unpack_state(current)
            denoised_state = None if denoised is None else unpack_state(denoised)
        on_state(
            SamplingStateEvent[object](
                step=event.step,
                total=event.total,
                sigma=event.sigma,
                phase=event.phase,
                current=current_state,
                denoised=denoised_state,
            )
        )

    info = SamplerInfo(
        parameterization=parameterization,
        seed=seed,
        percent_to_sigma=percent_to_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        on_state=report_state if on_state is not None else None,
    )
    if noise_sampler is None:
        if noise_kind is NoiseKind.GAUSSIAN:
            noise_sampler = GaussianNoise(x, seed=seed)
        elif noise_kind is NoiseKind.RES4LYF_GAUSSIAN:
            noise_sampler = RES4LYFTwoStreamNoise(x, seed=seed)
        elif noise_kind in (NoiseKind.BROWNIAN, NoiseKind.BROWNIAN_GPU) and len(sigmas) > 1:
            # The reference SDE solvers' bounds:
            # sigmas[sigmas > 0].min(), sigmas.max() (@ b78cec87).
            # A single-entry schedule skips construction exactly like
            # the reference solvers' len(sigmas) <= 1 early return
            # (zero steps: the solver never draws noise). Equal
            # bounds - a one-step (sigma, 0) schedule - are legal and
            # build the reference's zero-width, never-queried tree.
            positive = [s for s in sigmas if s > 0]
            if not positive:
                raise DenoiseError(
                    "brownian-tree noise needs a positive sigma in the"
                    f" schedule, got {tuple(sigmas)}"
                )
            noise_sampler = BrownianTreeNoise(
                x,
                min(positive),
                max(sigmas),
                seed=seed,
                cpu=noise_kind is NoiseKind.BROWNIAN,
            )
    if on_step_begin is None:
        x = solver(
            denoiser,
            x,
            sigmas,
            info,
            noise=noise_sampler,
            on_step=on_step,
        )
    else:
        if not solver_supports_step_begin(solver):
            raise DenoiseError(
                "scheduled step-begin control requires a solver that declares the internal"
                " on_step_begin capability"
            )
        x = run_step_begin_solver(
            solver,
            denoiser,
            x,
            sigmas,
            info,
            noise=noise_sampler,
            on_step=on_step,
            on_step_begin=on_step_begin,
        )
    x = inverse_noise_scaling(parameterization, float(sigmas[-1]), x)
    return process_out(x.to(torch.float32))


__all__ = [
    "FLUX_GUIDANCE_DEFAULT",
    "FLUX_GUIDANCE_DISABLED",
    "DenoiseError",
    "FluxDenoiser",
    "GaussianNoise",
    "latent_process_in",
    "latent_process_out",
    "prepare_noise",
    "run_denoise",
    "run_sampler_engine",
]
