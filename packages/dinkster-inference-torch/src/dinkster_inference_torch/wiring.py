"""Family plugin wiring: the torch realization of the runtime seam.

dinkster_inference.runtime declares the seam (probe_native +
FamilyRuntime); this module makes it executable. :func:`load_runtime`
is the reference's load_checkpoint_guess_config (comfy/sd.py
@ b78cec87) with the guessing replaced by the probe: detect, plan,
assemble, and hand back one typed :class:`FluxRuntime` instead of a
loose (ModelPatcher, CLIP, VAE) triple.

FluxRuntime composes only already-pinned pieces - the tokenizers,
ClipTextEncoder/T5TextEncoder, FluxDenoiser, run_denoise, and the KL
codec plugin - so its job is the reference's glue, not new math: the
FluxClipModel dual encode (comfy/text_encoders/flux.py), the KSampler
schedule/noise/drive pipeline (comfy/samplers.py), and the VAE
facade. The one ordering subtlety it owns: the reference SDE solvers
build their brownian tree from the schedule BEFORE
offset_first_sigma_for_snr nudges sigmas[0] on flow models
(comfy/k_diffusion/sampling.py @ b78cec87), so :meth:`FluxRuntime.sample_custom`
constructs the tree from pre-offset bounds and passes it explicitly.

Component geometry selects a typed assembly plan; family labels do not
control whether its registered implementation can execute.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import torch
from dinkster_inference import (
    BFLOAT16,
    CLIP_G_PROFILE,
    FLOAT16,
    FLOAT32,
    SD15_CONTROL_RESIDUAL_SITES,
    SDXL_CONTROL_RESIDUAL_SITES,
    AttentionPolicy,
    AttentionRouteToken,
    CompositeWindowPlan,
    Conditioning,
    ConditioningCarrier,
    ContinuousEDMSigmas,
    ContributionGain,
    DirectGainTableCurve,
    DiscreteSigmas,
    DType,
    FlowSigmas,
    FluxFlowSigmas,
    GuidanceRole,
    InferenceTypeRegistry,
    InpaintConditioning,
    ModelFamily,
    PromptTokenizer,
    RealizedGainRow,
    Registrable,
    Registry,
    SamplerDescriptor,
    SamplingGuidance,
    SamplingSegment,
    SamplingSpace,
    SamplingStateCallback,
    ScheduledEncodeRequest,
    ScheduledExecution,
    SchedulerDescriptor,
    SigmaSpace,
    StepCallback,
    Wan21AssemblyPlan,
    Wan21PoseBlockCacheSettings,
    WeightSource,
    build_runtime_identity,
    contribution_gain_slot_facts,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    executed_sampling_timeline,
    is_flow_parameterization,
    load_clip_bpe,
    load_t5_spm,
    prove_manifest_consensus,
    realize_gain_table,
    realize_sampling_timeline,
    require_realized_sampling_step,
)
from dinkster_inference.assembly import (
    Flux2AssemblyPlan,
    FluxAssemblyPlan,
    Lumina2AssemblyPlan,
    QwenImageAssemblyPlan,
    SDAssemblyPlan,
    ZImageAssemblyPlan,
)
from dinkster_inference.refusal import NativeRefusalError
from dinkster_inference.runtime import (
    AssemblyRegistration,
    FamilyRuntime,
    NativeAssemblyPlan,
    resolve_native_assembly,
    wired_runtime_family_ids,
)

from ._conditioning_layout import (
    bind_flux_layout,
    bind_sd_layout,
    conditioning_layout,
    conditioning_token_transforms,
    declared_token_count,
    flux_fused_layout,
    validate_flux_layout,
    validate_sd_layout,
)
from .assemble import (
    AssembledFlux,
    AssembledSD,
    assemble_flux,
    assemble_flux2,
    assemble_lumina2,
    assemble_qwen_image,
    assemble_sd,
    assemble_wan21,
    assemble_z_image,
)
from .autoencoder_kl import kl_codec_plugin
from .clip_text import (
    SD1_CLIP_L_POLICY,
    SDXL_CLIP_POLICY,
    ClipTextEncoder,
    EmbeddingLookup,
    compose_sdxl_conditioning,
)
from .codecs import CodecPlugin
from .conditioning_adapters import materialize_basic_conditioning
from .controlnet import (
    SDControlConditioning,
    SDXLControlLoRA,
    SDXLControlNet,
    SDXLControlNetUnion,
    _snapshot_sd_control_conditioning,  # pyright: ignore[reportPrivateUsage]
    compile_sd_effect_mask,
)
from .denoise import (
    DenoiseError,
    FluxCondition,
    FluxDenoiser,
    latent_process_in,
    prepare_noise,
)
from .distributed import distributed_sampling_config, ensure_process_group
from .flux2_runtime import Flux2Runtime
from .flux_window import (
    FluxWindowConditioningEvaluation,
    FluxWindowError,
    prepare_flux_window_plan,
)
from .flux_window_distributed import (
    DistributedFluxWindowEvaluation,
    WindowDigestConsensusTransport,
    build_flux_window_manifest,
    window_preflight_failed,
    window_route_mismatch,
)
from .guidance import ConditioningEvaluation, GuidanceExecutor
from .ipadapter import (
    SD15IPAdapterConditioning,
    SD15IPAdapterExecution,
    sd15_ipadapter_identity_facts,
)
from .lumina2_runtime import Lumina2Runtime
from .operations import module_compute_device
from .qwen_image_runtime import QwenImageRuntime
from .qwen_text import OvisTextEncoder
from .regional import flux_grouped_region_evaluator
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionInputs,
    SamplingExecutionRegistration,
    SingleStreamLatentAdapter,
    run_ksampler_as_custom,
    sampling_execution,
)
from .sampling_runtime import SingleStreamSamplingRuntime
from .scheduled import encode_text_scheduled
from .schedules import (
    continuous_edm_percent_to_sigma,
    discrete_percent_to_sigma,
    simple_schedule,
    torch_scheduler_registry,
)
from .sd_denoise import (
    CROSS_ATTN_REPEAT_LIMIT,
    SDXL_AESTHETIC_DEFAULT,
    SDXL_NEGATIVE_AESTHETIC_DEFAULT,
    SDControlGain,
    SDDenoiser,
    cross_attn_repeat,
    encode_sdxl_adm,
    encode_sdxl_refiner_adm,
)
from .solvers import torch_sampler_registry
from .t5_text import T5TextEncoder, compose_flux_conditioning
from .taesd import TAESD, taesd_codec_plugin
from .wan21_runtime import Wan21Runtime
from .z_image_runtime import ZImageRuntime

if TYPE_CHECKING:
    from .scheduled_sampling import ScheduledPatchResolver


class WiringError(ValueError):
    """A runtime cannot be built or driven as asked: the probe refused
    the sources, or a registry id resolves to nothing."""


_DescriptorT = TypeVar("_DescriptorT", bound=Registrable)


def _exact_registry(source: Registry[_DescriptorT]) -> Registry[_DescriptorT]:
    """Copy descriptors into a plain ``Registry`` so a caller-supplied
    ``Registry`` subclass (or an instance with shadowed methods) never
    executes its own code during sampling lookups."""
    registry: Registry[_DescriptorT] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


# Canonical identity construction lives in dinkster_inference.identity so
# dispatch hosts do not import torch.
_TORCH_TO_DTYPE = {
    torch.float32: FLOAT32,
    torch.float16: FLOAT16,
    torch.bfloat16: BFLOAT16,
}
_DTYPE_TO_TORCH = {value: key for key, value in _TORCH_TO_DTYPE.items()}


def _torch_dtype(dtype: DType) -> torch.dtype:
    try:
        return _DTYPE_TO_TORCH[dtype]
    except KeyError as error:
        raise WiringError(f"no torch dtype mapping for {dtype.name}") from error


def _identity_dtype(dtype: torch.dtype) -> DType:
    try:
        return _TORCH_TO_DTYPE[dtype]
    except KeyError as error:
        raise WiringError(f"no identity dtype mapping for {dtype}") from error


def _flux_sigma_space(family: ModelFamily) -> SigmaSpace:
    """The family's reference model-sampling space @ b78cec87: dev is
    ModelType.FLUX -> ModelSamplingFlux (exponential flux time shift,
    10000-entry table); schnell is ModelType.FLOW ->
    ModelSamplingDiscreteFlow with multiplier 1.0 over the default
    1000-entry table (supported_models.FluxSchnell). At shift 1.0 the
    two curves coincide, but the table length does not - the discrete
    schedulers (simple, ddim_uniform, beta) index the table directly,
    and sigma_min differs (1e-3 vs ~3.16e-4), so conflating the
    spaces moves every schnell schedule."""
    if family.engine.sigma_space == "default":
        return FlowSigmas(shift=family.sampling.shift, multiplier=1.0, timesteps=1000)
    return FluxFlowSigmas(shift=family.sampling.shift)


class _FluxSamplingDenoiser:
    evaluator_identity = "dinkster.flux.conditioning.v1"

    def __init__(
        self,
        evaluation: FluxDenoiser | FluxWindowConditioningEvaluation[FluxCondition],
    ) -> None:
        self.evaluation = evaluation

    def prepare_conditioning(self, value: object, _role: GuidanceRole) -> object:
        return self.evaluation.prepare_conditioning(value)

    def evaluate_conditioning(
        self, latent: torch.Tensor, sigma: float, condition: object
    ) -> torch.Tensor:
        return self.evaluation.evaluate_conditioning(latent, sigma, cast("Any", condition))

    def evaluate_conditioning_batch(
        self,
        latent: torch.Tensor,
        sigma: float,
        conditions: tuple[object, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self.evaluation.evaluate_conditioning_batch(latent, sigma, cast("Any", conditions))

    def batchable(self, conditions: tuple[object, ...]) -> bool:
        return self.evaluation.batchable(cast("Any", conditions))


def _validate_flux_latent(latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise WiringError("Flux latent must have shape [batch,channels,height,width]")


class _FluxLatentAdapter(SingleStreamLatentAdapter):
    def prepare(
        self,
        runtime: object,
        family: ModelFamily,
        *,
        latent: CustomSamplingLatentValue,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        denoise_mask: CustomSamplingLatentValue | None,
        context: SamplingAdapterContext,
        error: type[Exception],
    ) -> SamplingExecutionInputs:
        scheduled_request = (
            type(cond) is ConditioningCarrier or context.options.get("scheduled") is not None
        )
        if not scheduled_request:
            return super().prepare(
                runtime,
                family,
                latent=latent,
                noise=noise,
                cond=cond,
                cfg=cfg,
                denoise_mask=denoise_mask,
                context=context,
                error=error,
            )
        from .scheduled_sampling import ScheduledSamplingError, narrow_scheduled_values

        owner = cast("FluxRuntime", runtime)
        executor = owner.sampling_execution_registration.guidance_executor(owner)
        if executor is not None and executor.registry.active:
            raise ScheduledSamplingError("guidance-extensions")
        if cfg is not None and cfg.transforms:
            raise ScheduledSamplingError("guidance-extensions")
        if context.guidance is not None and owner.assembled.diffusion.guidance_in is None:
            raise DenoiseError(
                "guidance was given but this Flux model has no guidance"
                " embedder (schnell); pass guidance=None"
            )

        latent, noise, cond, cfg, denoise_mask = narrow_scheduled_values(
            family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
        )
        self.validate(latent)
        return SamplingExecutionInputs(
            latent,
            noise,
            cond,
            cfg,
            denoise_mask,
            family.single_stream_latent(),
        )


def _flux_scheduled_denoiser(
    owner: FluxRuntime,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    from .scheduled_sampling import (
        ScheduledConditioningDenoiser,
        ScheduledSamplingOptions,
        prepare_scheduled_carriers,
        validate_unconditional_carrier,
    )

    if (
        context.inputs is None
        or context.device is None
        or context.plan is None
        or context.request is None
        or context.schedule is None
    ):
        raise RuntimeError("scheduled Flux sampling context is unresolved")
    scheduled = context.options.get("scheduled")
    if scheduled is None:
        scheduled = ScheduledSamplingOptions()
    elif type(scheduled) is not ScheduledSamplingOptions:
        raise TypeError("scheduled must be an exact ScheduledSamplingOptions or None")
    unknown = set(context.options) - {"scheduled", "window_plan"}
    if unknown:
        raise WiringError(
            "Flux sampling does not accept adapter options: " + ", ".join(sorted(unknown))
        )
    executor = owner.sampling_execution_registration.guidance_executor(owner)
    if executor is not None and executor.registry.active:
        raise RuntimeError("scheduled guidance was admitted after validation")
    if context.inputs.cfg is not None and context.inputs.cfg.transforms:
        raise RuntimeError("scheduled guidance transforms were admitted after validation")
    if context.inputs.denoise_mask is not None:
        raise WiringError(f"scheduled-sampling:denoise-mask: {owner.family.id}")
    if context.inpaint is not None:
        raise WiringError(f"scheduled-sampling:inpaint: {owner.family.id}")
    if context.context_windows is not None:
        raise WiringError(f"scheduled-sampling:context-windows: {owner.family.id}")
    if context.options.get("window_plan") is not None:
        raise WiringError(f"scheduled-sampling:window-plan: {owner.family.id}")
    if context.guidance == "disabled":
        raise WiringError(f"scheduled-sampling:distilled-guidance: {owner.family.id}")
    validate_unconditional_carrier(context.plan)
    device = torch.device(context.device)
    realized_timeline = (
        None
        if context.request.timeline is None
        else realize_sampling_timeline(
            context.request.timeline,
            tuple(float(sigma) for sigma in context.schedule.sigmas),
        )
    )
    space = owner.sampling_sigma_space()
    conditional, unconditional, patch_sets, materialized_plan = prepare_scheduled_carriers(
        owner,
        context.inputs.latent,
        context.plan,
        resolver=scheduled.resolver,
        device=device,
        cancel=context.cancelled,
        timeline=realized_timeline,
        space=space,
    )
    evaluator = ScheduledConditioningDenoiser(
        conditional,
        unconditional,
        family_id=owner.family.id,
        space=space,
        model=owner.assembled.diffusion,
        evaluate=flux_grouped_region_evaluator(
            owner.assembled.diffusion,
            guidance=context.guidance,
            compute_dtype=compute_dtype,
        ),
        patch_sets=patch_sets,
        compute_dtype=compute_dtype,
        device=device,
        cancel=context.cancelled,
    )
    replacements = MappingProxyType(
        {
            condition.id: condition.conditioning
            for condition in materialized_plan.conditions
            if condition.conditioning is not None
        }
    )
    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        conditioning_evaluation=ConditioningEvaluation(
            evaluator.prepare_conditioning,
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: f"{owner.family.id}.scheduled-conditioning.v1",
            standard_activation_memory_factor=owner.family.memory_factor,
        ),
        conditioning_payloads=replacements,
        solver_options=MappingProxyType({"realized_timeline": realized_timeline}),
        defer_callback_cancellation=True,
        close=evaluator.close,
    )


def _flux_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("FluxRuntime", runtime)
    if (
        context.inputs is None
        or context.device is None
        or context.plan is None
        or context.request is None
        or context.sampler is None
        or context.schedule is None
    ):
        raise RuntimeError("Flux sampling context is unresolved")
    if type(context.inputs.cond) is ConditioningCarrier:
        return _flux_scheduled_denoiser(owner, compute_dtype, context)
    execution_device = torch.device(context.device)
    unknown = set(context.options) - {"window_plan", "scheduled"}
    if unknown:
        raise WiringError(
            "Flux sampling does not accept adapter options: " + ", ".join(sorted(unknown))
        )
    if context.options.get("scheduled") is not None:
        raise WiringError("scheduled Flux conditioning requires a ConditioningCarrier")
    window_plan = context.options.get("window_plan")
    latent = context.inputs.latent
    prepared_window_plan = None
    window_plan_error: BaseException | None = None
    try:
        if window_plan is not None and type(window_plan) is not CompositeWindowPlan:
            raise WiringError("invalid-window-plan: expected an exact CompositeWindowPlan")
        if window_plan is not None:
            prepared_window_plan = prepare_flux_window_plan(
                window_plan,
                latent_height=latent.shape[-2],
                latent_width=latent.shape[-1],
                patch_size=owner.assembled.diffusion.config.patch_size,
            )
    except FluxWindowError as error:
        window_plan_error = WiringError(str(error))
    except BaseException as error:
        window_plan_error = error
    requested_distributed = distributed_sampling_config()
    window_mode_requested = requested_distributed is not None and requested_distributed.mode in (
        "auto",
        "window",
    )
    window_distributed = (
        window_mode_requested
        and prepared_window_plan is not None
        and len(prepared_window_plan.windows) >= 2
    )
    distributed_config = None
    if not window_mode_requested and window_plan_error is not None:
        raise window_plan_error
    if window_mode_requested:
        distributed_config = ensure_process_group()
        assert distributed_config is not None
        if window_preflight_failed(window_plan_error is not None, execution_device):
            if window_plan_error is not None:
                raise window_plan_error from None
            raise WiringError("peer flux window preflight failed")
        if window_route_mismatch(window_distributed, execution_device):
            raise WiringError("distributed ranks disagree on Flux window route eligibility")
    patch_size = owner.assembled.diffusion.config.patch_size
    replacements: dict[str, object] = {}
    bound_conditions = []
    for condition in context.plan.conditions:
        if condition.conditioning is None:
            bound_conditions.append(condition)
            continue
        bound = bind_flux_layout(
            condition.conditioning,
            latent_height=latent.shape[-2],
            latent_width=latent.shape[-1],
            patch_size=patch_size,
        )
        replacements[condition.id] = bound
        bound_conditions.append(replace(condition, conditioning=cast("Any", bound)))
    bound_plan = replace(context.plan, conditions=tuple(bound_conditions))
    conditioning_evaluation: ConditioningEvaluation[Any]
    if prepared_window_plan is None:
        evaluator: FluxDenoiser | FluxWindowConditioningEvaluation[FluxCondition] = FluxDenoiser(
            owner.assembled.diffusion,
            guidance=context.guidance,
            compute_dtype=compute_dtype,
        )
        conditioning_evaluation = ConditioningEvaluation(
            lambda value, _role: evaluator.prepare_conditioning(value),
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: "dinkster.flux.conditioning.v1",
            standard_activation_memory_factor=owner.family.memory_factor,
            layout=conditioning_layout,
            fused_layout=flux_fused_layout,
            token_transforms=conditioning_token_transforms,
            validate_layout=lambda condition, layout: validate_flux_layout(
                condition,
                layout,
                latent_height=latent.shape[-2],
                latent_width=latent.shape[-1],
                patch_size=patch_size,
            ),
        )
    else:
        window_evaluation = FluxWindowConditioningEvaluation(
            prepared_window_plan,
            tuple(
                FluxDenoiser(
                    owner.assembled.diffusion,
                    guidance=context.guidance,
                    compute_dtype=compute_dtype,
                    image_grid_indices=(window.height_indices, window.width_indices),
                )
                for window in prepared_window_plan.windows
            ),
        )
        evaluation: (
            FluxWindowConditioningEvaluation[FluxCondition]
            | DistributedFluxWindowEvaluation[FluxCondition]
        ) = window_evaluation
        if window_distributed:
            assert distributed_config is not None
            manifest_error: BaseException | None = None
            manifest = None
            try:
                lanes = tuple(
                    condition.conditioning
                    for condition in bound_plan.conditions
                    if condition.conditioning is not None
                )
                token_counts = tuple(declared_token_count(lane) for lane in lanes)
                if not token_counts or any(count is None for count in token_counts):
                    raise WiringError(
                        "windowed distributed execution requires declared conditioning"
                    )
                manifest = build_flux_window_manifest(
                    runtime_identity=owner.runtime_identity,
                    config=distributed_config,
                    prepared_plan=prepared_window_plan,
                    text_token_counts=cast("tuple[int, ...]", token_counts),
                    sampler_id=context.sampler.id,
                    sampler_options=context.request.options,
                    seed=context.seed,
                    pre_offset_sigmas=context.request.sigmas,
                    sigmas=context.schedule.sigmas,
                )
            except BaseException as error:
                manifest_error = error
            if window_preflight_failed(manifest_error is not None, execution_device):
                if manifest_error is not None:
                    raise manifest_error
                raise WiringError("peer flux window preflight failed")
            assert manifest is not None
            transport = WindowDigestConsensusTransport(distributed_config, execution_device)
            if transport.physical_ranks != tuple(range(distributed_config.world_size)):
                raise WiringError("window consensus group differs from the collective group")
            evaluation = DistributedFluxWindowEvaluation(
                window_evaluation,
                prove_manifest_consensus(manifest, rank=transport.rank, transport=transport),
            )
        evaluator = cast("Any", evaluation)

        def prepare_window_conditioning(value: object, _role: GuidanceRole) -> object:
            try:
                return evaluation.prepare_conditioning(value)
            except FluxWindowError as error:
                raise WiringError(str(error)) from None

        conditioning_evaluation = ConditioningEvaluation(
            prepare_window_conditioning,
            evaluation.evaluate_conditioning,
            evaluation.batchable,
            evaluation.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: "dinkster.flux.conditioning.v1",
            standard_activation_memory_factor=owner.family.memory_factor,
            layout=conditioning_layout,
            token_transforms=conditioning_token_transforms,
            validate_layout=evaluation.validate_layout,
            inner_calls=evaluation.inner_calls,
        )
    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", _FluxSamplingDenoiser(evaluator)),
        conditioning_evaluation=conditioning_evaluation,
        conditioning_payloads=MappingProxyType(replacements),
        distributed_evaluation=window_distributed,
    )


def _flux_device(runtime: object) -> torch.device:
    return module_compute_device(cast("FluxRuntime", runtime).assembled.diffusion)


def _flux_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("FluxRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


class FluxRuntime(SingleStreamSamplingRuntime):
    """FamilyRuntime[torch.Tensor] over an assembled classic Flux.

    ``assembled`` and ``codec`` are public on purpose: model/device
    placement is caller business (the residency seams), and phase-2
    composition may want the tiled codec entry points - both stay
    reachable without widening the protocol.

    ``encode_text`` is FluxClipModel.encode_token_weights @ b78cec87:
    CLIP-L with the SD1 policy (last hidden state, raw pooled) feeds
    only its pooled vector; T5-XXL (Flux chunking profile, final
    chunk padded to 256) provides the sequence. ``sample`` composes
    KSampler inputs into ``sample_custom``, which owns windowed and
    distributed execution over the family's flow space.
    """

    sampling_error = WiringError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=_FluxLatentAdapter(_validate_flux_latent),
        denoiser=_flux_denoiser,
        device=_flux_device,
        compute_dtype=_flux_compute_dtype,
        flow=True,
    )

    def __init__(
        self,
        assembled: AssembledFlux,
        *,
        runtime_identity: str,
        receipt_identity: str | None = None,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._runtime_identity = runtime_identity
        self._receipt_identity = receipt_identity
        self._space = _flux_sigma_space(assembled.family)
        self.codec: CodecPlugin = replace(
            kl_codec_plugin(assembled.vae),
            compute_dtype=assembled.compute_dtype("vae"),
        )
        # Flux CLIP-L is the reference default SDClipModel with
        # return_projected_pooled=False - exactly the encoder
        # defaults (CLIP_L_PROFILE + SD1_CLIP_L_POLICY); the T5
        # default is the Flux profile (T5_XXL_FLUX_PROFILE).
        if assembled.qwen3_2b is None:
            assert assembled.clip_l is not None
            assert assembled.t5xxl is not None
            lookups = embedding_lookups or {}
            clip_lookup = lookups.get("clip_l")
            t5_lookup = lookups.get("t5xxl")
            self._clip_encoder = ClipTextEncoder(assembled.clip_l, embeddings=clip_lookup)
            self._t5_encoder = T5TextEncoder(assembled.t5xxl, embeddings=t5_lookup)
            self._clip_tokenizer = PromptTokenizer(
                encode_word=load_clip_bpe().encode,
                resolve=(
                    None
                    if clip_lookup is None
                    else lambda name: (
                        None if (vectors := clip_lookup(name)) is None else vectors.shape[0]
                    )
                ),
            )
            self._t5_tokenizer = PromptTokenizer(
                encode_word=load_t5_spm().encode,
                resolve=(
                    None
                    if t5_lookup is None
                    else lambda name: (
                        None if (vectors := t5_lookup(name)) is None else vectors.shape[0]
                    )
                ),
            )
            self._ovis_encoder = None
        else:
            self._clip_encoder = None
            self._t5_encoder = None
            self._clip_tokenizer = None
            self._t5_tokenizer = None
            self._ovis_encoder = OvisTextEncoder(assembled.qwen3_2b)
        self._samplers = torch_sampler_registry(sampler_registry)
        # torch_scheduler_registry, not the pure builtin: brownian
        # solvers need bit-exact reference sigmas (schedules.py).
        # Caller-supplied registries are copied into a plain Registry
        # (the sampler path already rebuilds one) so no subclass code
        # runs during sampling lookups.
        self._schedulers = (
            _exact_registry(scheduler_registry)
            if scheduler_registry is not None
            else torch_scheduler_registry()
        )
        self._guidance = guidance_executor

    def with_sampling_space(self, space: SigmaSpace) -> FluxRuntime:
        if type(space) is not FluxFlowSigmas or not math.isfinite(space.shift):
            raise WiringError("Flux sampling override requires a finite exponential-flow space")
        try:
            sigmas = simple_schedule(space.timesteps, space)
        except OverflowError as error:
            raise WiringError("Flux sampling shift overflows its exponential") from error
        if not all(math.isfinite(sigma) and sigma > 0 for sigma in sigmas[:-1]):
            raise WiringError("Flux sampling shift produces nonfinite or zero positive sigmas")
        derived = copy(self)
        derived._space = space
        return derived

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def retained_offload_storage_components(self) -> frozenset[str]:
        """Keep classic text sources available for copy-free offload."""
        if self.assembled.qwen3_2b is not None:
            return frozenset()
        return frozenset({"clip_l", "t5xxl"})

    @property
    def runtime_identity(self) -> str:
        """See :class:`dinkster_inference.FamilyRuntime` for the
        stability contract. :func:`load_runtime` derives this from
        the assembly plan and dtype knobs (:func:`_flux_identity`);
        direct constructions supply their own string and own its
        stability."""
        return self._runtime_identity

    @property
    def receipt_identity(self) -> str | None:
        """Optional declared identity for comparison with distributed measurements."""
        return self._receipt_identity

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        if self._ovis_encoder is not None:
            return self._ovis_encoder.encode(text)
        assert self._t5_encoder is not None
        assert self._t5_tokenizer is not None
        assert self._clip_encoder is not None
        assert self._clip_tokenizer is not None
        t5_spans = self._t5_tokenizer.tokenize(text)
        if min_padding is None and min_length is None:
            t5 = self._t5_encoder.encode(t5_spans)
        else:
            t5 = self._t5_encoder.encode(
                t5_spans,
                min_padding=min_padding,
                min_length=min_length,
            )
        clip_spans = self._clip_tokenizer.tokenize(text)
        if hidden_layer is None:
            clip_l = self._clip_encoder.encode(clip_spans)
        else:
            clip_l = self._clip_encoder.encode(clip_spans, hidden_layer=hidden_layer)
        return compose_flux_conditioning(t5, clip_l)

    def encode_text_scheduled(
        self,
        request: ScheduledEncodeRequest,
        *,
        execution: ScheduledExecution[object] | None = None,
        transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        cancelled: Callable[[], bool] | None = None,
        type_registry: InferenceTypeRegistry,
    ) -> ConditioningCarrier:
        """Explicit B1 scheduled producer; ordinary encoding is unchanged."""
        return encode_text_scheduled(
            self,
            request,
            execution=execution,
            transforms=transforms,
            cancelled=cancelled,
            type_registry=type_registry,
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return self._space

    @property
    def supports_distilled_guidance(self) -> bool:
        return self.assembled.diffusion.config.guidance_embed

    sample_custom = cast("Any", sampling_execution)  # noqa: F811

    def sample_scheduled(
        self,
        latent: torch.Tensor,
        *,
        cond: ConditioningCarrier,
        cfg: SamplingGuidance[ConditioningCarrier] | None = None,
        resolver: ScheduledPatchResolver | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        schedule_device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Compose scheduled conditioning onto the custom sampling engine."""
        from .scheduled_sampling import ScheduledSamplingOptions

        result = run_ksampler_as_custom(
            self,
            latent,
            samplers=self._samplers,
            schedulers=self._schedulers,
            space=self._space,
            flow=is_flow_parameterization(self.family.sampling.parameterization),
            device=schedule_device,
            cond=cond,
            cfg=cfg,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            guidance=guidance,
            segment=segment,
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            noise_inds=noise_inds,
            on_step=on_step,
            on_state=on_state,
            sample_custom_kwargs={
                "scheduled": ScheduledSamplingOptions(resolver, None),
                "cancelled": cancelled,
                "compute_dtype": compute_dtype,
                "device": device,
                "capture_denoised": False,
            },
            error=WiringError,
        )
        assert type(result.output) is torch.Tensor
        return result.output

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)


def _validate_sd_control_gain_keys(
    gain: ContributionGain,
    lane_ids: tuple[str, ...],
    sites: tuple[str, ...],
) -> None:
    site_keys = tuple(key for key, _ in gain.site_gains)
    if site_keys and site_keys != sites:
        raise WiringError(
            "operator_site_mismatch: SD control site gains must cover exactly " + ", ".join(sites)
        )
    lane_keys = tuple(key for key, _ in gain.lane_gains)
    if lane_keys and lane_keys != tuple(sorted(lane_ids)):
        raise WiringError(
            "invalid_gain_schedule: SD1.5 ControlNet lane gains must cover exactly the "
            f"active guidance lanes {tuple(sorted(lane_ids))!r}"
        )


def _sd_control_gain(
    row: RealizedGainRow,
    lane_ids: tuple[str, ...],
    sites: tuple[str, ...],
) -> SDControlGain:
    site_gains = dict(row.site_gains)
    lane_gains = dict(row.lane_gains)
    timeline_global = row.timeline_gain * row.global_gain
    values = tuple(
        tuple(
            (timeline_global * site_gains.get(site_id, 1.0)) * lane_gains.get(lane_id, 1.0)
            for lane_id in lane_ids
        )
        for site_id in sites
    )
    if any(not math.isfinite(value) for site in values for value in site):
        raise WiringError("gain_domain_mismatch: SD1.5 ControlNet effective gain must be finite")
    return SDControlGain(lane_ids, values, row.effect_mask_digests)


def _validate_sd_latent(latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise WiringError("SD latent must have shape [batch,channels,height,width]")


class _SDLatentAdapter(SingleStreamLatentAdapter):
    def prepare(
        self,
        runtime: object,
        family: ModelFamily,
        *,
        latent: CustomSamplingLatentValue,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        denoise_mask: CustomSamplingLatentValue | None,
        context: SamplingAdapterContext,
        error: type[Exception],
    ) -> SamplingExecutionInputs:
        scheduled_request = (
            type(cond) is ConditioningCarrier or context.options.get("scheduled") is not None
        )
        if not scheduled_request:
            return super().prepare(
                runtime,
                family,
                latent=latent,
                noise=noise,
                cond=cond,
                cfg=cfg,
                denoise_mask=denoise_mask,
                context=context,
                error=error,
            )
        from .scheduled_sampling import ScheduledSamplingError, narrow_scheduled_values

        owner = cast("SDRuntime", runtime)
        executor = owner.sampling_execution_registration.guidance_executor(owner)
        if executor is not None and executor.registry.active:
            raise ScheduledSamplingError("guidance-extensions")
        if cfg is not None and cfg.transforms:
            raise ScheduledSamplingError("guidance-extensions")
        control = context.options.get("control")
        if control is not None:
            raise ScheduledSamplingError("control", family.id)
        contributions = context.options.get("sd15_attention_contributions", ())
        if type(contributions) is not tuple or contributions:
            raise ScheduledSamplingError("attention-contributions", family.id)
        if context.guidance is not None:
            raise ScheduledSamplingError("distilled-guidance", family.id)
        if (
            denoise_mask is not None
            or context.inpaint is not None
            or owner.assembled.diffusion.config.in_channels == 9
        ):
            raise ScheduledSamplingError("scheduled-sd-inpaint")
        if context.context_windows is not None:
            raise ScheduledSamplingError("context-windows", family.id)
        latent, noise, cond, cfg, denoise_mask = narrow_scheduled_values(
            family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
        )
        self.validate(latent)
        return SamplingExecutionInputs(
            latent,
            noise,
            cond,
            cfg,
            denoise_mask,
            family.single_stream_latent(),
        )


def _sd_scheduled_denoiser(
    owner: SDRuntime,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    from .scheduled_sampling import (
        ScheduledConditioningDenoiser,
        ScheduledSamplingError,
        ScheduledSamplingOptions,
        prepare_scheduled_carriers,
        validate_unconditional_carrier,
    )

    if (
        context.inputs is None
        or context.device is None
        or context.plan is None
        or context.request is None
        or context.schedule is None
    ):
        raise RuntimeError("scheduled SD sampling context is unresolved")
    inputs = context.inputs
    scheduled = context.options.get("scheduled")
    if scheduled is None:
        scheduled = ScheduledSamplingOptions()
    elif type(scheduled) is not ScheduledSamplingOptions:
        raise TypeError("scheduled must be an exact ScheduledSamplingOptions or None")
    unknown = set(context.options) - {"scheduled", "control", "sd15_attention_contributions"}
    if unknown:
        raise WiringError(
            "SD sampling does not accept adapter options: " + ", ".join(sorted(unknown))
        )
    if context.inputs.denoise_mask is not None:
        raise ScheduledSamplingError("denoise-mask", owner.family.id)
    if context.inpaint is not None:
        raise ScheduledSamplingError("scheduled-sd-inpaint")
    if context.context_windows is not None:
        raise ScheduledSamplingError("context-windows", owner.family.id)
    if context.options.get("control") is not None:
        raise ScheduledSamplingError("scheduled-sd-control")
    contributions = context.options.get("sd15_attention_contributions", ())
    if contributions:
        raise ScheduledSamplingError("scheduled-sd-attention")
    validate_unconditional_carrier(context.plan)
    device = torch.device(context.device)
    realized_timeline = (
        None
        if context.request.timeline is None
        else realize_sampling_timeline(
            context.request.timeline,
            tuple(float(sigma) for sigma in context.schedule.sigmas),
        )
    )
    conditional, unconditional, patch_sets, materialized_plan = prepare_scheduled_carriers(
        owner,
        inputs.latent,
        context.plan,
        resolver=scheduled.resolver,
        device=device,
        cancel=context.cancelled,
        timeline=realized_timeline,
        space=owner.sampling_sigma_space(),
    )

    adm = None
    if owner.assembled.diffusion.config.adm_in_channels is not None:

        def resolve_adm(region: Any, role: GuidanceRole) -> torch.Tensor | None:
            return owner._adm(  # pyright: ignore[reportPrivateUsage]
                region.conditioning,
                inputs.latent,
                negative=role is GuidanceRole.UNCONDITIONAL,
            )

        adm = resolve_adm

    from . import scheduled_sampling as scheduled_module

    evaluator = ScheduledConditioningDenoiser(
        conditional,
        unconditional,
        family_id=owner.family.id,
        space=owner.sampling_sigma_space(),
        model=owner.assembled.diffusion,
        evaluate=scheduled_module.sd_grouped_region_evaluator(
            owner.assembled.diffusion,
            owner.sampling_sigma_space(),
            parameterization=owner.sampling.parameterization,
            adm=adm,
            compute_dtype=compute_dtype,
        ),
        patch_sets=patch_sets,
        compute_dtype=compute_dtype,
        device=device,
        cancel=context.cancelled,
    )
    replacements = MappingProxyType(
        {
            condition.id: condition.conditioning
            for condition in materialized_plan.conditions
            if condition.conditioning is not None
        }
    )
    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        conditioning_evaluation=ConditioningEvaluation(
            evaluator.prepare_conditioning,
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: f"{owner.family.id}.scheduled-conditioning.v1",
            standard_activation_memory_factor=owner.family.memory_factor,
        ),
        conditioning_payloads=replacements,
        solver_options=MappingProxyType({"realized_timeline": realized_timeline}),
        sampling=owner.sampling,
        percent_to_sigma=owner._percent_to_sigma,  # pyright: ignore[reportPrivateUsage]
        close=evaluator.close,
    )


def _sd_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("SDRuntime", runtime)
    if (
        context.inputs is None
        or context.device is None
        or context.plan is None
        or context.request is None
        or context.sampler is None
        or context.schedule is None
    ):
        raise RuntimeError("SD sampling context is unresolved")
    if type(context.inputs.cond) is ConditioningCarrier:
        return _sd_scheduled_denoiser(owner, compute_dtype, context)
    unknown = set(context.options) - {"scheduled", "control", "sd15_attention_contributions"}
    if unknown:
        raise WiringError(
            "SD sampling does not accept adapter options: " + ", ".join(sorted(unknown))
        )
    if context.options.get("scheduled") is not None:
        raise WiringError("scheduled SD conditioning requires a ConditioningCarrier")
    control = context.options.get("control")
    controls: tuple[SDControlConditioning, ...] = ()
    control_sites = SD15_CONTROL_RESIDUAL_SITES
    if control is not None:
        if type(control) is not SDControlConditioning:
            raise TypeError("control must be an exact SDControlConditioning or None")
        control = _snapshot_sd_control_conditioning(control)
        newest_to_oldest: list[SDControlConditioning] = []
        current: SDControlConditioning | None = control
        while current is not None:
            newest_to_oldest.append(current)
            current = current.previous
        controls = tuple(reversed(newest_to_oldest))
        sdxl_control = tuple(
            type(current.model) in (SDXLControlLoRA, SDXLControlNet, SDXLControlNetUnion)
            for current in controls
        )
        if any(sdxl_control) and not all(sdxl_control):
            raise WiringError("control chain mixes SD1.5 and SDXL providers")
        if all(sdxl_control):
            if owner.family.engine.controlnet_profile != "sdxl":
                raise WiringError("SDXL control providers require the registered SDXL profile")
            control_sites = SDXL_CONTROL_RESIDUAL_SITES
        elif owner.family.engine.controlnet_profile != "sd15":
            raise WiringError("SD1.5 control providers require the registered SD1.5 profile")
        for current in controls:
            if current.gain is not None and (
                current.application.strength != 1.0
                or current.application.window.start_percent != 0.0
                or current.application.window.end_percent != 1.0
            ):
                raise WiringError(
                    "explicit ControlNet gain requires application strength 1.0 and the full"
                    " [0, 1] window; set the application to identity or fold the intended"
                    " scaling/window into the gain keyframes"
                )
    contributions = context.options.get("sd15_attention_contributions", ())
    if type(contributions) is not tuple or any(
        type(contribution) is not SD15IPAdapterConditioning for contribution in contributions
    ):
        raise TypeError("sd15_attention_contributions must be an exact tuple")
    ipadapter = owner._ipadapter_executions(contributions)  # pyright: ignore[reportPrivateUsage]
    inputs = context.inputs
    sigmas = context.schedule.sigmas
    control_effect_fields = tuple(
        tuple(
            compile_sd_effect_mask(
                source,
                latent_height=inputs.latent.shape[2],
                latent_width=inputs.latent.shape[3],
            )
            for source in current.effect_masks
        )
        for current in controls
    )
    if any(
        source.mask.shape[0] not in (1, inputs.latent.shape[0])
        for current in controls
        for source in current.effect_masks
    ):
        raise WiringError(
            "mask_layout_mismatch: effect-mask batch must be one or equal latent batch"
        )
    off_grid_control = bool(controls) and context.sampler.id in {
        "dinkster.dpm_fast",
        "dinkster.dpm_adaptive",
    }
    realized_timeline = (
        None
        if context.request.timeline is None
        else realize_sampling_timeline(
            context.request.timeline,
            tuple(float(sigma) for sigma in sigmas),
        )
    )
    plan = context.plan
    admit_extra_lanes = plan.needs_unconditional or plan.has_strategy
    admitted = [True] + [
        condition.conditioning is not None and admit_extra_lanes
        for condition in plan.conditions[1:]
    ]
    control_lane_ids = tuple(
        "positive"
        if condition.role is GuidanceRole.CONDITIONAL
        else "negative"
        if condition.role is GuidanceRole.UNCONDITIONAL
        else "empty"
        for condition, admit in zip(plan.conditions, admitted, strict=True)
        if admit
    )
    control_tables = []
    constant_control_gains: tuple[float, ...] | None = None
    constant_control_gain_rows: tuple[SDControlGain, ...] | None = None
    structured_control_gains = False
    control_facts: tuple[str, ...] = ()
    if controls and len(sigmas) > 1:
        timeline = (
            realized_timeline.executed
            if realized_timeline is not None
            else executed_sampling_timeline(tuple(float(sigma) for sigma in sigmas))
        )
        fact_parts: list[str] = []
        off_grid_gains: list[float] = []
        off_grid_gain_rows: list[SDControlGain] = []
        for index, current in enumerate(controls):
            gain = current.gain
            if gain is None:
                start_sigma = owner._percent_to_sigma(  # pyright: ignore[reportPrivateUsage]
                    current.application.window.start_percent
                )
                end_sigma = owner._percent_to_sigma(  # pyright: ignore[reportPrivateUsage]
                    current.application.window.end_percent
                )
                gain = ContributionGain(
                    DirectGainTableCurve(
                        tuple(
                            current.application.strength
                            if end_sigma <= sigma <= start_sigma
                            else 0.0
                            for sigma in sigmas[:-1]
                        )
                    ),
                    1.0,
                )
            _validate_sd_control_gain_keys(gain, control_lane_ids, control_sites)
            if (gain.site_gains or gain.lane_gains) and plan.has_strategy:
                raise WiringError(
                    "unsupported_control_partition: SD1.5 site/lane gains require the "
                    "builtin guidance lane plan"
                )
            table = realize_gain_table(gain, timeline)
            fields = control_effect_fields[index]
            resolved_digests = tuple(field.compiled.input_digest for field in fields)
            required_digests = tuple(
                dict.fromkeys(digest for row in table.rows for digest in row.effect_mask_digests)
            )
            if resolved_digests != required_digests:
                raise WiringError(
                    "mask_role_mismatch: realized effect-mask declarations must resolve "
                    "exactly in first-use order"
                )
            effective_gains = tuple(row.timeline_gain * row.global_gain for row in table.rows)
            gain_rows = tuple(
                _sd_control_gain(row, control_lane_ids, control_sites) for row in table.rows
            )
            structured_control_gains = structured_control_gains or bool(
                gain.site_gains or gain.lane_gains or required_digests
            )
            if off_grid_control:
                full_window = (
                    current.application.window.start_percent == 0.0
                    and current.application.window.end_percent == 1.0
                )
                if not full_window or any(value != gain_rows[0] for value in gain_rows[1:]):
                    raise WiringError(
                        f"sampler {context.sampler.id} supports only constant ControlNet gain over"
                        " the full application window because its internal evaluation timeline"
                        " does not map to executed sigma rows"
                    )
                off_grid_gains.append(effective_gains[0])
                off_grid_gain_rows.append(gain_rows[0])
            control_tables.append(table)
            prefix = f"control[{index}]"
            fact_parts.extend(
                (
                    f"{prefix}.child={current.application.child_id}",
                    f"{prefix}.model={current.model_digest}",
                    f"{prefix}.hint={current.hint_digest}",
                    *(
                        ()
                        if current.application.mode is None
                        else (
                            f"{prefix}.mode.provider={current.application.mode.provider}",
                            f"{prefix}.mode.token={current.application.mode.token}",
                        )
                    ),
                    *(f"{prefix}.{fact}" for fact in contribution_gain_slot_facts(gain, table)),
                    *(
                        fact
                        for mask_index, field in enumerate(fields)
                        for fact in (
                            f"{prefix}.effect_mask[{mask_index}].source="
                            f"{field.compiled.source_digest}",
                            f"{prefix}.effect_mask[{mask_index}].field={field.compiled.digest}",
                            f"{prefix}.effect_mask[{mask_index}].layout="
                            f"{field.compiled.layout_digest}",
                            f"{prefix}.effect_mask[{mask_index}].transform="
                            f"{field.compiled.transform_digest}",
                        )
                    ),
                )
            )
        if off_grid_control:
            if structured_control_gains:
                constant_control_gain_rows = tuple(off_grid_gain_rows)
            else:
                constant_control_gains = tuple(off_grid_gains)
        control_facts = tuple(fact_parts)
    admitted_sources = [
        condition.conditioning
        for condition, admit in zip(plan.conditions, admitted, strict=True)
        if admit
    ]
    token_counts = [
        declared_token_count(source) if isinstance(source, Conditioning) else None
        for source in admitted_sources
    ]
    declared_counts = [count for count in token_counts if count is not None]
    if len(declared_counts) == len(token_counts):
        repeats = cross_attn_repeat(declared_counts)
        target_counts = (
            [math.lcm(*declared_counts)] * len(declared_counts)
            if repeats is not None
            else declared_counts
        )
    else:
        target_counts = [0 if count is None else count for count in token_counts]
    bound_sources = [
        bind_sd_layout(source, target_count)
        for source, target_count in zip(admitted_sources, target_counts, strict=True)
    ]
    bound_iter = iter(bound_sources)
    replacements = MappingProxyType(
        {
            condition.id: next(bound_iter)
            for condition, admit in zip(plan.conditions, admitted, strict=True)
            if admit
        }
    )
    is_inpaint = owner.assembled.diffusion.config.in_channels == 9
    inpaint = cast("InpaintConditioning[torch.Tensor] | None", context.inpaint)
    evaluator = SDDenoiser(
        owner.assembled.diffusion,
        owner.sampling_sigma_space(),
        parameterization=owner.sampling.parameterization,
        inpaint_mask=(
            inpaint.mask if inpaint is not None else inputs.denoise_mask if is_inpaint else None
        ),
        inpaint_masked_image=(
            latent_process_in(
                inpaint.masked_image if inpaint is not None else inputs.latent,
                owner.family.single_stream_latent(),
            )
            if is_inpaint
            else None
        ),
        control_model=(
            None
            if not controls
            else controls[0].model
            if len(controls) == 1
            else tuple(current.model for current in controls)
        ),
        control_hint=(
            None
            if not controls
            else controls[0].hint
            if len(controls) == 1
            else tuple(current.hint for current in controls)
        ),
        control_mode=(
            None
            if not controls
            else controls[0].application.mode
            if len(controls) == 1
            else tuple(current.application.mode for current in controls)
        ),
        control_effect_masks=control_effect_fields,
        ipadapter=ipadapter,
        compute_dtype=compute_dtype,
    )
    if constant_control_gains is not None:
        if len(constant_control_gains) == 1:
            evaluator.set_control_gain(constant_control_gains[0])
        else:
            evaluator.set_control_gains(constant_control_gains)
    elif constant_control_gain_rows is not None:
        evaluator.set_control_gain_rows(constant_control_gain_rows)

    def prepare_conditioning(
        value: object, role: GuidanceRole
    ) -> tuple[torch.Tensor, torch.Tensor | None, str]:
        if not isinstance(value, Conditioning):
            raise WiringError("SD guidance lane requires Conditioning")
        return evaluator.prepare_conditioning(
            value,
            adm=owner._adm(  # pyright: ignore[reportPrivateUsage]
                value,
                inputs.latent,
                negative=role is not GuidanceRole.CONDITIONAL,
            ),
            lane_id=(
                "positive"
                if role is GuidanceRole.CONDITIONAL
                else "negative"
                if role is GuidanceRole.UNCONDITIONAL
                else "empty"
            ),
        )

    evaluator_identity = "dinkster.sd.conditioning.v1"
    if control_facts:
        digest = hashlib.sha256("\n".join((*control_facts, "")).encode()).hexdigest()
        evaluator_identity += f":intervention-plan={digest}"
    evaluator_identity = owner._extend_ipadapter_evaluator_identity(  # pyright: ignore[reportPrivateUsage]
        evaluator_identity, contributions
    )
    conditioning = ConditioningEvaluation(
        prepare_conditioning,
        evaluator.evaluate_conditioning,
        evaluator.batchable,
        evaluator.evaluate_conditioning_batch,
        evaluate_batch_attention=evaluator.evaluate_conditioning_batch_attention,
        evaluator_identity=lambda _role: evaluator_identity,
        standard_activation_memory_factor=owner.family.memory_factor,
        layout=conditioning_layout,
        token_transforms=conditioning_token_transforms,
        validate_layout=lambda condition, layout: validate_sd_layout(
            condition[:2],
            layout,
            repeat_limit=CROSS_ATTN_REPEAT_LIMIT,
        ),
    )

    def apply_control_step(index: int) -> None:
        anchor = control_tables[0].rows[index]
        require_realized_sampling_step(index, anchor.sigma, anchor.progress)
        if structured_control_gains:
            evaluator.set_control_gain_rows(
                tuple(
                    _sd_control_gain(table.rows[index], control_lane_ids, control_sites)
                    for table in control_tables
                )
            )
        elif len(control_tables) == 1:
            row = control_tables[0].rows[index]
            evaluator.set_control_gain(row.timeline_gain * row.global_gain)
        else:
            evaluator.set_control_gains(
                tuple(
                    table.rows[index].timeline_gain * table.rows[index].global_gain
                    for table in control_tables
                )
            )

    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        conditioning_evaluation=conditioning,
        conditioning_payloads=replacements,
        solver_options=MappingProxyType({"realized_timeline": realized_timeline}),
        sampling=owner.sampling,
        percent_to_sigma=owner._percent_to_sigma,  # pyright: ignore[reportPrivateUsage]
        on_step_begin=(apply_control_step if control_tables and not off_grid_control else None),
        inpaint_noise=(
            prepare_noise(inputs.latent, context.seed + 1)
            if context.sampler.random_inpaint_noise
            else None
        ),
    )


def _sd_device(runtime: object) -> torch.device:
    return cast("SDRuntime", runtime)._compute_device  # pyright: ignore[reportPrivateUsage]


def _sd_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("SDRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.float16


class SDRuntime(SingleStreamSamplingRuntime):
    """FamilyRuntime[torch.Tensor] over an assembled SD 1.5 / SDXL
    base / SDXL refiner.

    ``assembled`` and ``codec`` are public for the same reasons as
    :class:`FluxRuntime`. ``encode_text`` is the family's reference
    CLIP stack @ b78cec87: SD 1.5 is SD1ClipModel (CLIP-L, final
    hidden state, raw pooled); SDXL base is SDXLClipModel (CLIP-L +
    CLIP-G, both penultimate-without-norm, feature-concatenated,
    CLIP-G's projected pooled); the refiner is SDXLRefinerClipModel
    (CLIP-G alone). ``sample`` is the KSampler body over the shared
    checkpoint-selected discrete linear-beta or continuous EDM sigma
    space, with EPS or v-prediction parameterization and optional
    zero-terminal-SNR rescale on the discrete path, and with
    SDXL's ADM vectors built from each conditioning's pooled output
    and the latent's pixel dimensions - the reference
    encode_model_conds defaults (width/height = latent * 8, targets
    = sizes, crops 0, refiner aesthetic 6.0 positive / 2.5
    negative), exactly what a plain CLIPTextEncode + KSampler
    workflow hits.
    """

    sampling_error = WiringError
    sampling_compute_dtype = torch.float16
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=_SDLatentAdapter(_validate_sd_latent, admit_perp_neg=True),
        denoiser=_sd_denoiser,
        device=_sd_device,
        compute_dtype=_sd_compute_dtype,
        flow=False,
    )

    def __init__(
        self,
        assembled: AssembledSD,
        *,
        runtime_identity: str,
        receipt_identity: str | None = None,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._runtime_identity = runtime_identity
        self._receipt_identity = receipt_identity
        self.codec: CodecPlugin = replace(
            (
                taesd_codec_plugin(assembled.vae)
                if isinstance(assembled.vae, TAESD)
                else kl_codec_plugin(assembled.vae)
            ),
            compute_dtype=assembled.compute_dtype("vae"),
        )
        self.sampling = assembled.sampling or assembled.family.sampling
        if self.sampling.space is SamplingSpace.CONTINUOUS_EDM:
            edm_space = ContinuousEDMSigmas(
                min_sigma=self.sampling.sigma_min,
                max_sigma=self.sampling.sigma_max,
            )
            self._space: SigmaSpace = edm_space

            def percent_to_sigma(percent: float) -> float:
                return continuous_edm_percent_to_sigma(edm_space, percent)

            self._percent_to_sigma = percent_to_sigma
        else:
            # ModelSamplingDiscrete's SD15/SDXL registration: linear
            # betas 0.00085..0.012 over 1000 steps.
            discrete_space = DiscreteSigmas.linear_beta(zsnr=self.sampling.zsnr)
            self._space = discrete_space

            def percent_to_sigma(percent: float) -> float:
                return discrete_percent_to_sigma(discrete_space, percent)

            self._percent_to_sigma = percent_to_sigma
        sd1 = assembled.family.engine.clip_text_profile == "sd1"
        lookups = embedding_lookups or {}
        self._clip_l_encoder = (
            None
            if assembled.clip_l is None
            else ClipTextEncoder(
                assembled.clip_l,
                policy=SD1_CLIP_L_POLICY if sd1 else SDXL_CLIP_POLICY,
                embeddings=lookups.get("clip_l"),
            )
        )
        self._clip_g_encoder = (
            None
            if assembled.clip_g is None
            else ClipTextEncoder(
                assembled.clip_g,
                profile=CLIP_G_PROFILE,
                policy=SDXL_CLIP_POLICY,
                embeddings=lookups.get("clip_g"),
            )
        )
        active_lookups = tuple(
            lookup
            for component in ("clip_l", "clip_g")
            if (lookup := lookups.get(component)) is not None
        )

        def resolve_embedding_rows(name: str) -> int | None:
            vectors = tuple(lookup(name) for lookup in active_lookups)
            if not vectors or all(value is None for value in vectors):
                return None
            if any(value is None for value in vectors):
                raise WiringError(f"embedding {name!r} is missing an active SD text component")
            rows = {value.shape[0] for value in vectors if value is not None}
            if len(rows) != 1:
                raise WiringError(
                    f"embedding {name!r} has incompatible component row counts {sorted(rows)}"
                )
            return next(iter(rows))

        # Both towers share the merges/vocab BPE; the per-tower
        # differences (pad token, projected pooled) live in the
        # encoder profile/policy above.
        self._tokenizer = PromptTokenizer(
            encode_word=load_clip_bpe().encode,
            resolve=resolve_embedding_rows if active_lookups else None,
        )
        self._samplers = torch_sampler_registry(sampler_registry)
        # torch_scheduler_registry, not the pure builtin: brownian
        # solvers need bit-exact reference sigmas (schedules.py).
        # Caller-supplied registries are copied into a plain Registry
        # (the sampler path already rebuilds one) so no subclass code
        # runs during sampling lookups.
        self._schedulers = (
            _exact_registry(scheduler_registry)
            if scheduler_registry is not None
            else torch_scheduler_registry()
        )
        self._guidance = guidance_executor

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        """See :class:`dinkster_inference.FamilyRuntime` for the
        stability contract; derived like :class:`FluxRuntime`'s."""
        return self._runtime_identity

    @property
    def receipt_identity(self) -> str | None:
        """Optional declared identity for comparison with distributed measurements."""
        return self._receipt_identity

    @property
    def conditioning_identity(self) -> str:
        return self._runtime_identity

    @property
    def _compute_device(self) -> torch.device:
        return module_compute_device(self.assembled.diffusion)

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[torch.Tensor]:
        return materialize_basic_conditioning(carrier, device=self._compute_device)

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        tokens = self._tokenizer.tokenize(text)
        clip_l = (
            None
            if self._clip_l_encoder is None
            else (
                self._clip_l_encoder.encode(tokens)
                if hidden_layer is None
                else self._clip_l_encoder.encode(tokens, hidden_layer=hidden_layer)
            )
        )
        clip_g = (
            None
            if self._clip_g_encoder is None
            else (
                self._clip_g_encoder.encode(tokens)
                if hidden_layer is None
                else self._clip_g_encoder.encode(tokens, hidden_layer=hidden_layer)
            )
        )
        if clip_l is not None and clip_g is not None:
            return compose_sdxl_conditioning(clip_l, clip_g)
        conditioning = clip_l if clip_l is not None else clip_g
        assert conditioning is not None  # AssembledSD wires >= 1 tower
        return conditioning

    def encode_text_scheduled(
        self,
        request: ScheduledEncodeRequest,
        *,
        execution: ScheduledExecution[object] | None = None,
        transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        cancelled: Callable[[], bool] | None = None,
        type_registry: InferenceTypeRegistry,
    ) -> ConditioningCarrier:
        """Explicit B1 scheduled producer; ordinary encoding is unchanged."""
        return encode_text_scheduled(
            self,
            request,
            execution=execution,
            transforms=transforms,
            cancelled=cancelled,
            type_registry=type_registry,
        )

    def _adm(
        self,
        conditioning: Conditioning[torch.Tensor] | None,
        latent: torch.Tensor,
        *,
        negative: bool,
    ) -> torch.Tensor | None:
        """The reference extra_conds ADM for one conditioning: None
        for SD 1.5 (no ADM), encode_adm with encode_model_conds'
        latent-derived pixel sizes for SDXL base/refiner
        (comfy/samplers.py encode_model_conds, comfy/model_base.py
        SDXL/SDXLRefiner.encode_adm @ b78cec87)."""
        if conditioning is None:
            return None
        adm_channels = self.assembled.diffusion.config.adm_in_channels
        if adm_channels is None:
            return None
        pooled = conditioning.pooled
        if pooled is None:
            raise WiringError(
                "this family's ADM conditioning needs the pooled text"
                " output, but the conditioning has none"
            )
        scale = self.family.single_stream_latent().spatial_downscale
        height = latent.shape[2] * scale
        width = latent.shape[3] * scale
        if self.family.engine.adm_profile == "sdxl_refiner":
            return encode_sdxl_refiner_adm(
                pooled,
                width=width,
                height=height,
                aesthetic_score=(
                    SDXL_NEGATIVE_AESTHETIC_DEFAULT if negative else SDXL_AESTHETIC_DEFAULT
                ),
            )
        return encode_sdxl_adm(pooled, width=width, height=height)

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return self._space

    def _sampling_percent_to_sigma(self, space: SigmaSpace, percent: float) -> float:
        return self._percent_to_sigma(percent)

    def _ipadapter_executions(
        self,
        contributions: tuple[SD15IPAdapterConditioning, ...],
    ) -> tuple[SD15IPAdapterExecution, ...]:
        if type(contributions) is not tuple or any(
            type(contribution) is not SD15IPAdapterConditioning for contribution in contributions
        ):
            raise TypeError("sd15_attention_contributions must be an exact tuple")
        if contributions and self.family.engine.ipadapter_profile != "sd15":
            raise WiringError("this family does not register the standard IP-Adapter profile")
        return tuple(
            SD15IPAdapterExecution(
                contribution,
                self._percent_to_sigma(contribution.declaration.window.start_percent),
                self._percent_to_sigma(contribution.declaration.window.end_percent),
            )
            for contribution in contributions
        )

    @staticmethod
    def _extend_ipadapter_evaluator_identity(
        identity: str,
        contributions: tuple[SD15IPAdapterConditioning, ...],
    ) -> str:
        if contributions:
            digest = hashlib.sha256(
                "\n".join((*sd15_ipadapter_identity_facts(contributions), "")).encode()
            ).hexdigest()
            identity += f":ipadapter={digest}"
        return identity

    @property
    def supports_inpaint(self) -> bool:
        return self.assembled.diffusion.config.in_channels == 9

    sample_custom = cast("Any", sampling_execution)  # noqa: F811

    def sample_scheduled(
        self,
        latent: torch.Tensor,
        *,
        cond: ConditioningCarrier,
        cfg: SamplingGuidance[ConditioningCarrier] | None = None,
        resolver: ScheduledPatchResolver | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        compute_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        schedule_device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Compose scheduled conditioning onto the custom sampling engine."""
        from .scheduled_sampling import ScheduledSamplingOptions

        result = run_ksampler_as_custom(
            self,
            latent,
            samplers=self._samplers,
            schedulers=self._schedulers,
            space=self._space,
            flow=False,
            device=schedule_device,
            cond=cond,
            cfg=cfg,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            guidance=guidance,
            segment=segment,
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            noise_inds=noise_inds,
            on_step=on_step,
            on_state=on_state,
            sample_custom_kwargs={
                "scheduled": ScheduledSamplingOptions(resolver, cancelled),
                "compute_dtype": compute_dtype,
                "device": device,
                "capture_denoised": False,
            },
            error=WiringError,
        )
        assert type(result.output) is torch.Tensor
        return result.output

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)


def _load_flux(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> FluxRuntime:
    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if not isinstance(plan, FluxAssemblyPlan):
        raise WiringError("Flux loader requires a Flux assembly plan")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_flux(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    applicable = frozenset() if assembled.qwen3_2b is not None else frozenset({"clip_l", "t5xxl"})
    supplied = frozenset(embedding_lookups or {})
    if embedding_lookups is not None and not applicable <= supplied:
        raise WiringError(
            f"family {plan.family.id} is missing embedding lookups {sorted(applicable - supplied)}"
        )
    applicable_lookups = (
        None if embedding_lookups is None else {key: embedding_lookups[key] for key in applicable}
    )
    return FluxRuntime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            embedding_binding_digest=embedding_binding_digest,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        guidance_executor=guidance_executor,
        embedding_lookups=applicable_lookups,
    )


def _load_qwen_image(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> QwenImageRuntime:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
    from dinkster_inference.qwen_image_assembly import qwen_image_checkpoint_assembly

    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if isinstance(plan, ComponentCheckpointPlan):
        plan = qwen_image_checkpoint_assembly(plan)
    if not isinstance(plan, QwenImageAssemblyPlan):
        raise WiringError("Qwen Image loader requires a Qwen Image assembly plan")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise WiringError("Qwen Image does not support textual-inversion bindings")
    if guidance_executor is not None:
        raise WiringError("Qwen Image does not support custom guidance execution")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_qwen_image(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    return QwenImageRuntime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
    )


def _load_sd(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> SDRuntime:
    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if not isinstance(plan, SDAssemblyPlan):
        raise WiringError("SD loader requires an SD assembly plan")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_sd(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    applicable = frozenset(
        component for component in ("clip_l", "clip_g") if getattr(assembled, component) is not None
    )
    supplied = frozenset(embedding_lookups or {})
    if embedding_lookups is not None and not applicable <= supplied:
        raise WiringError(
            f"family {plan.family.id} is missing embedding lookups {sorted(applicable - supplied)}"
        )
    applicable_lookups = (
        None if embedding_lookups is None else {key: embedding_lookups[key] for key in applicable}
    )
    return SDRuntime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            embedding_binding_digest=embedding_binding_digest,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        guidance_executor=guidance_executor,
        embedding_lookups=applicable_lookups,
    )


def _load_wan(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> Wan21Runtime:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
    from dinkster_inference.wan21_component import wan_checkpoint_assembly

    if pose_cache_settings is not None and type(pose_cache_settings) is not (
        Wan21PoseBlockCacheSettings
    ):
        raise TypeError("pose_cache_settings must be exact Wan21PoseBlockCacheSettings")
    if guidance_executor is not None:
        raise WiringError("Wan does not support guidance extensions")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise WiringError("Wan does not support textual-inversion bindings")
    if isinstance(plan, ComponentCheckpointPlan):
        plan = wan_checkpoint_assembly(plan)
    if not isinstance(plan, Wan21AssemblyPlan):
        raise WiringError("Wan loader requires a Wan assembly plan")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_wan21(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    return Wan21Runtime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            embedding_binding_digest=None,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            runtime_facts=(
                () if pose_cache_settings is None else pose_cache_settings.runtime_facts
            ),
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        pose_cache_settings=pose_cache_settings,
    )


def _load_z_image(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> ZImageRuntime:
    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if not isinstance(plan, ZImageAssemblyPlan):
        raise WiringError("Z-Image loader requires a Z-Image assembly plan")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise WiringError("Z-Image does not support textual-inversion bindings")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_z_image(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    return ZImageRuntime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        guidance_executor=guidance_executor,
    )


def _load_flux2(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> Flux2Runtime:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
    from dinkster_inference.flux2_assembly import flux2_checkpoint_assembly

    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if isinstance(plan, ComponentCheckpointPlan):
        plan = flux2_checkpoint_assembly(plan)
    if not isinstance(plan, Flux2AssemblyPlan):
        raise WiringError("Flux2 loader requires a Flux2 assembly plan")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise WiringError("Flux2 does not support textual-inversion bindings")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_flux2(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(
        assembled,
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    return Flux2Runtime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        guidance_executor=guidance_executor,
    )


def _load_lumina2(
    plan: NativeAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype | None,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool,
    fp8_matmul: bool,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None,
    scheduler_registry: Registry[SchedulerDescriptor] | None,
    registry_token: str | None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
) -> Lumina2Runtime:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
    from dinkster_inference.lumina2_component import lumina2_checkpoint_assembly

    if pose_cache_settings is not None:
        raise WiringError("Animate2 pose cache settings require a Wan 2.1 runtime")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise WiringError("Lumina2 does not support textual-inversion bindings")
    if isinstance(plan, ComponentCheckpointPlan):
        plan = lumina2_checkpoint_assembly(plan)
    if not isinstance(plan, Lumina2AssemblyPlan):
        raise WiringError("Lumina2 loader requires a Lumina2 assembly plan")
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    assembled = assemble_lumina2(
        plan,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    assembled = replace(assembled, _storage_dtype_follows_compute=storage_dtype_follows_compute)
    return Lumina2Runtime(
        assembled,
        runtime_identity=build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=_identity_dtype(diffusion_dtype),
            text_dtype=_identity_dtype(text_dtype),
            vae_dtype=_identity_dtype(vae_dtype),
            fp8_matmul=fp8_matmul,
            registry_token=registry_token,
            extension_behavior_hash=extension_behavior_hash,
            patch_overlay_digests=patch_overlay_digests,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        guidance_executor=guidance_executor,
    )


def _resolve_assembly_loader(assembly: AssemblyRegistration) -> Callable[..., Any]:
    module, attribute = assembly.load.split(":")
    try:
        loader = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise WiringError(
            f"assembly {assembly.id!r} loader {assembly.load!r} is unavailable: {error}"
        ) from error
    if not callable(loader):
        raise WiringError(f"assembly {assembly.id!r} loader {assembly.load!r} is not callable")
    return loader


@overload
def load_runtime(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    diffusion_dtype: torch.dtype | None = None,
    text_dtype: torch.dtype | None = None,
    vae_dtype: torch.dtype | None = None,
    storage_dtype_follows_compute: bool = False,
    fp8_matmul: bool = False,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
    scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    registry_token: str | None = None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    expected_identity: str | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
) -> FamilyRuntime[torch.Tensor] | Wan21Runtime: ...


@overload
def load_runtime(
    checkpoint: Any = None,
    *,
    diffusion: Any = None,
    clip_l: Any = None,
    clip_g: Any = None,
    clip_vision: Any = None,
    t5xxl: Any = None,
    gemma3_12b: Any = None,
    mistral3_24b: Any = None,
    qwen2_5_vl_7b: Any = None,
    qwen3_06b: Any = None,
    qwen3_2b: Any = None,
    qwen3_4b: Any = None,
    qwen3_8b: Any = None,
    qwen3vl_4b: Any = None,
    vae: Any = None,
    diffusion_dtype: Any = None,
    text_dtype: Any = None,
    vae_dtype: Any = None,
    storage_dtype_follows_compute: Any = False,
    fp8_matmul: Any = False,
    sampler_registry: Any = None,
    scheduler_registry: Any = None,
    registry_token: Any = None,
    extension_behavior_hash: Any = None,
    patch_overlay_digests: Any = None,
    expected_identity: Any = None,
    guidance_executor: Any = None,
    embedding_lookups: Any = None,
    embedding_binding_digest: Any = None,
    attention_policy: Any = "auto",
    attention_route_token: Any = None,
    pose_cache_settings: Any = None,
    assembly_registry: Any = None,
) -> FamilyRuntime[torch.Tensor] | Wan21Runtime: ...


def load_runtime(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    diffusion_dtype: torch.dtype | None = None,
    text_dtype: torch.dtype | None = None,
    vae_dtype: torch.dtype | None = None,
    storage_dtype_follows_compute: bool = False,
    fp8_matmul: bool = False,
    sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
    scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    registry_token: str | None = None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    expected_identity: str | None = None,
    guidance_executor: GuidanceExecutor | None = None,
    embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
) -> FamilyRuntime[torch.Tensor] | Wan21Runtime:
    """Plan component geometry, assemble, and bind the shared runtime.

    Sources mirror probe_native: a combined checkpoint with optional split
    overrides for compatible components. Dtype knobs mirror the
    assemblers, and ``diffusion_dtype=None`` picks the family's
    reference default (bf16 Flux/Wan, fp16 SD-era). ``text_dtype=None``
    picks the family's reference text dtype and ``vae_dtype=None`` picks
    the family's preferred VAE dtype. Planning uses the same geometry and
    scalar-configuration validation as probe_native, not a family-ID gate.

    When ``storage_dtype_follows_compute`` is true, later residency
    enrollment stores eligible components in their compute dtype unless that
    would widen checkpoint storage; wider compute uses cast-at-use instead.
    The residency layer owns eligibility and reports a closed per-component
    outcome set; callers select no component or storage kind. The default
    false value preserves checkpoint storage exactly. Active patches make
    the whole assembly preserve its original storage, while anomalous
    unrouted or unmanaged state on an otherwise eligible component refuses
    before any conversion.

    Custom registries change assembly, sampler, or scheduler execution, so
    they must arrive with a ``registry_token`` - a stable string the
    registry owner rotates whenever a registered implementation
    changes - which folds into runtime_identity (a same-id solver
    swap without a rotated token would serve stale native cache
    entries). The builtin registries need no token; passing a token
    without a custom registry is refused as a likely mistake.
    Animate2 pose-cache settings are accepted only by its Wan 2.1 profile;
    lossy cache storage modes fold into runtime identity while default storage
    preserves the uncached identity.
    ``expected_identity`` asserts the host-computed cache tag after
    construction (the stage-6 dispatch contract)."""
    custom_registries = (
        sampler_registry is not None
        or scheduler_registry is not None
        or assembly_registry is not None
    )
    if (embedding_lookups is None) != (embedding_binding_digest is None):
        raise WiringError(
            "embedding_lookups and embedding_binding_digest must be provided together"
        )
    if embedding_lookups is not None:
        unknown = set(embedding_lookups) - {"clip_l", "clip_g", "t5xxl"}
        if unknown:
            raise WiringError(f"unknown embedding lookup components: {sorted(unknown)}")
    if custom_registries and registry_token is None:
        raise WiringError(
            "custom assembly/sampler/scheduler registries need a registry_token"
            " for runtime_identity rotation"
        )
    if registry_token is not None and not custom_registries:
        raise WiringError("registry_token given without a custom registry")
    try:
        resolution = resolve_native_assembly(
            checkpoint,
            diffusion=diffusion,
            clip_l=clip_l,
            clip_g=clip_g,
            clip_vision=clip_vision,
            t5xxl=t5xxl,
            gemma3_12b=gemma3_12b,
            mistral3_24b=mistral3_24b,
            qwen2_5_vl_7b=qwen2_5_vl_7b,
            qwen3_06b=qwen3_06b,
            qwen3_2b=qwen3_2b,
            qwen3_4b=qwen3_4b,
            qwen3_8b=qwen3_8b,
            qwen3vl_4b=qwen3vl_4b,
            vae=vae,
            assembly_registry=assembly_registry,
            fp8_matmul=fp8_matmul,
        )
    except NativeRefusalError as error:
        raise WiringError(
            "cannot assemble checkpoint components: " + "; ".join(error.reasons)
        ) from error
    plan = resolution.plan
    loader = _resolve_assembly_loader(resolution.registration)
    if diffusion_dtype is None:
        diffusion_dtype = _torch_dtype(default_diffusion_dtype(plan.family.id))
    if text_dtype is None:
        text_dtype = _torch_dtype(default_text_dtype(plan.family.id))
    if vae_dtype is None:
        vae_dtype = _torch_dtype(default_vae_dtype(plan.family.id))
    options: dict[str, Any] = dict(
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        storage_dtype_follows_compute=storage_dtype_follows_compute,
        fp8_matmul=fp8_matmul,
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        registry_token=registry_token,
        extension_behavior_hash=extension_behavior_hash,
        patch_overlay_digests=patch_overlay_digests,
        guidance_executor=guidance_executor,
        embedding_lookups=embedding_lookups,
        embedding_binding_digest=embedding_binding_digest,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        pose_cache_settings=pose_cache_settings,
    )
    runtime = loader(plan, **options)
    if expected_identity is not None and expected_identity != runtime.runtime_identity:
        raise WiringError(
            f"expected runtime identity {expected_identity!r},"
            f" constructed {runtime.runtime_identity!r}"
        )
    return runtime


__all__ = [
    "Flux2Runtime",
    "FluxRuntime",
    "QwenImageRuntime",
    "SDRuntime",
    "Wan21Runtime",
    "ZImageRuntime",
    "WiringError",
    "load_runtime",
    "wired_runtime_family_ids",
    "_load_flux",
    "_load_flux2",
    "_load_lumina2",
    "_load_qwen_image",
    "_load_sd",
    "_load_wan",
    "_load_z_image",
]
