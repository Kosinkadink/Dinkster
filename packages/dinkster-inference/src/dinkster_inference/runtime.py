"""Torch-free component planning and the shared runtime contracts.

Planning selects component configurations from source geometry without
materializing weights. Family labels provide optional specializations, not
permission to execute. The Torch realization lives in
``dinkster_inference_torch.wiring``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, overload, runtime_checkable

from .assembly import (
    AssemblyError,
    ComponentPlan,
    _detect_flux_family_normalized,  # pyright: ignore[reportPrivateUsage]
    plan_flux_assembly,
    plan_sd_assembly,
    plan_wan22_assembly,
    plan_z_image_assembly,
)
from .cfg import DualSamplingGuidance, PerpNegSamplingGuidance, SamplingGuidance
from .clip_text import ClipTextConfig
from .component_registry import ComponentRegistry
from .conditioning_wire import ConditioningCarrier
from .context_windows import ContextWindowsSpec
from .devices import FLOAT8_E5M2
from .families import FamilyRegistry, ModelFamily
from .latents import MultiStreamLatent
from .observation import ExecutionObserverAttachment
from .patches import SizedTensor
from .quantization import QuantizationError
from .refusal import NativeRefusalCategory, NativeRefusalError
from .registry import Registry
from .sampling import (
    ArithTensor,
    CustomSamplingRequest,
    CustomSamplingResult,
    SamplingSegment,
    SamplingStateCallback,
    StepCallback,
)
from .spaces import SigmaSpace
from .sparse import SparseLatent
from .text_encoders import Conditioning
from .weights import WeightSource


class RuntimeTensor(ArithTensor, SizedTensor, Protocol):
    """What the runtime seam demands of a tensor: the sampling
    arithmetic (ArithTensor - the denoise drive) plus a shape
    (SizedTensor - the Conditioning payloads). torch.Tensor satisfies
    both structurally; tests use plain value types."""


RuntimeTensorT = TypeVar("RuntimeTensorT", bound=RuntimeTensor)


@dataclass(frozen=True)
class InpaintConditioning(Generic[RuntimeTensorT]):
    """Model-local SD inpaint mask and masked-image latent."""

    mask: RuntimeTensorT
    masked_image: RuntimeTensorT


@dataclass(frozen=True, slots=True)
class PreparedMultiStreamConditioning:
    """Family-owned conditioning payload bound to one compatible runtime contract."""

    runtime_identity: str
    payload: object

    def __post_init__(self) -> None:
        if type(self.runtime_identity) is not str or not self.runtime_identity:
            raise ValueError("prepared conditioning requires a runtime identity")


@dataclass(frozen=True, slots=True)
class AudioPreview(Generic[RuntimeTensorT]):
    waveform: RuntimeTensorT
    sample_rate: int

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("audio preview sample rate must be a positive integer")


@runtime_checkable
class MultiStreamFamilyRuntime(Protocol[RuntimeTensorT]):
    """Family adapter used by ordinary KSampler for structural latents."""

    @property
    def runtime_identity(self) -> str: ...

    def sample_multistream(
        self,
        latent: MultiStreamLatent[RuntimeTensorT],
        *,
        conditioning: object,
        cfg: SamplingGuidance[object] | DualSamplingGuidance[object] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float,
        seed: int,
        segment: SamplingSegment | None = None,
        denoise_mask: RuntimeTensorT | MultiStreamLatent[RuntimeTensorT] | None = None,
        noise_inds: Sequence[int] | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        observer: ExecutionObserverAttachment | None = None,
        parent_span_id: int | None = None,
    ) -> MultiStreamLatent[RuntimeTensorT]: ...


@runtime_checkable
class MultiStreamLatentAdapterRuntime(Protocol[RuntimeTensorT]):
    """Model-declared conversion from a plain latent to structural streams."""

    def adapt_multistream_latent(
        self,
        latent: RuntimeTensorT,
        *,
        source_spatial_downscale: int | None = None,
        source_temporal_downscale: int | None = None,
    ) -> MultiStreamLatent[RuntimeTensorT]: ...


@runtime_checkable
class DenoiseMaskRuntime(Protocol):
    """Runtime-owned admission for sampler denoise masks."""

    @property
    def supports_denoise_mask(self) -> bool: ...


@runtime_checkable
class SamplingSpaceOverrideRuntime(Protocol):
    """Derive a runtime sharing weights but owning an immutable sampling space.

    Schedules, percent conversions, SNR offsets and Brownian bounds must all
    use the override. Unsupported spaces raise rather than being ignored.
    """

    def with_sampling_space(self, space: SigmaSpace) -> CustomSamplingRuntime[Any]: ...


@runtime_checkable
class ContextWindowsRuntime(Protocol):
    """Runtime-owned admission for context-windowed sampling.

    A True value admits the family; the runtime still validates the
    active model profile when sampling starts.
    """

    @property
    def supports_context_windows(self) -> bool: ...


@runtime_checkable
class MultiStreamConditioningRuntime(Protocol):
    """Family adapter that materializes canonical conditioning carriers.

    Generic sampling consumers narrow to this protocol to turn one
    canonical :class:`ConditioningCarrier` into the family-owned opaque
    payload, then bind it as
    ``PreparedMultiStreamConditioning(conditioning_identity, payload)``.
    The runtime's ``sample_multistream`` keeps receiving the unwrapped
    payload; the identity check stays at the consumer seam.
    ``prepare_conditioning`` must raise on carriers the runtime's profile
    cannot materialize.
    """

    @property
    def runtime_identity(self) -> str: ...

    @property
    def conditioning_identity(self) -> str: ...

    def prepare_conditioning(self, carrier: ConditioningCarrier) -> object: ...


@runtime_checkable
class ConditioningRuntime(Protocol):
    """Single-stream runtime that materializes canonical conditioning carriers."""

    @property
    def runtime_identity(self) -> str: ...

    @property
    def conditioning_identity(self) -> str: ...

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[Any]: ...


@runtime_checkable
class CustomSamplingRuntime(Protocol[RuntimeTensorT]):
    """The one sampling execution seam: an exact custom sampling request
    over the family's latent representation.

    Every family implements only this contract; KSampler-flavored
    surfaces are compositions of it. The latent and noise are a plain
    tensor for dense single-stream families, a :class:`MultiStreamLatent`
    for structural streams, or a support-authenticated :class:`SparseLatent`;
    each family refuses the representations it does not execute with
    exact-type checks at admission. Conditioning arrives as
    :class:`Conditioning`, a canonical :class:`ConditioningCarrier`, or
    a family-opaque :class:`PreparedMultiStreamConditioning` whose runtime
    identity the family checks before unwrapping. Distributed-mode admission
    belongs to implementations of this seam, never to a node or dispatch path."""

    @property
    def family(self) -> ModelFamily: ...

    @property
    def runtime_identity(self) -> str: ...

    def custom_sampling_sigmas(
        self,
        scheduler_id: str,
        steps: int,
        denoise: float,
        *,
        device: Any = None,
    ) -> tuple[float, ...]: ...

    def custom_sampling_beta_sigmas(
        self,
        steps: int,
        alpha: float,
        beta: float,
        *,
        device: Any = None,
    ) -> tuple[float, ...]: ...

    def custom_sampling_sd_turbo_sigmas(
        self,
        steps: int,
        denoise: float,
        *,
        device: Any = None,
    ) -> tuple[float, ...]: ...

    def custom_sampling_percent_to_sigma(
        self,
        percent: float,
        *,
        return_actual_sigma: bool,
    ) -> float: ...

    def check_custom_sampling(
        self,
        request: CustomSamplingRequest[RuntimeTensorT],
        *,
        has_denoise_mask: bool,
        has_inpaint: bool,
        has_context_windows: bool,
        guidance: float | None = None,
    ) -> None:
        """Refuse unsupported invocation modes before residency staging."""
        ...

    def sample_custom(
        self,
        latent: RuntimeTensorT | MultiStreamLatent[RuntimeTensorT] | SparseLatent[RuntimeTensorT],
        *,
        noise: RuntimeTensorT | MultiStreamLatent[RuntimeTensorT] | SparseLatent[RuntimeTensorT],
        cond: (
            Conditioning[RuntimeTensorT] | ConditioningCarrier | PreparedMultiStreamConditioning
        ),
        cfg: (
            SamplingGuidance[Conditioning[RuntimeTensorT]]
            | SamplingGuidance[ConditioningCarrier]
            | SamplingGuidance[PreparedMultiStreamConditioning]
            | DualSamplingGuidance[Conditioning[RuntimeTensorT]]
            | DualSamplingGuidance[PreparedMultiStreamConditioning]
            | PerpNegSamplingGuidance[Conditioning[RuntimeTensorT]]
            | PerpNegSamplingGuidance[PreparedMultiStreamConditioning]
            | None
        ),
        request: CustomSamplingRequest[RuntimeTensorT],
        seed: int = 0,
        guidance: float | None = None,
        denoise_mask: (
            RuntimeTensorT | MultiStreamLatent[RuntimeTensorT] | SparseLatent[RuntimeTensorT] | None
        ) = None,
        inpaint: InpaintConditioning[RuntimeTensorT] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
    ) -> (
        CustomSamplingResult[RuntimeTensorT]
        | CustomSamplingResult[MultiStreamLatent[RuntimeTensorT]]
        | CustomSamplingResult[SparseLatent[RuntimeTensorT]]
    ): ...


@runtime_checkable
class MultiStreamPreviewRuntime(Protocol[RuntimeTensorT]):
    """Family codec capabilities used by generic stream preview nodes."""

    def preview_visual(self, role: str, latent: RuntimeTensorT) -> RuntimeTensorT: ...

    def preview_audio(self, role: str, latent: RuntimeTensorT) -> AudioPreview[RuntimeTensorT]: ...


@runtime_checkable
class NativeAssemblyPlan(Protocol):
    @property
    def family(self) -> ModelFamily: ...

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any] | None, ...]: ...


def uses_classic_embedding_bindings(plan: NativeAssemblyPlan) -> bool:
    """Textual inversion applies to declared classic CLIP text components."""
    return any(
        component is not None and isinstance(component.config, ClipTextConfig)
        for component in plan.identity_components
    )


@dataclass(frozen=True, slots=True)
class AssemblyRegistration:
    """One geometry planner and its lazily imported runtime constructor."""

    id: str
    plan: Callable[..., object]
    load: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.load.count(":") != 1 or not all(self.load.split(":")):
            raise ValueError("assembly loader must name a module:attribute")


@dataclass(frozen=True, slots=True)
class NativeAssemblyResolution:
    registration: AssemblyRegistration
    plan: NativeAssemblyPlan


def _plan_flux(
    *,
    checkpoint: WeightSource | None,
    diffusion: WeightSource | None,
    clip_l: WeightSource | None,
    clip_g: WeightSource | None,
    clip_vision: WeightSource | None,
    t5xxl: WeightSource | None,
    gemma3_12b: WeightSource | None,
    mistral3_24b: WeightSource | None,
    qwen3_06b: WeightSource | None,
    qwen3_2b: WeightSource | None,
    qwen3_4b: WeightSource | None,
    qwen3_8b: WeightSource | None,
    qwen2_5_vl_7b: WeightSource | None,
    qwen3vl_4b: WeightSource | None,
    vae: WeightSource | None,
) -> object:
    if clip_g is not None:
        raise AssemblyError("a split clip_g source was given but this family wires no CLIP-G slot")
    if clip_vision is not None:
        raise AssemblyError(
            "a split clip_vision source was given but this family wires no vision slot"
        )
    if gemma3_12b is not None:
        raise AssemblyError(
            "a split gemma3_12b source was given but this family wires no Gemma 3 slot"
        )
    if mistral3_24b is not None:
        raise AssemblyError(
            "a split mistral3_24b source was given but this family wires no Mistral3 slot"
        )
    if qwen3_06b is not None:
        raise AssemblyError(
            "a split qwen3_06b source was given but this family wires no Qwen3-0.6B slot"
        )
    if qwen3_4b is not None:
        raise AssemblyError(
            "a split qwen3_4b source was given but this family wires no Qwen3-4B slot"
        )
    if qwen3_8b is not None:
        raise AssemblyError(
            "a split qwen3_8b source was given but this family wires no Qwen3-8B slot"
        )
    if qwen2_5_vl_7b is not None:
        raise AssemblyError(
            "a split qwen2_5_vl_7b source was given but this family wires no Qwen2.5-VL slot"
        )
    if qwen3vl_4b is not None:
        raise AssemblyError(
            "a split qwen3vl_4b source was given but this family wires no Qwen3-VL-4B slot"
        )
    return plan_flux_assembly(
        checkpoint=checkpoint,
        diffusion=diffusion,
        clip_l=clip_l,
        t5xxl=t5xxl,
        qwen3_2b=qwen3_2b,
        vae=vae,
    )


def _plan_sd(
    *,
    checkpoint: WeightSource | None,
    diffusion: WeightSource | None,
    clip_l: WeightSource | None,
    clip_g: WeightSource | None,
    clip_vision: WeightSource | None,
    t5xxl: WeightSource | None,
    gemma3_12b: WeightSource | None,
    mistral3_24b: WeightSource | None,
    qwen3_06b: WeightSource | None,
    qwen3_2b: WeightSource | None,
    qwen3_4b: WeightSource | None,
    qwen3_8b: WeightSource | None,
    qwen2_5_vl_7b: WeightSource | None,
    qwen3vl_4b: WeightSource | None,
    vae: WeightSource | None,
) -> object:
    if clip_vision is not None:
        raise AssemblyError(
            "a split clip_vision source was given but this family wires no vision slot"
        )
    if t5xxl is not None:
        raise AssemblyError("a split t5xxl source was given but this family wires no T5 slot")
    if gemma3_12b is not None:
        raise AssemblyError(
            "a split gemma3_12b source was given but this family wires no Gemma 3 slot"
        )
    if mistral3_24b is not None:
        raise AssemblyError(
            "a split mistral3_24b source was given but this family wires no Mistral3 slot"
        )
    if qwen3_06b is not None:
        raise AssemblyError("a split qwen3_06b source was given but this family wires no Qwen slot")
    if qwen3_2b is not None:
        raise AssemblyError("a split qwen3_2b source was given but this family wires no Qwen slot")
    if qwen3_4b is not None:
        raise AssemblyError("a split qwen3_4b source was given but this family wires no Qwen slot")
    if qwen3_8b is not None:
        raise AssemblyError("a split qwen3_8b source was given but this family wires no Qwen slot")
    if qwen2_5_vl_7b is not None:
        raise AssemblyError(
            "a split qwen2_5_vl_7b source was given but this family wires no Qwen slot"
        )
    if qwen3vl_4b is not None:
        raise AssemblyError(
            "a split qwen3vl_4b source was given but this family wires no Qwen slot"
        )
    return plan_sd_assembly(
        checkpoint=checkpoint,
        diffusion=diffusion,
        clip_l=clip_l,
        clip_g=clip_g,
        vae=vae,
    )


def _plan_wan22(
    *,
    checkpoint: WeightSource | None,
    diffusion: WeightSource | None,
    clip_l: WeightSource | None,
    clip_g: WeightSource | None,
    clip_vision: WeightSource | None,
    t5xxl: WeightSource | None,
    gemma3_12b: WeightSource | None,
    mistral3_24b: WeightSource | None,
    qwen3_06b: WeightSource | None,
    qwen3_2b: WeightSource | None,
    qwen3_4b: WeightSource | None,
    qwen3_8b: WeightSource | None,
    qwen2_5_vl_7b: WeightSource | None,
    qwen3vl_4b: WeightSource | None,
    vae: WeightSource | None,
) -> object:
    if any(
        source is not None
        for source in (
            clip_l,
            clip_g,
            clip_vision,
            gemma3_12b,
            mistral3_24b,
            qwen3_06b,
            qwen3_2b,
            qwen3_4b,
            qwen3_8b,
            qwen2_5_vl_7b,
            qwen3vl_4b,
        )
    ):
        raise AssemblyError("Wan 2.2 wires only diffusion, UMT5-XXL, and VAE sources")
    return plan_wan22_assembly(
        checkpoint=checkpoint,
        diffusion=diffusion,
        umt5xxl=t5xxl,
        vae=vae,
    )


def _plan_z_image(
    *,
    checkpoint: WeightSource | None,
    diffusion: WeightSource | None,
    clip_l: WeightSource | None,
    clip_g: WeightSource | None,
    clip_vision: WeightSource | None,
    t5xxl: WeightSource | None,
    gemma3_12b: WeightSource | None,
    mistral3_24b: WeightSource | None,
    qwen3_06b: WeightSource | None,
    qwen3_2b: WeightSource | None,
    qwen3_4b: WeightSource | None,
    qwen3_8b: WeightSource | None,
    qwen2_5_vl_7b: WeightSource | None,
    qwen3vl_4b: WeightSource | None,
    vae: WeightSource | None,
) -> object:
    if clip_vision is not None:
        raise AssemblyError(
            "a split clip_vision source was given but this family wires no vision slot"
        )
    if any(
        source is not None
        for source in (
            clip_l,
            clip_g,
            t5xxl,
            gemma3_12b,
            mistral3_24b,
            qwen3_06b,
            qwen3_2b,
            qwen3_8b,
            qwen2_5_vl_7b,
            qwen3vl_4b,
        )
    ):
        raise AssemblyError("Z-Image wires only the qwen3_4b text encoder slot")
    return plan_z_image_assembly(
        checkpoint=checkpoint,
        diffusion=diffusion,
        qwen3_4b=qwen3_4b,
        vae=vae,
    )


_ASSEMBLIES = (
    AssemblyRegistration(
        "dinkster.flux",
        _plan_flux,
        "dinkster_inference_torch.wiring:_load_flux",
        aliases=("dinkster.flux_dev", "dinkster.flux_schnell"),
    ),
    AssemblyRegistration(
        "dinkster.sd",
        _plan_sd,
        "dinkster_inference_torch.wiring:_load_sd",
        aliases=("dinkster.sd15", "dinkster.sdxl", "dinkster.sdxl_refiner"),
    ),
    AssemblyRegistration(
        "dinkster.wan22", _plan_wan22, "dinkster_inference_torch.wiring:_load_wan"
    ),
    AssemblyRegistration(
        "dinkster.z_image_runtime",
        _plan_z_image,
        "dinkster_inference_torch.wiring:_load_z_image",
        aliases=("dinkster.z_image", "dinkster.z_image_pixel_space"),
    ),
)


def build_builtin_assembly_registry(
    component_registry: ComponentRegistry,
) -> Registry[AssemblyRegistration]:
    from .component_checkpoint import plan_component_checkpoint

    registry: Registry[AssemblyRegistration] = Registry()
    for assembly in _ASSEMBLIES:
        registry.register(assembly)
    registry.register(
        AssemblyRegistration(
            "dinkster.components",
            plan_component_checkpoint,
            "dinkster_inference_torch.component_runtime:load_component_checkpoint",
            aliases=tuple(
                descriptor.id
                for descriptor in component_registry
                if descriptor.checkpoint_loader is not None
            ),
        )
    )
    return registry


def builtin_assembly_registry() -> Registry[AssemblyRegistration]:
    """Build the builtin assembly registry through the aggregate factory."""
    from .registries import builtin_registries

    return builtin_registries().assemblies


def wired_runtime_family_ids() -> tuple[str, ...]:
    """Registered runtime labels for diagnostics, never an admission predicate."""
    from .registries import builtin_registries

    return tuple(
        sorted(
            name
            for assembly in builtin_registries().assemblies
            for name in (assembly.aliases or (assembly.id,))
        )
    )


def _require_refusal_category(category: object) -> NativeRefusalCategory:
    if not isinstance(category, NativeRefusalCategory):
        raise TypeError("category must be a NativeRefusalCategory")
    return category


@dataclass(frozen=True)
class NativeCapability:
    """The family and typed native-planning result from :func:`probe_native`.

    ``native_ineligible`` requires a complete valid ordinary-owner plan
    unsupported by the requested native capability. Invalid, ambiguous,
    incomplete, identity, configuration, and unknown cases fail closed as
    ``invalid_input_or_identity``. Refusal reasons are diagnostics only;
    callers branch on ``refusal_category`` and never parse reason text.
    """

    family_id: str | None
    native: bool
    reasons: tuple[str, ...] = ()
    refusal_category: NativeRefusalCategory | None = None

    def __post_init__(self) -> None:
        if self.refusal_category is not None:
            _require_refusal_category(self.refusal_category)
        if self.native and self.reasons:
            raise ValueError("a native=True capability carries no reasons")
        if not self.native and not self.reasons:
            raise ValueError("a native=False capability must name reasons")
        if self.native and not self.family_id:
            raise ValueError("a native=True capability names its family")
        if self.native and self.refusal_category is not None:
            raise ValueError("a native=True capability carries no refusal category")
        if any(not reason for reason in self.reasons):
            raise ValueError("a refusal reason cannot be blank")
        if not self.native and self.refusal_category is None:
            object.__setattr__(
                self,
                "refusal_category",
                NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY,
            )


class _NativePlanRefusal(NativeRefusalError):
    def __init__(
        self,
        family_id: str | None,
        reasons: tuple[str, ...],
        category: NativeRefusalCategory = NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY,
    ) -> None:
        self.family_id = family_id
        super().__init__(reasons, category)


def _validated_native_plan(plan: NativeAssemblyPlan, *, fp8_matmul: bool) -> NativeAssemblyPlan:
    if fp8_matmul and any(
        dtype == FLOAT8_E5M2
        for component in plan.identity_components
        if component is not None
        for dtype in component.dtypes.values()
    ):
        raise _NativePlanRefusal(
            plan.family.id,
            ("fp8 matmul does not support float8_e5m2 checkpoint storage",),
            NativeRefusalCategory.NATIVE_INELIGIBLE,
        )
    return plan


def resolve_native_assembly(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    registry: FamilyRegistry | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
    fp8_matmul: bool = False,
) -> NativeAssemblyResolution:
    detect_source = diffusion if diffusion is not None else checkpoint
    if detect_source is None:
        raise ValueError(
            "probe_native needs a diffusion or checkpoint source to detect a family from"
        )
    if registry is None or assembly_registry is None:
        from .registries import builtin_registries

        registries = builtin_registries()
        if registry is None:
            registry = registries.families
        if assembly_registry is None:
            assembly_registry = registries.assemblies
    result = registry.detect(detect_source)
    family_id = result.best.family_id if result.best is not None else None
    plans: list[NativeAssemblyResolution] = []
    problems: list[str] = []
    for assembly in assembly_registry:
        try:
            plan = assembly.plan(
                checkpoint=checkpoint,
                diffusion=diffusion,
                clip_l=clip_l,
                clip_g=clip_g,
                clip_vision=clip_vision,
                t5xxl=t5xxl,
                gemma3_12b=gemma3_12b,
                mistral3_24b=mistral3_24b,
                qwen3_06b=qwen3_06b,
                qwen3_2b=qwen3_2b,
                qwen3_4b=qwen3_4b,
                qwen3_8b=qwen3_8b,
                qwen2_5_vl_7b=qwen2_5_vl_7b,
                qwen3vl_4b=qwen3vl_4b,
                vae=vae,
            )
        except AssemblyError as error:
            problems.append(f"{assembly.id}: {error}")
            continue
        if not isinstance(plan, NativeAssemblyPlan):
            raise TypeError(f"assembly {assembly.id!r} did not produce a NativeAssemblyPlan")
        plans.append(NativeAssemblyResolution(assembly, plan))
    if not plans:
        # Retain bounded quantization diagnostics hidden by no-match detectors.
        try:
            _detect_flux_family_normalized(checkpoint=checkpoint, diffusion=diffusion)
        except AssemblyError as error:
            cause: BaseException | None = error
            while cause is not None:
                if isinstance(cause, QuantizationError):
                    raise _NativePlanRefusal(family_id, (str(error),)) from error
                cause = cause.__cause__
        raise _NativePlanRefusal(
            family_id,
            (
                "checkpoint components do not match an executable architecture: "
                + f"detected labels={tuple(item.family_id for item in result.candidates)!r}; "
                + "; ".join(dict.fromkeys(problems)),
            ),
        )
    if len(plans) > 1:
        raise _NativePlanRefusal(
            family_id,
            (
                "checkpoint component geometry is ambiguous between assembly configurations: "
                + ", ".join(item.registration.id for item in plans),
            ),
        )
    resolution = plans[0]
    plan = _validated_native_plan(resolution.plan, fp8_matmul=fp8_matmul)
    if family_id != plan.family.id:
        logging.getLogger(__name__).warning(
            "Checkpoint label %r has no matching specialization; defaulting model, text, "
            "codec, sampling and dtype behavior from detected %s component configuration",
            family_id if not result.ambiguous else result.ambiguous,
            type(plan).__name__,
        )
    return resolution


@overload
def plan_native(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    registry: FamilyRegistry | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
    fp8_matmul: bool = False,
) -> NativeAssemblyPlan: ...


@overload
def plan_native(
    checkpoint: Any = None,
    *,
    diffusion: Any = None,
    clip_l: Any = None,
    clip_g: Any = None,
    clip_vision: Any = None,
    t5xxl: Any = None,
    gemma3_12b: Any = None,
    qwen2_5_vl_7b: Any = None,
    mistral3_24b: Any = None,
    qwen3_06b: Any = None,
    qwen3_2b: Any = None,
    qwen3_4b: Any = None,
    qwen3_8b: Any = None,
    qwen3vl_4b: Any = None,
    vae: Any = None,
    registry: Any = None,
    assembly_registry: Any = None,
    fp8_matmul: Any = False,
) -> Any: ...


def plan_native(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    registry: FamilyRegistry | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
    fp8_matmul: bool = False,
) -> NativeAssemblyPlan:
    """Plan component assembly from the union of native source slots.

    Detection uses diffusion, then checkpoint.
    """
    return resolve_native_assembly(
        checkpoint,
        diffusion=diffusion,
        clip_l=clip_l,
        clip_g=clip_g,
        clip_vision=clip_vision,
        t5xxl=t5xxl,
        gemma3_12b=gemma3_12b,
        mistral3_24b=mistral3_24b,
        qwen3_06b=qwen3_06b,
        qwen3_2b=qwen3_2b,
        qwen3_4b=qwen3_4b,
        qwen3_8b=qwen3_8b,
        qwen2_5_vl_7b=qwen2_5_vl_7b,
        qwen3vl_4b=qwen3vl_4b,
        vae=vae,
        registry=registry,
        assembly_registry=assembly_registry,
        fp8_matmul=fp8_matmul,
    ).plan


def probe_native(
    checkpoint: WeightSource | None = None,
    *,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    gemma3_12b: WeightSource | None = None,
    mistral3_24b: WeightSource | None = None,
    qwen3_06b: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    qwen3_8b: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    qwen3vl_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
    registry: FamilyRegistry | None = None,
    assembly_registry: Registry[AssemblyRegistration] | None = None,
    fp8_matmul: bool = False,
) -> NativeCapability:
    """Can this checkpoint (or split-source set) go native?

    Raw family detection is header-only. A zero-match packed-quantization
    fallback may read bounded ``.comfy_quant`` configuration bytes to
    normalize logical geometry. Assembly planning reads keys, tensor
    geometry, metadata, and only explicitly modeled scalar configuration
    tensors whose values affect execution and pre-load identity. It never
    materializes model weights, so the probe remains
    cheap enough to run per dispatch decision. Pure per
    its inputs (seam pin): no filesystem enumeration beyond the
    passed sources, no global state - the default registry is
    rebuilt fresh from the immutable builtin catalog on every call
    (:func:`builtin_family_registry`), so equal sources with stable
    contents always yield equal verdicts for the life of the process.
    Sources are the union of component planner slots (a combined
    ``checkpoint``, with split files overriding their component); a split
    source incompatible with the detected component configuration refuses. Detection
    runs on diffusion, then checkpoint. At least one is required.

    The verdict is based on component geometry and configuration, not family
    labels. Missing label specializations use the detected component defaults
    with a diagnostic. Invalid or ambiguous geometry, missing components, and
    incompatible quantization remain explicit structural refusals.
    """
    try:
        plan = plan_native(
            checkpoint,
            diffusion=diffusion,
            clip_l=clip_l,
            clip_g=clip_g,
            clip_vision=clip_vision,
            t5xxl=t5xxl,
            gemma3_12b=gemma3_12b,
            mistral3_24b=mistral3_24b,
            qwen3_06b=qwen3_06b,
            qwen3_2b=qwen3_2b,
            qwen3_4b=qwen3_4b,
            qwen3_8b=qwen3_8b,
            qwen2_5_vl_7b=qwen2_5_vl_7b,
            qwen3vl_4b=qwen3vl_4b,
            vae=vae,
            registry=registry,
            assembly_registry=assembly_registry,
            fp8_matmul=fp8_matmul,
        )
    except _NativePlanRefusal as error:
        return NativeCapability(
            family_id=error.family_id,
            native=False,
            reasons=error.reasons,
            refusal_category=error.category,
        )
    return NativeCapability(family_id=plan.family.id, native=True)


@runtime_checkable
class FamilyRuntime(Protocol[RuntimeTensorT]):
    """The operations exposed by an assembled checkpoint - the typed
    counterpart of the reference's (ModelPatcher, CLIP, VAE) triple,
    shaped after the consuming node bodies:
    clip_text_encode -> :meth:`encode_text`, ksampler ->
    :meth:`sample`, vae_decode/vae_encode -> :meth:`decode_latent` /
    :meth:`encode_content`.

    Implementations own their family's composition choices (which
    text encoders, which sigma space, which denoiser); callers speak
    registry ids and tensors. Model/device placement stays caller
    business (the residency seams), exactly as it is for the pieces
    this protocol fronts.
    """

    @property
    def family(self) -> ModelFamily: ...

    @property
    def runtime_identity(self) -> str:
        """The execution body's identity, for cache-key rotation
        (seam pin, backend amendment 1). Stability contract: the
        same component plans and knobs (dtypes, quantization,
        matmul policy) produce the same string across processes;
        ANY behavior-affecting change produces a new string - a
        different component set, a different knob, or a wiring-math
        change without a component change (implementations bind
        those changes into identity facts). It identifies the execution
        body, not the weight bytes: the dispatcher mixes in asset
        identity separately."""
        ...

    def encode_text(self, text: str) -> Conditioning[RuntimeTensorT]:
        """One prompt -> this family's conditioning (the reference's
        CLIP.encode_from_tokens over the family's tokenizer set)."""
        ...

    def sample(
        self,
        latent: RuntimeTensorT,
        *,
        cond: Conditioning[RuntimeTensorT],
        cfg: SamplingGuidance[Conditioning[RuntimeTensorT]] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: RuntimeTensorT | None = None,
        inpaint: InpaintConditioning[RuntimeTensorT] | None = None,
        noise_inds: Sequence[int] | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
    ) -> RuntimeTensorT:
        """One full denoise run - the KSampler node body: schedule
        the sigmas (``denoise`` is the img2img strength), draw the
        deterministic initial noise from ``seed``, and drive the
        registered solver. ``cfg`` bundles the classifier-free
        guidance inputs (negative conditioning and scale); None means
        plain conditional sampling. Implementations hand it to the
        shared guidance executor, which owns the lane fan-out and
        combine arithmetic; they may consult its
        ``needs_unconditional_lane`` predicate to plan lanes but never
        reimplement the CFG math.
        ``sampler_id`` / ``scheduler_id`` resolve
        via the (injectable) registries, aliases included, so legacy
        ComfyUI names keep working. ``guidance`` is the distilled
        guidance embedding input for families that embed it (Flux
        dev); families without the embedder refuse a non-None value.
        ``denoise_mask`` is the sampler mask. ``inpaint`` separately
        carries a native inpaint model's local mask/masked-image inputs.
        ``noise_inds`` selects per-batch noise draws: None keeps the
        full-batch draw byte-identical to a call without the argument;
        a sequence follows the reference ``prepare_noise`` batch_inds
        semantics (rows before a skipped index are drawn and discarded,
        repeated indices share a draw). Families that cannot express
        per-batch indices refuse a non-None value loudly.
        """
        ...

    def decode_latent(self, latent: RuntimeTensorT) -> RuntimeTensorT:
        """Latent -> content through the family codec (VAEDecode)."""
        ...

    def encode_content(self, content: RuntimeTensorT) -> RuntimeTensorT:
        """Content -> latent through the family codec (VAEEncode)."""
        ...


__all__ = [
    "AssemblyRegistration",
    "AudioPreview",
    "CustomSamplingRuntime",
    "DenoiseMaskRuntime",
    "FamilyRuntime",
    "InpaintConditioning",
    "MultiStreamFamilyRuntime",
    "MultiStreamLatentAdapterRuntime",
    "MultiStreamPreviewRuntime",
    "NativeAssemblyPlan",
    "NativeAssemblyResolution",
    "NativeCapability",
    "NativeRefusalCategory",
    "NativeRefusalError",
    "PreparedMultiStreamConditioning",
    "RuntimeTensor",
    "SamplingSpaceOverrideRuntime",
    "builtin_assembly_registry",
    "build_builtin_assembly_registry",
    "plan_native",
    "probe_native",
    "resolve_native_assembly",
    "wired_runtime_family_ids",
]
