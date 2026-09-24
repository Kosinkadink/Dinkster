"""Native text encoding, flow denoising, and codec wiring for Flux2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    FLUX2_LATENT_CHANNELS,
    Conditioning,
    ConditioningCarrier,
    ConditioningSet,
    FluxFlowSigmas,
    GuidanceRole,
    ModelFamily,
    PayloadReference,
    Registry,
    SamplerDescriptor,
    SchedulerDescriptor,
    TekkenBpe,
    encode_conditioning_carrier,
    load_flux2_tekken_bpe,
    make_conditioning_carrier,
)

from .assemble import AssembledFlux2
from .autoencoder_kl import kl_codec_plugin
from .codecs import CodecPlugin
from .conditioning_adapters import materialize_basic_conditioning
from .denoise import (
    FluxCondition,
    FluxDenoiser,
    _flux_condition_parts,  # pyright: ignore[reportPrivateUsage]
)
from .flux import Flux
from .flux2_assembly import FLUX2_TEKKEN_ATTRIBUTE
from .guidance import GuidanceExecutor
from .operations import module_compute_device
from .payloads import payload_binding_to_tensor
from .qwen_text import Flux2DevTextEncoder, Flux2KleinTextEncoder, QwenTextModel
from .sampling_execution import (
    CONTEXT_WINDOWS_UNSUPPORTED,
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionRegistration,
    SingleStreamLatentAdapter,
    sampling_execution,
)
from .sampling_runtime import SingleStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry

_TARGET_TEXT_ROWS = 512
FLUX2_REFERENCE_LATENTS_KEY = "dinkster-model-flux2/reference-latents"


class Flux2RuntimeError(ValueError):
    """A Flux2 runtime request violates its native contract."""


@dataclass(frozen=True)
class Flux2Conditioning(Conditioning[torch.Tensor]):
    reference_latents: tuple[torch.Tensor, ...] = ()


def materialize_flux2_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> Flux2Conditioning:
    """Materialize basic Flux2 text plus its ordered edit references."""

    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise Flux2RuntimeError("Flux2 conditioning requires one record")
    record = records[0]
    metadata = dict(record.extension_metadata)
    unknown = set(metadata) - {FLUX2_REFERENCE_LATENTS_KEY}
    if unknown:
        raise Flux2RuntimeError("Flux2 conditioning contains unknown extension metadata")
    reference_value = metadata.get(FLUX2_REFERENCE_LATENTS_KEY, ())
    if not isinstance(reference_value, tuple) or any(
        type(item) is not PayloadReference for item in reference_value
    ):
        raise Flux2RuntimeError("Flux2 references metadata must be an ordered payload tuple")
    channel_ids = {descriptor.reference.id for _, descriptor in record.channels}
    if record.mask is not None:
        channel_ids.add(record.mask.payload.id)
    if record.scale_vector is not None:
        channel_ids.add(record.scale_vector.values.reference.id)
    stripped = make_conditioning_carrier(
        ConditioningSet((replace(record, extension_metadata=()),)),
        tuple(binding for binding in carrier.bindings if binding.reference_id in channel_ids),
    )
    text = materialize_basic_conditioning(stripped, device=device)
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    references: list[torch.Tensor] = []
    for item in reference_value:
        reference = cast("PayloadReference", item)
        binding = bindings[reference.id]
        if binding.space != "flux2-reference-latent":
            raise Flux2RuntimeError("Flux2 reference payloads use the wrong space")
        tensor = payload_binding_to_tensor(binding).to(device)
        if (
            tensor.ndim != 4
            or tensor.shape[0] < 1
            or tensor.shape[1] != FLUX2_LATENT_CHANNELS
            or tensor.shape[2] < 1
            or tensor.shape[3] < 1
            or not tensor.is_floating_point()
        ):
            raise Flux2RuntimeError(
                f"Flux2 references must be floating [batch,{FLUX2_LATENT_CHANNELS},height,width]"
            )
        references.append(tensor)
    return Flux2Conditioning(text.embeddings, text.pooled, tuple(references))


class Flux2Denoiser(FluxDenoiser):
    """FluxDenoiser with the reference Flux2 512-row context floor.

    model_base.Flux2.extra_conds @ b78cec87 left-pads the text context
    with zero rows up to 512 before the transformer call; longer
    contexts pass through unchanged.
    """

    evaluator_identity = "dinkster.flux2.conditioning.v1"

    def prepare_conditioning(
        self,
        conditioning: object,
        _role: GuidanceRole = GuidanceRole.CONDITIONAL,
    ) -> FluxCondition:
        references: tuple[torch.Tensor, ...] = ()
        if type(conditioning) is Flux2Conditioning:
            references = conditioning.reference_latents
        context, pooled, _ = _flux_condition_parts(super().prepare_conditioning(conditioning))
        width = self.model.config.context_in_dim
        if context.shape[0] < 1 or context.shape[2] != width:
            raise Flux2RuntimeError(
                f"Flux2 conditioning must be [batch x tokens x {width}],"
                f" got shape {tuple(context.shape)}"
            )
        rows = context.shape[1]
        if rows < _TARGET_TEXT_ROWS:
            context = torch.nn.functional.pad(context, (0, 0, _TARGET_TEXT_ROWS - rows, 0))
        return (context, pooled, references) if references else (context, pooled)


def _flux2_sigma_space(family: ModelFamily, sampling_shift: float | None = None) -> FluxFlowSigmas:
    """The family's reference model-sampling space @ b78cec87:
    ModelSamplingFlux with the registered shift."""
    if sampling_shift is not None and (
        type(sampling_shift) is not float
        or not math.isfinite(sampling_shift)
        or sampling_shift <= 0.0
    ):
        raise Flux2RuntimeError("sampling_shift must be a positive finite float")
    return FluxFlowSigmas(shift=family.sampling.shift if sampling_shift is None else sampling_shift)


def _exact_scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


def _validate_flux2_latent(latent: torch.Tensor) -> None:
    if latent.ndim != 4 or latent.shape[1] != FLUX2_LATENT_CHANNELS:
        raise Flux2RuntimeError(
            f"Flux2 input must have shape [batch,{FLUX2_LATENT_CHANNELS},height,width]"
        )


def _flux2_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("Flux2Runtime | Flux2DiffusionRuntime", runtime)
    if context.options:
        names = ", ".join(sorted(context.options))
        raise Flux2RuntimeError(f"Flux2 sampling does not accept adapter options: {names}")
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            Flux2Denoiser(
                owner.assembled.diffusion,
                guidance=context.guidance,
                compute_dtype=compute_dtype,
            ),
        )
    )


def _flux2_device(runtime: object) -> torch.device:
    owner = cast("Flux2Runtime | Flux2DiffusionRuntime", runtime)
    return module_compute_device(owner.assembled.diffusion)


def _flux2_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("Flux2Runtime | Flux2DiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


_FLUX2_SAMPLING_EXECUTION = SamplingExecutionRegistration(
    latent=SingleStreamLatentAdapter(_validate_flux2_latent),
    denoiser=_flux2_denoiser,
    device=_flux2_device,
    compute_dtype=_flux2_compute_dtype,
    flow=True,
    capabilities=CONTEXT_WINDOWS_UNSUPPORTED,
    forbidden_options=frozenset({"_compute_dtype", "_device"}),
    forbidden_options_message="Flux2 KSampler does not accept private compute placement arguments",
)


class Flux2Runtime(SingleStreamSamplingRuntime):
    """Family runtime for the published Flux2 releases.

    ``encode_text`` runs the family's reference encoder (Mistral3-Small
    for dev, Qwen3 for Klein) into stacked sequence conditioning;
    ``sample`` composes KSampler inputs into ``sample_custom``;
    ``sample_custom`` executes an exact pre-computed sigma sequence over
    the family's exponential flow space, with an optional per-run
    ``sampling_shift`` override for empirical-mu schedules; the codec is
    the packed batch-norm KL VAE.
    """

    streamed_residency_components = frozenset({"text_encoder"})
    sampling_error = Flux2RuntimeError
    supports_sampling_shift = True
    supports_denoised_capture = True
    sampling_execution_registration = _FLUX2_SAMPLING_EXECUTION

    def __init__(
        self,
        assembled: AssembledFlux2,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._runtime_identity = runtime_identity
        self.codec: CodecPlugin = replace(
            kl_codec_plugin(assembled.vae),
            compute_dtype=assembled.compute_dtype("vae"),
        )
        architecture = assembled.text_encoder.config.architecture
        self._text_encoder: Flux2DevTextEncoder | Flux2KleinTextEncoder
        if architecture in ("mistral3_24b", "mistral3_24b_pruned"):
            self._text_encoder = Flux2DevTextEncoder(
                assembled.text_encoder, load_flux2_tekken_bpe()
            )
        else:
            self._text_encoder = Flux2KleinTextEncoder(assembled.text_encoder)
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

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FluxFlowSigmas:
        return _flux2_sigma_space(self.family, sampling_shift)

    @property
    def supports_distilled_guidance(self) -> bool:
        return self.assembled.diffusion.guidance_in is not None

    sample_custom = sampling_execution

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)


class Flux2TextRuntime:
    """Text-only Flux2 encoder over an independently resident component."""

    def __init__(self, text: QwenTextModel) -> None:
        architecture = text.config.architecture
        self._encoder: Flux2DevTextEncoder | Flux2KleinTextEncoder
        if architecture in ("mistral3_24b", "mistral3_24b_pruned"):
            tokenizer = text.__dict__.get(FLUX2_TEKKEN_ATTRIBUTE)
            if not isinstance(tokenizer, TekkenBpe):
                raise Flux2RuntimeError("Flux2 dev text component carries no tekken tokenizer")
            self._encoder = Flux2DevTextEncoder(text, tokenizer)
        elif architecture in ("klein_qwen3_4b", "klein_qwen3_8b"):
            self._encoder = Flux2KleinTextEncoder(text)
        else:
            raise Flux2RuntimeError(f"not a Flux2 text profile: {architecture!r}")
        self.text = text

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        return self._encoder.encode(text)


@dataclass(frozen=True)
class _Flux2DiffusionAssembly:
    diffusion: Flux
    family: ModelFamily
    # Residency enrollment reads these; the component is strict-loaded at
    # its compute dtype already, so storage never follows a separate target.
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, role: str) -> torch.dtype | None:
        return next(self.diffusion.parameters()).dtype if role == "diffusion" else None


class Flux2DiffusionRuntime(SingleStreamSamplingRuntime):
    """Diffusion-only Flux2 sampling facade over an independently
    resident component. Compute dtype and device resolve from the
    loaded module rather than per-call arguments."""

    sampling_error = Flux2RuntimeError
    supports_sampling_shift = True
    supports_denoised_capture = True
    sampling_execution_registration = _FLUX2_SAMPLING_EXECUTION

    def __init__(
        self,
        diffusion: Flux,
        family: ModelFamily,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = _Flux2DiffusionAssembly(diffusion, family)
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

    @property
    def conditioning_identity(self) -> str:
        config = self.assembled.diffusion.config
        return (
            "dinkster.flux2.conditioning:v1:"
            f"{self.family.id}:{config.context_in_dim}:{config.hidden_size}:"
            f"{config.depth}:{config.depth_single_blocks}:{config.guidance_embed}"
        )

    def prepare_flux2_conditioning(self, carrier: ConditioningCarrier) -> Flux2Conditioning:
        device = module_compute_device(self.assembled.diffusion)
        return materialize_flux2_conditioning(carrier, device=device)

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[torch.Tensor]:
        return self.prepare_flux2_conditioning(carrier)

    # The FamilyRuntime members a diffusion-only component cannot serve
    # refuse explicitly; text and VAE ride their own component handles.
    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise Flux2RuntimeError("Flux2 diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise Flux2RuntimeError("Flux2 diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise Flux2RuntimeError("Flux2 diffusion component carries no VAE codec")

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FluxFlowSigmas:
        return _flux2_sigma_space(self.family, sampling_shift)

    supports_distilled_guidance = Flux2Runtime.supports_distilled_guidance  # pyright: ignore[reportIncompatibleMethodOverride]

    sample_custom = sampling_execution


__all__ = [
    "FLUX2_REFERENCE_LATENTS_KEY",
    "Flux2Conditioning",
    "Flux2Denoiser",
    "Flux2DiffusionRuntime",
    "Flux2Runtime",
    "Flux2RuntimeError",
    "Flux2TextRuntime",
    "materialize_flux2_conditioning",
]
