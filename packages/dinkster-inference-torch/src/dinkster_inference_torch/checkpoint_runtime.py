"""Checkpoint companions around an unchanged component sampling runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    Conditioning,
    ConditioningCarrier,
    ModelFamily,
    build_runtime_identity,
)
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
from dinkster_inference.component_registry import build_component_runtime, execution_symbol

from .operations import module_compute_device
from .sampling_runtime import SamplingRuntime


@dataclass(frozen=True)
class ComponentAssembly:
    family: ModelFamily
    model_role: str
    components: Mapping[str, torch.nn.Module]
    component_dtypes: Mapping[str, torch.dtype]
    attention_status: Mapping[str, object]
    _storage_dtype_follows_compute: bool = False

    def __post_init__(self) -> None:
        components = dict(self.components)
        dtypes = dict(self.component_dtypes)
        if self.model_role not in components:
            raise ValueError(f"checkpoint has no declared model role {self.model_role!r}")
        if components.keys() != dtypes.keys():
            raise ValueError("checkpoint compute dtypes must name every realized component")
        if len({id(module) for module in components.values()}) != len(components):
            raise ValueError("checkpoint roles must not share a module instance")
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(self, "component_dtypes", MappingProxyType(dtypes))
        object.__setattr__(self, "attention_status", MappingProxyType(dict(self.attention_status)))

    @property
    def diffusion(self) -> torch.nn.Module:
        return self.components[self.model_role]

    def compute_dtype(self, role: str) -> torch.dtype | None:
        return self.component_dtypes.get(self.model_role if role == "diffusion" else role)


class ComponentCheckpointRuntime(SamplingRuntime):
    """Own companions while sampling stays bound to the original diffusion runtime."""

    preserves_text_conditioning = True

    def __init__(
        self,
        diffusion: Any,
        assembled: ComponentAssembly,
        *,
        text_runtime: Any | None = None,
        codec: Any | None = None,
        prepare_conditioning: str = "materialize_basic_conditioning",
    ) -> None:
        if assembled.diffusion is not diffusion.assembled.diffusion:
            raise ValueError("checkpoint and diffusion runtime must share the same model module")
        if assembled.family != diffusion.family:
            raise ValueError("checkpoint and diffusion runtime disagree about architecture")
        if assembled.compute_dtype("diffusion") != diffusion.assembled.compute_dtype("diffusion"):
            raise ValueError("checkpoint and diffusion runtime disagree about compute dtype")
        self.diffusion = diffusion
        self.assembled = assembled
        self.attention_status = assembled.attention_status
        self._text_runtime = text_runtime
        self._codec = codec
        self._prepare_conditioning = prepare_conditioning
        # Concrete bound methods remain visible to Python's runtime protocol checks.
        if callable(getattr(diffusion, "sample", None)):
            self.sample = diffusion.sample
        if callable(getattr(diffusion, "sample_multistream", None)):
            self.sample_multistream = diffusion.sample_multistream
        if callable(getattr(diffusion, "run_ksampler_as_custom", None)):
            self.run_ksampler_as_custom = diffusion.run_ksampler_as_custom
        if callable(getattr(diffusion, "adapt_multistream_latent", None)):
            self.adapt_multistream_latent = diffusion.adapt_multistream_latent
        if callable(getattr(diffusion, "prepare_conditioning", None)):
            self.prepare_conditioning = diffusion.prepare_conditioning
            self.prepare_text_conditioning = self._prepare_text_conditioning
        self.sample_custom = diffusion.sample_custom
        self.check_custom_sampling = diffusion.check_custom_sampling
        self.custom_sampling_sigmas = diffusion.custom_sampling_sigmas
        self.custom_sampling_beta_sigmas = diffusion.custom_sampling_beta_sigmas
        self.custom_sampling_sd_turbo_sigmas = diffusion.custom_sampling_sd_turbo_sigmas
        self.custom_sampling_percent_to_sigma = diffusion.custom_sampling_percent_to_sigma

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self.diffusion.runtime_identity

    @property
    def conditioning_identity(self) -> str:
        return getattr(self.diffusion, "conditioning_identity", self.runtime_identity)

    def sampling_runtime(self) -> SamplingRuntime:
        return self.diffusion

    def with_sampling_runtime(self, diffusion: Any) -> ComponentCheckpointRuntime:
        return ComponentCheckpointRuntime(
            diffusion,
            self.assembled,
            text_runtime=self._text_runtime,
            codec=self._codec,
            prepare_conditioning=self._prepare_conditioning,
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> Any:
        return self.diffusion.sampling_sigma_space(sampling_shift)

    @property
    def text_encode_options(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
    ) -> frozenset[str]:
        return self.diffusion.text_encode_options

    @property
    def supports_denoised_capture(self) -> bool:
        return getattr(self.diffusion, "supports_denoised_capture", False)

    @property
    def supports_sampling_shift(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
    ) -> bool:
        return getattr(self.diffusion, "supports_sampling_shift", False)

    @property
    def supports_denoise_mask(self) -> bool:
        return getattr(self.diffusion, "supports_denoise_mask", False)

    @property
    def supports_context_windows(self) -> bool:
        return getattr(self.diffusion, "supports_context_windows", False)

    @property
    def supports_inpaint(self) -> bool:
        return getattr(self.diffusion, "supports_inpaint", False)

    @property
    def supports_distilled_guidance(self) -> bool:
        return getattr(self.diffusion, "supports_distilled_guidance", False)

    @property
    def streamed_residency_components(self) -> frozenset[str]:
        return getattr(self.diffusion, "streamed_residency_components", frozenset())

    @property
    def codec(self) -> Any:
        if self._codec is None:
            raise ValueError(
                "checkpoint has no codec binding; "
                f"detected roles={tuple(self.assembled.components)!r}"
            )
        return self._codec

    def encode_text(self, text: str) -> Any:
        if self._text_runtime is None:
            raise ValueError(
                "checkpoint has no text binding; "
                f"detected roles={tuple(self.assembled.components)!r}"
            )
        return self._text_runtime.encode_text(text)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content)

    def text_conditioning_carrier(self, value: Conditioning[torch.Tensor]) -> ConditioningCarrier:
        from .conditioning_adapters import basic_conditioning_to_carrier

        encode = getattr(
            self._text_runtime, "text_conditioning_carrier", basic_conditioning_to_carrier
        )
        return encode(value)

    def _prepare_text_conditioning(self, value: Conditioning[torch.Tensor]) -> Any:
        return self.prepare_conditioning(self.text_conditioning_carrier(value))

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[torch.Tensor]:
        prepare = getattr(self.diffusion, self._prepare_conditioning, None)
        if prepare is not None:
            return prepare(carrier)
        device = module_compute_device(self.assembled.diffusion)
        return execution_symbol(self._prepare_conditioning)(carrier, device=device)


def assemble_component_checkpoint(
    plan: ComponentCheckpointPlan,
    *,
    diffusion_dtype: torch.dtype,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    storage_dtype_follows_compute: bool = False,
    fp8_matmul: bool = False,
    sampler_registry: Any = None,
    scheduler_registry: Any = None,
    registry_token: str | None = None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    guidance_executor: Any = None,
    embedding_lookups: Any = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    pose_cache_settings: Any = None,
) -> ComponentCheckpointRuntime:
    """Realize named plans once and bind companions through declared behavior."""
    from .assemble import _select_attention_runtime  # pyright: ignore[reportPrivateUsage]
    from .wiring import _identity_dtype  # pyright: ignore[reportPrivateUsage]

    descriptor = plan.descriptor
    if descriptor.component_realizer is None:
        raise ValueError(f"{descriptor.id} has no component realization contract")
    if pose_cache_settings is not None:
        raise ValueError("checkpoint composition has no pose-cache binding")
    if embedding_lookups is not None or embedding_binding_digest is not None:
        raise ValueError("checkpoint composition has no textual-inversion binding")
    realizer = execution_symbol(descriptor.component_realizer)
    kernels, attention_status = _select_attention_runtime(attention_policy, attention_route_token)
    modules: dict[str, torch.nn.Module] = {}
    dtypes: dict[str, torch.dtype] = {}
    for role, component in plan.components.items():
        dtype = (
            text_dtype
            if role in descriptor.text_encoder_roles
            else vae_dtype
            if role in descriptor.codec_roles
            else diffusion_dtype
        )
        modules[role] = realizer(
            component,
            compute_dtype=dtype,
            fp8_matmul=fp8_matmul,
            attention_kernels=kernels,
        )
        dtypes[role] = dtype
    assembled = ComponentAssembly(
        plan.family,
        descriptor.model_role,
        modules,
        dtypes,
        {role: status for role, status in attention_status.items()},
        _storage_dtype_follows_compute=storage_dtype_follows_compute,
    )
    identity = build_runtime_identity(
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
    )
    diffusion = build_component_runtime(
        descriptor,
        SimpleNamespace(
            module=assembled.diffusion,
            plan=plan.components[descriptor.model_role],
            attention_status=attention_status,
        ),
        identity,
        diffusion_dtype,
        sampler_registry=sampler_registry,
        scheduler_registry=scheduler_registry,
        **({} if guidance_executor is None else {"guidance_executor": guidance_executor}),
    )
    text = (
        None
        if descriptor.checkpoint_text_factory is None
        else execution_symbol(descriptor.checkpoint_text_factory)(assembled)
    )
    codec = (
        None
        if descriptor.checkpoint_codec_factory is None
        else execution_symbol(descriptor.checkpoint_codec_factory)(assembled)
    )
    return ComponentCheckpointRuntime(
        diffusion,
        assembled,
        text_runtime=text,
        codec=codec,
        prepare_conditioning=descriptor.prepare_conditioning,
    )
