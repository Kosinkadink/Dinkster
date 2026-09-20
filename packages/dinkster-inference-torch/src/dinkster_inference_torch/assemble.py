"""Executing a classic-Flux assembly plan: planned slices -> modules.

The torch-free planner (dinkster_inference.assembly) turned checkpoint
HEADERS into a :class:`~dinkster_inference.assembly.FluxAssemblyPlan`:
per component, which source keys feed which model keys, at what
storage dtype, with what per-layer quantization. This module is the
executing half - it reads exactly the planned payload slices
(sources.load_tensors) and produces live native modules. Nothing here
re-detects or re-guesses; a plan/checkpoint mismatch is a loud
:class:`AssembleError`.

Loading semantics, pinned:

- Per-parameter storage dtypes are PRESERVED (``load_state_dict``
  with ``assign=True``, the house pattern): a mixed-dtype checkpoint
  loads exactly as shipped, no model-wide cast. Each component
  constructs with :data:`~.operations.INITLESS` when every
  non-quantized tensor already matches its compute dtype, and with
  :class:`~.operations.CastOperations` (cast-at-use, the reference's
  manual_cast route) otherwise.
- Scale-quantized layers (the plan's ``quant`` maps) are swapped to
  :class:`~.quant_linear.Fp8Linear` BEFORE loading: fp8 qdata,
  weight/input scales, and the full-precision-matmul pin load into
  registered storage. Layers whose format lives in payload bytes
  (the per-layer ``.comfy_quant`` JSON spelling) are resolved here,
  where payloads are finally readable.
- Plain-fp8 checkpoints (fp8 storage, no scales) are not quantization:
  their fp8 dtypes simply differ from the compute dtype, so the
  component lands on CastOperations and dequantizes at use - the
  reference's manual_cast route for flux1-dev-fp8-style files.
- ``absent`` keys default-fill: ``text_projection.weight`` becomes
  the identity, a deliberate deviation from the reference (which
  loads strict=False and leaves the constructed empty tensor) -
  deterministic, and unconsumed under the Flux CLIP-L policy either
  way.

Compute dtypes are caller policy with reference-shaped defaults
(bf16 DiT, fp32 text encoders, fp32 VAE - what the reference picks on
a modern CUDA host with T5's fp16-overflow pin honored). Device
placement and residency stay with their own layers: modules assemble
on CPU and move with ``.to()`` / the residency mechanism.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import BinaryIO, TypeVar, cast

import torch
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    resolve_attention_runtime_status,
    resolve_role_policy,
)
from dinkster_inference.assembly import (
    ComponentPlan,
    ControlNetAssemblyPlan,
    Flux2AssemblyPlan,
    FluxAssemblyPlan,
    Lumina2AssemblyPlan,
    QwenImageAssemblyPlan,
    QwenImageControlPlan,
    QwenImageDiffSynthPlan,
    SD15IPAdapterAssemblyPlan,
    SDAssemblyPlan,
    SDXLControlLoRAAssemblyPlan,
    SDXLControlNetAssemblyPlan,
    SDXLControlNetUnionAssemblyPlan,
    T2IAdapterAssemblyPlan,
    TAESDCodecPlan,
    Wan21AssemblyPlan,
    Wan21MultiTalkPlan,
    Wan21Uni3CPlan,
    ZImageAssemblyPlan,
    ZImageControlPlan,
)
from dinkster_inference.families import ModelFamily
from dinkster_inference.gguf import GGUFWeightSource
from dinkster_inference.quantization import LayerQuant
from dinkster_inference.qwen_image_control import QwenImageControlKind, QwenImageDiffSynthKind
from dinkster_inference.qwen_image_text import QWEN_IMAGE_TEXT_CONFIG, QwenImageTextConfig
from dinkster_inference.sampling import SamplingDescriptor
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header
from dinkster_inference.t5_text import load_umt5_spiece
from dinkster_inference.taehv import TAEHVConfig
from dinkster_inference.taesd import TAESDConfig
from dinkster_inference.unet import SDXL_UNET_CONFIG
from dinkster_inference.wan21_vae import Wan21VAEConfig
from dinkster_inference.wan22_vae import Wan22VAEConfig
from dinkster_inference.weights import LinearToConv2D, RowChunk, TensorTransform

from ._nvfp4_diagnostics import Nvfp4DiagnosticsRecorder
from .attention import (
    AttentionKernel,
    AttentionRole,
    AttentionSelectionError,
    AttentionStatus,
    schedule_aware_attention_kernel,
    select_attention,
    unauthenticated_auto_attention,
    verify_attention_route_evidence,
)
from .autoencoder_kl import AutoencoderKL
from .clip_text import ClipTextModel
from .clip_vision import SD15IPAdapterClipVisionEncoder, Wan21ClipVisionEncoder
from .controlnet import (
    SD15ControlNet,
    SDXLControlLoRA,
    SDXLControlNet,
    SDXLControlNetUnion,
    _bind_sd15_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_base_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_control_lora_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_union_resource,  # pyright: ignore[reportPrivateUsage]
    _validate_sdxl_base_resource,  # pyright: ignore[reportPrivateUsage]
    sd15_controlnet_resource_digest,
    sdxl_control_lora_resource_digest,
    sdxl_controlnet_resource_digest,
    sdxl_controlnet_union_resource_digest,
)
from .flux import Flux
from .gemma_text import GemmaTextModel
from .gemma_tokenizer import LUMINA2_TOKENIZER_ATTRIBUTE, LUMINA2_TOKENIZER_BYTE_CAP
from .gguf_linear import GGUF_BLOCK_SHAPES, GgufDecodedCache, GgufEncodedLinear
from .ipadapter import (
    SD15IPAdapter,
    _bind_sd15_ipadapter_resource,  # pyright: ignore[reportPrivateUsage]
    sd15_ipadapter_resource_digest,
)
from .operations import (
    INITLESS,
    CastOperations,
    Operations,
    bind_fp8_matmul_layer,
    bound_compute_dtype,
)
from .quant_linear import Fp8Linear, Int8Embedding, Int8Linear, Nvfp4Linear
from .qwen_image import QwenImage
from .qwen_image_control import (
    QwenImageControlModel,
    QwenImageDiffSynthPatch,
    QwenImageFunControlNet,
    QwenImageInstantXControlNet,
    _bind_qwen_image_control_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_qwen_image_diffsynth_resource,  # pyright: ignore[reportPrivateUsage]
    qwen_image_control_resource_digest,
    qwen_image_diffsynth_resource_digest,
    validate_qwen_image_control_resource,
    validate_qwen_image_diffsynth_resource,
)
from .qwen_image_text import QwenImageTextModel
from .qwen_text import QwenTextModel
from .sources import (
    FP8_QUANT_DTYPES,
    load_gguf_encoded_blocks,
    load_gguf_tensors,
    load_tensors,
    load_tensors_from_file,
)
from .t2i_adapter import (
    SD15T2IAdapter,
    _bind_sd15_t2i_adapter_resource,  # pyright: ignore[reportPrivateUsage]
    sd15_t2i_adapter_resource_digest,
)
from .t5_text import T5TextModel
from .taehv import TAEHVDecoder
from .taesd import TAESD, TAESDDecoder, TAESDEncoder
from .unet import UNetModel
from .wan21_animate2 import WanAnimate2Model
from .wan21_causal import Wan21CausalModel
from .wan21_humo import Wan21HumoModel
from .wan21_model import Wan21Model
from .wan21_multitalk import (
    Wan21MultiTalk,
    _bind_wan21_multitalk_resource,  # pyright: ignore[reportPrivateUsage]
    validate_wan21_multitalk_resource,
    wan21_multitalk_resource_digest,
)
from .wan21_scail import WanScailModel
from .wan21_uni3c import (
    Wan21Uni3C,
    _bind_wan21_uni3c_resource,  # pyright: ignore[reportPrivateUsage]
    validate_wan21_uni3c_resource,
    wan21_uni3c_resource_digest,
)
from .wan21_vae import WanVAE
from .wan21_vae import WanVAEConfig as TorchWanVAEConfig
from .wan22_dancer import Wan22DancerModel
from .wan22_s2v import Wan22S2VModel
from .wan22_vae import Wan22VAE
from .z_image import ZImage, ZImagePixelCodec, ZImagePixelSpace
from .z_image_control import (
    ZImageControl,
    _bind_z_image_control_resource,  # pyright: ignore[reportPrivateUsage]
    validate_z_image_control_resource,
    z_image_control_resource_digest,
)

__all__ = [
    "AssembledControlNet",
    "AssembledSDXLControlLoRA",
    "AssembledSDXLControlNet",
    "AssembledSDXLControlNetUnion",
    "AssembledT2IAdapter",
    "AssembledFlux",
    "AssembledFlux2",
    "AssembledLumina2",
    "AssembledQwenImage",
    "AssembledQwenImageControl",
    "AssembledQwenImageDiffSynth",
    "AssembledSD",
    "AssembledSD15IPAdapter",
    "AssembledWan21",
    "AssembledWan21MultiTalk",
    "AssembledWan21Uni3C",
    "AssembledZImage",
    "AssembledZImageControl",
    "AssembleError",
    "assemble_sd15_controlnet",
    "assemble_sd15_t2i_adapter",
    "assemble_sdxl_control_lora",
    "assemble_sdxl_controlnet",
    "assemble_sdxl_controlnet_union",
    "assemble_flux",
    "assemble_flux2",
    "assemble_lumina2",
    "assemble_qwen_image",
    "assemble_qwen_image_control",
    "assemble_qwen_image_diffsynth",
    "assemble_sd",
    "assemble_sd15_ipadapter",
    "assemble_wan21",
    "assemble_wan21_multitalk",
    "assemble_wan21_uni3c",
    "assemble_z_image",
    "assemble_z_image_control",
]


C = TypeVar("C")
M = TypeVar("M", bound=torch.nn.Module)
log = logging.getLogger("dinkster.inference_torch.assemble")


class AssembleError(ValueError):
    """A plan this executor cannot realize against the actual payload:
    an unported quantization format surfacing from payload-borne
    config, artifact tensors that contradict their declaration, or an
    absent key with no known default."""


_ATTENTION_ROLES: tuple[AttentionRole, ...] = (
    "unet",
    "flux",
    "vae",
    "clip",
    "t5",
    "qwen",
)


def _default_attention_statuses() -> Mapping[AttentionRole, AttentionStatus]:
    return MappingProxyType(
        {role: unauthenticated_auto_attention(role).status for role in _ATTENTION_ROLES}
    )


def _select_attention_runtime(
    policy: AttentionPolicy,
    token: AttentionRouteToken | None,
) -> tuple[
    Mapping[AttentionRole, AttentionKernel],
    Mapping[AttentionRole, AttentionStatus],
]:
    resolved = resolve_attention_runtime_status(policy, token)
    try:
        verify_attention_route_evidence(resolved, token)
    except AttentionSelectionError as error:
        raise AssembleError(str(error)) from None
    routes = {route.role: route for route in resolved.routes}
    kernels: dict[AttentionRole, AttentionKernel] = {}
    statuses: dict[AttentionRole, AttentionStatus] = {}
    for role in _ATTENTION_ROLES:
        effective_policy = resolve_role_policy(
            resolved.requested_policy, resolved.requested_role_policies, role
        )
        selection = (
            unauthenticated_auto_attention(role)
            if token is None and effective_policy == "auto"
            else select_attention(role, effective_policy)
        )
        status = selection.status
        route = routes[role]
        if (
            status.role,
            status.requested_policy,
            status.primary,
            status.fallback,
        ) != (
            route.role,
            effective_policy,
            route.primary,
            route.fallback,
        ):
            raise AssembleError(
                f"attention selection for role {role!r} does not match "
                "the authenticated runtime route"
            )
        kernels[role] = (
            schedule_aware_attention_kernel(status.primary, selection.kernel)
            if role in ("unet", "flux")
            else selection.kernel
        )
        statuses[role] = AttentionStatus(
            requested_policy=status.requested_policy,
            role=status.role,
            primary=status.primary,
            fallback=status.fallback,
            reason=status.reason,
            authenticated=resolved.authenticated,
            provider_versions=resolved.provider_versions,
            adapter_contract=(
                resolved.adapter_contract_revision
                if resolved.authenticated
                else status.adapter_contract
            ),
            device_kind=resolved.device_kind,
            device_sm=resolved.device_sm,
            sdpa_torch_runtime=resolved.sdpa_torch_runtime,
        )
    return MappingProxyType(kernels), MappingProxyType(statuses)


@dataclass(frozen=True)
class AssembledFlux2:
    """Loaded Flux2 diffusion, family text encoder (Mistral3-Small for
    dev, Qwen3 for Klein), and the packed batch-norm KL VAE."""

    family: ModelFamily
    diffusion: Flux
    text_encoder: QwenTextModel
    vae: AutoencoderKL
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(
            self,
            "attention_status",
            MappingProxyType(dict(self.attention_status)),
        )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledFlux:
    """The live result of one Flux plan: the family descriptor plus
    loaded diffusion, text, and codec modules at their planned mixed
    storage dtypes."""

    family: ModelFamily
    diffusion: Flux
    clip_l: ClipTextModel | None
    t5xxl: T5TextModel | None
    vae: AutoencoderKL
    qwen3_2b: QwenTextModel | None = None
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(
            self,
            "attention_status",
            MappingProxyType(dict(self.attention_status)),
        )
        if self.diffusion.config.vec_in_dim is None and self.qwen3_2b is None:
            raise ValueError("assembled vector-free Ovis Flux requires qwen3_2b text")
        classic_complete = self.clip_l is not None and self.t5xxl is not None
        if self.qwen3_2b is None and not classic_complete:
            raise ValueError("assembled classic Flux requires clip_l and t5xxl")
        if self.qwen3_2b is not None and not (self.clip_l is None and self.t5xxl is None):
            raise ValueError("assembled Qwen Flux cannot also carry classic text models")
        nvfp4_layers = tuple(
            module for module in self.diffusion.modules() if isinstance(module, Nvfp4Linear)
        )
        if nvfp4_layers:
            recorders = {
                module._diagnostics  # pyright: ignore[reportPrivateUsage]
                for module in nvfp4_layers
                if module._diagnostics is not None  # pyright: ignore[reportPrivateUsage]
            }
            attached = getattr(self.diffusion, "_nvfp4_diagnostics", None)
            if isinstance(attached, Nvfp4DiagnosticsRecorder):
                recorders.add(attached)
            if len(recorders) > 1:
                raise RuntimeError("assembled NVFP4 layers use different diagnostics recorders")
            recorder = next(iter(recorders)) if recorders else Nvfp4DiagnosticsRecorder()
            for module in nvfp4_layers:
                module._bind_diagnostics(recorder)  # pyright: ignore[reportPrivateUsage]
            object.__setattr__(self.diffusion, "_nvfp4_diagnostics", recorder)

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledSD:
    """The live result of one SD-era plan: family descriptor plus one
    loaded module per component the family wires (SD 1.5 has no
    CLIP-G, the refiner no CLIP-L - mirroring SDAssemblyPlan)."""

    family: ModelFamily
    diffusion: UNetModel
    clip_l: ClipTextModel | None
    clip_g: ClipTextModel | None
    vae: AutoencoderKL | TAESD
    sampling: SamplingDescriptor | None = None
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(
            self,
            "attention_status",
            MappingProxyType(dict(self.attention_status)),
        )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledWan21:
    """Loaded Wan core diffusion, UMT5, tokenizer, and VAE."""

    family: ModelFamily
    diffusion: Wan21Model
    umt5xxl: T5TextModel
    vae: WanVAE | Wan22VAE
    tokenizer_model: bytes = field(repr=False)
    clip_vision: Wan21ClipVisionEncoder | None = field(default=None, kw_only=True)
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(self, "attention_status", MappingProxyType(dict(self.attention_status)))
        if not self.tokenizer_model:
            raise ValueError("assembled Wan runtime requires tokenizer model bytes")
        model_type = self.diffusion.config.model_type
        if (model_type == "i2v") != (self.clip_vision is not None):
            raise ValueError("assembled Wan 2.1 I2V requires CLIP vision and T2V must not carry it")

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledZImage:
    family: ModelFamily
    diffusion: ZImage | ZImagePixelSpace
    qwen3_4b: QwenTextModel
    vae: AutoencoderKL | ZImagePixelCodec
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(
            self,
            "attention_status",
            MappingProxyType(dict(self.attention_status)),
        )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledLumina2:
    family: ModelFamily
    diffusion: ZImage
    gemma2_2b: GemmaTextModel
    vae: AutoencoderKL
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(self, "attention_status", MappingProxyType(dict(self.attention_status)))

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledQwenImage:
    """Loaded Qwen Image diffusion, Qwen2.5-VL-7B, and Wan VAE."""

    diffusion: QwenImage
    text: QwenImageTextModel
    vae: WanVAE
    family: ModelFamily | None = None
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=_default_attention_statuses
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_component_compute_dtypes",
            MappingProxyType(dict(self._component_compute_dtypes)),
        )
        object.__setattr__(
            self,
            "attention_status",
            MappingProxyType(dict(self.attention_status)),
        )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self._component_compute_dtypes.get(component)


@dataclass(frozen=True)
class AssembledZImageControl:
    control: ZImageControl
    attention_status: AttentionStatus
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if type(self.control) is not ZImageControl or self.control.resource_digest is None:
            raise ValueError("assembled Z-Image control must carry sealed provenance")
        validate_z_image_control_resource(self.control, self.control.resource_digest)

    @property
    def resource_digest(self) -> str:
        digest = self.control.resource_digest
        assert digest is not None
        validate_z_image_control_resource(self.control, digest)
        return digest


@dataclass(frozen=True)
class AssembledQwenImageControl:
    control: QwenImageControlModel
    kind: QwenImageControlKind
    attention_status: AttentionStatus
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        expected_type = (
            QwenImageFunControlNet if self.kind == "fun" else QwenImageInstantXControlNet
        )
        if type(self.control) is not expected_type or self.control.resource_digest is None:
            raise ValueError("assembled Qwen Image control must match its kind and provenance")
        validate_qwen_image_control_resource(self.control, self.control.resource_digest)

    @property
    def resource_digest(self) -> str:
        digest = self.control.resource_digest
        assert digest is not None
        validate_qwen_image_control_resource(self.control, digest)
        return digest


@dataclass(frozen=True)
class AssembledQwenImageDiffSynth:
    patch: QwenImageDiffSynthPatch
    kind: QwenImageDiffSynthKind
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if type(self.patch) is not QwenImageDiffSynthPatch or self.patch.resource_digest is None:
            raise ValueError("assembled Qwen Image DiffSynth must carry sealed provenance")
        validate_qwen_image_diffsynth_resource(self.patch, self.patch.resource_digest)

    @property
    def resource_digest(self) -> str:
        digest = self.patch.resource_digest
        assert digest is not None
        validate_qwen_image_diffsynth_resource(self.patch, digest)
        return digest


@dataclass(frozen=True)
class AssembledWan21Uni3C:
    patch: Wan21Uni3C
    attention_status: AttentionStatus
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if type(self.patch) is not Wan21Uni3C or self.patch.resource_digest is None:
            raise ValueError("assembled Wan 2.1 Uni3C patch must carry sealed provenance")
        validate_wan21_uni3c_resource(self.patch, self.patch.resource_digest)

    @property
    def resource_digest(self) -> str:
        digest = self.patch.resource_digest
        assert digest is not None
        validate_wan21_uni3c_resource(self.patch, digest)
        return digest


@dataclass(frozen=True)
class AssembledWan21MultiTalk:
    patch: Wan21MultiTalk
    attention_status: AttentionStatus
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if type(self.patch) is not Wan21MultiTalk or self.patch.resource_digest is None:
            raise ValueError("assembled Wan 2.1 MultiTalk patch must carry sealed provenance")
        validate_wan21_multitalk_resource(self.patch, self.patch.resource_digest)

    @property
    def resource_digest(self) -> str:
        digest = self.patch.resource_digest
        assert digest is not None
        validate_wan21_multitalk_resource(self.patch, digest)
        return digest


@dataclass(frozen=True)
class AssembledControlNet:
    """A loaded classic SD1.5 ControlNet and its compute dtype."""

    controlnet: SD15ControlNet
    compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.controlnet), SD15ControlNet):
            raise TypeError("assembled controlnet must be an SD15ControlNet")
        if not self.compute_dtype.is_floating_point:
            raise TypeError("ControlNet compute dtype must be floating")
        if self.controlnet.resource_digest is None:
            raise ValueError("assembled ControlNet must carry its resource digest")

    @property
    def resource_digest(self) -> str:
        digest = self.controlnet.resource_digest
        assert digest is not None
        return digest


@dataclass(frozen=True)
class AssembledT2IAdapter:
    adapter: SD15T2IAdapter
    compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if type(self.adapter) is not SD15T2IAdapter or self.adapter.resource_digest is None:
            raise ValueError("assembled T2I Adapter must carry sealed provenance")

    @property
    def resource_digest(self) -> str:
        digest = self.adapter.resource_digest
        assert digest is not None
        return digest


@dataclass(frozen=True)
class AssembledSD15IPAdapter:
    adapter: SD15IPAdapter
    clip_vision: SD15IPAdapterClipVisionEncoder
    adapter_compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if type(self.adapter) is not SD15IPAdapter or self.adapter.resource_digest is None:
            raise ValueError("assembled IP-Adapter must carry sealed provenance")
        if type(self.clip_vision) is not SD15IPAdapterClipVisionEncoder:
            raise TypeError("assembled IP-Adapter requires its exact CLIP vision encoder")

    @property
    def resource_digest(self) -> str:
        digest = self.adapter.resource_digest
        assert digest is not None
        return digest


@dataclass(frozen=True)
class AssembledSDXLControlLoRA:
    control_lora: SDXLControlLoRA
    compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if (
            type(self.control_lora) is not SDXLControlLoRA
            or self.control_lora.resource_digest is None
        ):
            raise ValueError("assembled Control-LoRA must carry sealed provenance")

    @property
    def resource_digest(self) -> str:
        digest = self.control_lora.resource_digest
        assert digest is not None
        return digest


@dataclass(frozen=True)
class AssembledSDXLControlNet:
    controlnet: SDXLControlNet
    compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if type(self.controlnet) is not SDXLControlNet or self.controlnet.resource_digest is None:
            raise ValueError("assembled SDXL ControlNet must carry sealed provenance")

    @property
    def resource_digest(self) -> str:
        digest = self.controlnet.resource_digest
        assert digest is not None
        return digest


@dataclass(frozen=True)
class AssembledSDXLControlNetUnion:
    controlnet_union: SDXLControlNetUnion
    compute_dtype: torch.dtype

    def __post_init__(self) -> None:
        if (
            type(self.controlnet_union) is not SDXLControlNetUnion
            or self.controlnet_union.resource_digest is None
        ):
            raise ValueError("assembled SDXL ControlNet Union must carry sealed provenance")

    @property
    def resource_digest(self) -> str:
        digest = self.controlnet_union.resource_digest
        assert digest is not None
        return digest


def _build_taesd_encoder(_config: TAESDConfig, *, operations: Operations) -> TAESDEncoder:
    return TAESDEncoder(operations=operations)


def _build_taesd_decoder(_config: TAESDConfig, *, operations: Operations) -> TAESDDecoder:
    return TAESDDecoder(operations=operations)


def _build_taehv_decoder(config: TAEHVConfig, *, operations: Operations) -> TAEHVDecoder:
    return TAEHVDecoder(config, operations=operations)


@dataclass(frozen=True)
class _ResolvedQuant:
    """A LayerQuant with payload-borne facts resolved and format tagged."""

    format: str
    fp8_dtype: torch.dtype | None
    full_precision_matmul: bool
    convrot: bool = False
    convrot_groupsize: int = 256


def _decode_layer_config(component: str, layer: str, config: torch.Tensor) -> dict[str, object]:
    """The per-layer ``.comfy_quant`` JSON (what the reference's
    _load_quantized_module json.loads from the popped tensor
    @ b78cec87)."""
    if config.dtype != torch.uint8:
        raise AssembleError(
            f"{component}: layer {layer!r} config tensor is"
            f" {config.dtype}, expected uint8 JSON bytes"
        )
    try:
        decoded = json.loads(
            bytes(config.flatten().tolist()), object_pairs_hook=_unique_json_object
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise AssembleError(f"{component}: layer {layer!r} config is not JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise AssembleError(f"{component}: layer {layer!r} config JSON is not an object")
    return decoded


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON object member {key!r}")
        decoded[key] = value
    return decoded


def _resolve_quant(
    component: str,
    quant: LayerQuant,
    tensors: Mapping[str, torch.Tensor],
) -> _ResolvedQuant:
    format_name = quant.format
    full_precision = quant.full_precision_matmul
    decoded: dict[str, object] = {}
    if quant.config is not None:
        decoded = _decode_layer_config(component, quant.layer, tensors[quant.config])
        raw_format = decoded.get("format")
        if not isinstance(raw_format, str):
            raise AssembleError(f"{component}: layer {quant.layer!r} config carries no format")
        if format_name is not None and raw_format != format_name:
            raise AssembleError(
                f"{component}: layer {quant.layer!r} payload format {raw_format!r}"
                f" contradicts planned format {format_name!r}"
            )
        format_name = raw_format
        raw_full_precision = decoded.get("full_precision_matrix_mult", False)
        if raw_format in {"int8_tensorwise", "nvfp4"}:
            allowed = {"format", "full_precision_matrix_mult"}
            if raw_format == "int8_tensorwise":
                allowed.update({"params", "convrot", "convrot_groupsize"})
            unexpected = sorted(set(decoded) - allowed)
            if unexpected:
                raise AssembleError(
                    f"{component}: layer {quant.layer!r} config has unexpected"
                    f" fields: {', '.join(unexpected)}"
                )
            if not isinstance(raw_full_precision, bool):
                raise AssembleError(
                    f"{component}: layer {quant.layer!r} full_precision_matrix_mult must be a bool"
                )
            if quant.format is not None and raw_full_precision != quant.full_precision_matmul:
                raise AssembleError(
                    f"{component}: layer {quant.layer!r} payload"
                    " full_precision_matrix_mult contradicts the planned value"
                )
            full_precision = raw_full_precision
        else:
            full_precision = bool(raw_full_precision)
    if format_name is None:
        raise AssembleError(
            f"{component}: layer {quant.layer!r} has no resolved quantization format"
        )
    if format_name == "int8_tensorwise":
        parameters = dict(quant.parameters)
        if quant.config is not None:
            raw_parameters = decoded.get("params", {})
            if not isinstance(raw_parameters, dict):
                raise AssembleError(
                    f"{component}: layer {quant.layer!r} int8_tensorwise params must be an object"
                )
            payload_parameters = dict(raw_parameters)
            for name in ("convrot", "convrot_groupsize"):
                if name in decoded:
                    payload_parameters[name] = decoded[name]
            payload_parameters.setdefault("convrot", False)
            payload_parameters.setdefault("convrot_groupsize", 256)
            if payload_parameters != parameters:
                raise AssembleError(
                    f"{component}: layer {quant.layer!r} payload parameters contradict the plan"
                )
        convrot = parameters.get("convrot", False)
        group = parameters.get("convrot_groupsize", 256)
        if not isinstance(convrot, bool) or isinstance(group, bool) or not isinstance(group, int):
            raise AssembleError(
                f"{component}: layer {quant.layer!r} carries invalid int8_tensorwise parameters"
            )
        if group < 4 or group & (group - 1) or group.bit_length() % 2 == 0:
            raise AssembleError(
                f"{component}: layer {quant.layer!r} int8_tensorwise convrot_groupsize"
                " must be a power of 4 >= 4"
            )
        return _ResolvedQuant(format_name, None, full_precision, convrot, group)
    if format_name == "nvfp4":
        return _ResolvedQuant(format_name, None, full_precision)
    fp8_dtype = FP8_QUANT_DTYPES.get(format_name)
    if fp8_dtype is None:
        raise AssembleError(
            f"{component}: layer {quant.layer!r} uses quantization"
            f" format {format_name!r}, which has no port (ROADMAP:"
            " Native inference)"
        )
    return _ResolvedQuant(format_name, fp8_dtype, full_precision)


def _scalar_scale(component: str, layer: str, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a per-tensor scale to the registered 0-dim float32
    form (checkpoints ship () and (1,) interchangeably)."""
    if tensor.numel() != 1 or tensor.dtype != torch.float32:
        raise AssembleError(
            f"{component}: layer {layer!r} {name} must be one float32"
            f" value, got {tensor.dtype} with {tensor.numel()} elements"
        )
    return tensor.reshape(())


def _swap_in_fp8_linear(
    component: str,
    module: torch.nn.Module,
    layer: str,
    resolved: _ResolvedQuant,
    *,
    compute_dtype: torch.dtype,
) -> None:
    try:
        existing = module.get_submodule(layer)
    except AttributeError as error:
        raise AssembleError(
            f"{component}: quantized layer {layer!r} does not exist in the constructed module"
        ) from error
    if not isinstance(existing, torch.nn.Linear):
        raise AssembleError(
            f"{component}: quantized layer {layer!r} is"
            f" {type(existing).__name__}, only Linear layers have an"
            " fp8 port (ROADMAP: Native inference)"
        )
    # The stub types Linear.bias as Parameter, but bias=False layers
    # genuinely hold None at runtime.
    existing_bias = cast("torch.nn.Parameter | None", existing.bias)
    fp8_dtype = resolved.fp8_dtype
    assert fp8_dtype is not None
    replacement = Fp8Linear(
        existing.in_features,
        existing.out_features,
        bias=existing_bias is not None,
        fp8_dtype=fp8_dtype,
        compute_dtype=compute_dtype,
        full_precision_matmul=resolved.full_precision_matmul,
    )
    parent_name, _, attr = layer.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, attr, replacement)


def _swap_in_nvfp4_linear(
    component: str,
    module: torch.nn.Module,
    layer: str,
    quant: LayerQuant,
    resolved: _ResolvedQuant,
    *,
    compute_dtype: torch.dtype,
) -> None:
    try:
        existing = module.get_submodule(layer)
    except AttributeError as error:
        raise AssembleError(
            f"{component}: quantized layer {layer!r} does not exist in the constructed module"
        ) from error
    if not isinstance(existing, torch.nn.Linear):
        raise AssembleError(
            f"{component}: NVFP4 layer {layer!r} is {type(existing).__name__},"
            " only Linear layers have an NVFP4 port"
        )
    existing_bias = cast("torch.nn.Parameter | None", existing.bias)
    replacement = Nvfp4Linear(
        existing.in_features,
        existing.out_features,
        bias=existing_bias is not None,
        compute_dtype=compute_dtype,
        pre_quant_scale=quant.pre_quant_scale is not None,
        input_scale=quant.input_scale is not None,
        full_precision_matmul=resolved.full_precision_matmul,
    )
    parent_name, _, attr = layer.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, attr, replacement)


def _swap_in_int8_layer(
    component: str,
    module: torch.nn.Module,
    layer: str,
    resolved: _ResolvedQuant,
    *,
    compute_dtype: torch.dtype,
) -> None:
    try:
        existing = module.get_submodule(layer)
    except AttributeError as error:
        raise AssembleError(
            f"{component}: quantized layer {layer!r} does not exist in the constructed module"
        ) from error
    if isinstance(existing, torch.nn.Linear):
        existing_bias = cast("torch.nn.Parameter | None", existing.bias)
        replacement: torch.nn.Module = Int8Linear(
            existing.in_features,
            existing.out_features,
            bias=existing_bias is not None,
            compute_dtype=compute_dtype,
            convrot=resolved.convrot,
            convrot_groupsize=resolved.convrot_groupsize,
            full_precision_matmul=resolved.full_precision_matmul,
        )
    elif isinstance(existing, torch.nn.Embedding):
        if existing.max_norm is not None:
            raise AssembleError(
                f"{component}: INT8 embedding layer {layer!r} uses max_norm,"
                " which packed INT8 embeddings do not support"
            )
        replacement = Int8Embedding(
            existing.num_embeddings,
            existing.embedding_dim,
            compute_dtype=compute_dtype,
            convrot=resolved.convrot,
            convrot_groupsize=resolved.convrot_groupsize,
        )
    else:
        raise AssembleError(
            f"{component}: INT8 layer {layer!r} is {type(existing).__name__},"
            " only Linear and Embedding layers have an INT8 port"
        )
    parent_name, _, attr = layer.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, attr, replacement)


def _gguf_encoded_candidates(
    plan: ComponentPlan[C], authority: GGUFWeightSource
) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """Model keys the memory residency mode can hold as encoded
    blocks: untransformed rank-2 quantized weights in a decodable
    layout, read through their own source key. Everything else keeps
    the eager reference decode."""

    usage: dict[str, int] = {}
    for source_key in plan.keys.values():
        usage[source_key] = usage.get(source_key, 0) + 1
    public = authority.component_map.public_tensors()
    candidates: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for model_key, source_key in plan.keys.items():
        if not model_key.endswith(".weight") or model_key in plan.transforms:
            continue
        if model_key[: -len(".weight")] in plan.quant or usage[source_key] != 1:
            continue
        mapped = public.get(source_key)
        if mapped is None or len(mapped.logical_shape) != 2:
            continue
        if not mapped.ggml_type.quantized:
            continue
        shape = GGUF_BLOCK_SHAPES.get(mapped.ggml_type.name)
        if shape is None or math.prod(mapped.logical_shape) % shape[0]:
            continue
        candidates[model_key] = (source_key, mapped.logical_shape, mapped.ggml_type.name)
    return candidates


def _swap_in_gguf_encoded_linears(
    plan: ComponentPlan[C],
    authority: GGUFWeightSource,
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    candidates: dict[str, tuple[str, tuple[int, ...], str]],
    *,
    compute_dtype: torch.dtype,
    decoded_cache: GgufDecodedCache | None = None,
) -> None:
    """Replace candidate Linear layers with encoded-resident modules,
    filling ``state`` with their block payloads. A candidate whose
    module is not a matching Linear (a convolution port or reshaped
    consumer) falls back to the eager reference decode."""

    for model_key, (source_key, logical_shape, ggml_type) in candidates.items():
        layer = model_key[: -len(".weight")]
        try:
            existing: torch.nn.Module | None = module.get_submodule(layer)
        except AttributeError:
            existing = None
        if (
            not isinstance(existing, torch.nn.Linear)
            or (existing.out_features, existing.in_features) != logical_shape
        ):
            state[model_key] = load_gguf_tensors(
                authority, (source_key,), expected_runtime_facts=plan.runtime_facts
            )[source_key]
            continue
        existing_bias = cast("torch.nn.Parameter | None", existing.bias)
        replacement = GgufEncodedLinear(
            existing.in_features,
            existing.out_features,
            ggml_type=ggml_type,
            bias=existing_bias is not None,
            compute_dtype=compute_dtype,
            decoded_cache=decoded_cache,
            cache_key=layer,
        )
        parent_name, _, attr = layer.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, replacement)
        state[f"{layer}.weight_blocks"] = load_gguf_encoded_blocks(
            authority, source_key, expected_runtime_facts=plan.runtime_facts
        )
        bias_key = f"{layer}.bias"
        if bias_key in state:
            # Encoded-resident biases use the component dtype.
            state[bias_key] = state[bias_key].to(compute_dtype)


def _apply_transform(
    component: str,
    model_key: str,
    tensor: torch.Tensor,
    transform: TensorTransform,
) -> torch.Tensor:
    """Derive a planned model tensor from its source tensor - the
    reference's OpenCLIP conversion tensor math (comfy/utils.py
    transformers_convert equal-thirds in_proj rows +
    clip_text_transformers_convert text_projection transpose
    @ b78cec87). Row chunks stay zero-copy views: the three q/k/v
    slices of one contiguous fused tensor are themselves contiguous
    and together cover its storage exactly. The transpose is
    materialized (``contiguous()``, as in the reference) so the
    assigned parameter has the Linear memory layout."""
    if isinstance(transform, RowChunk):
        if tensor.ndim == 0:
            raise AssembleError(f"{component}: {model_key!r} row-chunk source is a scalar")
        rows, rem = divmod(tensor.shape[0], transform.parts)
        if rem:
            raise AssembleError(
                f"{component}: {model_key!r} row-chunk source axis 0"
                f" ({tensor.shape[0]}) does not split into"
                f" {transform.parts} equal chunks"
            )
        return tensor[rows * transform.part : rows * (transform.part + 1)].contiguous()
    if isinstance(transform, LinearToConv2D):
        if tensor.ndim != 2:
            raise AssembleError(
                f"{component}: {model_key!r} linear-to-conv source has"
                f" rank {tensor.ndim}, expected 2"
            )
        return tensor.reshape(*tensor.shape, 1, 1)
    if tensor.ndim != 2:
        raise AssembleError(
            f"{component}: {model_key!r} transpose source has rank {tensor.ndim}, expected 2"
        )
    return tensor.transpose(0, 1).contiguous()


def _component_state(
    plan: ComponentPlan[C],
    tensors: Mapping[str, torch.Tensor],
    resolved: Mapping[str, _ResolvedQuant],
    skip: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Model-key state dict: planned renames applied, quant artifacts
    at their registered names, scales normalized. Quantized weights
    must arrive at their declared fp8 dtype - anything else means the
    header lied or the port is missing. ``skip`` names model keys the
    caller resolves itself (encoded GGUF residency weights)."""
    state: dict[str, torch.Tensor] = {}
    for model_key, source_key in plan.keys.items():
        if model_key in skip:
            continue
        tensor = tensors[source_key]
        transform = plan.transforms.get(model_key)
        if transform is not None:
            tensor = _apply_transform(plan.component, model_key, tensor, transform)
        state[model_key] = tensor
    for layer, quant in plan.quant.items():
        weight = state[f"{layer}.weight"]
        selected = resolved[layer]
        if selected.format == "int8_tensorwise":
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} INT8 weight must be rank-2 int8,"
                    f" got {weight.dtype} with shape {tuple(weight.shape)}"
                )
            if selected.convrot and weight.shape[1] % selected.convrot_groupsize:
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} ConvRot width {weight.shape[1]}"
                    " must be divisible by convrot_groupsize"
                    f" {selected.convrot_groupsize}"
                )
            expected_scale = (weight.shape[0], 1) if selected.convrot else ()
            scale = tensors[quant.weight_scale]
            if scale.dtype != torch.float32 or tuple(scale.shape) != expected_scale:
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} INT8 weight_scale must be"
                    f" float32 {expected_scale}, got {scale.dtype} {tuple(scale.shape)}"
                )
            if not torch.isfinite(scale).all().item():
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} weight_scale contains non-finite values"
                )
            state[f"{layer}.weight_scale"] = scale
            continue
        if selected.format == "nvfp4":
            if weight.dtype != torch.uint8 or weight.ndim != 2:
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} NVFP4 weight must"
                    f" be rank-2 uint8, got {weight.dtype} with shape"
                    f" {tuple(weight.shape)}"
                )
            logical = (weight.shape[0], weight.shape[1] * 2)
            expected_block = (
                ((logical[0] + 127) // 128) * 128,
                (((logical[1] // 16) + 3) // 4) * 4,
            )
            block_scale = tensors[quant.weight_scale]
            if (
                block_scale.dtype != torch.float8_e4m3fn
                or tuple(block_scale.shape) != expected_block
            ):
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} NVFP4 weight_scale"
                    f" must be float8_e4m3fn {expected_block}, got"
                    f" {block_scale.dtype} {tuple(block_scale.shape)}"
                )
            if not torch.isfinite(block_scale.float()).all().item():
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} weight_scale contains non-finite values"
                )
            state[f"{layer}.weight_scale"] = block_scale
            if quant.weight_scale_2 is None:
                raise AssembleError(
                    f"{plan.component}: layer {layer!r} NVFP4 plan is missing weight_scale_2"
                )
            scales = [("weight_scale_2", quant.weight_scale_2)]
            if quant.input_scale is not None:
                scales.append(("input_scale", quant.input_scale))
            for name, source_key in scales:
                scale = tensors[source_key]
                if scale.dtype != torch.float32 or scale.shape != ():
                    raise AssembleError(
                        f"{plan.component}: layer {layer!r} NVFP4 {name}"
                        f" must be scalar float32, got {scale.dtype}"
                        f" with shape {tuple(scale.shape)}"
                    )
                if not torch.isfinite(scale).item():
                    raise AssembleError(f"{plan.component}: layer {layer!r} {name} is not finite")
                state[f"{layer}.{name}"] = scale
            if quant.pre_quant_scale is not None:
                pre = tensors[quant.pre_quant_scale]
                if not pre.is_floating_point() or tuple(pre.shape) != (logical[1],):
                    raise AssembleError(
                        f"{plan.component}: layer {layer!r} pre_quant_scale"
                        f" must be floating with shape {(logical[1],)}, got"
                        f" {pre.dtype} {tuple(pre.shape)}"
                    )
                if not torch.isfinite(pre).all().item():
                    raise AssembleError(
                        f"{plan.component}: layer {layer!r} pre_quant_scale"
                        " contains non-finite values"
                    )
                state[f"{layer}.pre_quant_scale"] = pre
            continue
        fp8_dtype = selected.fp8_dtype
        assert fp8_dtype is not None
        if weight.dtype != fp8_dtype:
            raise AssembleError(
                f"{plan.component}: layer {layer!r} declares"
                f" {fp8_dtype} but its weight payload is {weight.dtype}"
            )
        state[f"{layer}.weight_scale"] = _scalar_scale(
            plan.component, layer, "weight_scale", tensors[quant.weight_scale]
        )
        if quant.input_scale is not None:
            state[f"{layer}.input_scale"] = _scalar_scale(
                plan.component,
                layer,
                "input_scale",
                tensors[quant.input_scale],
            )
        else:
            # The neutral scale: quantize-input against 1.0 is the
            # reference's behavior for checkpoints without one
            # (convert_old_quants drops stored 1.0 scales; kitchen
            # from_float defaults scale to ones).
            state[f"{layer}.input_scale"] = torch.ones(())
    return state


def _quant_member_keys(quant: Mapping[str, LayerQuant]) -> frozenset[str]:
    """Model keys owned by quantized layers (excluded from the
    storage-dtype scan that picks the operations set - Fp8Linear
    casts its own weight and bias at use)."""
    keys = set()
    for layer in quant:
        keys.add(f"{layer}.weight")
        keys.add(f"{layer}.bias")
    return frozenset(keys)


def _pick_operations(
    state: Mapping[str, torch.Tensor],
    quant_keys: frozenset[str],
    compute_dtype: torch.dtype,
) -> Operations:
    """INITLESS when every non-quantized tensor is already at the
    compute dtype; CastOperations (the reference's manual_cast
    decision in pick_operations @ b78cec87) when storage and compute
    dtypes decouple. Non-float tensors (position ids) do not force a
    cast set."""
    for key, tensor in state.items():
        if key in quant_keys or key.endswith(
            (
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
                ".pre_quant_scale",
            )
        ):
            continue
        if tensor.is_floating_point() and tensor.dtype != compute_dtype:
            return CastOperations(compute_dtype)
    return INITLESS


def _load_component(
    plan: ComponentPlan[C],
    build: Callable[..., M],
    *,
    compute_dtype: torch.dtype,
    fp8_matmul: bool,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    transform: Callable[[M, dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    storage_dtype_follows_compute: bool = False,
    preserve_equal_width_cast_storage: bool = False,
) -> M:
    source_keys = set(plan.keys.values())
    for quant in plan.quant.values():
        source_keys.add(quant.weight_scale)
        if quant.weight_scale_2 is not None:
            source_keys.add(quant.weight_scale_2)
        if quant.input_scale is not None:
            source_keys.add(quant.input_scale)
        if quant.pre_quant_scale is not None:
            source_keys.add(quant.pre_quant_scale)
        if quant.config is not None:
            source_keys.add(quant.config)
    if (source_file is None) != (source is None):
        raise TypeError("source_file and source must be supplied together")
    gguf_authority: GGUFWeightSource | None = None
    gguf_encoded: dict[str, tuple[str, tuple[int, ...], str]] = {}
    if plan.source_format == "gguf":
        if source_file is not None or source is not None:
            raise TypeError("GGUF component loading does not accept a safetensors source handle")
        if not isinstance(plan.payload_source, GGUFWeightSource):
            raise TypeError("GGUF component loading requires a verified artifact authority")
        gguf_authority = plan.payload_source
        if gguf_authority.residency_mode != "speed":
            gguf_encoded = _gguf_encoded_candidates(plan, gguf_authority)
        tensors = load_gguf_tensors(
            gguf_authority,
            source_keys - {source_key for source_key, _shape, _ggml_type in gguf_encoded.values()},
            expected_runtime_facts=plan.runtime_facts,
        )
    else:
        tensors = (
            load_tensors(plan.path, source_keys)
            if source_file is None or source is None
            else load_tensors_from_file(source_file, source, source_keys)
        )

    resolved = {
        layer: _resolve_quant(plan.component, quant, tensors) for layer, quant in plan.quant.items()
    }
    state = _component_state(plan, tensors, resolved, skip=frozenset(gguf_encoded))
    quant_keys = _quant_member_keys(plan.quant)
    source_dtypes = sorted(
        {
            str(tensor.dtype)
            for key, tensor in state.items()
            if key not in quant_keys
            and tensor.is_floating_point()
            and tensor.dtype != compute_dtype
        }
    )
    for source_dtype in source_dtypes:
        log.info(
            "%s: casting checkpoint storage dtype %s to compute dtype %s",
            plan.component,
            source_dtype,
            compute_dtype,
        )
    if storage_dtype_follows_compute:
        if plan.quant or gguf_encoded:
            raise AssembleError("compute-dtype storage requires an unquantized component")
        state = {
            key: tensor.to(compute_dtype) if tensor.is_floating_point() else tensor
            for key, tensor in state.items()
        }

    for key in plan.absent:
        if key == "text_projection.weight":
            hidden = state["text_model.embeddings.token_embedding.weight"].shape[1]
            state[key] = torch.eye(hidden, dtype=compute_dtype)
        else:
            raise AssembleError(f"{plan.component}: no default fill for absent key {key!r}")

    # Encoded residency weights are planned float32 (their decoded
    # dtype), so the cast-operations decision must see them exactly as
    # the eager loader would have materialized them.
    operations_probe = (
        state
        if not gguf_encoded
        else {**state, **dict.fromkeys(gguf_encoded, torch.empty(0, dtype=torch.float32))}
    )
    if not plan.quant:
        # Ordinary storage wider than the compute dtype rounds down at
        # load, so it cannot force a cast set.
        operations_probe = {
            key: (
                torch.empty(0, dtype=compute_dtype)
                if tensor.is_floating_point() and tensor.dtype.itemsize > compute_dtype.itemsize
                else tensor
            )
            for key, tensor in operations_probe.items()
        }
    operations = _pick_operations(operations_probe, quant_keys, compute_dtype)
    module = build(plan.config, operations=operations)
    if gguf_encoded:
        assert gguf_authority is not None
        decoded_cache = (
            GgufDecodedCache(gguf_authority.decoded_cache_budget)
            if gguf_authority.residency_mode == "balanced"
            else None
        )
        _swap_in_gguf_encoded_linears(
            plan,
            gguf_authority,
            module,
            state,
            gguf_encoded,
            compute_dtype=compute_dtype,
            decoded_cache=decoded_cache,
        )
    if plan.quant:
        # Quantized biases use the component dtype.
        for layer in plan.quant:
            key = f"{layer}.bias"
            if key in state:
                state[key] = state[key].to(compute_dtype)
    for layer, owner in module.named_modules():
        if layer in plan.quant:
            continue
        owner_dtype = bound_compute_dtype(owner)
        members = tuple(name for name, _value in owner.named_parameters(recurse=False)) + tuple(
            name for name, _value in owner.named_buffers(recurse=False)
        )
        for member in members:
            key = f"{layer}.{member}" if layer else member
            tensor = state.get(key)
            if tensor is None or not tensor.is_floating_point():
                continue
            if owner_dtype is None:
                # Initless parameters live at the compute dtype, so
                # wider checkpoint storage rounds down to it, matching
                # the reference loader's model-dtype parameters.
                if not plan.quant and tensor.dtype.itemsize > compute_dtype.itemsize:
                    state[key] = tensor.to(compute_dtype)
                continue
            # Preserve ordinary narrower storage for cast-at-use; wider
            # storage rounds down to the compute dtype, matching the
            # reference loader's model-dtype parameters. Equal-width retention
            # is opt-in because stored-weight patches depend on prior rounding.
            if owner_dtype == compute_dtype and (
                tensor.dtype.itemsize < compute_dtype.itemsize
                or (
                    preserve_equal_width_cast_storage
                    and tensor.dtype.itemsize == compute_dtype.itemsize
                )
            ):
                continue
            state[key] = tensor.to(owner_dtype)
    for layer, quant in resolved.items():
        if quant.format == "int8_tensorwise":
            _swap_in_int8_layer(plan.component, module, layer, quant, compute_dtype=compute_dtype)
        elif quant.format == "nvfp4":
            _swap_in_nvfp4_linear(
                plan.component,
                module,
                layer,
                plan.quant[layer],
                quant,
                compute_dtype=compute_dtype,
            )
        else:
            _swap_in_fp8_linear(plan.component, module, layer, quant, compute_dtype=compute_dtype)
    if transform is not None:
        state = transform(module, state)
    module.load_state_dict(state, strict=True, assign=True)
    if fp8_matmul:
        for submodule in module.modules():
            if isinstance(submodule, Fp8Linear) and (not submodule.full_precision_matmul):
                submodule.bind_fp8_matmul(True)
            else:
                bind_fp8_matmul_layer(submodule, True)
    return module


def assemble_flux(
    plan: FluxAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledFlux:
    """Realize a planned classic-Flux assembly.

    Compute dtypes follow the reference's defaults for the family
    (T5-XXL MUST NOT compute at fp16 - its RMS variance overflows;
    fp32 is the supported text-encode shape over any storage).
    ``fp8_matmul`` opts scaled and plain e4m3fn Linear storage into the
    Kitchen input-quantization and scaled-mm routes, with eager torch as a
    capability fallback; enable it only where
    :func:`~.quant_linear.supports_fp8_matmul` holds (dequant is the
    universal default).
    """
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    diffusion = _load_component(
        plan.diffusion,
        partial(Flux, attention_kernel=attention_kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()
    if plan.qwen3_2b is None:
        assert plan.clip_l is not None and plan.t5xxl is not None
        clip_l = _load_component(
            plan.clip_l,
            partial(ClipTextModel, attention_kernel=attention_kernels["clip"]),
            compute_dtype=text_dtype,
            fp8_matmul=fp8_matmul,
        )
        t5xxl = _load_component(
            plan.t5xxl,
            partial(T5TextModel, attention_kernel=attention_kernels["t5"]),
            compute_dtype=text_dtype,
            fp8_matmul=fp8_matmul,
        )
        qwen3_2b = None
    else:
        clip_l = None
        t5xxl = None
        qwen3_2b = _load_component(
            plan.qwen3_2b,
            partial(QwenTextModel, attention_kernel=attention_kernels["qwen"]),
            compute_dtype=text_dtype,
            fp8_matmul=fp8_matmul,
        )
    vae = _load_component(
        plan.vae,
        partial(AutoencoderKL, attention_kernel=attention_kernels["vae"]),
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    return AssembledFlux(
        family=plan.family,
        diffusion=diffusion,
        clip_l=clip_l,
        t5xxl=t5xxl,
        vae=vae,
        qwen3_2b=qwen3_2b,
        attention_status=attention_status,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "vae": vae_dtype,
            **(
                {"qwen3_2b": text_dtype}
                if qwen3_2b is not None
                else {"clip_l": text_dtype, "t5xxl": text_dtype}
            ),
        },
    )


def assemble_flux2(
    plan: Flux2AssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledFlux2:
    """Realize one published Flux2 release: DiT, family text encoder,
    and packed batch-norm KL VAE at their planned storage dtypes."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    diffusion = _load_component(
        plan.diffusion,
        partial(Flux, attention_kernel=attention_kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()
    text_encoder = _load_component(
        plan.text_encoder,
        partial(QwenTextModel, attention_kernel=attention_kernels["qwen"]),
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )
    vae = _load_component(
        plan.vae,
        partial(AutoencoderKL, attention_kernel=attention_kernels["vae"]),
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    return AssembledFlux2(
        family=plan.family,
        diffusion=diffusion,
        text_encoder=text_encoder,
        vae=vae,
        attention_status=attention_status,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "text_encoder": text_dtype,
            "vae": vae_dtype,
        },
    )


def assemble_qwen_image(
    plan: QwenImageAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledQwenImage:
    """Realize one exact Qwen Image variant and its shared text/VAE components."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    diffusion = _load_component(
        plan.diffusion,
        partial(QwenImage, attention_kernel=attention_kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()

    def build_text(config: QwenImageTextConfig, *, operations: Operations) -> QwenImageTextModel:
        if config != QWEN_IMAGE_TEXT_CONFIG:
            raise AssembleError("Qwen Image text plan does not carry the exact supported profile")
        return QwenImageTextModel(
            operations=operations,
            attention_kernel=attention_kernels["qwen"],
        )

    text = _load_component(
        plan.qwen2_5_vl_7b,
        build_text,
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )

    def build_vae(_config: Wan21VAEConfig, *, operations: Operations) -> WanVAE:
        return WanVAE(
            operations=operations,
            attention_kernel=attention_kernels["vae"],
        )

    vae = _load_component(
        plan.vae,
        build_vae,
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    return AssembledQwenImage(
        family=plan.family,
        diffusion=diffusion,
        text=text,
        vae=vae,
        attention_status=attention_status,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "text": text_dtype,
            "vae": vae_dtype,
        },
    )


def assemble_z_image(
    plan: ZImageAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledZImage:
    """Realize exact native Z-Image, Qwen3-4B, and Flux AE components."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    diffusion_type = ZImagePixelSpace if plan.vae is None else ZImage
    diffusion = _load_component(
        plan.diffusion,
        partial(diffusion_type, attention_kernel=attention_kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()
    qwen3_4b = _load_component(
        plan.qwen3_4b,
        partial(QwenTextModel, attention_kernel=attention_kernels["qwen"]),
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )
    vae = (
        ZImagePixelCodec()
        if plan.vae is None
        else _load_component(
            plan.vae,
            partial(AutoencoderKL, attention_kernel=attention_kernels["vae"]),
            compute_dtype=vae_dtype,
            fp8_matmul=fp8_matmul,
        )
    )
    return AssembledZImage(
        family=plan.family,
        diffusion=diffusion,
        qwen3_4b=qwen3_4b,
        vae=vae,
        attention_status=attention_status,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "qwen3_4b": text_dtype,
            "vae": vae_dtype,
        },
    )


def assemble_lumina2(
    plan: Lumina2AssemblyPlan,
    *,
    diffusion_dtype: torch.dtype,
    text_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledLumina2:
    """Realize the checkpoint's strict diffusion, Gemma, tokenizer, and VAE plans."""
    kernels, statuses = _select_attention_runtime(attention_policy, attention_route_token)
    tokenizer_source = load_safetensors_header(plan.gemma2_2b.path)
    tokenizer = tokenizer_source.read_uint8_configuration(
        plan.tokenizer_source_key, limit=LUMINA2_TOKENIZER_BYTE_CAP
    )
    diffusion = _load_component(
        plan.diffusion,
        partial(ZImage, attention_kernel=kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    text = _load_component(
        plan.gemma2_2b,
        partial(GemmaTextModel, attention_kernel=kernels["qwen"]),
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )
    text.__dict__[LUMINA2_TOKENIZER_ATTRIBUTE] = tokenizer
    vae = _load_component(
        plan.vae,
        partial(AutoencoderKL, attention_kernel=kernels["vae"]),
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    return AssembledLumina2(
        family=plan.family,
        diffusion=diffusion,
        gemma2_2b=text,
        vae=vae,
        attention_status=statuses,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "gemma2_2b": text_dtype,
            "vae": vae_dtype,
        },
    )


def assemble_z_image_control(
    plan: ZImageControlPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledZImageControl:
    """Strict-load a standalone Z-Image control patch."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )

    def build(_config: object, *, operations: Operations) -> ZImageControl:
        return ZImageControl(operations=operations, attention_kernel=attention_kernels["flux"])

    control = _load_component(
        plan.control,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    _bind_z_image_control_resource(
        control,
        z_image_control_resource_digest(plan.asset_digest, compute_dtype),
    )
    return AssembledZImageControl(
        control=control,
        attention_status=attention_status["flux"],
        compute_dtype=compute_dtype,
    )


def assemble_qwen_image_control(
    plan: QwenImageControlPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledQwenImageControl:
    """Strict-load a maintained InstantX or Fun Qwen Image ControlNet."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )

    def build(_config: object, *, operations: Operations) -> QwenImageControlModel:
        if plan.control.config.kind == "fun":
            return QwenImageFunControlNet(
                control_in_features=plan.control.config.input_features,
                operations=operations,
                attention_kernel=attention_kernels["qwen"],
            )
        return QwenImageInstantXControlNet(
            extra_condition_channels=plan.control.config.input_features - 64,
            operations=operations,
            attention_kernel=attention_kernels["qwen"],
        )

    control = _load_component(
        plan.control,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    _bind_qwen_image_control_resource(
        control,
        qwen_image_control_resource_digest(
            plan.asset_digest, plan.control.config.kind, compute_dtype
        ),
    )
    return AssembledQwenImageControl(
        control=control,
        kind=plan.control.config.kind,
        attention_status=attention_status["qwen"],
        compute_dtype=compute_dtype,
    )


def assemble_qwen_image_diffsynth(
    plan: QwenImageDiffSynthPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> AssembledQwenImageDiffSynth:
    """Strict-load a maintained Qwen Image DiffSynth block patch."""

    def build(_config: object, *, operations: Operations) -> QwenImageDiffSynthPatch:
        return QwenImageDiffSynthPatch(
            input_features=plan.patch.config.input_features,
            operations=operations,
        )

    patch = _load_component(
        plan.patch,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    _bind_qwen_image_diffsynth_resource(
        patch,
        qwen_image_diffsynth_resource_digest(
            plan.asset_digest, plan.patch.config.kind, compute_dtype
        ),
    )
    return AssembledQwenImageDiffSynth(patch, plan.patch.config.kind, compute_dtype)


def assemble_wan21_uni3c(
    plan: Wan21Uni3CPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledWan21Uni3C:
    """Strict-load the maintained Wan 2.1 Uni3C patch."""
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )

    def build(_config: object, *, operations: Operations) -> Wan21Uni3C:
        return Wan21Uni3C(
            operations=operations,
            attention_kernel=attention_kernels["flux"],
        )

    patch = _load_component(
        plan.patch,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    _bind_wan21_uni3c_resource(
        patch,
        wan21_uni3c_resource_digest(plan.asset_digest, compute_dtype),
    )
    return AssembledWan21Uni3C(
        patch,
        attention_status["flux"],
        compute_dtype,
    )


def assemble_wan21_multitalk(
    plan: Wan21MultiTalkPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledWan21MultiTalk:
    """Strict-load the maintained Wan 2.1 InfiniteTalk/MultiTalk patch."""
    if not compute_dtype.is_floating_point:
        raise TypeError("Wan 2.1 MultiTalk compute dtype must be floating")
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )

    def build(_config: object, *, operations: Operations) -> Wan21MultiTalk:
        return Wan21MultiTalk(
            operations=operations,
            audio_operations=CastOperations(torch.float16),
            attention_kernel=attention_kernels["flux"],
        )

    patch = _load_component(
        plan.patch,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    _bind_wan21_multitalk_resource(
        patch,
        wan21_multitalk_resource_digest(plan.asset_digest, compute_dtype),
    )
    return AssembledWan21MultiTalk(
        patch,
        attention_status["flux"],
        compute_dtype,
    )


def assemble_sd(
    plan: SDAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.float16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    diffusion_asset_digest: str | None = None,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledSD:
    """Realize a planned SD 1.5 / SDXL base / SDXL refiner assembly.

    Same executor as :func:`assemble_flux` per component (INITLESS vs
    CastOperations selection, planned-transform application, strict
    assign loading); only the module builders differ. Text encoders
    the family does not wire stay None. Compute-dtype defaults follow
    the reference's SD-era picks (fp16 UNet on supported hardware,
    fp32 text encode over any storage)."""
    if (
        diffusion_asset_digest is not None
        and plan.diffusion_asset_digest is not None
        and diffusion_asset_digest != plan.diffusion_asset_digest
    ):
        raise AssembleError("diffusion asset digest does not match the planned source")
    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    diffusion = _load_component(
        plan.diffusion,
        partial(UNetModel, attention_kernel=attention_kernels["unet"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()
    source_digest = diffusion_asset_digest or plan.diffusion_asset_digest
    if source_digest is not None and plan.diffusion.config == SDXL_UNET_CONFIG:
        _bind_sdxl_base_resource(diffusion, source_digest)
    clip_l = (
        None
        if plan.clip_l is None
        else _load_component(
            plan.clip_l,
            partial(ClipTextModel, attention_kernel=attention_kernels["clip"]),
            compute_dtype=text_dtype,
            fp8_matmul=fp8_matmul,
        )
    )
    clip_g = (
        None
        if plan.clip_g is None
        else _load_component(
            plan.clip_g,
            partial(ClipTextModel, attention_kernel=attention_kernels["clip"]),
            compute_dtype=text_dtype,
            fp8_matmul=fp8_matmul,
        )
    )
    if isinstance(plan.vae, TAESDCodecPlan):
        encoder = _load_component(
            plan.vae.encoder,
            _build_taesd_encoder,
            compute_dtype=vae_dtype,
            fp8_matmul=fp8_matmul,
        )
        decoder = _load_component(
            plan.vae.decoder,
            _build_taesd_decoder,
            compute_dtype=vae_dtype,
            fp8_matmul=fp8_matmul,
        )
        if plan.vae.scale_keys is not None:
            scalars = load_tensors(plan.vae.encoder.path, plan.vae.scale_keys)
            scale, shift = (scalars[key].item() for key in plan.vae.scale_keys)
            expected_scale = torch.tensor(plan.vae.config.vae_scale, dtype=torch.float32).item()
            expected_shift = torch.tensor(plan.vae.config.vae_shift, dtype=torch.float32).item()
            if scale != expected_scale or shift != expected_shift:
                raise AssembleError(
                    "TAESD vae_scale/vae_shift payload contradicts the selected family"
                )
        codec: AutoencoderKL | TAESD = TAESD(
            plan.vae.config, encoder, decoder, compute_dtype=vae_dtype
        )
    else:
        codec = _load_component(
            plan.vae,
            partial(AutoencoderKL, attention_kernel=attention_kernels["vae"]),
            compute_dtype=vae_dtype,
            fp8_matmul=fp8_matmul,
        )
    return AssembledSD(
        family=plan.family,
        diffusion=diffusion,
        clip_l=clip_l,
        clip_g=clip_g,
        vae=codec,
        sampling=plan.sampling or plan.family.sampling,
        attention_status=attention_status,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "vae": vae_dtype,
            **({"clip_l": text_dtype} if clip_l is not None else {}),
            **({"clip_g": text_dtype} if clip_g is not None else {}),
        },
    )


def assemble_wan21(
    plan: Wan21AssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    fp8_matmul: bool = False,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledWan21:
    """Realize an official Wan assembly on CPU."""

    attention_kernels, attention_status = _select_attention_runtime(
        attention_policy, attention_route_token
    )
    model_types = {
        "animate2": WanAnimate2Model,
        "causal_ar": Wan21CausalModel,
        "humo": Wan21HumoModel,
        "scail": WanScailModel,
        "scail2": WanScailModel,
        "s2v": Wan22S2VModel,
        "wandancer": Wan22DancerModel,
    }
    model_type = model_types.get(plan.diffusion.config.model_variant, Wan21Model)
    diffusion = _load_component(
        plan.diffusion,
        partial(model_type, attention_kernel=attention_kernels["flux"]),
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    diffusion.__dict__["_dinkster_component_plan"] = plan.diffusion.without_payload_source()
    umt5xxl = _load_component(
        plan.umt5xxl,
        partial(T5TextModel, attention_kernel=attention_kernels["t5"]),
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )

    def build_vae(
        config: Wan21VAEConfig | Wan22VAEConfig, *, operations: Operations
    ) -> WanVAE | Wan22VAE:
        if isinstance(config, Wan22VAEConfig):
            return Wan22VAE(
                config,
                operations=operations,
                attention_kernel=attention_kernels["vae"],
            )
        return WanVAE(
            TorchWanVAEConfig(
                dim=config.dim,
                z_dim=config.z_dim,
                dim_mult=config.dim_mult,
                num_res_blocks=config.num_res_blocks,
                attn_scales=config.attn_scales,
                temporal_downsample=config.temporal_downsample,
                image_channels=config.image_channels,
                conv_out_channels=config.conv_out_channels,
                dropout=config.dropout,
            ),
            operations=operations,
            attention_kernel=attention_kernels["vae"],
        )

    vae = _load_component(
        plan.vae,
        build_vae,
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    clip_vision = (
        None
        if plan.clip_vision is None
        else _load_component(
            plan.clip_vision,
            partial(Wan21ClipVisionEncoder, attention_kernel=attention_kernels["clip"]),
            compute_dtype=torch.float32,
            fp8_matmul=fp8_matmul,
        )
    )
    if plan.tokenizer_vendored:
        tokenizer_model = load_umt5_spiece()
    else:
        tokenizer = load_tensors(plan.umt5xxl.path, (plan.tokenizer_source_key,))[
            plan.tokenizer_source_key
        ]
        if tokenizer.dtype != torch.uint8 or tokenizer.ndim != 1 or tokenizer.numel() == 0:
            raise AssembleError("umt5xxl: spiece_model payload must be nonempty rank-1 uint8")
        tokenizer_model = tokenizer.contiguous().numpy().tobytes()
    return AssembledWan21(
        family=plan.family,
        diffusion=diffusion,
        umt5xxl=umt5xxl,
        vae=vae,
        tokenizer_model=tokenizer_model,
        clip_vision=clip_vision,
        attention_status=attention_status,
        _storage_dtype_follows_compute=False,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "umt5xxl": text_dtype,
            "vae": vae_dtype,
            **({"clip_vision": torch.float32} if clip_vision is not None else {}),
        },
    )


def assemble_sd15_controlnet(
    plan: ControlNetAssemblyPlan,
    *,
    controlnet_dtype: torch.dtype = torch.float16,
) -> AssembledControlNet:
    """Strict-load one planned classic SD1.5 ControlNet on CPU."""
    if not controlnet_dtype.is_floating_point:
        raise TypeError("ControlNet compute dtype must be floating")
    resource_digest = sd15_controlnet_resource_digest(
        plan.asset_digest,
        plan.source_layout,
        controlnet_dtype,
    )
    controlnet = _load_component(
        plan.controlnet,
        SD15ControlNet,
        compute_dtype=controlnet_dtype,
        fp8_matmul=False,
    )
    _bind_sd15_controlnet_resource(controlnet, resource_digest)
    return AssembledControlNet(controlnet, controlnet_dtype)


def assemble_taesd_decoder(
    plan: ComponentPlan[TAESDConfig],
    *,
    decoder_dtype: torch.dtype = torch.float32,
) -> TAESDDecoder:
    """Strict-load one planned standalone TAESD decoder on CPU."""
    if not decoder_dtype.is_floating_point:
        raise TypeError("TAESD decoder compute dtype must be floating")
    if plan.config.role != "decoder":
        raise ValueError("assemble_taesd_decoder requires a decoder-role plan")
    return _load_component(
        plan,
        _build_taesd_decoder,
        compute_dtype=decoder_dtype,
        fp8_matmul=False,
    )


def assemble_taehv_decoder(
    plan: ComponentPlan[TAEHVConfig],
    *,
    decoder_dtype: torch.dtype = torch.float32,
) -> TAEHVDecoder:
    """Strict-load one planned TAEHV video-TAE decoder on CPU."""
    if not decoder_dtype.is_floating_point:
        raise TypeError("TAEHV decoder compute dtype must be floating")
    return _load_component(
        plan,
        _build_taehv_decoder,
        compute_dtype=decoder_dtype,
        fp8_matmul=False,
    )


def assemble_sd15_t2i_adapter(
    plan: T2IAdapterAssemblyPlan,
    *,
    adapter_dtype: torch.dtype = torch.float32,
) -> AssembledT2IAdapter:
    """Strict-load one planned SD1.5 full adapter on CPU."""
    if not adapter_dtype.is_floating_point:
        raise TypeError("T2I Adapter compute dtype must be floating")
    digest = sd15_t2i_adapter_resource_digest(plan.asset_digest, adapter_dtype)
    if plan.adapter.path.suffix.lower() in (".pt", ".pth"):
        payload = torch.load(plan.adapter.path, map_location="cpu", weights_only=True)
        if type(payload) is not dict and not isinstance(payload, Mapping):
            raise AssembleError("t2i_adapter: checkpoint payload must be a tensor mapping")
        state = {key: payload[source] for key, source in plan.adapter.keys.items()}
        if set(payload) != set(plan.adapter.keys.values()) or any(
            type(tensor) is not torch.Tensor for tensor in state.values()
        ):
            raise AssembleError("t2i_adapter: payload keys do not exactly match the plan")
        operations = _pick_operations(state, frozenset(), adapter_dtype)
        adapter = SD15T2IAdapter(plan.adapter.config, operations=operations)
        adapter.load_state_dict(state, strict=True, assign=True)
    else:
        adapter = _load_component(
            plan.adapter, SD15T2IAdapter, compute_dtype=adapter_dtype, fp8_matmul=False
        )
    _bind_sd15_t2i_adapter_resource(adapter, digest)
    return AssembledT2IAdapter(adapter, adapter_dtype)


def assemble_sd15_ipadapter(
    plan: SD15IPAdapterAssemblyPlan,
    *,
    adapter_dtype: torch.dtype = torch.float16,
) -> AssembledSD15IPAdapter:
    """Strict-load the standard adapter and its F32 CLIP image encoder."""
    if not adapter_dtype.is_floating_point:
        raise TypeError("IP-Adapter compute dtype must be floating")
    adapter = _load_component(
        plan.adapter,
        SD15IPAdapter,
        compute_dtype=adapter_dtype,
        fp8_matmul=False,
    )
    clip_vision = _load_component(
        plan.clip_vision,
        SD15IPAdapterClipVisionEncoder,
        compute_dtype=torch.float32,
        fp8_matmul=False,
    )
    digest = sd15_ipadapter_resource_digest(plan.adapter_asset_digest, adapter_dtype)
    _bind_sd15_ipadapter_resource(adapter, digest)
    return AssembledSD15IPAdapter(adapter, clip_vision, adapter_dtype)


def assemble_sdxl_control_lora(
    plan: SDXLControlLoRAAssemblyPlan,
    base: AssembledSD,
    *,
    control_lora_dtype: torch.dtype = torch.float16,
) -> AssembledSDXLControlLoRA:
    """Bind one official Control-LoRA artifact to its assembled SDXL base."""
    if base.diffusion.config != plan.control_lora.config.base:
        raise AssembleError("Control-LoRA requires the exact SDXL base UNet geometry")
    _validate_sdxl_base_resource(base.diffusion, plan.base_asset_digest)
    if not control_lora_dtype.is_floating_point:
        raise TypeError("Control-LoRA compute dtype must be floating")
    tensors = load_tensors(plan.control_lora.path, plan.control_lora.keys.values())
    model = SDXLControlLoRA(plan.control_lora.config)
    model_state = model.state_dict()
    base_state = base.diffusion.state_dict()
    state: dict[str, torch.Tensor] = {}
    for key in model_state:
        if key in tensors:
            state[key] = tensors[key].to(dtype=control_lora_dtype)
        elif key in base_state:
            state[key] = base_state[key].to(dtype=control_lora_dtype)
    if set(state) != set(model_state):
        missing = sorted(set(model_state) - set(state))
        raise AssembleError("Control-LoRA cannot resolve model state: " + ", ".join(missing[:3]))
    model.load_state_dict(state, strict=True, assign=True)
    factor_keys = sorted(key for key in tensors if key.endswith((".up", ".down")))
    for key in factor_keys:
        module_name, attribute = key.rsplit(".", 1)
        module = model.get_submodule(module_name)
        if type(module) not in (
            type(model.time_embed[0]),
            type(cast("torch.nn.Sequential", model.input_blocks[0])[0]),
        ):
            raise AssembleError(f"Control-LoRA factor targets unsupported module {module_name!r}")
        setattr(
            module,
            attribute,
            torch.nn.Parameter(tensors[key].to(dtype=control_lora_dtype), requires_grad=False),
        )
    digest = sdxl_control_lora_resource_digest(
        plan.asset_digest,
        plan.base_asset_digest,
        control_lora_dtype,
    )
    _bind_sdxl_control_lora_resource(model, digest)
    return AssembledSDXLControlLoRA(model, control_lora_dtype)


def assemble_sdxl_controlnet_union(
    plan: SDXLControlNetUnionAssemblyPlan,
    *,
    controlnet_dtype: torch.dtype = torch.float16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> AssembledSDXLControlNetUnion:
    """Strict-load one planned xinsir SDXL ControlNet Union on CPU."""
    if not controlnet_dtype.is_floating_point:
        raise TypeError("SDXL ControlNet Union compute dtype must be floating")
    attention_kernels, _ = _select_attention_runtime(attention_policy, attention_route_token)
    model = _load_component(
        plan.controlnet_union,
        partial(SDXLControlNetUnion, attention_kernel=attention_kernels["unet"]),
        compute_dtype=controlnet_dtype,
        fp8_matmul=False,
    )
    digest = sdxl_controlnet_union_resource_digest(
        plan.asset_digest,
        plan.controlnet_union.config,
        controlnet_dtype,
    )
    _bind_sdxl_controlnet_union_resource(model, digest)
    return AssembledSDXLControlNetUnion(model, controlnet_dtype)


def assemble_sdxl_controlnet(
    plan: SDXLControlNetAssemblyPlan,
    *,
    controlnet_dtype: torch.dtype = torch.float16,
) -> AssembledSDXLControlNet:
    """Strict-load one planned classic SDXL ControlNet on CPU."""
    if not controlnet_dtype.is_floating_point:
        raise TypeError("SDXL ControlNet compute dtype must be floating")
    model = _load_component(
        plan.controlnet,
        SDXLControlNet,
        compute_dtype=controlnet_dtype,
        fp8_matmul=False,
    )
    digest = sdxl_controlnet_resource_digest(
        plan.asset_digest,
        plan.controlnet.config,
        controlnet_dtype,
    )
    _bind_sdxl_controlnet_resource(model, digest)
    return AssembledSDXLControlNet(model, controlnet_dtype)
