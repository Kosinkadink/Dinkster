"""Strict split-checkpoint planning and loading for MiniMax H3."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, TypeVar, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_DIT_PROVIDER_REVISION,
    AttentionRouteToken,
)
from dinkster_inference import minimax_h3_assembly as _native_h3_assembly
from dinkster_inference.assembly import ComponentPlan, NativePlanningContext
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.minimax_h3_assembly import (
    MINIMAX_H3_SPLIT_COMMON_ROLES,
    MiniMaxH3CommonComponentRole,
    MiniMaxH3ModelAssemblyPlan,
    MiniMaxH3SplitAssemblyError,
    component_plan_claims,
    minimax_h3_component_runtime_identity,
    plan_minimax_h3_audio_vae_component,
    plan_minimax_h3_common_component,
    plan_minimax_h3_conditioner_component,
    plan_minimax_h3_diffusion_component,
    plan_minimax_h3_video_vae_component,
)
from dinkster_inference.minimax_h3_assembly import (
    plan_minimax_h3_model_assembly as plan_minimax_h3_model_assembly_native,
)
from dinkster_inference.minimax_h3_conditioner import (
    MINIMAX_H3_CONDITIONER_CONFIG,
    MiniMaxH3ConditionerConfig,
)
from dinkster_inference.minimax_h3_dit import (
    MiniMaxH3DiTLayout,
    MiniMaxH3DiTRole,
    minimax_h3_dit_runtime_identity,
)
from dinkster_inference.sources import (
    SafetensorsSource,
    load_safetensors_header_from_file,
)
from dinkster_inference.weights import (
    WeightEntry,
    WeightSource,
)

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import (
    AttentionPolicy,
    AttentionRole,
    AttentionSelection,
    AttentionStatus,
    resolve_role_attention,
)
from .distributed import (
    guidance_receipt_identity,
)
from .minimax_h3_audio import MiniMaxH3AudioVAE
from .minimax_h3_conditioner import MiniMaxH3ConditionerModel
from .minimax_h3_dit import (
    MiniMaxH3DiT,
    assemble_minimax_h3_dit,
    minimax_h3_guidance_integration_facts,
)
from .minimax_h3_video_vae import MiniMaxH3VideoVAE, MiniMaxH3VideoVAEConfig
from .module_residency import declare_residency_materialization_ceilings
from .operations import CastOperations, Operations, ResidencyRouted

C = TypeVar("C")
M = TypeVar("M", bound=torch.nn.Module)

_COMMON_ROLES = MINIMAX_H3_SPLIT_COMMON_ROLES
_TORCH_TO_DTYPE = {
    torch.float16: FLOAT16,
    torch.bfloat16: BFLOAT16,
    torch.float32: FLOAT32,
}


def worker_planning_context() -> NativePlanningContext:
    """Return the live worker facts bound into MiniMax H3 identity."""

    return NativePlanningContext(
        torch_version=str(torch.__version__),
        dinkster_kitchen_version=version("dinkster-kitchen"),
    )


def _identity_dtype(dtype: torch.dtype) -> DType:
    try:
        return _TORCH_TO_DTYPE[dtype]
    except KeyError:
        raise TypeError(f"MiniMax H3 compute dtype is unsupported: {dtype}") from None


def minimax_h3_conditioner_runtime_identity(
    conditioner: ComponentPlan[MiniMaxH3ConditionerConfig],
    *,
    conditioner_dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Build the native identity for one H3 conditioner component."""

    return minimax_h3_component_runtime_identity(
        cast("ComponentPlan[object]", conditioner),
        "qwen3vl-32b-conditioner",
        _identity_dtype(conditioner_dtype),
    )


def minimax_h3_video_vae_runtime_identity(
    video_vae: ComponentPlan[MiniMaxH3VideoVAEConfig],
    *,
    video_vae_dtype: torch.dtype = torch.float16,
) -> str:
    """Build the native identity for one H3 video VAE component."""

    return minimax_h3_component_runtime_identity(
        cast("ComponentPlan[object]", video_vae),
        "video-vae",
        _identity_dtype(video_vae_dtype),
    )


def minimax_h3_audio_vae_runtime_identity(
    audio_vae: ComponentPlan[None],
    *,
    audio_vae_dtype: torch.dtype = torch.float16,
) -> str:
    """Build the native identity for one H3 audio VAE component."""

    return minimax_h3_component_runtime_identity(
        cast("ComponentPlan[object]", audio_vae),
        "audio-vae",
        _identity_dtype(audio_vae_dtype),
    )


_plan_claims = component_plan_claims
_extract_quantized_source = (
    _native_h3_assembly._extract_quantized_source  # pyright: ignore[reportPrivateUsage]
)
_component_plan = _native_h3_assembly._component_plan  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True, slots=True)
class MiniMaxH3ArtifactPaths:
    """Verified assets for one selected DiT and the three shared components."""

    diffusion_role: MiniMaxH3DiTRole
    paths: Mapping[str, Path]
    assets: Mapping[str, AssetRef]
    provider_revision: str = MINIMAX_H3_DIT_PROVIDER_REVISION

    def __post_init__(self) -> None:
        if self.diffusion_role not in ("fl2va-dit", "ref2va-dit"):
            raise ValueError("MiniMax H3 diffusion role must be fl2va-dit or ref2va-dit")
        paths_obj = cast("object", self.paths)
        if not isinstance(paths_obj, Mapping):
            raise TypeError("MiniMax H3 artifact paths must be a mapping")
        paths = dict(cast("Mapping[str, object]", paths_obj))
        expected = {self.diffusion_role, *_COMMON_ROLES}
        if set(paths) != expected:
            raise ValueError("MiniMax H3 artifact paths must name one complete split graph")
        if any(not isinstance(path, Path) for path in paths.values()):
            raise TypeError("MiniMax H3 artifact paths must be pathlib.Path values")
        if len(set(paths.values())) != len(paths):
            raise ValueError("MiniMax H3 artifact paths must be distinct")
        assets_obj = cast("object", self.assets)
        if not isinstance(assets_obj, Mapping):
            raise TypeError("MiniMax H3 artifacts must be a mapping")
        raw_assets = dict(cast("Mapping[str, object]", assets_obj))
        if set(raw_assets) != expected:
            raise ValueError("MiniMax H3 artifacts must cover the selected graph")
        assets: dict[str, AssetRef] = {}
        for role, asset in raw_assets.items():
            if type(asset) is not AssetRef:
                raise TypeError("MiniMax H3 artifacts must be AssetRef values")
            digest = asset.digest
            if (
                not digest.startswith("blake3:")
                or len(digest) != 71
                or any(char not in "0123456789abcdef" for char in digest[7:])
            ):
                raise ValueError("MiniMax H3 artifact digests must be canonical blake3")
            size = asset.size
            if type(size) is not int or size < 0:
                raise ValueError("MiniMax H3 artifact sizes must be nonnegative integers")
            assets[role] = asset
        if self.provider_revision != MINIMAX_H3_DIT_PROVIDER_REVISION:
            raise ValueError("MiniMax H3 provider revision is not the immutable authority")
        object.__setattr__(self, "paths", MappingProxyType(cast("dict[str, Path]", paths)))
        object.__setattr__(
            self,
            "assets",
            MappingProxyType(assets),
        )


@dataclass(frozen=True)
class MiniMaxH3SplitAssemblyPlan:
    """Header-complete plans for one DiT, conditioner, and both codecs."""

    diffusion: ComponentPlan[MiniMaxH3DiTLayout]
    conditioner: ComponentPlan[MiniMaxH3ConditionerConfig]
    video_vae: ComponentPlan[MiniMaxH3VideoVAEConfig]
    audio_vae: ComponentPlan[None]
    artifacts: MiniMaxH3ArtifactPaths
    claims: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        components = {
            self.artifacts.diffusion_role: self.diffusion,
            "qwen3vl-32b-conditioner": self.conditioner,
            "video-vae": self.video_vae,
            "audio-vae": self.audio_vae,
        }
        expected_names = {
            self.artifacts.diffusion_role: "diffusion",
            "qwen3vl-32b-conditioner": "conditioner",
            "video-vae": "video_vae",
            "audio-vae": "audio_vae",
        }
        claims = dict(self.claims)
        if set(claims) != set(components):
            raise ValueError("MiniMax H3 claims must cover the complete selected graph")
        for role, component in components.items():
            if component.component != expected_names[role]:
                raise ValueError(f"MiniMax H3 {role} component name is inconsistent")
            if component.path != self.artifacts.paths[role]:
                raise ValueError(f"MiniMax H3 {role} path differs from artifact authority")
            expected_claims = _plan_claims(cast("ComponentPlan[object]", component))
            if claims[role] != expected_claims:
                raise ValueError(f"MiniMax H3 {role} claims are not exact")
        object.__setattr__(self, "claims", MappingProxyType(claims))


@dataclass(frozen=True, slots=True)
class AssembledMiniMaxH3Model:
    """One H3 DiT behind the generic diffusion component name."""

    diffusion: MiniMaxH3DiT
    _storage_dtype_follows_compute: bool = field(default=True, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({"diffusion": torch.bfloat16}),
        repr=False,
        compare=False,
    )
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        dtypes = dict(self._component_compute_dtypes)
        if set(dtypes) != {"diffusion"} or dtypes["diffusion"] not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("MiniMax H3 model requires one supported diffusion dtype")
        statuses = dict(self.attention_status)
        if set(statuses) not in (set(), {"flux"}) or any(
            type(status) is not AttentionStatus for status in statuses.values()
        ):
            raise ValueError("MiniMax H3 model attention status must describe only Flux attention")
        object.__setattr__(self, "_component_compute_dtypes", MappingProxyType(dtypes))
        object.__setattr__(self, "attention_status", MappingProxyType(statuses))


@dataclass(frozen=True, slots=True)
class MiniMaxH3Model:
    """An ordinary MODEL owner for one role-specific H3 DiT."""

    assembled: AssembledMiniMaxH3Model
    runtime_identity: str
    model_role: MiniMaxH3DiTRole
    runtime_facts: tuple[str, ...] = ()
    receipt_identity: str | None = None

    def __post_init__(self) -> None:
        if type(self.runtime_identity) is not str or not self.runtime_identity:
            raise ValueError("MiniMax H3 model requires a runtime identity")
        if self.model_role not in ("fl2va-dit", "ref2va-dit"):
            raise ValueError("MiniMax H3 model requires a recognized DiT role")
        facts = tuple(self.runtime_facts)
        if any(type(fact) is not str or not fact for fact in facts):
            raise ValueError("MiniMax H3 model runtime facts must be non-empty strings")
        object.__setattr__(self, "runtime_facts", facts)
        if self.receipt_identity is not None and (
            type(self.receipt_identity) is not str or not self.receipt_identity
        ):
            raise ValueError("MiniMax H3 model receipt identity must be non-empty when present")


def _source_path(source: WeightSource, role: str) -> Path:
    path = getattr(source, "path", None)
    if not isinstance(path, Path):
        raise TypeError(f"MiniMax H3 {role} source must expose a pathlib.Path")
    return path


def _module_layout(  # pyright: ignore[reportUnusedFunction]
    module: torch.nn.Module,
) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _artifact_identity_facts(artifacts: MiniMaxH3ArtifactPaths, role: str) -> tuple[str, ...]:
    asset = artifacts.assets[role]
    return (
        f"asset_digest={asset.digest}",
        f"asset_size={asset.size}",
    )


def plan_minimax_h3_model_assembly(
    diffusion: WeightSource,
    *,
    role: MiniMaxH3DiTRole,
    path: Path,
    attention_policy: AttentionPolicy = "auto",
) -> MiniMaxH3ModelAssemblyPlan:
    """Plan one role-specific H3 DiT without requiring unrelated assets."""

    return plan_minimax_h3_model_assembly_native(
        diffusion,
        role=role,
        path=path,
        context=worker_planning_context(),
        attention_policy=attention_policy,
    )


def plan_minimax_h3_split_assembly(
    *,
    diffusion: WeightSource,
    conditioner: WeightSource,
    video_vae: WeightSource,
    audio_vae: WeightSource,
    artifacts: MiniMaxH3ArtifactPaths,
) -> MiniMaxH3SplitAssemblyPlan:
    """Plan one exact split H3 graph from headers without reading payloads."""

    if type(artifacts) is not MiniMaxH3ArtifactPaths:
        raise TypeError("artifacts must be MiniMaxH3ArtifactPaths")
    sources = {
        artifacts.diffusion_role: diffusion,
        "qwen3vl-32b-conditioner": conditioner,
        "video-vae": video_vae,
        "audio-vae": audio_vae,
    }
    for role, source in sources.items():
        if _source_path(source, role) != artifacts.paths[role]:
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {role} source path differs from artifact selection"
            )

    context = worker_planning_context()
    diffusion_plan, diffusion_claims = plan_minimax_h3_diffusion_component(
        diffusion,
        artifacts.diffusion_role,
        artifacts.paths[artifacts.diffusion_role],
        context=context,
        identity_facts=_artifact_identity_facts(artifacts, artifacts.diffusion_role),
    )
    conditioner_plan, conditioner_claims = plan_minimax_h3_conditioner_component(
        conditioner,
        artifacts.paths["qwen3vl-32b-conditioner"],
        context=context,
        identity_facts=_artifact_identity_facts(artifacts, "qwen3vl-32b-conditioner"),
    )
    video_plan, video_claims = plan_minimax_h3_video_vae_component(
        video_vae,
        artifacts.paths["video-vae"],
        context=context,
        identity_facts=_artifact_identity_facts(artifacts, "video-vae"),
    )
    audio_plan, audio_claims = plan_minimax_h3_audio_vae_component(
        audio_vae,
        artifacts.paths["audio-vae"],
        identity_facts=_artifact_identity_facts(artifacts, "audio-vae"),
    )
    return MiniMaxH3SplitAssemblyPlan(
        diffusion=diffusion_plan,
        conditioner=conditioner_plan,
        video_vae=video_plan,
        audio_vae=audio_plan,
        artifacts=artifacts,
        claims={
            artifacts.diffusion_role: diffusion_claims,
            "qwen3vl-32b-conditioner": conditioner_claims,
            "video-vae": video_claims,
            "audio-vae": audio_claims,
        },
    )


def verify_minimax_h3_artifacts(
    artifacts: MiniMaxH3ArtifactPaths,
) -> None:
    """Validate selected paths against asset identity and provider metadata."""

    if type(artifacts) is not MiniMaxH3ArtifactPaths:
        raise TypeError("artifacts must be MiniMaxH3ArtifactPaths")
    for role, path in artifacts.paths.items():
        _verify_artifact(role, path, artifacts.assets[role])


def _verify_artifact_size(handle: BinaryIO, asset_size: int, role: str) -> None:
    try:
        size = os.fstat(handle.fileno()).st_size
    except OSError as error:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {role} artifact is unavailable: {error}"
        ) from error
    if size != asset_size:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {role} byte size differs from its asset metadata"
        )


def _verify_artifact(role: str, path: Path, asset: AssetRef) -> None:
    try:
        with asset.open() as handle:
            _verify_artifact_file(role, path, handle, asset.digest, asset.size)
    except (AssetError, OSError) as error:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {role} artifact could not be verified: {error}"
        ) from error


def _verify_artifact_file(
    role: str,
    path: Path,
    handle: BinaryIO,
    asset_digest: str,
    asset_size: int,
) -> None:
    del path, asset_digest
    _verify_artifact_size(handle, asset_size, role)


@dataclass(frozen=True)
class _VerifiedArtifact:
    file: BinaryIO
    source: WeightSource


@dataclass(frozen=True)
class _DescriptorPinnedSource:
    source: SafetensorsSource
    file: BinaryIO
    asset_digest: str | None = None
    asset_size: int | None = None

    @property
    def path(self) -> Path:
        return self.source.path

    @property
    def entries(self) -> Mapping[str, WeightEntry]:
        return self.source.entries

    def keys(self) -> tuple[str, ...]:
        return tuple(self.source.keys())

    def entry(self, key: str) -> WeightEntry:
        return self.source.entry(key)

    def metadata(self) -> Mapping[str, str]:
        return self.source.metadata()

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key)


def _load_verified_component(
    plan: ComponentPlan[C],
    build: Callable[..., M],
    verified: _VerifiedArtifact,
    *,
    compute_dtype: torch.dtype,
    transform: Callable[[M, dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> M:
    return _load_component(
        plan,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
        source_file=verified.file,
        source=cast("SafetensorsSource", verified.source),
        transform=transform,
    )


def _load_verified_diffusion(
    plan: ComponentPlan[MiniMaxH3DiTLayout],
    verified: _VerifiedArtifact,
    *,
    compute_dtype: torch.dtype,
    attention_selection: AttentionSelection,
    transform: Callable[[MiniMaxH3DiT, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
) -> MiniMaxH3DiT:
    def build(layout: MiniMaxH3DiTLayout, *, operations: Operations) -> MiniMaxH3DiT:
        return _build_diffusion(
            layout, operations=operations, attention_selection=attention_selection
        )

    component = _load_verified_component(
        plan,
        build,
        verified,
        compute_dtype=compute_dtype,
        transform=transform,
    )
    for module in component.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        direct = tuple(module.named_parameters(recurse=False, remove_duplicate=False)) + tuple(
            module.named_buffers(recurse=False, remove_duplicate=False)
        )
        declare_residency_materialization_ceilings(
            module,
            {
                name: module.residency_materialization_dtype(stored).itemsize
                for name, stored in direct
                if stored.is_floating_point() or stored.is_complex()
            },
        )
    return component


def _build_diffusion(
    layout: MiniMaxH3DiTLayout,
    *,
    operations: Operations,
    attention_selection: AttentionSelection,
) -> MiniMaxH3DiT:
    if layout.config != MINIMAX_H3_CONFIG:
        raise MiniMaxH3SplitAssemblyError("diffusion builder requires exact H3 layout")
    # The reference computes patch projections and the final layer at
    # float32, but projects and refines its float16 text encoder output
    # before joining the bfloat16 diffusion stream.
    fp32_operations = CastOperations(torch.float32)
    text_operations = CastOperations(torch.float16)
    return assemble_minimax_h3_dit(
        operations=operations,
        fp32_operations=fp32_operations,
        text_operations=text_operations,
        time_embedding_kind=layout.time_embedding_kind,
        attention_selection=attention_selection,
    )


def _build_conditioner(
    config: MiniMaxH3ConditionerConfig, *, operations: Operations
) -> MiniMaxH3ConditionerModel:
    if config != MINIMAX_H3_CONDITIONER_CONFIG:
        raise MiniMaxH3SplitAssemblyError("conditioner builder requires exact H3 config")
    return MiniMaxH3ConditionerModel(operations=operations)


def _build_audio(_config: None, *, operations: Operations) -> MiniMaxH3AudioVAE:
    return MiniMaxH3AudioVAE(operations=operations)


@dataclass(frozen=True)
class MiniMaxH3LoadedComponent:
    """One independently verified and loaded H3 component."""

    role: MiniMaxH3CommonComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
    runtime_identity: str


def load_minimax_h3_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: MiniMaxH3CommonComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> MiniMaxH3LoadedComponent:
    """Verify, plan, and strict-load one common H3 component."""

    if type(asset) is not AssetRef:
        raise TypeError("MiniMax H3 component asset must be an AssetRef")
    if expected_role not in _COMMON_ROLES:
        raise ValueError(f"unsupported MiniMax H3 component role {expected_role!r}")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("MiniMax H3 component requires an expected identity")
    _identity_dtype(compute_dtype)
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _DescriptorPinnedSource(source, handle, asset.digest, asset.size)
        plan = plan_minimax_h3_common_component(
            pinned,
            role=expected_role,
            path=path,
            context=worker_planning_context(),
        )
        if expected_role == "qwen3vl-32b-conditioner":
            runtime_identity = minimax_h3_conditioner_runtime_identity(
                cast("ComponentPlan[MiniMaxH3ConditionerConfig]", plan),
                conditioner_dtype=compute_dtype,
            )
        elif expected_role == "video-vae":
            runtime_identity = minimax_h3_video_vae_runtime_identity(
                cast("ComponentPlan[MiniMaxH3VideoVAEConfig]", plan),
                video_vae_dtype=compute_dtype,
            )
        else:
            runtime_identity = minimax_h3_audio_vae_runtime_identity(
                cast("ComponentPlan[None]", plan),
                audio_vae_dtype=compute_dtype,
            )
        if runtime_identity != expected_identity:
            raise MiniMaxH3SplitAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        verified = _VerifiedArtifact(handle, pinned)
        if expected_role == "qwen3vl-32b-conditioner":
            module = _load_verified_component(
                plan,
                _build_conditioner,
                verified,
                compute_dtype=compute_dtype,
            )
        elif expected_role == "video-vae":
            module = _load_verified_component(
                plan,
                MiniMaxH3VideoVAE,
                verified,
                compute_dtype=compute_dtype,
            )
        else:
            module = _load_verified_component(
                plan,
                _build_audio,
                verified,
                compute_dtype=compute_dtype,
            )
    return MiniMaxH3LoadedComponent(expected_role, module, plan, runtime_identity)


def load_minimax_h3_model(
    path: Path,
    *,
    asset: AssetRef,
    role: MiniMaxH3DiTRole,
    expected_identity: str,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backend: AttentionRole,
) -> MiniMaxH3Model:
    """Verify and load one H3 DiT as an ordinary MODEL component."""

    if type(asset) is not AssetRef:
        raise TypeError("MiniMax H3 model asset must be an AssetRef")
    if role not in ("fl2va-dit", "ref2va-dit"):
        raise ValueError(f"unsupported MiniMax H3 DiT role {role!r}")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("MiniMax H3 model requires an expected identity")
    if diffusion_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("MiniMax H3 diffusion dtype must be bfloat16 or float32")
    attention_selection = resolve_role_attention(
        attention_backend, attention_policy, attention_route_token
    )
    effective_attention_policy = attention_selection.status.requested_policy
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 DiT artifact is unavailable: {error}"
        ) from error
    with handle:
        _verify_artifact_size(handle, asset.size, role)
        source = load_safetensors_header_from_file(handle, path=path)
        artifact = _VerifiedArtifact(
            handle, _DescriptorPinnedSource(source, handle, asset.digest, asset.size)
        )
        plan = plan_minimax_h3_model_assembly(
            artifact.source,
            role=role,
            path=path,
            attention_policy=effective_attention_policy,
        )
        runtime_facts = plan.diffusion.identity_facts
        artifact_identity = minimax_h3_dit_runtime_identity(
            asset_digest=asset.digest,
            asset_size=asset.size,
            role=role,
            diffusion_dtype=_identity_dtype(diffusion_dtype).name,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            runtime_facts=runtime_facts,
        )
        if artifact_identity != expected_identity:
            raise MiniMaxH3SplitAssemblyError(
                f"expected model identity {expected_identity!r}, constructed {artifact_identity!r}"
            )
        diffusion = _load_verified_diffusion(
            plan.diffusion,
            artifact,
            compute_dtype=diffusion_dtype,
            attention_selection=attention_selection,
        )
    assembled = AssembledMiniMaxH3Model(
        diffusion,
        _component_compute_dtypes=MappingProxyType({"diffusion": diffusion_dtype}),
        attention_status=MappingProxyType({"flux": attention_selection.status}),
    )
    return MiniMaxH3Model(
        assembled,
        expected_identity,
        role,
        runtime_facts,
    )


def minimax_h3_guidance_receipt_identity(
    plan: MiniMaxH3SplitAssemblyPlan | MiniMaxH3ModelAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Build a dual-GPU H3 measurement identity, not an assembly prerequisite."""

    return guidance_receipt_identity(
        MINIMAX_H3_CONFIG.family_id,
        minimax_h3_guidance_integration_facts(cast("ComponentPlan[object]", plan.diffusion)),
        diffusion_dtype,
        2,
    )


__all__ = [
    "AssembledMiniMaxH3Model",
    "MiniMaxH3ArtifactPaths",
    "MiniMaxH3DiTRole",
    "MiniMaxH3LoadedComponent",
    "MiniMaxH3Model",
    "MiniMaxH3ModelAssemblyPlan",
    "MiniMaxH3SplitAssemblyError",
    "MiniMaxH3SplitAssemblyPlan",
    "load_minimax_h3_component",
    "load_minimax_h3_model",
    "minimax_h3_audio_vae_runtime_identity",
    "minimax_h3_conditioner_runtime_identity",
    "minimax_h3_guidance_receipt_identity",
    "minimax_h3_video_vae_runtime_identity",
    "plan_minimax_h3_split_assembly",
    "plan_minimax_h3_model_assembly",
    "verify_minimax_h3_artifacts",
    "worker_planning_context",
]
