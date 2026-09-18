"""Torch-free split-checkpoint planning for MiniMax H3.

Plans the exact official split graph (one or both DiTs, the INT8
conditioner, and both codecs) from safetensors headers alone. Worker
environment facts that enter runtime identity (torch and comfy-kitchen
versions) arrive through an explicit :class:`NativePlanningContext`;
asset identity arrives on the sources themselves through the
AssetIdentifiedSource seam. Payload reading, artifact verification
against open descriptors, and module assembly stay in
``dinkster_inference_torch.minimax_h3_assembly``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from dinkster_protocol import AttentionPolicy

from .assembly import ComponentPlan, NativePlanningContext
from .devices import BFLOAT16, DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .minimax_h3 import MINIMAX_H3_CONFIG
from .minimax_h3_codecs import (
    MiniMaxH3VideoVAEConfig,
    minimax_h3_audio_vae_layout,
    minimax_h3_video_vae_layout,
)
from .minimax_h3_conditioner import (
    MINIMAX_H3_CONDITIONER_CONFIG,
    MiniMaxH3ConditionerConfig,
    minimax_h3_conditioner_layout,
)
from .minimax_h3_dit import (
    MINIMAX_H3_DIT_PROVIDER_REVISION,
    MiniMaxH3DiTLayout,
    MiniMaxH3DiTRole,
    minimax_h3_dit_layout,
    minimax_h3_dit_provider_facts,
    plan_minimax_h3_dit_assembly,
)
from .quantization import LayerQuant, QuantizationError, split_quantization
from .weights import (
    AssetIdentifiedSource,
    ConfigurationPayloadSource,
    TensorGeometry,
    WeightEntry,
    WeightSource,
)

MINIMAX_H3_SPLIT_COMMON_ROLES = ("qwen3vl-32b-conditioner", "video-vae", "audio-vae")
MiniMaxH3CommonComponentRole = Literal["qwen3vl-32b-conditioner", "video-vae", "audio-vae"]

_INT8_VIDEO_VAE_PROVIDER_FACTS = (
    "artifact_provider=Kijai/MiniMax-H3-experimental@7f5705937cc106963a9dd77c322f7631e3610e89",
    "artifact_sha256=9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
)


class MiniMaxH3SplitAssemblyError(ValueError):
    """The split source set does not form one exact H3 runtime graph."""


def minimax_h3_component_runtime_identity(
    plan: ComponentPlan[object],
    role: MiniMaxH3CommonComponentRole,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded H3 component."""

    expected_components = {
        "qwen3vl-32b-conditioner": "conditioner",
        "video-vae": "video_vae",
        "audio-vae": "audio_vae",
    }
    component = expected_components[role]
    if plan.component != component:
        raise ValueError(f"MiniMax H3 {role} identity requires the {component} plan")
    return build_runtime_identity_from_facts(
        MINIMAX_H3_CONFIG.family_id,
        runtime_component_identity(MINIMAX_H3_CONFIG.family_id, (plan,)),
        diffusion_dtype="unloaded",
        text_dtype=compute_dtype.name if role == "qwen3vl-32b-conditioner" else "unloaded",
        vae_dtype=compute_dtype.name if role != "qwen3vl-32b-conditioner" else "unloaded",
        fp8_matmul=False,
        runtime_facts=plan.runtime_facts,
    )


def _int8_provider_facts(context: NativePlanningContext) -> tuple[str, ...]:
    return (
        "int8_provider=comfy-kitchen.int8_linear",
        f"comfy_kitchen_version={context.comfy_kitchen_version}",
    )


@dataclass(frozen=True)
class _GeometrySource:
    geometries: Mapping[str, TensorGeometry]

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def component_plan_claims(plan: ComponentPlan[object]) -> tuple[str, ...]:
    """Every source key the component will read, sorted."""

    claims = set(plan.keys.values())
    for quant in plan.quant.values():
        claims.add(quant.weight_scale)
        claims.update(quant.payloads.values())
        if quant.input_scale is not None:
            claims.add(quant.input_scale)
        if quant.weight_scale_2 is not None:
            claims.add(quant.weight_scale_2)
        if quant.pre_quant_scale is not None:
            claims.add(quant.pre_quant_scale)
        if quant.config is not None:
            claims.add(quant.config)
    return tuple(sorted(claims))


@dataclass(frozen=True, slots=True)
class MiniMaxH3ModelAssemblyPlan:
    """One independently loadable H3 diffusion model."""

    diffusion_role: MiniMaxH3DiTRole
    diffusion: ComponentPlan[MiniMaxH3DiTLayout]
    claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.diffusion_role not in ("fl2va-dit", "ref2va-dit"):
            raise ValueError(f"unsupported MiniMax H3 DiT role {self.diffusion_role!r}")
        if self.diffusion.component != "diffusion":
            raise ValueError("MiniMax H3 model plan must own the diffusion component")
        if self.claims != component_plan_claims(cast("ComponentPlan[object]", self.diffusion)):
            raise ValueError("MiniMax H3 model claims must be exact")


def _source_path(source: WeightSource, role: str) -> Path:
    path = getattr(source, "path", None)
    if not isinstance(path, Path):
        raise TypeError(f"MiniMax H3 {role} source must expose a pathlib.Path")
    return path


def _require_exact_source(
    source: WeightSource,
    role: str,
    layout: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, DType], tuple[str, ...]]:
    keys = tuple(source.keys())
    if len(keys) != len(set(keys)):
        raise MiniMaxH3SplitAssemblyError(f"duplicate MiniMax H3 {role} keys")
    actual = set(keys)
    expected = set(layout)
    if actual != expected:
        missing = sorted(expected - actual)
        foreign = sorted(actual - expected)
        detail = missing[:1] if missing else foreign[:1]
        kind = "missing" if missing else "foreign"
        raise MiniMaxH3SplitAssemblyError(f"MiniMax H3 {role} has {kind} keys: {', '.join(detail)}")
    storage: dict[str, DType] = {}
    for key, shape in layout.items():
        geometry = source.entry(key).geometry
        if geometry.shape != shape:
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {role} geometry mismatch for {key}: "
                f"got {geometry.shape}, expected {shape}"
            )
        if geometry.dtype.kind != "float":
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {role} weight {key} requires floating storage, "
                f"got {geometry.dtype.name}"
            )
        storage[key] = geometry.dtype
    return storage, tuple(sorted(keys))


def _extract_quantized_source(
    source: WeightSource,
    role: str,
    layout: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, DType], dict[str, LayerQuant], tuple[str, ...]]:
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    try:
        split = split_quantization(
            geometries,
            source.metadata(),
            payload_reader=(
                source.read_uint8_configuration
                if isinstance(source, ConfigurationPayloadSource)
                else None
            ),
        )
    except QuantizationError as error:
        raise MiniMaxH3SplitAssemblyError(f"MiniMax H3 {role}: {error}") from error
    actual = set(split.architecture)
    expected = set(layout)
    if actual != expected:
        missing = sorted(expected - actual)
        foreign = sorted(actual - expected)
        detail = missing[:1] if missing else foreign[:1]
        kind = "missing" if missing else "foreign"
        raise MiniMaxH3SplitAssemblyError(f"MiniMax H3 {role} has {kind} keys: {', '.join(detail)}")
    storage: dict[str, DType] = {}
    quantized_weights = {entry.weight for entry in split.layers.values()}
    for key, shape in layout.items():
        geometry = split.architecture[key]
        if geometry.shape != shape:
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {role} geometry mismatch for {key}: "
                f"got {geometry.shape}, expected {shape}"
            )
        if key not in quantized_weights and geometry.dtype.kind != "float":
            raise MiniMaxH3SplitAssemblyError(
                f"MiniMax H3 {role} weight {key} requires floating storage, "
                f"got {geometry.dtype.name}"
            )
        storage[key] = geometry.dtype
    return storage, dict(split.layers), tuple(sorted(source.keys()))


def _component_plan(
    *,
    component: str,
    path: Path,
    config: object,
    storage: Mapping[str, DType],
    quant: Mapping[str, LayerQuant] = MappingProxyType({}),
    identity_facts: tuple[str, ...] = (),
) -> ComponentPlan[object]:
    return ComponentPlan(
        component=component,
        path=path,
        config=config,
        keys={key: key for key in storage},
        dtypes=storage,
        quant=quant,
        identity_facts=(
            f"provider_revision={MINIMAX_H3_DIT_PROVIDER_REVISION}",
            *identity_facts,
        ),
    )


def plan_minimax_h3_diffusion_component(
    diffusion: WeightSource,
    role: MiniMaxH3DiTRole,
    path: Path,
    *,
    context: NativePlanningContext,
    identity_facts: tuple[str, ...] = (),
    attention_policy: AttentionPolicy = "auto",
) -> tuple[ComponentPlan[MiniMaxH3DiTLayout], tuple[str, ...]]:
    """Plan one H3 DiT from its header; returns the plan and its claims."""

    if _source_path(diffusion, role) != path:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {role} source path differs from artifact selection"
        )
    diffusion_layout = minimax_h3_dit_layout(
        time_embedding_kind="curve" if "adaln_t_table" in diffusion.keys() else "mlp"
    )
    storage, quantized, claims = _extract_quantized_source(
        diffusion,
        "diffusion",
        diffusion_layout.keys,
    )
    logical_dtypes = dict(storage)
    for quant in quantized.values():
        logical_dtypes[quant.layer + ".weight"] = BFLOAT16
    source_plan = plan_minimax_h3_dit_assembly(
        _GeometrySource(
            {
                key: TensorGeometry(shape, logical_dtypes[key])
                for key, shape in diffusion_layout.keys.items()
            }
        )
    )
    if source_plan.source_prefix:
        raise MiniMaxH3SplitAssemblyError("official split H3 DiT must use bare keys")
    return (
        ComponentPlan(
            component="diffusion",
            path=path,
            config=source_plan.layout,
            keys=source_plan.keys,
            dtypes=storage,
            quant=quantized,
            identity_facts=minimax_h3_dit_provider_facts(
                role,
                quantized=bool(quantized),
                torch_version=context.torch_version,
                comfy_kitchen_version=(
                    context.comfy_kitchen_version
                    if quantized or attention_policy in ("comfy_kitchen_int8", "sol")
                    else None
                ),
                attention_policy=attention_policy,
            )
            + identity_facts,
        ),
        claims,
    )


def plan_minimax_h3_conditioner_component(
    conditioner: WeightSource,
    path: Path,
    *,
    context: NativePlanningContext,
    identity_facts: tuple[str, ...] = (),
) -> tuple[ComponentPlan[MiniMaxH3ConditionerConfig], tuple[str, ...]]:
    """Plan the shared INT8 conditioner from its header."""

    storage, quant, claims = _extract_quantized_source(
        conditioner, "conditioner", minimax_h3_conditioner_layout().keys
    )
    plan = cast(
        "ComponentPlan[MiniMaxH3ConditionerConfig]",
        _component_plan(
            component="conditioner",
            path=path,
            config=MINIMAX_H3_CONDITIONER_CONFIG,
            storage=storage,
            quant=quant,
            identity_facts=(*_int8_provider_facts(context), *identity_facts),
        ),
    )
    return plan, claims


def plan_minimax_h3_video_vae_component(
    video_vae: WeightSource,
    path: Path,
    *,
    context: NativePlanningContext,
    identity_facts: tuple[str, ...] = (),
) -> tuple[ComponentPlan[MiniMaxH3VideoVAEConfig], tuple[str, ...]]:
    """Plan the shared video VAE from its header."""

    storage, quant, claims = _extract_quantized_source(
        video_vae, "video VAE", minimax_h3_video_vae_layout()
    )
    plan = cast(
        "ComponentPlan[MiniMaxH3VideoVAEConfig]",
        _component_plan(
            component="video_vae",
            path=path,
            config=MiniMaxH3VideoVAEConfig(),
            storage=storage,
            quant=quant,
            identity_facts=(
                *(
                    (*_INT8_VIDEO_VAE_PROVIDER_FACTS, *_int8_provider_facts(context))
                    if quant
                    else ()
                ),
                *identity_facts,
            ),
        ),
    )
    return plan, claims


def plan_minimax_h3_audio_vae_component(
    audio_vae: WeightSource,
    path: Path,
    *,
    identity_facts: tuple[str, ...] = (),
) -> tuple[ComponentPlan[None], tuple[str, ...]]:
    """Plan the shared audio VAE from its header."""

    storage, claims = _require_exact_source(audio_vae, "audio VAE", minimax_h3_audio_vae_layout())
    plan = cast(
        "ComponentPlan[None]",
        _component_plan(
            component="audio_vae",
            path=path,
            config=None,
            storage=storage,
            identity_facts=identity_facts,
        ),
    )
    return plan, claims


def plan_minimax_h3_common_component(
    source: WeightSource,
    *,
    role: MiniMaxH3CommonComponentRole,
    path: Path,
    context: NativePlanningContext,
) -> ComponentPlan[object]:
    """Plan one independently loaded common component with official asset identity."""

    if _source_path(source, role) != path:
        raise MiniMaxH3SplitAssemblyError(
            f"MiniMax H3 {role} source path differs from artifact selection"
        )
    identity_facts = _identified_facts(source, role)
    if role == "qwen3vl-32b-conditioner":
        plan, _claims = plan_minimax_h3_conditioner_component(
            source,
            path,
            context=context,
            identity_facts=identity_facts,
        )
    elif role == "video-vae":
        plan, _claims = plan_minimax_h3_video_vae_component(
            source,
            path,
            context=context,
            identity_facts=identity_facts,
        )
    else:
        plan, _claims = plan_minimax_h3_audio_vae_component(
            source,
            path,
            identity_facts=identity_facts,
        )
    return cast("ComponentPlan[object]", plan)


def plan_minimax_h3_model_assembly(
    diffusion: WeightSource,
    *,
    role: MiniMaxH3DiTRole,
    path: Path,
    context: NativePlanningContext,
    attention_policy: AttentionPolicy = "auto",
) -> MiniMaxH3ModelAssemblyPlan:
    """Plan one role-specific H3 DiT without requiring unrelated assets."""

    diffusion_plan, claims = plan_minimax_h3_diffusion_component(
        diffusion, role, path, context=context, attention_policy=attention_policy
    )
    return MiniMaxH3ModelAssemblyPlan(role, diffusion_plan, claims)


def _identified_facts(source: WeightSource, role: str) -> tuple[str, ...]:
    digest: str | None = None
    size: int | None = None
    if isinstance(source, AssetIdentifiedSource):
        digest = source.asset_digest
        size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise MiniMaxH3SplitAssemblyError(f"MiniMax H3 {role} source must carry asset identity")
    return (
        f"asset_digest={digest}",
        f"asset_size={size}",
    )


__all__ = [
    "MINIMAX_H3_SPLIT_COMMON_ROLES",
    "MiniMaxH3CommonComponentRole",
    "MiniMaxH3ModelAssemblyPlan",
    "MiniMaxH3SplitAssemblyError",
    "component_plan_claims",
    "minimax_h3_component_runtime_identity",
    "plan_minimax_h3_audio_vae_component",
    "plan_minimax_h3_conditioner_component",
    "plan_minimax_h3_common_component",
    "plan_minimax_h3_diffusion_component",
    "plan_minimax_h3_model_assembly",
    "plan_minimax_h3_video_vae_component",
]
