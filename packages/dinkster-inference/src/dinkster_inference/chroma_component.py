"""Chroma per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import (
    FLUX_DIFFUSION_PREFIX,
    FLUX_T5XXL_PREFIX,
    FLUX_VAE_PREFIX,
    AssemblyError,
    ComponentPlan,
    _component_source,  # pyright: ignore[reportPrivateUsage]
    _kl_conversion,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from .autoencoder_kl import KLConfig, detect_kl_config
from .chroma import (
    CHROMA_FAMILY_ID,
    ChromaFamilyConfig,
    detect_chroma_config,
    normalize_chroma_keys,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .t5_text import T5_TEXT_OPTIONAL_KEYS, T5_XXL_CONFIG, T5Config, detect_t5_config
from .weights import AssetIdentifiedSource, WeightSource

ChromaComponentRole = Literal["diffusion", "t5xxl", "vae"]


class ChromaComponentAssemblyError(ValueError):
    """A source is not the exact Chroma component requested."""


def _plan_diffusion(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[ChromaFamilyConfig]:
    extracted = _component_source(
        "diffusion",
        source,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    try:
        config = detect_chroma_config(extracted.geometries)
        normalized = normalize_chroma_keys(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"diffusion: {error}") from error
    renames: dict[str, str] = {}
    for key in extracted.geometries:
        candidate = key.replace("distilled_guidance_layer.0.", "distilled_guidance_layer.", 1)
        if candidate.endswith(".scale"):
            candidate = candidate[: -len(".scale")] + ".weight"
        if candidate != key:
            renames[key] = candidate
    if set(normalized) != set(renames.get(key, key) for key in extracted.geometries):
        raise AssemblyError("diffusion: Chroma key normalization is ambiguous")
    return _plan("diffusion", extracted, config, renames=renames)


def _plan_t5(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[T5Config]:
    extracted = _component_source(
        "t5xxl",
        source,
        checkpoint,
        FLUX_T5XXL_PREFIX,
        root="text_encoders.t5xxl.",
    )
    if extracted.ignored:
        raise AssemblyError(
            "t5xxl: source contains unsupported tensors: " + ", ".join(extracted.ignored[:3])
        )
    try:
        config = detect_t5_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"t5xxl: {error}") from error
    if config != T5_XXL_CONFIG:
        raise AssemblyError("t5xxl: Chroma requires the classic T5-XXL layout")
    return _plan("t5xxl", extracted, config, drop=T5_TEXT_OPTIONAL_KEYS)


def _plan_vae(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[KLConfig]:
    extracted = _component_source(
        "vae",
        source,
        checkpoint,
        FLUX_VAE_PREFIX,
        split_prefixes=("", FLUX_VAE_PREFIX),
    )
    try:
        config = detect_kl_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"vae: {error}") from error
    if config.latent_channels != 16 or config.batch_norm_latent:
        raise AssemblyError("vae: Chroma requires the 16-channel Flux KL VAE")
    renames, transforms = _kl_conversion(extracted.geometries)
    return _plan("vae", extracted, config, renames=renames, transforms=transforms)


def plan_chroma_component(
    source: WeightSource,
    role: ChromaComponentRole,
) -> ComponentPlan[object]:
    """Plan one independently supplied Chroma component."""
    if role == "diffusion":
        return cast("ComponentPlan[object]", _plan_diffusion(source, None))
    if role == "t5xxl":
        return cast("ComponentPlan[object]", _plan_t5(source, None))
    return cast("ComponentPlan[object]", _plan_vae(source, None))


def chroma_component_family_id(plan: ComponentPlan[object]) -> str:
    if plan.component == "diffusion":
        config = cast("ChromaFamilyConfig", plan.config)
        return config.family_id
    return CHROMA_FAMILY_ID


def plan_chroma_split_component(
    source: WeightSource,
    *,
    role: ChromaComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    """Plan one split Chroma component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise ChromaComponentAssemblyError(f"Chroma {role} source path differs from selection")
    if not bind_asset_identity:
        return plan_chroma_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise ChromaComponentAssemblyError(f"Chroma {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise ChromaComponentAssemblyError(f"Chroma {role} source must carry asset identity")
    try:
        planned = plan_chroma_component(source, role)
    except ValueError as error:
        raise ChromaComponentAssemblyError(f"Chroma {role}: {error}") from error
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def chroma_component_runtime_identity(
    planned: ComponentPlan[object],
    role: ChromaComponentRole,
    compute_dtype: DType,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> str:
    """Build the native identity for one independently loaded Chroma component."""

    if planned.component != role:
        raise ValueError(f"Chroma {role} identity requires the {role} plan")
    family_id = chroma_component_family_id(planned)
    return build_runtime_identity_from_facts(
        family_id,
        runtime_component_identity(family_id, (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "t5xxl" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "ChromaComponentAssemblyError",
    "ChromaComponentRole",
    "chroma_component_family_id",
    "chroma_component_runtime_identity",
    "plan_chroma_component",
    "plan_chroma_split_component",
]
