"""Classic Flux diffusion-component planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import (
    ComponentPlan,
    _detect_flux_family_normalized,  # pyright: ignore[reportPrivateUsage]
    _norm_renames,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from .devices import DType
from .flux import FluxConfig
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource

FluxComponentRole = Literal["diffusion"]


class FluxComponentAssemblyError(ValueError):
    """A source is not the exact classic Flux component requested."""


def plan_flux_component(source: WeightSource, role: FluxComponentRole) -> ComponentPlan[FluxConfig]:
    """Plan one independently supplied classic Flux diffusion model."""
    if role != "diffusion":
        raise FluxComponentAssemblyError(f"unsupported classic Flux component role {role!r}")
    try:
        extracted, config, _family = _detect_flux_family_normalized(diffusion=source)
    except ValueError as error:
        raise FluxComponentAssemblyError(f"classic Flux diffusion: {error}") from error
    return _plan(
        "diffusion",
        extracted,
        config,
        renames=_norm_renames(extracted.geometries),
    )


def plan_flux_split_component(
    source: WeightSource,
    *,
    role: FluxComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[FluxConfig]:
    """Plan one split classic Flux model and bind its immutable asset identity."""
    if getattr(source, "path", None) != path:
        raise FluxComponentAssemblyError("classic Flux source path differs from selection")
    planned = plan_flux_component(source, role)
    if not bind_asset_identity:
        return planned
    if not isinstance(source, AssetIdentifiedSource):
        raise FluxComponentAssemblyError("classic Flux source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise FluxComponentAssemblyError("classic Flux source must carry asset identity")
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def flux_component_runtime_identity(
    planned: ComponentPlan[FluxConfig],
    compute_dtype: DType,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> str:
    """Build the native identity for one classic Flux diffusion component."""
    family_id = "dinkster.flux_dev" if planned.config.guidance_embed else "dinkster.flux_schnell"
    return build_runtime_identity_from_facts(
        family_id,
        runtime_component_identity(family_id, (planned,)),
        diffusion_dtype=compute_dtype.name,
        text_dtype="unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "FluxComponentAssemblyError",
    "FluxComponentRole",
    "flux_component_runtime_identity",
    "plan_flux_component",
    "plan_flux_split_component",
]
