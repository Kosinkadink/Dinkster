"""SeedVR2 per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import ComponentPlan, SeedVR2ComponentRole, plan_seedvr2_component
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class SeedVR2ComponentAssemblyError(ValueError):
    """A source is not the exact SeedVR2 component requested."""


def plan_seedvr2_split_component(
    source: WeightSource,
    *,
    role: SeedVR2ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    source_path = getattr(source, "path", None)
    if source_path != path:
        raise SeedVR2ComponentAssemblyError(f"SeedVR2 {role} source path differs from selection")
    if not bind_asset_identity:
        return plan_seedvr2_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise SeedVR2ComponentAssemblyError(f"SeedVR2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise SeedVR2ComponentAssemblyError(f"SeedVR2 {role} source must carry asset identity")
    try:
        planned = plan_seedvr2_component(source, role)
    except ValueError as error:
        raise SeedVR2ComponentAssemblyError(f"SeedVR2 {role}: {error}") from error
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def seedvr2_component_runtime_identity(
    planned: ComponentPlan[object],
    role: SeedVR2ComponentRole,
    compute_dtype: DType,
) -> str:
    if planned.component != role:
        raise ValueError(f"SeedVR2 {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.seedvr2",
        runtime_component_identity("dinkster.seedvr2", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype="unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "SeedVR2ComponentAssemblyError",
    "plan_seedvr2_split_component",
    "seedvr2_component_runtime_identity",
]
