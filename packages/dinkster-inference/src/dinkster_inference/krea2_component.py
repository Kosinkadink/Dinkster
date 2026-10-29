"""Krea 2 per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import ComponentPlan, Krea2ComponentRole, plan_krea2_component
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class Krea2ComponentAssemblyError(ValueError):
    """A source is not the exact Krea 2 component requested."""


def plan_krea2_split_component(
    source: WeightSource,
    *,
    role: Krea2ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    """Plan one split Krea 2 component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise Krea2ComponentAssemblyError(f"Krea 2 {role} source path differs from selection")
    if not bind_asset_identity:
        return plan_krea2_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise Krea2ComponentAssemblyError(f"Krea 2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Krea2ComponentAssemblyError(f"Krea 2 {role} source must carry asset identity")
    try:
        planned = plan_krea2_component(source, role)
    except ValueError as error:
        raise Krea2ComponentAssemblyError(f"Krea 2 {role}: {error}") from error
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def krea2_component_runtime_identity(
    planned: ComponentPlan[object],
    role: Krea2ComponentRole,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Krea 2 component."""

    if planned.component != role:
        raise ValueError(f"Krea 2 {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.krea2",
        runtime_component_identity("dinkster.krea2", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "qwen3vl_4b" else "unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "Krea2ComponentAssemblyError",
    "krea2_component_runtime_identity",
    "plan_krea2_split_component",
]
