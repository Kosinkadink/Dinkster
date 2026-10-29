"""Ideogram 4 per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import ComponentPlan, Ideogram4ComponentRole, plan_ideogram4_component
from .devices import FLOAT8_E4M3, DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class Ideogram4ComponentAssemblyError(ValueError):
    """A source is not the exact Ideogram 4 component requested."""


def plan_ideogram4_split_component(
    source: WeightSource,
    *,
    role: Ideogram4ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    source_path = getattr(source, "path", None)
    if source_path != path:
        raise Ideogram4ComponentAssemblyError(f"Ideogram 4 {role} source path differs")
    if not bind_asset_identity:
        return plan_ideogram4_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise Ideogram4ComponentAssemblyError(f"Ideogram 4 {role} source needs asset identity")
    if not source.asset_digest or type(source.asset_size) is not int or source.asset_size < 0:
        raise Ideogram4ComponentAssemblyError(f"Ideogram 4 {role} source needs asset identity")
    try:
        planned = plan_ideogram4_component(source, role)
    except ValueError as error:
        raise Ideogram4ComponentAssemblyError(f"Ideogram 4 {role}: {error}") from error
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={source.asset_digest}",
            f"asset_size={source.asset_size}",
        ),
    )


def ideogram4_component_uses_fp8_matmul(
    role: Ideogram4ComponentRole,
    component: ComponentPlan[object],
) -> bool:
    """Whether this component's checkpoint selects native FP8 matmul."""
    return role == "diffusion" and FLOAT8_E4M3 in component.dtypes.values()


def ideogram4_component_runtime_identity(
    planned: ComponentPlan[object],
    role: Ideogram4ComponentRole,
    compute_dtype: DType,
) -> str:
    if planned.component != role:
        raise ValueError(f"Ideogram 4 {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.ideogram4",
        runtime_component_identity("dinkster.ideogram4", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "qwen3vl_8b" else "unloaded",
        vae_dtype="unloaded",
        fp8_matmul=ideogram4_component_uses_fp8_matmul(role, planned),
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "Ideogram4ComponentAssemblyError",
    "ideogram4_component_runtime_identity",
    "ideogram4_component_uses_fp8_matmul",
    "plan_ideogram4_split_component",
]
