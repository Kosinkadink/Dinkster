"""Anima per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import AnimaComponentRole, ComponentPlan, plan_anima_component
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class AnimaComponentAssemblyError(ValueError):
    """A source is not the exact Anima component requested."""


def plan_anima_split_component(
    source: WeightSource,
    *,
    role: AnimaComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    """Plan one split Anima component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise AnimaComponentAssemblyError(f"Anima {role} source path differs from selection")
    if not bind_asset_identity:
        return plan_anima_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise AnimaComponentAssemblyError(f"Anima {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise AnimaComponentAssemblyError(f"Anima {role} source must carry asset identity")
    try:
        planned = plan_anima_component(source, role)
    except ValueError as error:
        raise AnimaComponentAssemblyError(f"Anima {role}: {error}") from error
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def anima_component_runtime_identity(
    planned: ComponentPlan[object],
    role: AnimaComponentRole,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Anima component."""

    if planned.component != role:
        raise ValueError(f"Anima {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.anima",
        runtime_component_identity("dinkster.anima", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "qwen3_06b" else "unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "AnimaComponentAssemblyError",
    "anima_component_runtime_identity",
    "plan_anima_split_component",
]
