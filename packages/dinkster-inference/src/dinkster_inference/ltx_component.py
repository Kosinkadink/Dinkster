"""Classic LTX-Video per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import (
    LTXVStandaloneComponentPlan,
    LTXVStandaloneComponentRole,
    plan_ltxv_standalone_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class LTXVComponentAssemblyError(ValueError):
    """A source is not the exact classic LTX-Video component requested."""


def plan_ltxv_split_component(
    source: WeightSource,
    *,
    role: LTXVStandaloneComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> LTXVStandaloneComponentPlan:
    """Plan one split component and bind its immutable asset identity."""
    source_path = getattr(source, "path", None)
    if source_path != path:
        raise LTXVComponentAssemblyError(f"LTX-Video {role} source path differs from selection")
    if not bind_asset_identity:
        return plan_ltxv_standalone_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise LTXVComponentAssemblyError(f"LTX-Video {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise LTXVComponentAssemblyError(f"LTX-Video {role} source must carry asset identity")
    try:
        planned = plan_ltxv_standalone_component(source, role)
    except ValueError as error:
        raise LTXVComponentAssemblyError(f"LTX-Video {role}: {error}") from error
    component = replace(
        planned.component,
        identity_facts=(
            *planned.component.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )
    return replace(planned, component=component)


def ltxv_component_runtime_identity(
    planned: LTXVStandaloneComponentPlan,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded component."""
    role = planned.role
    component = planned.component
    return build_runtime_identity_from_facts(
        "dinkster.ltxv",
        runtime_component_identity("dinkster.ltxv", (component,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "t5xxl" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=component.runtime_facts,
    )


__all__ = [
    "LTXVComponentAssemblyError",
    "ltxv_component_runtime_identity",
    "plan_ltxv_split_component",
]
