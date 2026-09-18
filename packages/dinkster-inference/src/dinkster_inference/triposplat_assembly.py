"""TripoSplat per-component artifact planning and identity.

Admission is geometry-only: any safetensors file whose weight geometry
detects as a TripoSplat component is admitted, and the file's immutable
asset digest and size are bound into the component's identity facts so
two byte-different files with the same geometry never share an identity.
All three components carry the shared ``dinkster.triposplat`` family scope,
so composed executions agree on one family and identity contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .assembly import ComponentPlan, TripoSplatComponentRole, plan_triposplat_component
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource

#: The shared identity family for every TripoSplat component.
TRIPOSPLAT_COMPONENT_FAMILY_ID = "dinkster.triposplat"


class TripoSplatComponentAssemblyError(ValueError):
    """A source is not the TripoSplat component requested."""


@dataclass(frozen=True)
class TripoSplatPlannedComponent:
    """One planned TripoSplat component under the shared family scope."""

    role: TripoSplatComponentRole
    plan: ComponentPlan[object]
    family_id: str = field(default=TRIPOSPLAT_COMPONENT_FAMILY_ID, init=False)

    def __post_init__(self) -> None:
        if self.plan.component != self.role:
            raise ValueError(
                f"TripoSplat {self.role} planned component requires the {self.role} plan"
            )


def plan_triposplat_split_component(
    source: WeightSource,
    *,
    role: TripoSplatComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> TripoSplatPlannedComponent:
    """Plan one split TripoSplat component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise TripoSplatComponentAssemblyError(
            f"TripoSplat {role} source path differs from artifact selection"
        )
    if not bind_asset_identity:
        return TripoSplatPlannedComponent(role, plan_triposplat_component(source, role))
    if not isinstance(source, AssetIdentifiedSource):
        raise TripoSplatComponentAssemblyError(
            f"TripoSplat {role} source must carry asset identity"
        )
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise TripoSplatComponentAssemblyError(
            f"TripoSplat {role} source must carry asset identity"
        )
    try:
        plan = plan_triposplat_component(source, role)
    except ValueError as error:
        raise TripoSplatComponentAssemblyError(f"TripoSplat {role}: {error}") from error
    plan = replace(
        plan,
        identity_facts=(
            *plan.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )
    return TripoSplatPlannedComponent(role, plan)


def triposplat_component_runtime_identity(
    planned: TripoSplatPlannedComponent,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded TripoSplat component."""

    role = planned.role
    return build_runtime_identity_from_facts(
        planned.family_id,
        runtime_component_identity(planned.family_id, (planned.plan,)),
        diffusion_dtype=compute_dtype.name if role == "dit" else "unloaded",
        text_dtype=compute_dtype.name if role == "dinov3-vision-conditioner" else "unloaded",
        vae_dtype=compute_dtype.name if role == "gaussian-decoder" else "unloaded",
        fp8_matmul=False,
        runtime_facts=planned.plan.runtime_facts,
    )


__all__ = [
    "TRIPOSPLAT_COMPONENT_FAMILY_ID",
    "TripoSplatComponentAssemblyError",
    "TripoSplatPlannedComponent",
    "plan_triposplat_split_component",
    "triposplat_component_runtime_identity",
]
