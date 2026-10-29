"""Flux2 per-component artifact planning and identity.

Admission is geometry-only: any safetensors file whose weight geometry
detects as a Flux2 component is admitted, and the file's immutable asset
digest and size are bound into the component's identity facts so two
byte-different files with the same geometry never share an identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .assembly import (
    AssemblyError,
    ComponentPlan,
    Flux2AssemblyPlan,
    Flux2ComponentRole,
    plan_flux2_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource

if TYPE_CHECKING:
    from .component_checkpoint import ComponentCheckpointPlan

#: The Flux2 component roles that carry a text encoder.
FLUX2_TEXT_COMPONENT_ROLES: tuple[Flux2ComponentRole, ...] = (
    "mistral3_24b",
    "qwen3_8b",
    "qwen3_4b",
)


class Flux2ComponentAssemblyError(ValueError):
    """A source is not the Flux2 component requested."""


def flux2_checkpoint_assembly(plan: ComponentCheckpointPlan) -> Flux2AssemblyPlan:
    """Validate the declared DiT, conditioning tower and packed codec composition."""
    components = cast("Mapping[str, ComponentPlan[Any]]", plan.components)
    try:
        (text_role,) = plan.descriptor.text_encoder_roles
        return Flux2AssemblyPlan(
            plan.family,
            components["diffusion"],
            components[text_role],
            components["vae"],
            plan.unclaimed,
        )
    except (KeyError, ValueError) as error:
        raise AssemblyError(f"incomplete or incompatible Flux2 checkpoint: {error}") from error


@dataclass(frozen=True)
class Flux2PlannedComponent:
    """One planned Flux2 component and the identity family it belongs to.

    ``family_id`` is the detected variant family for the diffusion and
    text roles and the shared Flux2 scope for the VAE.
    """

    role: Flux2ComponentRole
    family_id: str
    plan: ComponentPlan[object]

    def __post_init__(self) -> None:
        if self.plan.component != self.role:
            raise ValueError(f"Flux2 {self.role} planned component requires the {self.role} plan")


def plan_flux2_split_component(
    source: WeightSource,
    *,
    role: Flux2ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> Flux2PlannedComponent:
    """Plan one split Flux2 component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise Flux2ComponentAssemblyError(
            f"Flux2 {role} source path differs from artifact selection"
        )
    if not bind_asset_identity:
        family_id, plan = plan_flux2_component(source, role)
        return Flux2PlannedComponent(role, family_id, plan)
    if not isinstance(source, AssetIdentifiedSource):
        raise Flux2ComponentAssemblyError(f"Flux2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Flux2ComponentAssemblyError(f"Flux2 {role} source must carry asset identity")
    try:
        family_id, plan = plan_flux2_component(source, role)
    except ValueError as error:
        raise Flux2ComponentAssemblyError(f"Flux2 {role}: {error}") from error
    plan = replace(
        plan,
        identity_facts=(
            *plan.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )
    return Flux2PlannedComponent(role, family_id, plan)


def flux2_component_runtime_identity(
    planned: Flux2PlannedComponent,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Flux2 component."""

    role = planned.role
    return build_runtime_identity_from_facts(
        planned.family_id,
        runtime_component_identity(planned.family_id, (planned.plan,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role in FLUX2_TEXT_COMPONENT_ROLES else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=planned.plan.runtime_facts,
    )


__all__ = [
    "FLUX2_TEXT_COMPONENT_ROLES",
    "Flux2ComponentAssemblyError",
    "Flux2PlannedComponent",
    "flux2_component_runtime_identity",
    "plan_flux2_split_component",
]
