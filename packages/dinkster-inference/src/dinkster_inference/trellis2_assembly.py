"""TRELLIS.2 per-artifact planning and runtime identity."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeVar, cast

from .assembly import (
    ComponentPlan,
    Trellis2FlowRole,
    Trellis2ModelPlan,
    Trellis2SplitModelPlan,
    Trellis2VisionPlan,
    plan_trellis2_decoder_component,
    plan_trellis2_flow_component,
    plan_trellis2_model,
    plan_trellis2_vision,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .trellis2 import TRELLIS2_FAMILY_ID
from .weights import AssetIdentifiedSource, WeightSource

Trellis2ArtifactRole = Literal[
    "diffusion",
    "vision",
    "structure-decoder",
    "shape-decoder",
    "texture-decoder",
]
Trellis2ArtifactPlan = Trellis2ModelPlan | Trellis2VisionPlan | ComponentPlan[object]
C = TypeVar("C")

TRELLIS2_SPLIT_PROVIDER_REVISION = "microsoft/TRELLIS.2-4B@af44b45f2e35a493886929c6d786e563ec68364d"


class Trellis2AssemblyError(ValueError):
    """A source is not the requested TRELLIS.2 artifact."""


@dataclass(frozen=True)
class Trellis2PlannedArtifact:
    """One independently loaded artifact under the TRELLIS.2 family scope."""

    role: Trellis2ArtifactRole
    plan: Trellis2ArtifactPlan
    family_id: str = field(default=TRELLIS2_FAMILY_ID, init=False)

    @property
    def identity_components(self) -> tuple[ComponentPlan[object], ...]:
        if type(self.plan) is Trellis2ModelPlan:
            return cast("tuple[ComponentPlan[object], ...]", self.plan.identity_components)
        if type(self.plan) is Trellis2VisionPlan:
            return cast("tuple[ComponentPlan[object], ...]", self.plan.identity_components)
        component = cast("ComponentPlan[object]", self.plan)
        return (component,)

    def __post_init__(self) -> None:
        components = self.identity_components
        expected = {
            "diffusion": ("structure", "shape", "shape-512", "texture"),
            "vision": ("dino", "naf"),
            "structure-decoder": ("structure-decoder",),
            "shape-decoder": ("shape-decoder",),
            "texture-decoder": ("texture-decoder",),
        }[self.role]
        actual = tuple(component.component for component in components)
        if self.role == "vision" and actual == ("dino",):
            return
        if actual != expected:
            raise ValueError(
                f"TRELLIS.2 {self.role} artifact requires components {expected}, got {actual}"
            )


def _bind_asset(plan: ComponentPlan[C], digest: str, size: int) -> ComponentPlan[C]:
    return replace(
        plan,
        identity_facts=(
            *plan.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def plan_trellis2_flow_artifact(
    source: WeightSource,
    *,
    role: Trellis2FlowRole,
    path: Path,
) -> ComponentPlan[object]:
    """Plan one split flow and bind its immutable artifact identity."""

    if getattr(source, "path", None) != path:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source path differs from artifact selection")
    if not isinstance(source, AssetIdentifiedSource):
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source must carry asset identity")
    try:
        plan = cast(
            "ComponentPlan[object]",
            _bind_asset(plan_trellis2_flow_component(source, role), digest, size),
        )
    except ValueError as error:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role}: {error}") from error
    return replace(
        plan,
        identity_facts=(
            *plan.identity_facts,
            f"provider_revision={TRELLIS2_SPLIT_PROVIDER_REVISION}",
        ),
    )


def plan_trellis2_artifact(
    source: WeightSource,
    *,
    role: Trellis2ArtifactRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> Trellis2PlannedArtifact:
    """Plan one TRELLIS.2 artifact and bind its immutable asset identity."""

    if getattr(source, "path", None) != path:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source path differs from artifact selection")
    if not bind_asset_identity:
        if role == "diffusion":
            geometry: Trellis2ArtifactPlan = plan_trellis2_model(source)
        elif role == "vision":
            geometry = plan_trellis2_vision(source)
        else:
            geometry = cast("ComponentPlan[object]", plan_trellis2_decoder_component(source, role))
        return Trellis2PlannedArtifact(role, geometry)
    if not isinstance(source, AssetIdentifiedSource):
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role} source must carry asset identity")
    try:
        if role == "diffusion":
            model = plan_trellis2_model(source)
            bound: Trellis2ArtifactPlan = Trellis2ModelPlan(
                *(_bind_asset(part, digest, size) for part in model.identity_components)
            )
        elif role == "vision":
            vision = plan_trellis2_vision(source)
            bound = Trellis2VisionPlan(
                _bind_asset(vision.dino, digest, size),
                None if vision.naf is None else _bind_asset(vision.naf, digest, size),
            )
        else:
            bound = _bind_asset(
                cast("ComponentPlan[object]", plan_trellis2_decoder_component(source, role)),
                digest,
                size,
            )
    except ValueError as error:
        raise Trellis2AssemblyError(f"TRELLIS.2 {role}: {error}") from error
    return Trellis2PlannedArtifact(role, bound)


def trellis2_artifact_runtime_identity(
    planned: Trellis2PlannedArtifact,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded artifact."""

    role = planned.role
    components = planned.identity_components
    runtime_facts = tuple(sorted({fact for plan in components for fact in plan.runtime_facts}))
    return build_runtime_identity_from_facts(
        planned.family_id,
        runtime_component_identity(planned.family_id, components),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "vision" else "unloaded",
        vae_dtype=compute_dtype.name if role.endswith("decoder") else "unloaded",
        fp8_matmul=False,
        runtime_facts=runtime_facts,
    )


def trellis2_split_model_runtime_identity(
    plan: Trellis2SplitModelPlan,
    compute_dtype: DType,
) -> str:
    """Build the native identity for Microsoft's five split flow artifacts."""

    components = cast("tuple[ComponentPlan[object], ...]", plan.identity_components)
    runtime_facts = tuple(
        sorted({fact for component in components for fact in component.runtime_facts})
    )
    return build_runtime_identity_from_facts(
        TRELLIS2_FAMILY_ID,
        runtime_component_identity(TRELLIS2_FAMILY_ID, components),
        diffusion_dtype=compute_dtype.name,
        text_dtype="unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=runtime_facts,
    )


__all__ = [
    "TRELLIS2_FAMILY_ID",
    "TRELLIS2_SPLIT_PROVIDER_REVISION",
    "Trellis2ArtifactRole",
    "Trellis2AssemblyError",
    "Trellis2PlannedArtifact",
    "plan_trellis2_artifact",
    "plan_trellis2_flow_artifact",
    "trellis2_artifact_runtime_identity",
    "trellis2_split_model_runtime_identity",
]
