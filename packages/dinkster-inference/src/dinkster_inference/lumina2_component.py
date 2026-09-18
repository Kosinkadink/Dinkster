"""Lumina2 per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .assembly import (
    AssemblyError,
    ComponentPlan,
    Lumina2AssemblyPlan,
    Lumina2ComponentRole,
    plan_lumina2_checkpoint,
    plan_lumina2_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .quantization import QuantizationError, quantization_error_cause
from .weights import AssetIdentifiedSource, WeightSource

if TYPE_CHECKING:
    from .component_checkpoint import ComponentCheckpointPlan


class Lumina2ComponentAssemblyError(ValueError):
    """A source is not the exact Lumina2 component requested."""


@dataclass(frozen=True)
class Lumina2CheckpointText:
    component: ComponentPlan[object]
    tokenizer_source_key: str

    @property
    def role(self) -> str:
        return "gemma2_2b"

    @property
    def identity_components(self) -> tuple[ComponentPlan[object], ...]:
        return (self.component,)

    @property
    def auxiliary_source_claims(self) -> tuple[tuple[Path, str], ...]:
        return ((self.component.path, self.tokenizer_source_key),)


def lumina2_checkpoint_assembly(plan: ComponentCheckpointPlan) -> Lumina2AssemblyPlan:
    """Project validated role plans onto the Lumina2 conditioning and codec contract."""
    components = plan.components
    try:
        text = dict(plan.role_plans)["gemma2_2b"]
        if not isinstance(text, Lumina2CheckpointText):
            raise AssemblyError("gemma2_2b: the selected text plan has no tokenizer origin")
        return Lumina2AssemblyPlan(
            components["diffusion"],
            components["gemma2_2b"],
            components["vae"],
            tokenizer_source_key=text.tokenizer_source_key,
        )
    except (KeyError, ValueError) as error:
        raise AssemblyError(f"incomplete or incompatible Lumina2 checkpoint: {error}") from error


def _identified_plan(
    planned: ComponentPlan[object],
    digest: str,
    size: int,
) -> ComponentPlan[object]:
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def _asset_identity(source: WeightSource, role: str) -> tuple[str, int]:
    if not isinstance(source, AssetIdentifiedSource):
        raise Lumina2ComponentAssemblyError(f"Lumina2 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Lumina2ComponentAssemblyError(f"Lumina2 {role} source must carry asset identity")
    return digest, size


def plan_lumina2_split_component(
    source: WeightSource,
    *,
    role: Lumina2ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    """Plan one Lumina2 component and bind its immutable asset identity."""
    if getattr(source, "path", None) != path:
        raise Lumina2ComponentAssemblyError(
            f"Lumina2 {role} source path differs from artifact selection"
        )
    identity = _asset_identity(source, role) if bind_asset_identity else None
    try:
        planned = plan_lumina2_component(source, role)
    except ValueError as error:
        raise Lumina2ComponentAssemblyError(f"Lumina2 {role}: {error}") from error
    return planned if identity is None else _identified_plan(planned, *identity)


def plan_lumina2_checkpoint_components(
    source: WeightSource,
    *,
    path: Path,
    bind_asset_identity: bool = True,
) -> tuple[ComponentPlan[object], ComponentPlan[object], ComponentPlan[object]]:
    """Plan and asset-bind every component of one exact all-in-one checkpoint."""
    if getattr(source, "path", None) != path:
        raise Lumina2ComponentAssemblyError(
            "Lumina2 checkpoint source path differs from artifact selection"
        )
    identity = _asset_identity(source, "checkpoint") if bind_asset_identity else None
    try:
        plans = plan_lumina2_checkpoint(source)
    except ValueError as error:
        raise Lumina2ComponentAssemblyError(f"Lumina2 checkpoint: {error}") from error
    if identity is None:
        return plans
    digest, size = identity
    diffusion, text, vae = plans
    return (
        _identified_plan(diffusion, digest, size),
        _identified_plan(text, digest, size),
        _identified_plan(vae, digest, size),
    )


def plan_lumina2_artifact_components(
    source: WeightSource,
    *,
    path: Path,
    bind_asset_identity: bool = True,
) -> tuple[tuple[Lumina2ComponentRole, ComponentPlan[object]], ...]:
    """Identify every independently loadable component in one Lumina2 artifact."""
    quantization_error: QuantizationError | None = None
    checkpoint_error: Lumina2ComponentAssemblyError | None = None
    try:
        checkpoint = plan_lumina2_checkpoint_components(
            source, path=path, bind_asset_identity=bind_asset_identity
        )
    except Lumina2ComponentAssemblyError as error:
        checkpoint = None
        checkpoint_error = error
        quantization_error = quantization_error_cause(error)
    if checkpoint is not None:
        return tuple(zip(("diffusion", "gemma2_2b", "vae"), checkpoint, strict=True))

    components: list[tuple[Lumina2ComponentRole, ComponentPlan[object]]] = []
    roles: tuple[Lumina2ComponentRole, ...] = ("diffusion", "gemma2_2b", "vae")
    for role in roles:
        try:
            plan = plan_lumina2_split_component(
                source, role=role, path=path, bind_asset_identity=bind_asset_identity
            )
        except Lumina2ComponentAssemblyError as error:
            quantization_error = quantization_error or quantization_error_cause(error)
            continue
        components.append((role, plan))
    if len(components) != 1:
        if not components and quantization_error is not None:
            raise quantization_error
        raise Lumina2ComponentAssemblyError(
            "Lumina2 artifact is not an exact all-in-one checkpoint or split component"
        ) from checkpoint_error
    return tuple(components)


def lumina2_component_runtime_identity(
    planned: ComponentPlan[object],
    role: Lumina2ComponentRole,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Lumina2 component."""
    if planned.component != role:
        raise ValueError(f"Lumina2 {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.lumina2",
        runtime_component_identity("dinkster.lumina2", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "gemma2_2b" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "Lumina2ComponentAssemblyError",
    "lumina2_component_runtime_identity",
    "plan_lumina2_artifact_components",
    "plan_lumina2_checkpoint_components",
    "plan_lumina2_split_component",
]
