"""Descriptor-owned checkpoint components and their source claims."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from .assembly import AssemblyError, ComponentPlan
from .component_registry import (
    ComponentDescriptor,
    ComponentRegistry,
    DetectedComponents,
    component_plans,
)
from .families import ModelFamily
from .quantization import QuantizationError, quantization_error_cause
from .weights import WeightSource


def component_source_keys(component: ComponentPlan[object]) -> frozenset[str]:
    """Include quantization payloads without double-counting transformed model keys."""
    keys = set(component.keys.values())
    for quant in component.quant.values():
        keys.update(
            key
            for key in (
                quant.weight,
                quant.weight_scale,
                quant.input_scale,
                quant.config,
                quant.weight_scale_2,
                quant.pre_quant_scale,
                *quant.payloads.values(),
            )
            if key
        )
    return frozenset(keys)


def component_source_claims(planned: object) -> tuple[tuple[Path, str], ...]:
    auxiliary = cast(
        "tuple[tuple[Path, str], ...]", getattr(planned, "auxiliary_source_claims", ())
    )
    return (
        *(
            (part.path, key)
            for part in component_plans(planned)
            for key in component_source_keys(part)
        ),
        *auxiliary,
    )


@dataclass(frozen=True)
class ComponentCheckpointPlan:
    """Keep every actual role and its unchanged loading plan in descriptor order."""

    descriptor: ComponentDescriptor
    role_plans: tuple[tuple[str, object], ...]
    unclaimed: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        roles: dict[str, object] = {}
        atomic_names: set[str] = set()
        claims: dict[tuple[Path, str], str] = {}
        for role, planned in self.role_plans:
            if role in roles:
                raise AssemblyError(f"duplicate checkpoint component role {role!r}")
            if role not in self.descriptor.roles:
                raise AssemblyError(f"checkpoint role {role!r} is not declared by the descriptor")
            planned_role = (
                planned.component
                if isinstance(planned, ComponentPlan)
                else getattr(planned, "role", role)
            )
            if planned_role != role:
                raise AssemblyError(
                    f"checkpoint role {role!r} contains a plan for {planned_role!r}"
                )
            parts = component_plans(planned)
            if not parts:
                raise AssemblyError(f"checkpoint role {role!r} contains no atomic component plans")
            for component in parts:
                if component.component in atomic_names:
                    raise AssemblyError(
                        f"duplicate checkpoint atomic component name {component.component!r}"
                    )
                atomic_names.add(component.component)
            for claim in component_source_claims(cast("object", planned)):
                if claim in claims:
                    raise AssemblyError(
                        f"checkpoint source {claim[0]}:{claim[1]} is claimed by both "
                        f"{claims[claim]!r} and {role!r}"
                    )
                claims[claim] = role
            roles[role] = planned
        if self.descriptor.model_role not in roles:
            raise AssemblyError(
                f"checkpoint requires the declared model role {self.descriptor.model_role!r}"
            )
        object.__setattr__(
            self,
            "role_plans",
            tuple((role, roles[role]) for role in self.descriptor.roles if role in roles),
        )
        object.__setattr__(self, "unclaimed", tuple(self.unclaimed))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def family(self) -> ModelFamily:
        if self.descriptor.plan_family is not None:
            planned = next(
                planned for role, planned in self.role_plans if role == self.descriptor.model_role
            )
            return self.descriptor.plan_family(component_plans(planned)[0])
        return self.descriptor.family

    @property
    def components(self) -> Mapping[str, ComponentPlan[object]]:
        return MappingProxyType({part.component: part for part in self.identity_components})

    @property
    def identity_components(self) -> tuple[ComponentPlan[object], ...]:
        return tuple(
            part for _role, planned in self.role_plans for part in component_plans(planned)
        )


def plan_component_checkpoint(
    checkpoint: WeightSource | None = None,
    *,
    component_sources: Mapping[str, WeightSource] | None = None,
    component_registry: ComponentRegistry | None = None,
    **source_slots: WeightSource | None,
) -> ComponentCheckpointPlan:
    """Resolve complete declared composition contracts from component detections."""
    from .component_catalog import default_component_registry

    sources = {role: source for role, source in source_slots.items() if source is not None}
    if checkpoint is not None:
        sources["checkpoint"] = checkpoint
    for role, source in (component_sources or {}).items():
        if role in sources:
            raise AssemblyError(f"checkpoint source slot {role!r} was supplied twice")
        sources[role] = source
    registry = default_component_registry() if component_registry is None else component_registry
    detected: dict[str, tuple[DetectedComponents, ...]] = {}
    paths: dict[str, Path] = {}
    diagnostics: list[str] = []
    probe_diagnostics: list[tuple[str, AssemblyError | QuantizationError]] = []
    for slot, source in sources.items():
        path = getattr(source, "path", None)
        if not isinstance(path, Path):
            raise AssemblyError(f"checkpoint source slot {slot!r} has no file path")
        paths[slot] = path
        try:
            detected[slot] = registry.detect(
                source, path, bind_asset_identity=False, diagnostics=probe_diagnostics
            )
        except QuantizationError as error:
            raise AssemblyError(f"checkpoint source slot {slot!r}: {error}") from error

    candidates: list[ComponentCheckpointPlan] = []
    for descriptor in registry:
        matches = {
            slot: match
            for slot, possibilities in detected.items()
            for match in possibilities
            if match.descriptor is descriptor
        }
        if not any(
            role == descriptor.model_role
            for match in matches.values()
            for role, _planned in match.components
        ):
            continue
        if descriptor.checkpoint_loader is None:
            diagnostics.append(f"{descriptor.id}: no declared checkpoint composition loader")
            continue
        try:
            roles: list[tuple[str, object]] = []
            aliases = dict(descriptor.checkpoint_source_aliases)
            split_roles = {
                aliases.get(slot, descriptor.model_role if slot == "diffusion" else slot)
                for slot in sources
                if slot != "checkpoint"
            }
            for slot in sources:
                match = matches.get(slot)
                if match is None:
                    raise AssemblyError(f"source slot {slot!r} has no matching components")
                target = aliases.get(slot, descriptor.model_role if slot == "diffusion" else slot)
                selected = tuple(
                    (role, planned)
                    for role, planned in match.components
                    if (role not in split_roles if slot == "checkpoint" else role == target)
                )
                if not selected and slot != "checkpoint":
                    raise AssemblyError(
                        f"source slot {slot!r} requires role {target!r}; "
                        f"detected {tuple(role for role, _ in match.components)!r}"
                    )
                for role, planned in selected:
                    if any(part.path != paths[slot] for part in component_plans(planned)) or any(
                        path != paths[slot] for path, _key in component_source_claims(planned)
                    ):
                        raise AssemblyError(f"role {role!r} plan changed its selected source path")
                    roles.append((role, planned))
            plan = ComponentCheckpointPlan(descriptor, tuple(roles))
            claimed: dict[Path, set[str]] = {}
            for _role, planned in plan.role_plans:
                for path, key in component_source_claims(planned):
                    claimed.setdefault(path, set()).add(key)
            for component in plan.identity_components:
                claimed.setdefault(component.path, set()).update(component.ignored)
            unclaimed = tuple(
                sorted(
                    {
                        f"{paths[slot]}:{key}"
                        for slot, source in sources.items()
                        for key in source.keys()
                        if key not in claimed.get(paths[slot], set())
                    }
                )
            )
            plan = ComponentCheckpointPlan(descriptor, plan.role_plans, unclaimed)
            if descriptor.checkpoint_validator is not None:
                descriptor.checkpoint_validator(plan)
        except AssemblyError as error:
            probe_diagnostics.append((descriptor.id, error))
            continue
        candidates.append(plan)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise AssemblyError(
            "checkpoint component geometry has multiple complete compositions: "
            + ", ".join(plan.descriptor.id for plan in candidates)
        )
    summary = tuple(
        f"{match.descriptor.id}/{role}"
        for matches in detected.values()
        for match in matches
        for role, _planned in match.components
    )
    quantization_error: QuantizationError | None = None
    for descriptor_id, error in probe_diagnostics:
        cause = quantization_error_cause(error)
        quantization_error = quantization_error or cause
        detail = str(error) if cause is None or cause is error else f"{error}: {cause}"
        diagnostics.append(f"{descriptor_id}: {detail}")
    raise AssemblyError(
        f"no complete checkpoint composition; detected components={summary!r}; "
        + "; ".join(diagnostics)
    ) from quantization_error
