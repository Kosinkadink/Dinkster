"""The inference registries consumed by one worker generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import component_catalog
from .catalog import builtin_family_registry
from .component_registry import ComponentRegistry
from .families import FamilyRegistry
from .registry import Registry
from .runtime import AssemblyRegistration, build_builtin_assembly_registry
from .sampling import SamplerDescriptor, SchedulerDescriptor
from .schedules import builtin_scheduler_registry
from .solvers import builtin_sampler_registry

if TYPE_CHECKING:
    from .extensions import InferenceContribution


@dataclass(frozen=True)
class InferenceRegistries:
    """Every inference registry visible to one worker generation."""

    samplers: Registry[SamplerDescriptor[Any]]
    schedulers: Registry[SchedulerDescriptor]
    families: FamilyRegistry
    components: ComponentRegistry
    assemblies: Registry[AssemblyRegistration]


def builtin_registries() -> InferenceRegistries:
    """Build a fresh, mutually consistent set of builtin registries."""
    components = component_catalog.default_component_registry()
    return InferenceRegistries(
        samplers=builtin_sampler_registry(),
        schedulers=builtin_scheduler_registry(),
        families=builtin_family_registry(),
        components=components,
        assemblies=build_builtin_assembly_registry(components),
    )


def merge(
    base: InferenceRegistries,
    contributions: Iterable[InferenceContribution],
) -> InferenceRegistries:
    """Return fresh registries containing base entries and additive contributions."""
    samplers: Registry[SamplerDescriptor[Any]] = Registry()
    schedulers: Registry[SchedulerDescriptor] = Registry()
    families = FamilyRegistry()
    components = ComponentRegistry()
    assemblies: Registry[AssemblyRegistration] = Registry()
    for descriptor in base.samplers:
        samplers.register(descriptor)
    for descriptor in base.schedulers:
        schedulers.register(descriptor)
    for family_id in base.families.ids():
        family = base.families.get(family_id)
        assert family is not None
        families.register(family)
    for descriptor in base.components:
        components.register(descriptor)
    for descriptor in base.assemblies:
        assemblies.register(descriptor)
    for contribution in contributions:
        for descriptor in contribution.samplers:
            samplers.register(descriptor)
        for descriptor in contribution.schedulers:
            schedulers.register(descriptor)
    return InferenceRegistries(samplers, schedulers, families, components, assemblies)


def builtin_assembly_registry() -> Registry[AssemblyRegistration]:
    """Build the builtin assembly registry through the aggregate factory."""
    return builtin_registries().assemblies


def wired_runtime_family_ids() -> tuple[str, ...]:
    """Registered runtime labels for diagnostics, never an admission predicate."""
    return tuple(
        sorted(
            name
            for assembly in builtin_registries().assemblies
            for name in (assembly.aliases or (assembly.id,))
        )
    )


NATIVE_WIRED_FAMILY_IDS = wired_runtime_family_ids()


__all__ = [
    "InferenceRegistries",
    "NATIVE_WIRED_FAMILY_IDS",
    "builtin_assembly_registry",
    "builtin_registries",
    "merge",
    "wired_runtime_family_ids",
]
