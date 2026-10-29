"""Family-declared native residency policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast


def _string_frozenset(field_name: str, declaration: object) -> frozenset[str]:
    if not isinstance(declaration, frozenset):
        raise TypeError(f"{field_name} must be a frozenset of strings")
    values = cast("frozenset[object]", declaration)
    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"{field_name} must contain only strings")
    strings = cast("frozenset[str]", values)
    if any(not value for value in strings):
        raise ValueError(f"{field_name} must contain only non-empty strings")
    return strings


@dataclass(frozen=True, slots=True)
class NativeResidencyPolicy:
    enrollment_components: tuple[str, ...]
    enrollment_orders: tuple[tuple[str, ...], ...]
    component_roles: Mapping[str, str]
    diffusion_roles: frozenset[str]
    unload_after_stage: frozenset[str]
    resident_components: frozenset[str]

    def __post_init__(self) -> None:
        enrollment_components = cast("object", self.enrollment_components)
        if not isinstance(enrollment_components, tuple):
            raise TypeError("enrollment_components must be a tuple of strings")
        if not enrollment_components:
            raise ValueError("enrollment_components must not be empty")
        component_values = cast("tuple[object, ...]", enrollment_components)
        if not all(isinstance(component, str) for component in component_values):
            raise TypeError("enrollment_components must contain only strings")
        component_names = cast("tuple[str, ...]", component_values)
        if any(not component for component in component_names):
            raise ValueError("enrollment_components must contain only non-empty strings")
        if len(set(component_names)) != len(component_names):
            raise ValueError("enrollment_components must be unique")
        components = frozenset(component_names)

        enrollment_orders = cast("object", self.enrollment_orders)
        if not isinstance(enrollment_orders, tuple):
            raise TypeError("enrollment_orders must be a tuple of tuples")
        for order in cast("tuple[object, ...]", enrollment_orders):
            if not isinstance(order, tuple):
                raise TypeError("enrollment_orders must contain only tuples")
            order_values = cast("tuple[object, ...]", order)
            if not all(isinstance(component, str) for component in order_values):
                raise TypeError("enrollment_orders must contain only strings")
            unknown = set(cast("tuple[str, ...]", order_values)) - components
            if unknown:
                raise ValueError(
                    "enrollment_orders contains undeclared components: "
                    + ", ".join(sorted(unknown))
                )

        roles_declaration = cast("object", self.component_roles)
        if not isinstance(roles_declaration, Mapping):
            raise TypeError("component_roles must be a mapping of strings to strings")
        untyped_roles = dict(cast("Mapping[object, object]", roles_declaration))
        for component, role in untyped_roles.items():
            if not isinstance(component, str):
                raise TypeError("component_roles keys must be strings")
            if component not in components:
                raise ValueError(f"component_roles contains undeclared component {component!r}")
            if not isinstance(role, str):
                raise TypeError("component_roles values must be strings")
            if not role:
                raise ValueError("component_roles values must be non-empty strings")
        component_roles = cast("dict[str, str]", untyped_roles)
        object.__setattr__(self, "component_roles", MappingProxyType(component_roles))

        roles = frozenset(component_roles.values())
        for field_name in ("diffusion_roles", "unload_after_stage"):
            values = _string_frozenset(field_name, getattr(self, field_name))
            unknown = values - roles
            if unknown:
                raise ValueError(
                    f"{field_name} contains undeclared roles: " + ", ".join(sorted(unknown))
                )

        resident_components = _string_frozenset(
            "resident_components", cast("object", self.resident_components)
        )
        unknown_resident = resident_components - components
        if unknown_resident:
            raise ValueError(
                "resident_components contains undeclared components: "
                + ", ".join(sorted(unknown_resident))
            )
