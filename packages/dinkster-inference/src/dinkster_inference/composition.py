"""Torch-free contracts for composing per-component model executions.

Per-component models load independently and compose into one family
execution at invocation time. The composed identity binds cache keys and
receipt preimages to the exact component set.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from .conditioning import ConditioningRecord, ConditioningSet
from .conditioning_wire import ConditioningCarrier, make_conditioning_carrier

_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def _native_identity_parts(identity: object) -> tuple[str, str, str]:
    if not isinstance(identity, str):
        raise TypeError("component binding identity must be a string")
    parts = identity.split(":")
    if len(parts) != 3 or parts[0] != "native":
        raise ValueError("component binding identity must be a 3-part native identity")
    if not parts[1]:
        raise ValueError("component binding identity family must be non-empty")
    digest = parts[2]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("component binding identity digest must be a lowercase sha256 hex digest")
    return cast("tuple[str, str, str]", tuple(parts))


@dataclass(frozen=True)
class ComponentBinding:
    """One named component participating in a family execution."""

    role: str
    family_id: str
    identity: str

    def __post_init__(self) -> None:
        role = cast("object", self.role)
        if not isinstance(role, str):
            raise TypeError("component binding role must be a string")
        if _ROLE_RE.fullmatch(role) is None:
            raise ValueError("component binding role must be a canonical id")

        family_id = cast("object", self.family_id)
        if not isinstance(family_id, str):
            raise TypeError("component binding family_id must be a string")
        if not family_id:
            raise ValueError("component binding family_id must be non-empty")

        parts = _native_identity_parts(cast("object", self.identity))
        if parts[1] != family_id:
            raise ValueError("component binding identity family must match family_id")


COMPONENT_CONDITIONING_METADATA_KEY = "dinkster-inference/component-binding"


def _stamp_binding(stamp: object) -> ComponentBinding:
    if not isinstance(stamp, Mapping):
        raise ValueError("component binding conditioning metadata must be a mapping")
    entries = cast("Mapping[str, object]", stamp)
    if set(entries) != {"role", "family_id", "identity"}:
        raise ValueError(
            "component binding conditioning metadata must carry exactly "
            "role, family_id, and identity"
        )
    values: dict[str, str] = {}
    for name in ("role", "family_id", "identity"):
        value = entries[name]
        if not isinstance(value, str):
            raise ValueError(f"component binding conditioning metadata {name} must be a string")
        values[name] = value
    return ComponentBinding(**values)


def bind_component_conditioning(
    carrier: ConditioningCarrier, binding: ComponentBinding
) -> ConditioningCarrier:
    """Stamp the producing component on every record.

    The stamp rides ``extension_metadata`` inside the canonical carrier, so
    it survives the conditioning wire between workers.
    ``split_component_conditioning`` recovers and strips it before the
    carrier reaches a runtime."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be an exact ConditioningCarrier")
    if type(binding) is not ComponentBinding:
        raise TypeError("binding must be an exact ComponentBinding")
    records = carrier.conditioning.records
    if not records:
        raise ValueError("cannot bind a component to empty conditioning")
    stamp = {
        "role": binding.role,
        "family_id": binding.family_id,
        "identity": binding.identity,
    }
    stamped: list[ConditioningRecord] = []
    for record in records:
        metadata = dict(record.extension_metadata)
        existing = metadata.get(COMPONENT_CONDITIONING_METADATA_KEY)
        if existing is not None and _stamp_binding(existing) != binding:
            raise ValueError("conditioning already carries a different component binding")
        metadata[COMPONENT_CONDITIONING_METADATA_KEY] = stamp
        stamped.append(replace(record, extension_metadata=tuple(metadata.items())))
    return make_conditioning_carrier(ConditioningSet(tuple(stamped)), carrier.bindings)


def split_component_conditioning(
    carrier: ConditioningCarrier,
) -> tuple[ConditioningCarrier, ComponentBinding | None]:
    """Recover and strip the component stamp; ``(carrier, None)`` when unstamped."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be an exact ConditioningCarrier")
    records = carrier.conditioning.records
    stamps = [
        dict(record.extension_metadata).get(COMPONENT_CONDITIONING_METADATA_KEY)
        for record in records
    ]
    if all(stamp is None for stamp in stamps):
        return carrier, None
    if any(stamp is None for stamp in stamps):
        raise ValueError("conditioning records disagree on their component binding")
    bindings = {_stamp_binding(stamp) for stamp in stamps}
    if len(bindings) != 1:
        raise ValueError("conditioning records disagree on their component binding")
    stripped: list[ConditioningRecord] = []
    for record in records:
        metadata = dict(record.extension_metadata)
        del metadata[COMPONENT_CONDITIONING_METADATA_KEY]
        stripped.append(replace(record, extension_metadata=tuple(metadata.items())))
    return (
        make_conditioning_carrier(ConditioningSet(tuple(stripped)), carrier.bindings),
        next(iter(bindings)),
    )


@dataclass(frozen=True)
class ExecutionComposition:
    """A normalized component set for one family execution."""

    family_id: str
    components: tuple[ComponentBinding, ...]
    shared_component_families: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        family_id = cast("object", self.family_id)
        if not isinstance(family_id, str):
            raise TypeError("execution composition family_id must be a string")
        if not family_id:
            raise ValueError("execution composition family_id must be non-empty")

        components = cast("object", self.components)
        if not isinstance(components, tuple) or not all(
            isinstance(component, ComponentBinding)
            for component in cast("tuple[object, ...]", components)
        ):
            raise TypeError("execution composition components must be a tuple of ComponentBinding")
        bindings = cast("tuple[ComponentBinding, ...]", components)
        if not bindings:
            raise ValueError("execution composition components must be non-empty")

        shared_families = cast("object", self.shared_component_families)
        if type(shared_families) is not frozenset or not all(
            isinstance(shared_family, str)
            for shared_family in cast("frozenset[object]", shared_families)
        ):
            raise TypeError("shared component families must be a frozenset of strings")
        if any(not shared_family for shared_family in cast("frozenset[str]", shared_families)):
            raise ValueError("shared component family ids must be non-empty")

        normalized = tuple(sorted(bindings, key=lambda binding: binding.role))
        roles: set[str] = set()
        for binding in normalized:
            if (
                binding.family_id != family_id
                and binding.family_id not in self.shared_component_families
            ):
                raise ValueError(
                    "execution component "
                    f"{binding.role!r} family_id must match composition family_id"
                )
            if binding.role in roles:
                raise ValueError("execution composition component roles must be unique")
            roles.add(binding.role)

        object.__setattr__(self, "components", normalized)

    @property
    def execution_identity(self) -> str:
        """Return the canonical identity for this exact component set."""

        hasher = hashlib.sha256()
        hasher.update(f"family={self.family_id}\n".encode())
        for binding in self.components:
            hasher.update(f"component={binding.role} identity={binding.identity}\n".encode())
        return f"native:{self.family_id}:{hasher.hexdigest()}"


def compose_execution(
    family_id: str,
    components: Mapping[str, str],
    *,
    shared_component_families: frozenset[str] = frozenset(),
) -> ExecutionComposition:
    """Build a normalized execution composition from role-to-identity bindings."""

    if not isinstance(cast("object", components), Mapping):
        raise TypeError("components must be a mapping of role to identity")
    return ExecutionComposition(
        family_id=family_id,
        components=tuple(
            ComponentBinding(
                role=role,
                family_id=_native_identity_parts(identity)[1],
                identity=identity,
            )
            for role, identity in components.items()
        ),
        shared_component_families=shared_component_families,
    )


__all__ = [
    "COMPONENT_CONDITIONING_METADATA_KEY",
    "ComponentBinding",
    "ExecutionComposition",
    "bind_component_conditioning",
    "compose_execution",
    "split_component_conditioning",
]
