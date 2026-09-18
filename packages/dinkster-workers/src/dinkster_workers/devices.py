"""Worker-qualified device namespacing (DESIGN 3.10).

A worker pinned to one GPU (``CUDA_VISIBLE_DEVICES=1``) honestly reports
its device as ``cuda:0`` - that *is* the device in its namespace. But the
engine's admission lanes and the governor's budgets live in the parent's
namespace, where two pinned workers both claiming ``cuda:0`` would collide
on one lane while actually occupying different silicon.

The parent launched the worker with that env, so the parent owns the
translation: a DeviceMap rewrites device facts as they cross the boundary
into the parent - residency meta (lanes), cost meta (budgets), and
reservation lease requests. Child code never knows; nothing is sent back
translated because the child never consumes parent-namespace device
strings (its objects know their own devices).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from dinkster_values import (
    COST_META_KEY,
    RESOURCES_META_KEY,
    ListPayload,
    Value,
    ValueMeta,
    list_children,
)


@dataclass(frozen=True)
class DeviceMap:
    """child-namespace device -> parent-namespace device, e.g.
    ``{"cuda:0": "cuda:1"}`` for a worker launched with
    ``CUDA_VISIBLE_DEVICES=1``. Unmapped devices pass through unchanged, so
    an empty map is the identity.

    A ``qualifier`` changes the default for *unmapped* facts: instead of
    passing through, they are suffixed with ``@<qualifier>`` - including
    device-less residency classes (``ram`` -> ``ram@worker1``). This is the
    remote-worker posture: another machine's ``cuda:0`` is not this
    machine's ``cuda:0``, and its ram is not this machine's ram, so by
    default nothing a remote worker reports lands on a local budget or
    lane. Explicit ``mapping`` entries still win, so an operator can
    deliberately unify namespaces where that is meaningful.
    """

    mapping: Mapping[str, str]
    qualifier: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", dict(self.mapping))

    def device(self, device: str) -> str:
        mapped = self.mapping.get(device)
        if mapped is not None:
            return mapped
        if self.qualifier is not None:
            return f"{device}@{self.qualifier}"
        return device

    def residency(self, residency: str) -> str:
        """Map a residency class: ``vram:cuda:0`` -> ``vram:cuda:1``;
        device-less classes (``ram``, ``disk``) pass through unless a
        qualifier applies."""
        kind, sep, device = residency.partition(":")
        if not sep:
            if self.qualifier is not None:
                return f"{residency}@{self.qualifier}"
            return residency
        return f"{kind}:{self.device(device)}"

    def to_child_device(self, device: str) -> str | None:
        """Invert: parent-namespace device -> the child's name for it.
        None means the parent device has no counterpart in the child's
        namespace - pressure aimed there must not be forwarded, because
        the string would target the *wrong* silicon inside the child."""
        for child_device, parent_device in self.mapping.items():
            if parent_device == device:
                return child_device
        if self.qualifier is not None:
            suffix = f"@{self.qualifier}"
            if device.endswith(suffix):
                return device[: -len(suffix)]
            return None  # a local device; the remote worker cannot see it
        if device in self.mapping:
            return None  # this name means different silicon inside the child
        return device

    def to_child_residency(self, residency: str) -> str | None:
        """Invert a residency class, or None when it has no counterpart."""
        kind, sep, device = residency.partition(":")
        if not sep:
            if self.qualifier is not None:
                suffix = f"@{self.qualifier}"
                if residency.endswith(suffix):
                    return residency[: -len(suffix)]
                return None  # local ram/disk; not the remote worker's
            return residency
        child_device = self.to_child_device(device)
        if child_device is None:
            return None
        return f"{kind}:{child_device}"

    def value(self, value: Value) -> Value:
        """Rewrite a decoded value's device facts into the parent namespace.

        Only residency meta (lane binding) and cost meta (budget keys) are
        touched; identity - type, fingerprint, payload - is device-free by
        design (hazard H4) and crosses untouched. Lists rewrite their
        children recursively: identity is unchanged (fingerprints are
        device-free), only the child envelopes' device facts move.
        """
        children = list_children(value)
        if children is not None:
            mapped = tuple(self.value(child) for child in children)
            if all(m is c for m, c in zip(mapped, children, strict=True)):
                return value
            return Value(
                type_id=value.type_id,
                fingerprint=value.fingerprint,
                meta=value.meta,
                payload=ListPayload(mapped),
            )
        entries = dict(value.meta.entries)
        changed = False

        resources = entries.get(RESOURCES_META_KEY)
        if isinstance(resources, Mapping):
            mapped_resources: dict[str, object] = {}
            for kind, instance in cast("Mapping[str, object]", resources).items():
                if isinstance(instance, str):
                    mapped_resources[kind] = self.device(instance)
                else:
                    mapped_resources[kind] = tuple(
                        self.device(one) for one in cast("Iterable[str]", instance)
                    )
            if mapped_resources != dict(cast("Mapping[str, object]", resources)):
                entries[RESOURCES_META_KEY] = mapped_resources
                changed = True

        cost = entries.get(COST_META_KEY)
        if isinstance(cost, Mapping):
            mapped_cost = {
                self.residency(str(residency)): nbytes
                for residency, nbytes in cast("Mapping[str, object]", cost).items()
            }
            if mapped_cost != dict(cast("Mapping[str, object]", cost)):
                entries[COST_META_KEY] = mapped_cost
                changed = True

        if not changed:
            return value
        return Value(
            type_id=value.type_id,
            fingerprint=value.fingerprint,
            meta=ValueMeta(entries),
            payload=value.payload,
        )
