"""Public seam for one standalone component sharing runtime residency.

Standalone control modules are live hardware resources, not wire values.
Their handle exposes only identity, device placement, lifecycle, a standalone
lease, and a composite lease that stages the component together with a runtime
role. The implementation owns residency policy and must refuse handles from a
different residency domain.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, cast, runtime_checkable

from .runtime_handle import InferenceRuntimeHandle


@runtime_checkable
class InferenceComponentHandle(Protocol):
    """One resident component with standalone and runtime-coordinated leases."""

    @property
    def component(self) -> object:
        """The live component; raises after release."""
        ...

    @property
    def resource_identity(self) -> str:
        """Stable identity folded into admission and cache preimages."""
        ...

    @property
    def load_device(self) -> object:
        """Opaque backend device token; never None."""
        ...

    def require_active(self) -> None:
        """Raise if the component was terminally released."""
        ...

    def stage(self, *, memory_required: int = 0) -> AbstractContextManager[None]:
        """Lease this component with invocation-local device memory demand."""
        ...

    def stage_with(
        self,
        runtime_handle: InferenceRuntimeHandle,
        role: str,
    ) -> AbstractContextManager[None]:
        """Lease this component and one runtime role as one operation."""
        ...


def require_inference_component_handle(value: object, input_id: str) -> InferenceComponentHandle:
    """Fail-closed narrowing of one node input to a component handle."""

    if not isinstance(value, InferenceComponentHandle):
        raise TypeError(f"{input_id} must be a component handle, got {type(value).__name__}")
    for name in ("require_active", "stage", "stage_with"):
        if not callable(getattr(value, name)):
            raise TypeError(f"{input_id} {name} must be callable")
    value.require_active()
    identity = cast("object", value.resource_identity)
    if not isinstance(identity, str) or not identity:
        raise TypeError(f"{input_id} resource identity must be a non-empty string")
    if value.load_device is None:
        raise TypeError(f"{input_id} load device must not be None")
    return value


__all__ = [
    "InferenceComponentHandle",
    "require_inference_component_handle",
]
