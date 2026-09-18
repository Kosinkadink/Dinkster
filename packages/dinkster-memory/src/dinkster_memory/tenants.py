"""Neutral model-tenant registration contract for in-process packs."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol


class ModelTenantHandle(Protocol):
    pack: str
    model_id: str
    size_estimate_bytes: int
    device_preference: str | None

    async def offload(self) -> None: ...

    async def evict(self) -> None: ...


@dataclass(frozen=True)
class TenantRegistration:
    registration_id: str
    assigned_device: str


class ModelTenantRegistry(Protocol):
    async def register(self, handle: ModelTenantHandle) -> TenantRegistration: ...

    async def notify_resident(self, registration: TenantRegistration) -> None: ...

    async def unregister(self, registration: TenantRegistration) -> None: ...

    async def terminal_release(self, registration: TenantRegistration) -> None: ...


class TenantRegistryUnavailable(RuntimeError):
    """No engine-owned registry was provided to the active pack."""


_active_registry: ContextVar[ModelTenantRegistry | None] = ContextVar(
    "dinkster_model_tenant_registry", default=None
)


def model_tenant_registry() -> ModelTenantRegistry:
    """Return the engine-owned registry for the active pack context."""
    registry = _active_registry.get()
    if registry is None:
        raise TenantRegistryUnavailable(
            "no model tenant registry is available in this pack context"
        )
    return registry


@contextmanager
def use_model_tenant_registry(
    registry: ModelTenantRegistry | None,
) -> Generator[None]:
    """Provide a registry context across synchronous or asynchronous code."""
    token = _active_registry.set(registry)
    try:
        yield
    finally:
        _active_registry.reset(token)
