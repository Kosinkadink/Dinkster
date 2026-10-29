"""Task-local host publisher for standalone inference components."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

import torch
from dinkster_inference import InferenceComponentHandle


class ComponentPublisher(Protocol):
    """Publish one assembled module into the active host residency domain."""

    def publish(
        self,
        module: torch.nn.Module,
        *,
        resource_identity: str,
    ) -> InferenceComponentHandle: ...


_ACTIVE_PUBLISHER: ContextVar[ComponentPublisher | None] = ContextVar(
    "dinkster_component_publisher", default=None
)


def component_publisher() -> ComponentPublisher:
    """Return the component publisher for the active pack context."""

    publisher = _ACTIVE_PUBLISHER.get()
    if publisher is None:
        raise RuntimeError("no component publisher is available in this pack context")
    return publisher


@contextmanager
def use_component_publisher(publisher: ComponentPublisher | None) -> Generator[None]:
    """Provide one component publisher across synchronous or asynchronous code."""

    token = _ACTIVE_PUBLISHER.set(publisher)
    try:
        yield
    finally:
        _ACTIVE_PUBLISHER.reset(token)


__all__ = [
    "ComponentPublisher",
    "component_publisher",
    "use_component_publisher",
]
