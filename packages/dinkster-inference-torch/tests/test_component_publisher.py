"""Task-local component publisher contract tests."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext

import pytest
import torch
from dinkster_inference import InferenceComponentHandle, InferenceRuntimeHandle
from dinkster_inference_torch import (
    ComponentPublisher,
    component_publisher,
    use_component_publisher,
)


class _Handle:
    def __init__(self, component: torch.nn.Module, resource_identity: str) -> None:
        self.component = component
        self.resource_identity = resource_identity
        self.load_device = torch.device("cpu")

    def require_active(self) -> None:
        pass

    def stage(self, *, memory_required: int = 0):  # noqa: ANN201
        del memory_required
        return nullcontext()

    def stage_with(
        self,
        runtime_handle: InferenceRuntimeHandle,
        role: str,
    ):  # noqa: ANN201
        del runtime_handle, role
        return nullcontext()


class _Publisher:
    def __init__(self) -> None:
        self.published: list[tuple[torch.nn.Module, str]] = []

    def publish(
        self,
        module: torch.nn.Module,
        *,
        resource_identity: str,
    ) -> InferenceComponentHandle:
        self.published.append((module, resource_identity))
        return _Handle(module, resource_identity)


def test_component_publisher_fails_closed_without_host() -> None:
    with pytest.raises(RuntimeError, match="no component publisher"):
        component_publisher()


def test_component_publisher_scope_publishes_and_restores() -> None:
    outer = _Publisher()
    inner = _Publisher()
    module = torch.nn.Linear(2, 2)
    identity = "sha256:" + "2" * 64

    with use_component_publisher(outer):
        selected: ComponentPublisher = component_publisher()
        assert selected is outer
        with use_component_publisher(inner):
            handle = component_publisher().publish(module, resource_identity=identity)
        assert component_publisher() is outer

    assert inner.published == [(module, identity)]
    assert handle.component is module
    with pytest.raises(RuntimeError, match="no component publisher"):
        component_publisher()


def test_component_publisher_none_scope_fails_closed_and_restores() -> None:
    publisher = _Publisher()
    with use_component_publisher(publisher):
        with use_component_publisher(None):
            with pytest.raises(RuntimeError, match="no component publisher"):
                component_publisher()
        assert component_publisher() is publisher


def test_component_publisher_restores_after_exception() -> None:
    publisher = _Publisher()
    with pytest.raises(ValueError, match="body failed"):
        with use_component_publisher(publisher):
            raise ValueError("body failed")
    with pytest.raises(RuntimeError, match="no component publisher"):
        component_publisher()


def test_component_publisher_isolated_between_async_tasks() -> None:
    async def run() -> None:
        first = _Publisher()
        second = _Publisher()
        ready = asyncio.Event()
        release = asyncio.Event()

        async def scoped(publisher: _Publisher) -> None:
            with use_component_publisher(publisher):
                ready.set()
                await release.wait()
                assert component_publisher() is publisher

        first_task = asyncio.create_task(scoped(first))
        await ready.wait()
        ready.clear()
        second_task = asyncio.create_task(scoped(second))
        await ready.wait()
        release.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())
