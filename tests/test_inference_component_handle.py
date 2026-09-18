"""Standalone component-handle seam contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from dinkster_inference import (
    InferenceComponentHandle,
    InferenceRuntimeHandle,
    require_inference_component_handle,
    require_inference_runtime_handle,
)


class _ReleasedError(RuntimeError):
    pass


class _ComponentHandle:
    def __init__(self) -> None:
        self._component = object()
        self.resource_identity = "sha256:" + "1" * 64
        self.load_device = "cpu"
        self.released = False

    @property
    def component(self) -> object:
        self.require_active()
        return self._component

    def require_active(self) -> None:
        if self.released:
            raise _ReleasedError("component was released")

    @contextmanager
    def stage(self) -> Iterator[None]:
        self.require_active()
        yield

    @contextmanager
    def stage_with(
        self,
        runtime_handle: InferenceRuntimeHandle,
        role: str,
    ) -> Iterator[None]:
        del runtime_handle, role
        self.require_active()
        yield


class _RuntimeHandleShape:
    runtime = object()
    recipe = object()
    load_device = "cpu"

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self, role: str) -> Iterator[None]:
        del role
        yield


def test_component_handle_accepts_active_handle() -> None:
    handle = _ComponentHandle()
    assert isinstance(handle, InferenceComponentHandle)
    assert require_inference_component_handle(handle, "control") is handle


def test_component_handle_rejects_foreign_object() -> None:
    with pytest.raises(TypeError, match="control must be a component handle"):
        require_inference_component_handle(object(), "control")


@pytest.mark.parametrize("member", ["require_active", "stage", "stage_with"])
def test_component_handle_rejects_non_callable_protocol_member(member: str) -> None:
    handle = _ComponentHandle()
    setattr(handle, member, 1)
    with pytest.raises(TypeError, match=rf"{member} must be callable"):
        require_inference_component_handle(handle, "control")


def test_component_handle_propagates_released_lifecycle() -> None:
    handle = _ComponentHandle()
    handle.released = True
    with pytest.raises(_ReleasedError):
        require_inference_component_handle(handle, "control")
    with pytest.raises(_ReleasedError):
        _ = handle.component


@pytest.mark.parametrize("identity", ["", 1])
def test_component_handle_rejects_invalid_resource_identity(identity: object) -> None:
    handle = _ComponentHandle()
    handle.resource_identity = identity  # type: ignore[assignment]
    with pytest.raises(TypeError, match="resource identity must be a non-empty string"):
        require_inference_component_handle(handle, "control")


def test_component_handle_rejects_none_device() -> None:
    handle = _ComponentHandle()
    handle.load_device = None  # type: ignore[assignment]
    with pytest.raises(TypeError, match="load device must not be None"):
        require_inference_component_handle(handle, "control")


def test_component_and_runtime_handles_narrow_only_in_their_direction() -> None:
    component = _ComponentHandle()
    runtime = _RuntimeHandleShape()

    assert not isinstance(component, InferenceRuntimeHandle)
    assert isinstance(runtime, InferenceRuntimeHandle)
    assert not isinstance(runtime, InferenceComponentHandle)
    with pytest.raises(TypeError, match="native runtime handle"):
        require_inference_runtime_handle(component, "model")
    with pytest.raises(TypeError, match="component handle"):
        require_inference_component_handle(runtime, "component")
