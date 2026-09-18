from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest


class _Mechanism:
    load_device = SimpleNamespace(type="cpu")

    def total_bytes(self) -> int:
        return 1

    def loaded_bytes(self) -> int:
        return 0


class _Manager:
    def __init__(self, events: list[str], *, fail_remove: bool = False) -> None:
        self.events = events
        self.fail_remove = fail_remove

    def remove(self, _mechanisms: object, *, unload: bool = True, discard: bool = False) -> None:
        assert unload
        self.events.append("discard" if discard else "remove")
        if self.fail_remove:
            raise RuntimeError("remove failed")

    def empty_cache(self, _device: object) -> None:
        pass


class _Dependent:
    pass


def _handle(
    events: list[str], *, fail_remove: bool = False, discard_on_release: bool = False
) -> tuple[Any, object]:
    native_residency = importlib.import_module("dinkster_compat_comfy.native_residency")
    manager = _Manager(events, fail_remove=fail_remove)
    coordinator = native_residency.NativeResidencyCoordinator(manager)
    module = SimpleNamespace(_dinkster_residency_state_store=object())
    handle = native_residency.NativeComponentHandle(
        module,
        _Mechanism(),
        SimpleNamespace(type="cpu"),
        resource_identity="native:test:" + "1" * 64,
        coordinator=coordinator,
        discard_on_release=discard_on_release,
    )
    return handle, module


@pytest.mark.parametrize("discard_on_release", [False, True])
def test_terminal_release_detaches_after_manager_removal(
    monkeypatch: pytest.MonkeyPatch, discard_on_release: bool
) -> None:
    events: list[str] = []
    handle, module = _handle(events, discard_on_release=discard_on_release)

    def detach(detached_module: object, _mechanism: object) -> None:
        assert detached_module is module
        events.append("detach")

    inference_torch = SimpleNamespace(detach_residency_enrollment=detach)
    original_import = importlib.import_module

    def import_module(name: str) -> Any:
        return inference_torch if name == "dinkster_inference_torch" else original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    handle.terminal_release()

    assert events == ["discard" if discard_on_release else "remove", "detach"]
    assert handle.released
    assert handle.mechanisms == ()


@pytest.mark.parametrize("discard_on_release", [False, True])
def test_terminal_release_does_not_detach_when_manager_removal_fails(
    monkeypatch: pytest.MonkeyPatch, discard_on_release: bool
) -> None:
    events: list[str] = []
    handle, _module = _handle(events, fail_remove=True, discard_on_release=discard_on_release)

    def unexpected_detach(_module: object, _mechanism: object) -> None:
        events.append("detach")

    inference_torch = SimpleNamespace(detach_residency_enrollment=unexpected_detach)
    original_import = importlib.import_module

    def import_module(name: str) -> Any:
        return inference_torch if name == "dinkster_inference_torch" else original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="remove failed"):
        handle.terminal_release()

    assert events == ["discard" if discard_on_release else "remove"]
    assert handle.released
    assert handle.mechanisms == ()


@pytest.mark.parametrize("discard_on_release", [False, True])
def test_terminal_release_refuses_live_dependent(discard_on_release: bool) -> None:
    events: list[str] = []
    handle, _module = _handle(events, discard_on_release=discard_on_release)
    dependent = _Dependent()
    handle.register_dependent(dependent)

    with pytest.raises(RuntimeError, match="still referenced by a model overlay"):
        handle.terminal_release()

    assert events == []
    assert not handle.released


def test_discard_lifetime_is_read_only_and_refuses_pool_attachment() -> None:
    events: list[str] = []
    handle, _module = _handle(events, discard_on_release=True)
    assert handle.discard_on_release
    with pytest.raises(AttributeError):
        handle.discard_on_release = False
    pool = SimpleNamespace(set_loaded=lambda *_args: events.append("pool"))
    with pytest.raises(RuntimeError, match="cannot attach to a resident pool"):
        handle.attach_pool(pool)
    handle.reconcile_pool()
    assert events == []
    published, _module = _handle(events)
    assert not published.discard_on_release
    published.attach_pool(pool)
    assert events == ["pool"]


@pytest.mark.parametrize("discard_on_release", [False, True])
def test_terminal_release_drops_handle_when_unload_lifecycle_fails(
    monkeypatch: pytest.MonkeyPatch, discard_on_release: bool
) -> None:
    events: list[str] = []
    handle, _module = _handle(events, discard_on_release=discard_on_release)
    mechanism = handle.mechanisms[0]
    mechanism.loaded_bytes = lambda: 1

    def detach(_module: object, _mechanism: object) -> None:
        events.append("detach")

    def run_lifecycle(phase: str, *, source: object | None = None) -> None:
        del source
        events.append(phase)
        if phase == "unload":
            raise RuntimeError("unload failed")

    inference_torch = SimpleNamespace(detach_residency_enrollment=detach)
    original_import = importlib.import_module

    def import_module(name: str) -> Any:
        return inference_torch if name == "dinkster_inference_torch" else original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(handle, "run_lifecycle", run_lifecycle)

    with pytest.raises(RuntimeError, match="unload failed"):
        handle.terminal_release()

    assert events == ["discard" if discard_on_release else "remove", "detach", "unload", "cleanup"]
    assert handle.released
    assert handle.mechanisms == ()
