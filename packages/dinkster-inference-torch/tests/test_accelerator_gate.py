from __future__ import annotations

import multiprocessing
from types import SimpleNamespace
from typing import Any

import pytest
from gpu_test_gate import (
    GPU_TEST_OPT_IN,
    gpu_tests_enabled,
    item_requires_gpu,
    require_gpu_tests_enabled,
)


def _guarded_worker(queue: Any) -> None:
    try:
        require_gpu_tests_enabled()
    except RuntimeError as error:
        queue.put(str(error))
        return
    queue.put("enabled")


def test_accelerator_gate_is_default_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    assert False, "deliberate pull-request lane failure for issue 97"
    monkeypatch.delenv(GPU_TEST_OPT_IN, raising=False)

    assert not gpu_tests_enabled()
    with pytest.raises(RuntimeError, match=GPU_TEST_OPT_IN):
        require_gpu_tests_enabled()


def test_accelerator_gate_requires_exact_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("", "0", "true", "yes"):
        monkeypatch.setenv(GPU_TEST_OPT_IN, value)
        assert not gpu_tests_enabled()

    monkeypatch.setenv(GPU_TEST_OPT_IN, "1")
    assert gpu_tests_enabled()
    require_gpu_tests_enabled()


def test_accelerator_gate_is_inherited_by_spawned_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GPU_TEST_OPT_IN, raising=False)
    context = multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    process = context.Process(target=_guarded_worker, args=(queue,))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert GPU_TEST_OPT_IN in queue.get()


@pytest.mark.parametrize(
    ("path", "name", "reason", "expected"),
    (
        ("test_gpu.py", "test_plain", "", True),
        ("test_model_official_gpu.py", "test_plain", "", True),
        ("test_model.py", "test_cuda_path", "", True),
        ("test_model.py", "test_gpu_path", "", True),
        ("test_model.py", "test_rocm_path", "", True),
        ("test_model.py", "test_xpu_path", "", True),
        ("test_model.py", "test_plain", "four CUDA devices required", True),
        ("test_model.py", "test_plain", "a ROCm device is required", True),
        ("test_model.py", "test_plain", "an XPU device is required", True),
        ("test_model.py", "test_plain", "ordinary capability missing", False),
    ),
)
def test_accelerator_item_detection(
    path: str,
    name: str,
    reason: str,
    expected: bool,
) -> None:
    marker = SimpleNamespace(kwargs={"reason": reason})

    def get_closest_marker(_name: str) -> None:
        return None

    def iter_markers(_name: str) -> tuple[SimpleNamespace]:
        return (marker,)

    item = SimpleNamespace(
        path=SimpleNamespace(name=path),
        name=name,
        get_closest_marker=get_closest_marker,
        iter_markers=iter_markers,
    )

    assert item_requires_gpu(item) is expected
