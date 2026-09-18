from __future__ import annotations

import os

import pytest
from gpu_test_gate import GPU_TEST_OPT_IN, gpu_tests_enabled, item_requires_gpu

if not gpu_tests_enabled():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_ignore_collect(collection_path: object) -> bool | None:
    if not gpu_tests_enabled() and getattr(collection_path, "name", None) == "test_gpu.py":
        return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if gpu_tests_enabled():
        return
    disabled = pytest.mark.skip(reason=f"GPU tests require {GPU_TEST_OPT_IN}=1")
    for item in items:
        if item_requires_gpu(item):
            item.add_marker(disabled)
