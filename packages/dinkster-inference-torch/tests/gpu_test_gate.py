from __future__ import annotations

import os
from typing import Any

GPU_TEST_OPT_IN = "DINKSTER_ENABLE_GPU_TESTS"


def gpu_tests_enabled() -> bool:
    return os.environ.get(GPU_TEST_OPT_IN) == "1"


def require_gpu_tests_enabled() -> None:
    if not gpu_tests_enabled():
        raise RuntimeError(f"GPU test worker requires {GPU_TEST_OPT_IN}=1")


_GPU_NAME_TOKENS = ("cuda", "gpu", "rocm", "xpu")


def item_requires_gpu(item: Any) -> bool:
    test_name = str(getattr(item, "originalname", None) or item.name).lower()
    if item.path.name.endswith("_gpu.py") or any(token in test_name for token in _GPU_NAME_TOKENS):
        return True
    if item.get_closest_marker("gpu") is not None:
        return True
    return any(
        any(token in str(marker.kwargs.get("reason", "")).lower() for token in _GPU_NAME_TOKENS)
        for marker in item.iter_markers("skipif")
    )


__all__ = [
    "GPU_TEST_OPT_IN",
    "gpu_tests_enabled",
    "item_requires_gpu",
    "require_gpu_tests_enabled",
]
