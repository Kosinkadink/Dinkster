"""Process-local accelerator policy supplied by validated worker argv."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any

from dinkster_memory import (
    DEFAULT_ACCELERATOR_HEADROOM_BYTES,
    AcceleratorMemoryPolicyError,
)
from dinkster_workers.accelerator import ACCELERATOR_ENV

ACCELERATOR_HEADROOM_ENV = "DINKSTER_ACCELERATOR_HEADROOM_BYTES"
ACCELERATOR_BUDGETS_ENV = "DINKSTER_ACCELERATOR_BUDGETS"


@lru_cache(maxsize=16)
def _policy_values(
    headroom_text: str | None,
    budgets_text: str,
    accelerator: str,
) -> tuple[int, tuple[tuple[str, int], ...]]:
    if headroom_text is None:
        headroom = DEFAULT_ACCELERATOR_HEADROOM_BYTES
    else:
        try:
            headroom = int(headroom_text)
        except ValueError:
            raise AcceleratorMemoryPolicyError(
                f"invalid process-local accelerator headroom {headroom_text!r}"
            ) from None
        if headroom < 0:
            raise AcceleratorMemoryPolicyError(
                "process-local accelerator headroom must be non-negative"
            )

    budgets: dict[str, int] = {}
    if budgets_text:
        device_family = "xpu" if accelerator.strip() == "xpu" else "cuda"
        try:
            for entry in budgets_text.split(","):
                index_text, sep, bytes_text = entry.partition("=")
                if not sep:
                    raise ValueError("missing '='")
                index = int(index_text)
                nbytes = int(bytes_text)
                device = f"{device_family}:{index}"
                if index < 0 or nbytes < 0 or device in budgets:
                    raise ValueError("negative or duplicate entry")
                budgets[device] = nbytes
        except ValueError as exc:
            raise AcceleratorMemoryPolicyError(
                f"invalid process-local accelerator budgets {budgets_text!r}: {exc}"
            ) from None
    return headroom, tuple(sorted(budgets.items()))


def native_memory_policy() -> Any:
    """Read one fresh process-local policy without caching environment state."""
    headroom, budgets = _policy_values(
        os.environ.get(ACCELERATOR_HEADROOM_ENV),
        os.environ.get(ACCELERATOR_BUDGETS_ENV, ""),
        os.environ.get(ACCELERATOR_ENV, ""),
    )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    return inference_torch.MemoryPolicy(
        physical_headroom=headroom,
        hard_budgets=dict(budgets),
    )


__all__ = [
    "ACCELERATOR_BUDGETS_ENV",
    "ACCELERATOR_HEADROOM_ENV",
    "native_memory_policy",
]
