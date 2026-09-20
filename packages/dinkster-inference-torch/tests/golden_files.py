"""Executed-reference golden selection and host gating.

Goldens asserted bit-exactly are pinned to the CPU class of the host that
executed the generator: conv, FFT, and vectorized libm kernels dispatch on
microarchitecture and drift by ULPs across CPUs (the CPU analogue of the GPU
host drift in issue #636). A fixture records the mint host's CPU model under
``_meta.cpu`` (the generators write it via tools/golden_platform.py); on a
host with a different CPU the loader skips the consuming module instead of
failing. Consumers of fixtures without a recorded CPU use the portable
assertion helpers by default and opt into the unchanged numerical comparison
only on a CUDA reference host. A consumer can explicitly fall back to the
canonical fixture for portable contracts when its
``<platform>-py<ver>-torch<ver>`` tuple has not been minted. Strict consumers
still skip rather than compare machine-sensitive values on an unvalidated
host. Never widen a tolerance to absorb host drift.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from tools.golden_platform import (  # pyright: ignore[reportMissingImports]
    GoldenUnavailableError,
    GoldenVariantNotFoundError,
    fetch_platform_golden,
)

REFERENCE_VALIDATION_ENV = "DINKSTER_VALIDATE_REFERENCE_GOLDENS"


def reference_validation_enabled() -> bool:
    configured = os.environ.get(REFERENCE_VALIDATION_ENV)
    if configured is None:
        return False
    if configured != "1":
        raise RuntimeError(f"{REFERENCE_VALIDATION_ENV} must be unset or 1")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{REFERENCE_VALIDATION_ENV}=1 requires a CUDA-capable runtime")
    return True


def assert_reference_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    actual_magnitude = actual.abs().amax()
    expected_magnitude = expected.abs().amax()
    if expected_magnitude == 0:
        assert actual_magnitude == 0
    else:
        assert actual_magnitude >= expected_magnitude / 2
        assert actual_magnitude <= expected_magnitude * 2
    if (expected < 0).any():
        assert (actual < 0).any()
    if (expected > 0).any():
        assert (actual > 0).any()
    if expected.numel() > 1 and expected.amin() != expected.amax():
        assert actual.amin() != actual.amax()
    if reference_validation_enabled():
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def assert_reference_schedule(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    rel: float | None = None,
) -> None:
    assert len(actual) == len(expected)
    assert all(math.isfinite(value) and value >= 0 for value in actual)
    assert all(left >= right for left, right in zip(actual, actual[1:], strict=False))
    assert [value > 0 for value in actual] == [value > 0 for value in expected]
    assert [left > right for left, right in zip(actual, actual[1:], strict=False)] == [
        left > right for left, right in zip(expected, expected[1:], strict=False)
    ]
    if not reference_validation_enabled():
        return
    if rel is None:
        assert list(actual) == list(expected)
    else:
        assert list(actual) == pytest.approx(expected, rel=rel)


def assert_reference_values(actual: Sequence[float], expected: Sequence[float]) -> None:
    assert len(actual) == len(expected)
    assert all(math.isfinite(value) for value in actual)
    actual_magnitude = max((abs(value) for value in actual), default=0.0)
    expected_magnitude = max((abs(value) for value in expected), default=0.0)
    if expected_magnitude == 0:
        assert actual_magnitude == 0
    else:
        assert actual_magnitude >= expected_magnitude / 2
        assert actual_magnitude <= expected_magnitude * 2
    assert [value < 0 for value in actual] == [value < 0 for value in expected]
    assert [value > 0 for value in actual] == [value > 0 for value in expected]
    if reference_validation_enabled():
        assert list(actual) == list(expected)


def runtime_provenance() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "platform": sys.platform,
        "python": f"Python {platform.python_version()}",
        "torch": str(torch.__version__),
    }


def cpu_identity() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def runtime_key() -> str:
    return f"{sys.platform}-py{platform.python_version()}-torch{torch.__version__}"


def platform_golden_path(
    path: Path,
    *,
    key: str | None = None,
    allow_portable_fallback: bool = False,
) -> Path:
    if key is None and sys.platform.startswith("linux"):
        return path
    key = key or runtime_key()
    selected = path.with_name(f"{path.stem}.{key}{path.suffix}")
    if not selected.is_file():
        try:
            selected = fetch_platform_golden(path, key)
        except GoldenVariantNotFoundError:
            if allow_portable_fallback:
                return path
            pytest.skip(
                f"excluded: no executed-reference golden minted for {key}: {path.name}",
                allow_module_level=True,
            )
        except GoldenUnavailableError as error:
            if allow_portable_fallback:
                pytest.skip(
                    "excluded: platform evidence unavailable; "
                    f"baseline comparison skipped: {error}",
                    allow_module_level=True,
                )
            pytest.skip(str(error), allow_module_level=True)
    return selected


def load_platform_golden(
    path: Path,
    *,
    allow_portable_fallback: bool = False,
) -> dict[str, Any]:
    selected = platform_golden_path(path, allow_portable_fallback=allow_portable_fallback)
    document: dict[str, Any] = json.loads(selected.read_text())
    provenance = document.get("_meta", document.get("reference"))
    if selected != path:
        assert isinstance(provenance, dict)
        for key, value in runtime_provenance().items():
            assert provenance[key] == value
    minted_cpu = _minted_cpu(document)
    if not allow_portable_fallback and minted_cpu is not None and minted_cpu != cpu_identity():
        pytest.skip(
            f"golden {selected.name} was executed on CPU '{minted_cpu}'; "
            f"live CPU '{cpu_identity()}' drifts by ULPs in executed "
            "kernels (#636), so bit-exact enforcement is host-pinned",
            allow_module_level=True,
        )
    return document


def _minted_cpu(document: dict[str, Any]) -> str | None:
    for source in (document.get("_meta"), document.get("reference")):
        if isinstance(source, dict):
            minted = source.get("cpu")
            if isinstance(minted, str):
                return minted
    return None


def platform_digest(name: str) -> str:
    path = Path(__file__).parent / "goldens" / "wiring_digests.json"
    document = load_platform_golden(path)
    return str(document["digests"][name])
