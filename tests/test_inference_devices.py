"""NVIDIA dtype capability gates against the pinned reference.

The tables and tiers mirror should_use_fp16 / should_use_bf16 in
comfy/model_management.py @ b78cec87 for driver-reported facts (compute
capability major, device name). The torch backend consumes the same
gates through torch device properties; these tests pin the pure logic.
"""

from __future__ import annotations

import pytest
from dinkster_inference.devices import (
    NVIDIA_16_SERIES,
    nvidia_bf16_compute,
    nvidia_compute_dtypes,
    nvidia_fp16_support,
)


@pytest.mark.parametrize(
    ("major", "name", "windows", "expected"),
    [
        # Ampere and newer compute everything.
        (12, "NVIDIA GeForce RTX 5090", False, {"float16", "bfloat16", "float32"}),
        (8, "NVIDIA A100-SXM4-80GB", False, {"float16", "bfloat16", "float32"}),
        (8, "NVIDIA GeForce RTX 3060", True, {"float16", "bfloat16", "float32"}),
        # Volta/Turing: fp16 but never bf16.
        (7, "NVIDIA TITAN V", False, {"float16", "float32"}),
        (7, "NVIDIA GeForce RTX 2080 Ti", False, {"float16", "float32"}),
        # 16-series fp16 kernels are broken regardless of OS.
        (7, "NVIDIA GeForce GTX 1660", False, {"float32"}),
        (7, "NVIDIA GeForce GTX 1660", True, {"float32"}),
        (7, "NVIDIA T1000", False, {"float32"}),
        # The reference matches the 16-series list case-sensitively.
        (7, "nvidia t1000", False, {"float16", "float32"}),
        # 10-series runs fp16 kernels profitably only on Windows.
        (6, "NVIDIA GeForce GTX 1080", False, {"float32"}),
        (6, "NVIDIA GeForce GTX 1080", True, {"float16", "float32"}),
        (6, "Tesla P100-PCIE-16GB", True, {"float16", "float32"}),
        # Pascal outside the 10-series list never computes fp16.
        (6, "NVIDIA GeForce GT 1030", False, {"float32"}),
        (6, "NVIDIA GeForce GT 1030", True, {"float32"}),
        # Pre-Pascal is float32 only.
        (5, "NVIDIA GeForce GTX 980", True, {"float32"}),
        (3, "NVIDIA GeForce GTX 780", False, {"float32"}),
    ],
)
def test_nvidia_compute_dtypes_reference_gates(
    major: int, name: str, windows: bool, expected: set[str]
) -> None:
    assert nvidia_compute_dtypes(major, name, windows=windows) == frozenset(expected)


def test_nvidia_fp16_storage_facts() -> None:
    # Pre-Pascal refuses fp16 outright, cast included, on every OS.
    assert nvidia_fp16_support(5, "NVIDIA GeForce GTX 980", windows=False) == nvidia_fp16_support(
        5, "NVIDIA GeForce GTX 980", windows=True
    )
    assert not nvidia_fp16_support(5, "NVIDIA GeForce GTX 980", windows=False).storage
    # 16-series and Linux 10-series take fp16 storage with cast-at-use.
    assert nvidia_fp16_support(7, "NVIDIA T1000", windows=False).manual_cast
    assert nvidia_fp16_support(6, "NVIDIA GeForce GTX 1080", windows=False).manual_cast
    assert not nvidia_fp16_support(6, "NVIDIA GeForce GTX 1080", windows=True).manual_cast


def test_nvidia_16_series_matches_reference_list() -> None:
    # should_use_fp16 @ b78cec87 names twelve broken-fp16 devices; a
    # shorter list silently re-enables fp16 on one of them.
    assert len(NVIDIA_16_SERIES) == 12
    for entry in ("T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"):
        assert entry in NVIDIA_16_SERIES


def test_nvidia_bf16_compute_is_ampere_and_newer() -> None:
    assert nvidia_bf16_compute(8)
    assert nvidia_bf16_compute(12)
    assert not nvidia_bf16_compute(7)
    assert not nvidia_bf16_compute(6)
