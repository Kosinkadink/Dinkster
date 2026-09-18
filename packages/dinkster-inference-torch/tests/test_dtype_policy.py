"""CPU fixtures for every per-backend dtype policy row.

Test names deliberately avoid the GPU gate tokens so these policy rows
stay collected in default runs; they monkeypatch torch's observation
seams and never touch a real accelerator.
"""

from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest
import torch
from dinkster_inference_torch import DtypeSupport, bf16_support, fp16_support, supports_fp8_matmul
from dinkster_inference_torch.dtype_policy import (
    amd_fp8_matmul_supported,
    amd_gfx_arch,
    configure_amd_miopen,
    force_fp16_attention_upcast,
    rocm_version_numeric,
)

CPU = torch.device("cpu")
ACCEL = torch.device("cuda", 0)
INTEL = torch.device("xpu", 0)


def _nvidia_props(monkeypatch: pytest.MonkeyPatch, *, major: int, minor: int, name: str) -> None:
    def props(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(major=major, minor=minor, name=name)

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)


def _amd_props(monkeypatch: pytest.MonkeyPatch, *, arch: str, major: int, minor: int = 0) -> None:
    def props(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(
            major=major, minor=minor, name="AMD Radeon", gcnArchName=f"{arch}:sramecc+:xnack-"
        )

    monkeypatch.setattr(torch.version, "hip", "7.1.44064")
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)


# ------------------------------------------------------------- fp16


def test_fp16_cpu_is_refused() -> None:
    support = fp16_support(CPU)
    assert support == DtypeSupport(storage=False, compute=False)
    assert support.manual_cast is False


def test_fp16_apple_is_native() -> None:
    assert fp16_support(torch.device("mps")) == DtypeSupport(storage=True, compute=True)


def test_fp16_intel_asks_device_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    def props(has_fp16: bool) -> object:
        def query(_device: torch.device) -> SimpleNamespace:
            return SimpleNamespace(has_fp16=has_fp16)

        return query

    monkeypatch.setattr(torch.xpu, "get_device_properties", props(True))
    assert fp16_support(INTEL) == DtypeSupport(storage=True, compute=True)
    monkeypatch.setattr(torch.xpu, "get_device_properties", props(False))
    assert fp16_support(INTEL) == DtypeSupport(storage=False, compute=False)


def test_fp16_hip_builds_are_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=True)


def test_fp16_nvidia_ampere_and_newer_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=12, minor=0, name="NVIDIA RTX PRO 6000")
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=True)


def test_fp16_nvidia_pre_pascal_is_refused_outright(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=5, minor=2, name="GeForce GTX 970")
    support = fp16_support(ACCEL)
    assert support == DtypeSupport(storage=False, compute=False)
    assert support.manual_cast is False


def test_fp16_nvidia_10_series_computes_only_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _nvidia_props(monkeypatch, major=6, minor=0, name="Tesla P100-PCIE-16GB")
    expected = bool(any(platform.win32_ver()))
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=expected)


def test_fp16_nvidia_pascal_off_list_is_storage_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=6, minor=1, name="NVIDIA GeForce GT 1030")
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=False)


def test_fp16_nvidia_16_series_is_storage_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=7, minor=5, name="NVIDIA GeForce GTX 1660")
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=False)


def test_fp16_nvidia_turing_off_list_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=7, minor=5, name="NVIDIA GeForce RTX 2070")
    assert fp16_support(ACCEL) == DtypeSupport(storage=True, compute=True)


def test_fp16_unknown_device_type_raises() -> None:
    with pytest.raises(ValueError, match="no fp16 dtype policy"):
        fp16_support(torch.device("meta"))


# ------------------------------------------------------------- bf16


def test_bf16_cpu_is_refused() -> None:
    assert bf16_support(CPU) == DtypeSupport(storage=False, compute=False)


def test_bf16_apple_requires_macos_14(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.5", ("", "", ""), ""))
    assert bf16_support(torch.device("mps")) == DtypeSupport(storage=True, compute=True)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("13.6", ("", "", ""), ""))
    assert bf16_support(torch.device("mps")) == DtypeSupport(storage=False, compute=False)


def test_bf16_intel_asks_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.xpu, "is_bf16_supported", lambda: True)
    assert bf16_support(INTEL) == DtypeSupport(storage=True, compute=True)
    monkeypatch.setattr(torch.xpu, "is_bf16_supported", lambda: False)
    assert bf16_support(INTEL) == DtypeSupport(storage=False, compute=False)


def test_bf16_amd_rdna2_and_older_is_storage_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _amd_props(monkeypatch, arch="gfx1030", major=10, minor=3)
    support = bf16_support(ACCEL)
    assert support == DtypeSupport(storage=True, compute=False)
    assert support.manual_cast is True


def test_bf16_amd_rdna3_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    assert bf16_support(ACCEL) == DtypeSupport(storage=True, compute=True)


def test_bf16_nvidia_ampere_and_newer_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=8, minor=0, name="NVIDIA A100")
    assert bf16_support(ACCEL) == DtypeSupport(storage=True, compute=True)


def test_bf16_nvidia_pre_ampere_follows_cast_support(monkeypatch: pytest.MonkeyPatch) -> None:
    _nvidia_props(monkeypatch, major=7, minor=5, name="NVIDIA GeForce RTX 2070")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert bf16_support(ACCEL) == DtypeSupport(storage=True, compute=False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert bf16_support(ACCEL) == DtypeSupport(storage=False, compute=False)


def test_bf16_unknown_device_type_raises() -> None:
    with pytest.raises(ValueError, match="no bf16 dtype policy"):
        bf16_support(torch.device("meta"))


# ------------------------------------------- AMD architecture facts


def test_gfx_arch_is_none_off_hip_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    assert amd_gfx_arch(ACCEL) is None
    assert amd_gfx_arch(CPU) is None


def test_gfx_arch_strips_feature_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    assert amd_gfx_arch(ACCEL) == "gfx1100"


def test_miopen_remains_enabled_on_rdna2(monkeypatch: pytest.MonkeyPatch) -> None:
    _amd_props(monkeypatch, arch="gfx1030", major=10, minor=3)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    assert configure_amd_miopen() is False
    assert torch.backends.cudnn.enabled is True


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_miopen_is_disabled_on_newer_amd_architectures(
    monkeypatch: pytest.MonkeyPatch, arch: str
) -> None:
    _amd_props(monkeypatch, arch=arch, major=11)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.delenv("COMFYUI_ENABLE_MIOPEN", raising=False)

    assert configure_amd_miopen() is True
    assert torch.backends.cudnn.enabled is False


@pytest.mark.parametrize(("override", "disabled"), [("1", False), ("true", True), ("01", True)])
def test_miopen_override_requires_exact_one(
    monkeypatch: pytest.MonkeyPatch, override: str, disabled: bool
) -> None:
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setenv("COMFYUI_ENABLE_MIOPEN", override)

    assert configure_amd_miopen() is disabled
    assert torch.backends.cudnn.enabled is not disabled


@pytest.mark.parametrize("arch", ["", "AMD Radeon", "gfx-new"])
def test_miopen_preserves_absent_or_unparseable_architecture(
    monkeypatch: pytest.MonkeyPatch, arch: str
) -> None:
    _amd_props(monkeypatch, arch=arch, major=11)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    assert configure_amd_miopen() is False
    assert torch.backends.cudnn.enabled is True


def test_miopen_preserves_non_hip_backends_without_querying_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    def unexpected(_device: torch.device) -> object:
        raise AssertionError("non-HIP backends must not query AMD properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    assert configure_amd_miopen() is False
    assert torch.backends.cudnn.enabled is True


def test_hip_runtime_version_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.1.44064")
    assert rocm_version_numeric() == (7, 1)
    monkeypatch.setattr(torch.version, "hip", None)
    assert rocm_version_numeric() == (6, -1)
    monkeypatch.setattr(torch.version, "hip", "junk")
    assert rocm_version_numeric() == (6, -1)


# ------------------------------------------------ native fp8 matmul


def test_amd_fp8_matmul_needs_listed_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.12.0+rocm7.1")
    _amd_props(monkeypatch, arch="gfx1200", major=12)
    assert amd_fp8_matmul_supported(ACCEL) is True
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    assert amd_fp8_matmul_supported(ACCEL) is False


def test_amd_fp8_matmul_needs_recent_torch_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _amd_props(monkeypatch, arch="gfx1200", major=12)
    monkeypatch.setattr(torch, "__version__", "2.6.0+rocm6.4")
    assert amd_fp8_matmul_supported(ACCEL) is False
    monkeypatch.setattr(torch, "__version__", "2.12.0+rocm6.3")
    monkeypatch.setattr(torch.version, "hip", "6.3.42131")
    assert amd_fp8_matmul_supported(ACCEL) is False


def test_fp8_matmul_gate_covers_amd_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "__version__", "2.12.0+rocm7.1")
    _amd_props(monkeypatch, arch="gfx1200", major=12)
    assert supports_fp8_matmul(ACCEL) is True
    _amd_props(monkeypatch, arch="gfx1100", major=11)
    assert supports_fp8_matmul(ACCEL) is False


# ------------------------------------------- fp16 attention upcast


@pytest.mark.parametrize(
    ("release", "expected"),
    [
        ("14.5", True),
        ("14.5.1", True),
        ("15.0", True),
        ("14.4", False),
        ("14.4.9", False),
        ("14", False),
        ("13.6", False),
        ("", False),
    ],
)
def test_fp16_attention_upcast_requires_macos_14_5(
    monkeypatch: pytest.MonkeyPatch, release: str, expected: bool
) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: (release, ("", "", ""), ""))
    assert force_fp16_attention_upcast() is expected


def test_fp16_attention_upcast_tolerates_unparseable_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.5b1", ("", "", ""), ""))
    assert force_fp16_attention_upcast() is False
