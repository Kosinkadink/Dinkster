"""Owned fp8 scaled-matmul route tests.

Availability and dispatch are host checks testable on any device; the
execution test needs a CUDA device with hardware fp8 support (compute
capability 8.9+) and pins bit-identity against torch's own scaled-mm
operation called directly, since the route's whole contract is that it
adds no numerics of its own.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from dinkster_kernels import scaled_mm, scaled_mm_available

# The package rebinds its ``scaled_mm`` attribute to the function, so
# the module itself comes from the import system.
scaled_mm_module = importlib.import_module("dinkster_kernels.scaled_mm")

E4M3 = torch.float8_e4m3fn


def _cpu_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    input = (torch.randn(2, 32, generator=generator) * 0.1).to(E4M3)
    weight = (torch.randn(3, 32, generator=generator) * 0.1).to(E4M3).t()
    scale_a = torch.tensor(0.75, dtype=torch.float32)
    scale_b = torch.tensor(1.25, dtype=torch.float32)
    return input, weight, scale_a, scale_b


def test_available_requires_a_cuda_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert scaled_mm_available() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert scaled_mm_available() is True


def test_available_declines_hip_runtimes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    assert scaled_mm_available() is False


def test_functional_surface_resolves_on_torch_210_plus() -> None:
    resolved = scaled_mm_module._resolve_functional_scaled_mm()  # pyright: ignore[reportPrivateUsage]
    from torch.nn.functional import ScalingType, SwizzleType
    from torch.nn.functional import scaled_mm as functional_scaled_mm

    assert resolved is not None
    assert resolved.op is functional_scaled_mm
    assert resolved.tensor_wise is ScalingType.TensorWise
    assert resolved.no_swizzle is SwizzleType.NO_SWIZZLE


def test_functional_dispatch_pins_tensorwise_unswizzled_slow_accum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.functional import ScalingType, SwizzleType

    input, weight, scale_a, scale_b = _cpu_operands()
    bias = torch.randn(3, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(12))
    seen: dict[str, object] = {}
    expected = torch.zeros((2, 3), dtype=torch.bfloat16)

    def fake(input: torch.Tensor, weight: torch.Tensor, **kwargs: object) -> torch.Tensor:
        seen.update(input=input, weight=weight, **kwargs)
        return expected

    resolved = scaled_mm_module._resolve_functional_scaled_mm()  # pyright: ignore[reportPrivateUsage]
    assert resolved is not None
    monkeypatch.setattr(scaled_mm_module, "_FUNCTIONAL_SCALED_MM", resolved._replace(op=fake))
    result = scaled_mm(
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=torch.bfloat16,
    )
    assert result is expected
    assert seen["input"] is input
    assert seen["weight"] is weight
    assert seen["scale_a"] is scale_a
    assert seen["scale_b"] is scale_b
    assert seen["bias"] is bias
    assert seen["output_dtype"] is torch.bfloat16
    assert seen["scale_recipe_a"] is ScalingType.TensorWise
    assert seen["scale_recipe_b"] is ScalingType.TensorWise
    assert seen["swizzle_a"] is SwizzleType.NO_SWIZZLE
    assert seen["swizzle_b"] is SwizzleType.NO_SWIZZLE
    assert seen["use_fast_accum"] is False


def test_legacy_dispatch_passes_operands_through(monkeypatch: pytest.MonkeyPatch) -> None:
    input, weight, scale_a, scale_b = _cpu_operands()
    seen: dict[str, object] = {}
    expected = torch.zeros((2, 3), dtype=torch.float32)

    def fake(input: torch.Tensor, weight: torch.Tensor, **kwargs: object) -> torch.Tensor:
        seen.update(input=input, weight=weight, **kwargs)
        return expected

    monkeypatch.setattr(scaled_mm_module, "_FUNCTIONAL_SCALED_MM", None)
    monkeypatch.setattr(torch, "_scaled_mm", fake)
    result = scaled_mm(
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,
        out_dtype=torch.float32,
    )
    assert result is expected
    assert seen == {
        "input": input,
        "weight": weight,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "bias": None,
        "out_dtype": torch.float32,
    }


def test_legacy_dispatch_normalizes_the_torch_24_tuple_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input, weight, scale_a, scale_b = _cpu_operands()
    expected = torch.zeros((2, 3), dtype=torch.float32)

    def fake(
        input: torch.Tensor, weight: torch.Tensor, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return expected, torch.zeros(())

    monkeypatch.setattr(scaled_mm_module, "_FUNCTIONAL_SCALED_MM", None)
    monkeypatch.setattr(torch, "_scaled_mm", fake)
    result = scaled_mm(
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,
        out_dtype=torch.float32,
    )
    assert result is expected


requires_fp8_cuda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 9)
    ),
    reason="hardware fp8 matmul needs a CUDA device with compute capability 8.9+",
)


@requires_fp8_cuda
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_matches_the_direct_torch_operation_bitwise(out_dtype: torch.dtype) -> None:
    device = torch.device("cuda:0")
    generator = torch.Generator(device).manual_seed(7)
    input = (torch.randn(5, 32, device=device, dtype=torch.bfloat16, generator=generator) * 0.1).to(
        E4M3
    )
    weight = (
        (torch.randn(16, 32, device=device, dtype=torch.bfloat16, generator=generator) * 0.1)
        .to(E4M3)
        .t()
    )
    scale_a = torch.tensor(0.75, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.25, device=device, dtype=torch.float32)
    bias = torch.randn(16, device=device, dtype=out_dtype, generator=generator)
    got = scaled_mm(
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )
    reference = torch._scaled_mm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )
    assert got.dtype == out_dtype
    assert torch.equal(got.view(torch.int16), reference.view(torch.int16))
