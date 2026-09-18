"""Owned per-tensor FP8 input quantization tests."""

from __future__ import annotations

from importlib.util import find_spec

import dinkster_kernels.fp8_quantize as fp8_quantize_module
import pytest
import torch
from dinkster_kernels import (
    quantize_per_tensor_fp8,
    quantize_per_tensor_fp8_available,
    quantize_per_tensor_fp8_supported,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and find_spec("triton") is not None),
    reason="owned FP8 quantization needs a CUDA device and triton",
)


def _reference(
    input: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype,
) -> torch.Tensor:
    fp8_max = torch.finfo(output_type).max
    values = input.float() / scale.float()
    return torch.clamp(values, -fp8_max, fp8_max).to(output_type)


def _skip_unless_available() -> None:
    if not quantize_per_tensor_fp8_available():
        pytest.skip("owned FP8 quantizer probe reports unavailable")


def test_import_and_probe_never_raise() -> None:
    first = quantize_per_tensor_fp8_available()
    assert isinstance(first, bool)
    assert quantize_per_tensor_fp8_available() == first


@requires_cuda
def test_probe_checks_both_output_types(monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_unless_available()
    calls: list[torch.dtype] = []
    original = fp8_quantize_module.quantize_per_tensor_fp8

    def tracked_quantize(
        input: torch.Tensor,
        scale: torch.Tensor,
        output_type: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        calls.append(output_type)
        return original(input, scale, output_type)

    monkeypatch.setattr(fp8_quantize_module, "quantize_per_tensor_fp8", tracked_quantize)
    assert fp8_quantize_module._probe()  # pyright: ignore[reportPrivateUsage] - probe regression
    assert calls == [torch.float8_e4m3fn, torch.float8_e5m2]


def test_availability_declines_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    assert not quantize_per_tensor_fp8_available()


def test_supported_predicate_matches_contract() -> None:
    input = torch.zeros((2, 16), dtype=torch.bfloat16)
    scale = torch.ones((), dtype=torch.float32)
    assert not quantize_per_tensor_fp8_supported(input, scale)
    if torch.cuda.is_available():
        input = input.cuda()
        scale = scale.cuda()
        assert not quantize_per_tensor_fp8_supported(input, scale)
        eligible = torch.zeros((8, 32), dtype=torch.bfloat16, device="cuda")
        assert quantize_per_tensor_fp8_supported(eligible, scale)
        assert quantize_per_tensor_fp8_supported(eligible.float(), scale, torch.float8_e5m2)
        assert not quantize_per_tensor_fp8_supported(input.int(), scale)
        assert not quantize_per_tensor_fp8_supported(eligible, torch.ones(2, device="cuda"))
        assert not quantize_per_tensor_fp8_supported(eligible, scale.half())
        assert not quantize_per_tensor_fp8_supported(eligible, scale.cpu())
        assert not quantize_per_tensor_fp8_supported(eligible, scale, torch.float16)


@requires_cuda
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("shape", [(1,), (2, 17), (7, 512), (3, 5, 768)])
def test_bit_identical_to_eager_reference(
    input_dtype: torch.dtype,
    output_type: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    _skip_unless_available()
    generator = torch.Generator(device="cuda").manual_seed(sum(shape) + input_dtype.itemsize)
    input = torch.randn(shape, device="cuda", dtype=input_dtype, generator=generator) * 13
    scale = torch.tensor(0.03, device="cuda", dtype=torch.float32)
    expected = _reference(input, scale, output_type)
    actual = quantize_per_tensor_fp8(input, scale, output_type)
    assert actual.dtype == output_type and actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@requires_cuda
def test_thresholds_saturation_and_nonfinite_values_match_kitchen() -> None:
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    _skip_unless_available()
    scale = torch.tensor(0.03, device="cuda", dtype=torch.float32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    midpoint = (torch.tensor(432.0, device="cuda") * scale).to(torch.bfloat16)
    input = torch.tensor(
        [
            0.0,
            -0.0,
            fp8_max * 0.03,
            -fp8_max * 0.03,
            fp8_max * 0.06,
            -fp8_max * 0.06,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    input = torch.cat(
        (
            input,
            midpoint.reshape(1),
            torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))).reshape(1),
            torch.nextafter(midpoint, torch.full_like(midpoint, float("-inf"))).reshape(1),
        )
    )
    expected = comfy_kitchen.quantize_per_tensor_fp8(input, scale, torch.float8_e4m3fn)
    actual = quantize_per_tensor_fp8(input, scale, torch.float8_e4m3fn)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@requires_cuda
@pytest.mark.parametrize("output_type", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_float32_rounding_boundaries_match_eager(output_type: torch.dtype) -> None:
    _skip_unless_available()
    input = torch.tensor(
        [-10.083683967590332, -3.0018301010131836, 2.0410664081573486, 4.083658695220947],
        device="cuda",
        dtype=torch.float32,
    )
    scale = torch.tensor(0.03, device="cuda", dtype=torch.float32)
    expected = _reference(input, scale, output_type)
    actual = quantize_per_tensor_fp8(input, scale, output_type)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@requires_cuda
@pytest.mark.parametrize(
    ("output_type", "min_subnormal", "max_subnormal", "min_normal", "expected_positive_bits"),
    [
        (torch.float8_e4m3fn, 2.0**-9, 7.0 * 2.0**-9, 2.0**-6, [0, 0, 0, 1, 1, 7, 7, 8, 8, 8]),
        (torch.float8_e5m2, 2.0**-16, 3.0 * 2.0**-16, 2.0**-14, [0, 0, 0, 1, 1, 3, 3, 4, 4, 4]),
    ],
)
def test_float32_subnormal_transitions_match_eager(
    output_type: torch.dtype,
    min_subnormal: float,
    max_subnormal: float,
    min_normal: float,
    expected_positive_bits: list[int],
) -> None:
    _skip_unless_available()
    half_min_subnormal = torch.tensor(min_subnormal / 2, dtype=torch.float32)
    normal_midpoint = torch.tensor((max_subnormal + min_normal) / 2, dtype=torch.float32)
    negative_infinity = torch.full((), float("-inf"), dtype=torch.float32)
    positive_infinity = torch.full((), float("inf"), dtype=torch.float32)
    positive = torch.tensor(
        [
            0.0,
            torch.nextafter(half_min_subnormal, negative_infinity),
            half_min_subnormal,
            torch.nextafter(half_min_subnormal, positive_infinity),
            min_subnormal,
            max_subnormal,
            torch.nextafter(normal_midpoint, negative_infinity),
            normal_midpoint,
            torch.nextafter(normal_midpoint, positive_infinity),
            min_normal,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    input = torch.cat((-positive, positive))
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    expected = _reference(input, scale, output_type)
    expected_bits = torch.tensor(
        [bit | 0x80 for bit in expected_positive_bits] + expected_positive_bits,
        device="cuda",
        dtype=torch.uint8,
    )
    assert torch.equal(expected.view(torch.uint8), expected_bits)
    actual = quantize_per_tensor_fp8(input, scale, output_type)
    assert torch.equal(actual.view(torch.uint8), expected_bits)


@requires_cuda
def test_noncontiguous_input_and_scale() -> None:
    _skip_unless_available()
    input = torch.randn((5, 66), device="cuda", dtype=torch.float16)[:, ::2]
    scale = torch.tensor([0.5, 0.03], device="cuda", dtype=torch.float32)[1:]
    assert not input.is_contiguous() and scale.storage_offset() == 1
    expected = _reference(input, scale, torch.float8_e4m3fn)
    actual = quantize_per_tensor_fp8(input, scale)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@requires_cuda
def test_empty_input() -> None:
    _skip_unless_available()
    input = torch.empty((0, 32), dtype=torch.float32, device="cuda")
    output = quantize_per_tensor_fp8(input, torch.ones((), device="cuda"))
    assert output.shape == input.shape and output.dtype == torch.float8_e4m3fn


@requires_cuda
def test_validation_errors() -> None:
    _skip_unless_available()
    input = torch.zeros((3, 32), dtype=torch.float32, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="input must be"):
        quantize_per_tensor_fp8(input.int(), scale)
    with pytest.raises(ValueError, match="one float32 value"):
        quantize_per_tensor_fp8(input, torch.ones(2, device="cuda"))
    with pytest.raises(ValueError, match="one float32 value"):
        quantize_per_tensor_fp8(input, scale.half())
    with pytest.raises(ValueError, match="output_type must be"):
        quantize_per_tensor_fp8(input, scale, torch.float16)


@requires_cuda
def test_deterministic() -> None:
    _skip_unless_available()
    input = torch.randn((17, 1024), dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.07, dtype=torch.float32, device="cuda")
    first = quantize_per_tensor_fp8(input, scale)
    second = quantize_per_tensor_fp8(input, scale)
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))


@requires_cuda
def test_opcheck() -> None:
    _skip_unless_available()
    input = torch.randn((3, 32), dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.03, dtype=torch.float32, device="cuda")
    torch.library.opcheck(quantize_per_tensor_fp8, (input, scale, torch.float8_e4m3fn))

    input = torch.randn((3, 64), dtype=torch.float32, device="cuda")[:, ::2]
    assert not input.is_contiguous()
    torch.library.opcheck(quantize_per_tensor_fp8, (input, scale, torch.float8_e5m2))
