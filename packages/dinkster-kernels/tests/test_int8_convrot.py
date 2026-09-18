"""Owned ConvRot INT8 dequantization tests."""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from dinkster_kernels import (
    CONVROT_GROUP_SIZE,
    dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_available,
    dequantize_int8_convrot_weight_supported,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and find_spec("triton") is not None),
    reason="fused ConvRot dequantization needs a CUDA device and triton",
)


def _reference(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, columns = qdata.shape
    groups = columns // CONVROT_GROUP_SIZE
    values = (qdata.float() * scale).reshape(rows, groups, CONVROT_GROUP_SIZE)
    for stride in (1, 4, 16, 64):
        blocks = CONVROT_GROUP_SIZE // (4 * stride)
        grouped = values.reshape(rows, groups, blocks, 4, stride)
        x0, x1, x2, x3 = grouped.unbind(dim=-2)
        values = (
            torch.stack(
                (
                    ((x0 + x1) + x2) - x3,
                    ((x0 + x1) - x2) + x3,
                    ((x0 - x1) + x2) + x3,
                    ((-x0 + x1) + x2) + x3,
                ),
                dim=-2,
            )
            * 0.5
        ).reshape(rows, groups, CONVROT_GROUP_SIZE)
    return values.reshape_as(qdata)


def _skip_unless_available() -> None:
    if not dequantize_int8_convrot_weight_available():
        pytest.skip("fused ConvRot probe reports unavailable")


def test_import_and_probe_never_raise() -> None:
    first = dequantize_int8_convrot_weight_available()
    assert isinstance(first, bool)
    assert dequantize_int8_convrot_weight_available() == first


def test_availability_declines_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr("dinkster_kernels.int8_convrot._common.fused_ops_available", lambda: True)
    assert not dequantize_int8_convrot_weight_available()


def test_supported_predicate_matches_contract() -> None:
    qdata = torch.zeros((3, CONVROT_GROUP_SIZE), dtype=torch.int8)
    scale = torch.ones((3, 1), dtype=torch.float32)
    assert not dequantize_int8_convrot_weight_supported(qdata, scale, CONVROT_GROUP_SIZE)
    if torch.cuda.is_available():
        qdata = qdata.cuda()
        scale = scale.cuda()
        assert dequantize_int8_convrot_weight_supported(qdata, scale, CONVROT_GROUP_SIZE)
        assert not dequantize_int8_convrot_weight_supported(qdata, scale, 64)
        assert not dequantize_int8_convrot_weight_supported(qdata[:, :-1], scale, 256)
        assert not dequantize_int8_convrot_weight_supported(qdata, scale.half(), 256)
        assert not dequantize_int8_convrot_weight_supported(qdata, scale.cpu(), 256)


@requires_cuda
@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (1, 256),
        (7, 512),
        (3, 768),
        (5, 1280),
        (33, 4096),
    ],
)
def test_bit_identical_to_radix4_reference(rows: int, columns: int) -> None:
    _skip_unless_available()
    generator = torch.Generator(device="cuda").manual_seed(rows * 100 + columns)
    qdata = torch.randint(
        -128,
        128,
        (rows, columns),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    scale = torch.empty((rows, 1), device="cuda", dtype=torch.float32).uniform_(
        -0.25,
        0.25,
        generator=generator,
    )
    expected = _reference(qdata, scale)
    actual = dequantize_int8_convrot_weight(qdata, scale, CONVROT_GROUP_SIZE)
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@requires_cuda
def test_extreme_quants_and_scales_are_bit_identical() -> None:
    _skip_unless_available()
    values = torch.tensor([-128, -127, -1, 0, 1, 126, 127, 0], dtype=torch.int8)
    qdata = values.repeat(4, 32).cuda()
    scale = torch.tensor(
        [torch.finfo(torch.float32).tiny, 1.0 / 127.0, -0.0, -3.25],
        dtype=torch.float32,
        device="cuda",
    ).reshape(4, 1)
    expected = _reference(qdata, scale)
    actual = dequantize_int8_convrot_weight(qdata, scale, CONVROT_GROUP_SIZE)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@requires_cuda
def test_noncontiguous_inputs() -> None:
    _skip_unless_available()
    generator = torch.Generator(device="cuda").manual_seed(19)
    wide = torch.randint(
        -128,
        128,
        (5, 512),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    qdata = wide[:, ::2]
    scale = torch.rand((5, 2), device="cuda", generator=generator)[:, :1]
    assert not qdata.is_contiguous() and not scale.is_contiguous()
    actual = dequantize_int8_convrot_weight(qdata, scale, CONVROT_GROUP_SIZE)
    expected = _reference(qdata, scale)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@requires_cuda
def test_empty_rows() -> None:
    _skip_unless_available()
    qdata = torch.empty((0, 256), dtype=torch.int8, device="cuda")
    scale = torch.empty((0, 1), dtype=torch.float32, device="cuda")
    output = dequantize_int8_convrot_weight(qdata, scale, CONVROT_GROUP_SIZE)
    assert output.shape == qdata.shape and output.dtype == torch.float32


@requires_cuda
def test_validation_errors() -> None:
    _skip_unless_available()
    qdata = torch.zeros((3, 256), dtype=torch.int8, device="cuda")
    scale = torch.ones((3, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="rank-2 int8"):
        dequantize_int8_convrot_weight(qdata.float(), scale, 256)
    with pytest.raises(ValueError, match="group size 256"):
        dequantize_int8_convrot_weight(qdata, scale, 64)
    with pytest.raises(ValueError, match="positive multiple of 256"):
        dequantize_int8_convrot_weight(qdata[:, :-1], scale, 256)
    with pytest.raises(ValueError, match="float32 of shape"):
        dequantize_int8_convrot_weight(qdata, scale.half(), 256)


@requires_cuda
def test_deterministic() -> None:
    _skip_unless_available()
    qdata = torch.randint(-128, 128, (17, 1024), dtype=torch.int8, device="cuda")
    scale = torch.rand((17, 1), dtype=torch.float32, device="cuda")
    first = dequantize_int8_convrot_weight(qdata, scale, 256)
    second = dequantize_int8_convrot_weight(qdata, scale, 256)
    assert torch.equal(first.view(torch.int32), second.view(torch.int32))


@requires_cuda
def test_opcheck() -> None:
    _skip_unless_available()
    qdata = torch.randint(-128, 128, (3, 512), dtype=torch.int8, device="cuda")
    scale = torch.rand((3, 1), dtype=torch.float32, device="cuda")
    torch.library.opcheck(dequantize_int8_convrot_weight, (qdata, scale, 256))

    qdata = torch.randint(-128, 128, (3, 1024), dtype=torch.int8, device="cuda")[:, ::2]
    scale = torch.rand((3, 2), dtype=torch.float32, device="cuda")[:, :1]
    assert not qdata.is_contiguous() and not scale.is_contiguous()
    torch.library.opcheck(dequantize_int8_convrot_weight, (qdata, scale, 256))
