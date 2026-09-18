"""Strict graph-compiler payload adapter proofs."""

from __future__ import annotations

import struct

import pytest
import torch
from dinkster_inference import (
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    PayloadDescriptor,
    PayloadReference,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)
from dinkster_inference_torch import (
    TensorPayloadError,
    payload_binding_to_tensor,
    tensor_to_payload_binding,
)

DTYPES = (
    (torch.float64, "F64"),
    (torch.float32, "F32"),
    (torch.float16, "F16"),
    (torch.bfloat16, "BF16"),
    (torch.int64, "I64"),
    (torch.int32, "I32"),
    (torch.int16, "I16"),
    (torch.int8, "I8"),
    (torch.uint8, "U8"),
    (torch.bool, "BOOL"),
)


@pytest.mark.parametrize(("dtype", "wire"), DTYPES)
def test_every_supported_dtype_round_trips_with_owned_content(
    dtype: torch.dtype, wire: str
) -> None:
    source = torch.tensor([[0, 1, 2], [3, 4, 1]], dtype=dtype)
    binding = tensor_to_payload_binding("tensor", source, space="proof")
    assert binding.dtype == wire
    decoded = payload_binding_to_tensor(binding)
    assert decoded.dtype == dtype
    assert torch.equal(decoded, source)
    source.fill_(0)
    assert torch.count_nonzero(decoded).item() > 0
    decoded.fill_(0)
    assert binding.data != bytes(len(binding.data))


@pytest.mark.parametrize(("dtype", "wire"), DTYPES)
@pytest.mark.parametrize("shape", ((0,), (2, 0, 3)))
def test_zero_element_non_scalar_round_trips(
    dtype: torch.dtype, wire: str, shape: tuple[int, ...]
) -> None:
    source = torch.empty(shape, dtype=dtype)
    binding = tensor_to_payload_binding("empty", source, space="proof")
    assert binding.dtype == wire
    assert binding.data == b""
    decoded = payload_binding_to_tensor(binding)
    assert decoded.shape == shape
    assert decoded.dtype == dtype
    assert decoded.numel() == 0


def test_bytes_are_little_endian_bfloat_integer_view_and_bool_is_canonical() -> None:
    f32 = tensor_to_payload_binding(
        "f32", torch.tensor([1.0, -2.0], dtype=torch.float32), space="proof"
    )
    assert f32.data == struct.pack("<ff", 1.0, -2.0)
    bf16_source = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    bf16 = tensor_to_payload_binding("bf16", bf16_source, space="proof")
    assert bf16.data == bytes(bf16_source.view(torch.uint8).tolist())
    boolean = tensor_to_payload_binding("bool", torch.tensor([True, False, True]), space="proof")
    assert boolean.data == b"\x01\x00\x01"
    bad = type(boolean)(
        boolean.reference_id,
        boolean.shape,
        boolean.dtype,
        boolean.space,
        b"\x02\x00\x01",
    )
    with pytest.raises(TensorPayloadError, match="canonical"):
        payload_binding_to_tensor(bad)


def test_noncontiguous_detached_normalization_and_carrier_byte_stability() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    source.requires_grad_(True)
    binding = tensor_to_payload_binding("tensor", source, space="text")
    assert torch.equal(payload_binding_to_tensor(binding), source.detach())
    descriptor = PayloadDescriptor(PayloadReference("tensor"), binding.shape, "F32", "text")
    conditioning = ConditioningSet((ConditioningRecord(((ConditioningChannel.TEXT, descriptor),)),))
    first = encode_conditioning_carrier(make_conditioning_carrier(conditioning, (binding,)))
    rebuilt = tensor_to_payload_binding("tensor", payload_binding_to_tensor(binding), space="text")
    second = encode_conditioning_carrier(make_conditioning_carrier(conditioning, (rebuilt,)))
    assert first == second


def test_scalar_sparse_quantized_complex_unsigned_and_unknown_refuse_without_casts() -> None:
    values = (
        torch.tensor(1.0),
        torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,)),
        torch.quantize_per_tensor(torch.tensor([1.0]), 0.1, 0, torch.qint8),
        torch.tensor([1 + 2j]),
    )
    for value in values:
        with pytest.raises(TensorPayloadError):
            tensor_to_payload_binding("bad", value, space="proof")
    if hasattr(torch, "uint16"):
        with pytest.raises(TensorPayloadError, match="unsupported"):
            tensor_to_payload_binding("bad", torch.tensor([1], dtype=torch.uint16), space="proof")
    from dinkster_inference import PayloadBinding

    unsupported = PayloadBinding("bad", (1,), "U16", "proof", b"\x01\x00")
    with pytest.raises(TensorPayloadError, match="unsupported"):
        payload_binding_to_tensor(unsupported)
