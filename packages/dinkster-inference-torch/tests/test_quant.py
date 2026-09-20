"""Packed quantized-weight routing tests."""

from __future__ import annotations

import importlib

import pytest
import torch
from dinkster_inference_torch.quant import Int8PackedWeight


def _stored() -> Int8PackedWeight:
    qdata = torch.arange(256, dtype=torch.uint8).view(1, 256).view(torch.int8)
    scale = torch.tensor([[0.25]], dtype=torch.float32)
    return Int8PackedWeight(qdata, scale, torch.float32, True, 256)


def _patch_kitchen(
    monkeypatch: pytest.MonkeyPatch,
    result: torch.Tensor,
) -> list[bool]:
    importlib.import_module("dinkster_kitchen")
    called: list[bool] = []

    def fake(qdata: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        called.append(True)
        return result

    monkeypatch.setattr(
        torch.ops.dinkster_kitchen,
        "dequantize_int8_convrot_weight",
        fake,
    )
    return called


def test_int8_convrot_uses_kitchen_dequantization(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.arange(256, dtype=torch.float32).view(1, 256)
    kitchen_called = _patch_kitchen(monkeypatch, expected)
    actual = _stored().dequantize(torch.float16)
    assert torch.equal(actual, expected.half())
    assert kitchen_called == [True]


def test_int8_convrot_kitchen_failure_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module("dinkster_kitchen")

    def fail(*_args: object) -> torch.Tensor:
        raise RuntimeError("dequantization failed")

    monkeypatch.setattr(torch.ops.dinkster_kitchen, "dequantize_int8_convrot_weight", fail)
    with pytest.raises(RuntimeError, match="kitchen ConvRot INT8 dequantization failed"):
        _stored().dequantize()


def test_int8_convrot_kitchen_oom_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module("dinkster_kitchen")

    def oom(*_args: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("allocation failed")

    monkeypatch.setattr(torch.ops.dinkster_kitchen, "dequantize_int8_convrot_weight", oom)
    with pytest.raises(torch.OutOfMemoryError, match="allocation failed"):
        _stored().dequantize()
