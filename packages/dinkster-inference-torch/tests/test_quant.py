"""Packed quantized-weight routing tests."""

from __future__ import annotations

import importlib
from types import ModuleType

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
    importlib.import_module("comfy_kitchen")
    called: list[bool] = []

    def fake(qdata: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        called.append(True)
        return result

    monkeypatch.setattr(
        torch.ops.comfy_kitchen,
        "dequantize_int8_convrot_weight",
        fake,
    )
    return called


def _patch_owned_supported(monkeypatch: pytest.MonkeyPatch, kernels: ModuleType) -> None:
    def supported(_qdata: torch.Tensor, _scale: torch.Tensor, _group_size: int) -> bool:
        return True

    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight_available", lambda: True)
    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight_supported", supported)


def test_int8_convrot_prefers_owned_dequantization(monkeypatch: pytest.MonkeyPatch) -> None:
    kernels = importlib.import_module("dinkster_kernels")
    expected = torch.arange(256, dtype=torch.float32).view(1, 256)
    _patch_owned_supported(monkeypatch, kernels)

    def dequantize(_qdata: torch.Tensor, _scale: torch.Tensor, _group_size: int) -> torch.Tensor:
        return expected

    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight", dequantize)
    kitchen_called = _patch_kitchen(monkeypatch, torch.zeros_like(expected))
    actual = _stored().dequantize(torch.float16)
    assert torch.equal(actual, expected.half())
    assert not kitchen_called


def test_int8_convrot_falls_back_when_owned_route_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernels = importlib.import_module("dinkster_kernels")
    expected = torch.full((1, 256), 3.5)
    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight_available", lambda: False)
    kitchen_called = _patch_kitchen(monkeypatch, expected)
    actual = _stored().dequantize()
    assert torch.equal(actual, expected)
    assert kitchen_called == [True]


def test_int8_convrot_falls_back_when_owned_route_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernels = importlib.import_module("dinkster_kernels")
    expected = torch.full((1, 256), -2.0)
    _patch_owned_supported(monkeypatch, kernels)

    def fail(*_args: object) -> torch.Tensor:
        raise RuntimeError("kernel compile failed")

    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight", fail)
    kitchen_called = _patch_kitchen(monkeypatch, expected)
    actual = _stored().dequantize()
    assert torch.equal(actual, expected)
    assert kitchen_called == [True]


def test_int8_convrot_owned_oom_is_not_retried_with_kitchen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernels = importlib.import_module("dinkster_kernels")
    _patch_owned_supported(monkeypatch, kernels)

    def oom(*_args: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("allocation failed")

    monkeypatch.setattr(kernels, "dequantize_int8_convrot_weight", oom)
    kitchen_called = _patch_kitchen(monkeypatch, torch.zeros((1, 256)))
    with pytest.raises(torch.OutOfMemoryError, match="allocation failed"):
        _stored().dequantize()
    assert not kitchen_called
