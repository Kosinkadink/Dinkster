"""Exact source-generated resize vectors, with ComfyUI imports forbidden."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference_torch import resize

GOLDEN = json.loads((Path(__file__).parent / "goldens/tensor_resize_947c2749.json").read_text())


@pytest.fixture(autouse=True)
def without_comfy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.utils", None)


def tensor(record: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(record["values"], dtype=getattr(torch, record["dtype"])).reshape(
        record["shape"]
    )


@pytest.mark.parametrize("name", GOLDEN["cases"])
def test_exact_reference(name: str) -> None:
    case = GOLDEN["cases"][name]
    source = tensor(case["input"])
    before = source.clone()
    result = getattr(resize, case["function"])(source, **case["kwargs"])
    expected = tensor(case["output"])
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert torch.equal(result, expected)
    assert torch.equal(source, before)


def test_native_latent_resize_without_comfy() -> None:
    from dinkster_compat_comfy.native_arm import GenerationLatentResize

    source = torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 32
    result = GenerationLatentResize.execute(
        samples={"samples": source}, method="bislerp", width=128, height=64, crop="disabled"
    )
    latent: Any = result["latent"]
    assert torch.equal(
        latent["samples"], resize.common_upscale(source, 16, 8, "bislerp", "disabled")
    )
