"""Executed-reference parity and ownership tests for Wan 2.1 CLIP vision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_vision_fill import fill_state_dict
from dinkster_inference.clip_vision import ClipVisionConfig, clip_vision_layout
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.clip_vision import (
    Wan21ClipVisionEncoder,
    clip_vision_preprocess,
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import ResidencyRouted

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "clip_vision_goldens.json").read_text())


def _decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def _model(*, kernel: Any = None) -> Wan21ClipVisionEncoder:
    case = GOLDENS["reduced"]
    config = ClipVisionConfig(**case["config"])
    kwargs = {} if kernel is None else {"attention_kernel": kernel}
    model = Wan21ClipVisionEncoder(config, **kwargs)
    model.load_state_dict(fill_state_dict(case["state_dict"]), strict=True)
    return model


def test_full_official_layout_matches_reference_and_torch_free_contract() -> None:
    with torch.device("meta"):
        model = Wan21ClipVisionEncoder()
    actual = [[key, list(value.shape)] for key, value in sorted(model.state_dict().items())]
    assert actual == GOLDENS["official_layout"]
    assert {key: tuple(shape) for key, shape in actual} == clip_vision_layout()


def test_preprocess_and_penultimate_state_match_executed_reference() -> None:
    case = GOLDENS["reduced"]
    image = _decode(case["image"])
    pixels = clip_vision_preprocess(image, size=28)
    assert torch.equal(pixels, _decode(case["preprocessed"]))
    output = _model()(image)
    torch.testing.assert_close(output, _decode(case["penultimate"]), rtol=0, atol=1e-6)
    assert output.shape == (1, 5, 16)


def test_preprocess_stretches_non_square_images_when_crop_is_disabled() -> None:
    image = torch.linspace(0.0, 1.0, 90).reshape(1, 5, 6, 3)
    actual = clip_vision_preprocess(image, size=4, crop=False)
    resized = functional.interpolate(
        image.movedim(-1, 1),
        size=(4, 4),
        mode="bicubic",
        antialias=True,
    )
    resized = (255.0 * resized).clamp(0, 255).round() / 255.0
    mean = resized.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
    std = resized.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)

    assert torch.equal(actual, (resized - mean) / std)
    assert not torch.equal(actual, clip_vision_preprocess(image, size=4))


def test_encoder_forces_reference_float32_compute_for_lower_precision_images() -> None:
    image = _decode(GOLDENS["reduced"]["image"]).to(torch.float16)
    output = _model()(image)
    assert output.dtype is torch.float32
    assert torch.isfinite(output).all()


def test_attention_uses_selected_noncausal_kernel() -> None:
    spy = CallableModuleKernel(select_attention("clip").kernel)
    model = _model(kernel=spy)
    output = model(_decode(GOLDENS["reduced"]["image"]))
    assert output.shape == (1, 5, 16)
    assert len(spy.calls) == 2
    assert all(call["q_shape"] == (1, 4, 5, 4) for call in spy.calls)
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(model, spy)


def test_all_state_owners_are_routed_and_offloaded_forward_matches() -> None:
    model = _model()
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]
    assert owners
    assert all(isinstance(module, ResidencyRouted) for module in owners)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    image = _decode(GOLDENS["reduced"]["image"])
    resident = model(image)
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = model(image)
    assert torch.equal(offloaded, resident)


@pytest.mark.parametrize(
    "image",
    (
        torch.empty(1, 16, 16, 2),
        torch.zeros(1, 16, 16, 3, dtype=torch.int64),
        torch.full((1, 16, 16, 3), -0.1),
        torch.full((1, 16, 16, 3), float("nan")),
    ),
)
def test_invalid_public_images_refuse_before_model_work(image: torch.Tensor) -> None:
    model = _model()
    called = False

    def mark_called(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal called
        called = True

    handle = model.vision_model.embeddings.patch_embedding.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError):
            model(image)
    finally:
        handle.remove()
    assert not called
