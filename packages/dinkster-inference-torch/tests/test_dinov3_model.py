"""The native DINOv3 image encoder against the executed reference.

Every golden in goldens/triposplat_goldens.json was produced by
RUNNING the reference encoder (comfy/image_encoders/dino3.py
DINOv3ViTModel and comfy/clip_model.py clip_preprocess @ the audited
baseline, tools/gen_triposplat_goldens.py) with attention forced to
pytorch SDPA. Weights come from the shared deterministic hash
(unet_fill.py) and inputs from its ``hashed_input`` namespace.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import DINOv3ViTConfig
from dinkster_inference.dinov3 import dinov3_vith_layout
from dinkster_inference_torch import (
    CastOperations,
    DINOv3ViTModel,
    ResidencyRouted,
    dinov3_preprocess,
    enroll_component,
)
from dinkster_inference_torch.operations import bound_compute_dtype
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "triposplat_goldens.json").read_text())


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries() -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"]["dinov3_encode"]["state_dict"]]


def tiny_config() -> DINOv3ViTConfig:
    """The golden case's generator kwargs as a real (tiny)
    DINOv3ViTConfig; unlike the frozen family configs it accepts any
    consistent architecture."""
    spec = GOLDENS["cases"]["dinov3_encode"]["config"]
    return DINOv3ViTConfig(
        hidden_size=spec["hidden_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        num_register_tokens=spec["num_register_tokens"],
        intermediate_size=spec["intermediate_size"],
        image_size=spec["image_size"],
        patch_size=spec["patch_size"],
        rope_theta=spec["rope_theta"],
        layer_norm_eps=spec["layer_norm_eps"],
        image_mean=tuple(spec["image_mean"]),
        image_std=tuple(spec["image_std"]),
    )


def build_model() -> DINOv3ViTModel:
    model = DINOv3ViTModel(tiny_config())
    model.load_state_dict(fill_state_dict(golden_entries()), strict=True)
    return model


def encode_pixels() -> torch.Tensor:
    spec = GOLDENS["cases"]["dinov3_encode"]
    size = spec["config"]["image_size"]
    return hashed_input("dinov3_encode:pixels", (spec["batch"], 3, size, size))


# ------------------------------------------------------ key layout


def test_state_dict_layout_matches_executed_reference() -> None:
    ours = sorted(
        (key, list(value.shape))
        for key, value in DINOv3ViTModel(tiny_config()).state_dict().items()
    )
    assert ours == golden_entries()


def test_torch_free_layout_predicts_the_module() -> None:
    predicted = sorted(
        (key, list(shape)) for key, shape in dinov3_vith_layout(tiny_config()).items()
    )
    assert predicted == golden_entries()


def test_full_size_module_matches_reference_layout() -> None:
    """The real ViT-H/16+ architecture, constructed on the meta device
    (initless factories never touch the storage), against the reference
    encoder's own full-size listing."""
    with torch.device("meta"):
        model = DINOv3ViTModel()
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["dinov3_vith"]]
    assert ours == golden
    predicted = sorted((key, list(shape)) for key, shape in dinov3_vith_layout().items())
    assert predicted == golden


# ---------------------------------------------------- golden replay


def test_encoded_sequence_matches_executed_reference() -> None:
    spec = GOLDENS["cases"]["dinov3_encode"]
    model = build_model()
    with torch.no_grad():
        sequence = model(encode_pixels())
    torch.testing.assert_close(sequence, dec(spec["sequence"]), rtol=1e-4, atol=1e-5)
    # The reference's pooled output is the class token at row zero.
    torch.testing.assert_close(sequence[:, 0], dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_preprocess_matches_executed_reference() -> None:
    """The reference clip_preprocess leg over a non-square image:
    bicubic resize, center crop, 8-bit quantization, and ImageNet
    normalization."""
    spec = GOLDENS["cases"]["dinov3_preprocess"]
    image = (hashed_input("dinov3_preprocess:image", tuple(spec["image_shape"])) + 1.0).clamp(
        0, 2
    ) * 0.5
    with torch.no_grad():
        preprocessed = dinov3_preprocess(image, tiny_config(), crop=True)
    torch.testing.assert_close(preprocessed, dec(spec["output"]), rtol=1e-4, atol=1e-5)


def test_fp16_forward_stays_finite() -> None:
    model = build_model().half()
    with torch.no_grad():
        sequence = model(encode_pixels().half())
    assert sequence.dtype == torch.float16
    assert torch.isfinite(sequence).all()


# ------------------------------------------------------ residency


def test_enrolled_prefetch_covers_exactly_the_persistent_state() -> None:
    """The rope table is a derived constant outside the residency
    store; prefetch must request every stored key and nothing else."""
    model = build_model()
    assert all("inv_freq" not in key for key in model.state_dict())

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    pixels = encode_pixels()
    with torch.no_grad():
        resident = model(pixels)

    mechanism.unload()
    requests: dict[str, torch.dtype | None] = {}
    for module in model.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        prefetch = module.residency_prefetch()
        if prefetch is None:
            continue
        for key, dtype in prefetch[1]:
            assert key not in requests
            requests[key] = dtype
    assert requests == {key: torch.float32 for key in model.state_dict()}

    with torch.no_grad():
        offloaded = model(pixels)
    assert torch.equal(offloaded, resident)


def test_fp32_manual_cast_runs_over_reduced_storage_when_loaded_or_offloaded() -> None:
    model = DINOv3ViTModel(
        tiny_config(),
        operations=CastOperations(torch.float32),
    )
    state = {
        key: value.to(torch.bfloat16) for key, value in fill_state_dict(golden_entries()).items()
    }
    model.load_state_dict(state, strict=True, assign=True)
    assert {value.dtype for value in model.state_dict().values() if value.is_floating_point()} == {
        torch.bfloat16
    }
    assert bound_compute_dtype(model.embeddings.patch_embeddings) is torch.float32

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    pixels = encode_pixels().float()
    with torch.no_grad():
        resident = model(pixels)
    assert resident.dtype is torch.float32

    mechanism.unload()
    with torch.no_grad():
        offloaded = model(pixels)
    assert torch.equal(offloaded, resident)


# ------------------------------------------------------- refusals


def test_hidden_size_must_divide_over_heads() -> None:
    spec = SimpleNamespace(hidden_size=10, num_attention_heads=3)
    with pytest.raises(ValueError, match="divide evenly"):
        DINOv3ViTModel(cast(DINOv3ViTConfig, spec))


def test_head_dimension_must_carry_the_patch_rope() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        DINOv3ViTModel(DINOv3ViTConfig(hidden_size=12, num_attention_heads=2))


def test_forward_refuses_malformed_pixels() -> None:
    model = build_model()
    size = tiny_config().image_size
    with pytest.raises(ValueError, match="pixel values must be"):
        model(torch.zeros(1, 4, size, size))
    with pytest.raises(ValueError, match="floating-point"):
        model(torch.zeros(1, 3, size, size, dtype=torch.int64))
    with pytest.raises(ValueError, match="positive multiples"):
        model(torch.zeros(1, 3, size + 2, size))


def test_preprocess_refuses_malformed_images() -> None:
    good = torch.rand(1, 8, 8, 3)
    with pytest.raises(TypeError, match="exact strided"):
        dinov3_preprocess(torch.nn.Parameter(good), tiny_config())
    with pytest.raises(ValueError, match="non-empty NHWC"):
        dinov3_preprocess(good[0], tiny_config())
    with pytest.raises(ValueError, match="at least three channels"):
        dinov3_preprocess(good[..., :2], tiny_config())
    with pytest.raises(ValueError, match="floating-point"):
        dinov3_preprocess((good * 255).to(torch.uint8), tiny_config())
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        dinov3_preprocess(good + 1.0, tiny_config())
