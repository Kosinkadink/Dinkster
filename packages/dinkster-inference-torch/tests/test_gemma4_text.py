"""Gemma 4 LTX text math against the pinned ComfyUI implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_fill import fill_state_dict
from dinkster_inference import (
    GEMMA4_LTX_12B_CONFIG,
    GemmaTextConfig,
    gemma_text_layout,
    tokenize_ltx_gemma_prompt,
)
from dinkster_inference_torch import GemmaTextModel, select_attention
from golden_files import assert_reference_tensor, load_platform_golden

GOLDENS = load_platform_golden(
    Path(__file__).parents[3] / "tests" / "goldens" / "ltx_gemma4_text_goldens.json",
    allow_portable_fallback=True,
)


def tiny_config() -> GemmaTextConfig:
    spec = dict(GOLDENS["model"]["config"])
    spec["sliding_pattern"] = tuple(spec["sliding_pattern"])
    return GemmaTextConfig(**spec)


def build_tiny(*, attention_kernel: object | None = None) -> GemmaTextModel:
    kwargs = {} if attention_kernel is None else {"attention_kernel": attention_kernel}
    model = GemmaTextModel(tiny_config(), **kwargs)  # type: ignore[arg-type]
    model.load_state_dict(fill_state_dict(GOLDENS["model"]["state_dict"]), strict=True)
    return model


def test_tiny_gemma4_state_layout_matches_executed_reference() -> None:
    assert sorted((key, list(value.shape)) for key, value in build_tiny().state_dict().items()) == [
        (key, list(shape)) for key, shape in GOLDENS["model"]["state_dict"]
    ]


def test_full_gemma4_layout_has_global_geometry_and_no_global_value_projection() -> None:
    layout = gemma_text_layout(GEMMA4_LTX_12B_CONFIG)
    assert len(layout) == 48 * 14 + 2 - 8
    assert layout["layers.0.self_attn.q_proj.weight"] == (4096, 3840)
    assert layout["layers.0.self_attn.k_proj.weight"] == (2048, 3840)
    assert layout["layers.5.self_attn.q_proj.weight"] == (8192, 3840)
    assert layout["layers.5.self_attn.k_proj.weight"] == (512, 3840)
    assert "layers.5.self_attn.v_proj.weight" not in layout
    assert layout["layers.5.self_attn.o_proj.weight"] == (3840, 8192)
    assert layout["layers.47.layer_scalar"] == (1,)

    with torch.device("meta"):
        model = GemmaTextModel(GEMMA4_LTX_12B_CONFIG)
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == sorted(
        layout.items()
    )


@pytest.mark.parametrize("case", range(len(GOLDENS["model"]["cases"])))
def test_gemma4_stack_matches_executed_reference_probes(case: int) -> None:
    spec = GOLDENS["model"]["cases"][case]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    with torch.no_grad():
        got = build_tiny()(ids, mask)
    stack = spec["stack"]
    assert list(got.shape) == stack["shape"]
    assert str(got.dtype).removeprefix("torch.") == stack["dtype"]
    assert_reference_tensor(
        got.float().flatten()[:: stack["sample_stride"]],
        torch.tensor(stack["samples"]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("case", "local_gqa"),
    ((0, False), (1, True)),
)
def test_gemma4_uses_reference_gqa_and_expanded_sliding_paths(
    case: int,
    local_gqa: bool,
) -> None:
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    model = build_tiny(attention_kernel=spy)
    spec = GOLDENS["model"]["cases"][case]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)

    with torch.no_grad():
        model(ids, mask)

    assert_kernel_is_not_model_state(model, spy)
    assert len(spy.calls) == 6
    for index, call in enumerate(spy.calls):
        if index == 5:
            assert call["q_shape"][1] == 4
            assert call["k_shape"][1] == call["v_shape"][1] == 1
            assert call["enable_gqa"] is True
        else:
            expected_kv_heads = 2 if local_gqa else 4
            assert call["q_shape"][1] == 4
            assert call["k_shape"][1] == call["v_shape"][1] == expected_kv_heads
            assert call["enable_gqa"] is local_gqa
        assert call["scale"] == 1.0


def test_gemma4_prompt_policy_adds_bos_without_eos_and_left_pads() -> None:
    tokens = tokenize_ltx_gemma_prompt(
        "cat",
        encode=lambda text: [9307] if text == "cat" else [],
        config=GEMMA4_LTX_12B_CONFIG,
    )
    assert len(tokens.ids) == 1024
    assert tokens.ids[-2:] == (2, 9307)
    assert tokens.attention_mask[-2:] == (1, 1)
    assert not any(tokens.attention_mask[:-2])
    assert 1 not in tokens.ids
