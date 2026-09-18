"""The native LTX-Video 2B transformer against the executed reference.

Every golden in goldens/ltx_model_goldens.json was produced by
RUNNING the reference LTXVModel (comfy/ldm/lightricks/model.py @ the
audited baseline, tools/gen_ltx_model_goldens.py) with attention
forced to pytorch SDPA. Weights come from the shared deterministic
hash (unet_fill.py - the only rank-1 ``.weight`` keys are the
attention RMS q/k scales, so the rank rule holds) and inputs from its
``hashed_input`` namespace.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from dinkster_inference import (
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V095_CONFIG,
    LTXVConfig,
    ltxv_layout,
)
from dinkster_inference_torch import LTXVModel
from dinkster_inference_torch import ltx_model as ltx_model_module
from dinkster_inference_torch.ltx_media import LTXVGuideConditioning
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "ltx_model_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])

FULL_CONFIGS = {
    "ltxv_2b_v09": LTXV_2B_V09_CONFIG,
    "ltxv_2b_v095": LTXV_2B_V095_CONFIG,
}


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> LTXVConfig:
    spec = GOLDENS["cases"][case]["config"]
    return LTXVConfig(
        in_channels=spec["in_channels"],
        cross_attention_dim=spec["cross_attention_dim"],
        attention_head_dim=spec["attention_head_dim"],
        num_attention_heads=spec["num_attention_heads"],
        caption_channels=spec["caption_channels"],
        num_layers=spec["num_layers"],
        causal_temporal_positioning=spec["causal_temporal_positioning"],
    )


def build_model(case: str) -> LTXVModel:
    model = LTXVModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
    torch.Tensor | None,
    tuple[LTXVGuideConditioning, ...],
]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    x = hashed_input(
        f"{case}:x",
        (spec["batch"], config["in_channels"], spec["frames"], spec["height"], spec["width"]),
    )
    timestep = torch.tensor(spec["timestep"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context",
        (spec["batch"], spec["context_len"], config["caption_channels"]),
    )
    attention_mask = None
    if spec["attention_mask"] is not None:
        attention_mask = torch.tensor(spec["attention_mask"], dtype=torch.int64)
    denoise_mask = None
    if spec.get("denoise_mask") is not None:
        denoise_mask = torch.tensor(spec["denoise_mask"], dtype=torch.float32)

    def guide_attention_mask(value: dict[str, Any] | None) -> torch.Tensor | None:
        if value is None:
            return None
        assert value["kind"] == "linear"
        return torch.linspace(0.0, 1.0, math.prod(value["shape"]), dtype=torch.float32).reshape(
            value["shape"]
        )

    guides = tuple(
        LTXVGuideConditioning(
            torch.tensor(guide["keyframe_indices"], dtype=torch.int64),
            tuple(guide["latent_shape"]),
            guide["strength"],
            guide_attention_mask(guide["attention_mask"]),
        )
        for guide in spec.get("guides", ())
    )
    return x, timestep, context, attention_mask, spec["frame_rate"], denoise_mask, guides


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in ltxv_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


@pytest.mark.parametrize("name", sorted(FULL_CONFIGS))
def test_full_size_module_matches_reference_layout(name: str) -> None:
    """The real 2B v0.9/v0.9.5 architectures, constructed on the meta
    device (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = LTXVModel(FULL_CONFIGS[name])
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


@pytest.mark.parametrize("name", sorted(FULL_CONFIGS))
def test_torch_free_layout_predicts_the_full_size_reference(name: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in ltxv_layout(FULL_CONFIGS[name]).items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert predicted == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_and_block_match_executed_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}
    hook = model.transformer_blocks[0].register_forward_hook(
        lambda _module, _inputs, output: observed.update(block0=cast(torch.Tensor, output))
    )
    x, timestep, context, attention_mask, frame_rate, denoise_mask, guides = case_inputs(case)
    try:
        with torch.no_grad():
            output = model(
                x,
                timestep,
                context,
                attention_mask=attention_mask,
                frame_rate=frame_rate,
                denoise_mask=denoise_mask,
                guides=guides,
            )
    finally:
        hook.remove()
    golden = GOLDENS["cases"][case]
    torch.testing.assert_close(
        observed["block0"],
        dec(golden["block_outputs"]["block0"]),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(output, dec(golden["output"]), rtol=1e-4, atol=1e-5)


def test_rope_and_rms_adaln_use_kitchen_without_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = torch.randn(1, 3, 4)
    matrix = ltx_model_module._rope_matrix(  # pyright: ignore[reportPrivateUsage]
        frequencies, 0, False, 2, torch.float32
    )
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    calls: list[str] = []

    def apply_rope(
        query: torch.Tensor, key: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("rope")
        return (
            ltx_model_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
                query, table, False
            ),
            ltx_model_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
                key, table, False
            ),
        )

    def rms_adaln(value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        calls.append("rms_adaln")
        return ltx_model_module._rms(value) * (1 + scale) + shift  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(comfy_kitchen, "apply_rope", apply_rope)
    monkeypatch.setattr(comfy_kitchen, "rms_adaln", rms_adaln)
    with torch.no_grad():
        observed_q, observed_k = ltx_model_module._apply_rope_qk(  # pyright: ignore[reportPrivateUsage]
            q, k, (matrix, False)
        )
        observed_norm = ltx_model_module._rms_adaln(  # pyright: ignore[reportPrivateUsage]
            q, torch.zeros_like(q), torch.zeros_like(q)
        )
    assert calls == ["rope", "rms_adaln"]
    torch.testing.assert_close(
        observed_q,
        ltx_model_module._apply_rope_torch(q, matrix, False),  # pyright: ignore[reportPrivateUsage]
    )
    torch.testing.assert_close(
        observed_k,
        ltx_model_module._apply_rope_torch(k, matrix, False),  # pyright: ignore[reportPrivateUsage]
    )
    torch.testing.assert_close(observed_norm, ltx_model_module._rms(q))  # pyright: ignore[reportPrivateUsage]


def test_rope_and_rms_adaln_keep_autograd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object) -> None:
        raise AssertionError("kitchen operation must not run under autograd")

    monkeypatch.setattr(comfy_kitchen, "apply_rope", refuse)
    monkeypatch.setattr(comfy_kitchen, "rms_adaln", refuse)
    matrix = ltx_model_module._rope_matrix(  # pyright: ignore[reportPrivateUsage]
        torch.randn(1, 3, 4), 0, False, 2, torch.float32
    )
    q = torch.randn(1, 3, 8, requires_grad=True)
    k = torch.randn(1, 3, 8, requires_grad=True)
    observed_q, observed_k = ltx_model_module._apply_rope_qk(  # pyright: ignore[reportPrivateUsage]
        q, k, (matrix, False)
    )
    observed_norm = ltx_model_module._rms_adaln(  # pyright: ignore[reportPrivateUsage]
        q, torch.zeros_like(q), torch.zeros_like(q)
    )
    (observed_q.sum() + observed_k.sum() + observed_norm.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None


@pytest.mark.parametrize(
    "storage_dtype,compute_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_direct_table_cast_preserves_aimdo_raw_storage_allocation(
    storage_dtype: torch.dtype, compute_dtype: torch.dtype
) -> None:
    assert_direct_table_cast_preserves_aimdo_allocation(storage_dtype, compute_dtype, "cpu")


def assert_direct_table_cast_preserves_aimdo_allocation(
    storage_dtype: torch.dtype, compute_dtype: torch.dtype, device: str
) -> None:
    from functools import partial

    from dinkster_inference_torch import INITLESS, AimdoWeights, ComfyAimdoBackend, enroll_component
    from dinkster_inference_torch.attention import select_attention
    from test_aimdo_residency import FakeVbarBackend

    with torch.device("meta"):
        block = ltx_model_module.LTXTransformerBlock(
            2048,
            32,
            64,
            2048,
            8192,
            operations=INITLESS,
            attention_kernel=select_attention("flux").kernel,
        )
    for name, _child in tuple(block.named_children()):
        delattr(block, name)
    stored = torch.arange(6 * 2048).reshape(6, 2048).to(storage_dtype)
    block.scale_shift_table = torch.nn.Parameter(stored)
    assert stored.numel() * stored.element_size() > 16 * 1024
    backend = FakeVbarBackend() if device == "cpu" else ComfyAimdoBackend()
    mechanism = enroll_component(
        block,
        load_device="cuda:0",
        offload_device="cpu",
        mechanism_factory=partial(AimdoWeights, backend=backend, stream_count=0),
    )
    binding = block.residency_binding()
    assert binding is not None
    try:
        mechanism.partially_load(0)
        assert not binding.unit_state.loaded
        with block.materialized_state(
            "scale_shift_table", device=torch.device(device), dtype=compute_dtype
        ) as actual:
            assert actual.dtype == compute_dtype
            torch.testing.assert_close(
                actual, stored.to(device=device, dtype=compute_dtype), rtol=0, atol=0
            )
        if isinstance(backend, FakeVbarBackend):
            assert backend.fault_calls
    finally:
        mechanism.unload()
