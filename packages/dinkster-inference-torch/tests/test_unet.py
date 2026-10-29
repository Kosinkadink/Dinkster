"""Stage 5 slice 5: the native SD1/SDXL diffusion UNet.

Every golden in goldens/unet_goldens.json was produced by RUNNING the
reference UNetModel
(comfy/ldm/modules/diffusionmodules/openaimodel.py @ the audited
baseline, tools/gen_unet_goldens.py). Weights come from the shared
deterministic hash (unet_fill.py) and inputs from its ``hashed_input``
namespace, so both sides run bit-identical parameters and activations
without storing megabytes.

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
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import (
    SD15_INPAINT_UNET_CONFIG,
    SD15_UNET_CONFIG,
    SDXL_INPAINT_UNET_CONFIG,
    SDXL_REFINER_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    UNetConfig,
    unet_layout,
)
from dinkster_inference_torch import (
    SDControlResiduals,
    UNetModel,
    select_attention,
    timestep_embedding,
)
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "unet_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> UNetConfig:
    spec = dict(GOLDENS["cases"][case]["config"])
    for field in (
        "num_res_blocks",
        "channel_mult",
        "transformer_depth",
        "transformer_depth_output",
    ):
        spec[field] = tuple(spec[field])
    return UNetConfig(**spec)


def build_model(case: str) -> UNetModel:
    model = UNetModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The generator's exact inputs, rebuilt from the shared hash."""
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    batch = spec["batch"]
    x = hashed_input(
        f"{case}:x",
        (batch, config["in_channels"], spec["height"], spec["width"]),
    )
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context",
        (batch, spec["context_len"], config["context_dim"]),
    )
    y = None
    if config["adm_in_channels"] is not None:
        y = hashed_input(f"{case}:y", (batch, config["adm_in_channels"]))
    return x, timesteps, context, y


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in unet_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("sd15", SD15_UNET_CONFIG),
        ("sd15_inpaint", SD15_INPAINT_UNET_CONFIG),
        ("sdxl", SDXL_UNET_CONFIG),
        ("sdxl_inpaint", SDXL_INPAINT_UNET_CONFIG),
        ("sdxl_refiner", SDXL_REFINER_UNET_CONFIG),
    ],
)
def test_full_size_module_matches_reference_layout(name: str, config: UNetConfig) -> None:
    """The real SD1.5/SDXL/refiner architectures, constructed on the
    meta device (initless factories never touch the storage), against
    the reference model's own full-size listing."""
    with torch.device("meta"):
        model = UNetModel(config)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_matches_executed_reference(case: str) -> None:
    x, timesteps, context, y = case_inputs(case)
    got = build_model(case)(x, timesteps, context, y)
    torch.testing.assert_close(got, dec(GOLDENS["cases"][case]["output"]), rtol=1e-4, atol=1e-5)


def test_odd_spatial_output_matches_input_extents() -> None:
    """The Upsample output_shape override: odd extents shrink through
    the strided downsample and must round-trip exactly - executed on
    the Dinkster model, not just read from the golden record."""
    spec = GOLDENS["cases"]["sd1_odd_spatial"]
    assert (spec["height"], spec["width"]) == (15, 10)
    x, timesteps, context, y = case_inputs("sd1_odd_spatial")
    assert y is None
    got = build_model("sd1_odd_spatial")(x, timesteps, context)
    assert tuple(got.shape[2:]) == (15, 10)


def test_control_residuals_apply_to_reversed_skips_and_middle() -> None:
    model = build_model("sd1_conv")
    x, timesteps, context, y = case_inputs("sd1_conv")
    assert y is None
    skips: list[torch.Tensor] = []
    middle: list[torch.Tensor] = []
    baseline_inputs: list[torch.Tensor] = []
    handles = [
        module.register_forward_hook(lambda _module, _args, output: skips.append(output.detach()))
        for module in model.input_blocks
    ]
    handles.append(
        model.middle_block.register_forward_hook(
            lambda _module, _args, output: middle.append(output.detach())
        )
    )
    handles.extend(
        module.register_forward_pre_hook(
            lambda _module, args: baseline_inputs.append(args[0].detach())
        )
        for module in model.output_blocks
    )
    try:
        model(x, timesteps, context)
    finally:
        for handle in handles:
            handle.remove()

    down = tuple(torch.full_like(value, float(index + 1)) for index, value in enumerate(skips))
    middle_control = torch.full_like(middle[0], 20.0)
    control = cast(
        SDControlResiduals,
        SimpleNamespace(down=down, middle=middle_control),
    )
    controlled_inputs: list[torch.Tensor] = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args: controlled_inputs.append(args[0].detach())
        )
        for module in model.output_blocks
    ]
    try:
        model(x, timesteps, context, control=control)
    finally:
        for handle in handles:
            handle.remove()

    assert len(controlled_inputs) == len(baseline_inputs) == len(down)
    middle_channels = middle[0].shape[1]
    torch.testing.assert_close(
        controlled_inputs[0][:, :middle_channels] - baseline_inputs[0][:, :middle_channels],
        middle_control,
    )
    for output_index, (controlled, baseline) in enumerate(
        zip(controlled_inputs, baseline_inputs, strict=True)
    ):
        residual = down[-1 - output_index]
        channels = residual.shape[1]
        torch.testing.assert_close(controlled[:, -channels:] - baseline[:, -channels:], residual)


# ------------------------------------------------------ embeddings


def test_timestep_embedding_is_cos_then_sin_float32() -> None:
    """The reference concatenates cos before sin; t=0 pins the halves
    (cos 0 = 1, sin 0 = 0) so a swapped order cannot pass."""
    emb = timestep_embedding(torch.tensor([0.0, 999.0]), 64)
    assert emb.shape == (2, 64)
    assert emb.dtype == torch.float32
    torch.testing.assert_close(emb[0, :32], torch.ones(32))
    torch.testing.assert_close(emb[0, 32:], torch.zeros(32))
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(32, dtype=torch.float32) / 32
    )
    torch.testing.assert_close(emb[1, :32], torch.cos(999.0 * freqs))
    torch.testing.assert_close(emb[1, 32:], torch.sin(999.0 * freqs))


def test_timestep_embedding_pads_odd_dim_with_zero() -> None:
    """The reference appends one zero column when dim is odd."""
    emb = timestep_embedding(torch.tensor([0.0, 999.0]), 65)
    assert emb.shape == (2, 65)
    assert emb.dtype == torch.float32
    torch.testing.assert_close(emb[:, 64], torch.zeros(2))
    torch.testing.assert_close(emb[:, :64], timestep_embedding(torch.tensor([0.0, 999.0]), 64))


# ------------------------------------------------- batched attention


def test_unet_injects_one_kernel_without_changing_state_or_output() -> None:
    x, timesteps, context, y = case_inputs("xl_linear_adm")
    baseline = build_model("xl_linear_adm")
    spy = CallableModuleKernel(select_attention("unet").kernel)
    model = UNetModel(case_config("xl_linear_adm"), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model(x, timesteps, context, y), baseline(x, timesteps, context, y))
    assert spy.calls
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)


# ------------------------------------------------------ ADM gating


def test_adm_model_requires_y() -> None:
    x, timesteps, context, _ = case_inputs("xl_linear_adm")
    with pytest.raises(ValueError, match="ADM-conditioned"):
        build_model("xl_linear_adm")(x, timesteps, context, None)


def test_plain_model_rejects_y() -> None:
    x, timesteps, context, _ = case_inputs("sd1_conv")
    with pytest.raises(ValueError, match="ADM-conditioned"):
        build_model("sd1_conv")(x, timesteps, context, torch.zeros(2, 8))


def test_adm_model_rejects_batch_mismatched_y() -> None:
    """The reference asserts y.shape[0] == x.shape[0]; a silent
    broadcast would condition every sample on one embedding."""
    x, timesteps, context, y = case_inputs("xl_linear_adm")
    assert y is not None and x.shape[0] == 2
    with pytest.raises(ValueError, match="does not match x batch"):
        build_model("xl_linear_adm")(x, timesteps, context, y[:1])


# -------------------------------------------------------- autograd


def test_gradients_flow_to_every_parameter() -> None:
    """No inference_mode/no_grad anywhere in the module (training
    program): the loss must reach EVERY parameter - conv, linear,
    norm, attention projection, and embedding MLP alike (dropout is
    0, so nothing is stochastically dropped)."""
    model = build_model("xl_linear_adm")
    x, timesteps, context, y = case_inputs("xl_linear_adm")
    model(x, timesteps, context, y).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
