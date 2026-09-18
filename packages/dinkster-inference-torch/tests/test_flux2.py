"""The native Flux2 diffusion transformer against the executed reference.

Every golden in goldens/flux2_goldens.json was produced by RUNNING
the reference Flux in its Flux2 configuration (comfy/ldm/flux/model.py
with global_modulation / mlp_silu_act / ops_bias=False @ the audited
baseline, tools/gen_flux2_goldens.py) with attention forced to
pytorch SDPA and RoPE to the reference's pure-torch path. Weights
come from the shared deterministic hash (unet_fill.py - Flux2's only
rank-1 weights are the QKNorm RMS scales, so the rank rule holds) and
inputs from its ``hashed_input`` namespace.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B_CONFIG,
    FluxConfig,
    flux2_layout,
    normalize_flux_keys,
)
from dinkster_inference_torch import Flux
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "flux2_goldens.json")

CASES = sorted(GOLDENS["cases"])

FULL_CONFIGS = {
    "flux2_dev": FLUX2_DEV_CONFIG,
    "flux2_klein_9b": FLUX2_KLEIN_9B_CONFIG,
    "flux2_klein_4b": FLUX2_KLEIN_4B_CONFIG,
}


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> FluxConfig:
    spec = dict(GOLDENS["cases"][case]["config"])
    spec["axes_dim"] = tuple(spec["axes_dim"])
    spec["txt_ids_dims"] = tuple(spec["txt_ids_dims"])
    return FluxConfig(**spec)


def build_model(case: str) -> Flux:
    model = Flux(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
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
        (batch, spec["context_len"], config["context_in_dim"]),
    )
    guidance = None
    if spec["guidance"] is not None:
        guidance = torch.tensor(spec["guidance"], dtype=torch.float32)
    return x, timesteps, context, None, guidance


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in flux2_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


@pytest.mark.parametrize("name", sorted(FULL_CONFIGS))
def test_full_size_module_matches_reference_layout(name: str) -> None:
    """The real dev/Klein architectures, constructed on the meta
    device (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = Flux(FULL_CONFIGS[name])
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


def test_flux2_owns_no_per_block_modulation() -> None:
    with torch.device("meta"):
        model = Flux(FLUX2_KLEIN_4B_CONFIG)
    assert model.double_stream_modulation_img is not None
    assert model.double_stream_modulation_txt is not None
    assert model.single_stream_modulation is not None
    assert model.vector_in is None
    for block in model.double_blocks:
        assert block.img_mod is None
        assert block.txt_mod is None
    for block in model.single_blocks:
        assert block.modulation is None


def test_bare_bfl_scale_spelling_loads_after_normalization() -> None:
    """Flux2 exports spell the RMSNorm scales ``*_norm.scale``; the
    renamed mapping must strict-load."""
    tensors = fill_state_dict(golden_entries("flux2_no_guidance"))
    bare = {
        (
            key[: -len(".weight")] + ".scale"
            if key.endswith(("query_norm.weight", "key_norm.weight"))
            else key
        ): value
        for key, value in tensors.items()
    }
    assert any(key.endswith(".scale") for key in bare)
    model = Flux(case_config("flux2_no_guidance"))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(bare, strict=True)
    model.load_state_dict(normalize_flux_keys(bare), strict=True)


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_and_blocks_match_executed_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}
    hooks = [
        model.double_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(
                double_img=output[0], double_txt=output[1]
            )
        ),
        model.single_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(single=cast(torch.Tensor, output))
        ),
    ]
    try:
        output = model(*case_inputs(case))
    finally:
        for hook in hooks:
            hook.remove()
    golden = GOLDENS["cases"][case]
    for name in ("double_img", "double_txt", "single"):
        assert_reference_tensor(
            observed[name],
            dec(golden["block_outputs"][name]),
            rtol=1e-4,
            atol=1e-5,
        )
    assert_reference_tensor(output, dec(golden["output"]), rtol=1e-4, atol=1e-5)


# ------------------------------------------------- forward contract


def test_distilled_guidance_may_be_omitted_and_plain_model_rejects_it() -> None:
    x, timesteps, context, _, guidance = case_inputs("flux2_guidance")
    output = build_model("flux2_guidance")(x, timesteps, context, None, None)
    assert output.shape == x.shape
    x, timesteps, context, _, _ = case_inputs("flux2_no_guidance")
    with pytest.raises(ValueError, match="no guidance embedder"):
        build_model("flux2_no_guidance")(x, timesteps, context, None, guidance)


def test_declared_four_axis_position_ids_match_the_derived_grid() -> None:
    """image_position_ids generalizes to Flux2's four axes: the
    derived local grid declared explicitly is bit-identical, and a
    nonzero non-spatial axis refuses."""
    model = build_model("flux2_no_guidance")
    x, timesteps, context, _, _ = case_inputs("flux2_no_guidance")
    baseline = model(x, timesteps, context)
    _, ids = model._patchify(x)  # pyright: ignore[reportPrivateUsage]
    declared = model(x, timesteps, context, image_position_ids=ids.contiguous())
    assert torch.equal(baseline, declared)
    bad = ids.contiguous()
    bad[..., 3] = 1.0
    with pytest.raises(ValueError, match="non-spatial axes must be zero"):
        model(x, timesteps, context, image_position_ids=bad)
