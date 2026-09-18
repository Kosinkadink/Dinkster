"""The native Krea 2 diffusion transformer against the executed
reference.

Every golden in goldens/krea2_dit_goldens.json was produced by
RUNNING the reference Krea 2 SingleStreamDiT (comfy/ldm/krea2/model.py
@ the audited baseline, tools/gen_krea2_dit_goldens.py) with attention
forced to pytorch SDPA and RoPE to the reference's pure-torch path.
Weights come from the shared deterministic hash (unet_fill.py - Krea 2
spells its RMS scales ``.scale``, so the rank-1 ``.weight`` rule holds
trivially) and inputs from its ``hashed_input`` namespace.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import KREA2_CONFIG, krea2_layout
from dinkster_inference_torch import Krea2DiT
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "krea2_dit_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


@dataclass(frozen=True)
class SmallConfig:
    features: int
    transformer_blocks: int
    attention_heads: int
    kv_heads: int
    mlp_width: int
    time_width: int
    text_width: int
    text_layers: int
    text_heads: int
    text_kv_heads: int
    text_mlp_width: int
    text_fusion_layerwise_blocks: int
    text_fusion_refiner_blocks: int
    latent_channels: int
    patch: tuple[int, int]
    rope_axes: tuple[int, int, int]
    rope_theta: float
    rms_norm_eps: float


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> SmallConfig:
    spec = GOLDENS["cases"][case]["config"]

    def swiglu(width: int) -> int:
        raw = int(2 * width / 3) * spec["multiplier"]
        return -(-raw // 128) * 128

    head_dim = spec["features"] // spec["heads"]
    sixteenth = head_dim // 16
    return SmallConfig(
        features=spec["features"],
        transformer_blocks=spec["layers"],
        attention_heads=spec["heads"],
        kv_heads=spec["kvheads"],
        mlp_width=swiglu(spec["features"]),
        time_width=spec["tdim"],
        text_width=spec["txtdim"],
        text_layers=spec["txtlayers"],
        text_heads=spec["txtheads"],
        text_kv_heads=spec["txtkvheads"],
        text_mlp_width=swiglu(spec["txtdim"]),
        text_fusion_layerwise_blocks=2,
        text_fusion_refiner_blocks=2,
        latent_channels=spec["channels"],
        patch=(spec["patch"], spec["patch"]),
        rope_axes=(head_dim - 12 * sixteenth, 6 * sixteenth, 6 * sixteenth),
        rope_theta=float(spec["theta"]),
        rms_norm_eps=1e-5,
    )


def build_model(case: str) -> Krea2DiT:
    model = Krea2DiT(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    batch = spec["batch"]
    frames = spec["frames"]
    if frames is None:
        x = hashed_input(f"{case}:x", (batch, config["channels"], spec["height"], spec["width"]))
        context_batch = batch
    else:
        x = hashed_input(
            f"{case}:x", (batch, config["channels"], frames, spec["height"], spec["width"])
        )
        context_batch = batch * frames
    fused = config["txtlayers"] * config["txtdim"]
    context = hashed_input(f"{case}:context", (context_batch, spec["context_len"], fused))
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    return x, timesteps, context


def case_refs(case: str) -> list[torch.Tensor]:
    spec = GOLDENS["cases"][case]
    channels = spec["config"]["channels"]
    return [
        hashed_input(f"{case}:ref{index}", (1, channels, height, width))
        for index, (height, width) in enumerate(spec["ref_shapes"])
    ]


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


def test_full_size_module_matches_reference_layout() -> None:
    """The real Krea 2 architecture, constructed on the meta device
    (initless factories never touch the storage), against the
    reference model's own full-size listing and the torch-free
    layout."""
    with torch.device("meta"):
        model = Krea2DiT(KREA2_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["krea2"]]
    assert ours == golden
    predicted = sorted((key, list(shape)) for key, shape in krea2_layout().items())
    assert ours == predicted


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_and_blocks_match_executed_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}
    hooks = [
        model.txtfusion.register_forward_hook(
            lambda _module, _inputs, output: observed.update(txtfusion=cast(torch.Tensor, output))
        ),
        model.blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(block0=cast(torch.Tensor, output))
        ),
    ]
    spec = GOLDENS["cases"][case]
    x, timesteps, context = case_inputs(case)
    refs = case_refs(case)
    try:
        output = model(
            x,
            timesteps,
            context,
            ref_latents=refs if refs else None,
            ref_latents_method=spec["ref_method"],
        )
    finally:
        for hook in hooks:
            hook.remove()
    for name in ("txtfusion", "block0"):
        torch.testing.assert_close(
            observed[name],
            dec(spec["block_outputs"][name]),
            rtol=1e-4,
            atol=1e-5,
        )
    torch.testing.assert_close(output, dec(spec["output"]), rtol=1e-4, atol=1e-5)


# ------------------------------------------------- forward contract


def test_rejects_unfused_context_widths() -> None:
    model = build_model("krea2_base")
    x, timesteps, context = case_inputs("krea2_base")
    with pytest.raises(ValueError, match="stacked text-fusion taps"):
        model(x, timesteps, context[..., :-1])


def test_rejects_unknown_reference_latent_methods() -> None:
    model = build_model("krea2_ref_index")
    x, timesteps, context = case_inputs("krea2_ref_index")
    with pytest.raises(ValueError, match="unknown reference-latent method"):
        model(
            x,
            timesteps,
            context,
            ref_latents=case_refs("krea2_ref_index"),
            ref_latents_method="offset",
        )


def test_rejects_mismatched_timestep_batches() -> None:
    model = build_model("krea2_base")
    x, timesteps, context = case_inputs("krea2_base")
    with pytest.raises(ValueError, match="timesteps must be"):
        model(x, timesteps[:1], context)


def test_reference_latents_without_method_join_at_the_denoising_timestep() -> None:
    """A ref method of None ignores the reference latents entirely
    (the reference gates the whole block on the method)."""
    model = build_model("krea2_ref_index")
    x, timesteps, context = case_inputs("krea2_ref_index")
    baseline = model(x, timesteps, context)
    ignored = model(x, timesteps, context, ref_latents=case_refs("krea2_ref_index"))
    assert torch.equal(baseline, ignored)
