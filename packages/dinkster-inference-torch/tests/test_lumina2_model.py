"""Lumina Image 2.0 diffusion math against the executed ComfyUI reference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import LUMINA2_CONFIG, lumina2_layout
from dinkster_inference_torch import ZImage
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "lumina2_goldens.json").read_text())
CASES = sorted(GOLDENS["cases"])


@dataclass(frozen=True)
class SmallConfig:
    family_id: str
    hidden_width: int
    caption_width: int
    main_blocks: int
    noise_refiner_blocks: int
    context_refiner_blocks: int
    attention_heads: int
    kv_heads: int
    attention_head_dim: int
    ffn_width: int
    latent_channels: int
    patch: tuple[int, int]
    rope_axes: tuple[int, int, int]
    rope_theta: float
    qk_norm_eps: float
    timestep_embedding_width: int
    modulation_width: int
    timestep_multiplier: float
    block_modulation_silu: bool
    pad_tokens_multiple: int
    learned_padding: bool


def decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> SmallConfig:
    spec = GOLDENS["cases"][case]["config"]
    hidden = spec["dim"]
    ffn = int(spec["ffn_dim_multiplier"] * hidden)
    multiple = spec["multiple_of"]
    ffn = multiple * ((ffn + multiple - 1) // multiple)
    return SmallConfig(
        family_id="dinkster.lumina2",
        hidden_width=hidden,
        caption_width=spec["cap_feat_dim"],
        main_blocks=spec["n_layers"],
        noise_refiner_blocks=spec["n_refiner_layers"],
        context_refiner_blocks=spec["n_refiner_layers"],
        attention_heads=spec["n_heads"],
        kv_heads=spec["n_kv_heads"],
        attention_head_dim=hidden // spec["n_heads"],
        ffn_width=ffn,
        latent_channels=spec["in_channels"],
        patch=(spec["patch_size"], spec["patch_size"]),
        rope_axes=tuple(spec["axes_dims"]),
        rope_theta=spec["rope_theta"],
        qk_norm_eps=spec["norm_eps"],
        timestep_embedding_width=256,
        modulation_width=min(hidden, 1024),
        timestep_multiplier=spec["time_scale"],
        block_modulation_silu=True,
        pad_tokens_multiple=1,
        learned_padding=False,
    )


def build_model(case: str) -> ZImage:
    model = ZImage(cast("Any", case_config(case)))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    latent = hashed_input(
        f"{case}:latent",
        (spec["batch"], config["in_channels"], spec["height"], spec["width"]),
    )
    context = hashed_input(
        f"{case}:context",
        (spec["batch"], spec["context_tokens"], config["cap_feat_dim"]),
    )
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    return latent, timesteps, context


@pytest.mark.parametrize("case", CASES)
def test_tiny_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


def test_full_size_layout_matches_executed_reference_and_detector() -> None:
    with torch.device("meta"):
        model = ZImage(LUMINA2_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["lumina2"]]
    predicted = sorted(
        (key, list(shape)) for key, shape in lumina2_layout().items() if key != "norm_final.weight"
    )
    assert ours == golden == predicted
    assert len(ours) == 399


@pytest.mark.parametrize("case", CASES)
def test_forward_and_blocks_match_executed_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}

    def capture(name: str):  # noqa: ANN202
        def hook(_module: object, _inputs: object, output: object) -> None:
            observed[name] = cast("torch.Tensor", output)

        return hook

    hooks = [
        cast("torch.nn.Sequential", model.context_refiner)[0].register_forward_hook(
            capture("context")
        ),
        cast("torch.nn.Sequential", model.noise_refiner)[0].register_forward_hook(capture("noise")),
        cast("torch.nn.Sequential", model.layers)[0].register_forward_hook(capture("main")),
    ]
    try:
        output = model(*case_inputs(case))
    finally:
        for hook in hooks:
            hook.remove()
    golden = GOLDENS["cases"][case]
    for name in ("context", "noise", "main"):
        torch.testing.assert_close(
            observed[name], decode(golden["block_outputs"][name]), rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(output, decode(golden["output"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", CASES)
def test_fused_qk_rope_matches_differentiable_path(case: str) -> None:
    model = build_model(case)
    inputs = case_inputs(case)
    eager = model(*inputs)
    with torch.no_grad():
        fused = model(*inputs)
    torch.testing.assert_close(fused, eager.detach(), rtol=1e-5, atol=1e-6)


def test_lumina_and_z_image_keep_distinct_modulation_and_padding() -> None:
    model = build_model(CASES[0])
    assert model.config.family_id == "dinkster.lumina2"
    assert not hasattr(model, "cap_pad_token")
    assert not hasattr(model, "x_pad_token")
    first_layer = cast("Any", cast("torch.nn.Sequential", model.layers)[0])
    assert isinstance(first_layer.adaLN_modulation[0], torch.nn.SiLU)
    assert model.t_embedder.embedding_width == 256
