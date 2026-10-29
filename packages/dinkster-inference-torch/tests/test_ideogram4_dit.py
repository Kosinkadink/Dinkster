"""Ideogram 4 DiT and scheduler parity with executed ComfyUI goldens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import IDEOGRAM4_CONFIG, ideogram4_layout
from dinkster_inference_torch import Ideogram4DiT, ideogram4_sigmas
from dinkster_inference_torch.ideogram4_dit import (
    _rope_matrix,  # pyright: ignore[reportPrivateUsage]
)
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "ideogram4_goldens.json",
    allow_portable_fallback=True,
)


@dataclass(frozen=True)
class SmallConfig:
    hidden_size: int
    layers: int
    attention_heads: int
    attention_head_dim: int
    intermediate_size: int
    adaln_dim: int
    latent_channels: int
    ae_channels: int
    patch: tuple[int, int]
    text_width: int
    rope_theta: float
    rope_dims: tuple[int, int, int]
    norm_eps: float


def decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def case_config(case: str) -> SmallConfig:
    config = GOLDENS["cases"][case]["config"]
    patch = 2
    return SmallConfig(
        hidden_size=config["num_attention_heads"] * config["attention_head_dim"],
        layers=config["num_layers"],
        attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        intermediate_size=config["intermediate_size"],
        adaln_dim=config["adaln_dim"],
        latent_channels=config["in_channels"],
        ae_channels=config["in_channels"] // (patch * patch),
        patch=(patch, patch),
        text_width=config["llm_features_dim"],
        rope_theta=float(config["rope_theta"]),
        rope_dims=tuple(config["mrope_section"]),
        norm_eps=float(config["norm_eps"]),
    )


def build_model(case: str) -> Ideogram4DiT:
    spec = GOLDENS["cases"][case]
    model = Ideogram4DiT(case_config(case))
    model.load_state_dict(fill_state_dict(spec["state_dict"]), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    latent = hashed_input(
        f"{case}:latent",
        (spec["batch"], config["in_channels"], spec["height"], spec["width"]),
    )
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    if spec["image_only"]:
        return latent, timesteps, None, None
    context = hashed_input(
        f"{case}:context",
        (spec["batch"], spec["text_length"], config["llm_features_dim"]),
    )
    mask = None
    if spec["padded"]:
        mask = torch.tensor(((1, 1, 1, 0, 0), (1, 1, 1, 1, 1)), dtype=torch.long)
    return latent, timesteps, context, mask


def test_full_model_and_predicted_layout_match_reference() -> None:
    with torch.device("meta"):
        model = Ideogram4DiT(IDEOGRAM4_CONFIG)
    actual = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert actual == [(key, list(shape)) for key, shape in GOLDENS["layout"]]
    assert actual == sorted((key, list(shape)) for key, shape in ideogram4_layout().items())


def test_full_profile_rope_keeps_unspecialized_dimensions_temporal() -> None:
    positions = torch.tensor([[[7, 11, 13], [17, 19, 23]]])
    rope = _rope_matrix(
        positions,
        head_dim=IDEOGRAM4_CONFIG.attention_head_dim,
        theta=IDEOGRAM4_CONFIG.rope_theta,
        rope_dims=IDEOGRAM4_CONFIG.rope_dims,
    )
    assert rope.shape == (1, 2, 1, 128, 2, 2)
    assert bool(torch.isfinite(rope).all())


@pytest.mark.parametrize("case", sorted(GOLDENS["cases"]))
def test_forward_and_intermediates_match_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}
    hooks = (
        model.layers[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(block0=cast(torch.Tensor, output))
        ),
        model.final_layer.register_forward_hook(
            lambda _module, _inputs, output: observed.update(final=cast(torch.Tensor, output))
        ),
    )
    try:
        output = model(*case_inputs(case))
    finally:
        for hook in hooks:
            hook.remove()
    spec = GOLDENS["cases"][case]
    assert_reference_tensor(observed["block0"], decode(spec["block0"]), rtol=1e-5, atol=1e-6)
    assert_reference_tensor(observed["final"], decode(spec["final"]), rtol=1e-5, atol=1e-6)
    assert_reference_tensor(output, decode(spec["output"]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "steps", "width", "height", "mu", "std"),
    (
        ("quality", 48, 1024, 1024, 0.0, 1.5),
        ("default", 20, 1024, 1024, 0.0, 1.75),
        ("maintained_template", 20, 1024, 1024, 0.5, 1.75),
        ("turbo", 12, 1024, 1024, 0.5, 1.75),
    ),
)
def test_scheduler_is_bit_exact_to_reference(
    name: str, steps: int, width: int, height: int, mu: float, std: float
) -> None:
    actual = ideogram4_sigmas(steps, width, height, mu, std)
    expected = torch.tensor(GOLDENS["schedules"][name], dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert actual.shape == (steps + 1,)
    assert actual[-1].item() == 0.0
    assert bool(torch.all(actual[:-1] > actual[1:]))


def test_forward_rejects_mismatched_boundaries() -> None:
    model = build_model("conditional")
    latent, timesteps, context, _ = case_inputs("conditional")
    assert context is not None
    with pytest.raises(ValueError, match="latent must have shape"):
        model(latent[:, :-1], timesteps, context)
    with pytest.raises(ValueError, match="timesteps must match"):
        model(latent, timesteps[:1], context)
    with pytest.raises(ValueError, match="context must have shape"):
        model(latent, timesteps, context[..., :-1])
    with pytest.raises(ValueError, match="attention mask must match"):
        model(latent, timesteps, context, torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="image-only"):
        model(latent, timesteps, None, torch.ones(2, 5, dtype=torch.long))
