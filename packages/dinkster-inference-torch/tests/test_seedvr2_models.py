"""Native SeedVR2 model math against the executed ComfyUI reference.

``tools/gen_seedvr2_goldens.py`` runs ComfyUI at the pinned commit with
deterministic name-hashed weights. The cases cover all three published DiT
architectures plus image and video execution through the causal VAE.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import (
    SEEDVR2_3B,
    SEEDVR2_7B,
    SEEDVR2_7B_MLP,
    SeedVR2Config,
    seedvr2_layout,
    seedvr2_vae_layout,
)
from dinkster_inference_torch import INITLESS, NaDiT, VideoAutoencoderKLWrapper
from dinkster_inference_torch import seedvr2_dit as seedvr2_dit_module
from dinkster_inference_torch.module_residency import enroll_component
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_value, hashed_input

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "seedvr2_goldens.json",
    allow_portable_fallback=True,
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def fill_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for key, value in module.state_dict().items():
            value.copy_(fill_value(key, value.shape))


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def build_dit(case: str) -> NaDiT:
    spec = GOLDENS["cases"][case]
    published_7b_width = seedvr2_dit_module.SEEDVR2_7B_VID_DIM
    seedvr2_dit_module.SEEDVR2_7B_VID_DIM = 32 if spec["seven_b"] else published_7b_width
    try:
        model = NaDiT(
            operations=INITLESS,
            attention_kernel=sdpa,
            **spec["config"],
        )
    finally:
        seedvr2_dit_module.SEEDVR2_7B_VID_DIM = published_7b_width
    fill_module(model)
    return model


def dit_inputs(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GOLDENS["cases"][case]
    batch = spec["batch"]
    frames = spec["frames"]
    return (
        hashed_input(f"{case}:latent", (batch, 16, frames, 4, 4)),
        torch.tensor(spec["timestep"], dtype=torch.float32),
        hashed_input(f"{case}:context", (batch, 3, 24)),
        hashed_input(f"{case}:condition", (batch, 17, frames, 4, 4)),
    )


def test_rotary_frequencies_enroll_and_forward_while_offloaded() -> None:
    container = torch.nn.Module()
    rotary = seedvr2_dit_module.RotaryEmbedding(dim=12, freqs_for="pixel")
    container.add_module("rotary", rotary)
    positions = torch.linspace(-1.0, 1.0, 7)
    resident = rotary(positions)

    mechanism = enroll_component(container, load_device="cpu", offload_device="cpu")
    mechanism.unload()

    prefetch = rotary.residency_prefetch()
    assert prefetch is not None
    assert prefetch[1] == (("rotary.freqs", torch.float32),)
    assert torch.equal(rotary(positions), resident)


@pytest.mark.parametrize("case", sorted(GOLDENS["cases"]))
def test_dit_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    model = build_dit(case)
    observed: dict[str, torch.Tensor] = {}

    def record_block(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        observed["block0_vid"] = output[0]
        observed["block0_txt"] = output[1]

    hook = model.blocks[0].register_forward_hook(record_block)
    latent, timestep, context, condition = dit_inputs(case)
    try:
        output = model(
            latent,
            timestep,
            context,
            condition=condition,
            transformer_options={"cond_or_uncond": spec["cond_or_uncond"]},
        )
    finally:
        hook.remove()
    for name in ("block0_vid", "block0_txt"):
        assert_reference_tensor(observed[name], dec(spec[name]), rtol=1e-4, atol=1e-5)
    assert_reference_tensor(output, dec(spec["output"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("config", "golden_name"),
    (
        (SEEDVR2_3B, "3b_swiglu"),
        (SEEDVR2_7B, "7b_swiglu"),
        (SEEDVR2_7B_MLP, "7b_mlp"),
    ),
)
def test_published_dit_layout_matches_reference(config: SeedVR2Config, golden_name: str) -> None:
    with torch.device("meta"):
        model = NaDiT(
            norm_eps=config.norm_eps,
            num_layers=config.layers,
            mlp_type=config.mlp_type,
            vid_dim=config.width,
            heads=config.heads,
            mm_layers=config.separate_layers,
            rope_type=config.rope_type,
            rope_dim=config.rope_dim,
            vid_out_norm="rms" if config.vid_out_norm else None,
        )
    observed = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    predicted = sorted((key, list(shape)) for key, shape in seedvr2_layout(config).items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][golden_name]]
    assert observed == predicted == golden


@pytest.fixture(scope="module")
def vae() -> Iterator[VideoAutoencoderKLWrapper]:
    model = VideoAutoencoderKLWrapper(attention_kernel=sdpa)
    fill_module(model)
    yield model


def test_published_vae_layout_matches_reference() -> None:
    with torch.device("meta"):
        model = VideoAutoencoderKLWrapper()
    observed = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    predicted = sorted((key, list(shape)) for key, shape in seedvr2_vae_layout().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["vae"]]
    assert observed == predicted == golden


@pytest.mark.parametrize(
    ("name", "window_fn"),
    (
        ("regular", seedvr2_dit_module.make_720Pwindows_bysize),
        ("shifted", seedvr2_dit_module.make_shifted_720Pwindows_bysize),
    ),
)
def test_window_plans_match_executed_reference(
    name: str,
    window_fn: Callable[
        [tuple[int, int, int], tuple[int, int, int]], list[tuple[slice, slice, slice]]
    ],
) -> None:
    spec = GOLDENS["windows"]
    observed = window_fn(tuple(spec["shape"]), tuple(spec["num_windows"]))
    encoded = [[[axis.start or 0, axis.stop or 0] for axis in window] for window in observed]
    assert encoded == spec[name]


@pytest.mark.parametrize("case", ("image", "video"))
def test_vae_matches_executed_reference(vae: VideoAutoencoderKLWrapper, case: str) -> None:
    spec = GOLDENS["vae"]["cases"][case]
    content = dec(spec["content"])
    with torch.no_grad():
        latent = vae.encode(content)
        decoded = vae.decode(latent)
    assert_reference_tensor(latent, dec(spec["latent"]), rtol=1e-4, atol=1e-5)
    assert_reference_tensor(decoded, dec(spec["decoded"]), rtol=1e-4, atol=1e-5)


def test_vae_spatial_tiling_matches_executed_reference(
    vae: VideoAutoencoderKLWrapper,
) -> None:
    spec = GOLDENS["vae"]["tiled"]
    with torch.no_grad():
        latent = vae.encode_tiled(dec(spec["content"]), tile_x=8, tile_y=8, overlap=0)
        decoded = vae.decode_tiled(latent, tile_x=1, tile_y=1, overlap=0)
    assert_reference_tensor(latent, dec(spec["latent"]), rtol=1e-4, atol=1e-5)
    assert_reference_tensor(decoded, dec(spec["decoded"]), rtol=1e-4, atol=1e-5)
