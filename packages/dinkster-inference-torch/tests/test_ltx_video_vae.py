"""The native LTX causal video VAE against the executed reference.

Every golden in goldens/ltx_vae_goldens.json was produced by RUNNING
the reference VideoVAE (comfy/ldm/lightricks/vae/
causal_video_autoencoder.py @ the audited baseline,
tools/gen_ltx_vae_goldens.py). Weights come from the shared
deterministic hash (ltx_vae_fill.py) and inputs from unet_fill's
``hashed_input`` namespace.

Encode and decode replay per streaming mode: the reference's own
streamed decode is not bit-equal to its single-shot decode, so each
mode is pinned to its own golden and the two are never compared to
each other. The timestep-conditioned case seeds torch's global RNG
with the recorded seed immediately before decoding, matching the
generator's draw order for the decode-noise mix and the inject-noise
draws.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    LTXAV_22B_V23_VAE_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    LTXV_2B_V095_VAE_CONFIG,
    LTXVAEBlock,
    LTXVideoVAEConfig,
    ltxv_vae_layout,
)
from dinkster_inference_torch import LTXVideoVAE, ltxv_vae_max_chunk_bytes
from ltx_vae_fill import fill_vae_state_dict
from unet_fill import hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "ltx_vae_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])
MODES = sorted(GOLDENS["budgets"])

FULL_CONFIGS = {
    "ltxav_22b_v23_vae": LTXAV_22B_V23_VAE_CONFIG,
    "ltxv_2b_v09_vae": LTXV_2B_V09_VAE_CONFIG,
    "ltxv_2b_v095_vae": LTXV_2B_V095_VAE_CONFIG,
}


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> LTXVideoVAEConfig:
    spec = GOLDENS["cases"][case]["config"]

    def blocks(specs: list[dict[str, Any]]) -> tuple[LTXVAEBlock, ...]:
        return tuple(
            LTXVAEBlock(
                kind=block["kind"],
                layers=block.get("layers", 1),
                multiplier=block.get("multiplier", 1),
                inject_noise=block.get("inject_noise", False),
                residual=block.get("residual", False),
            )
            for block in specs
        )

    return LTXVideoVAEConfig(
        encoder_blocks=blocks(spec["encoder_blocks"]),
        decoder_blocks=blocks(spec["decoder_blocks"]),
        latent_channels=spec["latent_channels"],
        base_channels=spec["base_channels"],
        patch_size=spec["patch_size"],
        timestep_conditioning=spec["timestep_conditioning"],
        encoder_spatial_padding_mode=spec.get("encoder_spatial_padding_mode", "zeros"),
        decoder_spatial_padding_mode=spec.get("decoder_spatial_padding_mode", "reflect"),
    )


def build_vae(case: str) -> LTXVideoVAE:
    vae = LTXVideoVAE(case_config(case))
    vae.load_state_dict(fill_vae_state_dict(golden_entries(case)), strict=True)
    return vae


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_vae(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted(
        (key, list(shape)) for key, shape in ltxv_vae_layout(case_config(case)).items()
    )
    assert predicted == golden_entries(case)


@pytest.mark.parametrize("name", sorted(FULL_CONFIGS))
def test_full_size_module_matches_reference_layout(name: str) -> None:
    """The real supported LTX VAE architectures, constructed on the
    meta device (initless factories never touch the storage), against
    the reference model's own full-size listing."""
    with torch.device("meta"):
        vae = LTXVideoVAE(FULL_CONFIGS[name])
    ours = sorted((key, list(value.shape)) for key, value in vae.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


@pytest.mark.parametrize("name", sorted(FULL_CONFIGS))
def test_torch_free_layout_predicts_the_full_size_reference(name: str) -> None:
    predicted = sorted(
        (key, list(shape)) for key, shape in ltxv_vae_layout(FULL_CONFIGS[name]).items()
    )
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert predicted == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_encode_matches_executed_reference(case: str, mode: str) -> None:
    spec = GOLDENS["cases"][case]
    vae = build_vae(case)
    x = hashed_input(f"{case}:x", spec["input_shape"])
    with torch.no_grad():
        latent = vae.encode(x, max_chunk_bytes=GOLDENS["budgets"][mode])
    torch.testing.assert_close(latent, dec(spec["encode"][mode]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_decode_matches_executed_reference(case: str, mode: str) -> None:
    spec = GOLDENS["cases"][case]
    vae = build_vae(case)
    latent = hashed_input(f"{case}:latent", spec["latent_shape"])
    if spec["seed"] is not None:
        torch.manual_seed(spec["seed"])
    with torch.no_grad():
        pixels = vae.decode(latent, max_chunk_bytes=GOLDENS["budgets"][mode])
    torch.testing.assert_close(pixels, dec(spec["decode"][mode]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_codec_residency_preserves_statistics_modulation_and_noise(case: str, device: str) -> None:
    from dinkster_inference_torch.module_residency import enroll_component

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA residency requires the GPU validation environment")
    spec = GOLDENS["cases"][case]
    vae = build_vae(case).to(device)
    content = hashed_input(f"{case}:x", spec["input_shape"]).to(device)
    latent = hashed_input(f"{case}:latent", spec["latent_shape"]).to(device)
    seed = 0 if spec["seed"] is None else spec["seed"]
    with torch.inference_mode():
        expected_encoded = vae.encode(content)
        torch.manual_seed(seed)
        expected_decoded = vae.decode(latent)
        mechanism = enroll_component(vae, load_device=device, offload_device="cpu")
        for budget in (None, 0, None):
            mechanism.unload()
            mechanism.partially_load(budget)
            assert bool(mechanism.loaded_unit_names()) == (budget is None)
            torch.testing.assert_close(vae.encode(content), expected_encoded, rtol=0, atol=0)
            torch.manual_seed(seed)
            torch.testing.assert_close(vae.decode(latent), expected_decoded, rtol=0, atol=0)
        mechanism.unload()


@pytest.mark.parametrize("case", CASES)
def test_decode_output_shape_matches_the_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    shape = build_vae(case).decode_output_shape(tuple(spec["latent_shape"]))
    assert list(shape) == spec["decode_output_shape"]


# ------------------------------------------------------- behavior


def test_encode_truncates_to_the_temporal_grid() -> None:
    """Frames beyond 1 + k*temporal_ratio never reach the encoder: a
    ten-frame clip encodes bit-identically to its first nine frames
    (the tiny v0 geometry has temporal ratio 2)."""
    vae = build_vae("vae_v0")
    x = hashed_input("truncation:x", (1, 3, 10, 16, 16))
    with torch.no_grad():
        assert torch.equal(vae.encode(x), vae.encode(x[:, :, :9]))


@pytest.mark.parametrize(
    ("free_memory", "expected"),
    (
        (6 * 1024**3 - 1, 32 * 1024**2),
        (6 * 1024**3, 32 * 1024**2),
        (6 * 1024**3 + 1, 32 * 1024**2),
        (24 * 1024**3 - 1, 128 * 1024**2 - 1),
        (24 * 1024**3, 128 * 1024**2),
        (24 * 1024**3 + 1, 128 * 1024**2),
    ),
)
def test_max_chunk_bytes_clamps_at_reference_boundaries(free_memory: int, expected: int) -> None:
    assert ltxv_vae_max_chunk_bytes(free_memory) == expected


def test_max_chunk_bytes_interpolates_an_asymmetric_reference_budget() -> None:
    gib = 1024**3
    free_memory = 10 * gib + 123_456_789

    assert ltxv_vae_max_chunk_bytes(free_memory) == 56_567_057


def test_causal_convolution_preserves_reference_spatial_padding_dispatch() -> None:
    vae = build_vae("vae_v0")
    assert vae.encoder.conv_in.conv.padding == (0, 1, 1)
    assert vae.decoder.conv_in.conv.padding == (0, 0, 0)
