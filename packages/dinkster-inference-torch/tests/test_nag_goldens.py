"""Replay the executed ComfyUI NAG oracle through Dinkster's attention seam.

The exact float values are platform-dependent (norm and conv kernels
drift by ULPs across torch builds and CPU microarchitectures), so the
fixture goes through the platform-tuple golden loader with the mint
host's CPU pinned: hosts whose CPU differs skip the module instead of
enforcing bit equality (generator: tools/gen_nag_goldens.py).

"patch_cases" pin the rewrite math at one attention site;
"unet_cases" pin the threading - a patched attn1 output feeding every
layer after it through the full tiny-UNet forward - and double as the
proof that the attention_guidance seam leaves the unpatched forward
bit-identical to the reference (output_plain).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import UNetConfig
from dinkster_inference_torch import UNetModel, guidance_transforms
from dinkster_inference_torch.unet import AttentionGuidanceContext
from golden_files import load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens/nag_goldens.json")


def dec(value: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(value["data"], dtype=torch.float32).reshape(value["shape"])


def guidance_context(case: dict[str, Any], batch: int) -> AttentionGuidanceContext:
    params = case["params"]
    descriptor = guidance_transforms.nag(
        params["nag_scale"], params["nag_alpha"], params["nag_tau"]
    ).attention
    assert descriptor is not None
    cond_or_uncond = [int(value) for value in case["cond_or_uncond"]]
    half = batch // len(cond_or_uncond)
    ind_pos = cond_or_uncond.index(0)
    ind_neg = cond_or_uncond.index(1)
    return AttentionGuidanceContext(
        transforms=(descriptor,),
        positive=(half * ind_pos, half * (ind_pos + 1)),
        negative=(half * ind_neg, half * (ind_neg + 1)),
    )


@pytest.mark.parametrize("name", sorted(GOLDENS["patch_cases"]))
def test_nag_rewrite_matches_executed_comfy_reference(name: str) -> None:
    case = GOLDENS["patch_cases"][name]
    out = dec(case["input"])
    got = guidance_context(case, out.shape[0]).apply(out)
    torch.testing.assert_close(got, dec(case["output"]), rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(GOLDENS["unet_cases"]))
def test_nag_multi_layer_forward_matches_executed_comfy_reference(name: str) -> None:
    case = GOLDENS["unet_cases"][name]
    spec = dict(case["config"])
    for field in (
        "num_res_blocks",
        "channel_mult",
        "transformer_depth",
        "transformer_depth_output",
    ):
        spec[field] = tuple(spec[field])
    model = UNetModel(UNetConfig(**spec))
    entries = [(key, list(shape)) for key, shape in case["state_dict"]]
    model.load_state_dict(fill_state_dict(entries), strict=True)

    batch = len(case["cond_or_uncond"])
    x = hashed_input(f"{name}:x", (batch, spec["in_channels"], case["height"], case["width"]))
    timesteps = torch.tensor(case["timesteps"], dtype=torch.float32)
    context = hashed_input(f"{name}:context", (batch, case["context_len"], spec["context_dim"]))

    with torch.no_grad():
        plain = model(x, timesteps, context)
        patched = model(x, timesteps, context, attention_guidance=guidance_context(case, batch))
    torch.testing.assert_close(plain, dec(case["output_plain"]), rtol=0, atol=0)
    torch.testing.assert_close(patched, dec(case["output"]), rtol=0, atol=0)
    assert not torch.equal(patched, plain)
