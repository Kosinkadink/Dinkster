"""Stage-6 flip parity: the FluxRuntime seam vs executed ComfyUI.

flux_pipeline_goldens.json holds full-pipeline outputs from EXECUTING
the reference node chain at the object level (CheckpointLoader ->
CLIPTextEncode [-> FluxGuidance] -> KSampler -> VAEDecode, plus a
VAEEncode roundtrip) @ 947c2749 - see tools/
gen_flux_pipeline_goldens.py. These tests replay every case through
the seam the stage-6 flips will consume (encode_text -> sample ->
decode_latent / encode_content) and pin the results, so the first
flip window (Flux dev + schnell) inherits end-to-end Comfy-as-oracle
evidence per node surface:

- ksampler: euler and dpmpp_2m_sde (pre-offset brownian sizing) over
  simple and karras schedules; denoise 0.5 from a stored nonzero
  init latent (the truncated-schedule img2img leg); batch 2 with a
  batch-1 conditioning (the repeat_to_batch_size broadcast); cfg 3
  with a real encoded negative; the guidance-default dev case
  (absent key = 3.5), an explicit FluxGuidance 2.0, and the
  guidance-free schnell family (ModelType.FLOW, shift 1.0).
- clip_text_encode: the FluxClipModel composition (T5 sequence +
  CLIP-L pooled) over the real BPE/spm tokenizers, including the
  empty prompt and a > 77-token prompt (CLIP multi-chunk, pooled
  from the first chunk).
- vae_decode / vae_encode: the KL codec over sampled latents and a
  deterministic content roundtrip. The reference VAE surface is
  NHWC in [0, 1]; the native codec seam is NCHW - goldens store the
  reference layout and the replay movedims at the comparison.

Models are the generator's tiny geometry with the SAME deterministic
hash fills (unet_fill for the DiT, clip_fill for both text towers,
kl_fill for the VAE), keyed on state-dict spellings the native ports
share with the reference inner modules - equal keys and shapes mean
bit-identical parameters on both sides. Architectures come from the
golden payload, not local constants, so this file cannot drift from
what actually executed.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from clip_fill import fill_state_dict as clip_fill_state_dict
from dinkster_inference import (
    FLUX_DEV,
    FLUX_SCHNELL,
    ClipTextConfig,
    Conditioning,
    FluxConfig,
    KLConfig,
    ModelFamily,
    SamplingGuidance,
    T5Config,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    AssembledFlux,
    AutoencoderKL,
    ClipTextModel,
    Flux,
    FluxRuntime,
    T5TextModel,
    torch_sampler_registry,
    torch_scheduler_registry,
)
from dinkster_inference_torch.schedules import (
    _TORCH_SCHEDULES,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.wiring import (
    _flux_sigma_space,  # pyright: ignore[reportPrivateUsage]
)
from golden_files import assert_reference_schedule, assert_reference_tensor, load_platform_golden
from kl_fill import fill_state_dict as kl_fill_state_dict
from unet_fill import fill_state_dict as unet_fill_state_dict

GOLDENS: dict[str, Any] = load_platform_golden(
    Path(__file__).parent / "goldens" / "flux_pipeline_goldens.json"
)

# The executed-golden comparison tolerance shared with the component
# pins (test_flux.py): the reference ran CPU float32 with the SDPA
# attention backend and pure-torch RoPE, both of which the native
# side reproduces, so only accumulation-order noise remains.
RTOL = 1e-4
ATOL = 1e-5

# The sampling loop keeps the strict band by default: the flow
# space's sigma_max is 1.0 and denoised = x - sigma * v, so the
# per-forward accumulation wobble passes through un-amplified
# (unlike SD's EPS 14.61x - see test_sd_pipeline.py).
SAMPLE_RTOL = RTOL
SAMPLE_ATOL = ATOL

# Documented per-case widenings of SAMPLE_ATOL - never blanket. Each
# entry must come with schedule- and noise-parity evidence pinning
# the drift to denoiser accumulation: the three brownian SDE cases
# below walk more high-sigma steps than dev_sde's karras schedule
# (normal/exponential/sgm_uniform stay near sigma 1.0 longer, where
# cfg and the brownian mixing recombine full-scale activations), and
# instrumented replays showed bit-exact sigmas and brownian draws
# (test_schedule_sigmas_match_the_reference pins the schedules)
# with end-to-end drift up to 1.3e-4. 5e-4 gives ~4x
# headroom while staying ~60x under decorrelation-scale divergence.
SAMPLE_ATOL_OVERRIDES = {
    "dev_sde_normal": 5e-4,
    "dev_sde_exponential": 5e-4,
    "schnell_sde_sgm_uniform": 5e-4,
}

FillState = Callable[[list[tuple[str, list[int]]]], Mapping[str, torch.Tensor]]


def dec(payload: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def filled(module: torch.nn.Module, fill_state: FillState) -> Any:
    entries = sorted((key, list(value.shape)) for key, value in module.state_dict().items())
    module.load_state_dict(dict(fill_state(entries)), strict=True)
    return module


def assembled(family: ModelFamily, guidance_embed: bool) -> AssembledFlux:
    arch = GOLDENS["arch"]
    flux = FluxConfig(
        in_channels=arch["flux"]["in_channels"],
        out_channels=arch["flux"]["out_channels"],
        vec_in_dim=arch["flux"]["vec_in_dim"],
        context_in_dim=arch["flux"]["context_in_dim"],
        hidden_size=arch["flux"]["hidden_size"],
        depth=arch["flux"]["depth"],
        depth_single_blocks=arch["flux"]["depth_single_blocks"],
        num_heads=arch["flux"]["num_heads"],
        axes_dim=tuple(arch["flux"]["axes_dim"]),
        theta=arch["flux"]["theta"],
        patch_size=arch["flux"]["patch_size"],
        mlp_ratio=arch["flux"]["mlp_ratio"],
        qkv_bias=arch["flux"]["qkv_bias"],
        guidance_embed=guidance_embed,
    )
    # The generator's t5xxl overlay rides the reference config JSON,
    # whose activation is the gated gelu; the native config states it.
    t5 = T5Config(
        **arch["t5xxl"],
        dense_act_fn="gelu_pytorch_tanh",
        is_gated_act=True,
    )
    ddconfig = arch["vae_ddconfig"]
    kl = KLConfig(
        in_channels=ddconfig["in_channels"],
        out_channels=ddconfig["out_ch"],
        ch=ddconfig["ch"],
        decoder_ch=ddconfig["ch"],
        ch_mult=tuple(ddconfig["ch_mult"]),
        num_res_blocks=ddconfig["num_res_blocks"],
        z_channels=ddconfig["z_channels"],
        embed_dim=arch["vae_embed_dim"],
        dropout=ddconfig["dropout"],
    )
    return AssembledFlux(
        family=family,
        diffusion=filled(Flux(flux), unet_fill_state_dict),
        clip_l=filled(
            ClipTextModel(ClipTextConfig(**arch["clip_l"])),
            clip_fill_state_dict,
        ),
        t5xxl=filled(T5TextModel(t5), clip_fill_state_dict),
        vae=filled(AutoencoderKL(kl), kl_fill_state_dict),
    )


@pytest.fixture(scope="module")
def runtimes() -> dict[str, FluxRuntime]:
    return {
        "flux_dev": FluxRuntime(
            assembled(FLUX_DEV, guidance_embed=True),
            runtime_identity="native:test:pipeline:dev",
        ),
        "flux_schnell": FluxRuntime(
            assembled(FLUX_SCHNELL, guidance_embed=False),
            runtime_identity="native:test:pipeline:schnell",
        ),
    }


def initial_latent(case: Mapping[str, Any]) -> torch.Tensor:
    if case["init"] != "zeros":
        return dec(case["init_latent"])
    height, width = case["latent_size"]
    return torch.zeros(
        case["batch"],
        GOLDENS["arch"]["flux"]["in_channels"],
        height,
        width,
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    "case",
    GOLDENS["cases"],
    ids=[case["name"] for case in GOLDENS["cases"]],
)
def test_sampling_case_matches_the_executed_reference(
    case: Mapping[str, Any], runtimes: dict[str, FluxRuntime]
) -> None:
    runtime = runtimes[case["family"]]
    cond: Conditioning[torch.Tensor] = runtime.encode_text(case["prompt"])
    uncond = (
        runtime.encode_text(case["negative_prompt"])
        if case["negative_prompt"] is not None
        else None
    )
    out = runtime.sample(
        initial_latent(case),
        cond=cond,
        cfg=SamplingGuidance(uncond, case["cfg"]),
        sampler_id=f"dinkster.{case['sampler']}",
        scheduler_id=f"dinkster.{case['scheduler']}",
        steps=case["steps"],
        denoise=case["denoise"],
        seed=case["seed"],
        guidance=case["guidance"],
        compute_dtype=torch.float32,
    )
    atol = SAMPLE_ATOL_OVERRIDES.get(case["name"], SAMPLE_ATOL)
    assert_reference_tensor(out, dec(case["output_latent"]), rtol=SAMPLE_RTOL, atol=atol)
    pixels = runtime.decode_latent(out)
    assert_reference_tensor(
        pixels,
        dec(case["output_pixels_nhwc"]).movedim(-1, 1),
        rtol=SAMPLE_RTOL,
        atol=atol,
    )


@pytest.mark.parametrize(
    "case",
    GOLDENS["cases"],
    ids=[case["name"] for case in GOLDENS["cases"]],
)
def test_schedule_sigmas_match_the_reference(
    case: Mapping[str, Any],
) -> None:
    """Goldens record KSampler.sigmas (@ 947c2749) as exact float64
    reads of the float32 values - the PRE-offset schedule, before
    offset_first_sigma_for_snr - and the replay walks the same path
    FluxRuntime.sample walks (the wiring's per-family flow space,
    torch_scheduler_registry, KSampler's denoise trim and
    discard-penultimate correction). Schedules rebound to reference
    kernels (schedules.py _TORCH_SCHEDULES) must match BIT-exactly -
    that is the whole point of the ports. Pure table-index schedules
    are contracted to one float32 ulp instead (spaces.py); rel 1e-6
    gives ~16x headroom over an ulp while sitting orders of magnitude
    under behavioral drift."""
    sampler = torch_sampler_registry().get(f"dinkster.{case['sampler']}")
    scheduler = torch_scheduler_registry().get(f"dinkster.{case['scheduler']}")
    assert sampler is not None and scheduler is not None
    family = {"flux_dev": FLUX_DEV, "flux_schnell": FLUX_SCHNELL}[case["family"]]
    sigmas = sampling_sigmas(
        scheduler,
        _flux_sigma_space(family),
        case["steps"],
        denoise=case["denoise"],
        discard_penultimate=sampler.discard_penultimate,
    )
    assert_reference_schedule(
        sigmas,
        case["sigmas"],
        rel=None if f"dinkster.{case['scheduler']}" in _TORCH_SCHEDULES else 1e-6,
    )


def test_encode_text_matches_the_executed_reference(
    runtimes: dict[str, FluxRuntime],
) -> None:
    golden = GOLDENS["encode_text"]
    cond = runtimes["flux_dev"].encode_text(golden["prompt"])
    torch.testing.assert_close(cond.embeddings, dec(golden["cond"]), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(cond.pooled, dec(golden["pooled"]), rtol=RTOL, atol=ATOL)


def test_vae_roundtrip_matches_the_executed_reference(
    runtimes: dict[str, FluxRuntime],
) -> None:
    golden = GOLDENS["vae_roundtrip"]
    runtime = runtimes["flux_dev"]
    latent = runtime.encode_content(dec(golden["content_nchw"]))
    torch.testing.assert_close(latent, dec(golden["latent"]), rtol=RTOL, atol=ATOL)
    decoded = runtime.decode_latent(dec(golden["latent"]))
    torch.testing.assert_close(
        decoded, dec(golden["decoded_pixels_nhwc"]).movedim(-1, 1), rtol=RTOL, atol=ATOL
    )
