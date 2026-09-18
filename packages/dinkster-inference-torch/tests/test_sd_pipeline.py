"""Stage-6 flip parity: the SDRuntime seam vs executed ComfyUI.

sd_pipeline_goldens.json holds full-pipeline outputs from EXECUTING
the reference node chain at the object level (CheckpointLoader ->
CLIPTextEncode -> KSampler -> VAEDecode, plus a VAEEncode roundtrip)
@ 947c2749 - see tools/gen_sd_pipeline_goldens.py. These tests replay
every case through the seam the stage-6 flips will consume
(encode_text -> sample -> decode_latent / encode_content) and pin the
results, so the second flip window (SD 1.5 + SDXL base + SDXL
refiner) inherits end-to-end Comfy-as-oracle evidence per node
surface, like test_pipeline.py does for the first (Flux) window:

- ksampler: euler and dpmpp_2m_sde over simple and karras schedules
  on the shared discrete linear-beta EPS table; denoise 0.5 from a
  stored nonzero init latent (the truncated-schedule img2img leg);
  batch 2 with a batch-1 conditioning (the repeat_to_batch_size
  broadcast); cfg 3 with a real encoded negative on all three
  families - for SDXL over a NON-SQUARE latent (the ADM
  encode_model_conds defaults derive width/height from the latent,
  so H/W transposition would move the goldens) and for the refiner
  pinning the 6.0/2.5 aesthetic polarity.
- clip_text_encode: the per-family CLIP stack (SD1 = CLIP-L final
  hidden + raw pooled; SDXL = L-then-G penultimate feature concat +
  CLIP-G projected pooled; refiner = CLIP-G alone), including the
  empty prompt and > 77-token prompts (multi-chunk, pooled from the
  first chunk) on one and two towers.
- vae_decode / vae_encode: the KL codec over sampled latents and a
  deterministic content roundtrip. The reference VAE surface is
  NHWC in [0, 1]; the native codec seam is NCHW - goldens store the
  reference layout and the replay movedims at the comparison.

Models are the generator's tiny geometry with the SAME deterministic
hash fills (unet_fill for the UNets, clip_fill for the text towers,
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
    SD15,
    SDXL,
    SDXL_REFINER,
    ClipTextConfig,
    Conditioning,
    DiscreteSigmas,
    KLConfig,
    SamplingGuidance,
    UNetConfig,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    AssembledSD,
    AutoencoderKL,
    ClipTextModel,
    SDRuntime,
    UNetModel,
    torch_sampler_registry,
    torch_scheduler_registry,
)
from dinkster_inference_torch.schedules import (
    _TORCH_SCHEDULES,  # pyright: ignore[reportPrivateUsage]
)
from golden_files import assert_reference_schedule, assert_reference_tensor, load_platform_golden
from kl_fill import fill_state_dict as kl_fill_state_dict
from unet_fill import fill_state_dict as unet_fill_state_dict

GOLDENS: dict[str, Any] = load_platform_golden(
    Path(__file__).parent / "goldens" / "sd_pipeline_goldens.json"
)

# The executed-golden comparison tolerance for the non-sampling
# seams, shared with the component pins (test_unet.py,
# test_clip_text.py): the reference ran CPU float32 with the SDPA
# attention backend, which the native side reproduces, so only
# accumulation-order noise remains.
RTOL = 1e-4
ATOL = 1e-5

# The sampling loop needs a wider absolute band than Flux
# (test_pipeline.py) because the EPS parameterization AMPLIFIES the
# per-forward accumulation wobble instead of passing it through:
# denoised = x - sigma * eps, and this family's sigma_max is 14.61
# (vs 1.0 on the flow space), so the first step multiplies the UNet
# implementations' order-of-accumulation noise by ~15x; cfg 3
# (uncond + 3 * (cond - uncond)) stacks up to another 4x, and each
# further step compounds. Instrumented replays (bit-exact x and
# noise in, single forward compared) show per-element per-forward
# wobble of ~1e-6 and a worst end-to-end drift of 9.7e-4
# (refiner_cfg3); 2e-3 gives 2x headroom while staying an order of
# magnitude under any behavioral divergence this harness exists to
# catch (the karras SDE sigma bug it already caught was 3.4e-2).
SAMPLE_RTOL = 1e-4
SAMPLE_ATOL = 2e-3

# Documented per-case widenings of SAMPLE_ATOL - never blanket. Each
# entry must come with schedule- and noise-parity evidence pinning
# the drift to denoiser accumulation:
# - sd15_sde_linear_quadratic: the linear leg descends from
#   sigma_max 14.61 to only 10.6 across the WHOLE schedule (the
#   quadratic drop to 0 is the single final step), so the EPS ~15x
#   amplification applies at every step instead of decaying, and
#   dpmpp_2m_sde compounds it; instrumented replay showed bit-exact
#   sigmas, brownian tree bounds, and every brownian draw (and
#   test_schedule_sigmas_match_the_reference pins the schedule),
#   with end-to-end drift 2.7e-3. 5e-3 gives ~2x headroom
#   and stays 7x under the 3.4e-2 decorrelation this harness exists
#   to catch.
# - The six binding-representative cases below use the same bit-exact
#   simple schedule and initial/step-noise streams as their executed
#   references. UniPC additionally uses its executed-reference float32
#   coefficient/update trajectory exactly; its remaining full-pipeline
#   drift is 1.63e-4 latent and 1.44e-6 decoded pixels, so 4e-4 gives
#   2.4x headroom for denoiser accumulation. Instrumented native replays
#   measured maximum latent drift of 2.06e-4 (sa_solver), 1.53e-4 (deis),
#   6.49e-4 (gradient-estimation CFG++), 1.23e-4 (exp Heun), and
#   3.67e-4 (SEEDS 2). Their individual bands give roughly 2x
#   headroom for denoiser float32 accumulation without weakening the
#   other cases; decoded-pixel drift was at most 4.3e-6.
SAMPLE_ATOL_OVERRIDES = {
    "sd15_sde_linear_quadratic": 5e-3,
    "sd15_uni_pc": 4e-4,
    "sd15_sa_solver": 5e-4,
    "sd15_deis": 4e-4,
    "sd15_gradient_estimation_cfg_pp_cfg3": 1.4e-3,
    "sd15_exp_heun_2_x0": 3e-4,
    "sd15_seeds_2": 8e-4,
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


def unet_config(arch: Mapping[str, Any]) -> UNetConfig:
    """The golden's reference-kwarg dict in native spelling; -1 head
    fields and a None adm ride the same sentinels as the reference."""
    return UNetConfig(
        in_channels=arch["in_channels"],
        out_channels=arch["out_channels"],
        model_channels=arch["model_channels"],
        num_res_blocks=tuple(arch["num_res_blocks"]),
        channel_mult=tuple(arch["channel_mult"]),
        transformer_depth=tuple(arch["transformer_depth"]),
        transformer_depth_output=tuple(arch["transformer_depth_output"]),
        transformer_depth_middle=arch["transformer_depth_middle"],
        context_dim=arch["context_dim"],
        use_linear_in_transformer=arch["use_linear_in_transformer"],
        adm_in_channels=arch["adm_in_channels"],
        num_heads=arch["num_heads"],
        num_head_channels=arch["num_head_channels"],
    )


def assembled(family_id: str) -> AssembledSD:
    arch = GOLDENS["arch"]
    clip_l = ClipTextConfig(**arch["clip_l"])
    clip_g = ClipTextConfig(**arch["clip_g"])
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
    family, unet_arch, with_l, with_g = {
        "sd15": (SD15, arch["sd1_unet"], True, False),
        "sdxl": (SDXL, arch["sdxl_unet"], True, True),
        "sdxl_refiner": (SDXL_REFINER, arch["refiner_unet"], False, True),
    }[family_id]
    return AssembledSD(
        family=family,
        diffusion=filled(UNetModel(unet_config(unet_arch)), unet_fill_state_dict),
        clip_l=(filled(ClipTextModel(clip_l), clip_fill_state_dict) if with_l else None),
        clip_g=(filled(ClipTextModel(clip_g), clip_fill_state_dict) if with_g else None),
        vae=filled(AutoencoderKL(kl), kl_fill_state_dict),
    )


@pytest.fixture(scope="module")
def runtimes() -> dict[str, SDRuntime]:
    return {
        family_id: SDRuntime(
            assembled(family_id),
            runtime_identity=f"native:test:pipeline:{family_id}",
        )
        for family_id in ("sd15", "sdxl", "sdxl_refiner")
    }


def initial_latent(case: Mapping[str, Any]) -> torch.Tensor:
    if case["init"] != "zeros":
        return dec(case["init_latent"])
    height, width = case["latent_size"]
    return torch.zeros(case["batch"], 4, height, width, dtype=torch.float32)


@pytest.mark.parametrize(
    "case",
    GOLDENS["cases"],
    ids=[case["name"] for case in GOLDENS["cases"]],
)
def test_sampling_case_matches_the_executed_reference(
    case: Mapping[str, Any], runtimes: dict[str, SDRuntime]
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
    reads of the float32 values; the replay walks the same path
    SDRuntime.sample walks (the shared discrete linear-beta space,
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
    sigmas = sampling_sigmas(
        scheduler,
        DiscreteSigmas.linear_beta(),
        case["steps"],
        denoise=case["denoise"],
        discard_penultimate=sampler.discard_penultimate,
    )
    assert_reference_schedule(
        sigmas,
        case["sigmas"],
        rel=None if f"dinkster.{case['scheduler']}" in _TORCH_SCHEDULES else 1e-6,
    )


@pytest.mark.parametrize("family_id", ["sd15", "sdxl", "sdxl_refiner"])
def test_encode_text_matches_the_executed_reference(
    family_id: str, runtimes: dict[str, SDRuntime]
) -> None:
    golden = GOLDENS["encode_text"][family_id]
    cond = runtimes[family_id].encode_text(golden["prompt"])
    torch.testing.assert_close(cond.embeddings, dec(golden["cond"]), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(cond.pooled, dec(golden["pooled"]), rtol=RTOL, atol=ATOL)


def test_vae_roundtrip_matches_the_executed_reference(
    runtimes: dict[str, SDRuntime],
) -> None:
    golden = GOLDENS["vae_roundtrip"]
    runtime = runtimes["sd15"]
    latent = runtime.encode_content(dec(golden["content_nchw"]))
    torch.testing.assert_close(latent, dec(golden["latent"]), rtol=RTOL, atol=ATOL)
    decoded = runtime.decode_latent(dec(golden["latent"]))
    torch.testing.assert_close(
        decoded, dec(golden["decoded_pixels_nhwc"]).movedim(-1, 1), rtol=RTOL, atol=ATOL
    )
