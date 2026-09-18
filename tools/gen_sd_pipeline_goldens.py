"""Generate full-pipeline SD1/SDXL/refiner goldens from executed ComfyUI.

Runs the REFERENCE txt2img pipeline @ the audited baseline - the
object-level body of the stage-6 node chain (CheckpointLoader ->
CLIPTextEncode -> KSampler -> VAEDecode, plus a VAEEncode roundtrip) -
and writes
packages/dinkster-inference-torch/tests/goldens/sd_pipeline_goldens.json.
The SDRuntime seam (encode_text -> sample -> decode_latent /
encode_content) is pinned against these outputs as flip-parity
evidence for the second stage-6 window (SD 1.5 + SDXL base + SDXL
refiner); the oracle is the executed reference, never a re-derivation.
tools/gen_flux_pipeline_goldens.py is the first-window sibling.

Node fidelity: the sampling body is nodes.py common_ksampler
@ 947c2749 verbatim at the object level (fix_empty_latent_channels,
prepare_noise from the case seed, comfy.sample.sample); text
conditioning is CLIPTextEncode's structure ([[cond, {"pooled_output":
pooled}]] from the per-family clip_target model over its tokenizer:
SD1ClipModel/SD1Tokenizer, SDXLClipModel/SDXLTokenizer,
SDXLRefinerClipModel over the same SDXLTokenizer); SDXL/refiner ADM
vectors are built INSIDE the sampler by encode_model_conds
(width/height from the noise shape * 8, crops 0, targets = sizes,
refiner aesthetic 6.0 positive / 2.5 negative via prompt_type) -
exactly what a plain CLIPTextEncode + KSampler workflow hits; VAE
decode/encode run through comfy.sd.VAE with an explicit config (the
tiny geometry is not inferable from a state dict).

Models are tiny (the test_wiring_sd TINY_* geometry: real CLIP
vocabulary so the real BPE drives both towers, everything else small;
the towers get DIFFERENT widths so the SDXL L-then-G feature-concat
order is observable) with deterministic hash-filled weights shared
with the replay tests: unet_fill for the UNets, clip_fill for the
text towers, kl_fill for the VAE - fills key on the INNER module
state dicts, whose spellings the native ports share.

Determinism: attention is forced to the pytorch SDPA backend, the one
Dinkster ports (recorded in the payload); the tiny VAE has no attention
resolutions and its mid-block attention picks the CPU pytorch path
under --cpu. Everything runs CPU float32.

Usage (a torch interpreter with the full comfy sampling closure:
numpy, torchsde, einops, transformers; the sibling comfy-aimdo and
comfy-kitchen checkouts are appended to sys.path in-script):

    /path/to/ref-venv/bin/python tools/gen_sd_pipeline_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (generation refuses on any other commit).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(
    os.environ.get("DINKSTER_COMFY_REFERENCE_ROOT", REPO.parent / "ComfyUI")
).resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
# Appended (never in front of the pin): comfy.ops imports comfy_aimdo
# and comfy.quant_ops imports comfy_kitchen; use the sibling checkouts
# when the interpreter lacks them.
sys.path.append(str(REPO.parent / "comfy-aimdo"))
sys.path.append(str(REPO.parent / "comfy-kitchen"))

# comfy.model_management probes CUDA at import time; --cpu keeps every
# device selection (sampling, VAE, intermediates) on CPU float32.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from clip_fill import fill_state_dict as clip_fill_state_dict  # noqa: E402
from comfy import (  # noqa: E402
    model_base,
    model_patcher,
    ops,
    sd1_clip,
    sdxl_clip,
    supported_models,
)
from comfy import sample as comfy_sample  # noqa: E402
from comfy import samplers as comfy_samplers  # noqa: E402
from comfy import sd as comfy_sd  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from kl_fill import fill_state_dict as kl_fill_state_dict  # noqa: E402
from unet_fill import fill_state_dict as unet_fill_state_dict  # noqa: E402

# The reference attention calls the ambient optimized_attention,
# selected per environment (xformers/sage/flash/pytorch). Goldens must
# not depend on which accelerators happen to be installed: force the
# pytorch SDPA backend, the one Dinkster ports.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
ATTENTION_BACKEND = "attention_pytorch (SDPA)"

OUT = platform_golden_path(
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "sd_pipeline_goldens.json",
    torch.__version__,
)

CPU = torch.device("cpu")

# --- tiny architectures (test_wiring_sd TINY_* geometry) --------------------
# The towers get different widths so the SDXL feature-concat order is
# observable: context_dim = clip_l + clip_g hidden for the base, one
# tower's hidden for SD1/refiner; adm_in_channels = the CLIP-G pooled
# width + the reference's Timestep(256) blocks (six base, five
# refiner).

TINY_CLIP_L = {
    "hidden_size": 32,
    # Three hidden layers, not test_wiring_sd's two: the reference's
    # penultimate selection asserts abs(-2) < num_hidden_layers.
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "hidden_act": "quick_gelu",
    "vocab_size": 49408,
    "eos_token_id": 49407,
}
TINY_CLIP_G = {
    "hidden_size": 48,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "intermediate_size": 96,
    "hidden_act": "gelu",
    "vocab_size": 49408,
    "eos_token_id": 49407,
}

# UNet arch dicts ride supported_models.*(arch) -> model_base like a
# CheckpointLoader product, so they carry the standard-UNet
# detect_unet_config base keys (comfy/model_detection.py @ 947c2749)
# on top of the geometry, in UNetModel kwarg spelling.
_UNET_BASE = {
    "use_checkpoint": False,
    "image_size": 32,
    "use_spatial_transformer": True,
    "legacy": False,
    "use_temporal_attention": False,
    "use_temporal_resblock": False,
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 32,
    "num_res_blocks": [1, 1],
    "channel_mult": [1, 2],
}
TINY_SD1_UNET = {
    **_UNET_BASE,
    "transformer_depth": [1, 1],
    "transformer_depth_output": [1, 1, 1, 1],
    "transformer_depth_middle": 1,
    "context_dim": TINY_CLIP_L["hidden_size"],
    "use_linear_in_transformer": False,
    "adm_in_channels": None,
    "num_heads": 8,
    "num_head_channels": -1,
}
TINY_SDXL_UNET = {
    **_UNET_BASE,
    "num_classes": "sequential",
    "transformer_depth": [0, 2],
    "transformer_depth_output": [0, 0, 2, 2],
    "transformer_depth_middle": 2,
    "context_dim": TINY_CLIP_L["hidden_size"] + TINY_CLIP_G["hidden_size"],
    "use_linear_in_transformer": True,
    "adm_in_channels": TINY_CLIP_G["hidden_size"] + 6 * 256,
    "num_heads": -1,
    "num_head_channels": 16,
}
TINY_REFINER_UNET = {
    **_UNET_BASE,
    "num_classes": "sequential",
    "transformer_depth": [0, 2],
    "transformer_depth_output": [0, 0, 2, 2],
    "transformer_depth_middle": 2,
    "context_dim": TINY_CLIP_G["hidden_size"],
    "use_linear_in_transformer": True,
    "adm_in_channels": TINY_CLIP_G["hidden_size"] + 5 * 256,
    "num_heads": -1,
    "num_head_channels": 16,
}
# ch rides the reference's fixed 32-group GroupNorm: every block
# channel count (ch * mult) must divide by 32. z_channels = the SD
# latent's 4.
TINY_VAE_DDCONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 32,
    "ch_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}
TINY_VAE_EMBED_DIM = 4

#: One sampling case = one executed reference KSampler chain. Prompts
#: stay plain (weight/embedding syntax is pinned at the tokenizer and
#: text-encoder layers); the long prompts exercise the CLIP
#: multi-chunk leg (pooled from the first chunk) on one and two
#: towers. init "randn:<seed>" latents are drawn once here and stored
#: verbatim, so the replay never depends on cross-version RNG.
#: sdxl_cfg3_rect's non-square latent pins the H/W order in the ADM
#: size vectors; refiner_cfg3's negative pins the 6.0/2.5 aesthetic
#: polarity.
LONG_PROMPT = (
    "a highly detailed oil painting of an ancient stone lighthouse on"
    " a cliff at dusk, crashing waves below, seagulls circling in the"
    " orange sky, dramatic clouds, distant sailing ships on the"
    " horizon, moss covered rocks in the foreground, warm lantern"
    " light glowing from the tower window, wind swept grass, a narrow"
    " winding path leading up the cliffside, masterful composition"
)

CASES = [
    {
        "name": "sd15_euler_baseline",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 7,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_cfg3",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 11,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    # CFG++ family through the REAL reference path: sampling_function's
    # post-CFG hook chain capturing the uncond inside CFGGuider, not the
    # mock-model harness of gen_sampling_goldens.py.
    {
        # Deterministic CFG++: proves the uncond seam with no step-noise
        # stream involved.
        "name": "sd15_euler_cfg_pp_cfg3",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "euler_cfg_pp",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 11,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        # cfg 1 WITH a negative: the cfg_pp hook's
        # disable_cfg1_optimization forces the uncond evaluation the
        # plain path would skip, and the ancestral variant additionally
        # pins the seeded gaussian step-noise stream end to end.
        "name": "sd15_euler_ancestral_cfg_pp_cfg1",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 1.0,
        "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 17,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_res_multistep_cfg3",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 23,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_res_multistep_ancestral_cfg_pp_cfg3",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "res_multistep_ancestral_cfg_pp",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 29,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    # Binding representatives for the remaining native solver families.
    # These execute the real Comfy denoiser/guider path rather than the
    # float64 mock used by gen_sampling_goldens.py. SA-Solver's default
    # stochastic interval calls model_sampling.percent_to_sigma(0.2/0.8),
    # and gradient-estimation CFG++ captures uncond through the post-CFG
    # hook chain.
    {
        "name": "sd15_uni_pc",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "uni_pc",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 131,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sa_solver",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "sa_solver",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 137,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_deis",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "deis",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 139,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_gradient_estimation_cfg_pp_cfg3",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "gradient_estimation_cfg_pp",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 149,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_exp_heun_2_x0",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "exp_heun_2_x0",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 151,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_seeds_2",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "seeds_2",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 157,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sde_karras",
        "family": "sd15",
        "prompt": LONG_PROMPT,
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 4,
        "denoise": 1.0,
        "seed": 13,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_denoise_half",
        "family": "sd15",
        "prompt": "a watercolor fox",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "karras",
        "steps": 4,
        "denoise": 0.5,
        "seed": 17,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "randn:1717",
    },
    {
        "name": "sd15_batch2",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 19,
        "batch": 2,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_empty_prompt",
        "family": "sd15",
        "prompt": "",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 23,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sdxl_euler_baseline",
        "family": "sdxl",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 29,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sdxl_cfg3_rect",
        "family": "sdxl",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 31,
        "batch": 1,
        "latent_size": [4, 6],
        "init": "zeros",
    },
    {
        "name": "sdxl_long_prompt",
        "family": "sdxl",
        "prompt": LONG_PROMPT,
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 37,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "refiner_euler_baseline",
        "family": "sdxl_refiner",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 41,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "refiner_cfg3",
        "family": "sdxl_refiner",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 43,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    # One brownian (dpmpp_2m_sde) case per remaining float32-sensitive
    # schedule: SDE replay needs bit-exact sigmas, so each schedule's
    # torch reference-kernel port (dinkster_inference_torch.schedules)
    # gets executed-golden coverage over the discrete SD15 space.
    {
        "name": "sd15_sde_exponential",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential",
        "steps": 4,
        "denoise": 1.0,
        "seed": 101,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sde_normal",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "normal",
        "steps": 4,
        "denoise": 1.0,
        "seed": 103,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sde_sgm_uniform",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "sgm_uniform",
        "steps": 4,
        "denoise": 1.0,
        "seed": 107,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sde_linear_quadratic",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "linear_quadratic",
        "steps": 4,
        "denoise": 1.0,
        "seed": 109,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "sd15_sde_kl_optimal",
        "family": "sd15",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "kl_optimal",
        "steps": 4,
        "denoise": 1.0,
        "seed": 113,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
]


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def build_diffusion(family: str) -> model_patcher.ModelPatcher:
    """The tiny reference diffusion model behind a CPU ModelPatcher -
    exactly the CheckpointLoader product for this family: the
    supported-model config carries the latent format (SD15 0.18215 /
    SDXL 0.13025) and the shared linear-beta EPS sampling; the model
    class carries the family's encode_adm (none / six-block /
    five-block-with-aesthetic)."""
    arch: dict = {
        "sd15": dict(TINY_SD1_UNET),
        "sdxl": dict(TINY_SDXL_UNET),
        "sdxl_refiner": dict(TINY_REFINER_UNET),
    }[family]
    arch["dtype"] = torch.float32
    if family == "sd15":
        config = supported_models.SD15(arch)
    elif family == "sdxl":
        config = supported_models.SDXL(arch)
    else:
        config = supported_models.SDXLRefiner(arch)
    # BASE.__init__ overlays the class's unet_extra_config head facts
    # (full-size SDXL uses 64-wide heads) over the passed dict;
    # restore the tiny geometry's heads. SD15's extra config (8
    # heads) already matches the tiny SD1.
    config.unet_config["num_heads"] = arch["num_heads"]
    config.unet_config["num_head_channels"] = arch["num_head_channels"]
    config.custom_operations = ops.disable_weight_init
    if family == "sd15":
        model = model_base.BaseModel(config, device=CPU)
    elif family == "sdxl":
        model = model_base.SDXL(config, device=CPU)
    else:
        model = model_base.SDXLRefiner(config, device=CPU)
    entries = sorted(
        (key, list(value.shape)) for key, value in model.diffusion_model.state_dict().items()
    )
    model.diffusion_model.load_state_dict(unet_fill_state_dict(entries), strict=True)
    return model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


def build_text(family: str) -> tuple[torch.nn.Module, object]:
    """The family's reference clip_target stack @ 947c2749 with the
    TINY overlays riding the model-options config channel (SDClipModel
    overlays '<model_name>_model_config' onto the JSON), every tower
    hash-filled on its inner transformer. The refiner shares
    SDXLTokenizer (its model indexes the 'g' half)."""
    model_options = {
        "custom_operations": ops.disable_weight_init,
        "clip_l_model_config": dict(TINY_CLIP_L),
        "clip_g_model_config": dict(TINY_CLIP_G),
    }
    if family == "sd15":
        model: torch.nn.Module = sd1_clip.SD1ClipModel(
            device="cpu", dtype=torch.float32, model_options=model_options
        )
        tokenizer: object = sd1_clip.SD1Tokenizer()
        towers = [model.clip_l]
    elif family == "sdxl":
        model = sdxl_clip.SDXLClipModel(
            device="cpu", dtype=torch.float32, model_options=model_options
        )
        tokenizer = sdxl_clip.SDXLTokenizer()
        towers = [model.clip_l, model.clip_g]
    else:
        model = sdxl_clip.SDXLRefinerClipModel(
            device="cpu", dtype=torch.float32, model_options=model_options
        )
        tokenizer = sdxl_clip.SDXLTokenizer()
        towers = [model.clip_g]
    for tower in towers:
        entries = sorted(
            (key, list(value.shape)) for key, value in tower.transformer.state_dict().items()
        )
        tower.transformer.load_state_dict(clip_fill_state_dict(entries), strict=True)
    return model, tokenizer


def build_vae() -> comfy_sd.VAE:
    """The tiny reference VAE through comfy.sd.VAE with an explicit
    config (state-dict detection hardcodes SD1 geometry and cannot
    infer this tiny ladder), hash-filled with the kl fill."""
    probe = comfy_sd.VAE(
        sd={},
        config={
            "params": {
                "ddconfig": dict(TINY_VAE_DDCONFIG),
                "embed_dim": TINY_VAE_EMBED_DIM,
            }
        },
        device=CPU,
        dtype=torch.float32,
    )
    entries = sorted(
        (key, list(value.shape)) for key, value in probe.first_stage_model.state_dict().items()
    )
    probe.first_stage_model.load_state_dict(kl_fill_state_dict(entries), strict=True)
    return probe


def encode_conditioning(text_model: torch.nn.Module, tokenizer: object, text: str) -> list:
    """CLIPTextEncode at the object level: [[cond, {"pooled_output":
    pooled}]]. The ADM metadata (width/height/prompt_type) is NOT set
    here - encode_model_conds fills it inside the sampler, exactly
    like the plain node graph."""
    tokens = tokenizer.tokenize_with_weights(text, return_word_ids=False)
    with torch.no_grad():
        cond, pooled = text_model.encode_token_weights(tokens)
    return [[cond, {"pooled_output": pooled}]]


def init_latent(case: dict) -> torch.Tensor:
    height, width = case["latent_size"]
    shape = (case["batch"], 4, height, width)
    if case["init"] == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    seed = int(case["init"].removeprefix("randn:"))
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator)


def run_case(
    case: dict,
    diffusion: model_patcher.ModelPatcher,
    text_model: torch.nn.Module,
    tokenizer: object,
    vae: comfy_sd.VAE,
) -> dict:
    """nodes.py common_ksampler @ 947c2749 at the object level, then
    VAEDecode. negative=[] is the reference-legal minimal uncond at
    cfg 1 (sampling_function drops uncond entirely); cfg > 1 cases
    encode a real negative prompt like the node graph would."""
    positive = encode_conditioning(text_model, tokenizer, case["prompt"])
    negative = (
        encode_conditioning(text_model, tokenizer, case["negative_prompt"])
        if case["negative_prompt"] is not None
        else []
    )
    latent_image = init_latent(case)
    latent_image = comfy_sample.fix_empty_latent_channels(diffusion, latent_image)
    noise = comfy_sample.prepare_noise(latent_image, case["seed"])
    with torch.no_grad():
        samples = comfy_sample.sample(
            diffusion,
            noise,
            case["steps"],
            case["cfg"],
            case["sampler"],
            case["scheduler"],
            positive,
            negative,
            latent_image,
            denoise=case["denoise"],
            disable_pbar=True,
            seed=case["seed"],
        )
        pixels = vae.decode(samples)
    out = dict(case)
    if case["init"] != "zeros":
        out["init_latent"] = enc(init_latent(case))
    out["output_latent"] = enc(samples)
    # comfy.sd.VAE.decode returns NHWC in [0, 1] (process_output +
    # movedim); stored as returned.
    out["output_pixels_nhwc"] = enc(pixels)
    # The exact schedule this run walked (KSampler.set_steps
    # @ 947c2749: scheduler + discard-penultimate + denoise trim),
    # recorded as exact float64 reads of the float32 values so the
    # replay can pin bit-equality on the scheduler seam.
    out["sigmas"] = [
        float(sigma)
        for sigma in comfy_samplers.KSampler(
            diffusion,
            steps=case["steps"],
            device=torch.device("cpu"),
            sampler=case["sampler"],
            scheduler=case["scheduler"],
            denoise=case["denoise"],
        ).sigmas
    ]
    return out


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(
            f"{COMFY_ROOT} is at {commit}; goldens must be generated"
            f" from the audited baseline {REFERENCE_COMMIT}"
        )
    module_file = Path(model_base.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            f"comfy.model_base was imported from {module_file}, not"
            f" the pinned checkout {COMFY_ROOT}"
        )

    families = ("sd15", "sdxl", "sdxl_refiner")
    text = {family: build_text(family) for family in families}
    vae = build_vae()
    diffusion = {family: build_diffusion(family) for family in families}

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention_backend": ATTENTION_BACKEND,
            **tuple_provenance(torch.__version__),
        },
        "arch": {
            "sd1_unet": dict(TINY_SD1_UNET),
            "sdxl_unet": dict(TINY_SDXL_UNET),
            "refiner_unet": dict(TINY_REFINER_UNET),
            "clip_l": dict(TINY_CLIP_L),
            "clip_g": dict(TINY_CLIP_G),
            "vae_ddconfig": dict(TINY_VAE_DDCONFIG),
            "vae_embed_dim": TINY_VAE_EMBED_DIM,
        },
        "cases": [],
    }

    for case in CASES:
        family = case["family"]
        text_model, tokenizer = text[family]
        payload["cases"].append(run_case(case, diffusion[family], text_model, tokenizer, vae))
        print(f"ran {case['name']}", file=sys.stderr)

    # The per-family composed text encode, pinned once with full
    # values: SD1 = CLIP-L final hidden + raw pooled; SDXL = L-then-G
    # penultimate feature concat + CLIP-G projected pooled; refiner =
    # CLIP-G alone.
    payload["encode_text"] = {}
    for family in families:
        text_model, tokenizer = text[family]
        tokens = tokenizer.tokenize_with_weights("a photo of a cat", return_word_ids=False)
        with torch.no_grad():
            cond, pooled = text_model.encode_token_weights(tokens)
        payload["encode_text"][family] = {
            "prompt": "a photo of a cat",
            "cond": enc(cond),
            "pooled": enc(pooled),
        }

    # VAEEncode/VAEDecode roundtrip on deterministic content: comfy
    # VAE.encode takes NHWC [0, 1] (the LoadImage surface); content is
    # stored NCHW to match the codec seam.
    from kl_fill import hash_uniform  # noqa: E402  (values in [0, 1))

    content_nchw = hash_uniform(0xC0FFEE, 1 * 3 * 16 * 16).reshape(1, 3, 16, 16)
    content_nchw = content_nchw.to(torch.float32)
    with torch.no_grad():
        encoded = vae.encode(content_nchw.movedim(1, -1))
        decoded = vae.decode(encoded)
    payload["vae_roundtrip"] = {
        "content_nchw": enc(content_nchw),
        "latent": enc(encoded),
        "decoded_pixels_nhwc": enc(decoded),
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
