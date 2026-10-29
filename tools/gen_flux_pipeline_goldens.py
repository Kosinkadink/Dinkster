"""Generate full-pipeline Flux goldens from executed ComfyUI.

Runs the REFERENCE txt2img pipeline @ the audited baseline - the
object-level body of the stage-6 node chain (CheckpointLoader ->
CLIPTextEncode [-> FluxGuidance] -> KSampler -> VAEDecode, plus a
VAEEncode roundtrip) - and writes
packages/dinkster-inference-torch/tests/goldens/flux_pipeline_goldens.json.
The FluxRuntime seam (encode_text -> sample -> decode_latent /
encode_content) is pinned against these outputs as flip-parity
evidence for the first stage-6 window (Flux dev + schnell); the
oracle is the executed reference, never a re-derivation.

Node fidelity: the sampling body is nodes.py common_ksampler
@ 947c2749 verbatim at the object level (fix_empty_latent_channels,
prepare_noise from the case seed, comfy.sample.sample); text
conditioning is CLIPTextEncode's structure ([[cond, {"pooled_output":
pooled}]] from FluxClipModel.encode_token_weights over FluxTokenizer),
with FluxGuidance's conditioning_set_values({"guidance": g}) applied
when the case carries an explicit guidance; VAE decode/encode run
through comfy.sd.VAE with an explicit config (the tiny geometry is
not inferable from a state dict). The CLIP wrapper's scheduling glue
(encode_from_tokens_scheduled) adds no math at this pin - hooks and
layer overrides are absent here, exactly like the golden cases.

Models are tiny (the test_wiring TINY_* geometry: real vocabulary
sizes so the real BPE/spm tokenizers drive them, everything else
small) with deterministic hash-filled weights shared with the replay
tests: unet_fill for the DiT, clip_fill for both text towers, kl_fill
for the VAE - fills key on the INNER module state dicts, whose
spellings the native ports share.

Determinism: attention is forced to the pytorch SDPA backend and
RoPE to the reference's pure-torch path (in_training=True), exactly
like tools/gen_flux_goldens.py; both are recorded in the payload.
Everything runs CPU float32 under --cpu.

Usage (a torch interpreter with the full comfy sampling closure:
numpy, torchsde, einops, transformers; the sibling comfy-aimdo and
comfy-kitchen checkouts are appended to sys.path in-script):

    /path/to/ref-venv/bin/python tools/gen_flux_pipeline_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (generation refuses on any other commit).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
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
    model_management,
    model_patcher,
    ops,
    supported_models,
)
from comfy import sample as comfy_sample  # noqa: E402
from comfy import samplers as comfy_samplers  # noqa: E402
from comfy import sd as comfy_sd  # noqa: E402
from comfy.ldm.flux import math as _flux_math  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy.text_encoders import flux as flux_te  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from kl_fill import fill_state_dict as kl_fill_state_dict  # noqa: E402
from unet_fill import fill_state_dict as unet_fill_state_dict  # noqa: E402

# The reference attention calls the ambient optimized_attention,
# selected per environment (xformers/sage/flash/pytorch). Goldens must
# not depend on which accelerators happen to be installed: force the
# pytorch SDPA backend, the one Dinkster ports. comfy/ldm/flux/math.py
# binds it with a from-import, so patch the bound global there too.
_flux_math.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert _flux_math.attention.__globals__["optimized_attention"] is (_attention.attention_pytorch), (
    "flux math is not calling the forced pytorch SDPA backend"
)
ATTENTION_BACKEND = "attention_pytorch (SDPA)"

# comfy/ldm/flux/math.py apply_rope dispatches to the comfy-kitchen
# kernel unless in_training is set; force the pure-torch reference
# path, the one the flux goldens pinned and Dinkster ports.
model_management.in_training = True
ROPE_BACKEND = "_apply_rope (pure torch, in_training=True)"

OUT = platform_golden_path(
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "flux_pipeline_goldens.json",
    torch.__version__,
)

CPU = torch.device("cpu")

# --- tiny architectures (test_wiring TINY_* geometry) ----------------------
# Cross-component dims cohere like a real Flux: context_in_dim = T5
# d_model, vec_in_dim = CLIP hidden, in_channels = VAE embed_dim.
# sum(axes_dim) == hidden_size // num_heads, every axis even (RoPE).

TINY_CLIP = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "hidden_act": "quick_gelu",
    "vocab_size": 49408,
    "eos_token_id": 49407,
}
TINY_T5 = {
    "d_model": 48,
    "d_ff": 96,
    "d_kv": 12,
    "num_heads": 4,
    "num_layers": 2,
    "vocab_size": 32128,
}
TINY_FLUX = {
    "in_channels": 16,
    "out_channels": 16,
    "vec_in_dim": TINY_CLIP["hidden_size"],
    "context_in_dim": TINY_T5["d_model"],
    "hidden_size": 32,
    "mlp_ratio": 4.0,
    "num_heads": 2,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [4, 6, 6],
    "theta": 10000,
    "patch_size": 2,
    "qkv_bias": True,
}
# ch rides the reference's fixed 32-group GroupNorm: every block
# channel count (ch * mult) must divide by 32.
TINY_VAE_DDCONFIG = {
    "double_z": True,
    "z_channels": 16,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 32,
    "ch_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}
TINY_VAE_EMBED_DIM = 16

#: One sampling case = one executed reference KSampler chain. Prompts
#: stay plain (weight/embedding syntax is pinned at the tokenizer and
#: text-encoder layers); dev_sde's long prompt exercises the CLIP
#: multi-chunk leg (pooled from the first chunk) while T5 pads to 256
#: either way. init "randn:<seed>" latents are drawn once here and
#: stored verbatim, so the replay never depends on cross-version RNG.
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
        "name": "dev_euler_baseline",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
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
        "name": "dev_guidance_2",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": 2.0,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 11,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "dev_sde",
        "family": "flux_dev",
        "prompt": LONG_PROMPT,
        "negative_prompt": None,
        "guidance": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 13,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "dev_karras_denoise_half",
        "family": "flux_dev",
        "prompt": "a watercolor fox",
        "negative_prompt": None,
        "guidance": None,
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
        "name": "dev_batch2",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
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
        "name": "dev_empty_prompt",
        "family": "flux_dev",
        "prompt": "",
        "negative_prompt": None,
        "guidance": None,
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
        "name": "dev_cfg3",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": "blurry, low quality",
        "guidance": None,
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 3,
        "denoise": 1.0,
        "seed": 29,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "schnell_euler",
        "family": "flux_schnell",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "denoise": 1.0,
        "seed": 31,
        "batch": 1,
        "latent_size": [6, 6],
        "init": "zeros",
    },
    # Brownian (dpmpp_2m_sde) coverage for the flow-space scheduler
    # ports (dinkster_inference_torch.schedules): normal/sgm_uniform
    # route through the space's timestep()/sigma() conversions and
    # exponential through its float32 table endpoints, so both flow
    # space kinds (dev: ModelSamplingFlux; schnell:
    # ModelSamplingDiscreteFlow) need executed-golden evidence.
    {
        "name": "dev_sde_normal",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "normal",
        "steps": 4,
        "denoise": 1.0,
        "seed": 101,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "dev_sde_exponential",
        "family": "flux_dev",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
        "cfg": 1.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential",
        "steps": 4,
        "denoise": 1.0,
        "seed": 103,
        "batch": 1,
        "latent_size": [4, 4],
        "init": "zeros",
    },
    {
        "name": "schnell_sde_sgm_uniform",
        "family": "flux_schnell",
        "prompt": "a photo of a cat",
        "negative_prompt": None,
        "guidance": None,
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
    supported-model config carries the sampling settings (dev:
    ModelSamplingFlux shift 1.15 default; schnell: ModelType.FLOW,
    shift 1.0, multiplier 1.0) and the Flux latent format."""
    arch = dict(TINY_FLUX)
    arch["image_model"] = "flux"
    arch["dtype"] = torch.float32
    arch["txt_ids_dims"] = []
    if family == "flux_dev":
        arch["guidance_embed"] = True
        config = supported_models.Flux(arch)
        config.custom_operations = ops.disable_weight_init
        model = model_base.Flux(config, device=CPU)
    else:
        arch["guidance_embed"] = False
        config = supported_models.FluxSchnell(arch)
        config.custom_operations = ops.disable_weight_init
        model = model_base.Flux(config, model_type=model_base.ModelType.FLOW, device=CPU)
    entries = sorted(
        (key, list(value.shape)) for key, value in model.diffusion_model.state_dict().items()
    )
    model.diffusion_model.load_state_dict(unet_fill_state_dict(entries), strict=True)
    return model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


def build_text() -> tuple[flux_te.FluxClipModel, flux_te.FluxTokenizer]:
    """The tiny reference Flux text stack: FluxClipModel with the
    TINY overlays riding the model-options config channel (the base
    SDClipModel overlays '<model_name>_model_config' onto the JSON),
    both towers hash-filled on their inner transformers."""
    model = flux_te.FluxClipModel(
        dtype_t5=torch.float32,
        device="cpu",
        dtype=torch.float32,
        model_options={
            "custom_operations": ops.disable_weight_init,
            "clip_l_model_config": dict(TINY_CLIP),
            "t5xxl_model_config": dict(TINY_T5),
        },
    )
    for tower in (model.clip_l, model.t5xxl):
        entries = sorted(
            (key, list(value.shape)) for key, value in tower.transformer.state_dict().items()
        )
        tower.transformer.load_state_dict(clip_fill_state_dict(entries), strict=True)
    return model, flux_te.FluxTokenizer()


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


def encode_conditioning(
    text_model: flux_te.FluxClipModel,
    tokenizer: flux_te.FluxTokenizer,
    text: str,
    guidance: float | None,
) -> list:
    """CLIPTextEncode at the object level ([[cond, {"pooled_output":
    pooled}]]), plus FluxGuidance's conditioning_set_values when the
    case carries an explicit guidance."""
    tokens = tokenizer.tokenize_with_weights(text, return_word_ids=False)
    with torch.no_grad():
        cond, pooled = text_model.encode_token_weights(tokens)
    metadata: dict = {"pooled_output": pooled}
    if guidance is not None:
        # FluxGuidance = node_helpers.conditioning_set_values(
        # conditioning, {"guidance": g}) @ 947c2749: a copied metadata
        # dict with the key set.
        metadata["guidance"] = guidance
    return [[cond, metadata]]


def init_latent(case: dict) -> torch.Tensor:
    height, width = case["latent_size"]
    shape = (case["batch"], TINY_FLUX["in_channels"], height, width)
    if case["init"] == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    seed = int(case["init"].removeprefix("randn:"))
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator)


def run_case(
    case: dict,
    diffusion: model_patcher.ModelPatcher,
    text_model: flux_te.FluxClipModel,
    tokenizer: flux_te.FluxTokenizer,
    vae: comfy_sd.VAE,
) -> dict:
    """nodes.py common_ksampler @ 947c2749 at the object level, then
    VAEDecode. negative=[] is the reference-legal minimal uncond at
    cfg 1 (sampling_function drops uncond entirely); cfg > 1 cases
    encode a real negative prompt like the node graph would."""
    positive = encode_conditioning(text_model, tokenizer, case["prompt"], case["guidance"])
    negative = (
        encode_conditioning(text_model, tokenizer, case["negative_prompt"], None)
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
    # replay can pin bit-equality on the scheduler seam. This is the
    # PRE-offset schedule: offset_first_sigma_for_snr happens inside
    # the solvers, after KSampler.sigmas is fixed.
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

    text_model, tokenizer = build_text()
    vae = build_vae()
    diffusion = {
        "flux_dev": build_diffusion("flux_dev"),
        "flux_schnell": build_diffusion("flux_schnell"),
    }

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention_backend": ATTENTION_BACKEND,
            "rope_backend": ROPE_BACKEND,
            **tuple_provenance(torch.__version__),
        },
        "arch": {
            "flux": dict(TINY_FLUX),
            "clip_l": dict(TINY_CLIP),
            "t5xxl": dict(TINY_T5),
            "vae_ddconfig": dict(TINY_VAE_DDCONFIG),
            "vae_embed_dim": TINY_VAE_EMBED_DIM,
        },
        "cases": [],
    }

    for case in CASES:
        payload["cases"].append(
            run_case(case, diffusion[case["family"]], text_model, tokenizer, vae)
        )
        print(f"ran {case['name']}", file=sys.stderr)

    # The composed text encode, pinned once with full values: the
    # FluxClipModel split (T5 sequence as cond, CLIP-L pooled as y)
    # over the real tokenizers.
    tokens = tokenizer.tokenize_with_weights("a photo of a cat", return_word_ids=False)
    with torch.no_grad():
        cond, pooled = text_model.encode_token_weights(tokens)
    payload["encode_text"] = {
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
