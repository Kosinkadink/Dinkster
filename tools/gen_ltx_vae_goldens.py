"""Generate LTX causal video VAE goldens from ComfyUI.

Runs the REFERENCE VideoVAE (comfy/ldm/lightricks/vae/
causal_video_autoencoder.py @ the audited baseline) and writes
packages/dinkster-inference-torch/tests/goldens/ltx_vae_goldens.json.
dinkster_inference.ltx (ltxv_vae_layout) and
dinkster_inference_torch.ltx_video_vae are pinned against these outputs;
the oracle is the executed reference, never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE 2B v0.9, 2B v0.9.5, and LTX-2.3 22B VAEs, built on the meta device from
  each checkpoint's verbatim metadata "config".vae dict (read from
  the published safetensors headers), exactly as the reference loads
  them (comfy/sd.py prefers the embedded config over its per-version
  defaults).
- "cases": tiny VAE architectures executed with deterministic
  hash-filled weights (ltx_vae_fill.py) covering the v0.9 shape
  (shared block list, res_x_y channel doubling, plain compress
  convolutions) and the v0.9.5 shape (space-to-depth encoder
  compression, residual depth-to-space decoder upsampling, noise
  injection, timestep conditioning). Every case records encode and
  decode BOTH single-shot and streamed: the reference's own streamed
  decode is not bit-equal to its single-shot decode, so the parity
  contract is per-mode against these goldens, never across modes.

Streaming is driven by monkeypatching the module-level
``get_max_chunk_size`` (the reference reads total device memory
there); the recorded budget equals the ``max_chunk_bytes`` argument
of the Dinkster port, which applies the reference's internal encoder
doubling itself. Decode noise (the timestep-conditioned latent mix
and the inject-noise draws) comes from torch's global RNG in the
reference, so the conditioned case seeds the global RNG immediately
before every decode and records the seed.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_ltx_vae_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
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

# comfy.model_management probes CUDA at import; goldens execute on
# CPU float32 either way, so force ComfyUI's CPU state and keep the
# generator runnable from a CPU-only torch venv. The reference only
# reads argv when args parsing is explicitly enabled.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

#: Bit-stability holds only for one interpreter: generation REFUSES
#: any other torch build so a regeneration cannot silently rotate the
#: payload hash through CPU-kernel drift.
GENERATOR_TORCH = "2.13.0+cpu"
if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has"
        f" {torch.__version__}. Regenerating on another build rotates the payload"
        " hash - update the pin deliberately and re-prove bit-stability."
    )

from comfy.ldm.lightricks.vae import causal_video_autoencoder as ref_vae  # noqa: E402
from ltx_vae_fill import fill_vae_state_dict  # noqa: E402
from unet_fill import hashed_input  # noqa: E402

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "ltx_vae_goldens.json"

#: The verbatim metadata "config".vae dicts of the two published 2B
#: checkpoints (safetensors headers of ltx-video-2b-v0.9.safetensors
#: and ltx-video-2b-v0.9.5.safetensors). The v0.9 shorthand
#: ``["res_x_y", 1]`` names one block whose channel multiplier is the
#: reference default, 2.
_2B_V09_METADATA = {
    "_class_name": "CausalVideoAutoencoder",
    "blocks": [
        ["res_x", 4],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x", 3],
        ["res_x", 4],
    ],
    "causal_decoder": False,
    "dims": 3,
    "in_channels": 3,
    "latent_channels": 128,
    "latent_log_var": "uniform",
    "norm_layer": "pixel_norm",
    "out_channels": 3,
    "patch_size": 4,
    "scaling_factor": 1.0,
    "use_quant_conv": False,
}

_2B_V095_METADATA = {
    "_class_name": "CausalVideoAutoencoder",
    "causal_decoder": False,
    "decoder_blocks": [
        ["res_x", {"inject_noise": False, "num_layers": 5}],
        ["compress_all", {"multiplier": 2, "residual": True}],
        ["res_x", {"inject_noise": False, "num_layers": 5}],
        ["compress_all", {"multiplier": 2, "residual": True}],
        ["res_x", {"inject_noise": False, "num_layers": 5}],
        ["compress_all", {"multiplier": 2, "residual": True}],
        ["res_x", {"inject_noise": False, "num_layers": 5}],
    ],
    "dims": 3,
    "encoder_blocks": [
        ["res_x", {"num_layers": 4}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
    ],
    "in_channels": 3,
    "latent_channels": 128,
    "latent_log_var": "uniform",
    "norm_layer": "pixel_norm",
    "normalize_latent_channels": False,
    "out_channels": 3,
    "patch_size": 4,
    "scaling_factor": 1.0,
    "timestep_conditioning": True,
    "use_quant_conv": False,
}

_22B_V23_METADATA = {
    "_class_name": "CausalVideoAutoencoder",
    "causal_decoder": False,
    "decoder_base_channels": 128,
    "decoder_blocks": [
        ["res_x", {"num_layers": 4}],
        ["compress_space", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_time", {"multiplier": 2}],
        ["res_x", {"num_layers": 4}],
        ["compress_all", {"multiplier": 1}],
        ["res_x", {"num_layers": 2}],
        ["compress_all", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
    ],
    "dims": 3,
    "encoder_base_channels": 128,
    "encoder_blocks": [
        ["res_x", {"num_layers": 4}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 4}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
        ["compress_all_res", {"multiplier": 1}],
        ["res_x", {"num_layers": 2}],
    ],
    "in_channels": 3,
    "latent_channels": 128,
    "latent_log_var": "uniform",
    "norm_layer": "pixel_norm",
    "normalize_latent_channels": False,
    "out_channels": 3,
    "patch_size": 4,
    "scaling_factor": 1.0,
    "spatial_padding_mode": "zeros",
    "timestep_conditioning": False,
    "use_quant_conv": False,
}

FULL_CONFIGS = {
    "ltxv_2b_v09_vae": _2B_V09_METADATA,
    "ltxv_2b_v095_vae": _2B_V095_METADATA,
    "ltxav_22b_v23_vae": _22B_V23_METADATA,
}

#: Tiny geometries for executed parity, in the reference config
#: vocabulary and in the Dinkster LTXVideoVAEConfig vocabulary side by
#: side (the replay tests rebuild LTXVideoVAEConfig from "config").
_TINY_SHARED = {
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "encoder_base_channels": 8,
    "decoder_base_channels": 8,
    "patch_size": 2,
    "norm_layer": "pixel_norm",
    "latent_log_var": "uniform",
    "use_quant_conv": False,
    "causal_decoder": False,
}

_TINY_V0_BLOCKS = [
    ["res_x", 1],
    ["compress_all", 1],
    ["res_x_y", {"multiplier": 2}],
    ["res_x", 1],
]

_TINY_V0_REFERENCE = {**_TINY_SHARED, "blocks": _TINY_V0_BLOCKS}

_TINY_V0_DINKSTER_BLOCKS = [
    {"kind": "res_x", "layers": 1},
    {"kind": "compress_all", "multiplier": 1},
    {"kind": "res_x_y", "multiplier": 2},
    {"kind": "res_x", "layers": 1},
]

_TINY_V0_DINKSTER = {
    "encoder_blocks": _TINY_V0_DINKSTER_BLOCKS,
    "decoder_blocks": _TINY_V0_DINKSTER_BLOCKS,
    "latent_channels": 4,
    "base_channels": 8,
    "patch_size": 2,
    "timestep_conditioning": False,
}

_TINY_V095_REFERENCE = {
    **_TINY_SHARED,
    "timestep_conditioning": True,
    "encoder_blocks": [
        ["res_x", {"num_layers": 2}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
    ],
    "decoder_blocks": [
        ["res_x", {"num_layers": 2, "inject_noise": True}],
        ["compress_all", {"multiplier": 2, "residual": True}],
        ["res_x", {"num_layers": 1, "inject_noise": True}],
        ["compress_all", {"multiplier": 2, "residual": True}],
        ["res_x", {"num_layers": 2}],
    ],
}

_TINY_V095_DINKSTER = {
    "encoder_blocks": [
        {"kind": "res_x", "layers": 2},
        {"kind": "compress_space_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_time_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_all_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
    ],
    "decoder_blocks": [
        {"kind": "res_x", "layers": 2, "inject_noise": True},
        {"kind": "compress_all", "multiplier": 2, "residual": True},
        {"kind": "res_x", "layers": 1, "inject_noise": True},
        {"kind": "compress_all", "multiplier": 2, "residual": True},
        {"kind": "res_x", "layers": 2},
    ],
    "latent_channels": 4,
    "base_channels": 8,
    "patch_size": 2,
    "timestep_conditioning": True,
}

_TINY_V23_REFERENCE = {
    **_TINY_SHARED,
    "spatial_padding_mode": "zeros",
    "encoder_blocks": [
        ["res_x", {"num_layers": 1}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_all_res", {"multiplier": 1}],
        ["res_x", {"num_layers": 1}],
    ],
    "decoder_blocks": [
        ["res_x", {"num_layers": 1}],
        ["compress_space", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_time", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
        ["compress_all", {"multiplier": 1}],
        ["res_x", {"num_layers": 1}],
        ["compress_all", {"multiplier": 2}],
        ["res_x", {"num_layers": 1}],
    ],
}

_TINY_V23_DINKSTER = {
    "encoder_blocks": [
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_space_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_time_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_all_res", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_all_res", "multiplier": 1},
        {"kind": "res_x", "layers": 1},
    ],
    "decoder_blocks": [
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_space", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_time", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_all", "multiplier": 1},
        {"kind": "res_x", "layers": 1},
        {"kind": "compress_all", "multiplier": 2},
        {"kind": "res_x", "layers": 1},
    ],
    "latent_channels": 4,
    "base_channels": 8,
    "patch_size": 2,
    "timestep_conditioning": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "zeros",
}

#: name -> (reference config, Dinkster config, input shape, seed).
#: T=9 satisfies 1 + k*temporal_ratio for both cases, so neither side
#: truncates frames (the reference hardcodes ratio 8 in its
#: truncation; the Dinkster port derives it from the block list).
CASES = {
    "vae_v0": (_TINY_V0_REFERENCE, _TINY_V0_DINKSTER, (1, 3, 9, 16, 16), None),
    "vae_v095_conditioned": (_TINY_V095_REFERENCE, _TINY_V095_DINKSTER, (1, 3, 9, 16, 16), 42),
    "vae_v23": (_TINY_V23_REFERENCE, _TINY_V23_DINKSTER, (1, 3, 9, 16, 16), None),
}

#: Streaming budgets, in the units of the Dinkster port's
#: ``max_chunk_bytes`` (== the reference ``get_max_chunk_size``
#: return). The single-shot budget makes every chunk loop degenerate;
#: 4096 forces multi-chunk streaming at the tiny geometry.
BUDGETS = {"single": 1 << 40, "streamed": 4096}


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def build_reference(config: dict, device: str) -> ref_vae.VideoVAE:
    if device == "meta":
        with torch.device("meta"):
            return ref_vae.VideoVAE(config=config)
    model = ref_vae.VideoVAE(config=config)
    # The reference processor hardcodes the published latent width;
    # tiny cases re-register the dataset statistics at their width.
    channels = config["latent_channels"]
    if channels != 128:
        model.per_channel_statistics.register_buffer("std-of-means", torch.empty(channels))
        model.per_channel_statistics.register_buffer("mean-of-means", torch.empty(channels))
    return model


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
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"{COMFY_ROOT} has uncommitted changes; a clean checkout"
            f" of {REFERENCE_COMMIT} is required:\n{dirty}"
        )
    module_file = Path(ref_vae.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference VideoVAE was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "budgets": BUDGETS,
        "layouts": {},
        "cases": {},
    }
    for name, config in FULL_CONFIGS.items():
        model = build_reference(config, "meta")
        payload["layouts"][name] = sorted(
            (key, list(value.shape)) for key, value in model.state_dict().items()
        )

    for name, (reference_config, dinkster_config, input_shape, seed) in sorted(CASES.items()):
        model = build_reference(reference_config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_vae_state_dict(entries), strict=True)

        x = hashed_input(f"{name}:x", input_shape)
        encoded: dict[str, object] = {}
        decoded: dict[str, object] = {}
        latent_shape: list[int] | None = None
        for mode, budget in BUDGETS.items():
            ref_vae.get_max_chunk_size = lambda device, budget=budget: budget
            with torch.no_grad():
                latent = model.encode(x)
            encoded[mode] = enc(latent)
            latent_shape = list(latent.shape)
            latent_input = hashed_input(f"{name}:latent", latent_shape)
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                pixels = model.decode(latent_input)
            decoded[mode] = enc(pixels)
        assert latent_shape is not None

        payload["cases"][name] = {
            "config": dinkster_config,
            "reference_config": reference_config,
            "state_dict": entries,
            "input_shape": list(input_shape),
            "latent_shape": latent_shape,
            "decode_output_shape": list(model.decode_output_shape(latent_shape)),
            "seed": seed,
            "encode": encoded,
            "decode": decoded,
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
