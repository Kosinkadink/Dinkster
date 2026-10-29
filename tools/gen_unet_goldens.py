"""Generate SD1/SDXL diffusion UNet goldens from the ComfyUI reference.

Runs the REFERENCE UNetModel
(comfy/ldm/modules/diffusionmodules/openaimodel.py) @ the audited
baseline and writes
packages/dinkster-inference-torch/tests/goldens/unet_goldens.json.
dinkster_inference_torch.unet is pinned against these outputs; the
oracle is the executed reference, never a re-derivation.

Two payloads:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE SD1.5, SDXL base, and SDXL refiner UNets, built from the
  reference's own detection listings
  (comfy/model_detection.py unet_config_from_diffusers_unet
  SD15/SDXL/SDXL_refiner @ 947c2749) on the meta device (weights
  never materialize). These pin torch-free detection and layout
  generation.
- "cases": tiny architectures executed with deterministic
  hash-filled weights (unet_fill.py, shared with the replay tests).
  GroupNorm(32) pins every width to a multiple of 32, so the tiny
  models use model_channels=32. Cases cover the SD1-style conv
  transformer projections, the SDXL-style linear projections with
  ADM ("sequential" class embedding), and odd spatial extents (the
  Upsample output_shape leg).

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_unet_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_REFERENCE", REPO.parent / "ComfyUI")).resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402
from comfy.cli_args import args  # noqa: E402

args.cpu = True

from comfy import ops  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy.ldm.modules.diffusionmodules.openaimodel import (  # noqa: E402
    UNetModel,
)
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# The reference CrossAttention calls the ambient optimized_attention,
# which comfy selects per environment (xformers/sage/flash/pytorch).
# Goldens must not depend on which accelerators happen to be
# installed: force the pytorch SDPA backend, the one Dinkster ports.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
ATTENTION_BACKEND = "attention_pytorch"

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "unet_goldens.json"

#: Full-size configs, verbatim from the reference's pinned listings
#: (comfy/model_detection.py unet_config_from_diffusers_unet @
#: 947c2749), in the kwarg shape detect_unet_config produces.
FULL_CONFIGS = {
    "sd15": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "transformer_depth_middle": 1,
        "context_dim": 768,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "num_heads": 8,
        "num_head_channels": -1,
    },
    "sdxl": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "channel_mult": [1, 2, 4],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "transformer_depth_middle": 10,
        "context_dim": 2048,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2816,
        "num_heads": -1,
        "num_head_channels": 64,
    },
    "sdxl_refiner": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 384,
        "num_res_blocks": [2, 2, 2, 2],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "transformer_depth_output": [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
        "transformer_depth_middle": 4,
        "context_dim": 1280,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2560,
        "num_heads": -1,
        "num_head_channels": 64,
    },
}
FULL_CONFIGS["sd15_inpaint"] = {**FULL_CONFIGS["sd15"], "in_channels": 9}
FULL_CONFIGS["sdxl_inpaint"] = {**FULL_CONFIGS["sdxl"], "in_channels": 9}

_TINY_SD1 = {
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 32,
    "num_res_blocks": [1, 1],
    "channel_mult": [1, 2],
    "transformer_depth": [1, 1],
    "transformer_depth_output": [1, 1, 1, 1],
    "transformer_depth_middle": 1,
    "context_dim": 16,
    "use_linear_in_transformer": False,
    "adm_in_channels": None,
    "num_heads": 8,
    "num_head_channels": -1,
}
_TINY_SD15_INPAINT = {**_TINY_SD1, "in_channels": 9}

_TINY_XL = {
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 32,
    "num_res_blocks": [1, 1],
    "channel_mult": [1, 2],
    "transformer_depth": [0, 2],
    "transformer_depth_output": [0, 0, 2, 2],
    "transformer_depth_middle": 2,
    "context_dim": 24,
    "use_linear_in_transformer": True,
    "adm_in_channels": 12,
    "num_heads": -1,
    "num_head_channels": 16,
}
_TINY_SDXL_INPAINT = {**_TINY_XL, "in_channels": 9}

#: name -> (config, batch, height, width, context_len). Odd spatial
#: extents exercise the Upsample output_shape override.
CASES = {
    "sd1_conv": (_TINY_SD1, 2, 16, 16, 7),
    "sd15_inpaint": (_TINY_SD15_INPAINT, 1, 8, 8, 5),
    "sd1_odd_spatial": (_TINY_SD1, 1, 15, 10, 5),
    "xl_linear_adm": (_TINY_XL, 2, 16, 16, 6),
    "xl_wide_inpaint": (_TINY_SDXL_INPAINT, 1, 8, 8, 5),
}


def build_reference(config: dict, device: str) -> UNetModel:
    return UNetModel(
        image_size=32,
        dims=2,
        use_spatial_transformer=True,
        legacy=False,
        num_classes=("sequential" if config["adm_in_channels"] is not None else None),
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
        **config,
    )


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


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
    module_file = Path(sys.modules[UNetModel.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference UNetModel was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
        },
        "layouts": {},
        "cases": {},
    }
    for name, config in FULL_CONFIGS.items():
        model = build_reference(config, "meta")
        payload["layouts"][name] = sorted(
            (key, list(value.shape)) for key, value in model.state_dict().items()
        )

    for name, (config, batch, height, width, context_len) in sorted(CASES.items()):
        model = build_reference(config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        x = hashed_input(f"{name}:x", (batch, config["in_channels"], height, width))
        timesteps = torch.linspace(3.0, 999.0, batch, dtype=torch.float32)
        context = hashed_input(f"{name}:context", (batch, context_len, config["context_dim"]))
        y = None
        if config["adm_in_channels"] is not None:
            y = hashed_input(f"{name}:y", (batch, config["adm_in_channels"]))
        with torch.no_grad():
            out = model(x, timesteps=timesteps, context=context, y=y)
        payload["cases"][name] = {
            "config": config,
            "state_dict": entries,
            "batch": batch,
            "height": height,
            "width": width,
            "context_len": context_len,
            "timesteps": timesteps.tolist(),
            "output": enc(out),
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
