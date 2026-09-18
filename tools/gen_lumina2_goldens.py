"""Generate Lumina Image 2.0 diffusion goldens from ComfyUI.

Runs the reference ``NextDiT`` at the audited baseline and writes the
full production state layout plus reduced deterministic forward cases.

Usage:

    COMFYUI_ROOT=/path/to/pinned/ComfyUI \
      .venv-torch/bin/python tools/gen_lumina2_goldens.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

GENERATOR_TORCH = "2.13.0+cpu"
if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
    )

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.lumina import model as lumina_model  # noqa: E402
from comfy.ldm.lumina.model import NextDiT  # noqa: E402
from comfy.ldm.modules import attention as attention  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

lumina_model.optimized_attention_masked = attention.attention_pytorch
model_management.in_training = True

ATTENTION_BACKEND = "attention_pytorch"
ROPE_BACKEND = "apply_rope (pure torch, in_training=True)"
OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "lumina2_goldens.json"

FULL_CONFIG = {
    "patch_size": 2,
    "in_channels": 16,
    "dim": 2304,
    "n_layers": 26,
    "n_refiner_layers": 2,
    "n_heads": 24,
    "n_kv_heads": 8,
    "multiple_of": 256,
    "ffn_dim_multiplier": 4.0,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2304,
    "axes_dims": [32, 32, 32],
    "axes_lens": [300, 512, 512],
    "rope_theta": 10000.0,
    "z_image_modulation": False,
    "time_scale": 1.0,
    "pad_tokens_multiple": None,
}

TINY_CONFIG = {
    "patch_size": 2,
    "in_channels": 4,
    "dim": 32,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 1,
    "multiple_of": 8,
    "ffn_dim_multiplier": 2.0,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 24,
    "axes_dims": [4, 6, 6],
    "axes_lens": [32, 32, 32],
    "rope_theta": 10000.0,
    "z_image_modulation": False,
    "time_scale": 1.0,
    "pad_tokens_multiple": None,
}

# name -> (batch, height, width, context tokens)
CASES = {
    "lumina2_even": (2, 6, 8, 5),
    "lumina2_odd": (1, 5, 7, 4),
}


def build_reference(config: dict[str, Any], device: str) -> NextDiT:
    return NextDiT(
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
        **config,
    )


def encode(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def capture_output(observed: dict[str, torch.Tensor], name: str):  # noqa: ANN202
    def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
        observed[name] = output

    return hook


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
            f"{COMFY_ROOT} is at {commit}; expected audited baseline {REFERENCE_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"{COMFY_ROOT} must be clean:\n{dirty}")
    module_file = Path(sys.modules[NextDiT.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"reference imported from {module_file}, not {COMFY_ROOT}")

    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
            "rope": ROPE_BACKEND,
        },
        "layouts": {},
        "cases": {},
    }
    full = build_reference(FULL_CONFIG, "meta")
    payload["layouts"]["lumina2"] = sorted(
        (key, list(value.shape)) for key, value in full.state_dict().items()
    )

    for name, (batch, height, width, context_tokens) in sorted(CASES.items()):
        model = build_reference(TINY_CONFIG, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)
        latent = hashed_input(
            f"{name}:latent",
            (batch, TINY_CONFIG["in_channels"], height, width),
        )
        context = hashed_input(
            f"{name}:context",
            (batch, context_tokens, TINY_CONFIG["cap_feat_dim"]),
        )
        timesteps = torch.linspace(0.05, 0.85, batch, dtype=torch.float32)
        observed: dict[str, torch.Tensor] = {}
        hooks = [
            model.context_refiner[0].register_forward_hook(capture_output(observed, "context")),
            model.noise_refiner[0].register_forward_hook(capture_output(observed, "noise")),
            model.layers[0].register_forward_hook(capture_output(observed, "main")),
        ]
        try:
            with torch.no_grad():
                output = model(
                    latent,
                    timesteps,
                    context,
                    num_tokens=context_tokens,
                )
        finally:
            for hook in hooks:
                hook.remove()
        payload["cases"][name] = {
            "config": TINY_CONFIG,
            "batch": batch,
            "height": height,
            "width": width,
            "context_tokens": context_tokens,
            "timesteps": timesteps.tolist(),
            "state_dict": entries,
            "block_outputs": {key: encode(value) for key, value in observed.items()},
            "output": encode(output),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
