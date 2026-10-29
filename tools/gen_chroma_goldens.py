"""Generate Chroma and Chroma Radiance goldens from pinned ComfyUI.

The executed cases retain the production 19 double and 38 single block
topology while reducing tensor widths. Attention is forced to PyTorch SDPA
and RoPE to the pure-torch path.

Run from the Dinkster root with the audited ComfyUI checkout selected by
``DINKSTER_COMFYUI_ROOT`` when it is not available at ``../ComfyUI``::

    DINKSTER_COMFYUI_ROOT=/path/to/ComfyUI .venv-gpu/bin/python \
        tools/gen_chroma_goldens.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("DINKSTER_COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
REFERENCE_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
GENERATOR_TORCH = "2.13.0+cu130"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens require torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
    )

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.chroma.model import Chroma  # noqa: E402
from comfy.ldm.chroma_radiance.model import ChromaRadiance  # noqa: E402
from comfy.ldm.flux import math as flux_math  # noqa: E402
from comfy.ldm.modules import attention  # noqa: E402
from golden_platform import tuple_provenance  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

flux_math.optimized_attention = attention.attention_pytorch
attention.optimized_attention = attention.attention_pytorch
attention.optimized_attention_masked = attention.attention_pytorch
assert flux_math.attention.__globals__["optimized_attention"] is attention.attention_pytorch
model_management.in_training = True

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "chroma_goldens.json"

_COMMON = {
    "context_in_dim": 4096,
    "hidden_size": 3072,
    "mlp_ratio": 4.0,
    "num_heads": 24,
    "depth": 19,
    "depth_single_blocks": 38,
    "axes_dim": [16, 56, 56],
    "theta": 10000,
    "qkv_bias": True,
    "in_dim": 64,
    "out_dim": 3072,
    "hidden_dim": 5120,
    "n_layers": 5,
    "txt_ids_dims": [],
    "vec_in_dim": None,
}

FULL_CONFIGS = {
    "chroma": {
        **_COMMON,
        "in_channels": 64,
        "out_channels": 64,
        "patch_size": 2,
    },
    "radiance_linear": {
        **_COMMON,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": 16,
        "nerf_hidden_size": 64,
        "nerf_mlp_ratio": 4,
        "nerf_depth": 4,
        "nerf_max_freqs": 8,
        "nerf_tile_size": 512,
        "nerf_final_head_type": "linear",
        "use_x0": False,
        "use_sequential_txt_ids": False,
    },
    "radiance_conv_x0": {
        **_COMMON,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": 16,
        "nerf_hidden_size": 64,
        "nerf_mlp_ratio": 4,
        "nerf_depth": 4,
        "nerf_max_freqs": 8,
        "nerf_tile_size": 512,
        "nerf_final_head_type": "conv",
        "use_x0": True,
        "use_sequential_txt_ids": False,
    },
    "radiance_sequential": {
        **_COMMON,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": 16,
        "nerf_hidden_size": 64,
        "nerf_mlp_ratio": 4,
        "nerf_depth": 4,
        "nerf_max_freqs": 8,
        "nerf_tile_size": 512,
        "nerf_final_head_type": "linear",
        "use_x0": False,
        "use_sequential_txt_ids": True,
    },
}

_TINY_COMMON = {
    "context_in_dim": 24,
    "hidden_size": 32,
    "mlp_ratio": 2.0,
    "num_heads": 2,
    "depth": 19,
    "depth_single_blocks": 38,
    "axes_dim": [4, 6, 6],
    "theta": 10000,
    "qkv_bias": True,
    "in_dim": 64,
    "out_dim": 32,
    "hidden_dim": 16,
    "n_layers": 1,
    "txt_ids_dims": [],
    "vec_in_dim": None,
}

CASES = {
    "chroma": {
        "kind": "chroma",
        "config": {
            **_TINY_COMMON,
            "in_channels": 8,
            "out_channels": 8,
            "patch_size": 2,
        },
        "batch": 2,
        "height": 3,
        "width": 5,
        "context_len": 3,
    },
    "radiance_linear": {
        "kind": "radiance",
        "config": {
            **_TINY_COMMON,
            "in_channels": 3,
            "out_channels": 3,
            "patch_size": 2,
            "nerf_hidden_size": 4,
            "nerf_mlp_ratio": 2,
            "nerf_depth": 2,
            "nerf_max_freqs": 2,
            "nerf_tile_size": 0,
            "nerf_final_head_type": "linear",
            "use_x0": False,
            "use_sequential_txt_ids": False,
        },
        "batch": 2,
        "height": 3,
        "width": 5,
        "context_len": 3,
    },
    "radiance_conv_x0": {
        "kind": "radiance",
        "config": {
            **_TINY_COMMON,
            "in_channels": 3,
            "out_channels": 3,
            "patch_size": 2,
            "nerf_hidden_size": 4,
            "nerf_mlp_ratio": 2,
            "nerf_depth": 2,
            "nerf_max_freqs": 2,
            "nerf_tile_size": 1,
            "nerf_final_head_type": "conv",
            "use_x0": True,
            "use_sequential_txt_ids": True,
        },
        "batch": 1,
        "height": 4,
        "width": 4,
        "context_len": 4,
    },
}


def _reference(kind: str, config: dict[str, Any], device: str) -> torch.nn.Module:
    arguments = dict(config)
    if kind == "radiance":
        arguments["nerf_embedder_dtype"] = torch.float32
        model = ChromaRadiance
    else:
        model = Chroma
    return model(
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
        **arguments,
    )


def _encoded(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    commit = _git("rev-parse", "HEAD")
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"{COMFY_ROOT} must be clean:\n{dirty}")
    for model in (Chroma, ChromaRadiance):
        module_file = Path(sys.modules[model.__module__].__file__ or "").resolve()
        if not module_file.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"reference module came from {module_file}, not {COMFY_ROOT}")

    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": "attention_pytorch",
            "rope": "_apply_rope (pure torch, in_training=True)",
            **tuple_provenance(torch.__version__, pin_cpu=True),
        },
        "layouts": {},
        "cases": {},
    }
    for name, config in FULL_CONFIGS.items():
        kind = "chroma" if name == "chroma" else "radiance"
        model = _reference(kind, config, "meta")
        payload["layouts"][name] = sorted(
            (key, list(value.shape)) for key, value in model.state_dict().items()
        )

    for name, case in sorted(CASES.items()):
        config = case["config"]
        model = _reference(case["kind"], config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)
        channels = config["in_channels"] // config["patch_size"] ** 2
        if case["kind"] == "radiance":
            channels = config["in_channels"]
        x = hashed_input(f"{name}:x", (case["batch"], channels, case["height"], case["width"]))
        timestep = torch.linspace(0.25, 0.75, case["batch"], dtype=torch.float32)
        context = hashed_input(
            f"{name}:context",
            (case["batch"], case["context_len"], config["context_in_dim"]),
        )
        guidance = torch.linspace(2.0, 4.0, case["batch"], dtype=torch.float32)
        outputs: dict[str, Any] = {}
        with torch.no_grad():
            outputs["default"] = _encoded(model(x, timestep, context, guidance))
            if case["kind"] == "radiance" and not config["use_x0"]:
                outputs["tiled"] = _encoded(
                    model(
                        x,
                        timestep,
                        context,
                        guidance,
                        transformer_options={"chroma_radiance_options": {"nerf_tile_size": 1}},
                    )
                )
                outputs["sequential"] = _encoded(
                    model(
                        x,
                        timestep,
                        context,
                        guidance,
                        transformer_options={
                            "chroma_radiance_options": {"use_sequential_txt_ids": True}
                        },
                    )
                )
        payload["cases"][name] = {
            **case,
            "state_dict": entries,
            "timestep": timestep.tolist(),
            "guidance": guidance.tolist(),
            "outputs": outputs,
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
