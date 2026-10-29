"""Generate SeedVR2 DiT and VAE goldens from the pinned ComfyUI reference.

The executed cases use deterministic name-hashed weights from ``unet_fill``.
Set ``SEEDVR2_COMFY_ROOT`` when the audited ComfyUI checkout is not the
repository's sibling ``ComfyUI`` directory.

Usage:

    SEEDVR2_COMFY_ROOT=/path/to/ComfyUI \
      PYTHONPATH=/path/to/comfy-aimdo \
      .venv-torch/bin/python tools/gen_seedvr2_goldens.py

Darwin fixtures use Python 3.12.11 and torch 2.13.0 in a platform-tuple file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("SEEDVR2_COMFY_ROOT", REPO.parent / "ComfyUI")).resolve()
REFERENCE_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
GENERATOR_TORCH = "2.13.0" if sys.platform == "darwin" else "2.13.0+cpu"

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
from comfy.ldm.modules import attention as reference_attention  # noqa: E402
from comfy.ldm.modules.diffusionmodules.model import pytorch_attention  # noqa: E402
from comfy.ldm.seedvr import attention as seedvr_attention  # noqa: E402
from comfy.ldm.seedvr import model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr import vae as seedvr_vae  # noqa: E402
from comfy.ldm.seedvr.model import NaDiT  # noqa: E402
from comfy.ldm.seedvr.vae import Attention, VideoAutoencoderKLWrapper  # noqa: E402
from comfy_extras import nodes_seedvr  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from unet_fill import fill_value, hashed_input  # noqa: E402

reference_attention.optimized_attention = reference_attention.attention_pytorch
seedvr_attention._attention.optimized_attention = reference_attention.attention_pytorch
seedvr_vae.optimized_attention = reference_attention.attention_pytorch
model_management.in_training = True

OUT = platform_golden_path(
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "seedvr2_goldens.json",
    torch.__version__,
)

FULL_CONFIGS: dict[str, dict[str, Any]] = {
    "3b_swiglu": {
        "norm_eps": 1e-5,
        "num_layers": 32,
        "mlp_type": "swiglu",
        "vid_dim": 2560,
        "heads": 20,
        "head_dim": 128,
        "mm_layers": 10,
        "rope_dim": 128,
        "rope_type": "mmrope3d",
        "vid_out_norm": "rms",
    },
    "7b_swiglu": {
        "norm_eps": 1e-5,
        "num_layers": 36,
        "mlp_type": "swiglu",
        "vid_dim": 3072,
        "heads": 24,
        "head_dim": 128,
        "mm_layers": 10,
        "rope_dim": 64,
        "rope_type": "rope3d",
    },
    "7b_mlp": {
        "norm_eps": 1e-5,
        "num_layers": 36,
        "mlp_type": "normal",
        "vid_dim": 3072,
        "heads": 24,
        "head_dim": 128,
        "mm_layers": 36,
        "rope_dim": 64,
        "rope_type": "rope3d",
    },
}

TINY_BASE: dict[str, Any] = {
    "norm_eps": 1e-5,
    "num_layers": 2,
    "vid_dim": 32,
    "txt_in_dim": 24,
    "heads": 2,
    "head_dim": 16,
    "rope_dim": 12,
}

CASES: dict[str, dict[str, Any]] = {
    "3b_image": {
        "config": {**TINY_BASE, "mlp_type": "swiglu", "mm_layers": 1, "vid_out_norm": "rms"},
        "seven_b": False,
        "batch": 1,
        "frames": 1,
        "cond_or_uncond": ["positive"],
    },
    "3b_cfg_video": {
        "config": {**TINY_BASE, "mlp_type": "swiglu", "mm_layers": 1, "vid_out_norm": "rms"},
        "seven_b": False,
        "batch": 2,
        "frames": 2,
        "cond_or_uncond": [0, 1],
    },
    "7b_swiglu_video": {
        "config": {**TINY_BASE, "mlp_type": "swiglu", "mm_layers": 1},
        "seven_b": True,
        "batch": 1,
        "frames": 2,
        "cond_or_uncond": ["positive"],
    },
    "7b_mlp_image": {
        "config": {**TINY_BASE, "mlp_type": "normal", "mm_layers": 2},
        "seven_b": True,
        "batch": 1,
        "frames": 1,
        "cond_or_uncond": ["positive"],
    },
}


def enc(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.detach().to(torch.float32).flatten().tolist(),
    }


def fill_module(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    entries = sorted((key, list(value.shape)) for key, value in module.state_dict().items())
    with torch.no_grad():
        for key, value in module.state_dict().items():
            value.copy_(fill_value(key, value.shape))
    return entries


def build_dit(config: dict[str, Any], device: str) -> NaDiT:
    return NaDiT(
        device=device,
        dtype=torch.float32,
        operations=ops.disable_weight_init,
        **config,
    )


def reference_commit() -> str:
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
        raise SystemExit(f"the reference checkout must be clean:\n{dirty}")
    module_file = Path(sys.modules[NaDiT.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"SeedVR2 reference imported from {module_file}, not {COMFY_ROOT}")
    return commit


def execute_dit_cases(payload: dict[str, Any]) -> None:
    published_7b_width = seedvr_model.SEEDVR2_7B_VID_DIM
    for name, spec in CASES.items():
        seedvr_model.SEEDVR2_7B_VID_DIM = 32 if spec["seven_b"] else published_7b_width
        model = build_dit(spec["config"], "cpu")
        entries = fill_module(model)
        batch = spec["batch"]
        frames = spec["frames"]
        latent = hashed_input(f"{name}:latent", (batch, 16, frames, 4, 4))
        condition = hashed_input(f"{name}:condition", (batch, 17, frames, 4, 4))
        context = hashed_input(f"{name}:context", (batch, 3, 24))
        timestep = torch.linspace(0.1, 0.9, batch, dtype=torch.float32)
        observed: dict[str, torch.Tensor] = {}

        def record_block(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: tuple[torch.Tensor, ...],
            record: dict[str, torch.Tensor] = observed,
        ) -> None:
            record["block0_vid"] = output[0]
            record["block0_txt"] = output[1]

        hook = model.blocks[0].register_forward_hook(record_block)
        try:
            output = model(
                latent,
                timestep,
                context,
                condition=condition,
                transformer_options={"cond_or_uncond": spec["cond_or_uncond"]},
            )
        finally:
            hook.remove()
        payload["cases"][name] = {
            **spec,
            "state_dict": entries,
            "timestep": timestep.tolist(),
            "block0_vid": enc(observed["block0_vid"]),
            "block0_txt": enc(observed["block0_txt"]),
            "output": enc(output),
        }
    seedvr_model.SEEDVR2_7B_VID_DIM = published_7b_width


def execute_vae_cases(payload: dict[str, Any]) -> None:
    model = VideoAutoencoderKLWrapper()
    entries = fill_module(model)
    for module in model.modules():
        if isinstance(module, Attention):
            module.optimized_vae_attention = pytorch_attention
    cases: dict[str, object] = {}
    for name, frames in (("image", 1), ("video", 5)):
        content = hashed_input(f"vae_{name}:content", (1, 3, frames, 8, 8))
        with torch.no_grad():
            latent = model.encode(content)
            decoded = model.decode(latent)
        cases[name] = {
            "frames": frames,
            "content": enc(content),
            "latent": enc(latent),
            "decoded": enc(decoded),
        }
    tiled_content = hashed_input("vae_tiled:content", (1, 3, 5, 16, 16))
    with torch.no_grad():
        tiled_latent = model.encode_tiled(tiled_content, tile_x=8, tile_y=8, overlap=0)
        tiled_decoded = model.decode_tiled(
            tiled_latent,
            tile_x=1,
            tile_y=1,
            overlap=0,
        )
    payload["vae"] = {
        "state_dict": entries,
        "cases": cases,
        "tiled": {
            "content": enc(tiled_content),
            "latent": enc(tiled_latent),
            "decoded": enc(tiled_decoded),
        },
    }


def execute_window_cases(payload: dict[str, Any]) -> None:
    shape = (40, 90, 160)
    num_windows = (4, 3, 3)

    def encode_slices(windows: list[tuple[slice, slice, slice]]) -> list[list[list[int]]]:
        return [[[axis.start or 0, axis.stop or 0] for axis in window] for window in windows]

    payload["windows"] = {
        "shape": list(shape),
        "num_windows": list(num_windows),
        "regular": encode_slices(seedvr_model.make_720Pwindows_bysize(shape, num_windows)),
        "shifted": encode_slices(seedvr_model.make_shifted_720Pwindows_bysize(shape, num_windows)),
    }


def execute_provider_cases(payload: dict[str, Any]) -> None:
    source = hashed_input("provider:source", (3, 7, 10, 4)).sigmoid()
    decoded = hashed_input("provider:decoded", (3, 9, 12, 3)).sigmoid()
    preprocessed = nodes_seedvr.SeedVR2Preprocess.execute(source).result[0]
    postprocessed = {
        method: enc(nodes_seedvr.SeedVR2PostProcessing.execute(decoded, source, method).result[0])
        for method in ("lab", "wavelet", "adain", "none")
    }
    latent = {
        "samples": hashed_input("provider:latent", (1, 16, 6, 2, 3)),
        "noise_mask": torch.ones((1, 16, 6, 2, 3)),
    }
    chunked = nodes_seedvr.SeedVR2TemporalChunk.execute(
        latent,
        2,
        {"chunking_mode": "manual", "frames_per_chunk": 13},
    ).result
    chunks = chunked[0]
    merged = nodes_seedvr.SeedVR2TemporalMerge.execute(chunks, [chunked[1]]).result[0]
    payload["provider"] = {
        "source": enc(source),
        "decoded": enc(decoded),
        "preprocessed": enc(preprocessed),
        "postprocessed": postprocessed,
        "chunks": [enc(chunk["samples"]) for chunk in chunks],
        "temporal_overlap": chunked[1],
        "merged": enc(merged["samples"]),
    }


def main() -> None:
    commit = reference_commit()
    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": "attention_pytorch",
            "vae_attention": "pytorch_attention",
            "rope": "pure torch (in_training=True)",
            **tuple_provenance(torch.__version__),
        },
        "layouts": {},
        "cases": {},
    }
    for name, config in FULL_CONFIGS.items():
        model = build_dit(config, "meta")
        payload["layouts"][name] = sorted(
            (key, list(value.shape)) for key, value in model.state_dict().items()
        )
    with torch.device("meta"):
        vae = VideoAutoencoderKLWrapper()
    payload["layouts"]["vae"] = sorted(
        (key, list(value.shape)) for key, value in vae.state_dict().items()
    )
    del vae

    execute_dit_cases(payload)
    execute_vae_cases(payload)
    execute_window_cases(payload)
    execute_provider_cases(payload)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
