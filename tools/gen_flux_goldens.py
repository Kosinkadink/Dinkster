"""Generate classic-Flux diffusion transformer goldens from ComfyUI.

Runs the REFERENCE Flux model (comfy/ldm/flux/model.py) @ the audited
baseline and writes
packages/dinkster-inference-torch/tests/goldens/flux_goldens.json.
dinkster_inference_torch.flux is pinned against these outputs; the
oracle is the executed reference, never a re-derivation.

The classic payload remains byte-stable. A separate
``flux_gated_goldens.json`` payload records the Ovis gated architecture so
adding the gated proof does not rotate the classic golden. The exact no-vector,
no-guidance Ovis architecture and its absent-y forward live in the additive
``flux_vector_free_goldens.json`` payload, so neither earlier golden rotates.

Payloads:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE Flux dev and schnell models, built with the reference's
  pinned detection facts (comfy/model_detection.py flux branch @
  947c2749) on the meta device (weights never materialize). These
  pin torch-free detection and layout generation. Schnell's
  guidance_in is nn.Identity() in the reference, so its listing has
  no guidance keys - exactly the torch-free layout.
- "cases": tiny architectures executed with deterministic
  hash-filled weights (unet_fill.py - all rank-1 Flux weights in
  these cases are RMS scales: QKNorm plus optional ``txt_norm``).
  Cases cover the guidance-distilled dev shape, the
  guidance-free schnell shape, odd spatial extents (the circular
  pad_to_patch_size leg), and context RMS normalization before
  ``txt_in``.
- The gated payload contains a full-size Ovis layout plus a tiny executed
  forward with the first double-block and first single-block outputs recorded
  independently.
- The vector-free payload contains the approved 397-key full-size artifact
  geometry and a tiny Ovis forward called with y=None and guidance=None.

Determinism: attention is forced to the pytorch SDPA backend and
RoPE to the reference's pure-torch path (comfy.model_management.
in_training = True routes comfy/ldm/flux/math.py apply_rope away
from the comfy-kitchen kernel; the kernel's eager backend is the
same math). Both forced selections are recorded in the payload.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_flux_goldens.py

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

import torch  # noqa: E402
from comfy import model_management, ops  # noqa: E402
from comfy.ldm.flux import math as _flux_math  # noqa: E402
from comfy.ldm.flux.model import Flux  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# The reference attention calls the ambient optimized_attention,
# which comfy selects per environment (xformers/sage/flash/pytorch).
# Goldens must not depend on which accelerators happen to be
# installed: force the pytorch SDPA backend, the one Dinkster ports.
# comfy/ldm/flux/math.py binds it with a from-import
# (`from comfy.ldm.modules.attention import optimized_attention`), so
# the name Flux actually calls is the BOUND GLOBAL in that module -
# patching the source module attribute alone would not reach it.
_flux_math.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert _flux_math.attention.__globals__["optimized_attention"] is (_attention.attention_pytorch), (
    "flux math is not calling the forced pytorch SDPA backend"
)
ATTENTION_BACKEND = "attention_pytorch"

# comfy/ldm/flux/math.py apply_rope routes through the comfy-kitchen
# kernel unless in_training is set; force the pure-torch reference
# path so goldens do not depend on the installed kitchen backend.
model_management.in_training = True
ROPE_BACKEND = "_apply_rope (pure torch, in_training=True)"

_GOLDENS = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens"
OUT = platform_golden_path(_GOLDENS / "flux_goldens.json", torch.__version__)
GATED_OUT = platform_golden_path(_GOLDENS / "flux_gated_goldens.json", torch.__version__)
VECTOR_FREE_OUT = platform_golden_path(
    _GOLDENS / "flux_vector_free_goldens.json", torch.__version__
)

#: Full-size configs: the reference detection's pinned classic-Flux
#: facts (comfy/model_detection.py flux branch @ 947c2749) in the
#: FluxParams kwarg shape. Schnell differs only in guidance_embed.
FULL_CONFIGS = {
    "flux_dev": {
        "in_channels": 16,
        "out_channels": 16,
        "vec_in_dim": 768,
        "context_in_dim": 4096,
        "hidden_size": 3072,
        "mlp_ratio": 4.0,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "axes_dim": [16, 56, 56],
        "theta": 10000,
        "patch_size": 2,
        "qkv_bias": True,
        "guidance_embed": True,
    },
    "flux_dev_txt_norm": {
        "in_channels": 16,
        "out_channels": 16,
        "vec_in_dim": 768,
        "context_in_dim": 4096,
        "hidden_size": 3072,
        "mlp_ratio": 4.0,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "axes_dim": [16, 56, 56],
        "theta": 10000,
        "patch_size": 2,
        "qkv_bias": True,
        "guidance_embed": True,
        "txt_norm": True,
    },
    "flux_schnell": {
        "in_channels": 16,
        "out_channels": 16,
        "vec_in_dim": 768,
        "context_in_dim": 4096,
        "hidden_size": 3072,
        "mlp_ratio": 4.0,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "axes_dim": [16, 56, 56],
        "theta": 10000,
        "patch_size": 2,
        "qkv_bias": True,
        "guidance_embed": False,
    },
}

#: Tiny geometry for executed parity: sum(axes_dim) must equal
#: hidden_size // num_heads and every axis must be even (RoPE
#: rotates pairs).
_TINY = {
    "in_channels": 16,
    "out_channels": 16,
    "vec_in_dim": 12,
    "context_in_dim": 24,
    "hidden_size": 32,
    "mlp_ratio": 4.0,
    "num_heads": 2,
    "depth": 2,
    "depth_single_blocks": 2,
    "axes_dim": [4, 6, 6],
    "theta": 10000,
    "patch_size": 2,
    "qkv_bias": True,
    "guidance_embed": True,
}

_TINY_SCHNELL = {**_TINY, "guidance_embed": False}
_TINY_TXT_NORM = {**_TINY, "txt_norm": True}
_TINY_GATED_OVIS = {**_TINY, "txt_norm": True, "yak_mlp": True}
_TINY_VECTOR_FREE_OVIS = {
    **_TINY,
    "vec_in_dim": None,
    "guidance_embed": False,
    "txt_norm": True,
    "yak_mlp": True,
    "txt_ids_dims": [1, 2],
}

FULL_GATED_OVIS = {
    **FULL_CONFIGS["flux_dev_txt_norm"],
    "context_in_dim": 2048,
    "yak_mlp": True,
}

FULL_VECTOR_FREE_OVIS = {
    **FULL_CONFIGS["flux_schnell"],
    "vec_in_dim": None,
    "context_in_dim": 2048,
    "depth": 6,
    "depth_single_blocks": 27,
    "txt_norm": True,
    "yak_mlp": True,
    "txt_ids_dims": [1, 2],
}

#: name -> (config, batch, height, width, context_len). Odd spatial
#: extents exercise the circular pad_to_patch_size leg.
CASES = {
    "dev_guidance": (_TINY, 2, 8, 8, 7),
    "dev_odd_spatial": (_TINY, 1, 7, 10, 5),
    "dev_txt_norm": (_TINY_TXT_NORM, 2, 8, 8, 7),
    "schnell_plain": (_TINY_SCHNELL, 2, 8, 8, 6),
}


def build_reference(config: dict, device: str) -> Flux:
    params = dict(config)
    txt_ids_dims = params.pop("txt_ids_dims", [])
    return Flux(
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
        txt_ids_dims=txt_ids_dims,
        **params,
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
    module_file = Path(sys.modules[Flux.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference Flux was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
            "rope": ROPE_BACKEND,
            **tuple_provenance(torch.__version__),
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
        timesteps = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
        context = hashed_input(f"{name}:context", (batch, context_len, config["context_in_dim"]))
        y = hashed_input(f"{name}:y", (batch, config["vec_in_dim"]))
        guidance = None
        if config["guidance_embed"]:
            guidance = torch.linspace(1.5, 4.0, batch, dtype=torch.float32)
        with torch.no_grad():
            out = model(
                x,
                timestep=timesteps,
                context=context,
                y=y,
                guidance=guidance,
            )
        payload["cases"][name] = {
            "config": config,
            "state_dict": entries,
            "batch": batch,
            "height": height,
            "width": width,
            "context_len": context_len,
            "timesteps": timesteps.tolist(),
            "guidance": None if guidance is None else guidance.tolist(),
            "output": enc(out),
        }

    OUT.write_bytes((json.dumps(payload, indent=1) + "\n").encode())
    print(f"wrote {OUT}", file=sys.stderr)

    gated_model = build_reference(FULL_GATED_OVIS, "meta")
    gated_payload: dict = {
        "reference": payload["reference"],
        "config": FULL_GATED_OVIS,
        "layout": sorted(
            (key, list(value.shape)) for key, value in gated_model.state_dict().items()
        ),
    }

    config = _TINY_GATED_OVIS
    case = "gated_ovis"
    batch, height, width, context_len = 2, 8, 8, 7
    model = build_reference(config, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    x = hashed_input(f"{case}:x", (batch, config["in_channels"], height, width))
    timesteps = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
    context = hashed_input(f"{case}:context", (batch, context_len, config["context_in_dim"]))
    y = hashed_input(f"{case}:y", (batch, config["vec_in_dim"]))
    guidance = torch.linspace(1.5, 4.0, batch, dtype=torch.float32)
    block_outputs: dict[str, object] = {}
    hooks = [
        model.double_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: block_outputs.update(
                double_img=enc(output[0]), double_txt=enc(output[1])
            )
        ),
        model.single_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: block_outputs.update(single=enc(output))
        ),
    ]
    try:
        with torch.no_grad():
            output = model(
                x,
                timestep=timesteps,
                context=context,
                y=y,
                guidance=guidance,
            )
    finally:
        for hook in hooks:
            hook.remove()
    gated_payload["case"] = {
        "name": case,
        "config": config,
        "state_dict": entries,
        "batch": batch,
        "height": height,
        "width": width,
        "context_len": context_len,
        "timesteps": timesteps.tolist(),
        "guidance": guidance.tolist(),
        "block_outputs": block_outputs,
        "output": enc(output),
    }
    GATED_OUT.write_bytes((json.dumps(gated_payload, indent=1) + "\n").encode())
    print(f"wrote {GATED_OUT}", file=sys.stderr)

    vector_free_model = build_reference(FULL_VECTOR_FREE_OVIS, "meta")
    vector_free_payload: dict = {
        "reference": payload["reference"],
        "config": FULL_VECTOR_FREE_OVIS,
        "layout": sorted(
            (key, list(value.shape)) for key, value in vector_free_model.state_dict().items()
        ),
        "artifact": {
            "name": "ovis_image_bf16.safetensors",
            "bytes": 14740943536,
            "sha256": "eb3d9e1b201412b3b527472cf82a2c5add6b5a40a37e94d99f11c381f18a9e2b",
            "key_count": 397,
        },
    }

    config = _TINY_VECTOR_FREE_OVIS
    case = "vector_free_ovis"
    batch, height, width, context_len = 2, 8, 8, 7
    model = build_reference(config, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    x = hashed_input(f"{case}:x", (batch, config["in_channels"], height, width))
    timesteps = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
    context = hashed_input(f"{case}:context", (batch, context_len, config["context_in_dim"]))
    with torch.no_grad():
        output = model(
            x,
            timestep=timesteps,
            context=context,
        )
    vector_free_payload["case"] = {
        "name": case,
        "config": config,
        "state_dict": entries,
        "batch": batch,
        "height": height,
        "width": width,
        "context_len": context_len,
        "timesteps": timesteps.tolist(),
        "y": None,
        "guidance": None,
        "output": enc(output),
    }
    VECTOR_FREE_OUT.write_bytes((json.dumps(vector_free_payload, indent=1) + "\n").encode())
    print(f"wrote {VECTOR_FREE_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
