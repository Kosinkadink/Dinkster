"""Generate Flux2 diffusion transformer goldens from ComfyUI.

Runs the REFERENCE Flux model in its Flux2 configuration
(comfy/ldm/flux/model.py with global_modulation / mlp_silu_act /
ops_bias=False @ the audited baseline) and writes
packages/dinkster-inference-torch/tests/goldens/flux2_goldens.json.
dinkster_inference.flux2 and dinkster_inference_torch.flux are pinned
against these outputs; the oracle is the executed reference, never a
re-derivation. This payload is additive - the classic, gated, and
vector-free flux goldens do not rotate.

Payload:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE Flux2 dev, Klein 9B, and Klein 4B models, built with the
  reference detection's pinned Flux2 facts (comfy/model_detection.py
  flux branch, the double_stream_modulation_img leg @ 947c2749) on
  the meta device (weights never materialize).
- "cases": tiny Flux2 architectures executed with deterministic
  hash-filled weights (unet_fill.py - the only rank-1 weights are
  QKNorm RMS scales, so the rank rule holds) covering the
  guidance-distilled dev shape and the guidance-free Klein shape,
  with the first double-block and first single-block outputs
  recorded independently.

Determinism: attention is forced to the pytorch SDPA backend and
RoPE to the reference's pure-torch path, exactly as
tools/gen_flux_goldens.py documents.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_flux2_goldens.py

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
if sys.platform.startswith("linux") and torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has"
        f" {torch.__version__}. Regenerating on another build rotates the payload"
        " hash - update the pin deliberately and re-prove bit-stability."
        " (Non-Linux runs write a platform-tuple file whose name carries the"
        " build, so they are exempt.)"
    )

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.flux import math as _flux_math  # noqa: E402
from comfy.ldm.flux.model import Flux  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports); see
# tools/gen_flux_goldens.py for why the bound global must be patched.
_flux_math.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert _flux_math.attention.__globals__["optimized_attention"] is (_attention.attention_pytorch), (
    "flux math is not calling the forced pytorch SDPA backend"
)
ATTENTION_BACKEND = "attention_pytorch"

# Force the pure-torch RoPE path away from the comfy-kitchen kernel.
model_management.in_training = True
ROPE_BACKEND = "_apply_rope (pure torch, in_training=True)"

OUT = platform_golden_path(
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "flux2_goldens.json",
    torch.__version__,
)

#: Full-size configs: the reference detection's pinned Flux2 facts
#: (comfy/model_detection.py flux branch, flux2 leg @ 947c2749) in
#: the FluxParams kwarg shape. All three share global modulation,
#: SiLU-gated packed MLPs, bias-free ops, 4 RoPE axes of 32 with
#: theta 2000, text ids on axis 3, patch size 1, no vector embedder.
_FLUX2_COMMON = {
    "in_channels": 128,
    "out_channels": 128,
    "vec_in_dim": None,
    "mlp_ratio": 3.0,
    "axes_dim": [32, 32, 32, 32],
    "theta": 2000,
    "patch_size": 1,
    "qkv_bias": False,
    "txt_ids_dims": [3],
    "global_modulation": True,
    "mlp_silu_act": True,
    "ops_bias": False,
}

FULL_CONFIGS = {
    "flux2_dev": {
        **_FLUX2_COMMON,
        "context_in_dim": 15360,
        "hidden_size": 6144,
        "num_heads": 48,
        "depth": 8,
        "depth_single_blocks": 48,
        "guidance_embed": True,
    },
    "flux2_klein_9b": {
        **_FLUX2_COMMON,
        "context_in_dim": 12288,
        "hidden_size": 4096,
        "num_heads": 32,
        "depth": 8,
        "depth_single_blocks": 24,
        "guidance_embed": False,
    },
    "flux2_klein_4b": {
        **_FLUX2_COMMON,
        "context_in_dim": 7680,
        "hidden_size": 3072,
        "num_heads": 24,
        "depth": 5,
        "depth_single_blocks": 20,
        "guidance_embed": False,
    },
}

#: Tiny geometry for executed parity: sum(axes_dim) must equal
#: hidden_size // num_heads and every axis must be even (RoPE
#: rotates pairs).
_TINY = {
    **_FLUX2_COMMON,
    "in_channels": 8,
    "out_channels": 8,
    "context_in_dim": 24,
    "hidden_size": 32,
    "num_heads": 2,
    "depth": 2,
    "depth_single_blocks": 2,
    "axes_dim": [4, 4, 4, 4],
    "guidance_embed": True,
}

_TINY_KLEIN = {**_TINY, "guidance_embed": False}

#: name -> (config, batch, height, width, context_len). Patch size 1
#: has no padding leg, so spatial extents just vary.
CASES = {
    "flux2_guidance": (_TINY, 2, 8, 8, 7),
    "flux2_no_guidance": (_TINY_KLEIN, 2, 6, 10, 5),
}


def build_reference(config: dict, device: str) -> Flux:
    return Flux(
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
        guidance = None
        if config["guidance_embed"]:
            guidance = torch.linspace(1.5, 4.0, batch, dtype=torch.float32)
        block_outputs: dict[str, object] = {}
        hooks = [
            model.double_blocks[0].register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    double_img=enc(output[0]), double_txt=enc(output[1])
                )
            ),
            model.single_blocks[0].register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    single=enc(output)
                )
            ),
        ]
        try:
            with torch.no_grad():
                out = model(
                    x,
                    timestep=timesteps,
                    context=context,
                    y=None,
                    guidance=guidance,
                )
        finally:
            for hook in hooks:
                hook.remove()
        payload["cases"][name] = {
            "config": config,
            "state_dict": entries,
            "batch": batch,
            "height": height,
            "width": width,
            "context_len": context_len,
            "timesteps": timesteps.tolist(),
            "guidance": None if guidance is None else guidance.tolist(),
            "block_outputs": block_outputs,
            "output": enc(out),
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
