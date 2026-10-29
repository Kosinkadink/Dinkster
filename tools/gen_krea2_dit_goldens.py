"""Generate Krea 2 diffusion transformer goldens from ComfyUI.

Runs the REFERENCE Krea 2 single-stream DiT (comfy/ldm/krea2/model.py
SingleStreamDiT @ the audited baseline) and writes
packages/dinkster-inference-torch/tests/goldens/krea2_dit_goldens.json.
dinkster_inference.krea2 and the torch runtime are pinned against these
outputs; the oracle is the executed reference, never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listing of the
  FULL-SIZE Krea 2 model, built with the reference constructor
  defaults (features 6144, 28 blocks, 48/12 GQA heads, 12-layer
  2560-wide text fusion @ 947c2749) on the meta device (weights never
  materialize).
- "cases": a tiny Krea 2 architecture executed with deterministic
  hash-filled weights (unet_fill.py - the only rank-1 ``.weight``
  keys would be norm scales and Krea 2 spells its RMS scales
  ``.scale``, so the rank rule holds trivially) covering the plain
  text-to-image path with patch padding/cropping, both reference-
  latent methods, and the temporal 5D reshape, with the text-fusion
  output and the first transformer block output recorded
  independently.

Determinism: attention is forced to the pytorch SDPA backend and
RoPE to the reference's pure-torch path, exactly as
tools/gen_flux2_goldens.py documents.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_krea2_dit_goldens.py

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

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.krea2 import model as krea2_model  # noqa: E402
from comfy.ldm.krea2.model import SingleStreamDiT  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). The Krea 2
# module binds optimized_attention_masked at import, so the module-
# level name must be patched, not just the attention module's.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
krea2_model.optimized_attention_masked = _attention.attention_pytorch
assert krea2_model.optimized_attention_masked is _attention.attention_pytorch, (
    "the Krea 2 module is not calling the forced pytorch SDPA backend"
)
ATTENTION_BACKEND = "attention_pytorch"

# Force the pure-torch RoPE path away from the comfy-kitchen kernel
# (comfy.ldm.flux.math apply_rope branches on in_training).
model_management.in_training = True
ROPE_BACKEND = "_apply_rope (pure torch, in_training=True)"

OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "krea2_dit_goldens.json"
)

#: The single published Krea 2 geometry: the reference constructor
#: defaults (comfy/ldm/krea2/model.py SingleStreamDiT @ 947c2749).
#: RAW and Turbo share this tensor layout.
FULL_CONFIG = {
    "features": 6144,
    "tdim": 256,
    "txtdim": 2560,
    "heads": 48,
    "kvheads": 12,
    "multiplier": 4,
    "layers": 28,
    "patch": 2,
    "channels": 16,
    "bias": False,
    "theta": 1000,
    "txtlayers": 12,
    "txtheads": 20,
    "txtkvheads": 20,
}

#: Tiny geometry for executed parity: heads must divide features,
#: kvheads must divide heads, and the derived head dim must satisfy
#: the constructor's 3-axis RoPE split (headdim 16 -> axes [4, 6, 6],
#: every axis even).
_TINY = {
    "features": 32,
    "tdim": 16,
    "txtdim": 16,
    "heads": 2,
    "kvheads": 1,
    "multiplier": 4,
    "layers": 2,
    "patch": 2,
    "channels": 4,
    "bias": False,
    "theta": 1000,
    "txtlayers": 3,
    "txtheads": 2,
    "txtkvheads": 1,
}

#: name -> (batch, height, width, context_len, ref_shapes, ref_method,
#: temporal_frames). Height 5 exercises pad_to_patch_size plus the
#: output crop; the ref cases cover both reference-latent methods; the
#: temporal case covers the 5D reshape seam.
CASES = {
    "krea2_base": (2, 5, 6, 7, (), None, None),
    "krea2_ref_index": (1, 4, 4, 5, ((2, 6), (4, 4)), "index", None),
    "krea2_ref_timestep_zero": (2, 4, 4, 5, ((4, 4),), "index_timestep_zero", None),
    "krea2_temporal": (1, 4, 4, 3, (), None, 2),
}


def build_reference(config: dict, device: str) -> SingleStreamDiT:
    return SingleStreamDiT(
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
    module_file = Path(sys.modules[SingleStreamDiT.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference SingleStreamDiT was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
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
    model = build_reference(FULL_CONFIG, "meta")
    payload["layouts"]["krea2"] = sorted(
        (key, list(value.shape)) for key, value in model.state_dict().items()
    )

    for name, (batch, height, width, context_len, ref_shapes, ref_method, frames) in sorted(
        CASES.items()
    ):
        config = _TINY
        model = build_reference(config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        channels = config["channels"]
        if frames is None:
            x = hashed_input(f"{name}:x", (batch, channels, height, width))
            context_batch = batch
        else:
            x = hashed_input(f"{name}:x", (batch, channels, frames, height, width))
            context_batch = batch * frames
        fused = config["txtlayers"] * config["txtdim"]
        context = hashed_input(f"{name}:context", (context_batch, context_len, fused))
        timesteps = torch.linspace(0.05, 0.95, context_batch, dtype=torch.float32)
        ref_latents = [
            hashed_input(f"{name}:ref{index}", (1, channels, rh, rw))
            for index, (rh, rw) in enumerate(ref_shapes)
        ]

        block_outputs: dict[str, object] = {}
        hooks = [
            model.txtfusion.register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    txtfusion=enc(output)
                )
            ),
            model.blocks[0].register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    block0=enc(output)
                )
            ),
        ]
        try:
            with torch.no_grad():
                out = model(
                    x,
                    timesteps,
                    context,
                    ref_latents=ref_latents if ref_latents else None,
                    ref_latents_method=ref_method,
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
            "ref_shapes": [list(shape) for shape in ref_shapes],
            "ref_method": ref_method,
            "frames": frames,
            "timesteps": timesteps.tolist(),
            "block_outputs": block_outputs,
            "output": enc(out),
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
