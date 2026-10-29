"""Generate SD/SDXL AutoencoderKL goldens from the ComfyUI reference.

Runs the REFERENCE comfy.ldm.models.autoencoder.AutoencoderKL @ the
audited baseline on two small deterministic architectures and writes
packages/dinkster-inference-torch/tests/goldens/kl_goldens.json.
dinkster_inference_torch.autoencoder_kl is pinned against these outputs
- the oracle is the reference code itself, never a re-derivation.

The architectures cannot be arbitrarily tiny (the reference hardcodes
GroupNorm(num_groups=32), so every width is a multiple of 32), which
makes the weights too large to store. The golden therefore records
the reference's sorted (key, shape) state-dict listing, and both the
reference run here and the replay tests fill each tensor from the
shared deterministic hash in
packages/dinkster-inference-torch/tests/kl_fill.py - bit-identical
across torch versions, keyed only on tensor names. Inputs and the
reference encode (posterior mode) / decode outputs ARE stored.

"standard" is the 4-level x8 ch_mult ladder (1,2,4,4); "x4" the
3-level upscaler ladder (1,2,4) with a non-square input and a
different num_res_blocks. Every reference block type is on the
execution path: resnet chains, nin shortcuts, downsample/upsample
convs, mid attention, quant convs. "regularizer" is the
regularizer-only AutoencodingEngine variant (comfy/sd.py builds it
when post_quant_conv is absent - the classic Flux ae.safetensors
layout): no quant convs, DiagonalGaussianRegularizer consumes
encoder.conv_out's moments directly, z_channels 16 like Flux.
"batch_norm" is the packed-latent variant (ddconfig
batch_norm_latent, the Flux2 VAE layout): posterior mode is 2x2
space-to-depth packed then normalized by frozen BatchNorm2d running
stats; decode inverts both before post_quant_conv. Its non-square
input exercises the packing axes independently.

Usage (needs a torch interpreter that imports the pinned checkout;
the workspace root venv is deliberately torch-free):

    PYTHONPATH=../ComfyUI /path/to/torch-venv/bin/python tools/gen_kl_goldens.py

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

# The reference must be imported in its default (CUDA) mode: forcing
# comfy into CPU mode changes its kernel selection (measured ~1e-5
# drift on these cases even though all tensors here live on CPU), and
# a CPU-only torch build cannot import comfy.model_management at all.
# Regenerate with the torch build recorded in the existing goldens to
# keep pre-existing case data byte-identical.
if not torch.cuda.is_available():
    raise SystemExit(
        "goldens must be generated with a CUDA-capable torch on a CUDA"
        " machine: the reference's CPU import mode changes kernel"
        " selection and moves the recorded float32 outputs"
    )

from comfy.ldm.models import autoencoder as _reference_module  # noqa: E402
from comfy.ldm.models.autoencoder import (  # noqa: E402
    AutoencoderKL,
    AutoencodingEngine,
)
from kl_fill import fill_state_dict  # noqa: E402

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "kl_goldens.json"

CASES = {
    "standard": {
        "seed": 0x5EED,  # pinned: was sorted-index 0 originally
        "embed_dim": 3,
        "ddconfig": {
            "double_z": True,
            "z_channels": 3,
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "input_shape": [1, 3, 16, 16],
    },
    "x4": {
        "seed": 0x5EED + 1,  # pinned: was sorted-index 1 originally
        "embed_dim": 3,
        "ddconfig": {
            "double_z": True,
            "z_channels": 3,
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "input_shape": [1, 3, 16, 24],
    },
    # Regularizer-only AutoencodingEngine (comfy/sd.py builds this
    # when post_quant_conv is absent - the classic Flux ae layout):
    # no quant convs, moments come straight off encoder.conv_out,
    # decoder consumes the latent directly. embed_dim is omitted
    # (there is no quant conv to define it); z_channels 16 like Flux.
    "regularizer": {
        "seed": 0x5EED + 2,
        "regularizer_only": True,
        "ddconfig": {
            "double_z": True,
            "z_channels": 16,
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "input_shape": [1, 3, 16, 16],
    },
    # Packed-latent (batch-norm) variant - the Flux2 VAE layout. The
    # reference AutoencoderKL builds the frozen BatchNorm2d from
    # ddconfig["batch_norm_latent"]; embed_dim equals z_channels as in
    # the real checkpoint (32/32 there, 4/4 here). The non-square input
    # gives a non-square packed latent, pinning the space-to-depth axis
    # order.
    "batch_norm": {
        "seed": 0x5EED + 3,
        "embed_dim": 4,
        "ddconfig": {
            "double_z": True,
            "z_channels": 4,
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
            "batch_norm_latent": True,
        },
        "input_shape": [1, 3, 32, 48],
    },
}


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
    module_file = Path(_reference_module.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "comfy.ldm.models.autoencoder was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "cases": {},
    }

    for name, spec in sorted(CASES.items()):
        if spec.get("regularizer_only"):
            # Exactly the comfy/sd.py no-post_quant_conv construction.
            model = AutoencodingEngine(
                regularizer_config={
                    "target": ("comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer")
                },
                encoder_config={
                    "target": ("comfy.ldm.modules.diffusionmodules.model.Encoder"),
                    "params": dict(spec["ddconfig"]),
                },
                decoder_config={
                    "target": ("comfy.ldm.modules.diffusionmodules.model.Decoder"),
                    "params": dict(spec["ddconfig"]),
                },
            )
        else:
            model = AutoencoderKL(embed_dim=spec["embed_dim"], ddconfig=dict(spec["ddconfig"]))
        model.eval()
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        gen = torch.Generator().manual_seed(spec["seed"])
        content = (
            torch.rand(spec["input_shape"], generator=gen, dtype=torch.float32) * 2.0 - 1.0
        )  # already in [-1, 1]: process_input is the wrapper's job

        with torch.no_grad():
            latent = model.encode(content)
            decoded = model.decode(latent)

        payload["cases"][name] = {
            **({} if spec.get("regularizer_only") else {"embed_dim": spec["embed_dim"]}),
            "ddconfig": spec["ddconfig"],
            "state_dict": entries,
            "input": enc(content),
            "latent": enc(latent),
            "decoded": enc(decoded),
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
