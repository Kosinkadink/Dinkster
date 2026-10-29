"""Generate Normalized Attention Guidance goldens from the ComfyUI reference.

Usage:
    PYTHONPATH=<comfy-deps> COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      .venv-gpu/bin/python tools/gen_nag_goldens.py

The reference may instead be supplied with ``--comfy-root``. The checkout
must be clean and exactly at the audited commit. Every case drives real
reference code: the attn1 output patch is extracted verbatim from
``comfy_extras.nodes_nag.NAGuidance.execute`` (this script never
re-implements the rewrite math).

Two case groups:

- "patch_cases": the extracted patch applied to recorded self-attention
  outputs, covering both cond_or_uncond lane orders, half sizes above 1,
  the tau clamp engaged and not engaged, and the alpha/scale edges.
- "unet_cases": the tiny reference UNetModel (multiple attn1 sites)
  executed with the patch installed through transformer_options, pinning
  how a patched attn1 output feeds the layers after it. The fused
  two-lane batch goes through ONE forward: upstream calc_cond_batch may
  split the lanes into separate forwards under free-memory pressure and
  the patch then silently no-ops, so the goldens deliberately force the
  single-chunk path that captures NAG-applied behavior.

The executed values drift by ULPs across CPU microarchitectures, so the
fixture records the mint host CPU (pin_cpu) and the replay suite skips on
other hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/nag_goldens.json"


def reference_root() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    value = args.comfy_root or os.environ.get("COMFYUI_REFERENCE")
    if value is None:
        raise SystemExit("pass --comfy-root or set COMFYUI_REFERENCE")
    return Path(value).resolve()


COMFY_ROOT = reference_root()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=COMFY_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


commit = git_output("rev-parse", "HEAD")
dirty = git_output("status", "--porcelain")
if commit != REFERENCE_COMMIT:
    raise SystemExit(f"reference is at {commit}; required {REFERENCE_COMMIT}")
if dirty:
    raise SystemExit(f"reference checkout must be clean:\n{dirty}")

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
from comfy_extras import nodes_nag  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# The reference CrossAttention calls the ambient optimized_attention,
# selected per environment. Force the pytorch SDPA backend Dinkster ports.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
ATTENTION_BACKEND = "attention_pytorch"


class _PatchRecorder:
    """Stands in for the ModelPatcher NAGuidance patches."""

    def __init__(self) -> None:
        self.patches: list[Any] = []
        self.cfg1_optimization_disabled = False

    def clone(self) -> _PatchRecorder:
        return self

    def set_model_attn1_output_patch(self, fn: Any) -> None:
        self.patches.append(fn)

    def disable_model_cfg1_optimization(self) -> None:
        self.cfg1_optimization_disabled = True


def extract_patch(nag_scale: float, nag_alpha: float, nag_tau: float) -> Any:
    recorder = _PatchRecorder()
    nodes_nag.NAGuidance.execute(recorder, nag_scale, nag_alpha, nag_tau)
    if len(recorder.patches) != 1 or not recorder.cfg1_optimization_disabled:
        raise SystemExit("NAGuidance.execute did not register the expected patch")
    return recorder.patches[0]


def seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator)


def enc(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": "float32", "data": value.flatten().tolist()}


def run_patch_case(
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    *,
    cond_or_uncond: tuple[int, ...],
    half_size: int = 1,
    seed: int,
    clamp: bool | None = None,
) -> dict[str, Any]:
    patch = extract_patch(nag_scale, nag_alpha, nag_tau)
    x = seeded((half_size * len(cond_or_uncond), 12, 16), seed)
    output = patch(x.clone(), {"cond_or_uncond": list(cond_or_uncond)})
    if clamp is not None:
        # Classify tau-clamp engagement with the reference itself: an
        # effectively infinite tau never clamps, so the case clamps iff
        # its output differs from that run's.
        unclamped_patch = extract_patch(nag_scale, nag_alpha, 1e9)
        unclamped = unclamped_patch(x.clone(), {"cond_or_uncond": list(cond_or_uncond)})
        if torch.equal(output, unclamped) == clamp:
            raise SystemExit(f"tau clamp expectation clamp={clamp} not met at seed {seed}")
    return {
        "params": {"nag_scale": nag_scale, "nag_alpha": nag_alpha, "nag_tau": nag_tau},
        "cond_or_uncond": list(cond_or_uncond),
        "input": enc(x),
        "output": enc(output),
    }


#: Tiny SD1-style config (multiple attn1 sites: two input blocks, the
#: middle block, and four output blocks each carry one).
TINY_SD1 = {
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


def run_unet_case(
    name: str,
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    *,
    cond_or_uncond: tuple[int, ...],
) -> dict[str, Any]:
    patch = extract_patch(nag_scale, nag_alpha, nag_tau)
    model = UNetModel(
        image_size=32,
        dims=2,
        use_spatial_transformer=True,
        legacy=False,
        num_classes=None,
        dtype=torch.float32,
        device="cpu",
        operations=ops.disable_weight_init,
        **TINY_SD1,
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)

    batch = len(cond_or_uncond)
    height, width, context_len = 16, 16, 7
    x = hashed_input(f"{name}:x", (batch, TINY_SD1["in_channels"], height, width))
    timesteps = torch.tensor([500.0] * batch, dtype=torch.float32)
    context = hashed_input(f"{name}:context", (batch, context_len, TINY_SD1["context_dim"]))
    with torch.no_grad():
        plain = model(x, timesteps=timesteps, context=context)
        patched = model(
            x,
            timesteps=timesteps,
            context=context,
            transformer_options={
                "patches": {"attn1_output_patch": [patch]},
                "cond_or_uncond": list(cond_or_uncond),
            },
        )
    if torch.equal(plain, patched):
        raise SystemExit(f"unet case {name}: the patch did not change the output")
    return {
        "params": {"nag_scale": nag_scale, "nag_alpha": nag_alpha, "nag_tau": nag_tau},
        "cond_or_uncond": list(cond_or_uncond),
        "config": TINY_SD1,
        "state_dict": entries,
        "height": height,
        "width": width,
        "context_len": context_len,
        "timesteps": timesteps.tolist(),
        "output_plain": enc(plain),
        "output": enc(patched),
    }


def main() -> None:
    for module in (nodes_nag,):
        path = Path(module.__file__ or "").resolve()
        if not path.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"{module.__name__} was imported from {path}, not {COMFY_ROOT}")

    patch_cases: dict[str, dict[str, Any]] = {}

    def patch_case(name: str, *args: float, **kw: Any) -> None:
        if name in patch_cases:
            raise SystemExit(f"duplicate case {name}")
        patch_cases[name] = run_patch_case(*args, seed=len(patch_cases) * 16 + 1, **kw)

    patch_case("nag_defaults", 5.0, 0.5, 1.5, cond_or_uncond=(0, 1), clamp=True)
    # Reversed lane order: upstream locates lanes with
    # cond_or_uncond.index, so both orders occur in the wild.
    patch_case("nag_reversed_lanes", 5.0, 0.5, 1.5, cond_or_uncond=(1, 0))
    patch_case("nag_half_size_two", 4.0, 0.6, 1.5, cond_or_uncond=(0, 1), half_size=2)
    # tau far above the observed norm ratios: the clamp never engages.
    patch_case("nag_tau_unclamped", 5.0, 0.5, 10.0, cond_or_uncond=(0, 1), clamp=False)
    # tau at its lower bound clamps everywhere the ratio exceeds one.
    patch_case("nag_tau_clamped", 5.0, 0.5, 1.0, cond_or_uncond=(0, 1), clamp=True)
    # alpha 0 blends fully back to the raw attention output.
    patch_case("nag_alpha_zero", 5.0, 0.0, 1.5, cond_or_uncond=(0, 1))
    # alpha 1 replaces the conditional rows outright.
    patch_case("nag_alpha_one", 5.0, 1.0, 1.5, cond_or_uncond=(0, 1))
    # scale 0 reduces guided to the negative stream before normalizing.
    patch_case("nag_scale_zero", 0.0, 0.5, 1.5, cond_or_uncond=(0, 1))

    unet_cases = {
        "nag_unet": run_unet_case("nag_unet", 5.0, 0.5, 1.5, cond_or_uncond=(0, 1)),
        "nag_unet_reversed": run_unet_case(
            "nag_unet_reversed", 3.0, 0.7, 1.4, cond_or_uncond=(1, 0)
        ),
    }

    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
        },
        "patch_cases": patch_cases,
        "unet_cases": unet_cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        payload["_meta"] = provenance
    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
