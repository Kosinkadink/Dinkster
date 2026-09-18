"""Generate Anima diffusion transformer goldens from ComfyUI.

Runs the REFERENCE Anima model (comfy/ldm/anima/model.py: a Cosmos
Predict2 MiniTrainDIT backbone plus the LLM adapter @ the audited
baseline) and writes
packages/dinkster-inference-torch/tests/goldens/anima_goldens.json.
dinkster_inference.anima and dinkster_inference_torch.anima_model /
cosmos_predict2 are pinned against these outputs; the oracle is the
executed reference, never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listing of the
  FULL-SIZE Anima 2B model, built with the reference detection's
  pinned kwargs (comfy/model_detection.py anima branch @ 82f839f5)
  on the meta device (weights never materialize).
- "cases": a tiny Anima architecture executed with deterministic
  hash-filled weights (unet_fill.py - every rank-1 weight is a norm
  scale, so the rank rule holds) covering an even-spatial case with
  per-token adapter weights and an odd-spatial case (exercising the
  circular pad + crop leg) without them, with the LLM adapter output
  and the first backbone block output recorded independently.

The reference Anima hardcodes the full-size LLMAdapter; the tiny
cases replace model.llm_adapter with a reduced LLMAdapter BEFORE the
hash fill so the executed state dict stays small.

Determinism: attention is forced to the pytorch SDPA backend (the
adapter's own F.scaled_dot_product_attention is already that
backend), and the backbone's fused norm+rope runs comfy_kitchen's
rms_rope_split_half, which dispatches to the deterministic eager
kernel on CPU.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_anima_goldens.py

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

from comfy import ops  # noqa: E402
from comfy.ldm.anima.model import Anima, LLMAdapter  # noqa: E402
from comfy.ldm.cosmos import predict2 as _predict2  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). The backbone
# imported optimized_attention as a module global at import time, so
# that bound name must be patched alongside the source module.
_predict2.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert _predict2.torch_attention_op.__globals__["optimized_attention"] is (
    _attention.attention_pytorch
), "the predict2 backbone is not calling the forced pytorch SDPA backend"
ATTENTION_BACKEND = "attention_pytorch"

# The backbone's self-attention norm+rope always runs comfy_kitchen's
# rms_rope_split_half; on CPU it dispatches to deterministic eager.
ROPE_BACKEND = "comfy_kitchen rms_rope_split_half (eager cpu backend)"

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "anima_goldens.json"

#: The reference detection's pinned Anima kwargs (comfy/
#: model_detection.py anima branch @ 82f839f5) in the MiniTrainDIT
#: kwarg shape. max_img_h/w, max_frames, and the fps range only shape
#: unused legs (rope fps modulation is off); they are recorded as
#: detection pins.
_ANIMA_COMMON = {
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 128,
    "concat_padding_mask": True,
    "mlp_ratio": 4.0,
    "pos_emb_cls": "rope3d",
    "pos_emb_learnable": True,
    "pos_emb_interpolation": "crop",
    "min_fps": 1,
    "max_fps": 30,
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
    "extra_per_block_abs_pos_emb": False,
    "extra_h_extrapolation_ratio": 1.0,
    "extra_w_extrapolation_ratio": 1.0,
    "extra_t_extrapolation_ratio": 1.0,
    "rope_enable_fps_modulation": False,
}

FULL_CONFIGS = {
    "anima_2b": {
        **_ANIMA_COMMON,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "num_blocks": 28,
        "num_heads": 16,
        "crossattn_emb_channels": 1024,
    },
}

#: Tiny backbone: head_dim 24 splits into rope axes
#: dim_h = dim_w = 8 and dim_t = 8, the smallest split whose NTK
#: exponents dim / (dim - 2) stay finite.
_TINY = {
    **_ANIMA_COMMON,
    "in_channels": 4,
    "out_channels": 4,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 48,
    "num_blocks": 2,
    "num_heads": 2,
    "crossattn_emb_channels": 16,
    "adaln_lora_dim": 8,
}

#: Tiny adapter: model_dim == target_dim keeps in_proj an Identity,
#: exactly like the full-size profile.
_TINY_ADAPTER = {
    "source_dim": 12,
    "target_dim": 16,
    "model_dim": 16,
    "num_layers": 2,
    "num_heads": 2,
}

#: name -> (batch, frames, height, width, source_rows, target_ids,
#: use_weights). The odd case exercises the circular pad + crop leg.
CASES = {
    "anima_even": (2, 1, 6, 6, 5, 7, True),
    "anima_odd": (2, 1, 5, 7, 5, 7, False),
}


def build_reference(config: dict, device: str) -> Anima:
    return Anima(
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


def target_ids(name: str, batch: int, rows: int) -> torch.Tensor:
    """Deterministic in-vocabulary T5 token ids, recorded per case."""
    flat = torch.arange(batch * rows, dtype=torch.int64)
    return (flat * 4093 + 17) % 32128


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
    module_file = Path(sys.modules[Anima.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference Anima was imported from"
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
    for name, config in FULL_CONFIGS.items():
        model = build_reference(config, "meta")
        payload["layouts"][name] = sorted(
            (key, list(value.shape)) for key, value in model.state_dict().items()
        )

    for name, (batch, frames, height, width, source_rows, id_rows, use_weights) in sorted(
        CASES.items()
    ):
        model = build_reference(_TINY, "cpu")
        model.llm_adapter = LLMAdapter(
            **_TINY_ADAPTER,
            device="cpu",
            dtype=torch.float32,
            operations=ops.disable_weight_init,
        )
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        x = hashed_input(f"{name}:x", (batch, _TINY["in_channels"], frames, height, width))
        timesteps = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
        context = hashed_input(f"{name}:context", (batch, source_rows, _TINY_ADAPTER["source_dim"]))
        ids = target_ids(name, batch, id_rows).view(batch, id_rows)
        weights = None
        if use_weights:
            weights = torch.linspace(0.5, 1.5, batch * id_rows, dtype=torch.float32).view(
                batch, id_rows, 1
            )
        block_outputs: dict[str, object] = {}
        hooks = [
            model.llm_adapter.register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    adapter=enc(output)
                )
            ),
            model.blocks[0].register_forward_hook(
                lambda _module, _inputs, output, record=block_outputs: record.update(
                    block=enc(output)
                )
            ),
        ]
        try:
            with torch.no_grad():
                out = model(
                    x,
                    timesteps,
                    context,
                    t5xxl_ids=ids,
                    t5xxl_weights=weights,
                )
        finally:
            for hook in hooks:
                hook.remove()
        payload["cases"][name] = {
            "config": _TINY,
            "adapter_config": _TINY_ADAPTER,
            "state_dict": entries,
            "batch": batch,
            "frames": frames,
            "height": height,
            "width": width,
            "source_rows": source_rows,
            "timesteps": timesteps.tolist(),
            "t5xxl_ids": ids.tolist(),
            "t5xxl_weights": None if weights is None else weights.tolist(),
            "block_outputs": block_outputs,
            "output": enc(out),
        }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
