"""Generate LTX-Video 2B transformer and context-window goldens from ComfyUI.

Runs the REFERENCE LTXVModel (comfy/ldm/lightricks/model.py @ the
audited baseline) and writes
packages/dinkster-inference-torch/tests/goldens/ltx_model_goldens.json.
dinkster_inference.ltx and dinkster_inference_torch.ltx_model are pinned
against these outputs; the oracle is the executed reference, never a
re-derivation.

The same pinned reference model and LTXVContextWindows node produce
ltxv_context_windows.json. That fixture executes overlapping windows
through ComfyUI's real handler and records the fused model output for
the baseline and retained-first-frame cases.

Payload:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE 2B v0.9 and v0.9.5 models, built exactly as the reference
  loads them - the detection facts (comfy/model_detection.py ltxv
  branch @ 947c2749) updated with each checkpoint's verbatim metadata
  "config".transformer dict (read from the published safetensors
  headers) - on the meta device (weights never materialize).
- "cases": tiny LTXV architectures executed with deterministic
  hash-filled weights (unet_fill.py - the only rank-1 ``.weight``
  keys are the attention RMS q/k scales, so the rank rule holds)
  covering the v0.9 shape (scalar timestep, no mask) and the v0.9.5
  causal-temporal shape (per-token timestep, integer attention mask),
  plus initial/final guides with per-token denoise timesteps, scalar
  strength, and a spatial attention mask. The first transformer block's
  output is recorded independently.
  The head width 16 leaves a nonzero RoPE pad (16 % 6 = 4), so the
  front ones/zeros padding leg executes.

Determinism: attention is forced to the pytorch SDPA backend; the
reference's LTX RoPE and norms are inline torch at this baseline, so
no other backend can rotate.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_ltx_model_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

from comfy import conds as comfy_conds  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.ldm.lightricks.model import LTXVModel  # noqa: E402
from comfy.ldm.lightricks.symmetric_patchifier import (  # noqa: E402
    SymmetricPatchifier,
    latent_to_pixel_coords,
)
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy_extras.nodes_context_windows import LTXVContextWindowsNode  # noqa: E402
from golden_platform import tuple_provenance  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). The LTX
# module resolves comfy.ldm.modules.attention.optimized_attention as
# a module attribute at call time, so patching the module suffices.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert (
    sys.modules["comfy.ldm.modules.attention"].optimized_attention is _attention.attention_pytorch
), "the LTX reference is not calling the forced pytorch SDPA backend"
ATTENTION_BACKEND = "attention_pytorch"
ROPE_BACKEND = "reference inline torch (interleaved, float32 freq grid)"

OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "ltx_model_goldens.json"
)
CONTEXT_OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "ltxv_context_windows.json"
)

#: The verbatim metadata "config".transformer dicts of the two
#: published 2B checkpoints (safetensors headers of
#: ltx-video-2b-v0.9.safetensors and ltx-video-2b-v0.9.5.safetensors),
#: as the reference detection merges them over its state-dict facts.
_2B_V09_METADATA = {
    "_class_name": "Transformer3DModel",
    "_diffusers_version": "0.25.1",
    "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_type": "default",
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 28,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
    "project_to_2d_pos": True,
}

_2B_V095_METADATA = {
    key: value for key, value in _2B_V09_METADATA.items() if key != "project_to_2d_pos"
} | {"causal_temporal_positioning": True}

#: The reference detection's state-dict facts (comfy/model_detection.py
#: ltxv branch): block count and the attn2.to_k shape split into head
#: width and context width. The metadata dict updates over them.
_DETECTION_FACTS = {
    "num_layers": 28,
    "attention_head_dim": 64,
    "cross_attention_dim": 2048,
}

FULL_CONFIGS = {
    "ltxv_2b_v09": {**_DETECTION_FACTS, **_2B_V09_METADATA},
    "ltxv_2b_v095": {**_DETECTION_FACTS, **_2B_V095_METADATA},
}

#: Tiny geometry for executed parity. The hidden width
#: (heads * head_dim) must stay indivisible by 6 so the RoPE padding
#: leg executes, and must equal cross_attention_dim (the reference
#: views projected context rows as hidden rows).
_TINY = {
    "in_channels": 8,
    "cross_attention_dim": 16,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "caption_channels": 12,
    "num_layers": 2,
    "causal_temporal_positioning": False,
}

_TINY_CAUSAL = {**_TINY, "causal_temporal_positioning": True}

#: name -> (config, batch, frames, height, width, context_len,
#: frame_rate, per_token_timestep, masked).
CASES = {
    "ltxv_scalar": (_TINY, 2, 3, 4, 5, 7, 25.0, False, False),
    "ltxv_per_token": (_TINY_CAUSAL, 2, 2, 3, 4, 5, 24.0, True, True),
}

GUIDE_CASE = "ltxv_initial_final_guides"
GUIDE_BATCH = 2
GUIDE_TARGET_FRAMES = 3
GUIDE_HEIGHT = 2
GUIDE_WIDTH = 2
GUIDE_CONTEXT_LEN = 5
GUIDE_FRAME_RATE = 25.0
GUIDE_STRENGTHS = (0.5, 2.0)
GUIDE_FRAME_INDICES = (0, 16)
GUIDE_PATCHIFIER = SymmetricPatchifier(1, start_end=True)


def guide_coordinates(frame_index: int) -> torch.Tensor:
    guide = torch.zeros((GUIDE_BATCH, _TINY_CAUSAL["in_channels"], 1, GUIDE_HEIGHT, GUIDE_WIDTH))
    _, latent_coordinates = GUIDE_PATCHIFIER.patchify(guide)
    coordinates = latent_to_pixel_coords(latent_coordinates, (8, 32, 32), causal_fix=True)
    coordinates[:, 0] += frame_index
    return coordinates


def guide_case_payload() -> dict[str, object]:
    name = GUIDE_CASE
    config = _TINY_CAUSAL
    model = build_reference(config, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)

    total_frames = GUIDE_TARGET_FRAMES + len(GUIDE_FRAME_INDICES)
    x = hashed_input(
        f"{name}:x",
        (GUIDE_BATCH, config["in_channels"], total_frames, GUIDE_HEIGHT, GUIDE_WIDTH),
    )
    context = hashed_input(
        f"{name}:context", (GUIDE_BATCH, GUIDE_CONTEXT_LEN, config["caption_channels"])
    )
    attention_mask = torch.ones(GUIDE_BATCH, GUIDE_CONTEXT_LEN, dtype=torch.int64)
    attention_mask[1, -1] = 0
    mask_frames = torch.tensor((1.0, 0.5, 0.25, 0.5, 0.0), dtype=torch.float32)
    denoise_mask = mask_frames.reshape(1, 1, total_frames, 1, 1).expand(
        GUIDE_BATCH, 1, total_frames, GUIDE_HEIGHT, GUIDE_WIDTH
    )
    scalar_timestep = torch.tensor((0.35, 0.8), dtype=torch.float32)
    timestep = GUIDE_PATCHIFIER.patchify(
        denoise_mask * scalar_timestep.reshape(GUIDE_BATCH, 1, 1, 1, 1)
    )[0]
    coordinates = tuple(guide_coordinates(index) for index in GUIDE_FRAME_INDICES)
    keyframes = torch.cat(coordinates, dim=2)
    pixel_mask = torch.linspace(0.0, 1.0, 64 * 64, dtype=torch.float32).reshape(1, 1, 1, 64, 64)
    guide_entries = (
        {
            "pre_filter_count": GUIDE_HEIGHT * GUIDE_WIDTH,
            "strength": GUIDE_STRENGTHS[0],
            "pixel_mask": None,
            "latent_shape": (1, GUIDE_HEIGHT, GUIDE_WIDTH),
        },
        {
            "pre_filter_count": GUIDE_HEIGHT * GUIDE_WIDTH,
            "strength": GUIDE_STRENGTHS[1],
            "pixel_mask": pixel_mask,
            "latent_shape": (1, GUIDE_HEIGHT, GUIDE_WIDTH),
        },
    )

    block_outputs: dict[str, object] = {}
    hook = model.transformer_blocks[0].register_forward_hook(
        lambda _module, _inputs, output, record=block_outputs: record.update(block0=enc(output))
    )
    try:
        with torch.no_grad():
            output = model(
                x,
                timestep=timestep,
                context=context,
                attention_mask=attention_mask,
                frame_rate=GUIDE_FRAME_RATE,
                keyframe_idxs=keyframes,
                denoise_mask=denoise_mask,
                guide_attention_entries=guide_entries,
            )
    finally:
        hook.remove()
    return {
        "config": config,
        "state_dict": entries,
        "batch": GUIDE_BATCH,
        "frames": total_frames,
        "height": GUIDE_HEIGHT,
        "width": GUIDE_WIDTH,
        "context_len": GUIDE_CONTEXT_LEN,
        "frame_rate": GUIDE_FRAME_RATE,
        "timestep": timestep.tolist(),
        "attention_mask": attention_mask.tolist(),
        "denoise_mask": denoise_mask.tolist(),
        "guides": [
            {
                "keyframe_indices": current.tolist(),
                "latent_shape": [1, GUIDE_HEIGHT, GUIDE_WIDTH],
                "strength": strength,
                "attention_mask": (
                    None if position == 0 else {"kind": "linear", "shape": list(pixel_mask.shape)}
                ),
            }
            for position, (current, strength) in enumerate(
                zip(coordinates, GUIDE_STRENGTHS, strict=True)
            )
        ],
        "block_outputs": block_outputs,
        "output": enc(output),
    }


def build_reference(config: dict, device: str) -> LTXVModel:
    return LTXVModel(
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


def tensor_sha256(x: torch.Tensor) -> str:
    raw = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


class _ReferenceModelPatch:
    def __init__(self) -> None:
        self.model_options: dict = {}

    def clone(self) -> _ReferenceModelPatch:
        cloned = _ReferenceModelPatch()
        cloned.model_options = dict(self.model_options)
        return cloned

    def add_wrapper_with_key(self, *_args, **_kwargs) -> None:
        pass


class _HandlerModel:
    latent_format = SimpleNamespace(temporal_downscale_ratio=8)

    def resize_cond_for_context_window(self, *_args, **_kwargs):
        return None


def context_window_payload(commit: str) -> dict:
    config = _TINY_CAUSAL
    model = build_reference(config, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)

    input_name = "ltxv_context_windows:x"
    context_name = "ltxv_context_windows:context"
    x = hashed_input(input_name, (1, config["in_channels"], 7, 2, 2))
    context = hashed_input(context_name, (1, 5, config["caption_channels"]))
    timestep = torch.tensor([0.5], dtype=torch.float32)
    attention_mask = torch.ones((1, 5), dtype=torch.int64)
    frame_rate = 25.0
    handler_model = _HandlerModel()
    conds = [
        [
            {
                "model_conds": {
                    "c_crossattn": comfy_conds.CONDRegular(context),
                    "attention_mask": comfy_conds.CONDRegular(attention_mask),
                }
            }
        ]
    ]
    cases: dict[str, object] = {}

    for name, retain_first_frame in (
        ("baseline", False),
        ("retain_first_frame", True),
    ):
        patched = LTXVContextWindowsNode.execute(
            _ReferenceModelPatch(),
            context_length=17,
            context_overlap=8,
            context_schedule="standard_uniform",
            context_stride=1,
            closed_loop=False,
            fuse_method="pyramid",
            freenoise=True,
            retain_first_frame=retain_first_frame,
        )[0]
        handler = patched.model_options["context_handler"]
        model_options = {"transformer_options": {"sample_sigmas": timestep}}
        windows = [
            window.index_list
            for window in handler.get_context_windows(handler_model, x, model_options)
        ]

        def evaluate(_model, sub_conds, latent, sub_timestep, _model_options):
            (actual,) = sub_conds
            model_conds = actual[0]["model_conds"]
            sub_context = model_conds["c_crossattn"].cond
            sub_attention_mask = model_conds["attention_mask"].cond
            assert torch.equal(sub_context, context)
            assert torch.equal(sub_attention_mask, attention_mask)
            return [
                model(
                    latent,
                    timestep=sub_timestep,
                    context=sub_context,
                    attention_mask=sub_attention_mask,
                    frame_rate=frame_rate,
                )
            ]

        with torch.no_grad():
            (output,) = handler.execute(
                evaluate,
                handler_model,
                conds,
                x,
                timestep,
                model_options,
            )
        cases[name] = {
            "retain_first_frame": retain_first_frame,
            "spec": {
                "schedule": handler.context_schedule.name,
                "fuse_method": handler.fuse_method.name,
                "length": handler.context_length,
                "overlap": handler.context_overlap,
                "stride": handler.context_stride,
                "closed_loop": handler.closed_loop,
                "dim": handler.dim,
                "freenoise": handler.freenoise,
                "causal_anchor": handler.causal_window_fix,
                "cond_retain_indices": [],
                "latent_retain_indices": [0] if retain_first_frame else [],
            },
            "reference_handler": {
                "cond_retain_indices": handler.cond_retain_index_list,
                "latent_retain_indices": handler.latent_retain_index_list,
            },
            "windows": windows,
            "output": enc(output),
            "output_sha256": tensor_sha256(output),
        }

    return {
        "_meta": {
            "generator": "tools/gen_ltx_model_goldens.py",
            **tuple_provenance(torch.__version__, pin_cpu=True),
        },
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "node": "LTXVContextWindows",
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
            "rope": ROPE_BACKEND,
        },
        "model": {
            "config": config,
            "state_dict": entries,
        },
        "input": {
            "name": input_name,
            "shape": list(x.shape),
            "context_name": context_name,
            "context_shape": list(context.shape),
            "timestep": timestep.tolist(),
            "attention_mask": attention_mask.tolist(),
            "frame_rate": frame_rate,
        },
        "cases": cases,
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
    module_file = Path(sys.modules[LTXVModel.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference LTXVModel was imported from"
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

    for name, case in sorted(CASES.items()):
        (
            config,
            batch,
            frames,
            height,
            width,
            context_len,
            frame_rate,
            per_token,
            masked,
        ) = case
        model = build_reference(config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        x = hashed_input(f"{name}:x", (batch, config["in_channels"], frames, height, width))
        tokens = frames * height * width
        if per_token:
            timestep = torch.linspace(0.05, 0.95, batch * tokens, dtype=torch.float32).reshape(
                batch, tokens
            )
        else:
            timestep = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
        context = hashed_input(f"{name}:context", (batch, context_len, config["caption_channels"]))
        attention_mask = None
        if masked:
            attention_mask = torch.ones(batch, context_len, dtype=torch.int64)
            attention_mask[0, -1] = 0
            attention_mask[1, -2:] = 0

        block_outputs: dict[str, object] = {}
        hook = model.transformer_blocks[0].register_forward_hook(
            lambda _module, _inputs, output, record=block_outputs: record.update(block0=enc(output))
        )
        try:
            with torch.no_grad():
                out = model(
                    x,
                    timestep=timestep,
                    context=context,
                    attention_mask=attention_mask,
                    frame_rate=frame_rate,
                )
        finally:
            hook.remove()
        payload["cases"][name] = {
            "config": config,
            "state_dict": entries,
            "batch": batch,
            "frames": frames,
            "height": height,
            "width": width,
            "context_len": context_len,
            "frame_rate": frame_rate,
            "timestep": timestep.tolist(),
            "attention_mask": None if attention_mask is None else attention_mask.tolist(),
            "block_outputs": block_outputs,
            "output": enc(out),
        }

    payload["cases"][GUIDE_CASE] = guide_case_payload()

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    CONTEXT_OUT.write_text(json.dumps(context_window_payload(commit), indent=1) + "\n")
    print(f"wrote {CONTEXT_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
