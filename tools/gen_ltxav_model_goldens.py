"""Generate LTX-2 audio-video diffusion transformer goldens from ComfyUI.

Runs the REFERENCE LTXAVModel (comfy/ldm/lightricks/av_model.py @ the
audited baseline) and writes
packages/dinkster-inference-torch/tests/goldens/ltxav_model_goldens.json.
dinkster_inference.ltx (the LTXAV config/layout half) and
dinkster_inference_torch.ltxav_model are pinned against these outputs;
the oracle is the executed reference, never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listing of the
  FULL-SIZE 19B and LTX-2.3 22B models, built exactly as the reference
  loads them from each checkpoint's verbatim metadata
  "config".transformer dict. The 19B listing includes 58 text-side
  connector keys that Dinkster filters; the 22B listing includes its
  diffusion-owned asymmetric connector towers.
- "cases": tiny LTXAV architectures executed with deterministic
  hash-filled weights (unet_fill.py - the only rank-1 ``.weight``
  keys are the attention RMS q/k scales and norm scales, so the rank
  rule holds), covering scalar and per-token timesteps, the per-frame
  compressed-timestep path, spatially varying denoise masks (the
  uncompressed path), video-only and audio-only denoising, empty
  audio, reference-audio injection, and the cross-attention-adaln
  variant. The first transformer block's video and audio outputs are
  recorded independently. The video head width 16 leaves a nonzero
  split-RoPE pad, so the front ones/zeros padding leg executes.
- "pack_example": comfy.utils.pack_latents applied to a hashed
  video/audio pair, pinning the packed single-tensor wire format that
  pack_av_latents/unpack_av_latents mirror.

Determinism: attention is forced to the pytorch SDPA backend; the
reference's split RoPE builds its frequency ladder in float64 via
numpy at this baseline, so no other backend can rotate.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_ltxav_model_goldens.py

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

import comfy.utils  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.ldm.lightricks.av_model import LTXAVModel  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). The LTX
# modules resolve comfy.ldm.modules.attention.optimized_attention as
# a module attribute at call time, so patching the module suffices.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
assert (
    sys.modules["comfy.ldm.modules.attention"].optimized_attention is _attention.attention_pytorch
), "the LTXAV reference is not calling the forced pytorch SDPA backend"
ATTENTION_BACKEND = "attention_pytorch"
ROPE_BACKEND = "reference split rope (float64 numpy frequency ladder)"

OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "ltxav_model_goldens.json"
)

#: The verbatim metadata "config".transformer dict of the published
#: 19B checkpoint (safetensors header of ltx-2-19b-dev.safetensors),
#: as the reference detection merges it over its state-dict facts.
_19B_METADATA = {
    "_class_name": "AVTransformer3DModel",
    "_diffusers_version": "0.25.1",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_type": "default",
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "audio_num_attention_heads": 32,
    "audio_out_channels": 128,
    "audio_positional_embedding_max_pos": [20],
    "av_ca_timestep_scale_multiplier": 1000.0,
    "av_cross_ada_norm": True,
    "caption_channels": 3840,
    "causal_temporal_positioning": True,
    "connector_attention_head_dim": 128,
    "connector_norm_output": True,
    "connector_num_attention_heads": 30,
    "connector_num_layers": 2,
    "connector_num_learnable_registers": 128,
    "connector_positional_embedding_max_pos": [4096],
    "cross_attention_dim": 4096,
    "cross_attention_norm": True,
    "double_self_attention": False,
    "dropout": 0.0,
    "frequencies_precision": "float64",
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 48,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "positional_embedding_theta": 10000.0,
    "positional_embedding_type": "rope",
    "qk_norm": "rms_norm",
    "rope_type": "split",
    "share_ff": False,
    "standardization_norm": "rms_norm",
    "timestep_scale_multiplier": 1000,
    "upcast_attention": False,
    "use_audio_video_cross_attention": True,
    "use_embeddings_connector": True,
    "use_linear_projection": False,
    "use_middle_indices_grid": True,
}

_22B_V23_METADATA = {
    "_class_name": "AVTransformer3DModel",
    "activation_fn": "gelu-approximate",
    "apply_gated_attention": True,
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_type": "default",
    "audio_attention_head_dim": 64,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "audio_cross_attention_dim": 2048,
    "audio_num_attention_heads": 32,
    "audio_out_channels": 128,
    "audio_positional_embedding_max_pos": [20],
    "av_ca_timestep_scale_multiplier": 1000.0,
    "av_cross_ada_norm": True,
    "caption_channels": 3840,
    "caption_proj_before_connector": True,
    "caption_proj_input_norm": False,
    "caption_projection_first_linear": False,
    "caption_projection_second_linear": False,
    "causal_temporal_positioning": True,
    "connector_apply_gated_attention": True,
    "connector_attention_head_dim": 128,
    "connector_learnable_registers_std": 1,
    "connector_norm_output": True,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_num_learnable_registers": 128,
    "connector_positional_embedding_max_pos": [4096],
    "cross_attention_adaln": True,
    "cross_attention_dim": 4096,
    "cross_attention_norm": True,
    "double_self_attention": False,
    "dropout": 0.0,
    "frequencies_precision": "float64",
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 48,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "positional_embedding_theta": 10000.0,
    "positional_embedding_type": "rope",
    "qk_norm": "rms_norm",
    "rope_type": "split",
    "share_ff": False,
    "standardization_norm": "rms_norm",
    "text_encoder_norm_type": "per_token_rms",
    "timestep_scale_multiplier": 1000,
    "upcast_attention": False,
    "use_audio_video_cross_attention": True,
    "use_embeddings_connector": True,
    "use_linear_projection": False,
    "use_middle_indices_grid": True,
}

#: The reference detection's state-dict facts (comfy/model_detection.py
#: ltxav branch): block count and the attn2.to_k shape split into head
#: width and context width. The metadata dict updates over them.
_DETECTION_FACTS = {
    "num_layers": 48,
    "attention_head_dim": 128,
    "cross_attention_dim": 4096,
}

FULL_CONFIGS = {
    "ltxav_19b": {**_DETECTION_FACTS, **_19B_METADATA},
    "ltxav_22b_v23": {**_DETECTION_FACTS, **_22B_V23_METADATA},
}

#: Tiny geometry for executed parity. The video hidden width
#: (heads * head_dim = 16) must equal cross_attention_dim and leave a
#: nonzero split-RoPE pad; the audio hidden width (8) must equal
#: audio_cross_attention_dim; audio_in_channels is pinned to the
#: 8-channel x 16-bin mel layout the reference hardcodes.
_TINY_AV = {
    "in_channels": 8,
    "audio_in_channels": 128,
    "cross_attention_dim": 16,
    "audio_cross_attention_dim": 8,
    "attention_head_dim": 8,
    "audio_attention_head_dim": 4,
    "num_attention_heads": 2,
    "audio_num_attention_heads": 2,
    "caption_channels": 12,
    "num_layers": 2,
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "audio_positional_embedding_max_pos": [20],
    "causal_temporal_positioning": True,
    "use_middle_indices_grid": True,
    "timestep_scale_multiplier": 1000,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "cross_attention_adaln": False,
    "connector_attention_head_dim": 4,
    "connector_num_attention_heads": 2,
    "connector_num_layers": 1,
}

_TINY_V23 = {
    **_TINY_AV,
    "apply_gated_attention": True,
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "connector_apply_gated_attention": True,
    "connector_attention_head_dim": 8,
    "connector_num_attention_heads": 2,
    "audio_connector_attention_head_dim": 4,
    "audio_connector_num_attention_heads": 2,
    "cross_attention_adaln": True,
}

#: name -> case spec. Masks are literal value ladders broadcast to the
#: latent shapes; "temporal" video masks are constant within each
#: frame (the compressed per-frame timestep path), "spatial" masks
#: vary within a frame (compression disabled). Timesteps for masked
#: cases are derived through the reference model's own patchifiers,
#: exactly as comfy/model_base.py LTXAV.process_timestep does.
CASES: dict[str, dict] = {
    "ltxav_22b_v23": {
        "config": _TINY_V23,
        "frames": 1,
        "height": 1,
        "width": 1,
        "audio_length": 2,
        "context_len": 3,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
    },
    "ltxav_t2av_scalar": {
        "config": _TINY_AV,
        "frames": 3,
        "height": 2,
        "width": 2,
        "audio_length": 5,
        "context_len": 7,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
    },
    "ltxav_per_frame": {
        "config": _TINY_AV,
        "frames": 4,
        "height": 2,
        "width": 2,
        "audio_length": 6,
        "context_len": 5,
        "frame_rate": 24.0,
        "timestep_arg": "tuple",
        "video_mask": "temporal",
        "video_mask_frames": [1.0, 1.0, 0.5, 0.0],
        "audio_mask_times": [1.0, 1.0, 1.0, 0.3, 0.3, 0.0],
        "attention_mask": True,
        "pass_denoise_mask": True,
    },
    "ltxav_spatial_mask": {
        "config": {**_TINY_AV, "av_ca_timestep_scale_multiplier": 500.0},
        "frames": 2,
        "height": 2,
        "width": 3,
        "audio_length": 4,
        "context_len": 6,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
        "video_mask": "spatial",
        "audio_mask_times": [1.0, 0.5, 0.5, 0.0],
        "pass_denoise_mask": True,
    },
    "ltxav_video_only": {
        "config": _TINY_AV,
        "frames": 3,
        "height": 2,
        "width": 2,
        "audio_length": 5,
        "context_len": 4,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
        "audio_mask_times": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "ltxav_audio_only": {
        "config": _TINY_AV,
        "frames": 3,
        "height": 2,
        "width": 2,
        "audio_length": 5,
        "context_len": 5,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
        "video_mask": "temporal",
        "video_mask_frames": [0.0, 0.0, 0.0],
        "pass_denoise_mask": True,
    },
    "ltxav_empty_audio": {
        "config": _TINY_AV,
        "frames": 3,
        "height": 2,
        "width": 2,
        "audio_length": 0,
        "context_len": 6,
        "frame_rate": 25.0,
        "timestep_arg": "single",
    },
    "ltxav_ref_audio": {
        "config": _TINY_AV,
        "frames": 3,
        "height": 2,
        "width": 2,
        "audio_length": 4,
        "context_len": 5,
        "frame_rate": 30.0,
        "timestep_arg": "tuple",
        "ref_audio_tokens": [1, 3, 128],
    },
    "ltxav_cross_attention_adaln": {
        "config": {
            **_TINY_AV,
            "cross_attention_adaln": True,
            "use_middle_indices_grid": False,
            "causal_temporal_positioning": False,
        },
        "frames": 2,
        "height": 2,
        "width": 2,
        "audio_length": 3,
        "context_len": 5,
        "frame_rate": 25.0,
        "timestep_arg": "tuple",
        "attention_mask": True,
    },
}

BATCH = 2


def build_reference(config: dict, device: str) -> LTXAVModel:
    return LTXAVModel(
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


def video_mask_tensor(spec: dict) -> torch.Tensor | None:
    kind = spec.get("video_mask")
    if kind is None:
        return None
    frames, height, width = spec["frames"], spec["height"], spec["width"]
    if kind == "temporal":
        values = torch.tensor(spec["video_mask_frames"], dtype=torch.float32)
        return values.view(1, 1, frames, 1, 1).expand(BATCH, 1, frames, height, width).clone()
    assert kind == "spatial"
    mask = torch.full((BATCH, 1, frames, height, width), 0.25, dtype=torch.float32)
    mask[:, :, 0, 0, 0] = 1.0
    mask[:, :, 1:] = 0.6
    return mask


def audio_mask_tensor(spec: dict) -> torch.Tensor | None:
    times = spec.get("audio_mask_times")
    if times is None:
        return None
    values = torch.tensor(times, dtype=torch.float32)
    return values.view(1, 1, -1, 1).expand(BATCH, 8, len(times), 16).clone()


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
    module_file = Path(sys.modules[LTXAVModel.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "the reference LTXAVModel was imported from"
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

    for name, spec in sorted(CASES.items()):
        config = spec["config"]
        frames, height, width = spec["frames"], spec["height"], spec["width"]
        audio_length = spec["audio_length"]
        model = build_reference(config, "cpu")
        entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(entries), strict=True)

        vx = hashed_input(f"{name}:x", (BATCH, config["in_channels"], frames, height, width))
        if audio_length:
            ax = hashed_input(f"{name}:audio", (BATCH, 8, audio_length, 16))
        else:
            ax = torch.zeros(BATCH, 8, 0, 16, dtype=torch.float32)
        context_width = (
            config["cross_attention_dim"] + config["audio_cross_attention_dim"]
            if config.get("caption_proj_before_connector", False)
            else 2 * config["caption_channels"]
        )
        context = hashed_input(f"{name}:context", (BATCH, spec["context_len"], context_width))
        reference_context = (
            model.preprocess_text_embeds(context, unprocessed=True)
            if config.get("caption_proj_before_connector", False)
            else context
        )
        t = torch.linspace(0.05, 0.95, BATCH, dtype=torch.float32)

        video_mask = video_mask_tensor(spec)
        audio_mask = audio_mask_tensor(spec)
        if video_mask is not None:
            v_timestep = model.patchifier.patchify((video_mask * t.view(BATCH, 1, 1, 1, 1))[:, :1])[
                0
            ]
        else:
            v_timestep = t
        if audio_mask is not None:
            a_timestep = model.a_patchifier.patchify(
                (audio_mask * t.view(BATCH, 1, 1, 1))[:, :1, :, :1]
            )[0]
        else:
            a_timestep = t

        attention_mask = None
        if spec.get("attention_mask"):
            attention_mask = torch.ones(BATCH, spec["context_len"], dtype=torch.int64)
            attention_mask[0, -1] = 0
            attention_mask[1, -2:] = 0

        ref_audio = None
        if "ref_audio_tokens" in spec:
            ref_audio = {
                "tokens": hashed_input(f"{name}:ref_audio", tuple(spec["ref_audio_tokens"]))
            }

        if spec["timestep_arg"] == "tuple":
            timestep_arg: object = (v_timestep, a_timestep)
        else:
            assert spec["timestep_arg"] == "single"
            timestep_arg = v_timestep

        kwargs: dict = {}
        if ref_audio is not None:
            kwargs["ref_audio"] = ref_audio

        block_outputs: dict[str, object] = {}

        def record_block(_module, _inputs, output, record=block_outputs):
            record["block0_video"] = enc(output[0])
            record["block0_audio"] = enc(output[1])

        hook = model.transformer_blocks[0].register_forward_hook(record_block)
        try:
            with torch.no_grad():
                out = model(
                    [vx, ax] if audio_length else [vx],
                    timestep=timestep_arg,
                    context=reference_context,
                    attention_mask=attention_mask,
                    frame_rate=spec["frame_rate"],
                    transformer_options={},
                    denoise_mask=video_mask if spec.get("pass_denoise_mask") else None,
                    **kwargs,
                )
        finally:
            hook.remove()

        if isinstance(out, list):
            output_video, output_audio = out
        else:
            output_video, output_audio = out, None

        payload["cases"][name] = {
            "config": config,
            "state_dict": entries,
            "batch": BATCH,
            "frames": frames,
            "height": height,
            "width": width,
            "audio_length": audio_length,
            "context_len": spec["context_len"],
            "frame_rate": spec["frame_rate"],
            "timestep_arg": spec["timestep_arg"],
            "timestep": v_timestep.tolist(),
            "audio_timestep": a_timestep.tolist(),
            "denoise_mask": (
                video_mask.tolist()
                if video_mask is not None and spec.get("pass_denoise_mask")
                else None
            ),
            "attention_mask": None if attention_mask is None else attention_mask.tolist(),
            "ref_audio_tokens": spec.get("ref_audio_tokens"),
            "block_outputs": block_outputs,
            "output_video": enc(output_video),
            "output_audio": None if output_audio is None else enc(output_audio),
        }

    pack_video = hashed_input("pack:video", (2, 4, 3, 2, 2))
    pack_audio = hashed_input("pack:audio", (2, 8, 5, 16))
    packed, shapes = comfy.utils.pack_latents([pack_video, pack_audio])
    payload["pack_example"] = {
        "video_shape": list(pack_video.shape),
        "audio_shape": list(pack_audio.shape),
        "packed": enc(packed),
        "latent_shapes": [list(shape) for shape in shapes],
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
