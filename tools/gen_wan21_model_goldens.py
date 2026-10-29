"""Generate core Wan DiT goldens by executing pinned ComfyUI b78cec87.

The source tree is materialized from the exact git object with ``git archive``;
the current ComfyUI checkout need not move to the audited commit. Run from the
Dinkster root with Python 3.12 and torch available:

    python tools/gen_wan21_model_goldens.py

Set ``COMFYUI_ROOT`` when ComfyUI is not beside Dinkster (or beside its parent,
as in station delegate clones).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from golden_platform import platform_golden_path, tuple_provenance

REPO = Path(__file__).resolve().parent.parent
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "wan21_model_goldens.json"
)


def _comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "ComfyUI",
        REPO.parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set COMFYUI_ROOT to the ComfyUI git checkout")


COMFY_ROOT = _comfy_root()


def _git(*args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=COMFY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _enc(value: Any) -> dict[str, Any]:
    import torch

    assert isinstance(value, torch.Tensor)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.float().flatten().tolist(),
    }


def main() -> None:
    resolved = _git("rev-parse", REFERENCE_COMMIT).decode().strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {REFERENCE_COMMIT}")

    archive = _git("archive", "--format=tar", REFERENCE_COMMIT)
    with tempfile.TemporaryDirectory(prefix="dinkster-wan21-reference-") as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
            payload.extractall(source, filter="data")
        sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
        sys.path.insert(0, str(source))
        sys.argv.append("--cpu")

        import comfy.options
        import torch

        torch.set_num_threads(1)
        comfy.options.enable_args_parsing()
        from comfy import model_management, ops
        from comfy.ldm.modules import attention as attention_module
        from comfy.ldm.wan import model as wan_module
        from comfy.ldm.wan import model_animate as animate_module
        from comfy.ldm.wan import model_multitalk as multitalk_module
        from unet_fill import fill_state_dict, hashed_input

        class MultiTalkModelPatch(torch.nn.Module):
            def __init__(
                self,
                audio_window: int = 5,
                intermediate_dim: int = 512,
                in_dim: int = 5120,
                out_dim: int = 768,
                context_tokens: int = 32,
                vae_scale: int = 4,
                num_layers: int = 40,
                *,
                device: str,
                dtype: torch.dtype,
                operations: Any,
            ) -> None:
                super().__init__()
                self.audio_proj = multitalk_module.MultiTalkAudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window + vae_scale - 1,
                    intermediate_dim=intermediate_dim,
                    out_dim=out_dim,
                    context_tokens=context_tokens,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                self.blocks = torch.nn.ModuleList(
                    multitalk_module.WanMultiTalkAttentionBlock(
                        in_dim,
                        out_dim,
                        device=device,
                        dtype=dtype,
                        operations=operations,
                    )
                    for _ in range(num_layers)
                )

        wan_module.optimized_attention = attention_module.attention_pytorch
        animate_module.optimized_attention = attention_module.attention_pytorch
        multitalk_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention_masked = attention_module.attention_pytorch
        model_management.in_training = True

        tiny_base = {
            "patch_size": (1, 2, 2),
            "dim": 24,
            "ffn_dim": 48,
            "freq_dim": 8,
            "text_dim": 12,
            "out_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
        }
        full = {
            "t2v_1_3b": {
                **tiny_base,
                "model_type": "t2v",
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 12,
                "num_layers": 30,
            },
            "t2v_14b": {
                **tiny_base,
                "model_type": "t2v",
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
            },
            "i2v_14b": {
                **tiny_base,
                "model_type": "i2v",
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
            },
            "flf_i2v_14b": {
                **tiny_base,
                "model_type": "i2v",
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
                "flf_pos_embed_token_number": 514,
            },
            "wan22_i2v_14b": {
                **tiny_base,
                "model_type": "t2v",
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
            },
            "animate_14b": {
                **tiny_base,
                "model_type": "animate",
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
            },
            "s2v_14b": {
                **tiny_base,
                "model_type": "s2v",
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
            },
            "humo_17b": {
                **tiny_base,
                "model_type": "humo",
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
            },
            "ti2v_5b": {
                **tiny_base,
                "model_type": "t2v",
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
            },
            "vace_1_3b": {
                **tiny_base,
                "model_type": "vace",
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 12,
                "num_layers": 30,
                "vace_in_dim": 96,
                "vace_layers": 15,
            },
            "vace_14b": {
                **tiny_base,
                "model_type": "vace",
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "num_layers": 40,
                "vace_in_dim": 96,
                "vace_layers": 8,
            },
        }
        layouts: dict[str, list[tuple[str, list[int]]]] = {}
        for name, config in full.items():
            with torch.device("meta"):
                model_class = (
                    wan_module.VaceWanModel
                    if config["model_type"] == "vace"
                    else animate_module.AnimateWanModel
                    if config["model_type"] == "animate"
                    else wan_module.WanModel_S2V
                    if config["model_type"] == "s2v"
                    else wan_module.HumoWanModel
                    if config["model_type"] == "humo"
                    else wan_module.WanModel
                )
                model = model_class(
                    operations=ops.disable_weight_init,
                    device="meta",
                    dtype=torch.float32,
                    **config,
                )
            layouts[name] = sorted(
                (key, list(value.shape)) for key, value in model.state_dict().items()
            )
        with torch.device("meta"):
            multitalk_patch = MultiTalkModelPatch(
                device="meta",
                dtype=torch.float32,
                operations=ops.disable_weight_init,
            )
        layouts["multitalk_patch"] = sorted(
            (key, list(value.shape)) for key, value in multitalk_patch.state_dict().items()
        )

        cases: dict[str, dict[str, Any]] = {}
        for (
            name,
            model_type,
            channels,
            out_channels,
            timestep_values,
            flf_tokens,
            reference_frames,
            context_latent_shapes,
        ) in (
            ("t2v_reduced", "t2v", 16, 16, (0.375,), None, None, ()),
            ("phantom_t2v_reduced", "t2v", 16, 16, (0.375,), None, 3, ()),
            ("i2v_36_reduced", "i2v", 36, 16, (0.375,), None, None, ()),
            ("flf_i2v_36_reduced", "i2v", 36, 16, (0.375,), 4, None, ()),
            (
                "wan22_i2v_36_reduced",
                "t2v",
                36,
                16,
                (0.375,),
                None,
                None,
                (),
            ),
            ("ti2v_48_reduced", "t2v", 48, 48, ((0.375, 0.125),), None, None, ()),
            ("vace_reduced", "vace", 16, 16, (0.375,), None, None, ()),
            (
                "bernini_reduced",
                "t2v",
                16,
                16,
                (0.375,),
                None,
                None,
                ((1, 16, 2, 3, 5), (1, 16, 1, 2, 2)),
            ),
        ):
            config = {
                **tiny_base,
                "model_type": model_type,
                "in_dim": channels,
                "out_dim": out_channels,
                **({"flf_pos_embed_token_number": flf_tokens} if flf_tokens is not None else {}),
                **({"vace_in_dim": 96, "vace_layers": 2} if model_type == "vace" else {}),
            }
            model_class = wan_module.VaceWanModel if model_type == "vace" else wan_module.WanModel
            model = model_class(
                operations=ops.disable_weight_init,
                device="cpu",
                dtype=torch.float32,
                **config,
            )
            state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
            model.load_state_dict(fill_state_dict(state), strict=True)
            spatial = (6, 6) if model_type == "vace" else (5, 6)
            x = hashed_input(f"{name}:x", (1, channels, 2, *spatial))
            timestep = torch.tensor(timestep_values, dtype=torch.float32)
            context = hashed_input(f"{name}:context", (1, 4, 12))
            vision = hashed_input(f"{name}:vision", (1, 3, 1280)) if model_type == "i2v" else None
            vace_context = (
                hashed_input(f"{name}:vace", (1, 2, 96, 2, *spatial))
                if model_type == "vace"
                else None
            )
            vace_strength = (0.25, 1.5) if model_type == "vace" else None
            temporal_reference = (
                hashed_input(
                    f"{name}:temporal-reference", (1, channels, reference_frames, *spatial)
                )
                if reference_frames is not None
                else None
            )
            context_latents = tuple(
                hashed_input(f"{name}:context-latent:{index}", shape)
                for index, shape in enumerate(context_latent_shapes)
            )
            forward_kwargs = (
                {"vace_context": vace_context, "vace_strength": vace_strength}
                if vace_context is not None
                else {"time_dim_concat": temporal_reference}
                if temporal_reference is not None
                else {"context_latents": context_latents}
                if context_latents
                else {}
            )
            output = model._forward(x, timestep, context, clip_fea=vision, **forward_kwargs)
            cases[name] = {
                "config": {
                    "model_type": "ti2v"
                    if channels == 48
                    else "t2v"
                    if model_type == "vace"
                    else model_type,
                    "in_channels": channels,
                    "hidden_size": 24,
                    "ffn_hidden_size": 48,
                    "num_heads": 2,
                    "num_layers": 2,
                    "text_dim": 12,
                    "time_freq_dim": 8,
                    "out_channels": out_channels,
                    **(
                        {"flf_pos_embed_token_number": flf_tokens} if flf_tokens is not None else {}
                    ),
                    **({"vace_layers": 2} if model_type == "vace" else {}),
                },
                "state_dict": state,
                "input_shape": list(x.shape),
                "timesteps": timestep.tolist(),
                "context_shape": list(context.shape),
                "vision_shape": None if vision is None else list(vision.shape),
                **(
                    {"temporal_reference_shape": list(temporal_reference.shape)}
                    if temporal_reference is not None
                    else {}
                ),
                **(
                    {"context_latent_shapes": [list(latent.shape) for latent in context_latents]}
                    if context_latents
                    else {}
                ),
                "vace_context_shape": None if vace_context is None else list(vace_context.shape),
                "vace_strength": vace_strength,
                "output": _enc(output),
            }

        name = "s2v_reduced"
        config = {
            **tiny_base,
            "model_type": "s2v",
            "in_dim": 16,
        }
        model = wan_module.WanModel_S2V(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **config,
        )
        state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(state), strict=True)
        x = hashed_input(f"{name}:x", (1, 16, 2, 8, 8))
        timestep = torch.tensor((0.375,), dtype=torch.float32)
        context = hashed_input(f"{name}:context", (1, 4, 12))
        audio = hashed_input(f"{name}:audio", (1, 25, 1024, 8))
        reference = hashed_input(f"{name}:reference", (1, 16, 1, 8, 8))
        motion = hashed_input(f"{name}:motion", (1, 16, 19, 8, 8))
        control = hashed_input(f"{name}:control", (1, 16, 2, 8, 8))
        output = model._forward(
            x,
            timestep,
            context,
            audio_embed=audio,
            reference_latent=reference,
            control_video=control,
            reference_motion=motion,
        )
        cases[name] = {
            "config": {
                "model_type": "t2v",
                "model_variant": "s2v",
                "in_channels": 16,
                "hidden_size": 24,
                "ffn_hidden_size": 48,
                "num_heads": 2,
                "num_layers": 2,
                "text_dim": 12,
                "time_freq_dim": 8,
                "out_channels": 16,
            },
            "state_dict": state,
            "input_shape": list(x.shape),
            "timesteps": timestep.tolist(),
            "context_shape": list(context.shape),
            "vision_shape": None,
            "audio_shape": list(audio.shape),
            "reference_latent_shape": list(reference.shape),
            "reference_motion_shape": list(motion.shape),
            "control_video_shape": list(control.shape),
            "vace_context_shape": None,
            "vace_strength": None,
            "output": _enc(output),
        }

        name = "humo_reduced"
        config = {
            **tiny_base,
            "model_type": "humo",
            "in_dim": 36,
        }
        model = wan_module.HumoWanModel(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **config,
        )
        state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(state), strict=True)
        x = hashed_input(f"{name}:x", (1, 36, 2, 4, 6))
        timestep = torch.tensor((0.375,), dtype=torch.float32)
        context = hashed_input(f"{name}:context", (1, 4, 12))
        audio = hashed_input(f"{name}:audio", (1, 2, 8, 5, 1280))
        reference = hashed_input(f"{name}:reference", (1, 36, 1, 4, 6))
        freqs = model.rope_encode(2, 4, 6, device=x.device, dtype=x.dtype)
        output = model.forward_orig(
            x,
            timestep,
            context,
            freqs=freqs,
            audio_embed=audio,
            reference_latent=reference,
        )
        cases[name] = {
            "config": {
                "model_type": "t2v",
                "model_variant": "humo",
                "in_channels": 36,
                "hidden_size": 24,
                "ffn_hidden_size": 48,
                "num_heads": 2,
                "num_layers": 2,
                "text_dim": 12,
                "time_freq_dim": 8,
                "out_channels": 16,
            },
            "state_dict": state,
            "input_shape": list(x.shape),
            "timesteps": timestep.tolist(),
            "context_shape": list(context.shape),
            "vision_shape": None,
            "audio_shape": list(audio.shape),
            "reference_latent_shape": list(reference.shape),
            "vace_context_shape": None,
            "vace_strength": None,
            "output": _enc(output),
        }

        name = "infinite_talk_reduced"
        config = {
            **tiny_base,
            "model_type": "i2v",
            "in_dim": 36,
            "dim": 80,
            "ffn_dim": 160,
        }
        model = wan_module.WanModel(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **config,
        )
        state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(state), strict=True)
        patch_config = {
            "audio_window": 5,
            "intermediate_dim": 4,
            "in_dim": 80,
            "out_dim": 6,
            "context_tokens": 2,
            "vae_scale": 4,
            "num_layers": 2,
        }
        multitalk_patch = MultiTalkModelPatch(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **patch_config,
        )
        patch_state = sorted(
            (key, list(value.shape)) for key, value in multitalk_patch.state_dict().items()
        )
        multitalk_patch.load_state_dict(fill_state_dict(patch_state), strict=True)
        x = hashed_input(f"{name}:x", (1, 36, 2, 4, 4))
        timestep = torch.tensor((0.375,), dtype=torch.float32)
        context = hashed_input(f"{name}:context", (1, 4, 12))
        vision = hashed_input(f"{name}:vision", (1, 3, 1280))
        audio_shapes = ((5, 12, 768), (5, 12, 768))
        audio = tuple(
            hashed_input(f"{name}:audio:{index}", shape) for index, shape in enumerate(audio_shapes)
        )
        projected_audio = multitalk_module.project_audio_features(
            multitalk_patch.audio_proj,
            audio,
            0,
            5,
        )
        target_masks = torch.tensor(
            ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)),
            dtype=torch.float32,
        )
        strength = 0.75
        transformer_options = {
            "audio_embeds": projected_audio,
            "patches": {
                "attn1_patch": [multitalk_module.MultiTalkGetAttnMapPatch(target_masks)],
                "attn2_patch": [
                    multitalk_module.MultiTalkCrossAttnPatch(
                        SimpleNamespace(model=multitalk_patch), strength
                    )
                ],
            },
        }
        output = model._forward(
            x,
            timestep,
            context,
            clip_fea=vision,
            transformer_options=transformer_options,
        )
        cases[name] = {
            "config": {
                "model_type": "i2v",
                "in_channels": 36,
                "hidden_size": 80,
                "ffn_hidden_size": 160,
                "num_heads": 2,
                "num_layers": 2,
                "text_dim": 12,
                "time_freq_dim": 8,
                "out_channels": 16,
            },
            "state_dict": state,
            "patch_config": patch_config,
            "patch_state_dict": patch_state,
            "input_shape": list(x.shape),
            "timesteps": timestep.tolist(),
            "context_shape": list(context.shape),
            "vision_shape": list(vision.shape),
            "audio_shapes": [list(shape) for shape in audio_shapes],
            "audio_start": 0,
            "audio_end": 5,
            "target_masks": target_masks.tolist(),
            "strength": strength,
            "vace_context_shape": None,
            "vace_strength": None,
            "output": _enc(output),
        }

        name = "animate_reduced"
        config = {
            **tiny_base,
            "model_type": "animate",
            "in_dim": 36,
            "num_layers": 5,
        }
        model = animate_module.AnimateWanModel(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **config,
        )
        state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(state), strict=True)
        x = hashed_input(f"{name}:x", (1, 36, 2, 4, 4))
        timestep = torch.tensor((0.375,), dtype=torch.float32)
        context = hashed_input(f"{name}:context", (1, 4, 12))
        vision = hashed_input(f"{name}:vision", (1, 3, 1280))
        pose_latents = hashed_input(f"{name}:pose", (1, 16, 1, 4, 4))
        face_pixel_values = hashed_input(f"{name}:face", (1, 3, 4, 512, 512))
        output = model._forward(
            x,
            timestep,
            context,
            clip_fea=vision,
            pose_latents=pose_latents,
            face_pixel_values=face_pixel_values,
        )
        cases[name] = {
            "config": {
                "model_type": "i2v",
                "model_variant": "animate",
                "in_channels": 36,
                "hidden_size": 24,
                "ffn_hidden_size": 48,
                "num_heads": 2,
                "num_layers": 5,
                "text_dim": 12,
                "time_freq_dim": 8,
                "out_channels": 16,
            },
            "state_dict": state,
            "input_shape": list(x.shape),
            "timesteps": timestep.tolist(),
            "context_shape": list(context.shape),
            "vision_shape": list(vision.shape),
            "pose_latents_shape": list(pose_latents.shape),
            "face_pixel_values_shape": list(face_pixel_values.shape),
            "vace_context_shape": None,
            "vace_strength": None,
            "output": _enc(output),
        }

        payload = {
            "_meta": {
                "reference": (
                    "ComfyUI WanModel, WanModel_S2V, HumoWanModel, AnimateWanModel, "
                    "and MultiTalkModelPatch"
                ),
                "commit": REFERENCE_COMMIT,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "attention": "attention_pytorch",
                "rope": "pure torch (model_management.in_training=True)",
                "source": "git archive of exact commit object",
                **tuple_provenance(torch.__version__),
            },
            "layouts": layouts,
            "cases": cases,
        }
        platform_golden_path(OUT, torch.__version__).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            newline="\n",
        )


if __name__ == "__main__":
    main()
