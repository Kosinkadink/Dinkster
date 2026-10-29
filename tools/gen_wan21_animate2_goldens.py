"""Generate Wan Animate2 goldens from pinned ComfyUI 7dde5617.

Run from the Dinkster root with Python 3.12 and torch available:

    python tools/gen_wan21_animate2_goldens.py

Set ``COMFYUI_ROOT`` when ComfyUI is not beside Dinkster or its parent.
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
from typing import Any

REPO = Path(__file__).resolve().parent.parent
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "wan21_animate2_goldens.json"
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


def _encode(value: Any) -> dict[str, Any]:
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
    with tempfile.TemporaryDirectory(prefix="dinkster-wan21-animate2-reference-") as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
            payload.extractall(source, filter="data")
        sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
        sys.path.insert(0, str(source))
        sys.argv.append("--cpu")

        import comfy.options
        import torch

        comfy.options.enable_args_parsing()
        from comfy import model_management, ops
        from comfy.ldm.modules import attention as attention_module
        from comfy.ldm.wan import model as wan_module
        from comfy.ldm.wan import model_animate2 as animate2_module
        from unet_fill import fill_state_dict, hashed_input

        wan_module.optimized_attention = attention_module.attention_pytorch
        animate2_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention_masked = attention_module.attention_pytorch
        model_management.in_training = True

        config = {
            "model_type": "animate2",
            "patch_size": (1, 2, 2),
            "in_dim": 36,
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
        model = animate2_module.WanAnimate2Model(
            operations=ops.disable_weight_init,
            device="cpu",
            dtype=torch.float32,
            **config,
        )
        state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
        model.load_state_dict(fill_state_dict(state), strict=True)

        x = hashed_input("animate2_reduced:x", (1, 36, 3, 5, 6))
        timestep = torch.tensor((0.375,), dtype=torch.float32)
        context = hashed_input("animate2_reduced:context", (1, 4, 12))
        vision = hashed_input("animate2_reduced:vision", (1, 3, 1280))
        pose = hashed_input("animate2_reduced:pose", (1, 16, 2, 5, 6))
        pose_context = hashed_input("animate2_reduced:pose-context", (1, 5, 12))
        pose_vision = hashed_input("animate2_reduced:pose-vision", (1, 2, 1280))

        no_pose = model._forward(
            x,
            timestep,
            context,
            clip_fea=vision,
            transformer_options={},
        )
        pose_default = model._forward(
            x,
            timestep,
            context,
            clip_fea=vision,
            pose_latents=pose,
            transformer_options={},
        )
        pose_distinct = model._forward(
            x,
            timestep,
            context,
            clip_fea=vision,
            pose_latents=pose,
            context_pose=pose_context,
            clip_fea_pose=pose_vision,
            pose_strength=0.625,
            reference_strength=0.75,
            transformer_options={},
        )
        payload = {
            "_meta": {
                "reference": "ComfyUI WanAnimate2Model",
                "commit": REFERENCE_COMMIT,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "attention": "attention_pytorch",
                "rope": "pure torch (model_management.in_training=True)",
                "source": "git archive of exact commit object",
            },
            "config": {
                "model_type": "i2v",
                "model_variant": "animate2",
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
            "vision_shape": list(vision.shape),
            "pose_shape": list(pose.shape),
            "pose_context_shape": list(pose_context.shape),
            "pose_vision_shape": list(pose_vision.shape),
            "outputs": {
                "no_pose": _encode(no_pose),
                "pose_default": _encode(pose_default),
                "pose_distinct": _encode(pose_distinct),
            },
        }
        OUT.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            newline="\n",
        )


if __name__ == "__main__":
    main()
