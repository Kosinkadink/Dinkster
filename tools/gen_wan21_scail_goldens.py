"""Generate Wan SCAIL/SCAIL2 goldens from pinned ComfyUI b78cec87.

Run from the Dinkster root with Python 3.12 and torch available:

    python tools/gen_wan21_scail_goldens.py

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

from golden_platform import platform_golden_path, tuple_provenance

REPO = Path(__file__).resolve().parent.parent
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "wan21_scail_goldens.json"
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
    with tempfile.TemporaryDirectory(prefix="dinkster-wan21-scail-reference-") as directory:
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
        from unet_fill import fill_state_dict, hashed_input

        wan_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention = attention_module.attention_pytorch
        attention_module.optimized_attention_masked = attention_module.attention_pytorch
        model_management.in_training = True

        base_config = {
            "patch_size": (1, 2, 2),
            "in_dim": 20,
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
        cases: dict[str, dict[str, Any]] = {}
        for name, variant, replacement in (
            ("scail_animation", "scail", False),
            ("scail2_animation", "scail2", False),
            ("scail2_replacement", "scail2", True),
        ):
            model_class = (
                wan_module.SCAIL2WanModel if variant == "scail2" else wan_module.SCAILWanModel
            )
            model = model_class(
                operations=ops.disable_weight_init,
                device="cpu",
                dtype=torch.float32,
                **base_config,
            )
            state = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
            model.load_state_dict(fill_state_dict(state), strict=True)

            x = hashed_input(f"{name}:x", (1, 20, 3, 5, 6))
            timestep = torch.tensor((0.375,), dtype=torch.float32)
            context = hashed_input(f"{name}:context", (1, 4, 12))
            vision = hashed_input(f"{name}:vision", (1, 3, 1280))
            reference = hashed_input(f"{name}:reference", (1, 20, 2, 5, 6))
            pose = hashed_input(f"{name}:pose", (1, 20, 3, 3, 3))
            reference_mask = (
                hashed_input(f"{name}:reference-mask", (1, 28, 5, 5, 6))
                if variant == "scail2"
                else None
            )
            driving_mask = (
                hashed_input(f"{name}:driving-mask", (1, 28, 3, 3, 3))
                if variant == "scail2"
                else None
            )
            output = model._forward(
                x,
                timestep,
                context,
                clip_fea=vision,
                pose_latents=pose,
                reference_latent=reference,
                ref_mask_latents=reference_mask,
                sam_latents=driving_mask,
                ref_mask_flag=False if replacement else None,
                transformer_options={},
            )
            cases[name] = {
                "variant": variant,
                "replacement": replacement,
                "state_dict": state,
                "input_shape": list(x.shape),
                "timesteps": timestep.tolist(),
                "context_shape": list(context.shape),
                "vision_shape": list(vision.shape),
                "reference_shape": list(reference.shape),
                "pose_shape": list(pose.shape),
                "reference_mask_shape": (
                    None if reference_mask is None else list(reference_mask.shape)
                ),
                "driving_mask_shape": (None if driving_mask is None else list(driving_mask.shape)),
                "output": _encode(output),
            }

        payload = {
            "_meta": {
                "reference": "ComfyUI SCAILWanModel/SCAIL2WanModel",
                "commit": REFERENCE_COMMIT,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "attention": "attention_pytorch",
                "rope": "pure torch (model_management.in_training=True)",
                "source": "git archive of exact commit object",
                **tuple_provenance(torch.__version__),
            },
            "config": {
                "model_type": "i2v",
                "in_channels": 20,
                "hidden_size": 24,
                "ffn_hidden_size": 48,
                "num_heads": 2,
                "num_layers": 2,
                "text_dim": 12,
                "time_freq_dim": 8,
                "out_channels": 16,
            },
            "cases": cases,
        }
        platform_golden_path(OUT, torch.__version__).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            newline="\n",
        )


if __name__ == "__main__":
    main()
