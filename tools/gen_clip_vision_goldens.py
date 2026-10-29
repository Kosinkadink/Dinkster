"""Generate CLIP ViT-H reduced goldens from pinned ComfyUI 947c2749.

The source is read from the exact git object without moving the checkout.
Run twice with Python 3.12 and torch available and require the output SHA256
to remain unchanged:

    python tools/gen_clip_vision_goldens.py
"""

from __future__ import annotations

import hashlib
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
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/clip_vision_goldens.json"


def _comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "ComfyUI",
        REPO.parent.parent / "ComfyUI",
        REPO.parent.parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set COMFYUI_ROOT to the ComfyUI git checkout")


COMFY_ROOT = _comfy_root()


def _git(*args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=COMFY_ROOT, check=True, capture_output=True).stdout


def _encode(tensor: Any) -> dict[str, Any]:
    import torch

    assert isinstance(tensor, torch.Tensor)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def main() -> None:
    if _git("rev-parse", REFERENCE_COMMIT).decode().strip() != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {REFERENCE_COMMIT}")
    archive = _git("archive", "--format=tar", REFERENCE_COMMIT)
    with tempfile.TemporaryDirectory(prefix="dinkster-clip-vision-reference-") as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
            payload.extractall(source, filter="data")
        sys.path.insert(0, str(REPO / "packages/dinkster-inference-torch/tests"))
        sys.path.insert(0, str(source))
        sys.argv.append("--cpu")

        import comfy.options
        import torch

        comfy.options.enable_args_parsing()
        from clip_vision_fill import fill_state_dict, image_input
        from comfy import clip_model, ops
        from comfy.ldm.modules import attention as attention_module

        clip_model.optimized_attention_for_device = lambda _device, **_kwargs: (
            attention_module.attention_pytorch
        )
        full_config = {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "intermediate_size": 5120,
            "hidden_act": "gelu",
            "model_type": "clip_vision_model",
            "num_channels": 3,
            "patch_size": 14,
            "image_size": 224,
            "projection_dim": 1024,
        }
        with torch.device("meta"):
            full_model = clip_model.CLIPVisionModelProjection(
                full_config, torch.float32, "meta", ops.disable_weight_init
            )
        full_layout = [(key, list(value.shape)) for key, value in full_model.state_dict().items()]
        full_layout.append(("vision_model.embeddings.position_ids", [1, 257]))
        full_layout.sort()

        reduced_config = {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "hidden_act": "gelu",
            "model_type": "clip_vision_model",
            "num_channels": 3,
            "patch_size": 14,
            "image_size": 28,
            "projection_dim": 8,
        }
        model = clip_model.CLIPVisionModelProjection(
            reduced_config, torch.float32, "cpu", ops.disable_weight_init
        )
        reference_state = [(key, list(value.shape)) for key, value in model.state_dict().items()]
        state = sorted(reference_state + [("vision_model.embeddings.position_ids", [1, 5])])
        filled = fill_state_dict(state)
        model.load_state_dict(
            {key: value for key, value in filled.items() if not key.endswith("position_ids")},
            strict=True,
        )
        image = image_input((1, 19, 31, 4))
        pixels = clip_model.clip_preprocess(image, size=28)
        output = model(pixel_values=pixels, intermediate_output=-2)[1]

        result = {
            "_meta": {
                "reference": (
                    "ComfyUI comfy/clip_model.py CLIPVisionModelProjection and clip_preprocess"
                ),
                "commit": REFERENCE_COMMIT,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "attention": "attention_pytorch",
                "source": "git archive of exact commit object",
            },
            "official_layout": full_layout,
            "reduced": {
                "config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 32,
                    "image_size": 28,
                    "patch_size": 14,
                    "projection_dim": 8,
                },
                "state_dict": state,
                "image": _encode(image),
                "preprocessed": _encode(pixels),
                "penultimate": _encode(output),
            },
        }
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        OUT.write_bytes(encoded.encode())
        print(hashlib.sha256(encoded.encode()).hexdigest())


if __name__ == "__main__":
    main()
