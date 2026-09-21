"""Generate Ideogram 4 model and scheduler goldens from pinned ComfyUI.

Usage from the Dinkster repository root:

    COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=../comfy-aimdo \
        /path/to/execution/python tools/gen_ideogram4_goldens.py

The generator refuses a dirty or differently pinned reference checkout and a
different torch build. It executes tiny conditional, padded, and image-only
models while constructing the published full model on the meta device.
Darwin fixtures use Python 3.12.11 and torch 2.13.0 in a platform-tuple file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
REFERENCE_COMMIT = "1af040bf022569d7a890241c8dd79b296cda483f"
GENERATOR_TORCH = "2.13.0" if sys.platform == "darwin" else "2.13.0+cu130"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens require torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
    )

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.ideogram4 import model as ideogram4_model  # noqa: E402
from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel  # noqa: E402
from comfy.ldm.modules import attention as reference_attention  # noqa: E402
from comfy.text_encoders import ideogram4 as ideogram4_text  # noqa: E402
from comfy.text_encoders import qwen3vl  # noqa: E402
from comfy_extras.nodes_ideogram4 import ideogram4_sigmas  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

ideogram4_model.optimized_attention_masked = reference_attention.attention_pytorch
model_management.in_training = True

OUT = platform_golden_path(
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "ideogram4_goldens.json",
    torch.__version__,
)

FULL_CONFIG = {
    "in_channels": 128,
    "num_layers": 34,
    "num_attention_heads": 18,
    "attention_head_dim": 256,
    "intermediate_size": 12288,
    "adaln_dim": 512,
    "llm_features_dim": 53248,
    "rope_theta": 5_000_000,
    "mrope_section": (24, 20, 20),
    "norm_eps": 1e-5,
}

TINY_CONFIG = {
    "in_channels": 8,
    "num_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "intermediate_size": 48,
    "adaln_dim": 16,
    "llm_features_dim": 24,
    "rope_theta": 5_000,
    "mrope_section": (3, 3, 2),
    "norm_eps": 1e-5,
}

PROMPTS = (
    "cat",
    "",
    "(cat:2.0)",
    "cafe\u0301 \u4e2d",
    "<|im_start|>literal",
    "<|endoftext|> masked tail",
)


def _reference_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _require_reference() -> str:
    commit = _reference_commit()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected reference {REFERENCE_COMMIT}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"{COMFY_ROOT} must be clean:\n{dirty}")
    module = Path(sys.modules[Ideogram4Transformer2DModel.__module__].__file__ or "").resolve()
    if not module.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"Ideogram 4 reference loaded from {module}, not {COMFY_ROOT}")
    return commit


def _build(config: dict[str, object], device: str) -> Ideogram4Transformer2DModel:
    return Ideogram4Transformer2DModel(
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
        **config,
    )


def _encode(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().to(torch.float32)
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "data": value.flatten().tolist(),
    }


def _text_layout() -> list[tuple[str, list[int]]]:
    model_class = qwen3vl._make_qwen3vl_model("qwen3vl_8b")
    model = model_class({}, torch.float32, "meta", ops.disable_weight_init)

    def artifact_key(key: str) -> str:
        if key.startswith("visual."):
            return "model.visual." + key.removeprefix("visual.")
        if key == "model.lm_head.weight":
            return "lm_head.weight"
        return key

    return sorted(
        (artifact_key(key), list(value.shape)) for key, value in model.state_dict().items()
    )


def _attention_mask(ids: list[int]) -> list[int]:
    first_pad = ids.index(151643) if 151643 in ids else len(ids)
    return [1] * first_pad + [0] * (len(ids) - first_pad)


def _tokenizer_cases() -> list[dict[str, object]]:
    tokenizer = ideogram4_text.Ideogram4Tokenizer()
    cases = []
    for text in PROMPTS:
        chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)["qwen3vl_8b"]
        if len(chunks) != 1:
            raise AssertionError(f"Ideogram 4 produced {len(chunks)} token chunks")
        ids = [int(token) for token, _, _ in chunks[0]]
        cases.append(
            {
                "text": text,
                "ids": ids,
                "attention_mask": _attention_mask(ids),
                "weights": [float(weight) for _, weight, _ in chunks[0]],
            }
        )
    return cases


def _model_case(name: str, *, padded: bool, image_only: bool) -> dict[str, object]:
    model = _build(TINY_CONFIG, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    batch, height, width, text_length = 2, 3, 4, 5
    latent = hashed_input(f"{name}:latent", (batch, TINY_CONFIG["in_channels"], height, width))
    timesteps = torch.tensor((0.2, 0.8), dtype=torch.float32)
    context = None
    attention_mask = None
    if not image_only:
        context = hashed_input(
            f"{name}:context", (batch, text_length, TINY_CONFIG["llm_features_dim"])
        )
        if padded:
            attention_mask = torch.tensor(((1, 1, 1, 0, 0), (1, 1, 1, 1, 1)), dtype=torch.long)

    observed: dict[str, torch.Tensor] = {}
    hooks = (
        model.layers[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(block0=output)
        ),
        model.final_layer.register_forward_hook(
            lambda _module, _inputs, output: observed.update(final=output)
        ),
    )
    try:
        output = model(latent, timesteps, context, attention_mask)
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "config": TINY_CONFIG,
        "state_dict": entries,
        "batch": batch,
        "height": height,
        "width": width,
        "text_length": text_length,
        "timesteps": timesteps.tolist(),
        "padded": padded,
        "image_only": image_only,
        "block0": _encode(observed["block0"]),
        "final": _encode(observed["final"]),
        "output": _encode(output),
    }


def main() -> None:
    commit = _require_reference()
    payload: dict[str, object] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": "attention_pytorch",
            "rope": "pure torch (model_management.in_training=True)",
            **tuple_provenance(torch.__version__),
        },
        "layout": sorted(
            (key, list(value.shape))
            for key, value in _build(FULL_CONFIG, "meta").state_dict().items()
        ),
        "text_layout": _text_layout(),
        "tokenizer": _tokenizer_cases(),
        "cases": {
            "conditional": _model_case("conditional", padded=False, image_only=False),
            "conditional_padded": _model_case("conditional_padded", padded=True, image_only=False),
            "image_only": _model_case("image_only", padded=False, image_only=True),
        },
        "schedules": {
            name: ideogram4_sigmas(steps, width, height, mu, std).tolist()
            for name, steps, width, height, mu, std in (
                ("quality", 48, 1024, 1024, 0.0, 1.5),
                ("default", 20, 1024, 1024, 0.0, 1.75),
                ("maintained_template", 20, 1024, 1024, 0.5, 1.75),
                ("turbo", 12, 1024, 1024, 0.5, 1.75),
            )
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
