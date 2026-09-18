"""Generate Gemma 2 2B text-tower goldens from ComfyUI.

Darwin fixtures use Python 3.12.11 and torch 2.13.0 in a platform-tuple file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

GENERATOR_TORCH = "2.13.0" if sys.platform == "darwin" else "2.13.0+cpu"
if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
    )

from clip_fill import fill_state_dict  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.ldm.modules import attention  # noqa: E402
from comfy.text_encoders import llama  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

llama.optimized_attention_for_device = lambda *_args, **_kwargs: attention.attention_pytorch

OUT = platform_golden_path(
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "lumina2_text_goldens.json",
    torch.__version__,
)

TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}

CASES = {
    "gemma2_attended": (
        (2, 11, 7, 19, 23),
        (1, 1, 1, 1, 1),
    ),
    "gemma2_padding": (
        (2, 0, 0, 13, 17, 0),
        (1, 0, 0, 1, 1, 0),
    ),
}


def tiny_config() -> Any:
    config = llama.Gemma2_2B_Config(**TINY_CONFIG)
    config.head_dim = 8
    return config


def build(config: Any, device: str) -> torch.nn.Module:
    return llama.Llama2_(config, device=device, dtype=torch.float32, ops=ops.disable_weight_init)


def encode(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def entries(model: torch.nn.Module) -> list[tuple[str, list[int]]]:
    return sorted((key, list(value.shape)) for key, value in model.state_dict().items())


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
            f"{COMFY_ROOT} is at {commit}; expected audited baseline {REFERENCE_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"{COMFY_ROOT} must be clean:\n{dirty}")
    module_file = Path(sys.modules[llama.Llama2_.__module__].__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"reference imported from {module_file}, not {COMFY_ROOT}")

    full = build(llama.Gemma2_2B_Config(), "meta")
    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": "attention_pytorch",
            **tuple_provenance(torch.__version__),
        },
        "layout": entries(full),
        "cases": {},
    }
    for name, (ids, mask) in sorted(CASES.items()):
        model = build(tiny_config(), "cpu")
        state = entries(model)
        model.load_state_dict(fill_state_dict(state), strict=True)
        token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            final, stack = model(
                token_ids,
                attention_mask=attention_mask,
                intermediate_output="all",
                final_layer_norm_intermediate=False,
            )
        payload["cases"][name] = {
            "config": {**TINY_CONFIG, "head_dim": 8},
            "ids": list(ids),
            "attention_mask": list(mask),
            "state_dict": state,
            "stack": encode(stack),
            "final": encode(final),
        }

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
