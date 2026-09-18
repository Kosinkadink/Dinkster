"""Generate ordered LoRA stack goldens from the pinned ComfyUI reference.

Usage from the Dinkster repository root:

    PYTHONPATH=../ComfyUI:../comfy-kitchen \
      /path/to/torch-2.13/bin/python tools/gen_lora_stack_goldens.py

``../ComfyUI`` must be commit b78cec879b9460d5cb25228a83a942fb78d2cd24.
Generate twice and require an unchanged SHA-256 before updating the fixture.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch
from gen_patch_goldens import APPLY_CASES, apply_case, e_lora
from golden_platform import platform_golden_path, tuple_provenance

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = platform_golden_path(
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "lora_stack_goldens.json",
    torch.__version__,
)


def main() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"ComfyUI must be {REFERENCE_COMMIT}, got {commit}")

    APPLY_CASES.clear()
    generator = torch.Generator().manual_seed(0x4B4)

    def tensor(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    weight = tensor(8, 6)
    linear_stack = [
        e_lora(
            tensor(8, rank, scale=0.4),
            tensor(rank, 6, scale=0.4),
            alpha,
            strength=strength,
        )
        for rank, alpha, strength in ((2, 1.0, 0.6), (3, 2.0, -0.25), (1, None, 1.1))
    ]
    apply_case("lora_stack_linear", weight, linear_stack)
    apply_case("lora_stack_linear_reversed", weight, list(reversed(linear_stack)))

    repeated_up = tensor(8, 2, scale=0.4)
    repeated_down = tensor(2, 6, scale=0.4)
    apply_case(
        "lora_stack_repeated_fractional",
        weight,
        [
            e_lora(repeated_up, repeated_down, 1.0, strength=0.25),
            e_lora(repeated_up, repeated_down, 1.0, strength=0.75),
        ],
    )
    apply_case(
        "lora_stack_repeated_combined",
        weight,
        [e_lora(repeated_up, repeated_down, 1.0, strength=1.0)],
    )

    conv_weight = tensor(5, 4, 3, 3)
    apply_case(
        "lora_stack_convolution",
        conv_weight,
        [
            e_lora(
                tensor(5, 2, 1, 1, scale=0.3),
                tensor(2, 4, 3, 3, scale=0.3),
                1.0,
                strength=0.5,
            ),
            e_lora(
                tensor(5, 3, 1, 1, scale=0.3),
                tensor(3, 4, 3, 3, scale=0.3),
                2.0,
                strength=-0.375,
            ),
        ],
    )

    payload = {
        "_meta": {
            "generator": "tools/gen_lora_stack_goldens.py",
            "reference_commit": commit,
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "apply_cases": APPLY_CASES,
    }
    encoded = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    OUT.write_bytes(encoded)
    print(f"wrote {OUT} ({len(APPLY_CASES)} cases, sha256 {hashlib.sha256(encoded).hexdigest()})")


if __name__ == "__main__":
    main()
