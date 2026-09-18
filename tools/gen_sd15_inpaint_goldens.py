"""Generate SD1.5 inpaint concatenation goldens from pinned ComfyUI.

Usage:
    COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      /path/to/ComfyUI/venv/bin/python tools/gen_sd15_inpaint_goldens.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from golden_platform import platform_golden_path, platform_provenance

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ["COMFYUI_REFERENCE"]).resolve()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=COMFY_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


commit = git_output("rev-parse", "HEAD")
dirty = git_output("status", "--porcelain")
if commit != REFERENCE_COMMIT:
    raise SystemExit(f"reference is at {commit}; required {REFERENCE_COMMIT}")
if dirty:
    raise SystemExit(f"reference checkout must be clean:\n{dirty}")

sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.model_base import BaseModel  # noqa: E402

OUT = platform_golden_path(
    REPO / "packages/dinkster-inference-torch/tests/goldens/sd15_inpaint_goldens.json",
    torch.__version__,
)


def encoded(value: torch.Tensor) -> dict[str, object]:
    return {
        "data": value.flatten().tolist(),
        "dtype": "float32",
        "shape": list(value.shape),
    }


def reference_model() -> BaseModel:
    model = object.__new__(BaseModel)
    model.concat_keys = ()
    model.set_inpaint()
    model.process_latent_in = lambda latent: latent * 0.18215
    return model


def main() -> None:
    model = reference_model()
    noise = torch.linspace(-1.0, 1.0, 2 * 4 * 2 * 3).reshape(2, 4, 2, 3)
    latent = torch.linspace(-0.8, 0.9, 4 * 1 * 2).reshape(1, 4, 1, 2)
    mask = torch.tensor([[[[0.49, 0.51]]]], dtype=torch.float32)
    crop_noise = noise[:1, :, :, :2]
    crop_latent = torch.linspace(-0.9, 0.8, 4 * 2 * 4).reshape(1, 4, 2, 4)
    crop_mask = torch.tensor(
        [[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]],
        dtype=torch.float32,
    )
    batch_noise = torch.linspace(-1.0, 1.0, 3 * 4 * 2 * 3).reshape(3, 4, 2, 3)
    batch_latent = torch.linspace(-0.8, 0.9, 2 * 4 * 1 * 2).reshape(2, 4, 1, 2)
    batch_mask = torch.tensor([[[[0.0, 0.0]]], [[[1.0, 1.0]]]], dtype=torch.float32)
    cases = {
        "default": {
            "noise": encoded(noise),
            "mask": None,
            "latent": encoded(latent),
            "output": encoded(
                torch.cat(
                    (
                        noise,
                        model.concat_cond(
                            noise=noise,
                            device=noise.device,
                            latent_image=latent,
                            denoise_mask=None,
                        ),
                    ),
                    dim=1,
                )
            ),
        },
        "resize_round_and_batch": {
            "noise": encoded(noise),
            "mask": encoded(mask),
            "latent": encoded(latent),
            "output": encoded(
                torch.cat(
                    (
                        noise,
                        model.concat_cond(
                            noise=noise,
                            device=noise.device,
                            concat_latent_image=latent,
                            denoise_mask=mask,
                        ),
                    ),
                    dim=1,
                )
            ),
        },
        "center_crop": {
            "noise": encoded(crop_noise),
            "mask": encoded(crop_mask),
            "latent": encoded(crop_latent),
            "output": encoded(
                torch.cat(
                    (
                        crop_noise,
                        model.concat_cond(
                            noise=crop_noise,
                            device=crop_noise.device,
                            concat_latent_image=crop_latent,
                            denoise_mask=crop_mask,
                        ),
                    ),
                    dim=1,
                )
            ),
        },
        "resize_batch_distribution": {
            "noise": encoded(batch_noise),
            "mask": encoded(batch_mask),
            "latent": encoded(batch_latent),
            "output": encoded(
                torch.cat(
                    (
                        batch_noise,
                        model.concat_cond(
                            noise=batch_noise,
                            device=batch_noise.device,
                            concat_latent_image=batch_latent,
                            denoise_mask=batch_mask,
                        ),
                    ),
                    dim=1,
                )
            ),
        },
    }
    payload = {
        "cases": cases,
        "reference": {
            "commit": commit,
            "repo": "ComfyUI",
            **platform_provenance(torch.__version__),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", newline="\n")
    print(OUT)


if __name__ == "__main__":
    main()
