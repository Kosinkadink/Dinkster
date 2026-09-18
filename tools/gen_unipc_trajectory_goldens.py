"""Generate exact UniPC float32 trajectory goldens from pinned ComfyUI.

Usage from the Dinkster repository root:

    PYTHONPATH=/home/kosin/ComfyUI:/home/kosin/comfy-aimdo:/home/kosin/comfy-kitchen \
        /home/kosin/ComfyUI/venv/bin/python tools/gen_unipc_trajectory_goldens.py
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import comfy.options
import torch

comfy.options.enable_args_parsing()

import comfy.model_sampling as model_sampling  # noqa: E402, I001
import comfy.sample as sample  # noqa: E402
import comfy.samplers as samplers  # noqa: E402
from comfy.extra_samplers.uni_pc import (  # noqa: E402
    sample_unipc,
    sample_unipc_bh2,
)
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402


REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
SEED = 808369199502636
STEPS = 20
REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/home/kosin/ComfyUI")).resolve()
OUT = platform_golden_path(
    REPO / "packages/dinkster-inference-torch/tests/goldens/unipc_trajectory_goldens.json",
    torch.__version__,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(COMFY_ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _values(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


class _Model:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor, sigma: torch.Tensor, **_: object) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        scalar = float(sigma.reshape(-1)[0])
        return value * (1.0 / (1.0 + scalar)) + (value * value) * (0.05 / (1.0 + scalar))


def _case(function: Any, noise: torch.Tensor, sigmas: torch.Tensor) -> dict[str, object]:
    model = _Model()
    trajectory: list[list[float]] = []

    def callback(event: dict[str, Any]) -> None:
        trajectory.append(_values(event["x"]))

    result = function(
        model,
        noise.clone(),
        sigmas,
        extra_args={},
        callback=callback,
        disable=True,
    )
    return {
        "first_call": _values(model.calls[0]),
        "model_calls": [_values(value) for value in model.calls],
        "steps": trajectory,
        "final": _values(result),
    }


def main() -> None:
    commit = _git("rev-parse", "HEAD")
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"refusing to generate: ComfyUI is {commit}, expected {REFERENCE_COMMIT}")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"refusing to generate: ComfyUI is dirty:\n{dirty}")
    source_path = Path(inspect.getfile(sample_unipc)).resolve()
    try:
        source_relative = source_path.relative_to(COMFY_ROOT.resolve())
    except ValueError as error:
        raise SystemExit(
            f"refusing to generate: imported UniPC from {source_path}, "
            f"outside {COMFY_ROOT.resolve()}"
        ) from error

    space = model_sampling.ModelSamplingDiscrete()
    sigmas = samplers.normal_scheduler(space, STEPS)
    noise = sample.prepare_noise(torch.zeros((1, 1, 2, 2)), SEED)
    data = {
        "_meta": {
            "generator": "tools/gen_unipc_trajectory_goldens.py",
            "reference_commit": commit,
            "source_path": str(source_relative),
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "seed": SEED,
        "sigmas": _values(sigmas),
        "initial_noise": _values(noise),
        "cases": {
            "uni_pc": _case(sample_unipc, noise, sigmas),
            "uni_pc_bh2": _case(sample_unipc_bh2, noise, sigmas),
        },
        "edge_cases": {
            "uni_pc_terminal_zero": _case(sample_unipc, noise, torch.tensor([1.0, 0.0])),
            "uni_pc_terminal_nonzero": _case(sample_unipc, noise, torch.tensor([1.0, 0.5])),
            "uni_pc_bh2_terminal_zero": _case(sample_unipc_bh2, noise, torch.tensor([1.0, 0.0])),
            "uni_pc_bh2_terminal_nonzero": _case(sample_unipc_bh2, noise, torch.tensor([1.0, 0.5])),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
