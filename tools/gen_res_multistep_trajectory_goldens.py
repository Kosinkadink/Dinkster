"""Generate exact res_multistep float32 trajectories from pinned ComfyUI.

Usage from the Dinkster repository root:

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    PYTHONPATH=/path/to/ComfyUI-at-b78cec87 \
        .venv-gpu/bin/python tools/gen_res_multistep_trajectory_goldens.py
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_REFERENCE", REPO.parent / "ComfyUI")).resolve()
sys.path.insert(0, str(COMFY_ROOT))

import comfy.options  # noqa: E402
import torch  # noqa: E402

comfy.options.enable_args_parsing()
from comfy.cli_args import args as comfy_args  # noqa: E402

comfy_args.cpu = True

import comfy.k_diffusion.sampling as sampling  # noqa: E402
import comfy.model_sampling as model_sampling  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = platform_golden_path(
    REPO / "packages/dinkster-inference-torch/tests/goldens/res_multistep_trajectory_goldens.json",
    torch.__version__,
)
INITIAL = (0.5, -1.0, 2.0, 0.25)
SIGMAS = {
    "eps": (14.614642, 6.0, 2.5, 1.0, 0.4, 0.0291675, 0.0),
    "flow": (
        0.9999857130611196,
        0.9723517894744873,
        0.9334266781806946,
        0.875,
        0.7782955765724182,
        0.5839160680770874,
        0.0,
    ),
}


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
    def __init__(self, space: object) -> None:
        self.inner_model = SimpleNamespace(
            model_patcher=SimpleNamespace(get_model_object=lambda _name: space)
        )
        self.calls: list[torch.Tensor] = []
        self.sigmas: list[float] = []

    def __call__(self, value: torch.Tensor, sigma: torch.Tensor, **_: object) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        scalar = float(sigma.reshape(-1)[0])
        self.sigmas.append(scalar)
        return value * (1.0 / (1.0 + scalar)) + value.square() * (0.05 / (1.0 + scalar))


def _case(space: object, sigmas: tuple[float, ...]) -> dict[str, object]:
    initial = torch.tensor(INITIAL, dtype=torch.float32).reshape(1, 1, 2, 2)
    schedule = torch.tensor(sigmas, dtype=torch.float32)
    model = _Model(space)
    steps: list[list[float]] = []

    def callback(event: dict[str, Any]) -> None:
        steps.append(_values(event["x"]))

    result = sampling.sample_res_multistep(
        model,
        initial.clone(),
        schedule,
        extra_args={},
        callback=callback,
        disable=True,
    )
    return {
        "sigmas": _values(schedule),
        "model_sigmas": model.sigmas,
        "model_calls": [_values(value) for value in model.calls],
        "steps": steps,
        "final": _values(result),
    }


def main() -> None:
    commit = _git("rev-parse", "HEAD")
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"refusing to generate: ComfyUI is {commit}, expected {REFERENCE_COMMIT}")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"refusing to generate: ComfyUI is dirty:\n{dirty}")
    source_path = Path(inspect.getfile(inspect.unwrap(sampling.sample_res_multistep))).resolve()
    try:
        source_relative = source_path.relative_to(COMFY_ROOT)
    except ValueError as error:
        raise SystemExit(
            f"refusing to generate: imported res_multistep from {source_path}, outside {COMFY_ROOT}"
        ) from error

    data = {
        "_meta": {
            "generator": "tools/gen_res_multistep_trajectory_goldens.py",
            "reference_commit": commit,
            "source_path": source_relative.as_posix(),
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "initial": list(INITIAL),
        "cases": {
            "eps": _case(model_sampling.EPS(), SIGMAS["eps"]),
            "flow": _case(model_sampling.CONST(), SIGMAS["flow"]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
