"""Generate exact ER-SDE float32 trajectories from pinned ComfyUI.

Usage from the Dinkster repository root:

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    PYTHONPATH=/path/to/ComfyUI-at-b78cec87 \
        .venv-gpu/bin/python tools/gen_er_sde_trajectory_goldens.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
parser.add_argument("--compare", action="store_true")
ARGS = parser.parse_args()
if ARGS.compare and ARGS.output is None:
    parser.error("--compare requires --output to preserve the tracked golden")
sys.argv[:] = sys.argv[:1]
sys.argv.append("--cpu")

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_REFERENCE", REPO.parent / "ComfyUI")).resolve()
sys.path.insert(0, str(COMFY_ROOT))

import comfy.options  # noqa: E402
import torch  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.k_diffusion.sampling as sampling  # noqa: E402
import comfy.model_sampling as model_sampling  # noqa: E402
from golden_platform import cpu_identity, platform_golden_path, tuple_provenance  # noqa: E402

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = platform_golden_path(
    REPO / "packages/dinkster-inference-torch/tests/goldens/er_sde_trajectory_goldens.json",
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
NOISE_DRAWS = (
    (0.099833414, 0.71735609, 0.997494987, 0.675463181),
    (0.98544973, 0.863209367, 0.33498815, -0.350783228),
    (0.42737988, -0.255541102, -0.818277111, -0.999923258),
    (-0.687766159, -0.982452613, -0.772764488, -0.182162504),
    (-0.925814682, -0.631266638, 0.0168139, 0.656986599),
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


def _first_difference(left: object, right: object, path: str = "") -> dict[str, object] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"actual": sorted(left), "expected": sorted(right), "path": path}
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}".lstrip("."))
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"actual": len(left), "expected": len(right), "path": f"{path}.length"}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"actual": left, "expected": right, "path": path}
    return None


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


class _Noise:
    def __init__(self, like: torch.Tensor) -> None:
        self.like = like
        self.bounds: list[tuple[float, float]] = []

    def __call__(self, sigma_from: torch.Tensor, sigma_to: torch.Tensor) -> torch.Tensor:
        self.bounds.append((float(sigma_from), float(sigma_to)))
        values = NOISE_DRAWS[len(self.bounds) - 1]
        return torch.tensor(values, dtype=self.like.dtype, device=self.like.device).reshape_as(
            self.like
        )


def _case(space: object, sigmas: tuple[float, ...]) -> dict[str, object]:
    initial = torch.tensor(INITIAL, dtype=torch.float32).reshape(1, 1, 2, 2)
    schedule = torch.tensor(sigmas, dtype=torch.float32)
    model = _Model(space)
    noise = _Noise(initial)
    steps: list[list[float]] = []

    def callback(event: dict[str, Any]) -> None:
        steps.append(_values(event["x"]))

    result = sampling.sample_er_sde(
        model,
        initial.clone(),
        schedule,
        extra_args={},
        callback=callback,
        disable=True,
        noise_sampler=noise,
    )
    return {
        "sigmas": _values(schedule),
        "model_sigmas": model.sigmas,
        "model_calls": [_values(value) for value in model.calls],
        "noise_bounds": [list(bounds) for bounds in noise.bounds],
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
    source_path = Path(inspect.getfile(inspect.unwrap(sampling.sample_er_sde))).resolve()
    try:
        source_relative = source_path.relative_to(COMFY_ROOT)
    except ValueError as error:
        raise SystemExit(
            f"refusing to generate: imported ER-SDE from {source_path}, outside {COMFY_ROOT}"
        ) from error

    data = {
        "_meta": {
            "generator": "tools/gen_er_sde_trajectory_goldens.py",
            "reference_commit": commit,
            "source_path": source_relative.as_posix(),
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "initial": list(INITIAL),
        "noise_draws": [list(values) for values in NOISE_DRAWS],
        "cases": {
            "eps": _case(model_sampling.EPS(), SIGMAS["eps"]),
            "flow": _case(model_sampling.CONST(), SIGMAS["flow"]),
        },
    }
    output = ARGS.output or OUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    if ARGS.compare:
        baseline = json.loads(OUT.read_text(encoding="utf-8"))
        report = {
            "cpu": cpu_identity(),
            "first_differences": {
                case: _first_difference(data["cases"][case], baseline["cases"][case])
                for case in data["cases"]
            },
            "output": str(output),
            "torch": torch.__version__,
        }
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
