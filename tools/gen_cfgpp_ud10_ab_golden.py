"""Generate CFG++ UD10 AB solver goldens from its pinned ComfyUI source."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from comfy import model_sampling
from comfy.k_diffusion import sampling

REFERENCE_COMMIT = "95539f56344958339e39b7582a476267d489b0ee"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "cfgpp_ud10_ab_95539f56.json"
X0 = [0.5, -1.0, 2.0, 0.25]
SIGMAS = {
    "eps": [14.614642, 6.0, 2.5, 1.0, 0.4, 0.0291675, 0.0],
    "flow": [0.98, 0.75, 0.5, 0.25, 0.1, 0.0],
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _denoised(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    value = float(sigma.reshape(-1)[0])
    return x * (1.0 / (1.0 + value)) + (x * x) * (0.05 / (1.0 + value))


def _unconditioned(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    value = float(sigma.reshape(-1)[0])
    return x * (0.8 / (1.2 + value)) - (x * x) * (0.03 / (1.0 + value))


class _Model:
    def __init__(self, space: object) -> None:
        self.inner_model = SimpleNamespace(
            inner_model=SimpleNamespace(model_sampling=space),
            model_patcher=SimpleNamespace(get_model_object=lambda name: space),
        )

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        denoised = _denoised(x, sigma)
        unconditioned = _unconditioned(x, sigma)
        options = kwargs.get("model_options") or {}
        assert isinstance(options, dict)
        for function in options.get("sampler_post_cfg_function", []):
            denoised = function(
                {
                    "denoised": denoised,
                    "uncond_denoised": unconditioned,
                    "cond_denoised": denoised,
                    "uncond": object(),
                    "input": x,
                    "sigma": sigma,
                    "model_options": options,
                }
            )
        return denoised


def main() -> None:
    configured = os.environ.get("COMFYUI_REFERENCE")
    if not configured:
        raise RuntimeError("COMFYUI_REFERENCE must name the pinned ComfyUI checkout")
    root = Path(configured).resolve()
    actual = _git(root, "rev-parse", "HEAD")
    if actual != REFERENCE_COMMIT:
        raise RuntimeError(f"expected ComfyUI {REFERENCE_COMMIT}, got {actual}")
    spaces = {"eps": model_sampling.EPS(), "flow": model_sampling.CONST()}
    x = torch.tensor(X0, dtype=torch.float64)
    cases = {}
    for name, sigmas in SIGMAS.items():
        result = sampling.sample_cfgpp_ud10_ab(
            _Model(spaces[name]),
            x.clone(),
            torch.tensor(sigmas, dtype=torch.float64),
            disable=True,
        )
        cases[name] = [float(value) for value in result.tolist()]
    data = {
        "reference_commit": REFERENCE_COMMIT,
        "reference_commit_date": _git(root, "show", "-s", "--format=%cI", REFERENCE_COMMIT),
        "reference_remote": _git(root, "remote", "get-url", "origin"),
        "dtype": "float64",
        "x0": X0,
        "sigmas": SIGMAS,
        "cases": cases,
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
