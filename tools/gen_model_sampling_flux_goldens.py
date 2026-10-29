"""Execute pinned ComfyUI ModelSamplingFlux and scheduler source on CPU.

Usage: .venv-torch/bin/python tools/gen_model_sampling_flux_goldens.py COMFY_ROOT
Only the patcher's clone/add_object_patch storage is substituted. All arithmetic
comes from the named, unmodified upstream definitions, executed with real torch.
Darwin fixtures use Python 3.12.11 and torch 2.13.0 in a platform-tuple file.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import scipy.stats
import torch
import torchsde
from golden_platform import platform_golden_path, tuple_provenance

REFERENCE_COMMIT = "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/dinkster-inference-torch/tests/goldens/model_sampling_flux.json"


def definitions(path: Path, names: tuple[str, ...], namespace: dict[str, Any]) -> None:
    source = ast.parse(path.read_text())
    body = [node for node in source.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(model_config=SimpleNamespace(sampling_settings={}))
        self.patches: dict[str, Any] = {}

    def clone(self) -> Patcher:
        result = Patcher()
        result.patches = dict(self.patches)
        return result

    def add_object_patch(self, name: str, value: Any) -> None:
        self.patches[name] = value


def main() -> None:
    comfy = Path(sys.argv[1]).resolve()
    actual = subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True)
    if actual.strip() != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI must be at {REFERENCE_COMMIT}")
    namespace: dict[str, Any] = {
        "torch": torch,
        "math": math,
        "numpy": numpy,
        "scipy": scipy,
        "torchsde": torchsde,
    }
    definitions(
        comfy / "comfy/model_sampling.py",
        ("flux_time_shift", "ModelSamplingFlux", "CONST"),
        namespace,
    )
    namespace["comfy"] = SimpleNamespace(model_sampling=SimpleNamespace(**namespace))
    definitions(comfy / "comfy_extras/nodes_model_advanced.py", ("ModelSamplingFlux",), namespace)
    definitions(
        comfy / "comfy/samplers.py",
        ("simple_scheduler", "normal_scheduler", "beta_scheduler", "ddim_scheduler"),
        namespace,
    )
    definitions(
        comfy / "comfy/k_diffusion/sampling.py",
        ("offset_first_sigma_for_snr", "BatchedBrownianTree", "BrownianTreeNoiseSampler"),
        namespace,
    )
    patch = namespace["ModelSamplingFlux"]().patch
    cases = []
    original = Patcher()
    for label, maximum, base, width, height in (
        ("default", 1.15, 0.5, 1024, 1024),
        ("non_square", 1.15, 0.5, 768, 1280),
        ("small", 1.15, 0.5, 16, 16),
        ("large", 1.15, 0.5, 4096, 4096),
        ("zero", 0.0, 0.0, 1024, 1024),
        ("negative_mu", 0.0, 1.0, 2048, 2048),
        ("equal_shifts", 1.0, 1.0, 16384, 16384),
    ):
        patched = patch(original, maximum, base, width, height)[0]
        assert original.patches == {}
        assert (
            patch(patched, maximum, base, width, height)[0].patches["model_sampling"].shift
            == patched.patches["model_sampling"].shift
        )
        sampling = patched.patches["model_sampling"]
        schedules = {}
        for name, function in (
            ("simple", "simple_scheduler"),
            ("normal", "normal_scheduler"),
            ("beta", "beta_scheduler"),
            ("ddim_uniform", "ddim_scheduler"),
        ):
            sigmas = namespace[function](sampling, 4)
            snr_sigmas = namespace["offset_first_sigma_for_snr"](sigmas, sampling)
            noise = namespace["BrownianTreeNoiseSampler"](
                torch.zeros(1, 2, 2, 2),
                sigmas[sigmas > 0].min(),
                sigmas.max(),
                seed=23,
                cpu=True,
            )
            schedules[name] = {
                "sigmas": sigmas.tolist(),
                "snr_sigmas": snr_sigmas.tolist(),
                "brownian_min": float(sigmas[sigmas > 0].min()),
                "brownian_max": float(sigmas.max()),
            }
            try:
                schedules[name]["brownian_draws"] = [
                    noise(snr_sigmas[index], snr_sigmas[index + 1]).tolist()
                    for index in range(len(sigmas) - 2)
                ]
            except RecursionError:
                schedules[name]["brownian_error"] = "RecursionError"
        cases.append(
            {
                "name": label,
                "max_shift": maximum,
                "base_shift": base,
                "width": width,
                "height": height,
                "shift": sampling.shift,
                "percent_sigmas": [sampling.percent_to_sigma(p) for p in (0, 0.0001, 0.5, 1)],
                "schedules": schedules,
            }
        )
    payload: dict[str, Any] = {"comfyui_commit": REFERENCE_COMMIT, "cases": cases}
    provenance = tuple_provenance(torch.__version__)
    if provenance:
        payload["_meta"] = provenance
    output = platform_golden_path(OUTPUT, torch.__version__)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
