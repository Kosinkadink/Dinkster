"""Execute ComfyUI flow schedules and Euler on Python 3.12 / torch 2.13 CPU.

Usage: .venv-torch/bin/python tools/gen_complete_sampling_shift_goldens.py COMFY_ROOT
The diffusion module is deterministic arithmetic; sampling and latent transforms
are unmodified definitions executed from the pinned reference checkout.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REFERENCE_COMMIT = "947c2749dd04c51ef0e21b069544d8b0b4f9b411"
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "packages/dinkster-inference-torch/tests/goldens/complete_sampling_shifts.json"
)


def definitions(path: Path, names: tuple[str, ...], namespace: dict[str, Any]) -> None:
    source = ast.parse(path.read_text())
    body = [node for node in source.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main() -> None:
    comfy = Path(sys.argv[1]).resolve()
    actual = subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True)
    if actual.strip() != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI must be at {REFERENCE_COMMIT}")
    torch.set_num_threads(1)
    namespace: dict[str, Any] = {"torch": torch, "math": math}
    definitions(
        comfy / "comfy/model_sampling.py",
        (
            "reshape_sigma",
            "flux_time_shift",
            "time_snr_shift",
            "CONST",
            "ModelSamplingFlux",
            "ModelSamplingDiscreteFlow",
        ),
        namespace,
    )
    definitions(comfy / "comfy/samplers.py", ("simple_scheduler",), namespace)
    definitions(comfy / "comfy/k_diffusion/utils.py", ("append_dims",), namespace)
    namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])
    namespace["trange"] = lambda count, **kwargs: range(count)
    definitions(comfy / "comfy/k_diffusion/sampling.py", ("to_d", "sample_euler"), namespace)
    definitions(
        comfy / "comfy/latent_formats.py", ("LatentFormat", "SD3", "Flux", "Flux2"), namespace
    )
    cases = []
    for family, sampling_name, format_name, channels, default in (
        ("flux2", "ModelSamplingFlux", "Flux2", 128, 2.02),
        ("lumina2", "ModelSamplingDiscreteFlow", "Flux", 16, 6.0),
    ):
        sampling_type = type("Sampling", (namespace[sampling_name], namespace["CONST"]), {})
        for shift in (None, 0.4, 2.0):
            sampling = sampling_type()
            kwargs = {"multiplier": 1.0} if family == "lumina2" else {}
            sampling.set_parameters(shift=default if shift is None else shift, **kwargs)
            sigmas = namespace["simple_scheduler"](sampling, 3)
            latent_format = namespace[format_name]()
            latent = torch.zeros((1, channels, 2, 2))
            noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(7))
            latent_in = latent_format.process_in(latent) if torch.count_nonzero(latent) else latent
            initial = sampling.noise_scaling(sigmas[0], noise, latent_in)

            def model(
                x: torch.Tensor, sigma: torch.Tensor, sampling: Any = sampling
            ) -> torch.Tensor:
                timestep = sampling.timestep(sigma).reshape(-1, 1, 1, 1)
                output = sampling.calculate_input(sigma, x) * 0.17 + timestep.square() * 0.07
                return sampling.calculate_denoised(sigma, output, x)

            output = namespace["sample_euler"](model, initial, sigmas, disable=True)
            output = latent_format.process_out(sampling.inverse_noise_scaling(sigmas[-1], output))
            cases.append(
                {
                    "family": family,
                    "shift": shift,
                    "sigmas": sigmas.tolist(),
                    "noise_hash": digest(noise),
                    "output_hash": digest(output),
                }
            )
    OUTPUT.write_text(json.dumps({"reference": REFERENCE_COMMIT, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
