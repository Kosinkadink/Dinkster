"""Execute pinned ComfyUI AuraFlow patch, schedules, and Euler/SDE solvers.

Usage: .venv-torch/bin/python tools/gen_model_sampling_aura_flow_goldens.py COMFY_ROOT
Patch storage and the deterministic denoiser are substitutes; model-sampling,
scheduler, Brownian noise, and solver arithmetic execute unmodified source.
Generate on Linux with Python 3.12.3, torch 2.13.0+cpu, torchsde 0.2.6,
ATEN_CPU_CAPABILITY=avx2, ONEDNN_MAX_CPU_ISA=AVX2, and OMP_NUM_THREADS=2.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import scipy.stats
import torch
import torchsde
from gen_model_sampling_flux_goldens import REFERENCE_COMMIT, ROOT, definitions
from golden_platform import tuple_provenance
from tqdm.auto import trange

OUTPUT = ROOT / "packages/dinkster-inference-torch/tests/goldens/model_sampling_aura_flow.json"


class Patcher:
    def __init__(self, sampling: Any) -> None:
        self.model = SimpleNamespace(model_config=SimpleNamespace(sampling_settings={}))
        self.sampling = sampling

    def clone(self) -> Patcher:
        return Patcher(self.sampling)

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return self.sampling

    def add_object_patch(self, name: str, value: Any) -> None:
        assert name == "model_sampling"
        self.sampling = value


class Denoiser:
    def __init__(self, patcher: Patcher) -> None:
        self.inner_model = SimpleNamespace(model_patcher=patcher)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x * 0.25


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
        "partial": partial,
        "trange": trange,
    }
    definitions(
        comfy / "comfy/model_sampling.py",
        ("time_snr_shift", "ModelSamplingDiscreteFlow", "CONST"),
        namespace,
    )
    namespace["comfy"] = SimpleNamespace(model_sampling=SimpleNamespace(**namespace))
    definitions(
        comfy / "comfy_extras/nodes_model_advanced.py",
        ("ModelSamplingSD3", "ModelSamplingAuraFlow"),
        namespace,
    )
    definitions(
        comfy / "comfy/samplers.py",
        ("simple_scheduler", "normal_scheduler", "beta_scheduler", "ddim_scheduler"),
        namespace,
    )
    definitions(comfy / "comfy/k_diffusion/utils.py", ("append_dims",), namespace)
    namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])
    definitions(
        comfy / "comfy/k_diffusion/sampling.py",
        (
            "append_zero",
            "get_sigmas_karras",
            "to_d",
            "sigma_to_half_log_snr",
            "offset_first_sigma_for_snr",
            "BatchedBrownianTree",
            "BrownianTreeNoiseSampler",
            "sample_euler",
            "sample_dpmpp_2m_sde",
        ),
        namespace,
    )
    cases = []
    for shift in (1.0, 1.73, 3.1, 6.0):
        original = Patcher(namespace["ModelSamplingDiscreteFlow"]())
        patched = namespace["ModelSamplingAuraFlow"]().patch_aura(original, shift)[0]
        sampling = patched.sampling
        assert original.sampling is not sampling
        assert original.sampling.shift == 1.0
        assert sampling.shift == shift and sampling.multiplier == 1.0
        for steps in (4, 30):
            schedules = {}
            for name in ("simple", "normal", "beta", "ddim_uniform", "karras"):
                if name == "karras":
                    sigmas = namespace["get_sigmas_karras"](
                        steps, float(sampling.sigma_min), float(sampling.sigma_max)
                    )
                else:
                    function = "ddim_scheduler" if name == "ddim_uniform" else f"{name}_scheduler"
                    sigmas = namespace[function](sampling, steps)
                latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
                schedules[name] = {
                    "sigmas": sigmas.tolist(),
                    "snr_sigmas": namespace["offset_first_sigma_for_snr"](
                        sigmas, sampling
                    ).tolist(),
                    "outputs": {
                        solver: namespace[f"sample_{solver}"](
                            Denoiser(patched),
                            latent.clone(),
                            sigmas,
                            extra_args={"seed": 23},
                            disable=True,
                        ).tolist()
                        for solver in ("euler", "dpmpp_2m_sde")
                    },
                }
            cases.append({"shift": shift, "steps": steps, "schedules": schedules})
    solver_cases = []
    for flow in (False, True):
        sampling = namespace["CONST"]() if flow else SimpleNamespace()
        patcher = Patcher(sampling)
        sigmas = torch.tensor([0.875, 0.61, 0.27, 0.04, 0.0])
        for solver_type in ("midpoint", "heun"):
            for eta, s_noise, noise_scale in (
                (1.0, 1.0, 1.0),
                (0.35, 0.7, 0.6),
                (0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0),
            ):
                sampling.noise_scale = noise_scale
                states: list[Any] = []
                noise_bounds: list[list[float]] = []
                latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8

                def noise(
                    sigma: torch.Tensor,
                    sigma_next: torch.Tensor,
                    bounds: list[list[float]] = noise_bounds,
                    like: torch.Tensor = latent,
                ) -> torch.Tensor:
                    bounds.append([float(sigma), float(sigma_next)])
                    return torch.full_like(like, 0.125 * len(bounds))

                def record(state: dict[str, Any], target: list[Any] = states) -> None:
                    target.append(state["x"].tolist())

                result = namespace["sample_dpmpp_2m_sde"](
                    Denoiser(patcher),
                    latent.clone(),
                    sigmas,
                    callback=record,
                    disable=True,
                    eta=eta,
                    s_noise=s_noise,
                    noise_sampler=noise,
                    solver_type=solver_type,
                )
                solver_cases.append(
                    {
                        "flow": flow,
                        "solver_type": solver_type,
                        "eta": eta,
                        "s_noise": s_noise,
                        "noise_scale": noise_scale,
                        "sigmas": sigmas.tolist(),
                        "states": states,
                        "noise_bounds": noise_bounds,
                        "output": result.tolist(),
                    }
                )
    OUTPUT.write_text(
        json.dumps(
            {
                "comfyui_commit": REFERENCE_COMMIT,
                "cases": cases,
                "solver_cases": solver_cases,
                "_meta": tuple_provenance(str(torch.__version__), pin_cpu=True),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
