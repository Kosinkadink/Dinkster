"""Generate continuous-EDM SDXL v-pred goldens from ComfyUI @ 947c2749.

Usage from the Dinkster root:

    DINKSTER_COMFYUI_ROOT=/path/to/ComfyUI-at-947c2749 \
        /path/to/comfyui-python tools/gen_sdxl_edm_vpred_goldens.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from golden_platform import platform_golden_path, platform_provenance

PIN = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
BASE_OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "sdxl_edm_vpred_goldens.json"
)


def main() -> None:
    root_value = os.environ.get("DINKSTER_COMFYUI_ROOT")
    if root_value is None:
        raise SystemExit("DINKSTER_COMFYUI_ROOT must name the pinned ComfyUI checkout")
    root = Path(root_value).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise SystemExit(f"ComfyUI HEAD is {head}, expected {PIN}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise SystemExit("ComfyUI golden checkout must be clean")

    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()

    import torch
    from comfy import model_sampling, samplers

    output = platform_golden_path(BASE_OUT, torch.__version__)

    sigma_min = 0.125
    sigma_max = 42.5
    sampling = model_sampling.ModelSamplingContinuousEDM()
    sampling.set_parameters(sigma_min, sigma_max, 1.0)
    vpred = model_sampling.V_PREDICTION()
    vpred.sigma_data = 1.0
    sigma = torch.tensor(0.7, dtype=torch.float32)
    model_input = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float32)
    model_output = torch.tensor([0.125, 0.75, -0.5, 1.5], dtype=torch.float32)
    table = sampling.sigmas
    close_sampling = model_sampling.ModelSamplingContinuousEDM()
    close_sampling.set_parameters(1.0, 1.00000001, 1.0)
    cosxl_sampling = model_sampling.ModelSamplingContinuousEDM()
    cosxl_sampling.set_parameters(0.0020000000949949026, 120.0, 1.0)
    payload = {
        "_meta": {
            "comfyui_commit": head,
            "device": "cpu",
            "dtype": "float32",
            "tolerance": {"atol": 1e-7, "rtol": 1e-6},
            **platform_provenance(torch.__version__),
        },
        "bounds": {"sigma_max": sigma_max, "sigma_min": sigma_min},
        "denoiser": {
            "calculate_denoised": vpred.calculate_denoised(
                sigma, model_output, model_input
            ).tolist(),
            "calculate_input": vpred.calculate_input(sigma, model_input).tolist(),
            "model_input": model_input.tolist(),
            "model_output": model_output.tolist(),
            "sigma": float(sigma),
            "timestep": float(sampling.timestep(sigma)),
        },
        "space": {
            "sigma_of_t": {
                "-0.5": float(sampling.sigma(torch.tensor(-0.5))),
                "0.0": float(sampling.sigma(torch.tensor(0.0))),
                "0.75": float(sampling.sigma(torch.tensor(0.75))),
            },
            "table_head": table[:3].tolist(),
            "table_mid": float(table[len(table) // 2]),
            "table_tail": table[-3:].tolist(),
            "timestep_of_sigma": {
                "0.125": float(sampling.timestep(torch.tensor(0.125))),
                "0.7": float(sampling.timestep(torch.tensor(0.7))),
                "42.5": float(sampling.timestep(torch.tensor(42.5))),
            },
        },
        "schedules_20": {
            name: samplers.calculate_sigmas(sampling, name, 20).tolist()
            for name in samplers.SCHEDULER_NAMES
        },
        "close_bounds": {
            "sigma_max": 1.00000001,
            "sigma_min": 1.0,
            "schedules_20": {
                name: samplers.calculate_sigmas(close_sampling, name, 20).tolist()
                for name in ("simple", "ddim_uniform", "beta")
            },
        },
        "cosxl": {
            "normal": samplers.calculate_sigmas(cosxl_sampling, "normal", 20).tolist(),
        },
        "percent_to_sigma": {
            str(percent): sampling.percent_to_sigma(percent) for percent in (0.0, 0.2, 0.8, 1.0)
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
