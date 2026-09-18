"""Generate discrete SDXL v-prediction goldens from ComfyUI @ 947c2749.

Usage from the Dinkster root:

    DINKSTER_COMFYUI_ROOT=/path/to/ComfyUI-at-947c2749 \
        /path/to/comfyui-python tools/gen_sdxl_vpred_goldens.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PIN = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "sdxl_vpred_goldens.json"
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
    import torch
    from comfy import model_sampling
    from golden_platform import platform_golden_path, tuple_provenance

    out = platform_golden_path(OUT, torch.__version__)

    sampling = model_sampling.ModelSamplingDiscrete(zsnr=True)
    vpred = model_sampling.V_PREDICTION()
    vpred.sigma_data = 1.0
    sigma = torch.tensor(0.7, dtype=torch.float32)
    model_input = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float32)
    model_output = torch.tensor([0.125, 0.75, -0.5, 1.5], dtype=torch.float32)
    payload = {
        "_meta": {
            "comfyui_commit": head,
            "device": "cpu",
            "dtype": "float32",
            "tolerance": {"atol": 1e-7, "rtol": 1e-6},
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "denoiser": {
            "calculate_denoised": vpred.calculate_denoised(
                sigma, model_output, model_input
            ).tolist(),
            "calculate_input": vpred.calculate_input(sigma, model_input).tolist(),
            "model_input": model_input.tolist(),
            "model_output": model_output.tolist(),
            "sigma": float(sigma),
        },
        "zsnr_sigmas": sampling.sigmas.tolist(),
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(out)


if __name__ == "__main__":
    main()
