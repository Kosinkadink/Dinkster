"""Generate the pinned SDXL normal 30-step CPU schedule.

Usage: .venv-torch/bin/python tools/gen_sdxl_normal_schedule_golden.py COMFY_ROOT
Generate on Linux with Python 3.12.3, torch 2.13.0+cpu,
ATEN_CPU_CAPABILITY=avx2, ONEDNN_MAX_CPU_ISA=AVX2, OMP_NUM_THREADS=4,
MKL_NUM_THREADS=4, and MKL_CBWR unset.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from golden_platform import tuple_provenance

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "packages/dinkster-inference-torch/tests/goldens/sdxl_normal_30_cpu_b78cec87.json"


def main() -> None:
    comfy_root = Path(sys.argv[1]).resolve()
    actual = subprocess.check_output(["git", "-C", str(comfy_root), "rev-parse", "HEAD"], text=True)
    if actual.strip() != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI must be at {REFERENCE_COMMIT}")

    sys.path.insert(0, str(comfy_root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy.model_sampling import ModelSamplingDiscrete
    from comfy.samplers import normal_scheduler

    torch.set_default_device("cpu")
    sampling = ModelSamplingDiscrete()
    sigmas = normal_scheduler(sampling, 30)
    assert sampling.sigmas.device.type == "cpu"
    assert sigmas.device.type == "cpu"
    source = (comfy_root / "comfy/samplers.py").read_bytes()
    document = {
        "reference": {
            "commit": REFERENCE_COMMIT,
            "path": "comfy/samplers.py",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "executed_symbol": "normal_scheduler",
        },
        "sigmas": sigmas.tolist(),
        "_meta": tuple_provenance(str(torch.__version__), pin_cpu=True),
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
