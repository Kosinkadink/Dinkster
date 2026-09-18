"""Generate RES4LYF scheduler goldens from the pinned reference checkouts.

Runs the REFERENCE implementations (RES4LYF sigmas.py
bong_tangent_scheduler and comfy.samplers.beta_scheduler with
alpha=0.5/beta=0.7, the two schedulers RES4LYF registers globally) and
writes tests/goldens/res4lyf_scheduler_goldens.json. Dinkster's native
ports are pinned against these values - the oracle is the reference
code itself, never a re-derivation.

RES4LYF's sigmas.py imports the ComfyUI server stack through its
package-relative modules; that scaffolding is inert for sigma math, so
this generator satisfies the two relative imports with minimal
in-process stubs and loads sigmas.py verbatim from the checkout.

Usage (needs a torch interpreter with the ComfyUI reference importable;
the workspace root venv is deliberately torch-free):

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    RES4LYF_REFERENCE=/path/to/RES4LYF-at-26036f64 \
        /path/to/torch-venv/bin/python tools/gen_res4lyf_scheduler_goldens.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

COMFYUI_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
RES4LYF_COMMIT = "26036f647ca15d3048a193daf99a40cecfc3820d"

STEPS = (2, 4, 12, 20, 30, 50)


def _verify_reference_commit(root: Path, expected: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != expected:
        raise SystemExit(f"reference checkout {root} is at {head}, expected {expected}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"refusing to generate goldens: {root} has uncommitted changes, "
            f"so the recorded commit would not describe the code that "
            f"actually ran:\n{dirty}"
        )


def _load_res4lyf_sigmas(res4lyf_root: Path) -> Any:
    """Load RES4LYF sigmas.py verbatim, stubbing its two relative imports
    (.res4lyf RESplain logging and .helper scheduler-list UI vocabulary,
    both inert for sigma values)."""
    package = types.ModuleType("res4lyf_reference")
    package.__path__ = [str(res4lyf_root)]
    sys.modules["res4lyf_reference"] = package

    res4lyf_stub = types.ModuleType("res4lyf_reference.res4lyf")
    res4lyf_stub.RESplain = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["res4lyf_reference.res4lyf"] = res4lyf_stub

    helper_stub = types.ModuleType("res4lyf_reference.helper")
    helper_stub.get_res4lyf_scheduler_list = lambda: ["normal"]  # type: ignore[attr-defined]
    sys.modules["res4lyf_reference.helper"] = helper_stub

    spec = importlib.util.spec_from_file_location(
        "res4lyf_reference.sigmas", res4lyf_root / "sigmas.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["res4lyf_reference.sigmas"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    comfy_root = Path(os.environ["COMFYUI_REFERENCE"]).resolve()
    res4lyf_root = Path(os.environ["RES4LYF_REFERENCE"]).resolve()
    _verify_reference_commit(comfy_root, COMFYUI_COMMIT)
    _verify_reference_commit(res4lyf_root, RES4LYF_COMMIT)

    sys.path.insert(0, str(comfy_root))
    from comfy.cli_args import args

    args.cpu = True

    import comfy.model_sampling as ms
    import comfy.samplers as samplers
    import torch
    from golden_platform import platform_golden_path, tuple_provenance

    sigmas_mod = _load_res4lyf_sigmas(res4lyf_root)

    discrete = ms.ModelSamplingDiscrete()  # SD15/SDXL defaults
    flow_sd3 = ms.ModelSamplingDiscreteFlow()
    flow_sd3.set_parameters(shift=3.0)
    spaces = {"discrete_sd": discrete, "flow_shift3": flow_sd3}

    def floats(tensor: torch.Tensor) -> list[float]:
        return [float(value) for value in tensor.reshape(-1).tolist()]

    schedules: dict[str, dict[str, dict[str, list[float]]]] = {
        "beta57": {
            space_name: {
                str(steps): floats(samplers.beta_scheduler(space, steps, alpha=0.5, beta=0.7))
                for steps in STEPS
            }
            for space_name, space in spaces.items()
        },
        # bong_tangent ignores model_sampling entirely; recording it under
        # both spaces proves the space-independence by execution.
        "bong_tangent": {
            space_name: {
                str(steps): floats(sigmas_mod.bong_tangent_scheduler(space, steps))
                for steps in STEPS
            }
            for space_name, space in spaces.items()
        },
    }

    payload = {
        "provenance": {
            "comfyui_commit": COMFYUI_COMMIT,
            "res4lyf_commit": RES4LYF_COMMIT,
            "generator": "tools/gen_res4lyf_scheduler_goldens.py",
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "schedules": schedules,
    }
    out = platform_golden_path(
        REPO / "tests" / "goldens" / "res4lyf_scheduler_goldens.json", torch.__version__
    )
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(schedules)} schedules x {len(spaces)} spaces x {len(STEPS)} steps)")


if __name__ == "__main__":
    main()
