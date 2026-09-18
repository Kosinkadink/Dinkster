"""Generate model-aware scheduler goldens from the ComfyUI reference checkout.

Runs the REFERENCE node implementations (comfy_extras
nodes_align_your_steps.py, nodes_gits.py, nodes_optimalsteps.py @ the
audited baseline) and writes tests/goldens/
model_aware_scheduler_goldens.json. Dinkster's native ports are pinned
against these values - the oracle is the reference code itself, never a
re-derivation.

The reference files compute sigmas with numpy/torch only; their sole
ComfyUI dependency is the comfy_api node-schema scaffolding, which pulls
the full server dependency stack at import. That scaffolding is inert
for sigma math, so this generator satisfies it with a minimal in-process
stub and loads each reference file verbatim from the checkout.

Usage (needs a torch interpreter; the workspace root venv is
deliberately torch-free):

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
        /path/to/torch-venv/bin/python tools/gen_model_aware_scheduler_goldens.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

OUT = platform_golden_path(
    REPO / "tests" / "goldens" / "model_aware_scheduler_goldens.json", torch.__version__
)
BASELINE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

AYS_MODELS = ("SD1", "SDXL", "SVD")
AYS_STEPS = (1, 4, 10, 12, 20, 40)
GITS_COEFFS = (0.80, 0.95, 1.20, 1.35, 1.50)
GITS_STEPS = (2, 5, 10, 20, 21, 35)
OPTIMAL_MODELS = ("FLUX", "Wan", "Chroma")
OPTIMAL_STEPS = (3, 10, 20, 40)
DENOISES = (1.0, 0.7, 0.5, 0.25)


class _NodeOutput:
    def __init__(self, *args: Any, **_: Any) -> None:
        self.args = args

    @property
    def result(self) -> tuple[Any, ...] | None:
        return self.args if len(self.args) > 0 else None


def _accept_anything(*_: Any, **__: Any) -> None:
    return None


def _install_comfy_api_stub() -> None:
    io_stub = SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        NodeOutput=_NodeOutput,
        Schema=_accept_anything,
        Combo=SimpleNamespace(Input=_accept_anything),
        Int=SimpleNamespace(Input=_accept_anything),
        Float=SimpleNamespace(Input=_accept_anything),
        Sigmas=SimpleNamespace(Output=_accept_anything),
    )
    latest = ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})  # type: ignore[attr-defined]
    latest.io = io_stub  # type: ignore[attr-defined]
    package = ModuleType("comfy_api")
    package.latest = latest  # type: ignore[attr-defined]
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest


def _load_reference_module(reference_root: Path, stem: str) -> Any:
    path = reference_root / "comfy_extras" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"reference_{stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmas(node: Any, *args: Any) -> list[float]:
    tensor = node.execute(*args).result[0]
    return [float(value) for value in tensor.reshape(-1).tolist()]


def _verify_reference_commit(reference_root: Path) -> None:
    """Refuse to generate against anything but the pinned reference commit.

    A git checkout is verified directly; a non-git export (e.g. from
    git archive) requires COMFYUI_REFERENCE_COMMIT to attest the pin.
    """
    result = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        head = result.stdout.strip()
        if head != BASELINE_COMMIT:
            raise SystemExit(f"reference checkout is at {head}, expected {BASELINE_COMMIT}")
        return
    attested = os.environ.get("COMFYUI_REFERENCE_COMMIT")
    if attested != BASELINE_COMMIT:
        raise SystemExit(
            "reference root is not a git checkout; set "
            f"COMFYUI_REFERENCE_COMMIT={BASELINE_COMMIT} to attest its origin"
        )


def main() -> None:
    reference_root = Path(os.environ["COMFYUI_REFERENCE"])
    _verify_reference_commit(reference_root)
    _install_comfy_api_stub()
    nodes_ays = _load_reference_module(reference_root, "nodes_align_your_steps")
    nodes_gits = _load_reference_module(reference_root, "nodes_gits")
    nodes_optimal = _load_reference_module(reference_root, "nodes_optimalsteps")

    ays = [
        {
            "model_type": model_type,
            "steps": steps,
            "denoise": denoise,
            "sigmas": sigmas(nodes_ays.AlignYourStepsScheduler, model_type, steps, denoise),
        }
        for model_type in AYS_MODELS
        for steps in AYS_STEPS
        for denoise in DENOISES
    ]
    gits = [
        {
            "coeff": coeff,
            "steps": steps,
            "denoise": denoise,
            "sigmas": sigmas(nodes_gits.GITSScheduler, coeff, steps, denoise),
        }
        for coeff in GITS_COEFFS
        for steps in GITS_STEPS
        for denoise in DENOISES
    ]
    optimal = [
        {
            "model_type": model_type,
            "steps": steps,
            "denoise": denoise,
            "sigmas": sigmas(nodes_optimal.OptimalStepsScheduler, model_type, steps, denoise),
        }
        for model_type in OPTIMAL_MODELS
        for steps in OPTIMAL_STEPS
        for denoise in DENOISES
    ]
    payload = {
        "provenance": {
            "comfyui_commit": BASELINE_COMMIT,
            "generator": "tools/gen_model_aware_scheduler_goldens.py",
            **tuple_provenance(torch.__version__),
        },
        "align_your_steps": ays,
        "gits": gits,
        "optimal_steps": optimal,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(ays) + len(gits) + len(optimal)} cases)")


if __name__ == "__main__":
    main()
