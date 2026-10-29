"""Generate brownian SDE pairing goldens for the RES4LYF schedulers.

Runs the REFERENCE pairing end to end: sigmas from RES4LYF's
bong_tangent_scheduler and ComfyUI's beta, ddim_uniform, and custom beta
schedulers over the real ModelSamplingDiscrete space, fed to
the real comfy.k_diffusion sample_dpmpp_2m_sde with the real seeded
BrownianTreeNoiseSampler (constructed with the solver's own default
arguments) and a deterministic mock denoiser. Brownian trees decorrelate
on one-float32-ulp sigma differences, so this is the executed evidence
the numerical parity discipline requires before a schedule ships paired
with an SDE sampler. The replay
(packages/dinkster-inference-torch/tests/test_res4lyf_sde_pairing.py)
asserts the decorrelation-critical seams bit-exactly (sigmas, tree
bounds, noise queries, per-draw noise) and the trajectory (model calls,
per-step latents, final latent) value-close at its documented
TRAJECTORY_ATOL, because the Dinkster solver uses float64 scalar
coefficients where the reference uses float32 tensor math.

The fixture is CPU-pinned (_meta.cpu): the tree draws ride vectorized
kernels that drift by ULPs across CPU microarchitectures, so hosts with
a different CPU module-skip instead of failing.

Usage (needs a torch+torchsde interpreter with the ComfyUI reference
importable; the workspace root venv is deliberately torch-free):

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    RES4LYF_REFERENCE=/path/to/RES4LYF-at-26036f64 \
        /path/to/torch-venv/bin/python tools/gen_res4lyf_sde_pairing_goldens.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

COMFYUI_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
RES4LYF_COMMIT = "26036f647ca15d3048a193daf99a40cecfc3820d"

INITIAL = (0.5, -1.0, 2.0, 0.25)
SEED = 1234
STEPS = (4, 8)


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

    import comfy.k_diffusion.sampling as sampling
    import comfy.model_sampling as ms
    import comfy.samplers as samplers
    import torch
    from golden_platform import platform_golden_path, tuple_provenance

    sigmas_mod = _load_res4lyf_sigmas(res4lyf_root)
    space = ms.ModelSamplingDiscrete()  # SD15/SDXL defaults

    def values(tensor: torch.Tensor) -> list[float]:
        return [float(item) for item in tensor.detach().reshape(-1).tolist()]

    class Model:
        def __init__(self) -> None:
            # sample_dpmpp_2m_sde reads the model's sampling space for its
            # half-logSNR conversion; hand it the same space the schedules
            # were minted over, exactly as a real SD model would.
            self.inner_model = SimpleNamespace(
                model_patcher=SimpleNamespace(get_model_object=lambda _name: space)
            )
            self.calls: list[torch.Tensor] = []
            self.sigmas: list[float] = []

        def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **_: object) -> torch.Tensor:
            self.calls.append(x.detach().clone())
            scalar = float(sigma.reshape(-1)[0])
            self.sigmas.append(scalar)
            return x * (1.0 / (1.0 + scalar)) + x.square() * (0.05 / (1.0 + scalar))

    class RecordingBrownian:
        """The solver's own default noise sampler (BrownianTreeNoiseSampler
        over the schedule's positive bounds, seeded, cpu=True), with every
        query recorded."""

        def __init__(self, x: torch.Tensor, sigmas: torch.Tensor) -> None:
            sigma_min = float(sigmas[sigmas > 0].min())
            sigma_max = float(sigmas.max())
            self.bounds_args = (sigma_min, sigma_max)
            self.inner = sampling.BrownianTreeNoiseSampler(
                x, sigma_min, sigma_max, seed=SEED, cpu=True
            )
            self.queries: list[tuple[float, float]] = []
            self.draws: list[list[float]] = []

        def __call__(self, sigma_from: torch.Tensor, sigma_to: torch.Tensor) -> torch.Tensor:
            self.queries.append((float(sigma_from), float(sigma_to)))
            draw = self.inner(sigma_from, sigma_to)
            self.draws.append(values(draw))
            return draw

    def case(schedule_sigmas: torch.Tensor) -> dict[str, object]:
        initial = torch.tensor(INITIAL, dtype=torch.float32).reshape(1, 1, 2, 2)
        model = Model()
        noise = RecordingBrownian(initial, schedule_sigmas)
        steps: list[list[float]] = []

        def callback(event: dict[str, Any]) -> None:
            steps.append(values(event["x"]))

        result = sampling.sample_dpmpp_2m_sde(
            model,
            initial.clone(),
            schedule_sigmas,
            extra_args={"seed": SEED},
            callback=callback,
            disable=True,
            noise_sampler=noise,
        )
        return {
            "sigmas": values(schedule_sigmas),
            "tree_bounds": list(noise.bounds_args),
            "model_sigmas": model.sigmas,
            "model_calls": [values(value) for value in model.calls],
            "noise_queries": [list(query) for query in noise.queries],
            "noise_draws": noise.draws,
            "steps": steps,
            "final": values(result),
        }

    cases = {
        **{
            f"bong_tangent_{steps}": case(sigmas_mod.bong_tangent_scheduler(space, steps))
            for steps in STEPS
        },
        **{
            f"beta57_{steps}": case(samplers.beta_scheduler(space, steps, alpha=0.5, beta=0.7))
            for steps in STEPS
        },
        **{
            f"beta_{steps}": case(samplers.calculate_sigmas(space, "beta", steps))
            for steps in STEPS
        },
        **{
            f"ddim_uniform_{steps}": case(samplers.calculate_sigmas(space, "ddim_uniform", steps))
            for steps in STEPS
        },
        "custom_beta_13_a2_b5": case(samplers.beta_scheduler(space, 13, alpha=2.0, beta=5.0)),
    }

    payload = {
        "_meta": {
            "comfyui_commit": COMFYUI_COMMIT,
            "res4lyf_commit": RES4LYF_COMMIT,
            "generator": "tools/gen_res4lyf_sde_pairing_goldens.py",
            "sampler": "sample_dpmpp_2m_sde",
            "seed": SEED,
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__, pin_cpu=True),
        },
        "initial": list(INITIAL),
        "cases": cases,
    }
    out = platform_golden_path(
        REPO
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "res4lyf_sde_pairing_goldens.json",
        torch.__version__,
    )
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
