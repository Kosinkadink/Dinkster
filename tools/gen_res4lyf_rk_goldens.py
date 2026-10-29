"""Generate executed goldens for the RES4LYF beta RK engine default path.

Drives the REAL sample_rk_beta at the pinned references (ComfyUI +
RES4LYF) with a deterministic mock denoiser, over the exact default
path the 16 named wrappers exercise (explicit tableaus, eta=0.5 hard
SDE noise or eta=0 for the _ode variants, gaussian noise streams).

Recorded per case, for the replay contract:
- the space's sigma_min / sigma_max (the model facts the engine
  consults for sigma preprocessing and the BONGMATH guard)
- preprocessed sigmas (after the engine's duplicate removal and
  sigma_min terminal handling): asserted bit-exact
- tableau A/B/C rows per outer step plus the resolved rk_type
  (captures multistep warm-up and the h >= 1 fallback): bit-exact
- both noise generator seeds and every noise draw in order (drawn
  float64, standardized by the generator): bit-exact
- per-stage model call inputs and sigmas: value-close (trajectory)
- per-step latents and the final latent: value-close (trajectory)

Seeding follows upstream's own seeded path: noise_seed = SEED + 1 is
passed explicitly (mirroring RES4LYF beta/samplers.py, which rewrites
noise_seed -1 to workflow_seed + 1 before sample_rk_beta), and the
substep seed is derived by upstream's own noise_seed + MAX_STEPS line.
The unseeded direct-wrapper default (torch.initial_seed() + 1) depends
on process-global RNG state and is out of parity scope.

Recording uses value-preserving wrappers only: a GaussianNoiseGenerator
subclass that records what the real implementation returns, and
wrappers around set_coeff / prepare_sigmas that record their real
results. No math is reimplemented or altered.

Usage (needs a torch interpreter with the ComfyUI reference importable
and mpmath/einops/PyWavelets available; the workspace root venv is
deliberately torch-free):

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    RES4LYF_REFERENCE=/path/to/RES4LYF-at-26036f64 \
        /path/to/torch-venv/bin/python tools/gen_res4lyf_rk_goldens.py
"""

from __future__ import annotations

import importlib
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

# The 16 named wrappers in RES4LYF beta/__init__.py: rk_type plus,
# for the _ode variants, eta=0 eta_substep=0. Everything else is the
# sample_rk_beta default.
NAMED_SAMPLERS: dict[str, dict[str, Any]] = {
    **{f"res_{kind}": {"rk_type": f"res_{kind}"} for kind in ("2m", "3m", "2s", "3s", "5s", "6s")},
    **{
        f"res_{kind}_ode": {"rk_type": f"res_{kind}", "eta": 0.0, "eta_substep": 0.0}
        for kind in ("2m", "3m", "2s", "3s", "5s", "6s")
    },
    "deis_2m": {"rk_type": "deis_2m"},
    "deis_3m": {"rk_type": "deis_3m"},
    "deis_2m_ode": {"rk_type": "deis_2m", "eta": 0.0, "eta_substep": 0.0},
    "deis_3m_ode": {"rk_type": "deis_3m", "eta": 0.0, "eta_substep": 0.0},
}


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


def _load_res4lyf_beta(res4lyf_root: Path) -> tuple[Any, Any, Any]:
    """Import the pinned RES4LYF beta modules as a real package so the
    relative imports resolve to the pin. Only res4lyf.py is stubbed
    (it registers PromptServer routes at import); the three names the
    beta modules use from it are logging/UI-only, inert for math."""
    package = types.ModuleType("res4lyf_reference")
    package.__path__ = [str(res4lyf_root)]
    sys.modules["res4lyf_reference"] = package

    stub = types.ModuleType("res4lyf_reference.res4lyf")
    stub.RESplain = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.is_debug_logging_enabled = lambda: False  # type: ignore[attr-defined]
    stub.get_display_sampler_category = lambda *args, **kwargs: ""  # type: ignore[attr-defined]
    sys.modules["res4lyf_reference.res4lyf"] = stub

    rk_sampler = importlib.import_module("res4lyf_reference.beta.rk_sampler_beta")
    rk_method = importlib.import_module("res4lyf_reference.beta.rk_method_beta")
    noise_mod = importlib.import_module("res4lyf_reference.beta.rk_noise_sampler_beta")
    return rk_sampler, rk_method, noise_mod


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

    rk_sampler, rk_method, noise_mod = _load_res4lyf_beta(res4lyf_root)
    noise_classes = importlib.import_module("res4lyf_reference.beta.noise_classes")

    eps_space = type("model_sampling", (ms.ModelSamplingDiscrete, ms.EPS), {})()
    const_space = type("model_sampling", (ms.ModelSamplingDiscreteFlow, ms.CONST), {})()

    def values(tensor: torch.Tensor) -> list[float]:
        return [float(item) for item in tensor.detach().reshape(-1).tolist()]

    # --- Recording seams (value-preserving). -------------------------------

    draw_log: list[dict[str, Any]] = []
    seed_log: list[int] = []

    real_gaussian = noise_classes.NOISE_GENERATOR_CLASSES_SIMPLE["gaussian"]

    class RecordingGaussian(real_gaussian):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seed_log.append(int(kwargs.get("seed")))
            self._stream_index = len(seed_log) - 1

        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            draw = super().__call__(*args, **kwargs)
            draw_log.append(
                {
                    "stream": self._stream_index,
                    "dtype": str(draw.dtype),
                    "values": values(draw),
                }
            )
            return draw

    noise_classes.NOISE_GENERATOR_CLASSES_SIMPLE["gaussian"] = RecordingGaussian

    coeff_log: list[dict[str, Any]] = []
    real_set_coeff = rk_method.RK_Method_Beta.set_coeff

    def recording_set_coeff(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_set_coeff(self, *args, **kwargs)
        coeff_log.append(
            {
                "rk_type": self.rk_type,
                "a": [[float(v) for v in row] for row in self.A],
                "b": [[float(v) for v in row] for row in self.B],
                "c": [float(v) for v in self.C],
                "multistep_stages": int(self.multistep_stages),
                "rows": int(self.rows),
            }
        )
        return result

    rk_method.RK_Method_Beta.set_coeff = recording_set_coeff

    prepared_log: list[list[float]] = []
    real_prepare = noise_mod.RK_NoiseSampler.prepare_sigmas

    def recording_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_prepare(self, *args, **kwargs)
        prepared_log.append(values(self.sigmas))
        return result

    noise_mod.RK_NoiseSampler.prepare_sigmas = recording_prepare

    # --- Mock model. --------------------------------------------------------

    class Model:
        """Black-box comfy-shaped denoiser: quadratic in x, sigma-damped."""

        def __init__(self, space: Any) -> None:
            inner_inner = SimpleNamespace(
                device=torch.device("cpu"),
                model_sampling=space,
                diffusion_model=SimpleNamespace(),
            )
            self.inner_model = SimpleNamespace(
                inner_model=inner_inner,
                model_patcher=SimpleNamespace(
                    model=SimpleNamespace(diffusion_model=SimpleNamespace())
                ),
            )
            self.sigmas: torch.Tensor | None = None
            self.calls: list[torch.Tensor] = []
            self.call_sigmas: list[float] = []

        def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **_: object) -> torch.Tensor:
            self.calls.append(x.detach().clone())
            scalar = float(sigma.reshape(-1)[0])
            self.call_sigmas.append(scalar)
            return x * (1.0 / (1.0 + scalar)) + x.square() * (0.05 / (1.0 + scalar))

    def case(
        space: Any,
        steps: int,
        wrapper_kwargs: dict[str, Any],
        schedule: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        draw_log.clear()
        seed_log.clear()
        coeff_log.clear()
        prepared_log.clear()

        if schedule is None:
            schedule = samplers.normal_scheduler(space, steps).to(torch.float64)
        initial = torch.tensor(INITIAL, dtype=torch.float32).reshape(1, 1, 2, 2)
        model = Model(space)
        step_latents: list[dict[str, Any]] = []

        def callback(event: dict[str, Any]) -> None:
            step_latents.append(
                {
                    "i": int(event["i"]),
                    "final": bool(event.get("final", False)),
                    "x": values(event["x"]),
                }
            )

        result = rk_sampler.sample_rk_beta(
            model,
            initial.clone(),
            schedule,
            None,
            {"model_options": {"transformer_options": {}}},
            callback,
            True,
            noise_seed=SEED + 1,
            **wrapper_kwargs,
        )
        assert len(prepared_log) == 1
        return {
            "sigma_min": float(space.sigma_min),
            "sigma_max": float(space.sigma_max),
            "schedule_sigmas": values(schedule),
            "prepared_sigmas": prepared_log[0],
            "seeds": list(seed_log),
            "coeffs": list(coeff_log),
            "model_call_sigmas": model.call_sigmas,
            "model_calls": [values(call) for call in model.calls],
            "noise_draws": list(draw_log),
            "steps": step_latents,
            "final": values(result),
        }

    cases: dict[str, dict[str, Any]] = {}
    for name, kwargs in NAMED_SAMPLERS.items():
        for steps in (4, 8):
            cases[f"{name}_{steps}"] = case(eps_space, steps, dict(kwargs))
    # h_no_eta >= 1 fallback: 2 steps over the full discrete range makes
    # every step's -log(sigma_next/sigma) large.
    cases["res_2m_fallback_2"] = case(eps_space, 2, {"rk_type": "res_2m"})
    cases["deis_2m_fallback_2"] = case(eps_space, 2, {"rk_type": "deis_2m"})
    # CONST space: variance-preserving eta formulas and the non-EPS
    # reconstruction branch.
    cases["res_2s_const_4"] = case(const_space, 4, {"rk_type": "res_2s"})
    cases["res_2m_const_8"] = case(const_space, 8, {"rk_type": "res_2m"})
    # The general rk_beta entry with a non-default rk_type.
    cases["rk_beta_res_3s_4"] = case(eps_space, 4, {"rk_type": "res_3s"})
    # Synthetic schedules exercising the remaining prepare_sigmas
    # branches no scheduler output reaches: a consecutive duplicate
    # (removed) with a terminal far from sigma_min (sigma_min inserted
    # before the trailing zero), and a terminal below sigma_min
    # (replaced by sigma_min).
    cases["res_2m_synth_insert_5"] = case(
        eps_space,
        5,
        {"rk_type": "res_2m"},
        schedule=torch.tensor([14.614642, 8.0, 8.0, 2.0, 0.5, 0.0], dtype=torch.float64),
    )
    cases["res_2m_synth_replace_4"] = case(
        eps_space,
        4,
        {"rk_type": "res_2m"},
        schedule=torch.tensor([14.614642, 2.0, 0.5, 0.01, 0.0], dtype=torch.float64),
    )

    payload = {
        "_meta": {
            "comfyui_commit": COMFYUI_COMMIT,
            "res4lyf_commit": RES4LYF_COMMIT,
            "generator": "tools/gen_res4lyf_rk_goldens.py",
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
        / "res4lyf_rk_goldens.json",
        torch.__version__,
    )
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
