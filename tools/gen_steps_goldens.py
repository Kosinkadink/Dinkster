"""Generate step-planning goldens from the ComfyUI reference checkout.

Runs the REFERENCE KSampler.set_steps / KSampler.calculate_sigmas /
Sampler.max_denoise (comfy/samplers.py @ the audited baseline) over a
minimal model stand-in and writes tests/goldens/steps_goldens.json.
Dinkster's steps.sampling_sigmas / steps.max_denoise ports are pinned
against these values - the oracle is the reference code itself, never
a re-derivation.

Usage (needs a torch interpreter; the workspace root venv is
deliberately torch-free):

    PYTHONPATH=../ComfyUI:../comfy-aimdo:../comfy-kitchen \
        /path/to/torch-venv/bin/python tools/gen_steps_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (the script records `git -C ../ComfyUI rev-parse HEAD` in the
output for provenance).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import comfy.model_sampling as ms  # noqa: E402
import comfy.samplers as samplers  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "steps_goldens.json"

STEPS = (4, 20)
# null rides as JSON null -> None on the native side.
DENOISE = (None, 1.0, 0.9999, 0.75, 0.5, 0.25, 0.0)
# euler: no penultimate discard; dpm_2: in the reference's
# DISCARD_PENULTIMATE_SIGMA_SAMPLERS set.
SAMPLERS = ("euler", "dpm_2")
# Schedulers per space: simple needs the discrete sigma table.
SCHEDULERS = {
    "discrete_sd": ("normal", "simple"),
    "flux_shift1.15": ("normal", "karras"),
}


def floats(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def set_steps_goldens(model_sampling: object) -> list[dict[str, object]]:
    model = SimpleNamespace(get_model_object=lambda name: model_sampling)
    out: list[dict[str, object]] = []
    space_key = None
    for key, sampling in SPACES.items():
        if sampling is model_sampling:
            space_key = key
    assert space_key is not None
    for scheduler in SCHEDULERS[space_key]:
        for sampler in SAMPLERS:
            for steps in STEPS:
                for denoise in DENOISE:
                    k = samplers.KSampler(
                        model,
                        steps,
                        "cpu",
                        sampler=sampler,
                        scheduler=scheduler,
                        denoise=denoise,
                    )
                    out.append(
                        {
                            "scheduler": scheduler,
                            "sampler": sampler,
                            "steps": steps,
                            "denoise": denoise,
                            "sigmas": floats(k.sigmas),
                        }
                    )
    return out


def max_denoise_goldens(model_sampling: object) -> list[dict[str, object]]:
    sampler = samplers.Sampler()
    sigma_max = float(model_sampling.sigma_max)
    model_wrap = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=model_sampling))
    probes = (
        sigma_max,
        sigma_max * (1.0 + 1e-6),
        sigma_max * (1.0 - 1e-6),
        sigma_max * (1.0 - 1e-4),
        sigma_max * 1.5,
        sigma_max * 0.5,
    )
    return [
        {
            "sigma0": probe,
            "result": bool(sampler.max_denoise(model_wrap, torch.tensor([probe, 0.0]))),
        }
        for probe in probes
    ]


SPACES = {
    "discrete_sd": ms.ModelSamplingDiscrete(),
    "flux_shift1.15": ms.ModelSamplingFlux(),
}


def main() -> None:
    comfy_root = (REPO.parent / "ComfyUI").resolve()
    baseline = subprocess.run(
        ["git", "-C", str(comfy_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(comfy_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"refusing to generate goldens: {comfy_root} has uncommitted "
            f"changes, so the recorded commit would not describe the code "
            f"that actually ran:\n{dirty}"
        )

    goldens = {
        "_meta": {
            "reference_commit": baseline,
            "torch": torch.__version__,
            "generator": "tools/gen_steps_goldens.py",
        },
        "discard_penultimate_samplers": sorted(
            samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS
        ),
        "set_steps": {key: set_steps_goldens(sampling) for key, sampling in SPACES.items()},
        "max_denoise": {key: max_denoise_goldens(sampling) for key, sampling in SPACES.items()},
    }
    OUT.write_text(json.dumps(goldens, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
