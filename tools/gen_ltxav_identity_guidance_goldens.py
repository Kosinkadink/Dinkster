"""Generate LTX-2 ID-LoRA identity-guidance goldens from pinned ComfyUI.

Usage:
    COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      /path/to/ComfyUI-goldenref/.venv/bin/python \
      tools/gen_ltxav_identity_guidance_goldens.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/ltxav_identity_guidance_goldens.json"


def reference_root() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path)
    args = parser.parse_args()
    value = args.comfy_root or os.environ.get("COMFYUI_REFERENCE")
    if value is None:
        raise SystemExit("pass --comfy-root or set COMFYUI_REFERENCE")
    return Path(value).resolve()


COMFY_ROOT = reference_root()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=COMFY_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


commit = git_output("rev-parse", "HEAD")
dirty = git_output("status", "--porcelain")
if commit != REFERENCE_COMMIT:
    raise SystemExit(f"reference is at {commit}; required {REFERENCE_COMMIT}")
if dirty:
    raise SystemExit(f"reference checkout must be clean:\n{dirty}")

sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402
from comfy import samplers  # noqa: E402
from comfy_extras.nodes_lt import LTXVReferenceAudio  # noqa: E402


class _Sampling:
    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        return 1.0 - percent


class _Model:
    last_clone: _Model | None = None

    def __init__(self) -> None:
        self.callback = None

    def clone(self) -> _Model:
        clone = _Model()
        _Model.last_clone = clone
        return clone

    def get_model_object(self, name: str) -> _Sampling:
        if name != "model_sampling":
            raise KeyError(name)
        return _Sampling()

    def set_model_sampler_post_cfg_function(self, callback) -> None:
        self.callback = callback


class _AudioVAE:
    audio_sample_rate = 44100

    @staticmethod
    def encode(waveform: torch.Tensor) -> torch.Tensor:
        if tuple(waveform.shape) != (1, 4, 1):
            raise ValueError("unexpected waveform shape")
        return torch.arange(256, dtype=torch.float32).reshape(1, 8, 2, 16)


COND = torch.tensor([11.0, -3.0, 0.5], dtype=torch.float32)
NO_REFERENCE = torch.tensor([5.0, 2.0, -1.5], dtype=torch.float32)
CFG_RESULT = torch.tensor([20.0, -7.0, 4.0], dtype=torch.float32)


def run_case(scale: float, sigma: float) -> dict[str, object]:
    _Model.last_clone = None
    LTXVReferenceAudio.execute(
        _Model(),
        [[torch.ones(1), {}]],
        [[torch.zeros(1), {}]],
        {"waveform": torch.ones((1, 1, 4)), "sample_rate": 44100},
        _AudioVAE(),
        scale,
        0.2,
        0.8,
    )
    clone = _Model.last_clone
    if clone is None or clone.callback is None:
        raise RuntimeError("reference node did not install its post-CFG callback")
    calls: list[dict[str, object]] = []
    original = samplers.calc_cond_batch

    def evaluate(model, conditions, input, current_sigma, model_options):
        del model, input, current_sigma, model_options
        no_reference = conditions[0]
        stripped = all("ref_audio" not in entry.get("model_conds", {}) for entry in no_reference)
        calls.append({"stripped": stripped})
        return (NO_REFERENCE.clone(),)

    samplers.calc_cond_batch = evaluate
    try:
        output = clone.callback(
            {
                "denoised": CFG_RESULT.clone(),
                "sigma": torch.tensor([sigma], dtype=torch.float32),
                "cond_denoised": COND.clone(),
                "cond": [
                    {
                        "model_conds": {"ref_audio": {"tokens": torch.ones(1)}},
                        "retained": True,
                    }
                ],
                "model_options": {},
                "input": torch.zeros_like(CFG_RESULT),
                "model": object(),
            }
        )
    finally:
        samplers.calc_cond_batch = original
    return {
        "scale": scale,
        "sigma": sigma,
        "sigma_start": 0.8,
        "sigma_end": 0.2,
        "conditional": COND.tolist(),
        "no_reference": NO_REFERENCE.tolist(),
        "cfg_result": CFG_RESULT.tolist(),
        "calls": calls,
        "output": output.tolist(),
    }


def main() -> None:
    module = Path(sys.modules[LTXVReferenceAudio.__module__].__file__ or "").resolve()
    if not module.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"nodes_lt was imported from {module}, not {COMFY_ROOT}")
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "generator": "tools/gen_ltxav_identity_guidance_goldens.py",
            "platform": sys.platform,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "cases": {
            "active": run_case(3.0, 0.5),
            "above_window": run_case(3.0, 0.9),
            "below_window": run_case(3.0, 0.1),
            "zero_scale": run_case(0.0, 0.5),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
