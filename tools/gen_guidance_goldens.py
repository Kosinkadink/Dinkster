"""Generate guidance callback goldens by executing pinned ComfyUI.

Usage:
    COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      .venv-torch/bin/python tools/gen_guidance_goldens.py

The reference may instead be supplied with ``--comfy-root``. The checkout must
be clean and exactly at the audited commit. Expected values are produced only
by ``comfy.samplers.sampling_function``; this script does not reproduce its
CFG or callback orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/guidance_goldens.json"


def reference_root() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
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
from comfy.samplers import sampling_function  # noqa: E402

X = torch.tensor([[[[0.25, -0.5], [1.25, 0.75]]]], dtype=torch.float32)
SIGMA = torch.tensor([0.625], dtype=torch.float32)
COND = torch.tensor([[[[1.5, -2.0], [0.5, 3.0]]]], dtype=torch.float32)
UNCOND = torch.tensor([[[[-0.5, 1.0], [2.0, -1.5]]]], dtype=torch.float32)


def enc(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": "float32", "data": value.flatten().tolist()}


def run_case(name: str, cfg: float, *, force: bool = False, mode: str = "standard") -> dict:
    calls: list[str] = []

    def evaluate(args):
        calls.append("evaluate")
        return [
            COND.clone(),
            UNCOND.clone() if args["conds"][1] is not None else torch.zeros_like(X),
        ]

    options = {"sampler_calc_cond_batch_function": evaluate}
    if force:
        options["disable_cfg1_optimization"] = True

    if mode in {"compose", "bypass"}:

        def pre_a(args):
            calls.append("pre_a")
            return [args["conds_out"][0] + 0.25, args["conds_out"][1] - 0.5]

        def pre_b(args):
            calls.append("pre_b")
            return [item * 1.5 for item in args["conds_out"]]

        def reducer(args):
            calls.append("reducer")
            # Comfy's custom reducer returns noise, not denoised output.
            return args["input"] - (args["cond_denoised"] * 0.75 + args["uncond_denoised"] * 0.25)

        def post_a(args):
            calls.append("post_a")
            return args["denoised"] + 0.125

        def post_b(args):
            calls.append("post_b")
            return args["denoised"] * 0.8

        options["sampler_cfg_function"] = reducer
        if mode == "compose":
            options["sampler_pre_cfg_function"] = [pre_a, pre_b]
            options["sampler_post_cfg_function"] = [post_a, post_b]

    if mode == "rescale":

        def rescale(args):
            calls.append("rescale")
            cond = args["cond"]
            uncond = args["uncond"]
            combined = uncond + (cond - uncond) * args["cond_scale"]
            dims = tuple(range(1, combined.ndim))
            ratio = cond.std(dim=dims, keepdim=True) / combined.std(dim=dims, keepdim=True)
            return combined * ratio * 0.7 + combined * 0.3

        options["sampler_cfg_function"] = rescale

    output = sampling_function(None, X, SIGMA, "u", "c", cfg, options)
    return {
        "cfg_scale": cfg,
        "force_uncond": force,
        "mode": mode,
        "calls": calls,
        "input": enc(X),
        "cond": enc(COND),
        "uncond": enc(UNCOND if force or cfg != 1.0 else torch.zeros_like(X)),
        "output": enc(output),
    }


def main() -> None:
    module = Path(sys.modules[sampling_function.__module__].__file__ or "").resolve()
    if not module.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"comfy.samplers was imported from {module}, not {COMFY_ROOT}")
    cases = {
        "standard_cfg": run_case("standard_cfg", 2.5),
        "cfg1_skip_uncond": run_case("cfg1_skip_uncond", 1.0),
        "cfg1_force_uncond": run_case("cfg1_force_uncond", 1.0, force=True),
        "custom_compose": run_case("custom_compose", 2.0, mode="compose"),
        "custom_bypass": run_case("custom_bypass", 2.0, mode="bypass"),
        "cfg_rescale": run_case("cfg_rescale", 3.25, mode="rescale"),
    }
    payload = {
        "reference": {"repo": "ComfyUI", "commit": commit, "torch": torch.__version__},
        "cases": cases,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
