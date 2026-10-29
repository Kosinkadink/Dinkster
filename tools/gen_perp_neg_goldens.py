"""Generate Perp-Neg guider goldens by executing pinned ComfyUI node code.

Usage:
    PYTHONPATH=<comfy-deps> COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      .venv-gpu/bin/python tools/gen_perp_neg_goldens.py

The reference may instead be supplied with ``--comfy-root``. The checkout
must be clean and exactly at the audited commit. Each case drives the real
``comfy_extras.nodes_perpneg.Guider_PerpNeg.predict_noise`` with recorded
lane predictions: ``comfy.samplers.calc_cond_batch`` is substituted with a
stub that returns the recorded prediction for each requested lane and
calc_cond_batch's untouched zeros for dropped lanes; this script never
re-implements the combination math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/perp_neg_goldens.json"


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

import comfy.options  # noqa: E402

# Honor forwarded ComfyUI flags such as --cpu, which a CPU-only torch build
# needs before comfy.model_management probes CUDA at import time.
comfy.options.enable_args_parsing()

import comfy.samplers  # noqa: E402
import torch  # noqa: E402
from comfy_extras import nodes_eps, nodes_perpneg, nodes_tcfg  # noqa: E402


def _recorded_calc_cond_batch(
    model: Any, conds: list[Any], x_in: torch.Tensor, timestep: Any, model_options: Any
) -> list[torch.Tensor]:
    del model, timestep, model_options
    return [cond.clone() if cond is not None else torch.zeros_like(x_in) for cond in conds]


comfy.samplers.calc_cond_batch = _recorded_calc_cond_batch


class _PatcherStub:
    """Stands in for the ModelPatcher Guider_PerpNeg wraps."""

    model_options: dict[str, Any] = {}


class _HookRecorder:
    """Records the sampler pre/post-CFG callbacks a node registers."""

    def __init__(self) -> None:
        self.pre: list[Any] = []
        self.post: list[Any] = []

    def clone(self) -> _HookRecorder:
        return self

    def set_model_sampler_pre_cfg_function(self, fn: Any, **_: Any) -> None:
        self.pre.append(fn)

    def set_model_sampler_post_cfg_function(self, fn: Any, **_: Any) -> None:
        self.post.append(fn)


def seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator)


def enc(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": "float32", "data": value.flatten().tolist()}


SHAPE = (2, 4, 8, 8)


def run_case(
    cfg: float,
    neg_scale: float,
    *,
    sigmas: tuple[float, ...] = (0.625,),
    seed: int = 0,
    shape: tuple[int, ...] = SHAPE,
    chain: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    x = seeded(shape, seed)
    positives = [seeded(shape, seed + 1 + 3 * step) for step in range(len(sigmas))]
    negatives = [seeded(shape, seed + 2 + 3 * step) for step in range(len(sigmas))]
    empties = [seeded(shape, seed + 3 + 3 * step) for step in range(len(sigmas))]

    model_options: dict[str, Any] = {}
    if chain is not None:
        transform, params = chain
        recorder = _HookRecorder()
        if transform == "epsilon_scaling":
            nodes_eps.EpsilonScaling.execute(recorder, **params)
        elif transform == "tcfg":
            nodes_tcfg.TCFG.execute(recorder, **params)
        else:
            raise SystemExit(f"unknown chain transform {transform!r}")
        if recorder.pre:
            model_options["sampler_pre_cfg_function"] = list(recorder.pre)
        if recorder.post:
            model_options["sampler_post_cfg_function"] = list(recorder.post)

    guider = nodes_perpneg.Guider_PerpNeg(_PatcherStub())
    guider.inner_model = object()
    guider.set_cfg(cfg, neg_scale)

    outputs = []
    for step in range(len(sigmas)):
        guider.conds = {
            "positive": positives[step],
            "negative": negatives[step],
            "empty_negative_prompt": empties[step],
        }
        sigma_t = torch.tensor([sigmas[step]] * shape[0], dtype=torch.float32)
        outputs.append(guider.predict_noise(x, sigma_t, model_options))

    result: dict[str, Any] = {
        "params": {"neg_scale": neg_scale},
        "cfg_scale": cfg,
        "sigmas": list(sigmas),
        "input": enc(x),
        "positives": [enc(value) for value in positives],
        "negatives": [enc(value) for value in negatives],
        "empties": [enc(value) for value in empties],
        "outputs": [enc(value) for value in outputs],
    }
    if chain is not None:
        result["chain"] = {"transform": chain[0], "params": chain[1]}
    return result


def main() -> None:
    for module in (nodes_eps, nodes_perpneg):
        path = Path(module.__file__ or "").resolve()
        if not path.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"{module.__name__} was imported from {path}, not {COMFY_ROOT}")

    cases: dict[str, dict[str, Any]] = {}

    def case(name: str, cfg: float, neg_scale: float, **kw: Any) -> None:
        if name in cases:
            raise SystemExit(f"duplicate case {name}")
        cases[name] = run_case(cfg, neg_scale, seed=len(cases) * 16 + 1, **kw)

    case("perp_neg_case1", 8.0, 1.0)
    case("perp_neg_case2", 4.5, 0.7)
    # neg_scale == 0 exercises the reference's negative-lane drop.
    case("perp_neg_case3", 3.0, 0.0)
    # neg_scale == 0 at cfg == 1 additionally drops the empty lane.
    case("perp_neg_case4", 1.0, 0.0)
    # cfg == 1 with an active neg_scale keeps every lane.
    case("perp_neg_case5", 1.0, 1.0)
    case("perp_neg_case6", 7.5, 0.8, sigmas=(0.9, 0.625, 0.35))
    case(
        "perp_neg_chain_case1",
        3.0,
        0.75,
        chain=("epsilon_scaling", {"scaling_factor": 1.005}),
    )
    # TCFG registers a pre-CFG hook, exercising the three-lane conds_out
    # surface the reference exposes to sampler_pre_cfg_function.
    case("perp_neg_pre_chain_case1", 5.5, 0.9, chain=("tcfg", {}))
    # With the negative lane dropped, the reference passes None in
    # conds[:2] and TCFG's hook skips; pins that skip parity.
    case("perp_neg_pre_chain_case2", 4.0, 0.0, chain=("tcfg", {}))

    payload: dict[str, Any] = {
        "reference": {"repo": "ComfyUI", "commit": commit, "torch": torch.__version__},
        "cases": cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        payload["_meta"] = provenance
    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
