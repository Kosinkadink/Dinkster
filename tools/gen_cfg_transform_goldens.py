"""Generate CFG-transform goldens by executing pinned ComfyUI node code.

Usage:
    PYTHONPATH=<comfy-deps> COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      .venv-gpu/bin/python tools/gen_cfg_transform_goldens.py

The reference may instead be supplied with ``--comfy-root``. The checkout must
be clean and exactly at the audited commit. Each case calls the real
comfy_extras node ``execute`` classmethod against a recording model stub, then
runs ``comfy.samplers.sampling_function`` with the callbacks the node
registered; this script never re-implements any transform math.
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
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/cfg_transform_goldens.json"


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

import torch  # noqa: E402
from comfy.model_sampling import CONST  # noqa: E402
from comfy.samplers import sampling_function  # noqa: E402
from comfy_extras import (  # noqa: E402
    nodes_apg,
    nodes_cfg,
    nodes_eps,
    nodes_fresca,
    nodes_lumina2,
    nodes_mahiro,
    nodes_model_advanced,
    nodes_tcfg,
)

FLOW_SAMPLING = type("FlowSampling", (CONST,), {})()
EPS_SAMPLING = object()


class _DiffusionModelStub:
    def __init__(self, in_channels: int) -> None:
        self.in_channels = in_channels


class _InnerModelStub:
    def __init__(self, in_channels: int) -> None:
        self.diffusion_model = _DiffusionModelStub(in_channels)


class RecordingModel:
    """Stands in for a ModelPatcher; records the sampler callbacks a node sets."""

    def __init__(self, *, model_sampling: Any = None, in_channels: int = 4) -> None:
        self.pre: list[Any] = []
        self.post: list[Any] = []
        self.cfg_fn: Any = None
        self.model_sampling = model_sampling
        self.model = _InnerModelStub(in_channels)
        self.current_patcher = self

    def clone(self) -> RecordingModel:
        return self

    def get_model_object(self, name: str) -> Any:
        if name != "model_sampling" or self.model_sampling is None:
            raise KeyError(name)
        return self.model_sampling

    def set_model_sampler_pre_cfg_function(self, fn: Any, **_: Any) -> None:
        self.pre.append(fn)

    def set_model_sampler_post_cfg_function(self, fn: Any, **_: Any) -> None:
        self.post.append(fn)

    def set_model_sampler_cfg_function(self, fn: Any) -> None:
        self.cfg_fn = fn


def seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator)


def enc(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": "float32", "data": value.flatten().tolist()}


SHAPE = (2, 4, 8, 8)


def run_case(
    apply_node: Any,
    params: dict[str, Any],
    cfg: float,
    *,
    sigmas: tuple[float, ...] = (0.625,),
    seed: int = 0,
    shape: tuple[int, ...] = SHAPE,
    model_sampling: Any = None,
    in_channels: int = 4,
    pass_model: bool = False,
) -> dict[str, Any]:
    x = seeded(shape, seed)
    conds = [seeded(shape, seed + 1 + 2 * step) for step in range(len(sigmas))]
    unconds = [seeded(shape, seed + 2 + 2 * step) for step in range(len(sigmas))]

    recorder = RecordingModel(model_sampling=model_sampling, in_channels=in_channels)
    apply_node(recorder, params)

    step_index = 0

    def evaluate(args: dict[str, Any]) -> list[torch.Tensor]:
        return [
            conds[step_index].clone(),
            unconds[step_index].clone() if args["conds"][1] is not None else torch.zeros_like(x),
        ]

    options: dict[str, Any] = {"sampler_calc_cond_batch_function": evaluate}
    if recorder.pre:
        options["sampler_pre_cfg_function"] = list(recorder.pre)
    if recorder.post:
        options["sampler_post_cfg_function"] = list(recorder.post)
    if recorder.cfg_fn is not None:
        options["sampler_cfg_function"] = recorder.cfg_fn

    outputs = []
    for step_index in range(len(sigmas)):
        sigma_t = torch.tensor([sigmas[step_index]] * shape[0], dtype=torch.float32)
        model_arg = recorder if pass_model else None
        outputs.append(sampling_function(model_arg, x, sigma_t, "u", "c", cfg, options))

    return {
        "params": params,
        "cfg_scale": cfg,
        "sigmas": list(sigmas),
        "input": enc(x),
        "conds": [enc(value) for value in conds],
        "unconds": [enc(value) for value in unconds],
        "outputs": [enc(value) for value in outputs],
    }


def apply_cfg_zero_star(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_cfg.CFGZeroStar.execute(model, **params)


def apply_cfg_norm(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_cfg.CFGNorm.execute(model, **params)


def apply_tcfg(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_tcfg.TCFG.execute(model, **params)


def apply_fresca(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_fresca.FreSca.execute(model, **params)


def apply_apg(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_apg.APG.execute(model, **params)


def apply_mahiro(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_mahiro.Mahiro.execute(model, **params)


def apply_epsilon_scaling(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_eps.EpsilonScaling.execute(model, **params)


def apply_rescale_cfg(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_model_advanced.RescaleCFG().patch(model, **params)


def apply_renorm_cfg(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_lumina2.RenormCFG.execute(model, **params)


def apply_tsr(model: RecordingModel, params: dict[str, Any]) -> None:
    nodes_eps.TemporalScoreRescaling.execute(model, **params)


def main() -> None:
    for module in (
        nodes_apg,
        nodes_cfg,
        nodes_eps,
        nodes_fresca,
        nodes_lumina2,
        nodes_mahiro,
        nodes_model_advanced,
        nodes_tcfg,
    ):
        path = Path(module.__file__ or "").resolve()
        if not path.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"{module.__name__} was imported from {path}, not {COMFY_ROOT}")

    descending = (0.9, 0.625, 0.35, 0.125)
    with_reset = (0.625, 0.35, 0.9, 0.625)
    cases: dict[str, dict[str, Any]] = {}

    def case(name: str, apply_node: Any, params: dict[str, Any], cfg: float, **kw: Any) -> None:
        if name in cases:
            raise SystemExit(f"duplicate case {name}")
        cases[name] = run_case(apply_node, params, cfg, seed=len(cases) * 16 + 1, **kw)
        cases[name]["transform"] = name.rsplit("_case", 1)[0]

    case("cfg_zero_star_case1", apply_cfg_zero_star, {}, 2.5)
    case("cfg_zero_star_case2", apply_cfg_zero_star, {}, 7.5)
    case("cfg_zero_star_case3", apply_cfg_zero_star, {}, 1.0)

    case("cfg_norm_case1", apply_cfg_norm, {"strength": 1.0}, 3.0)
    case("cfg_norm_case2", apply_cfg_norm, {"strength": 0.5}, 3.0)
    case("cfg_norm_case3", apply_cfg_norm, {"strength": 1.3}, 7.5)
    case("cfg_norm_pre_case1", apply_cfg_norm, {"strength": 1.0, "pre_cfg": True}, 3.0)
    case("cfg_norm_pre_case2", apply_cfg_norm, {"strength": 0.6, "pre_cfg": True}, 7.5)

    case("tcfg_case1", apply_tcfg, {}, 2.0)
    case("tcfg_case2", apply_tcfg, {}, 7.5)

    case(
        "fresca_case1",
        apply_fresca,
        {"scale_low": 1.0, "scale_high": 1.25, "freq_cutoff": 3},
        2.0,
    )
    case(
        "fresca_case2",
        apply_fresca,
        {"scale_low": 0.5, "scale_high": 2.0, "freq_cutoff": 1},
        4.5,
    )
    case(
        "fresca_case3",
        apply_fresca,
        {"scale_low": 1.0, "scale_high": 1.25, "freq_cutoff": 20},
        2.0,
    )

    case(
        "apg_case1",
        apply_apg,
        {"eta": 1.0, "norm_threshold": 0.0, "momentum": 0.0},
        3.0,
    )
    case(
        "apg_case2",
        apply_apg,
        {"eta": 0.5, "norm_threshold": 2.5, "momentum": 0.0},
        3.0,
    )
    case(
        "apg_case3",
        apply_apg,
        {"eta": -0.5, "norm_threshold": 5.0, "momentum": -0.5},
        4.0,
        sigmas=descending,
    )
    case(
        "apg_case4",
        apply_apg,
        {"eta": 1.0, "norm_threshold": 2.5, "momentum": 0.75},
        3.0,
        sigmas=descending,
    )
    case(
        "apg_case5",
        apply_apg,
        {"eta": 1.0, "norm_threshold": 0.0, "momentum": 0.75},
        3.0,
        sigmas=with_reset,
    )

    case("mahiro_case1", apply_mahiro, {}, 4.0)
    case("mahiro_case2", apply_mahiro, {}, 7.0)

    case("epsilon_scaling_case1", apply_epsilon_scaling, {"scaling_factor": 1.005}, 3.0)
    case("epsilon_scaling_case2", apply_epsilon_scaling, {"scaling_factor": 0.8}, 3.0)
    case("epsilon_scaling_case3", apply_epsilon_scaling, {"scaling_factor": 1.2}, 7.5)

    # RescaleCFG reads model_sampling at patch time: CONST instances take the
    # x0-space branch, everything else the v-space conversion branch.
    case(
        "rescale_cfg_flow_case1",
        apply_rescale_cfg,
        {"multiplier": 0.7},
        4.0,
        model_sampling=FLOW_SAMPLING,
    )
    case(
        "rescale_cfg_flow_case2",
        apply_rescale_cfg,
        {"multiplier": 0.3},
        7.5,
        model_sampling=FLOW_SAMPLING,
        sigmas=(0.9, 0.35),
    )
    case(
        "rescale_cfg_flow_case3",
        apply_rescale_cfg,
        {"multiplier": 1.0},
        2.0,
        model_sampling=FLOW_SAMPLING,
    )
    case(
        "rescale_cfg_eps_case1",
        apply_rescale_cfg,
        {"multiplier": 0.7},
        4.0,
        model_sampling=EPS_SAMPLING,
    )
    case(
        "rescale_cfg_eps_case2",
        apply_rescale_cfg,
        {"multiplier": 0.5},
        7.5,
        model_sampling=EPS_SAMPLING,
        sigmas=(5.0, 0.35),
    )
    case(
        "rescale_cfg_eps_case3",
        apply_rescale_cfg,
        {"multiplier": 0.0},
        3.0,
        model_sampling=EPS_SAMPLING,
    )

    # RenormCFG collapses a batched norm comparison to one Python bool, so
    # the reference supports only batch 1 when renorm_cfg is positive; the
    # goldens keep batch 1 throughout for it.
    batch1 = (1, 4, 8, 8)
    case(
        "renorm_cfg_case1",
        apply_renorm_cfg,
        {"cfg_trunc": 100.0, "renorm_cfg": 1.0},
        4.0,
        shape=batch1,
    )
    case(
        "renorm_cfg_case2",
        apply_renorm_cfg,
        {"cfg_trunc": 100.0, "renorm_cfg": 0.0},
        4.0,
        shape=batch1,
    )
    case(
        "renorm_cfg_case3",
        apply_renorm_cfg,
        {"cfg_trunc": 0.5, "renorm_cfg": 1.0},
        4.0,
        shape=batch1,
    )
    case(
        "renorm_cfg_case4",
        apply_renorm_cfg,
        {"cfg_trunc": 100.0, "renorm_cfg": 100.0},
        4.0,
        shape=batch1,
    )
    case(
        "renorm_cfg_split_case1",
        apply_renorm_cfg,
        {"cfg_trunc": 100.0, "renorm_cfg": 1.0},
        4.0,
        shape=batch1,
        in_channels=2,
    )
    cases["renorm_cfg_split_case1"]["in_channels"] = 2

    # TemporalScoreRescaling truth-tests the batched sigma tensor and
    # broadcasts alpha over the batch axis, so the reference supports only
    # batch 1; it also reads model_sampling at sample time through
    # args["model"].current_patcher.
    case(
        "tsr_flow_case1",
        apply_tsr,
        {"tsr_k": 0.95, "tsr_sigma": 1.0},
        3.0,
        shape=batch1,
        model_sampling=FLOW_SAMPLING,
        pass_model=True,
    )
    case(
        "tsr_flow_case2",
        apply_tsr,
        {"tsr_k": 5.0, "tsr_sigma": 0.5},
        7.5,
        shape=batch1,
        model_sampling=FLOW_SAMPLING,
        pass_model=True,
        sigmas=(0.9, 0.35),
    )
    case(
        "tsr_flow_case3",
        apply_tsr,
        {"tsr_k": 1.0, "tsr_sigma": 1.0},
        3.0,
        shape=batch1,
        model_sampling=FLOW_SAMPLING,
        pass_model=True,
    )
    case(
        "tsr_eps_case1",
        apply_tsr,
        {"tsr_k": 0.95, "tsr_sigma": 1.0},
        3.0,
        shape=batch1,
        model_sampling=EPS_SAMPLING,
        pass_model=True,
    )
    case(
        "tsr_eps_case2",
        apply_tsr,
        {"tsr_k": 0.5, "tsr_sigma": 2.0},
        5.0,
        shape=batch1,
        model_sampling=EPS_SAMPLING,
        pass_model=True,
        sigmas=(5.0, 0.35),
    )
    # sigma == 0 hits the reference's no-noise early return.
    case(
        "tsr_flow_case4",
        apply_tsr,
        {"tsr_k": 0.95, "tsr_sigma": 1.0},
        3.0,
        shape=batch1,
        model_sampling=FLOW_SAMPLING,
        pass_model=True,
        sigmas=(0.0,),
    )
    # Flow sigma == 1 gives logit(1) = inf, so snr == 0 hits that early return.
    case(
        "tsr_flow_case5",
        apply_tsr,
        {"tsr_k": 0.95, "tsr_sigma": 1.0},
        3.0,
        shape=batch1,
        model_sampling=FLOW_SAMPLING,
        pass_model=True,
        sigmas=(1.0,),
    )

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
