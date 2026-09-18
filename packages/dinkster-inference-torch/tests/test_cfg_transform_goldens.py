"""Replay the executed ComfyUI CFG-transform oracle through GuidanceExecutor.

The exact float values are platform-dependent (reduction order in the norm
and variance kernels differs by ULPs across torch builds), so the fixture
goes through the platform-tuple golden loader: the canonical file is
Linux-minted and non-Linux hosts mint a suffixed fixture with the
generator (tools/gen_cfg_transform_goldens.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationRequest,
    GuidancePlanContext,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    ProgressScope,
    SamplingExecutionContext,
)
from dinkster_inference_torch import guidance_transforms
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from golden_files import load_platform_golden

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens/cfg_transform_goldens.json")


def dec(value: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(value["data"], dtype=torch.float32).reshape(value["shape"])


def contribution_for(case: dict[str, Any]) -> GuidanceContribution[torch.Tensor]:
    transform = case["transform"]
    params = cast("dict[str, Any]", case["params"])
    if transform == "cfg_zero_star":
        return guidance_transforms.cfg_zero_star()
    if transform in {"cfg_norm", "cfg_norm_pre"}:
        return guidance_transforms.cfg_norm(
            strength=params["strength"], pre_cfg=params.get("pre_cfg", False)
        )
    if transform == "tcfg":
        return guidance_transforms.tcfg()
    if transform == "fresca":
        return guidance_transforms.fresca(**params)
    if transform == "apg":
        return guidance_transforms.apg(**params)
    if transform == "mahiro":
        return guidance_transforms.mahiro()
    if transform == "epsilon_scaling":
        return guidance_transforms.epsilon_scaling(**params)
    if transform in {"rescale_cfg_flow", "rescale_cfg_eps"}:
        return guidance_transforms.rescale_cfg(
            params["multiplier"], flow=transform == "rescale_cfg_flow"
        )
    if transform in {"renorm_cfg", "renorm_cfg_split"}:
        return guidance_transforms.renorm_cfg(
            params["cfg_trunc"],
            params["renorm_cfg"],
            in_channels=cast("int | None", case.get("in_channels")),
        )
    if transform in {"tsr_flow", "tsr_eps"}:
        return guidance_transforms.temporal_score_rescaling(
            params["tsr_k"], params["tsr_sigma"], flow=transform == "tsr_flow"
        )
    raise AssertionError(f"unknown transform {transform!r}")


@pytest.mark.parametrize("name", sorted(GOLDENS["cases"]))
def test_cfg_transform_matches_executed_comfy_reference(name: str) -> None:
    case = GOLDENS["cases"][name]
    sigmas = [float(sigma) for sigma in case["sigmas"]]
    x = dec(case["input"])
    registry = GuidanceRegistry((("golden", contribution_for(case)),))
    executor = GuidanceExecutor(registry)
    conditions = (
        GuidanceCondition("c", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0), None)),
        GuidanceCondition("u", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0), None)),
    )
    token = CancellationToken(lambda: False)
    base = SamplingExecutionContext(
        (*sigmas, 0.0),
        0,
        0,
        sigmas[0],
        1,
        token,
        ProgressScope(token),
        {guidance_transforms.APG_STATE_NAMESPACE: {}},
    )

    for step, sigma in enumerate(sigmas):
        execution = SamplingExecutionContext(
            (*sigmas, 0.0),
            step,
            step,
            sigma,
            1,
            token,
            ProgressScope(token),
            base.extension_state,
        )
        context = GuidancePlanContext(
            x,
            torch.tensor([sigma] * x.shape[0], dtype=torch.float32),
            case["cfg_scale"],
            conditions,
            False,
            execution,
        )
        values = {"c": dec(case["conds"][step]), "u": dec(case["unconds"][step])}

        def evaluate(
            request: GuidanceEvaluationRequest[torch.Tensor],
            values: dict[str, torch.Tensor] = values,
        ) -> GuidancePredictions[torch.Tensor]:
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(lane.id, values[lane.id], GuidancePredictionSource.MODEL)
                    for lane in request.plan.lanes
                )
            )

        result = executor.execute(context, evaluate)
        torch.testing.assert_close(result.denoised, dec(case["outputs"][step]), rtol=0, atol=0)
