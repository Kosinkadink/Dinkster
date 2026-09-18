"""Replay the executed ComfyUI Perp-Neg guider oracle through GuidanceExecutor.

The exact float values are platform-dependent (reduction order in the norm
and dot-product kernels differs by ULPs across torch builds and CPU
microarchitectures), so the fixture goes through the platform-tuple golden
loader with the mint host's CPU pinned: hosts whose CPU differs skip the
module instead of enforcing bit equality (generator:
tools/gen_perp_neg_goldens.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens/perp_neg_goldens.json")


def dec(value: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(value["data"], dtype=torch.float32).reshape(value["shape"])


@pytest.mark.parametrize("name", sorted(GOLDENS["cases"]))
def test_perp_neg_matches_executed_comfy_reference(name: str) -> None:
    case = GOLDENS["cases"][name]
    sigmas = [float(sigma) for sigma in case["sigmas"]]
    x = dec(case["input"])
    contributions: list[tuple[str, GuidanceContribution[torch.Tensor]]] = [
        ("golden.perp-neg", guidance_transforms.perp_neg(case["params"]["neg_scale"]))
    ]
    chain = case.get("chain")
    if chain is not None:
        factories = {
            "epsilon_scaling": guidance_transforms.epsilon_scaling,
            "tcfg": guidance_transforms.tcfg,
        }
        contributions.append(("golden.chain", factories[chain["transform"]](**chain["params"])))
    executor = GuidanceExecutor(GuidanceRegistry(tuple(contributions)))
    conditions = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0), None)),
        GuidanceCondition(
            "negative", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0), None)
        ),
        GuidanceCondition("empty", GuidanceRole.AUXILIARY, Conditioning(torch.empty(0), None)),
    )
    token = CancellationToken(lambda: False)

    for step, sigma in enumerate(sigmas):
        execution = SamplingExecutionContext(
            (*sigmas, 0.0),
            step,
            step,
            sigma,
            1,
            token,
            ProgressScope(token),
            {},
        )
        context = GuidancePlanContext(
            x,
            torch.tensor([sigma] * x.shape[0], dtype=torch.float32),
            case["cfg_scale"],
            conditions,
            False,
            execution,
        )
        values = {
            "positive": dec(case["positives"][step]),
            "negative": dec(case["negatives"][step]),
            "empty": dec(case["empties"][step]),
        }

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
