"""Replay the executed ComfyUI guidance oracle through GuidanceExecutor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationRequest,
    GuidancePlanContext,
    GuidancePostCFGContext,
    GuidancePostCFGDescriptor,
    GuidancePreCFGContext,
    GuidancePreCFGDescriptor,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceReduceContext,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    ProgressScope,
    SamplingExecutionContext,
)
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from dinkster_protocol import GuidancePhaseParticipation

GOLDENS = cast(
    "dict[str, Any]",
    json.loads((Path(__file__).parent / "goldens/guidance_goldens.json").read_text()),
)


def dec(value: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(value["data"], dtype=torch.float32).reshape(value["shape"])


def execution() -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((0.625, 0.0), 0, 0, 0.625, 1, token, ProgressScope(token), {})


@pytest.mark.parametrize("name", sorted(GOLDENS["cases"]))
def test_guidance_matches_executed_comfy_reference(name: str) -> None:
    case = GOLDENS["cases"][name]
    calls: list[str] = []
    contribution = None
    mode = case["mode"]

    def plan(context: GuidancePlanContext[torch.Tensor]) -> GuidanceEvaluationPlan[torch.Tensor]:
        return GuidanceEvaluationPlan(context.conditions, "c", "u")

    if mode in {"compose", "bypass"}:

        def pre_a(
            context: GuidancePreCFGContext[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            calls.append("pre_a")
            items = context.predictions.items
            return GuidancePredictions(
                (
                    GuidancePrediction("c", items[0].value + 0.25, items[0].source),
                    GuidancePrediction("u", items[1].value - 0.5, items[1].source),
                )
            )

        def pre_b(
            context: GuidancePreCFGContext[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            calls.append("pre_b")
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(item.lane_id, item.value * 1.5, item.source)
                    for item in context.predictions.items
                )
            )

        def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
            calls.append("reducer")
            return (
                context.predictions.items[0].value * 0.75
                + context.predictions.items[1].value * 0.25
            )

        def post_a(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
            calls.append("post_a")
            return context.reduced + 0.125

        def post_b(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
            calls.append("post_b")
            return context.reduced * 0.8

        participation = (
            GuidancePhaseParticipation.COMPOSE
            if mode == "compose"
            else GuidancePhaseParticipation.BYPASS_TRANSFORMS
        )
        contribution = GuidanceContribution(
            pre_cfg=(
                GuidancePreCFGDescriptor("golden.pre_a", pre_a, order=1),
                GuidancePreCFGDescriptor("golden.pre_b", pre_b, order=2),
            ),
            strategy=GuidanceStrategyDescriptor(
                "golden.reducer", plan, reduce, participation=participation
            ),
            post_cfg=(
                GuidancePostCFGDescriptor("golden.post_a", post_a, order=1),
                GuidancePostCFGDescriptor("golden.post_b", post_b, order=2),
            ),
        )
    elif mode == "rescale":

        def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
            calls.append("rescale")
            cond, uncond = (item.value for item in context.predictions.items)
            cond_noise = context.request.input - cond
            uncond_noise = context.request.input - uncond
            combined = uncond_noise + (cond_noise - uncond_noise) * context.cfg_scale
            dims = tuple(range(1, combined.ndim))
            ratio = cond_noise.std(dim=dims, keepdim=True) / combined.std(dim=dims, keepdim=True)
            return context.request.input - (combined * ratio * 0.7 + combined * 0.3)

        contribution = GuidanceContribution(
            strategy=GuidanceStrategyDescriptor("golden.rescale", plan, reduce)
        )

    conditions = (
        GuidanceCondition("c", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0), None)),
        GuidanceCondition("u", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0), None)),
    )
    context = __import__("dinkster_inference").GuidancePlanContext(
        dec(case["input"]),
        torch.tensor(0.625),
        case["cfg_scale"],
        conditions,
        case["force_uncond"]
        or case["cfg_scale"] != 1.0
        or mode in {"compose", "bypass", "rescale"},
        execution(),
    )

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        calls.append("evaluate")
        values = {"c": dec(case["cond"]), "u": dec(case["uncond"])}
        return GuidancePredictions(
            tuple(
                GuidancePrediction(lane.id, values[lane.id], GuidancePredictionSource.MODEL)
                for lane in request.plan.lanes
            )
        )

    registry = (
        GuidanceRegistry()
        if contribution is None
        else GuidanceRegistry((("golden", contribution),))
    )
    result = GuidanceExecutor(registry).execute(context, evaluate)
    assert calls == case["calls"]
    torch.testing.assert_close(result.denoised, dec(case["output"]), rtol=0, atol=0)
