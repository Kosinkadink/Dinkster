from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    GuidanceCondition,
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

REPO = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = Path(os.environ.get("DINKSTER_EVIDENCE_ROOT", REPO.parent / "dinkster-evidence"))
RECEIPT = json.loads(
    (
        EVIDENCE_ROOT
        / "docs/comfy-confidence-receipts/comfyui-inspire-pack/scheduled-cfg-guider.receipt.json"
    ).read_text(encoding="utf-8")
)
GOLDEN = json.loads(
    (
        EVIDENCE_ROOT
        / "docs/comfy-confidence-receipts/artifacts/comfyui-inspire-pack"
        / "scheduled-cfg-guider.native.json"
    ).read_text(encoding="utf-8")
)


def _array_value(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "data": value.reshape(-1).tolist(),
    }


def _recording_scale(
    transform: Callable[[GuidancePlanContext[torch.Tensor]], float],
    scales: list[float],
) -> Callable[[GuidancePlanContext[torch.Tensor]], float]:
    def record(context: GuidancePlanContext[torch.Tensor]) -> float:
        scale = transform(context)
        scales.append(scale)
        return scale

    return record


def _case_evaluator(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    scales: list[float],
    trace: list[dict[str, object]],
) -> Callable[[GuidanceEvaluationRequest[torch.Tensor]], GuidancePredictions[torch.Tensor]]:
    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        trace.append({"cfg": scales[-1], "lanes": [lane.id for lane in request.plan.lanes]})
        predictions = {"positive": cond, "negative": uncond}
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    predictions[lane.id],
                    GuidancePredictionSource.MODEL,
                )
                for lane in request.plan.lanes
            )
        )

    return evaluate


@pytest.mark.parametrize("family", ("sd-eps", "flux-flow"))
def test_scheduled_cfg_matches_inspire_for_sd_and_flow(family: str) -> None:
    expected = {case["id"]: case for case in GOLDEN["cases"]}
    cases = [case for case in RECEIPT["parameters"]["cases"] if case["family"] == family]
    channels = {"sd-eps": 4, "flux-flow": 16}[family]
    values = torch.arange(channels * 4, dtype=torch.float32).reshape(1, channels, 2, 2)
    cond = values / 11.0
    uncond = -values / 7.0 - 0.25

    for case in cases:
        sigma_tensor = torch.tensor(case["sigmas"], dtype=torch.float32)
        sigmas = tuple(float(value) for value in sigma_tensor.tolist())
        contribution = guidance_transforms.scheduled_cfg(
            sigmas,
            case["from_cfg"],
            case["to_cfg"],
            case["schedule"],
        )
        scales: list[float] = []
        descriptor = contribution.scale[0]
        contribution = replace(
            contribution,
            scale=(replace(descriptor, transform=_recording_scale(descriptor.transform, scales)),),
        )
        executor = GuidanceExecutor(GuidanceRegistry((("scheduled-cfg-golden", contribution),)))
        conditions = (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0))),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0))),
        )
        token = CancellationToken(lambda: False)
        trace: list[dict[str, object]] = []
        outputs = []
        evaluate = _case_evaluator(cond, uncond, scales, trace)
        admission_cfg = (
            case["from_cfg"] if not math.isclose(case["from_cfg"], 1.0) else case["to_cfg"]
        )
        for index, sigma in enumerate(torch.tensor(case["evaluations"], dtype=torch.float32)):
            execution = SamplingExecutionContext(
                sigmas,
                index,
                index,
                float(sigma),
                1,
                token,
                ProgressScope(token),
                {},
            )
            context = GuidancePlanContext(
                cond,
                sigma.reshape(1),
                admission_cfg,
                conditions,
                False,
                execution,
            )
            outputs.append(executor.execute(context, evaluate).denoised)

        actual = {
            "id": case["id"],
            "trace": trace,
            "outputs": [_array_value(cast("torch.Tensor", output)) for output in outputs],
        }
        assert actual == expected[case["id"]]
