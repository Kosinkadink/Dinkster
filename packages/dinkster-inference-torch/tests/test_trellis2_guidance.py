from __future__ import annotations

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationRequest,
    GuidancePlanContext,
    GuidancePostCFGDescriptor,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    ProgressScope,
    SamplingExecutionContext,
    cfg_combine,
)
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from dinkster_inference_torch.guidance_transforms import (
    cfg_override,
    rescale_cfg,
    trellis2_rescale_cfg,
)


def _execution(sigma: float) -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((1.0, 0.0), 0, 0, sigma, 1083, token, ProgressScope(token), {})


def _conditions() -> tuple[GuidanceCondition[torch.Tensor], ...]:
    conditioning = Conditioning(torch.empty(0), None)
    return (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, conditioning),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, conditioning),
    )


def _run(
    x_t: torch.Tensor,
    sigma: float,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    base_cfg: float,
    override_cfg: float,
    interval: tuple[float, float],
    rescale: float | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    contributions = [("override", cfg_override(override_cfg, *interval))]
    if rescale is not None:
        contributions.append(("rescale", trellis2_rescale_cfg(rescale)))
    executor = GuidanceExecutor(GuidanceRegistry(tuple(contributions)))
    context = GuidancePlanContext(
        x_t,
        torch.tensor(sigma, dtype=x_t.dtype),
        base_cfg,
        _conditions(),
        False,
        _execution(sigma),
    )
    evaluated: tuple[str, ...] = ()

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        nonlocal evaluated
        evaluated = tuple(lane.id for lane in request.plan.lanes)
        velocities = {"positive": positive, "negative": negative}
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    x_t - sigma * velocities[lane.id],
                    GuidancePredictionSource.MODEL,
                )
                for lane in request.plan.lanes
            )
        )

    return executor.execute(context, evaluate).denoised, evaluated


def _source(
    x_t: torch.Tensor,
    sigma: float,
    positive: torch.Tensor,
    negative: torch.Tensor,
    strength: float,
    rescale: float,
) -> torch.Tensor:
    if strength == 1.0:
        prediction = positive
    elif strength == 0.0:
        prediction = negative
    else:
        prediction = strength * positive + (1.0 - strength) * negative
        if rescale > 0.0:
            sigma_min = 1e-5
            source_sigma = sigma_min + (1.0 - sigma_min) * sigma
            source_origin = (1.0 - sigma_min) * x_t
            x_0_pos = source_origin - source_sigma * positive
            x_0_cfg = source_origin - source_sigma * prediction
            dims = tuple(range(1, x_0_pos.ndim))
            x_0_rescaled = x_0_cfg * (
                x_0_pos.std(dim=dims, keepdim=True) / x_0_cfg.std(dim=dims, keepdim=True)
            )
            x_0 = rescale * x_0_rescaled + (1.0 - rescale) * x_0_cfg
            prediction = (source_origin - x_0) / source_sigma
    return x_t - sigma * prediction


@pytest.mark.parametrize(
    ("sigma", "strength", "lanes"),
    (
        (0.59, 1.0, ("positive",)),
        (0.6, 7.5, ("positive", "negative")),
        (0.8, 7.5, ("positive", "negative")),
        (1.0, 7.5, ("positive", "negative")),
        (1.01, 1.0, ("positive",)),
    ),
)
def test_trellis2_guidance_matches_source_interval_and_rescale(
    sigma: float, strength: float, lanes: tuple[str, ...]
) -> None:
    x_t = torch.tensor(
        [[[[0.5, -0.25], [1.5, -2.0]], [[0.75, 2.0], [-1.0, 0.125]]]],
        dtype=torch.float64,
    )
    positive = torch.tensor(
        [[[[1.0, -1.0], [0.5, 2.0]], [[-0.5, 1.5], [0.25, -2.0]]]],
        dtype=torch.float64,
    )
    negative = torch.tensor(
        [[[[0.0, 0.5], [-1.0, 1.0]], [[1.0, -0.5], [2.0, 0.75]]]],
        dtype=torch.float64,
    )

    actual, evaluated = _run(
        x_t,
        sigma,
        positive,
        negative,
        base_cfg=1.0,
        override_cfg=7.5,
        interval=(0.6, 1.0),
        rescale=0.7,
    )
    expected = _source(x_t, sigma, positive, negative, strength, 0.7)

    assert evaluated == lanes
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-14)


@pytest.mark.parametrize(
    ("strength", "lanes"),
    (
        (0.0, ("negative",)),
        (1.0, ("positive",)),
        (1.0 + 1e-10, ("positive", "negative")),
    ),
)
def test_cfg_override_uses_exact_source_branches(strength: float, lanes: tuple[str, ...]) -> None:
    x_t = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    positive = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    negative = torch.tensor([[-4.0, 1.0]], dtype=torch.float64)
    actual, evaluated = _run(
        x_t,
        0.8,
        positive,
        negative,
        base_cfg=7.5,
        override_cfg=strength,
        interval=(0.0, 1.0),
    )

    assert evaluated == lanes
    torch.testing.assert_close(
        actual,
        _source(x_t, 0.8, positive, negative, strength, 0.0),
        rtol=0.0,
        atol=2e-15,
    )


def test_cfg_override_preserves_comfy_cfg_operation_order() -> None:
    dtype = torch.float32
    actual, _ = _run(
        torch.tensor([0.0], dtype=dtype),
        1.0,
        torch.tensor([-0.1], dtype=dtype),
        torch.tensor([-0.2], dtype=dtype),
        base_cfg=7.0,
        override_cfg=3.0,
        interval=(0.0, 1.0),
    )
    expected = cfg_combine(torch.tensor([0.1], dtype=dtype), torch.tensor([0.2], dtype=dtype), 3.0)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("sigma", "lanes"),
    ((0.59, ("positive",)), (0.6, ("positive",)), (0.61, ("positive", "negative"))),
)
def test_cfg_override_matches_official_comfy_reverse_interval(
    sigma: float, lanes: tuple[str, ...]
) -> None:
    value = torch.zeros((1, 2), dtype=torch.float64)
    _, evaluated = _run(
        value,
        sigma,
        value,
        value,
        base_cfg=7.5,
        override_cfg=1.0,
        interval=(0.0, 0.6),
    )
    assert evaluated == lanes


def test_cfg_override_uses_full_precision_execution_sigma() -> None:
    contribution = cfg_override(7.5, 0.60005, 0.60015)
    executor = GuidanceExecutor(GuidanceRegistry((("override", contribution),)))
    context = GuidancePlanContext(
        torch.zeros((1, 2), dtype=torch.bfloat16),
        torch.tensor(0.6001, dtype=torch.bfloat16),
        1.0,
        _conditions(),
        False,
        _execution(0.6001),
    )
    evaluated: tuple[str, ...] = ()

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        nonlocal evaluated
        evaluated = tuple(lane.id for lane in request.plan.lanes)
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    torch.zeros_like(request.input),
                    GuidancePredictionSource.MODEL,
                )
                for lane in request.plan.lanes
            )
        )

    executor.execute(context, evaluate)
    assert not 0.60005 <= float(context.sigma) <= 0.60015
    assert evaluated == ("positive", "negative")


def test_trellis2_source_rescale_preserves_zero_variance_semantics() -> None:
    x_t = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    actual, _ = _run(
        x_t,
        0.8,
        torch.ones_like(x_t),
        torch.full_like(x_t, 2.0),
        base_cfg=1.0,
        override_cfg=7.5,
        interval=(0.6, 1.0),
        rescale=0.7,
    )
    assert torch.isnan(actual).all()


def test_cfg_override_refuses_unproven_post_transform_composition() -> None:
    post = GuidanceContribution(
        post_cfg=(GuidancePostCFGDescriptor("test.scale", lambda context: context.reduced),)
    )
    with pytest.raises(RuntimeError, match="strategy cannot compose"):
        GuidanceRegistry((("override", cfg_override(7.5, 0.6, 1.0)), ("post", post)))


@pytest.mark.parametrize("strength", (0.0, 7.5))
def test_cfg_override_refuses_missing_real_unconditional_conditioning(strength: float) -> None:
    value = torch.zeros((1, 2), dtype=torch.float64)
    conditions = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0))),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, None),
    )
    context = GuidancePlanContext(
        value,
        torch.tensor(0.8, dtype=value.dtype),
        1.0,
        conditions,
        False,
        _execution(0.8),
    )
    executor = GuidanceExecutor(GuidanceRegistry((("override", cfg_override(strength, 0.0, 1.0)),)))
    with pytest.raises(RuntimeError, match="requires unconditional conditioning"):
        executor.execute(context, lambda _request: pytest.fail("must refuse before evaluation"))


def test_cfg_override_accepts_at_most_one_proven_rescaler() -> None:
    with pytest.raises(RuntimeError, match="at most one compatible post-CFG"):
        GuidanceRegistry(
            (
                ("override", cfg_override(7.5, 0.6, 1.0)),
                ("standard", rescale_cfg(0.7, flow=True)),
                ("trellis", trellis2_rescale_cfg(0.7)),
            )
        )


def test_trellis2_source_rescale_refuses_terminal_sigma() -> None:
    value = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="positive executed sigma"):
        _run(
            value,
            0.0,
            torch.tensor([[2.0, 3.0]], dtype=value.dtype),
            torch.tensor([[-4.0, 1.0]], dtype=value.dtype),
            base_cfg=7.5,
            override_cfg=7.5,
            interval=(0.0, 1.0),
            rescale=0.7,
        )
