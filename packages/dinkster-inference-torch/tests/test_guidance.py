# pyright: basic
"""Focused proofs for the torch guidance phase executor."""

from __future__ import annotations

import ast
import weakref
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    LTXV,
    AttentionGuidanceDescriptor,
    CancellationToken,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    ConditionScaleVector,
    DualSamplingGuidance,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationWrapperDescriptor,
    GuidanceExtensionError,
    GuidancePlanAugmentationDescriptor,
    GuidancePostCFGDescriptor,
    GuidancePreCFGDescriptor,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    GuidanceScaleDescriptor,
    GuidanceStrategyDescriptor,
    ModelTokenLayout,
    ModelTokenSegment,
    ProgressScope,
    SamplingCancelled,
    SamplingExecutionContext,
    TokenGridTransform,
    TokenLayoutError,
)
from dinkster_inference_torch import distributed
from dinkster_inference_torch.guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
    ConditioningPlanCompiler,
    ConditioningValidationPath,
    GuidanceExecutor,
    GuidanceRegistry,
    GuidedDenoiser,
    dual_cfg_executor,
    evaluate_conditioning_batch,
)
from dinkster_protocol import GuidancePhaseParticipation


def execution(cancelled: Callable[[], bool] = lambda: False) -> SamplingExecutionContext:  # noqa: B008
    token = CancellationToken(cancelled)
    return SamplingExecutionContext((1.0, 0.0), 0, 0, 1.0, 7, token, ProgressScope(token), {})


DEFAULT_CONDITIONING = Conditioning(torch.empty(0), None)


def test_conditioning_batch_engine_owns_lane_stacking_and_flow_conversion() -> None:
    class Adapter:
        compute_dtype = torch.float64

        def _validate_conditioning_batch(
            self, x: torch.Tensor, conditions: tuple[int, ...]
        ) -> None:
            assert x.shape == (2, 3)
            assert conditions == (5, 8)

        def _evaluate_conditioning_model(self, batch: ConditioningBatch[int]) -> torch.Tensor:
            assert batch.model_input.shape == (4, 3)
            assert batch.model_input.dtype == torch.float64
            assert torch.equal(batch.timestep, torch.full((4,), 0.25))
            return torch.zeros_like(batch.model_input, dtype=torch.float32)

    latent = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    first, second = evaluate_conditioning_batch(Adapter(), latent, 0.25, (5, 8))

    assert torch.equal(first, latent)
    assert torch.equal(second, latent)


def test_conditioning_batch_engine_rejects_wrong_adapter_cardinality() -> None:
    class Adapter:
        def _evaluate_conditioning_batch(
            self,
            x: torch.Tensor,
            sigma: float,
            conditions: tuple[int, ...],
        ) -> tuple[torch.Tensor, ...]:
            del sigma, conditions
            return (x,)

    with pytest.raises(GuidanceContractError, match="one tensor per condition"):
        evaluate_conditioning_batch(Adapter(), torch.zeros(2, 3), 0.5, (1, 2))


def test_conditioning_batch_engine_rejects_incomplete_adapter_contract() -> None:
    with pytest.raises(GuidanceContractError, match="must validate conditions"):
        evaluate_conditioning_batch(object(), torch.zeros(2, 3), 0.5, (1, 2))


def test_conditioning_batch_overrides_delegate_to_engine_evaluators() -> None:
    source_root = Path(__file__).parents[1] / "src" / "dinkster_inference_torch"
    overrides: list[tuple[str, str]] = []
    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == "_evaluate_conditioning_batch"
            ):
                body = ast.get_source_segment(source, node)
                assert body is not None
                overrides.append((path.name, body))

    assert {path for path, _body in overrides} == {
        "flux_window.py",
        "flux_window_distributed.py",
    }
    assert all("evaluate_conditioning_batch(" in body for _path, body in overrides)


def lane(
    name: str,
    role: GuidanceRole,
    conditioning: Conditioning[torch.Tensor] | None = DEFAULT_CONDITIONING,
) -> GuidanceCondition[torch.Tensor]:
    return GuidanceCondition(name, role, conditioning)


def context(*lanes: Any, cfg: float = 2.0, force: bool = False, cancelled=lambda: False):  # noqa: B008
    return __import__("dinkster_inference").GuidancePlanContext(
        torch.zeros(1, 2), torch.tensor(1.0), cfg, lanes, force, execution(cancelled)
    )


def predictions(request, values: tuple[float, ...] | None = None):
    values = values or tuple(float(i + 1) for i in range(len(request.plan.lanes)))
    return GuidancePredictions(
        tuple(
            GuidancePrediction(
                item.id,
                torch.zeros_like(request.input)
                if item.conditioning is None
                else torch.full_like(request.input, value),
                GuidancePredictionSource.MODEL
                if item.conditioning is not None
                else GuidancePredictionSource.SYNTHETIC_ZERO,
            )
            for item, value in zip(request.plan.lanes, values, strict=True)
        )
    )


def test_wrapper_pre_reducer_post_order_and_unwind() -> None:
    calls: list[str] = []

    def wrapper(name):
        def call(request, next):
            calls.append(f"{name}-enter")
            result = next(request)
            calls.append(f"{name}-exit")
            return result

        return call

    def pre(name):
        def call(context):
            calls.append(name)
            return context.predictions

        return call

    def post(name):
        def call(context):
            calls.append(name)
            return context.reduced

        return call

    contribution = GuidanceContribution(
        evaluation_wrappers=(
            GuidanceEvaluationWrapperDescriptor("x.b", wrapper("B"), order=2),
            GuidanceEvaluationWrapperDescriptor("x.a", wrapper("A"), order=1),
        ),
        pre_cfg=(
            GuidancePreCFGDescriptor("x.pb", pre("pre-B"), order=2),
            GuidancePreCFGDescriptor("x.pa", pre("pre-A"), order=1),
        ),
        post_cfg=(
            GuidancePostCFGDescriptor("x.qb", post("post-B"), order=2),
            GuidancePostCFGDescriptor("x.qa", post("post-A"), order=1),
        ),
    )
    result = GuidanceExecutor(GuidanceRegistry((("ext", contribution),))).execute(
        context(lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)),
        lambda request: calls.append("core") or predictions(request, (3.0, 1.0)),
    )
    assert calls == [
        "A-enter",
        "B-enter",
        "core",
        "B-exit",
        "A-exit",
        "pre-A",
        "pre-B",
        "post-A",
        "post-B",
    ]
    assert torch.equal(result.denoised, torch.full((1, 2), 5.0))


def test_scale_transforms_run_in_order_before_every_downstream_phase() -> None:
    calls: list[tuple[str, float]] = []
    c, u = lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)

    def record(name: str, scale: float) -> None:
        calls.append((name, scale))

    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: (
            record("plan", context.cfg_scale)
            or GuidanceEvaluationPlan(context.conditions, "c", "u")
        ),
        lambda context: record("reduce", context.cfg_scale) or context.predictions.items[0].value,
        participation=GuidancePhaseParticipation.COMPOSE,
    )
    contribution = GuidanceContribution(
        scale=(
            GuidanceScaleDescriptor(
                "x.second", lambda context: record("scale-B", context.cfg_scale) or 6.0, order=2
            ),
            GuidanceScaleDescriptor(
                "x.first", lambda context: record("scale-A", context.cfg_scale) or 4.0, order=1
            ),
        ),
        pre_cfg=(
            GuidancePreCFGDescriptor(
                "x.pre",
                lambda context: record("pre", context.cfg_scale) or context.predictions,
            ),
        ),
        strategy=strategy,
        post_cfg=(
            GuidancePostCFGDescriptor(
                "x.post", lambda context: record("post", context.cfg_scale) or context.reduced
            ),
        ),
    )

    GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
        context(c, u, cfg=2.0),
        lambda request: record("evaluate", 6.0) or predictions(request),
    )

    assert calls == [
        ("scale-A", 2.0),
        ("scale-B", 4.0),
        ("plan", 6.0),
        ("evaluate", 6.0),
        ("pre", 6.0),
        ("reduce", 6.0),
        ("post", 6.0),
    ]


def test_scale_transform_refuses_a_non_float_result() -> None:
    contribution = GuidanceContribution(
        scale=(GuidanceScaleDescriptor("x.bad", lambda context: 2),)
    )

    with pytest.raises(GuidanceContractError, match="returned a non-float scale"):
        GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )


def test_plan_augmentation_appends_one_model_evaluation_before_reduction() -> None:
    extra = lane("extra", GuidanceRole.AUXILIARY)

    def augment(context, plan):
        del context
        return replace(plan, lanes=(*plan.lanes, extra))

    def post(context):
        values = {item.lane_id: item.value for item in context.predictions.items}
        return context.reduced + values["extra"]

    contribution = GuidanceContribution(
        plan_augmentations=(GuidancePlanAugmentationDescriptor("x.extra", augment),),
        post_cfg=(GuidancePostCFGDescriptor("x.apply-extra", post),),
    )
    result = GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
        context(lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)),
        lambda request: predictions(request, (3.0, 1.0, 7.0)),
    )

    assert [item.lane_id for item in result.predictions.items] == ["c", "u", "extra"]
    assert torch.equal(result.denoised, torch.full((1, 2), 12.0))


@pytest.mark.parametrize("case", ("replace", "duplicate", "conditional", "synthetic"))
def test_plan_augmentation_is_append_only_auxiliary_evaluation(case: str) -> None:
    def augment(context, plan):
        del context
        if case == "replace":
            return replace(plan, lanes=(replace(plan.lanes[0]), *plan.lanes[1:]))
        role = GuidanceRole.CONDITIONAL if case == "conditional" else GuidanceRole.AUXILIARY
        conditioning = None if case == "synthetic" else DEFAULT_CONDITIONING
        name = plan.lanes[0].id if case == "duplicate" else "extra"
        return replace(plan, lanes=(*plan.lanes, lane(name, role, conditioning)))

    contribution = GuidanceContribution(
        plan_augmentations=(GuidancePlanAugmentationDescriptor("x.bad", augment),)
    )
    with pytest.raises(
        (GuidanceContractError, GuidanceExtensionError),
        match="preserve|duplicate|auxiliary|lane ids must be unique",
    ):
        GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )


def test_plan_augmentation_refuses_a_strategy_that_bypasses_transforms() -> None:
    extra = GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor("x.extra", lambda context, plan: plan),
        )
    )
    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: GuidanceEvaluationPlan(context.conditions, "c", None),
        lambda context: context.predictions.items[0].value,
    )

    with pytest.raises(GuidanceContractError, match="bypasses transforms"):
        GuidanceRegistry(
            (
                ("strategy", GuidanceContribution(strategy=strategy)),
                ("augmentation", extra),
            )
        )


@pytest.mark.parametrize("phase", ("wrapper", "pre"))
def test_plan_augmentation_refuses_evaluation_wrappers_and_pre_cfg_transforms(
    phase: str,
) -> None:
    extra = GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor("x.extra", lambda context, plan: plan),
        )
    )
    if phase == "wrapper":
        paired = GuidanceContribution(
            evaluation_wrappers=(
                GuidanceEvaluationWrapperDescriptor(
                    "x.wrapper", lambda request, next: next(request)
                ),
            )
        )
    else:
        paired = GuidanceContribution(
            pre_cfg=(GuidancePreCFGDescriptor("x.pre", lambda context: context.predictions),)
        )

    with pytest.raises(GuidanceContractError, match="cannot compose"):
        GuidanceRegistry((("augmentation", extra), (phase, paired)))


def test_execute_releases_core_evaluator_without_cyclic_collection() -> None:
    class Evaluator:
        def __call__(self, request):
            return predictions(request)

    evaluator = Evaluator()
    reference = weakref.ref(evaluator)
    GuidanceExecutor(GuidanceRegistry()).execute(
        context(lane("c", GuidanceRole.CONDITIONAL)), evaluator
    )

    del evaluator

    assert reference() is None


@pytest.mark.parametrize(
    "participation,transforms",
    [
        (GuidancePhaseParticipation.COMPOSE, True),
        (GuidancePhaseParticipation.BYPASS_TRANSFORMS, False),
    ],
)
def test_strategy_compose_or_bypass_and_exactly_one_reducer(participation, transforms) -> None:
    calls: list[str] = []
    c, u = lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)
    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: GuidanceEvaluationPlan(context.conditions, "c", "u"),
        lambda context: calls.append("reduce") or context.predictions.items[0].value,
        participation=participation,
    )
    contribution = GuidanceContribution(
        pre_cfg=(
            GuidancePreCFGDescriptor(
                "x.pre", lambda context: calls.append("pre") or context.predictions
            ),
        ),
        strategy=strategy,
        post_cfg=(
            GuidancePostCFGDescriptor(
                "x.post", lambda context: calls.append("post") or context.reduced
            ),
        ),
    )
    GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
        context(c, u), predictions
    )
    assert calls == (["pre", "reduce", "post"] if transforms else ["reduce"])


@pytest.mark.parametrize(
    ("nested", "expected", "expected_uncond"),
    ((False, 35.0, 2.0), (True, 83.0, 5.0)),
)
def test_dual_cfg_executor_matches_comfy_three_lane_formula(
    nested: bool, expected: float, expected_uncond: float
) -> None:
    guidance = DualSamplingGuidance(DEFAULT_CONDITIONING, DEFAULT_CONDITIONING, 4.0, 3.0, nested)
    executor = dual_cfg_executor(cast("Any", guidance))
    lanes = (
        lane("positive", GuidanceRole.CONDITIONAL),
        lane(
            "middle",
            GuidanceRole.UNCONDITIONAL if nested else GuidanceRole.AUXILIARY,
        ),
        lane(
            "negative",
            GuidanceRole.AUXILIARY if nested else GuidanceRole.UNCONDITIONAL,
        ),
    )

    def evaluate(request):
        values = {"negative": 2.0, "middle": 5.0, "positive": 11.0}
        return predictions(request, tuple(values[item.id] for item in request.plan.lanes))

    result = executor.execute(context(*lanes, cfg=guidance.middle_scale), evaluate)

    assert [item.lane_id for item in result.predictions.items] == [
        "negative",
        "middle",
        "positive",
    ]
    assert torch.equal(result.denoised, torch.full((1, 2), expected))
    assert result.unconditional is not None
    assert torch.equal(result.unconditional, torch.full((1, 2), expected_uncond))


def test_dual_cfg_regular_optimizes_unit_scales_but_force_uncond_runs_all_lanes() -> None:
    guidance = DualSamplingGuidance(DEFAULT_CONDITIONING, DEFAULT_CONDITIONING, 1.0, 1.0)
    executor = dual_cfg_executor(cast("Any", guidance))
    lanes = (
        lane("positive", GuidanceRole.CONDITIONAL),
        lane("middle", GuidanceRole.AUXILIARY),
        lane("negative", GuidanceRole.UNCONDITIONAL),
    )

    optimized = executor.execute(context(*lanes, cfg=1.0), predictions)
    forced = executor.execute(context(*lanes, cfg=1.0, force=True), predictions)

    assert [item.lane_id for item in optimized.predictions.items] == ["positive"]
    assert torch.equal(optimized.denoised, torch.ones((1, 2)))
    assert [item.lane_id for item in forced.predictions.items] == [
        "negative",
        "middle",
        "positive",
    ]


@pytest.mark.parametrize("count", [0, 2])
def test_wrapper_must_call_next_exactly_once(count: int) -> None:
    def wrapper(request, next):
        result = predictions(request)
        for _ in range(count):
            result = next(request)
        return result

    registry = GuidanceRegistry(
        (
            (
                "owner",
                GuidanceContribution(
                    evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("x.wrap", wrapper),)
                ),
            ),
        )
    )
    with pytest.raises(GuidanceContractError, match="did not call next|multiple times"):
        GuidanceExecutor(registry).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )


@pytest.mark.parametrize("case", ["unavailable", "role", "primary-role", "uncond-role", "scale"])
def test_plan_validation(case: str) -> None:
    c, u = lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)

    def plan(context):
        if case == "unavailable":
            return GuidanceEvaluationPlan((lane("other", GuidanceRole.CONDITIONAL),), "other", None)
        if case == "role":
            return GuidanceEvaluationPlan((replace(c, role=GuidanceRole.AUXILIARY),), "c", None)
        if case == "primary-role":
            return GuidanceEvaluationPlan((c, u), "u", None)
        if case == "uncond-role":
            return GuidanceEvaluationPlan((c, u), "c", "c")
        return GuidanceEvaluationPlan(
            (replace(c, scale_vector=ConditionScaleVector(torch.ones(1))),), "c", None
        )

    strategy = GuidanceStrategyDescriptor(
        "x.strategy", plan, lambda context: context.predictions.items[0].value
    )
    with pytest.raises(GuidanceContractError):
        GuidanceExecutor(
            GuidanceRegistry((("owner", GuidanceContribution(strategy=strategy)),))
        ).execute(context(c, u), predictions)


@pytest.mark.parametrize("case", ["lane", "shape", "dtype", "device", "source", "zero-value"])
def test_prediction_validation(case: str) -> None:
    c = lane("c", GuidanceRole.CONDITIONAL)

    def bad(request):
        value = torch.ones_like(request.input)
        lane_id, source = "c", GuidancePredictionSource.MODEL
        if case == "lane":
            lane_id = "wrong"
        if case == "shape":
            value = torch.ones(3)
        if case == "dtype":
            value = value.double()
        if case == "device":
            value = value.to("meta")
        if case == "source":
            source = GuidancePredictionSource.SYNTHETIC_ZERO
        if case == "zero-value":
            request = replace(
                request,
                plan=GuidanceEvaluationPlan(
                    (lane("c", GuidanceRole.CONDITIONAL, None),), "c", None
                ),
            )
            source = GuidancePredictionSource.SYNTHETIC_ZERO
        return GuidancePredictions((GuidancePrediction(lane_id, value, source),))

    with pytest.raises(GuidanceContractError):
        GuidanceExecutor(GuidanceRegistry()).execute(context(c), bad)


@pytest.mark.parametrize("phase", ["wrapper", "pre", "reducer", "post"])
def test_cancellation_stops_later_callbacks(phase: str) -> None:
    calls: list[str] = []
    armed = [False]

    def cancelled():
        return armed[0]

    def mark(name, result):
        calls.append(name)
        if name == phase:
            armed[0] = True
        return result

    def wrapper(request, next):
        return mark("wrapper", next(request))

    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: GuidanceEvaluationPlan(context.conditions, "c", None),
        lambda context: mark("reducer", context.predictions.items[0].value),
        participation=GuidancePhaseParticipation.COMPOSE,
    )
    contribution = GuidanceContribution(
        evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("x.wrapper", wrapper),),
        pre_cfg=(
            GuidancePreCFGDescriptor("x.pre", lambda context: mark("pre", context.predictions)),
        ),
        strategy=strategy,
        post_cfg=(
            GuidancePostCFGDescriptor("x.post", lambda context: mark("post", context.reduced)),
        ),
    )
    with pytest.raises(SamplingCancelled):
        GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL), cancelled=cancelled), predictions
        )
    assert calls[-1] == phase


def test_sampling_cancelled_is_not_wrapped() -> None:
    def fail(context):
        raise SamplingCancelled("original")

    contribution = GuidanceContribution(pre_cfg=(GuidancePreCFGDescriptor("x.pre", fail),))
    with pytest.raises(SamplingCancelled, match="original"):
        GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )


def test_callback_error_names_owner_phase_and_preserves_cause() -> None:
    cause = ValueError("boom")

    def fail(context):
        raise cause

    contribution = GuidanceContribution(pre_cfg=(GuidancePreCFGDescriptor("x.pre", fail),))
    with pytest.raises(
        GuidanceExtensionError, match="extension=owner contribution=x.pre phase=pre"
    ) as caught:
        GuidanceExecutor(GuidanceRegistry((("owner", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )
    assert caught.value.__cause__ is cause


def test_core_failure_is_not_attributed_to_outer_wrapper() -> None:
    cause = ValueError("core boom")

    def wrapper(request, next):
        return next(request)

    contribution = GuidanceContribution(
        evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("x.outer", wrapper),)
    )

    def fail(_request):
        raise cause

    with pytest.raises(ValueError, match="core boom") as caught:
        GuidanceExecutor(GuidanceRegistry((("outer", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), fail
        )
    assert caught.value is cause


def test_inner_wrapper_failure_names_inner_only() -> None:
    def outer(request, next):
        return next(request)

    def inner(request, next):
        del request, next
        raise ValueError("inner boom")

    registry = GuidanceRegistry(
        (
            (
                "outer",
                GuidanceContribution(
                    evaluation_wrappers=(
                        GuidanceEvaluationWrapperDescriptor("x.outer", outer, order=0),
                    )
                ),
            ),
            (
                "inner",
                GuidanceContribution(
                    evaluation_wrappers=(
                        GuidanceEvaluationWrapperDescriptor("x.inner", inner, order=1),
                    )
                ),
            ),
        )
    )
    with pytest.raises(GuidanceExtensionError) as caught:
        GuidanceExecutor(registry).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )
    assert "extension=inner contribution=x.inner phase=wrapper" in str(caught.value)
    assert "extension=outer" not in str(caught.value)


def test_outer_wrapper_own_failure_names_outer() -> None:
    def outer(request, next):
        del request, next
        raise ValueError("outer boom")

    contribution = GuidanceContribution(
        evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("x.outer", outer),)
    )
    with pytest.raises(
        GuidanceExtensionError, match="extension=outer contribution=x.outer phase=wrapper"
    ):
        GuidanceExecutor(GuidanceRegistry((("outer", contribution),))).execute(
            context(lane("c", GuidanceRole.CONDITIONAL)), predictions
        )


def test_requires_uncond_cfg1_synthetic_zero_final_uncond_and_more_than_two_lanes() -> None:
    descriptor = GuidancePreCFGDescriptor(
        "x.pre", lambda context: context.predictions, requires_uncond=True
    )
    registry = GuidanceRegistry((("owner", GuidanceContribution(pre_cfg=(descriptor,))),))
    assert registry.requires_uncond
    lanes = (
        lane("c", GuidanceRole.CONDITIONAL),
        lane("aux", GuidanceRole.CONDITIONAL),
        lane("u", GuidanceRole.UNCONDITIONAL, None),
    )
    result = GuidanceExecutor(registry).execute(context(*lanes, cfg=1.0, force=True), predictions)
    assert len(result.predictions.items) == 3
    assert result.predictions.items[-1].source is GuidancePredictionSource.SYNTHETIC_ZERO
    assert result.unconditional is not None
    assert torch.count_nonzero(result.unconditional) == 0


def test_unforced_synthetic_uncond_lane_is_not_planned() -> None:
    """With no unconditional CONDITIONING and no force_uncond, the plan
    has no unconditional lane: the primary prediction is returned
    untouched at any scale, exactly like CfgDenoiser's uncond-None
    single evaluation - never synthetic-zero CFG."""
    result = GuidanceExecutor(GuidanceRegistry()).execute(
        context(
            lane("c", GuidanceRole.CONDITIONAL),
            lane("u", GuidanceRole.UNCONDITIONAL, None),
            cfg=3.0,
        ),
        predictions,
    )
    assert len(result.predictions.items) == 1
    assert result.unconditional is None
    assert torch.equal(result.denoised, torch.ones_like(result.denoised))


def test_active_contributions_refuse_a_conditioning_free_uncond_lane_when_cfg_needs_it() -> None:
    """Contributions registered around CFG combination cannot execute
    when the unconditional lane carries no CONDITIONING and the cfg
    scale would need one: the builtin plan returns the conditional
    prediction untouched, so the pairing is ambiguous and refuses
    instead of silently running on the untouched prediction. A wholly
    absent unconditional lane stays cond-only (pinned by the wrapper
    and attribution tests, which run active registries over
    conditional-only lanes at cfg 2)."""
    contribution = GuidanceContribution(
        post_cfg=(GuidancePostCFGDescriptor("x.post", lambda context: context.reduced),)
    )
    registry = GuidanceRegistry((("owner", contribution),))
    lanes = (
        lane("c", GuidanceRole.CONDITIONAL),
        lane("u", GuidanceRole.UNCONDITIONAL, None),
    )
    with pytest.raises(GuidanceContractError, match="unconditional conditioning"):
        GuidanceExecutor(registry).execute(context(*lanes, cfg=3.0), predictions)


def test_active_contributions_run_without_uncond_at_cfg_one() -> None:
    calls: list[str] = []
    contribution = GuidanceContribution(
        post_cfg=(
            GuidancePostCFGDescriptor(
                "x.post", lambda context: calls.append("post") or context.reduced
            ),
        )
    )
    registry = GuidanceRegistry((("owner", contribution),))
    result = GuidanceExecutor(registry).execute(
        context(
            lane("c", GuidanceRole.CONDITIONAL),
            lane("u", GuidanceRole.UNCONDITIONAL, None),
            cfg=1.0,
        ),
        predictions,
    )
    assert calls == ["post"]
    assert torch.equal(result.denoised, torch.ones_like(result.denoised))


def test_strategy_plans_missing_uncond_without_refusal() -> None:
    """A strategy owns planning and reduction, so the standard plan's
    missing-uncond refusal does not apply to it."""
    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: GuidanceEvaluationPlan(
            tuple(item for item in context.conditions if item.conditioning is not None),
            "c",
            None,
        ),
        lambda context: context.predictions.items[0].value,
    )
    registry = GuidanceRegistry((("owner", GuidanceContribution(strategy=strategy)),))
    result = GuidanceExecutor(registry).execute(
        context(
            lane("c", GuidanceRole.CONDITIONAL),
            lane("u", GuidanceRole.UNCONDITIONAL, None),
            cfg=3.0,
        ),
        predictions,
    )
    assert torch.equal(result.denoised, torch.ones_like(result.denoised))


def test_attention_contribution_validation() -> None:
    descriptor = AttentionGuidanceDescriptor("x.att", lambda positive, negative: positive)
    assert GuidanceContribution(attention=descriptor).attention is descriptor
    with pytest.raises(TypeError, match="AttentionGuidanceDescriptor or None"):
        GuidanceContribution(attention=cast("Any", object()))
    with pytest.raises(ValueError, match="globally unique"):
        GuidanceContribution(
            post_cfg=(GuidancePostCFGDescriptor("x.att", lambda context: context.reduced),),
            attention=descriptor,
        )
    with pytest.raises(ValueError, match="must be namespaced"):
        AttentionGuidanceDescriptor("att", lambda positive, negative: positive)
    with pytest.raises(TypeError, match="callable"):
        AttentionGuidanceDescriptor("x.att", cast("Any", object()))


def test_attention_contributions_keep_contribution_order_and_mark_the_registry() -> None:
    first = AttentionGuidanceDescriptor("b.att", lambda positive, negative: positive)
    second = AttentionGuidanceDescriptor(
        "a.att", lambda positive, negative: positive, requires_uncond=True
    )
    registry = GuidanceRegistry(
        (
            ("owner-b", GuidanceContribution(attention=first)),
            ("owner-a", GuidanceContribution(attention=second)),
        )
    )
    assert [item.value.id for item in registry.attention] == ["b.att", "a.att"]
    assert registry.active
    assert registry.requires_uncond


def test_strategy_refuses_attention_contributions() -> None:
    strategy = GuidanceStrategyDescriptor(
        "x.strategy",
        lambda context: GuidanceEvaluationPlan(
            tuple(item for item in context.conditions if item.conditioning is not None),
            "c",
            None,
        ),
        lambda context: context.predictions.items[0].value,
    )
    attention = GuidanceContribution(
        attention=AttentionGuidanceDescriptor("x.att", lambda positive, negative: positive)
    )
    with pytest.raises(GuidanceContractError, match="cannot compose with attention-kind"):
        GuidanceRegistry(
            (("s", GuidanceContribution(strategy=strategy)), ("a", attention)),
        )


def test_executor_refuses_unconsumed_attention_contributions() -> None:
    registry = GuidanceRegistry(
        (
            (
                "owner",
                GuidanceContribution(
                    attention=AttentionGuidanceDescriptor(
                        "x.att", lambda positive, negative: positive
                    )
                ),
            ),
        )
    )
    lanes = (lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL))
    with pytest.raises(GuidanceContractError, match="must be consumed"):
        GuidanceExecutor(registry).execute(context(*lanes), predictions)
    result = GuidanceExecutor(registry).execute(
        context(*lanes), predictions, attention_consumed=True
    )
    assert len(result.predictions.items) == 2
    assert result.unconditional is not None


@pytest.mark.parametrize("provider", ["flux", "sd"])
def test_guided_denoiser_executes_both_provider_shapes(provider: str) -> None:
    seen: list[str] = []

    def evaluate(x: torch.Tensor, _sigma: float, value: float) -> torch.Tensor:
        seen.append(provider)
        return torch.full_like(x, value)

    guided = GuidedDenoiser(
        ConditioningEvaluation(
            lambda _value, role: 2.0 if role is GuidanceRole.UNCONDITIONAL else 4.0,
            evaluate,
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)),
        cfg_scale=2.0,
        force_uncond=True,
        input=torch.zeros(1, 2),
        execution=execution(),
    )
    denoised, uncond = guided.call_with_uncond(torch.zeros(1, 2), 1.0)
    assert seen == [provider, provider]
    assert torch.equal(denoised, torch.full((1, 2), 6.0))
    assert torch.equal(uncond, torch.full((1, 2), 2.0))


def test_rank_zero_sampling_bypasses_a_prebuilt_replica_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local: list[float] = []
    replica: list[float] = []

    def evaluate(x: torch.Tensor, _sigma: float, value: float) -> torch.Tensor:
        local.append(value)
        return torch.full_like(x, value)

    def replica_factory(inner: Any) -> Any:
        def evaluate_replica(x: torch.Tensor, sigma: float, request: object) -> object:
            replica.append(sigma)
            return inner(x, sigma, request)

        return evaluate_replica

    guided = GuidedDenoiser(
        ConditioningEvaluation(
            lambda _value, role: 2.0 if role is GuidanceRole.UNCONDITIONAL else 4.0,
            evaluate,
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (lane("c", GuidanceRole.CONDITIONAL), lane("u", GuidanceRole.UNCONDITIONAL)),
        cfg_scale=2.0,
        force_uncond=True,
        input=torch.zeros(1, 2),
        execution=execution(),
        replica_evaluator_factory=replica_factory,
    )
    config = distributed.DistributedSamplingConfig(0, 2, "guidance", "file:///group", "1" * 32)
    monkeypatch.setattr(distributed, "ensure_process_group", lambda: config)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda _tensor, src, group=None: None,
    )
    template = torch.zeros(1, 2)

    output = distributed.run_rank_zero_sampling(lambda: guided(template, 1.0), template, config)

    assert torch.equal(output, torch.full_like(template, 6.0))
    assert local == [4.0, 2.0]
    assert replica == []


def test_guided_denoiser_prepares_and_freezes_the_builtin_plan_at_admission() -> None:
    prepared: list[GuidanceRole] = []

    def prepare(value: object, role: GuidanceRole) -> float:
        prepared.append(role)
        return cast(float, value)

    guided = GuidedDenoiser(
        ConditioningEvaluation(
            prepare,
            lambda x, _sigma, value: torch.full_like(x, value),
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (
            cast(Any, lane("c", GuidanceRole.CONDITIONAL, cast(Any, 4.0))),
            cast(Any, lane("u", GuidanceRole.UNCONDITIONAL, cast(Any, 2.0))),
        ),
        cfg_scale=2.0,
        force_uncond=False,
        input=torch.zeros(1, 2),
        execution=execution(),
    )
    assert prepared == [GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL]
    assert guided.conditioning_plan is not None
    assert torch.equal(guided(torch.zeros(1, 2), 1.0), torch.full((1, 2), 6.0))
    assert torch.equal(guided(torch.zeros(1, 2), 0.5), torch.full((1, 2), 6.0))
    assert prepared == [GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL]


@pytest.mark.parametrize(
    ("batching", "free_memory", "expected"),
    (
        (ConditioningBatching(), 400, (("lane-1", "lane-0"), ("lane-3", "lane-2"))),
        (
            ConditioningBatching(ConditioningBatchingMode.FORCE_SEPARATE),
            10_000,
            (("lane-0",), ("lane-1",), ("lane-2",), ("lane-3",)),
        ),
        (
            ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 3),
            0,
            (("lane-2", "lane-1", "lane-0"), ("lane-3",)),
        ),
        (
            ConditioningBatching(),
            0,
            (("lane-0",), ("lane-1",), ("lane-2",), ("lane-3",)),
        ),
    ),
)
def test_conditioning_plan_sizes_compatible_batches_from_policy_and_free_memory(
    batching: ConditioningBatching,
    free_memory: int,
    expected: tuple[tuple[str, ...], ...],
) -> None:
    conditions = tuple(Conditioning(torch.tensor(float(index))) for index in range(4))
    lanes = tuple(
        lane(f"lane-{index}", GuidanceRole.AUXILIARY, condition)
        for index, condition in enumerate(conditions)
    )
    plan = GuidanceEvaluationPlan(lanes, "lane-0", None)
    memory_reads: list[torch.device] = []

    def read_memory(device: torch.device) -> int:
        memory_reads.append(device)
        return free_memory

    compiler = ConditioningPlanCompiler(
        ConditioningEvaluation(
            lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
            lambda x, _sigma, _condition: x,
            lambda _conditions: True,
            lambda x, _sigma, values: tuple(x for _value in values),
            estimate_activation_memory=lambda _shape, values: len(values) * 100,
        ),
        lanes,
        torch.zeros(1, 4, 8, 8),
        batching,
        read_memory,
    )

    assert tuple(call.lane_ids for call in compiler.prepare_plan(plan).calls) == expected
    assert bool(memory_reads) is (batching.mode is ConditioningBatchingMode.AUTO)


def test_conditioning_plan_uses_the_family_memory_factor_for_auto_batching() -> None:
    conditions = tuple(Conditioning(torch.tensor(float(index))) for index in range(2))
    lanes = tuple(
        lane(f"lane-{index}", GuidanceRole.AUXILIARY, condition)
        for index, condition in enumerate(conditions)
    )
    plan = GuidanceEvaluationPlan(lanes, "lane-0", None)
    compiler = ConditioningPlanCompiler(
        ConditioningEvaluation(
            lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
            lambda x, _sigma, _condition: x,
            lambda _conditions: True,
            lambda x, _sigma, values: tuple(x for _value in values),
            standard_activation_memory_factor=LTXV.memory_factor,
        ),
        lanes,
        torch.zeros(1, 4, 17, 32, 32),
        ConditioningBatching(),
        lambda _device: 20 * 1024**3,
    )

    assert tuple(call.lane_ids for call in compiler.prepare_plan(plan).calls) == (
        ("lane-0",),
        ("lane-1",),
    )


def test_conditioning_plan_defaults_to_exact_tensor_shape_compatibility() -> None:
    conditions = (
        Conditioning(torch.zeros(1, 2)),
        Conditioning(torch.ones(1, 2)),
        Conditioning(torch.zeros(1, 3)),
    )
    lanes = tuple(
        lane(f"lane-{index}", GuidanceRole.AUXILIARY, condition)
        for index, condition in enumerate(conditions)
    )
    plan = GuidanceEvaluationPlan(lanes, "lane-0", None)
    compiler = ConditioningPlanCompiler(
        ConditioningEvaluation(
            lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
            lambda x, _sigma, _condition: x,
            evaluate_batch=lambda x, _sigma, values: tuple(x for _value in values),
        ),
        lanes,
        torch.zeros(1, 4, 8, 8),
        ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 2),
    )

    assert tuple(call.lane_ids for call in compiler.prepare_plan(plan).calls) == (
        ("lane-1", "lane-0"),
        ("lane-2",),
    )


def test_declared_layout_owns_fusion_and_records_canonical_plan_facts() -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    cond_transform = TokenGridTransform(
        "test.positive-text.v1",
        "text",
        "text",
        (2,),
        None,
    )
    uncond_transform = TokenGridTransform(
        "test.negative-text.v1",
        "text",
        "text",
        (2,),
        None,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 2), 2.0))
    physical: list[tuple[torch.Tensor, ...]] = []
    compatible: list[tuple[torch.Tensor, ...]] = []

    def family_compatible(conditions: tuple[torch.Tensor, ...]) -> bool:
        compatible.append(conditions)
        return True

    def validate(value: torch.Tensor, declared: ModelTokenLayout) -> None:
        if value.shape[1] != declared.valid_rows:
            raise TokenLayoutError("conditioning rows do not match the declared layout")

    def evaluate_batch(
        _x: torch.Tensor,
        _sigma: float,
        conditions: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        physical.append(conditions)
        return conditions

    guided = GuidedDenoiser(
        ConditioningEvaluation(
            lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
            lambda _x, _sigma, value: value,
            family_compatible,
            evaluate_batch,
            evaluator_identity=lambda _role: "test.conditioning.v1",
            layout=lambda _value: layout,
            token_transforms=lambda value: (cond_transform if value is cond else uncond_transform,),
            validate_layout=validate,
            standard_activation_memory_factor=1.0,
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (
            lane("c", GuidanceRole.CONDITIONAL, cond),
            lane("u", GuidanceRole.UNCONDITIONAL, uncond),
        ),
        cfg_scale=2.0,
        force_uncond=False,
        input=torch.zeros(1, 2),
        execution=execution(),
    )

    assert torch.equal(guided(torch.zeros(1, 2), 1.0), torch.full((1, 2), 6.0))
    assert compatible == [(cond.embeddings, uncond.embeddings)]
    assert physical == [(uncond.embeddings, cond.embeddings)]
    compiled = guided.conditioning_plan
    assert compiled is not None
    assert compiled.calls[0].lane_ids == ("u", "c")
    assert compiled.calls[0].evaluator_identity == "test.conditioning.v1"
    assert compiled.calls[0].layout_digest == layout.digest
    assert tuple(lane.validation for lane in compiled.lanes) == (
        ConditioningValidationPath.LAYOUT_BACKED,
        ConditioningValidationPath.LAYOUT_BACKED,
    )
    assert all(lane.layout_digest == layout.digest for lane in compiled.lanes)
    assert compiled.lanes[0].token_transforms == (
        (cond_transform.transform, cond_transform.digest),
    )
    assert compiled.lanes[1].token_transforms == (
        (uncond_transform.transform, uncond_transform.digest),
    )
    assert compiled.fact_lines == (
        "conditioning-lane-plan.v1",
        '{"call":0,"lane":"c","layout":"079d0e1f1c4b1a537d89bc57a86ee19b1a5a7198404ac500b54e52cb4a407576","physical":1,"role":"conditional","transforms":[{"digest":"75c00bbfe58e1ee83bfef66e4ba2ac5560f4c0f3dadfbdb1bcd9459feb3f74db","identity":"test.positive-text.v1"}],"validation":"layout-backed"}',
        '{"call":0,"lane":"u","layout":"079d0e1f1c4b1a537d89bc57a86ee19b1a5a7198404ac500b54e52cb4a407576","physical":0,"role":"unconditional","transforms":[{"digest":"bdb85eae0b0c6ad265caa155ae584ca5f0b42972035446b66ba436df43ab3e15","identity":"test.negative-text.v1"}],"validation":"layout-backed"}',
        '{"evaluator":"test.conditioning.v1","lanes":["u","c"],"layout":"079d0e1f1c4b1a537d89bc57a86ee19b1a5a7198404ac500b54e52cb4a407576","validation":"layout-backed"}',
    )
    assert (
        compiled.plan_digest == "959aeb0775b87be59fe7f317240884710d8b649924da1f7eb33f51586de7a7fb"
    )
    changed = replace(
        compiled,
        lanes=(
            replace(
                compiled.lanes[0],
                token_transforms=(("test.changed.v1", "0" * 64),),
            ),
            compiled.lanes[1],
        ),
    )
    assert changed.plan_digest != compiled.plan_digest
    assert len(compiled.plan_digest) == 64
    assert all("\n" not in line and "\r" not in line for line in compiled.fact_lines)


def test_declared_layout_tensor_mismatch_refuses_before_model_evaluation() -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    calls = 0

    def validate(value: torch.Tensor, declared: ModelTokenLayout) -> None:
        if value.shape[1] != declared.valid_rows:
            raise TokenLayoutError("conditioning rows do not match the declared layout")

    def evaluate(_x: torch.Tensor, _sigma: float, value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return value

    with pytest.raises(GuidanceContractError, match="conditioning rows do not match"):
        GuidedDenoiser(
            ConditioningEvaluation(
                lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
                evaluate,
                evaluator_identity=lambda _role: "test.conditioning.v1",
                layout=lambda _value: layout,
                validate_layout=validate,
            ),
            GuidanceExecutor(GuidanceRegistry()),
            (lane("c", GuidanceRole.CONDITIONAL, Conditioning(torch.zeros(1, 3))),),
            cfg_scale=1.0,
            force_uncond=False,
            input=torch.zeros(1, 2),
            execution=execution(),
        )
    assert calls == 0


def test_declared_layout_disagreement_refuses_instead_of_splitting_fusion() -> None:
    first = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    second = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 3, (3,)),),
        0,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 3), 2.0))
    calls = 0

    def evaluate_batch(
        _x: torch.Tensor,
        _sigma: float,
        conditions: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return conditions

    with pytest.raises(GuidanceContractError, match="layout mismatch"):
        GuidedDenoiser(
            ConditioningEvaluation(
                lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
                lambda _x, _sigma, value: value,
                lambda _conditions: True,
                evaluate_batch,
                evaluator_identity=lambda _role: "test.conditioning.v1",
                layout=lambda value: first if value is cond else second,
                validate_layout=lambda _value, _layout: None,
            ),
            GuidanceExecutor(GuidanceRegistry()),
            (
                lane("c", GuidanceRole.CONDITIONAL, cond),
                lane("u", GuidanceRole.UNCONDITIONAL, uncond),
            ),
            cfg_scale=2.0,
            force_uncond=False,
            input=torch.zeros(1, 2),
            execution=execution(),
        )
    assert calls == 0


def test_inner_call_layout_disagreement_refuses_admitted_fusion() -> None:
    outer = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    first_inner = ModelTokenLayout(
        (ModelTokenSegment("image", "image", "latent", 0, 4, (2, 2)),),
        0,
    )
    second_inner = ModelTokenLayout(
        (ModelTokenSegment("image", "image", "latent", 0, 4, (1, 4)),),
        0,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 2), 2.0))
    calls = 0

    def evaluate_batch(
        _x: torch.Tensor,
        _sigma: float,
        conditions: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return conditions

    with pytest.raises(GuidanceContractError, match="inner-call layout mismatch"):
        GuidedDenoiser(
            ConditioningEvaluation(
                lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
                lambda _x, _sigma, value: value,
                lambda _conditions: True,
                evaluate_batch,
                evaluator_identity=lambda _role: "test.conditioning.v1",
                layout=lambda _value: outer,
                validate_layout=lambda _value, _layout: None,
                inner_calls=lambda value: (
                    ((first_inner if value is cond.embeddings else second_inner), ()),
                ),
            ),
            GuidanceExecutor(GuidanceRegistry()),
            (
                lane("c", GuidanceRole.CONDITIONAL, cond),
                lane("u", GuidanceRole.UNCONDITIONAL, uncond),
            ),
            cfg_scale=2.0,
            force_uncond=False,
            input=torch.zeros(1, 2),
            execution=execution(),
        )
    assert calls == 0


def test_declared_layouts_form_separate_calls_when_family_batching_refuses() -> None:
    first = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    second = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 3, (3,)),),
        0,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 3), 2.0))

    def prepare(value: object, _role: GuidanceRole) -> tuple[int, float]:
        source = cast("Conditioning[torch.Tensor]", value).embeddings
        return source.shape[1], float(source[0, 0])

    guided = GuidedDenoiser(
        ConditioningEvaluation(
            prepare,
            lambda x, _sigma, value: torch.full_like(x, value[1]),
            lambda values: all(value[0] == values[0][0] for value in values[1:]),
            lambda _x, _sigma, _values: (_ for _ in ()).throw(
                AssertionError("incompatible layouts must not be fused")
            ),
            evaluator_identity=lambda _role: "test.conditioning.v1",
            layout=lambda value: first if value is cond else second,
            validate_layout=lambda _value, _layout: None,
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (
            lane("c", GuidanceRole.CONDITIONAL, cond),
            lane("u", GuidanceRole.UNCONDITIONAL, uncond),
        ),
        cfg_scale=2.0,
        force_uncond=False,
        input=torch.zeros(1, 2),
        execution=execution(),
    )

    assert torch.equal(guided(torch.zeros(1, 2), 1.0), torch.full((1, 2), 6.0))
    compiled = guided.conditioning_plan
    assert compiled is not None
    assert tuple(call.lane_ids for call in compiled.calls) == (("c",), ("u",))
    assert tuple(call.layout_digest for call in compiled.calls) == (first.digest, second.digest)


def test_mixed_declared_and_absent_layouts_refuse_before_grouping() -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 2), 2.0))

    with pytest.raises(GuidanceContractError, match="mix declared and absent"):
        GuidedDenoiser(
            ConditioningEvaluation(
                lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
                lambda _x, _sigma, value: value,
                lambda _conditions: True,
                lambda _x, _sigma, values: values,
                evaluator_identity=lambda _role: "test.conditioning.v1",
                layout=lambda value: layout if value is cond else None,
                validate_layout=lambda _value, _layout: None,
            ),
            GuidanceExecutor(GuidanceRegistry()),
            (
                lane("c", GuidanceRole.CONDITIONAL, cond),
                lane("u", GuidanceRole.UNCONDITIONAL, uncond),
            ),
            cfg_scale=2.0,
            force_uncond=False,
            input=torch.zeros(1, 2),
            execution=execution(),
        )


def test_fused_evaluator_identity_disagreement_refuses() -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("text", "text", "context", 0, 2, (2,)),),
        0,
    )
    cond = Conditioning(torch.full((1, 2), 4.0))
    uncond = Conditioning(torch.full((1, 2), 2.0))
    calls = 0

    def evaluate_batch(
        _x: torch.Tensor,
        _sigma: float,
        conditions: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return conditions

    with pytest.raises(GuidanceContractError, match="evaluator identity mismatch"):
        GuidedDenoiser(
            ConditioningEvaluation(
                lambda value, _role: cast("Conditioning[torch.Tensor]", value).embeddings,
                lambda _x, _sigma, value: value,
                lambda _conditions: True,
                evaluate_batch,
                evaluator_identity=lambda role: f"test.{role.value}.v1",
                layout=lambda _value: layout,
                validate_layout=lambda _value, _layout: None,
            ),
            GuidanceExecutor(GuidanceRegistry()),
            (
                lane("c", GuidanceRole.CONDITIONAL, cond),
                lane("u", GuidanceRole.UNCONDITIONAL, uncond),
            ),
            cfg_scale=2.0,
            force_uncond=False,
            input=torch.zeros(1, 2),
            execution=execution(),
        )
    assert calls == 0


def test_cfg_override_skips_unconditional_and_composes_with_flow_rescale() -> None:
    from dinkster_inference_torch.guidance_transforms import cfg_override, rescale_cfg

    contributions = (
        ("override", cfg_override(1.0, 0.0, 0.6)),
        ("rescale", rescale_cfg(0.7, flow=True)),
    )
    executor = GuidanceExecutor(GuidanceRegistry(contributions))
    conditions = (
        lane("c", GuidanceRole.CONDITIONAL),
        lane("u", GuidanceRole.UNCONDITIONAL),
    )
    evaluated: list[tuple[str, ...]] = []

    def evaluate(request):
        evaluated.append(tuple(item.id for item in request.plan.lanes))
        values = {"c": (1.0, 3.0), "u": (-2.0, 2.0)}
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    item.id,
                    torch.tensor([values[item.id]]),
                    GuidancePredictionSource.MODEL,
                )
                for item in request.plan.lanes
            )
        )

    inside = context(*conditions, cfg=7.5)
    inside = replace(
        inside,
        sigma=torch.tensor(0.8),
        execution=replace(inside.execution, current_sigma=0.5),
    )
    result = executor.execute(inside, evaluate)
    assert evaluated == [("c",)]
    cond = torch.tensor([[1.0, 3.0]])
    zero = torch.zeros_like(cond)
    cfg_one = zero + (cond - zero) * 1.0
    cfg_one_rescaled = cfg_one * (
        cond.std(dim=1, keepdim=True) / cfg_one.std(dim=1, keepdim=True).clamp(min=1e-8)
    )
    cfg_one_final = 0.7 * cfg_one_rescaled + 0.3 * cfg_one
    expected_cfg_one = inside.input - (inside.input - cfg_one_final)
    assert torch.equal(result.denoised, expected_cfg_one)

    outside = context(*conditions, cfg=7.5)
    outside = replace(
        outside,
        sigma=torch.tensor(0.5),
        execution=replace(outside.execution, current_sigma=0.8),
    )
    result = executor.execute(outside, evaluate)
    cond = torch.tensor([[1.0, 3.0]])
    uncond = torch.tensor([[-2.0, 2.0]])
    cfg = uncond + (cond - uncond) * 7.5
    rescaled = cfg * (cond.std(dim=1, keepdim=True) / cfg.std(dim=1, keepdim=True))
    expected = 0.7 * rescaled + 0.3 * cfg
    assert evaluated[-1] == ("c", "u")
    torch.testing.assert_close(result.denoised, expected, rtol=0, atol=0)


def test_unconditional_primary_requires_explicit_strategy_declaration() -> None:
    condition = lane("u", GuidanceRole.UNCONDITIONAL)

    def plan(context):
        del context
        return GuidanceEvaluationPlan((condition,), "u", "u")

    def reduce(context):
        return context.predictions.items[0].value

    undeclared = GuidanceContribution(
        strategy=GuidanceStrategyDescriptor("x.undeclared", plan, reduce)
    )
    with pytest.raises(GuidanceContractError, match="primary lane must be conditional"):
        GuidanceExecutor(GuidanceRegistry((("owner", undeclared),))).execute(
            context(condition), predictions
        )

    declared = GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "x.declared", plan, reduce, allows_unconditional_primary=True
        )
    )
    result = GuidanceExecutor(GuidanceRegistry((("owner", declared),))).execute(
        context(condition), predictions
    )
    assert torch.equal(result.denoised, torch.ones_like(result.denoised))
