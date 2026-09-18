"""Out-of-tree S3 proof pack B: a composing guidance strategy."""

from dinkster_api.v1 import (
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePhaseParticipation,
    GuidancePostCFGDescriptor,
    GuidancePreCFGDescriptor,
    GuidanceStrategyDescriptor,
    InferenceContribution,
)


def _events(execution):
    return execution.extension_state["proof_b"].setdefault("events", [])


def _wrapper(request, next):
    events = _events(request.execution)
    events.append("B-enter")
    result = next(request)
    events.append("B-exit")
    return result


def _pre(context):
    _events(context.request.execution).append("B-pre")
    return context.predictions


def _plan(context):
    lanes = tuple(lane for lane in context.conditions if lane.conditioning is not None)
    uncond = next((lane for lane in lanes if lane.role.value == "unconditional"), None)
    primary = next(lane for lane in lanes if lane.role.value == "conditional")
    return GuidanceEvaluationPlan(lanes, primary.id, None if uncond is None else uncond.id)


def _reduce(context):
    _events(context.request.execution).append("B-reduce")
    values = {item.lane_id: item.value for item in context.predictions.items}
    primary = values[context.request.plan.primary_id]
    uncond_id = context.request.plan.unconditional_id
    if uncond_id is None:
        return primary
    uncond = values[uncond_id]
    return uncond + (primary - uncond) * context.cfg_scale


def _post(context):
    _events(context.request.execution).append("B-post")
    return context.reduced


GUIDANCE = GuidanceContribution(
    evaluation_wrappers=(
        GuidanceEvaluationWrapperDescriptor(
            "proof_b.guidance_strategy.wrapper", _wrapper, order=-10
        ),
    ),
    pre_cfg=(GuidancePreCFGDescriptor("proof_b.guidance_strategy.pre", _pre, order=-10),),
    strategy=GuidanceStrategyDescriptor(
        "proof_b.guidance_strategy",
        _plan,
        _reduce,
        # Spell out the fixture's intended default contract.
        GuidancePhaseParticipation.COMPOSE,
    ),
    post_cfg=(GuidancePostCFGDescriptor("proof_b.guidance_strategy.post", _post, order=-10),),
)


def register():
    return InferenceContribution(guidance=GUIDANCE)


BYPASS_GUIDANCE = GuidanceContribution(
    evaluation_wrappers=GUIDANCE.evaluation_wrappers,
    pre_cfg=GUIDANCE.pre_cfg,
    strategy=GuidanceStrategyDescriptor(
        "proof_b.guidance_strategy.bypass",
        _plan,
        _reduce,
        GuidancePhaseParticipation.BYPASS_TRANSFORMS,
    ),
    post_cfg=GUIDANCE.post_cfg,
)


def register_bypass():
    return InferenceContribution(guidance=BYPASS_GUIDANCE)


def register_declaration_mismatch():
    return InferenceContribution(guidance=BYPASS_GUIDANCE)


def register_materialization_raises():
    raise RuntimeError("proof_b materialization raised")


def register_callback_raises():
    def callback(request, next):
        del request, next
        raise RuntimeError("proof_b callback raised")

    return InferenceContribution(
        guidance=GuidanceContribution(
            evaluation_wrappers=(
                GuidanceEvaluationWrapperDescriptor(
                    "proof_b.callback_raise.wrapper", callback, order=-10
                ),
            )
        )
    )
