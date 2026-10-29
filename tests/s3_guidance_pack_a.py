"""Out-of-tree S3 proof pack A: cfg-rescale phase contributions."""

from dinkster_api.v1 import (
    GuidanceContribution,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePostCFGDescriptor,
    GuidancePreCFGDescriptor,
    GuidanceStrategyDescriptor,
    InferenceContribution,
)


def _events(execution):
    return execution.extension_state["proof_a"].setdefault("events", [])


def _wrapper(request, next):
    events = _events(request.execution)
    events.append("A-enter")
    result = next(request)
    events.append("A-exit")
    return result


def _pre(context):
    _events(context.request.execution).append("A-pre")
    return context.predictions


def _post(context):
    _events(context.request.execution).append("A-post")
    # Deterministic CFG-rescale: preserve the guided mean while matching the
    # primary prediction's per-sample standard deviation.
    guided = context.reduced
    primary = next(
        item.value
        for item in context.predictions.items
        if item.lane_id == context.request.plan.primary_id
    )
    dims = tuple(range(1, guided.ndim))
    guided_std = guided.std(dim=dims, keepdim=True, correction=0)
    primary_std = primary.std(dim=dims, keepdim=True, correction=0)
    guided_mean = guided.mean(dim=dims, keepdim=True)
    scale = primary_std / guided_std.clamp_min(1e-12)
    return guided_mean + (guided - guided_mean) * scale


GUIDANCE = GuidanceContribution(
    evaluation_wrappers=(
        GuidanceEvaluationWrapperDescriptor("proof_a.cfg_rescale.wrapper", _wrapper, order=20),
    ),
    pre_cfg=(GuidancePreCFGDescriptor("proof_a.cfg_rescale.pre", _pre, order=20),),
    post_cfg=(GuidancePostCFGDescriptor("proof_a.cfg_rescale.post", _post, order=20),),
)


ALTERNATE_GUIDANCE = GuidanceContribution(
    strategy=GuidanceStrategyDescriptor(
        "proof_a.alternate_strategy",
        lambda context: __import__("s3_guidance_pack_b", fromlist=["_plan"])._plan(context),
        lambda context: __import__("s3_guidance_pack_b", fromlist=["_reduce"])._reduce(context),
    )
)


def register():
    return InferenceContribution(guidance=GUIDANCE)


def register_with_strategy():
    return InferenceContribution(guidance=ALTERNATE_GUIDANCE)


def register_raises():
    raise RuntimeError("proof_a callback raised")
