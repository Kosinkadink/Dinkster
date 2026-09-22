"""PAG through declared attention points and auxiliary condition evaluation."""

from dataclasses import replace

from dinkster_api.v1 import (
    AttentionContribution,
    AttentionSelector,
    AttentionWrapperDescriptor,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidancePostCFGDescriptor,
    InferenceContribution,
)

NODES = ()


def make(*, scale=1.0, torch_version="2.13.0+cu130", aimdo_version="0.5.5.post2"):
    if not 0 <= scale <= 100:
        raise ValueError("PAG scale must be between zero and 100")

    def perturb(q, k, v, context, next):
        if context.state.get("perturbed", False):
            return v
        return next(q, k, v)

    def post(context):
        if scale == 0:
            return context.reduced
        evaluate = context.evaluate_conditions
        if evaluate is None:
            raise RuntimeError("PAG requires auxiliary condition evaluation")
        request = context.request
        primary = next(lane for lane in request.plan.lanes if lane.id == request.plan.primary_id)
        conditional = next(
            item.value for item in context.predictions.items if item.lane_id == primary.id
        )
        state = request.execution.extension_state["proof_pag"]
        previous = state.get("perturbed", False)
        state["perturbed"] = True
        try:
            result = evaluate(
                replace(
                    request,
                    plan=GuidanceEvaluationPlan((primary,), primary.id, None),
                )
            )
        finally:
            state["perturbed"] = previous
        return context.reduced + (conditional - result.items[0].value) * scale

    metadata = (("config.scale", str(scale)),)
    return InferenceContribution(
        attention=AttentionContribution(
            torch_version=torch_version,
            aimdo_version=aimdo_version,
            wrappers=(
                AttentionWrapperDescriptor(
                    "proof_pag.unet",
                    AttentionSelector("unet", "middle_block.1.transformer_blocks.0", "self"),
                    perturb,
                    terminal=True,
                    behavior_metadata=metadata,
                ),
                AttentionWrapperDescriptor(
                    "proof_pag.flux",
                    AttentionSelector("flux", "double_blocks.0", "joint"),
                    perturb,
                    terminal=True,
                    behavior_metadata=metadata,
                ),
            ),
        ),
        guidance=GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor("proof_pag.guidance", post, behavior_metadata=metadata),
            )
        ),
    )


def register():
    return make()
