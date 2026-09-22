# pyright: reportMissingImports=false
"""Blend prompt-specific attention outputs with two spatial region masks."""

from dinkster_api.v1 import (
    AttentionContribution,
    AttentionOutputDescriptor,
    AttentionSelector,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceStrategyDescriptor,
    InferenceContribution,
)

NODES = ()


def make(*, split=0.5, torch_version="2.13.0+cu130", aimdo_version="0.5.5.post2"):
    if not 0 < split < 1:
        raise ValueError("region split must be strictly inside the image")

    def plan(context):
        lanes = tuple(lane for lane in context.conditions if lane.id in ("positive", "negative"))
        if tuple(lane.id for lane in lanes) != ("positive", "negative"):
            raise ValueError("attention-couple requires positive and negative prompt lanes")
        return GuidanceEvaluationPlan(lanes, "positive", None)

    def reduce(context):
        import torch

        predictions = {item.lane_id: item.value for item in context.predictions.items}
        left, right = predictions["positive"], predictions["negative"]
        width = left.shape[-1]
        mask_left = torch.arange(width, device=left.device) < width * split
        mask_left = mask_left.to(left.dtype).view(1, 1, 1, width)
        return left * mask_left + right * (1 - mask_left)

    def couple(output, context):
        import torch

        spans = [span for span in context.spans if span.axis == "query" and span.stream == "image"]
        regions = {span.condition_id: span for span in spans}
        lane_ids = (
            ("left", "right")
            if "left" in regions and "right" in regions
            else (
                "positive",
                "negative",
            )
        )
        if any(lane_id not in regions for lane_id in lane_ids):
            raise ValueError(
                "attention-couple requires fused positive and negative conditioning lanes"
            )
        left, right = (regions[lane_id] for lane_id in lane_ids)
        height, width = context.spatial_shape
        if left.end - left.start != height * width or right.end - right.start != height * width:
            raise ValueError("region masks require an image-token grid without reference tokens")
        if left.batch_end - left.batch_start != right.batch_end - right.batch_start:
            raise ValueError("region conditioning batches differ")
        mask_left = (torch.arange(width, device=output.device) < width * split).repeat(height)
        mask_left = mask_left.to(output.dtype).view(1, 1, height * width, 1)
        mask_right = 1 - mask_left
        result = output.clone()
        mixed = (
            output[left.batch_start : left.batch_end, :, left.start : left.end] * mask_left
            + output[right.batch_start : right.batch_end, :, right.start : right.end] * mask_right
        )
        for span in (left, right):
            result[span.batch_start : span.batch_end, :, span.start : span.end] = mixed
        return result

    metadata = (("config.regions", "left,right"), ("config.split", str(split)))
    return InferenceContribution(
        attention=AttentionContribution(
            torch_version=torch_version,
            aimdo_version=aimdo_version,
            outputs=(
                AttentionOutputDescriptor(
                    "proof_couple.unet",
                    AttentionSelector("unet", kind="cross"),
                    couple,
                    behavior_metadata=metadata,
                ),
                AttentionOutputDescriptor(
                    "proof_couple.flux",
                    AttentionSelector("flux", kind="joint"),
                    couple,
                    behavior_metadata=metadata,
                ),
            ),
        ),
        guidance=GuidanceContribution(
            strategy=GuidanceStrategyDescriptor(
                "proof_couple.regional_reducer",
                plan,
                reduce,
                behavior_metadata=metadata,
            )
        ),
    )


def register():
    return make()
