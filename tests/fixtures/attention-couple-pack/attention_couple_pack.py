# pyright: reportMissingImports=false
"""Blend prompt-specific attention outputs with two spatial region masks."""

from dinkster_api.v1 import (
    AttentionContribution,
    AttentionOutputDescriptor,
    AttentionQKVDescriptor,
    AttentionSelector,
    InferenceContribution,
)

NODES = ()


def make(
    *,
    split=0.5,
    attention_strength=1.0,
    torch_version="2.13.0+cu130",
    aimdo_version="0.5.5.post2",
):
    if not 0 < split < 1:
        raise ValueError("region split must be strictly inside the image")
    if not 0 <= attention_strength <= 1:
        raise ValueError("attention strength must be between zero and one")

    def top_region_weights(torch, height, *, device, dtype):
        positions = torch.arange(height, device=device)
        return (positions < height * split).to(dtype)

    def regional_qkv(q, k, v, context):
        next_q, next_k, next_v = q.clone(), k.clone(), v.clone()
        spans = {(span.axis, span.stream, span.condition_id): span for span in context.spans}
        for axis, tensors in (("query", (q, next_q)), ("key", (k, next_k)), ("key", (v, next_v))):
            if context.kind == "cross" and axis == "key":
                continue
            positive = spans[(axis, "image", "positive")]
            middle = spans[(axis, "image", "middle")]
            source, target = tensors
            target[middle.batch_start : middle.batch_end, :, middle.start : middle.end] = source[
                positive.batch_start : positive.batch_end, :, positive.start : positive.end
            ]
        return next_q, next_k, next_v

    def couple(output, context):
        import torch

        spans = [span for span in context.spans if span.axis == "query" and span.stream == "image"]
        regions = {span.condition_id: span for span in spans}
        lane_ids = ("positive", "middle")
        if any(lane_id not in regions for lane_id in lane_ids):
            raise ValueError("attention-couple requires fused positive and regional prompt lanes")
        top, bottom = (regions[lane_id] for lane_id in lane_ids)
        height, width = context.spatial_shape
        if top.end - top.start != height * width or bottom.end - bottom.start != height * width:
            raise ValueError("region masks require an image-token grid without reference tokens")
        if top.batch_end - top.batch_start != bottom.batch_end - bottom.batch_start:
            raise ValueError("region conditioning batches differ")
        mask_top = top_region_weights(
            torch, height, device=output.device, dtype=output.dtype
        ).repeat_interleave(width)
        mask_top = mask_top.view(1, 1, height * width, 1)
        mask_bottom = 1 - mask_top
        result = output.clone()
        mixed = (
            output[top.batch_start : top.batch_end, :, top.start : top.end] * mask_top
            + output[bottom.batch_start : bottom.batch_end, :, bottom.start : bottom.end]
            * mask_bottom
        )
        current = output[top.batch_start : top.batch_end, :, top.start : top.end]
        result[top.batch_start : top.batch_end, :, top.start : top.end] = current.lerp(
            mixed, attention_strength
        )
        return result

    metadata = (
        ("config.attention-strength", str(attention_strength)),
        ("config.regions", "top,bottom"),
        ("config.split", str(split)),
    )
    return InferenceContribution(
        attention=AttentionContribution(
            torch_version=torch_version,
            aimdo_version=aimdo_version,
            qkv=(
                AttentionQKVDescriptor(
                    "proof_couple.unet.inputs",
                    AttentionSelector("unet", kind="cross"),
                    regional_qkv,
                    behavior_metadata=metadata,
                ),
                AttentionQKVDescriptor(
                    "proof_couple.flux.inputs",
                    AttentionSelector("flux", kind="joint"),
                    regional_qkv,
                    behavior_metadata=metadata,
                ),
            ),
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
    )


def register():
    return make()
