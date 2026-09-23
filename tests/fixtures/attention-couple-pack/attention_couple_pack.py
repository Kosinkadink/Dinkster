# pyright: reportMissingImports=false
"""Blend prompt-specific attention outputs with two spatial region masks."""

from dinkster_api.v1 import (
    AttentionContribution,
    AttentionOutputDescriptor,
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
        for span in (top, bottom):
            current = output[span.batch_start : span.batch_end, :, span.start : span.end]
            result[span.batch_start : span.batch_end, :, span.start : span.end] = current.lerp(
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
