"""Optical-flow-correlated noise for VOID video refinement."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    OPTICAL_FLOW_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class VOIDWarpedNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        return NodeSchema(
            node_type="VOIDWarpedNoise",
            category="model/latent/void",
            inputs=(
                InputSpec(
                    "optical_flow",
                    TypeExpr.concrete(OPTICAL_FLOW_TYPE),
                    doc="Optical flow model from OpticalFlowLoader (RAFT-large).",
                ),
                InputSpec(
                    "video",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Pass 1 output video frames [T, H, W, 3]",
                ),
                InputSpec(
                    "width",
                    integer,
                    default=672,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
                InputSpec(
                    "height",
                    integer,
                    default=384,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
                InputSpec(
                    "length",
                    integer,
                    default=45,
                    widget=NumberWidget(min=1, max=16384, step=1),
                    doc=(
                        "Number of pixel frames. Rounded down to make latent_t even "
                        "(patch_size_t=2 requirement), e.g. 49 -> 45."
                    ),
                ),
                InputSpec("batch_size", integer, default=1, widget=NumberWidget(min=1, max=64)),
            ),
            outputs=(
                OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE), display_name="warped_noise"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        optical_flow: Any,
        video: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        warp = importlib.import_module("dinkster_nodes_comfy._void_noise_warp")

        latent_t = ((length - 1) // 4) + 1
        if latent_t % 2:
            adjusted_length = (max(2, (latent_t // 2) * 2) - 1) * 4 + 1
            logging.warning(
                "VOIDWarpedNoise: rounding length %d down to %d so that latent_t "
                "is even (required by CogVideoX-Fun-V1.5 patch_size_t=2).",
                length,
                adjusted_length,
            )
            length = adjusted_length
        latent_t = ((length - 1) // 4) + 1
        latent_h, latent_w = height // 8, width // 8
        device = management.get_torch_device()
        management.load_model_gpu(optical_flow)
        raft = warp.RaftOpticalFlow(optical_flow.model, device=device)
        frames = materialize_image(video)[:length].to(device)
        frames = utils.common_upscale(
            frames.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        warped = warp.get_noise_from_video(
            frames,
            raft,
            noise_channels=16,
            resize_frames=0.5,
            resize_flow=8,
            downscale_factor=32,
            device=device,
        )
        if warped.shape[0] != latent_t:
            indices = torch.linspace(0, warped.shape[0] - 1, latent_t, device=device).long()
            warped = warped[indices]
        if warped.shape[1] != latent_h or warped.shape[2] != latent_w:
            warped = torch.nn.functional.interpolate(
                warped.permute(0, 3, 1, 2),
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        samples = warped.permute(3, 0, 1, 2).unsqueeze(0)
        if batch_size > 1:
            samples = samples.repeat(batch_size, 1, 1, 1, 1)
        return cls.outputs(LATENT={"samples": samples.to(management.intermediate_device())})
