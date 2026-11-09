"""CPU contracts for Wan SCAIL conditioning."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.wan_scail_conditioning import WanSCAILToVideo


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning(start: float = 0.0, end: float = 1.0) -> list[list[object]]:
    return [[torch.ones((1, 1)), {"start_percent": start, "end_percent": end}]]


def test_mask_color_channels_temporal_order_and_strict_threshold() -> None:
    colors = torch.zeros((5, 8, 8, 3))
    colors[0, :, :, :] = 225 / 255
    colors[1, :, :, 0] = 1
    colors[2, :, :, 1] = 1
    colors[3, :, :, 2] = 1
    colors[4, :, :, :] = 1
    output = WanSCAILToVideo.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=VAE(),
        width=16,
        height=16,
        length=5,
        batch_size=1,
        pose_strength=1,
        pose_start=0,
        pose_end=1,
        video_frame_offset=0,
        previous_frame_count=1,
        pose_video_mask=colors,
    )
    result = output["positive"][0][1]["driving_mask_28ch"]
    assert result.shape == (1, 2, 28, 1, 1)
    assert not result[0, 0].count_nonzero()
    assert result[0, 1, [1, 9, 17, 21], 0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_schema_and_reference_append_replacement_without_mutation() -> None:
    schema = WanSCAILToVideo.define_schema()
    assert [item.id for item in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "pose_video",
        "pose_video_mask",
        "replacement_mode",
        "pose_strength",
        "pose_start",
        "pose_end",
        "reference_image",
        "reference_image_mask",
        "clip_vision_output",
        "video_frame_offset",
        "previous_frame_count",
        "previous_frames",
    ]
    assert schema.category == "model/conditioning/wan/scail"
    positive, negative = _conditioning(), _conditioning()
    references = torch.stack((torch.ones((8, 8, 3)), torch.full((8, 8, 3), 2.0)))
    masks = torch.zeros_like(references)
    masks[1] = 1
    references_before, masks_before = references.clone(), masks.clone()
    vae = VAE()
    result = WanSCAILToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=8,
        height=8,
        length=5,
        batch_size=1,
        pose_strength=1,
        pose_start=0,
        pose_end=1,
        video_frame_offset=0,
        previous_frame_count=1,
        replacement_mode=True,
        reference_image=references,
        reference_image_mask=masks,
    )
    assert len(result["positive"][0][1]["reference_latents"]) == 2
    assert not vae.images[0].count_nonzero() and vae.images[1].count_nonzero()
    assert torch.equal(references, references_before) and torch.equal(masks, masks_before)


def test_reference_masks_follow_reference_order_without_replacement() -> None:
    references = torch.stack((torch.ones((8, 8, 3)), torch.full((8, 8, 3), 2.0)))
    masks = torch.zeros_like(references)
    masks[0, :, :, 0] = 1
    masks[1, :, :, 2] = 1
    vae = VAE()
    result = WanSCAILToVideo.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=vae,
        width=8,
        height=8,
        length=5,
        batch_size=1,
        pose_strength=1,
        pose_start=0,
        pose_end=1,
        video_frame_offset=0,
        previous_frame_count=1,
        reference_image=references,
        reference_image_mask=masks,
    )
    metadata = result["positive"][0][1]
    assert vae.images[0].mean() == 1 and vae.images[1].mean() == 2
    assert metadata["ref_mask_flag"] is True
    ref_mask = metadata["ref_mask_28ch"]
    assert ref_mask.shape == (1, 4, 28, 1, 1)
    assert ref_mask[0, 0, 3, 0, 0] == 1 and ref_mask[0, 1, 1, 0, 0] == 1
    assert not ref_mask[0, 2:].count_nonzero()


def test_pose_ranges_joint_truncation_and_masks() -> None:
    positive, negative = _conditioning(0.2, 0.9), _conditioning(0.2, 0.9)
    pose = torch.ones((9, 8, 8, 3))
    mask = torch.zeros((6, 8, 8, 3))
    mask[:, :, :, 0] = 1
    result = WanSCAILToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=8,
        height=8,
        length=9,
        batch_size=1,
        pose_strength=2,
        pose_start=0.4,
        pose_end=0.6,
        video_frame_offset=0,
        previous_frame_count=1,
        pose_video=pose,
        pose_video_mask=mask,
    )
    entries = result["positive"]
    assert [(entry[1]["start_percent"], entry[1]["end_percent"]) for entry in entries] == [
        (0.2, 0.4 - 1e-5),
        (0.4, 0.6),
        (0.6 + 1e-5, 0.9),
    ]
    assert entries[1][1]["pose_video_latent"].shape[2] == 2
    assert entries[0][1]["driving_mask_28ch"].shape[1] == 2


def test_continuation_offset_noise_mask_and_sources() -> None:
    positive, negative = _conditioning(), _conditioning()
    previous = torch.stack([torch.full((8, 8, 3), float(index)) for index in range(1, 7)])
    before = previous.clone()
    result = WanSCAILToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=8,
        height=8,
        length=9,
        batch_size=1,
        pose_strength=1,
        pose_start=0,
        pose_end=1,
        video_frame_offset=5,
        previous_frame_count=5,
        previous_frames=previous,
    )
    latent, noise = result["latent"]["samples"], result["latent"]["noise_mask"]
    assert result["video_frame_offset"] == 9
    assert latent[0, 0, :2, 0, 0].tolist() == [2.0, 6.0]
    assert noise[0, 0, :, 0, 0].tolist() == [0.0, 0.0, 1.0]
    assert torch.equal(previous, before)
