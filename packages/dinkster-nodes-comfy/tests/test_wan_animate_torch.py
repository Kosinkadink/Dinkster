"""CPU contracts for native Wan animation conditioning."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_nodes_comfy.wan_animate import (
    WanAnimate2Cache,
    WanAnimate2ToVideo,
    WanAnimateToVideo,
)


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image[::4, ::8, ::8, :1].movedim(-1, 0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning(label: str) -> list[list[Any]]:
    return [[torch.tensor([[1.0]]), {"label": label}]]


def _image(values: list[float], channels: int = 3) -> torch.Tensor:
    return torch.stack([torch.full((8, 16, channels), value) for value in values])


def test_wan_animate_schema_and_continuation_offset_preserve_sources() -> None:
    schema = WanAnimateToVideo.define_schema()
    assert [input.id for input in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "reference_image",
        "face_video",
        "pose_video",
        "continue_motion_max_frames",
        "background_video",
        "character_mask",
        "continue_motion",
        "video_frame_offset",
    ]
    positive, negative = _conditioning("positive"), _conditioning("negative")
    reference, motion, pose = _image([2.0]), _image([4.0, 5.0, 6.0]), _image([7.0, 8.0, 9.0])
    face = torch.zeros((3, 8, 16, 3))
    face[-1, :, :, :] = torch.tensor([0.25, 0.5, 0.75])
    reference_before, motion_before, pose_before = reference.clone(), motion.clone(), pose.clone()
    output = WanAnimateToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=16,
        height=8,
        length=9,
        batch_size=2,
        continue_motion_max_frames=2,
        video_frame_offset=4,
        reference_image=reference,
        continue_motion=motion,
        pose_video=pose,
        face_video=face,
    )
    assert output["trim_latent"] == 1
    assert output["trim_image"] == 1
    assert output["video_frame_offset"] == 11
    assert output["latent"]["samples"].shape == (2, 16, 4, 1, 2)
    assert output["positive"][0][1]["pose_video_latent"].shape == (1, 16, 3, 1, 2)
    face_pixels = output["positive"][0][1]["face_video_pixels"]
    assert face_pixels.shape == (1, 3, 1, 512, 512)
    assert face_pixels[0, :, 0, 0, 0].tolist() == [-0.5, 0.0, 0.5]
    assert output["negative"][0][1]["face_video_pixels"].unique().tolist() == [-1.0]
    assert positive == _conditioning("positive") and negative == _conditioning("negative")
    assert torch.equal(reference, reference_before)
    assert torch.equal(motion, motion_before)
    assert torch.equal(pose, pose_before)


def test_wan_animate_asymmetric_mask_and_background() -> None:
    character = torch.zeros((1, 8, 16))
    character[:, :, 8:] = 1.0
    vae = VAE()
    output = WanAnimateToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=vae,
        width=16,
        height=8,
        length=5,
        batch_size=1,
        continue_motion_max_frames=1,
        video_frame_offset=0,
        continue_motion=_image([2.0]),
        background_video=_image([7.0, 8.0, 9.0]),
        character_mask=character,
    )
    conditioning = output["positive"][0][1]
    assert vae.images[-1][0, 0, 0, 0] == 2.0
    assert vae.images[-1][1, 0, 0, 0] == 8.0
    assert conditioning["concat_mask"].shape == (1, 4, 3, 1, 2)
    assert not conditioning["concat_mask"][:, :, 0].count_nonzero()
    assert conditioning["concat_mask"][:, :, 1:].count_nonzero()
    assert set(conditioning["concat_mask"].flatten().tolist()) == {0.0, 1.0}


def test_wan_animate2_pose_hold_window_complements_and_error() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    pose = _image([3.0, 4.0])
    output = WanAnimate2ToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=16,
        height=8,
        length=9,
        batch_size=1,
        video_frame_offset=0,
        pose_video=pose,
        pose_start_percent=0.2,
        pose_end_percent=0.7,
        pose_strength=2.0,
        reference_image_strength=0.5,
    )
    entries = output["positive"]
    assert [(item[1]["start_percent"], item[1]["end_percent"]) for item in entries] == [
        (0.2, 0.7),
        (0.0, 0.2),
        (0.7, 1.0),
    ]
    assert entries[0][1]["pose_video_latent"][0, 0, :, 0, 0].tolist() == [3.0, 4.0, 4.0]
    assert entries[0][1]["cross_attn_pose"] is positive[0][0]
    assert entries[0][1]["pose_strength"] == 2.0
    assert entries[0][1]["reference_strength"] == 0.5
    assert "pose_video_latent" not in entries[1][1]
    assert output["negative"][0][1]["cross_attn_pose"] is positive[0][0]
    try:
        WanAnimate2ToVideo.execute(
            positive=_conditioning("positive"),
            negative=_conditioning("negative"),
            vae=VAE(),
            width=16,
            height=8,
            length=1,
            batch_size=1,
            video_frame_offset=2,
            pose_video=_image([1.0, 2.0]),
        )
    except ValueError as error:
        assert (
            str(error)
            == "pose_video has 2 frames but video_frame_offset is 2 -- nothing left to read."
        )
    else:
        raise AssertionError("expected exhausted pose video to fail")


def test_wan_animate2_continuation_and_cache_callback() -> None:
    from dinkster_inference.model_patcher import CoreModelPatcher
    from dinkster_inference.patcher_extension import CallbacksMP

    output = WanAnimate2ToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=VAE(),
        width=16,
        height=8,
        length=5,
        batch_size=1,
        video_frame_offset=1,
        continue_motion=_image([9.0, 10.0]),
    )
    assert output["trim_latent"] == 1
    assert output["trim_image"] == 1
    assert output["video_frame_offset"] == 5
    model = CoreModelPatcher(torch.nn.Identity(), torch.device("cpu"), torch.device("cpu"))
    cached = WanAnimate2Cache.execute(model=model, device="cpu", dtype="int8")["MODEL"]
    cache = cached.model_options["transformer_options"]["animate2_cache"]
    assert (
        cached is not model and cache.store_device == torch.device("cpu") and cache.dtype == "int8"
    )
    cache.slots.append({"pinned": []})
    cached.cleanup()
    assert not cache.slots
    assert cached.get_all_callbacks(CallbacksMP.ON_CLEANUP)
