from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from dinkster_inference import (
    MINIMAX_H3_AUDIO_MASK_MAPPING,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    LatentMaskMapping,
    MultiStreamLatent,
)

ROOT = Path(__file__).resolve().parents[3]
for source in sorted((ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(source))

native = importlib.import_module("dinkster_compat_comfy.native")
native_arm = importlib.import_module("dinkster_compat_comfy.native_arm")


class MaskRuntime:
    def __init__(self, mapping: LatentMaskMapping) -> None:
        self.mapping = mapping

    @property
    def latent_mask_mapping(self) -> LatentMaskMapping:
        return self.mapping


def h3_latent() -> dict[str, object]:
    streams = MultiStreamLatent.from_pairs(
        (
            ("video", torch.zeros(1, 24, 7, 2, 3)),
            ("audio", torch.zeros(1, 32, 2, 40)),
        )
    )
    return {"samples": streams, "metadata": "preserved"}


def test_mask_nodes_compose_exact_h3_roles_and_preserve_latent_metadata() -> None:
    latent = h3_latent()
    frame_mask = native.FrameRangeMask.execute(
        width=48,
        height=32,
        frames=22,
        ranges="0, 1:5",
    )["mask"]
    video_result = native_arm.NativeSetLatentMaskFromFrames.execute(
        latent=latent,
        vae=MaskRuntime(MINIMAX_H3_VIDEO_MASK_MAPPING),
        mask=frame_mask,
        spatial_reduction="max",
        temporal_reduction="max",
        operation="replace",
    )["latent"]
    audio_result = native_arm.NativeSetLatentMaskFromTimeRanges.execute(
        latent=video_result,
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
        ranges="0.25:0.5",
        selected=0.4,
        unselected=0.1,
        operation="replace",
    )["latent"]

    assert audio_result["samples"] is latent["samples"]
    assert audio_result["metadata"] == "preserved"
    masks = audio_result["noise_mask"]
    assert type(masks) is MultiStreamLatent
    assert masks.roles == ("video", "audio")
    video = masks.by_role("video")
    audio = masks.by_role("audio")
    assert tuple(video.shape) == (1, 24, 7, 2, 3)
    assert torch.all(video[:, :, :2] == 1.0)
    assert torch.all(video[:, :, 2:] == 0.0)
    assert tuple(audio.shape) == (1, 32, 2, 40)
    assert torch.all(audio[..., :10] == 0.1)
    assert torch.all(audio[..., 10:20] == 0.4)
    assert torch.all(audio[..., 20:] == 0.1)


def test_frame_mask_operations_preserve_reversed_roles_and_singular_latents() -> None:
    video = torch.zeros(1, 24, 2, 1, 1)
    audio = torch.zeros(1, 32, 2, 4)
    streams = MultiStreamLatent.from_pairs((("audio", audio), ("video", video)))
    audio_mask = torch.zeros_like(audio)
    audio_mask[:, 1, 1, 2] = 0.75
    masks = MultiStreamLatent.from_pairs(
        (
            ("audio", audio_mask),
            ("video", torch.full_like(video, 0.4)),
        )
    )
    content = torch.full((5, 16, 16), 0.5)

    expected = {"replace": 0.5, "max": 0.5, "min": 0.4, "multiply": 0.2}
    for operation, value in expected.items():
        result = native_arm.NativeSetLatentMaskFromFrames.execute(
            latent={"samples": streams, "noise_mask": masks, "metadata": "preserved"},
            vae=MaskRuntime(MINIMAX_H3_VIDEO_MASK_MAPPING),
            mask=content,
            spatial_reduction="mean",
            temporal_reduction="mean",
            operation=operation,
        )["latent"]
        result_masks = result["noise_mask"]
        assert type(result_masks) is MultiStreamLatent
        assert result_masks.roles == ("audio", "video")
        assert torch.equal(result_masks.by_role("audio"), audio_mask)
        assert torch.allclose(
            result_masks.by_role("video"),
            torch.full_like(video, value),
        )
        assert result["metadata"] == "preserved"

    singular = native_arm.NativeSetLatentMaskFromFrames.execute(
        latent={"samples": video, "metadata": "preserved"},
        vae=MaskRuntime(MINIMAX_H3_VIDEO_MASK_MAPPING),
        mask=content,
        spatial_reduction="mean",
        temporal_reduction="mean",
        operation="replace",
    )["latent"]
    assert type(singular["noise_mask"]) is torch.Tensor
    assert tuple(singular["noise_mask"].shape) == tuple(video.shape)
    assert singular["metadata"] == "preserved"


def test_inspect_latent_mask_reports_ranges_values_and_missing_roles() -> None:
    latent = h3_latent()
    absent = native_arm.NativeInspectLatentMask.execute(
        latent=latent,
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
    )
    assert tuple(absent["mask"].shape) == (1, 2, 40)
    assert torch.all(absent["mask"] == 1.0)
    assert "no noise mask on the latent; audio defaults to generate" in absent["report"]

    video_only = dict(latent)
    video_only["noise_mask"] = MultiStreamLatent.from_pairs(
        (("video", torch.zeros(1, 24, 7, 2, 3)),)
    )
    missing = native_arm.NativeInspectLatentMask.execute(
        latent=video_only,
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
    )
    assert "noise mask has no audio role; it defaults to generate" in missing["report"]

    varied_audio = torch.full((1, 32, 2, 40), 0.1)
    varied_audio[..., 10:20] = 0.4
    varied_audio[:, 0, 0, 10:20] = 0.6
    masked = dict(latent)
    masked["noise_mask"] = MultiStreamLatent.from_pairs(
        (
            ("video", torch.ones(1, 24, 7, 2, 3)),
            ("audio", varied_audio),
        )
    )
    inspected = native_arm.NativeInspectLatentMask.execute(
        latent=masked,
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
    )
    assert "0-0.25s = 0.1 (soft)" in inspected["report"]
    assert "0.25-0.5s = 0.6 (soft)" in inspected["report"]
    assert "warning: values vary within timeline frames" in inspected["report"]


def test_inspect_latent_mask_preserves_soft_values_and_audio_boundaries() -> None:
    video = torch.zeros(1, 24, 1, 1, 1)
    audio = torch.zeros(1, 32, 2, 3)
    streams = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    audio_mask = torch.empty_like(audio)
    audio_mask[..., 0] = 0.0004
    audio_mask[..., 1] = 0.4004
    audio_mask[..., 2] = 0.40049
    masks = MultiStreamLatent.from_pairs((("video", torch.ones_like(video)), ("audio", audio_mask)))

    inspected = native_arm.NativeInspectLatentMask.execute(
        latent={"samples": streams, "noise_mask": masks},
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
    )

    assert "audio: 3 latent frames, 0.075s" in inspected["report"]
    assert "0-0.025s = 0.0004 (soft)" in inspected["report"]
    assert "0.025-0.05s = 0.4004 (soft)" in inspected["report"]
    assert "0.05-0.075s = 0.40049 (soft)" in inspected["report"]


def test_inspect_latent_mask_reads_values_outside_the_first_audio_channel() -> None:
    video = torch.zeros(1, 24, 1, 1, 1)
    audio = torch.zeros(1, 32, 2, 3)
    streams = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    audio_mask = torch.zeros_like(audio)
    audio_mask[:, 1, 1, 1] = 0.75

    inspected = native_arm.NativeInspectLatentMask.execute(
        latent={
            "samples": streams,
            "noise_mask": MultiStreamLatent.from_pairs(
                (("video", torch.ones_like(video)), ("audio", audio_mask))
            ),
        },
        vae=MaskRuntime(MINIMAX_H3_AUDIO_MASK_MAPPING),
    )

    assert "0.025-0.05s = 0.75 (soft)" in inspected["report"]
    assert "warning: values vary within timeline frames" in inspected["report"]
