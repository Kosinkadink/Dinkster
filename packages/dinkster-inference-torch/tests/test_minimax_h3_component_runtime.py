from __future__ import annotations

import pytest
import torch
from dinkster_inference import (
    MiniMaxH3AudioReference,
    MiniMaxH3FL2VARequest,
    MiniMaxH3ImageReference,
    MiniMaxH3Keyframe,
    MiniMaxH3KeyframeRole,
    MiniMaxH3REF2VARequest,
    MiniMaxH3T2VARequest,
    PayloadDescriptor,
    PayloadReference,
)
from dinkster_inference_torch import (
    MiniMaxH3ConditionerRuntime,
    MiniMaxH3RuntimeError,
    MiniMaxH3VideoVaeRuntime,
    empty_minimax_h3_av,
)
from dinkster_inference_torch.minimax_h3_conditioning import MiniMaxH3ConditionerInputs


class _Conditioner:
    def encode(self, inputs: MiniMaxH3ConditionerInputs) -> torch.Tensor:
        return torch.zeros((1, inputs.ids.shape[1], 16), device=inputs.ids.device)


class _VideoVae:
    @staticmethod
    def encode_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
        batch, _, frames, height, width = input_shape
        temporal = 1 if frames == 1 else 5 * ((frames + 16) // 17) - 3
        return batch, 24, temporal, (height + 15) // 16, (width + 15) // 16

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            self.encode_output_shape(tuple(content.shape)),
            device=content.device,
            dtype=content.dtype,
        )


def _target():  # noqa: ANN202
    return empty_minimax_h3_av(
        width=32,
        height=32,
        frame_count=5,
        device="cpu",
        dtype=torch.float32,
    )


def _descriptor(name: str, shape: tuple[int, ...]) -> PayloadDescriptor:
    return PayloadDescriptor(
        PayloadReference(name),
        shape,
        "float32",
        "worker:minimax-h3",
    )


def _runtime(
    video_vae: MiniMaxH3VideoVaeRuntime | None = None,
) -> MiniMaxH3ConditionerRuntime:
    return MiniMaxH3ConditionerRuntime(
        _Conditioner(),  # type: ignore[arg-type]
        video_vae,
        runtime_identity="native:dinkster.minimax_h3:" + "1" * 64,
    )


def test_t2va_conditioning_needs_no_codec_components() -> None:
    prepared = _runtime().condition(
        MiniMaxH3T2VARequest("text only"),
        target=_target(),
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    assert prepared.task.name == "T2VA"


def test_fl2va_conditioning_requires_video_vae_component() -> None:
    image = torch.zeros((1, 32, 32, 3))
    descriptor = _descriptor("first", tuple(image.shape))
    request = MiniMaxH3FL2VARequest(
        "animate",
        (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, descriptor),),
    )
    with pytest.raises(
        MiniMaxH3RuntimeError,
        match="^MiniMax H3 conditioning request requires a video VAE component$",
    ):
        _runtime().condition(
            request,
            target=_target(),
            frame_count=5,
            payloads={"first": image},
            cancelled=lambda: False,
        )


def test_ref2va_audio_conditioning_requires_audio_vae_component() -> None:
    waveform = torch.zeros((1, 2, 1600))
    descriptor = _descriptor("audio", tuple(waveform.shape))
    request = MiniMaxH3REF2VARequest(
        "continue",
        (MiniMaxH3AudioReference(descriptor, 32_000),),
    )
    with pytest.raises(
        MiniMaxH3RuntimeError,
        match="^MiniMax H3 conditioning request requires an audio VAE component$",
    ):
        _runtime().condition(
            request,
            target=_target(),
            frame_count=5,
            payloads={"audio": waveform},
            cancelled=lambda: False,
        )


def test_ref2va_image_conditioning_needs_only_video_vae_component() -> None:
    image = torch.zeros((1, 32, 32, 3))
    descriptor = _descriptor("image", tuple(image.shape))
    video_vae = MiniMaxH3VideoVaeRuntime(
        _VideoVae(),  # type: ignore[arg-type]
        runtime_identity="native:dinkster.minimax_h3:" + "2" * 64,
        compute_dtype=torch.float32,
    )
    prepared = _runtime(video_vae).condition(
        MiniMaxH3REF2VARequest(
            "continue",
            (MiniMaxH3ImageReference(descriptor),),
        ),
        target=_target(),
        frame_count=5,
        payloads={"image": image},
        cancelled=lambda: False,
    )
    assert prepared.task.name == "REF2VA"
