"""Parity proofs for declared and realized MiniMax H3 packed token geometry."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    MiniMaxH3DiTPayloadKind,
    MiniMaxH3KeyframeRole,
    MiniMaxH3ReferenceTokenGeometry,
    MiniMaxH3VideoLatentGeometry,
    plan_minimax_h3_token_layout,
)
from dinkster_inference_torch import minimax_h3_dit as dit_module
from dinkster_inference_torch.minimax_h3_dit import (
    MiniMaxH3DiTConditioning,
    MiniMaxH3KeyframeLatent,
    MiniMaxH3ReferenceKind,
    MiniMaxH3ReferenceLatents,
)


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    text_tokens: int
    target_video: MiniMaxH3VideoLatentGeometry
    target_audio_temporal: int
    keyframes: tuple[tuple[MiniMaxH3KeyframeRole, int], ...] = ()
    references: tuple[MiniMaxH3ReferenceTokenGeometry, ...] = ()


def _video_tensor(geometry: MiniMaxH3VideoLatentGeometry) -> torch.Tensor:
    return torch.empty(
        1,
        MINIMAX_H3_CONFIG.video_latent_channels,
        geometry.temporal,
        geometry.height,
        geometry.width,
    )


def _reference_latents(
    reference: MiniMaxH3ReferenceTokenGeometry,
) -> MiniMaxH3ReferenceLatents:
    video = None
    if reference.video is not None:
        video = dit_module._pad_video(  # pyright: ignore[reportPrivateUsage]
            _video_tensor(reference.video), MINIMAX_H3_CONFIG.patch
        )
    audio = None
    if reference.audio_temporal is not None:
        audio = torch.empty(
            1,
            MINIMAX_H3_CONFIG.audio_latent_channels,
            MINIMAX_H3_CONFIG.audio_content_channels,
            reference.audio_temporal,
        )
    return MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind(reference.kind.value), video, audio)


def _declared_kind(identity: str, modality: str, role: str) -> str:
    if identity == "text":
        return "text"
    if role == "condition":
        return "condition"
    if role == "reference":
        return f"reference_{modality}"
    return modality


_T2_NON_SQUARE = MiniMaxH3VideoLatentGeometry(2, 4, 6)
_T7_NON_SQUARE = MiniMaxH3VideoLatentGeometry(7, 6, 4)
_IMAGE = MiniMaxH3ReferenceTokenGeometry(
    MiniMaxH3DiTPayloadKind.IMAGE,
    MiniMaxH3VideoLatentGeometry(1, 3, 5),
)
_AUDIO = MiniMaxH3ReferenceTokenGeometry(
    MiniMaxH3DiTPayloadKind.AUDIO,
    audio_temporal=3,
)
_VIDEO_ONLY = MiniMaxH3ReferenceTokenGeometry(
    MiniMaxH3DiTPayloadKind.VIDEO,
    MiniMaxH3VideoLatentGeometry(3, 5, 3),
)
_VIDEO_WITH_AUDIO = MiniMaxH3ReferenceTokenGeometry(
    MiniMaxH3DiTPayloadKind.VIDEO,
    MiniMaxH3VideoLatentGeometry(4, 3, 5),
    6,
)


@pytest.mark.parametrize(
    "scenario",
    (
        _Scenario("text-only-short", 3, _T2_NON_SQUARE, 8),
        _Scenario(
            "first-keyframe",
            11,
            _T7_NON_SQUARE,
            37,
            ((MiniMaxH3KeyframeRole.FIRST, 0),),
        ),
        _Scenario(
            "last-keyframe",
            5,
            _T2_NON_SQUARE,
            8,
            ((MiniMaxH3KeyframeRole.LAST, 4),),
        ),
        _Scenario(
            "two-keyframes",
            7,
            _T7_NON_SQUARE,
            37,
            (
                (MiniMaxH3KeyframeRole.FIRST, 0),
                (MiniMaxH3KeyframeRole.LAST, 21),
            ),
        ),
        _Scenario("image-reference", 3, _T2_NON_SQUARE, 8, references=(_IMAGE,)),
        _Scenario("audio-reference", 5, _T7_NON_SQUARE, 37, references=(_AUDIO,)),
        _Scenario(
            "video-reference-without-audio",
            7,
            _T2_NON_SQUARE,
            8,
            references=(_VIDEO_ONLY,),
        ),
        _Scenario(
            "video-reference-with-audio",
            11,
            _T7_NON_SQUARE,
            37,
            references=(_VIDEO_WITH_AUDIO,),
        ),
        _Scenario(
            "mixed-reference-order",
            5,
            _T2_NON_SQUARE,
            8,
            references=(_VIDEO_WITH_AUDIO, _IMAGE, _AUDIO, _VIDEO_ONLY),
        ),
    ),
    ids=lambda scenario: scenario.name,
)
def test_declared_layout_matches_runtime_packed_boundaries(scenario: _Scenario) -> None:
    target_video = dit_module._pad_video(  # pyright: ignore[reportPrivateUsage]
        _video_tensor(scenario.target_video), MINIMAX_H3_CONFIG.patch
    )
    target_audio = torch.empty(
        1,
        MINIMAX_H3_CONFIG.audio_latent_channels,
        MINIMAX_H3_CONFIG.audio_content_channels,
        scenario.target_audio_temporal,
    )
    keyframes = tuple(
        MiniMaxH3KeyframeLatent(
            resolved_frame_index,
            torch.empty(
                1,
                MINIMAX_H3_CONFIG.video_latent_channels,
                1,
                target_video.shape[3],
                target_video.shape[4],
            ),
        )
        for _, resolved_frame_index in scenario.keyframes
    )
    conditioning = MiniMaxH3DiTConditioning(
        keyframes=keyframes,
        references=tuple(_reference_latents(reference) for reference in scenario.references),
    )
    packed = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        scenario.text_tokens, target_video, target_audio, conditioning
    )
    declared = plan_minimax_h3_token_layout(
        text_tokens=scenario.text_tokens,
        target_video=scenario.target_video,
        target_audio_temporal=scenario.target_audio_temporal,
        keyframes=tuple(role for role, _ in scenario.keyframes),
        references=scenario.references,
    )
    declared_segments = tuple(
        (
            segment.start,
            segment.stop,
            _declared_kind(segment.identity, segment.modality, segment.role),
        )
        for segment in declared.layout.segments
    )

    assert declared.layout.valid_rows == packed.sequence_length
    assert declared_segments == packed.segments
