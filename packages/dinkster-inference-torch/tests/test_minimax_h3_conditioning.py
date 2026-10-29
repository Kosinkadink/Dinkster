from __future__ import annotations

from dataclasses import replace

import torch
from dinkster_inference import (
    MiniMaxH3FL2VARequest,
    MiniMaxH3Keyframe,
    MiniMaxH3KeyframeRole,
    MiniMaxH3PresentationKind,
    MiniMaxH3REF2VARequest,
    MiniMaxH3T2VARequest,
    MiniMaxH3VideoReference,
    PayloadDescriptor,
    PayloadReference,
    normalize_minimax_h3_conditioning,
)
from dinkster_inference_torch.minimax_h3_conditioner import (
    MiniMaxH3ConditionerModel,
    MiniMaxH3VisionModel,
)
from dinkster_inference_torch.minimax_h3_conditioning import (
    MiniMaxH3VisionValue,
    realize_minimax_h3_conditioner_inputs,
)
from dinkster_inference_torch.qwen_image_text import QwenImageLanguageModel


def _payload(name: str) -> PayloadDescriptor:
    return PayloadDescriptor(PayloadReference(name), (1, 32, 32, 3), "float32", "image")


def _reduced_conditioner() -> MiniMaxH3ConditionerModel:
    language = QwenImageLanguageModel.reduced(
        vocab_size=151656,
        hidden_size=16,
        intermediate_size=32,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        qkv_bias=False,
        qk_norm=True,
        final_norm=False,
        rope_theta=5_000_000.0,
        interleaved_mrope=True,
    )
    visual = MiniMaxH3VisionModel.reduced(
        hidden_size=8,
        output_size=16,
        intermediate_size=12,
        heads=2,
        layers=3,
        patch=(2, 16, 16),
        merge_size=2,
        position_embeddings=16,
        deepstack_layers=(0, 1, 2),
    )
    return MiniMaxH3ConditionerModel(language=language, visual=visual)


def test_raw_text_uses_qwen_bpe_without_chat_template_and_empty_uses_pad() -> None:
    inputs = realize_minimax_h3_conditioner_inputs(
        normalize_minimax_h3_conditioning(MiniMaxH3T2VARequest("hello"))
    )
    assert inputs.ids.tolist() == [[14990]]
    assert inputs.position_ids.tolist() == [[0]]
    assert inputs.token_tags.tolist() == [[1]]
    assert inputs.patches is None and inputs.grids is None
    empty = realize_minimax_h3_conditioner_inputs(
        normalize_minimax_h3_conditioning(MiniMaxH3T2VARequest(""))
    )
    assert empty.ids.tolist() == [[151643]]


def test_image_preprocessing_tokens_tags_and_mrope_match_pinned_layout() -> None:
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "go",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),),
        )
    )
    value = MiniMaxH3VisionValue(
        MiniMaxH3PresentationKind.IMAGE_CONTENT,
        torch.full((1, 32, 32, 3), 0.75),
    )
    inputs = realize_minimax_h3_conditioner_inputs(plan, (value,))
    assert inputs.grids is not None and inputs.grids.tolist() == [[1, 4, 4]]
    assert inputs.patches is not None and inputs.patches.shape == (16, 1536)
    torch.testing.assert_close(inputs.patches, torch.full_like(inputs.patches, 0.5))
    assert int(inputs.visual_mask.count_nonzero()) == 4
    content = torch.nonzero(inputs.ids[0] == 151655).flatten()
    assert content.numel() == 4
    start, end = int(content[0]), int(content[-1]) + 1
    assert inputs.position_ids[0, start:end].tolist() == [start] * 4
    assert inputs.position_ids[1, start:end].tolist() == [start, start, start + 1, start + 1]
    assert inputs.position_ids[2, start:end].tolist() == [start, start + 1, start, start + 1]
    assert inputs.token_tags[0, start - 1 : end + 1].tolist() == [0] * 6


def test_two_frame_video_preserves_temporal_patch_order() -> None:
    frames = torch.stack((torch.zeros(32, 32, 3), torch.ones(32, 32, 3)))
    value = MiniMaxH3VisionValue(MiniMaxH3PresentationKind.VIDEO_CONTENT, frames)
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3REF2VARequest(
            "x",
            (
                MiniMaxH3VideoReference(
                    tuple(_payload(f"frame-{index}") for index in range(13)),
                    (0, 12),
                    (0.0, 0.5),
                ),
            ),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(plan, (value,))
    assert inputs.patches is not None
    assert inputs.patches.shape == (16, 1536)
    assert inputs.patches[0, :256].tolist() == [-1.0] * 256
    assert inputs.patches[0, 256:512].tolist() == [1.0] * 256


def test_realization_refuses_missing_vision_values() -> None:
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "x",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),),
        )
    )
    try:
        realize_minimax_h3_conditioner_inputs(plan)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("missing vision realization was accepted")


def test_realized_inputs_execute_through_the_conditioner_boundary() -> None:
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "go",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(
        plan,
        (
            MiniMaxH3VisionValue(
                MiniMaxH3PresentationKind.IMAGE_CONTENT,
                torch.rand(1, 32, 32, 3),
            ),
        ),
    )
    output = _reduced_conditioner().encode(inputs)
    assert output.shape == (1, inputs.ids.shape[1], 16)


def test_realized_inputs_refuse_count_preserving_visual_corruption() -> None:
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "go",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(
        plan,
        (
            MiniMaxH3VisionValue(
                MiniMaxH3PresentationKind.IMAGE_CONTENT,
                torch.rand(1, 32, 32, 3),
            ),
        ),
    )
    visual_indices = torch.nonzero(inputs.visual_mask[0]).flatten()
    shifted_mask = inputs.visual_mask.clone()
    shifted_mask[0, visual_indices[0]] = False
    shifted_mask[0, visual_indices[-1] + 1] = True
    corruptions = (
        {"visual_mask": shifted_mask},
        {"ids": inputs.ids.scatter(1, visual_indices[:1].view(1, 1), 1)},
        {"token_tags": torch.ones_like(inputs.token_tags)},
        {"position_ids": inputs.position_ids + 1},
    )
    for corruption in corruptions:
        try:
            replace(inputs, **corruption)
        except ValueError:
            pass
        else:
            raise AssertionError(f"conditioner accepted corruption: {tuple(corruption)}")


def test_image_patch_rows_follow_spatial_merge_order() -> None:
    pixels = torch.empty(1, 64, 64, 3)
    for row in range(4):
        for column in range(4):
            pixels[:, row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16] = (
                row * 4 + column
            ) / 15
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "x",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(
        plan,
        (MiniMaxH3VisionValue(MiniMaxH3PresentationKind.IMAGE_CONTENT, pixels),),
    )
    assert inputs.patches is not None
    merge_order = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    expected = torch.tensor(tuple(index / 15 * 2 - 1 for index in merge_order))
    torch.testing.assert_close(inputs.patches[:, 0], expected, rtol=0.0, atol=1e-7)


def test_two_vision_spans_use_exact_cumulative_mrope_offsets() -> None:
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3FL2VARequest(
            "x",
            (
                MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, _payload("first")),
                MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.LAST, _payload("last")),
            ),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(
        plan,
        (
            MiniMaxH3VisionValue(
                MiniMaxH3PresentationKind.IMAGE_CONTENT, torch.zeros(1, 64, 64, 3)
            ),
            MiniMaxH3VisionValue(
                MiniMaxH3PresentationKind.IMAGE_CONTENT, torch.zeros(1, 64, 96, 3)
            ),
        ),
    )
    content = torch.nonzero(inputs.visual_mask[0]).flatten().tolist()
    groups = ((content[0], content[3] + 1), (content[4], content[-1] + 1))
    first, second = groups
    first_start, first_end = first
    second_start, second_end = second
    assert inputs.position_ids[:, first_start:first_end].tolist() == [
        [first_start] * 4,
        [first_start, first_start, first_start + 1, first_start + 1],
        [first_start, first_start + 1, first_start, first_start + 1],
    ]
    second_origin = second_start - 2
    assert inputs.position_ids[:, second_start:second_end].tolist() == [
        [second_origin] * 6,
        [second_origin] * 3 + [second_origin + 1] * 3,
        [second_origin, second_origin + 1, second_origin + 2] * 2,
    ]


def test_video_half_inputs_normalize_in_float32() -> None:
    frames = torch.zeros(2, 64, 64, 3, dtype=torch.float16)
    plan = normalize_minimax_h3_conditioning(
        MiniMaxH3REF2VARequest(
            "x",
            (
                MiniMaxH3VideoReference(
                    tuple(_payload(f"frame-{index}") for index in range(13)),
                    (0, 12),
                    (0.0, 0.5),
                ),
            ),
        )
    )
    inputs = realize_minimax_h3_conditioner_inputs(
        plan,
        (MiniMaxH3VisionValue(MiniMaxH3PresentationKind.VIDEO_CONTENT, frames),),
    )
    assert inputs.patches is not None and inputs.patches.dtype == torch.float32
