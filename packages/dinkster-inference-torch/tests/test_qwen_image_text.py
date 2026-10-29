"""Reduced CPU proofs for the Qwen Image Qwen2.5-VL text source."""

from __future__ import annotations

from typing import Any

import dinkster_inference_torch.qwen_image_text as qwen_image_text_module
import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference.qwen_image_text import (
    QWEN_IMAGE_TEXT_CONFIG,
    qwen_image_text_layout,
)
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.qwen_image_text import (
    QwenImageLanguageAttention,
    QwenImageLanguageModel,
    QwenImageTextModel,
    QwenImageVisionAttention,
    QwenImageVisionTransformer,
    _apply_language_rope,  # pyright: ignore[reportPrivateUsage]
    _language_rope,  # pyright: ignore[reportPrivateUsage]
    prepare_qwen_image_vision,
    qwen_image_mrope_position_ids,
    resize_qwen_image_content,
)


def test_full_meta_state_layout_exactly_matches_s4a() -> None:
    with torch.device("meta"):
        model = QwenImageTextModel()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == dict(qwen_image_text_layout())
    assert len(actual) == 728
    state = {key: torch.empty(shape, device="meta") for key, shape in actual.items()}
    model.load_state_dict(state, strict=True, assign=True)


def test_language_attention_uses_gqa_mrope_mask_and_injected_kernel() -> None:
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    attention = QwenImageLanguageAttention(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        attention_kernel=spy,
    )
    hidden = torch.randn(1, 5, 32)
    position_ids = torch.tensor(
        (
            (0, 1, 2, 3, 4),
            (0, 1, 2, 2, 3),
            (0, 1, 2, 3, 3),
        )
    )
    mask = torch.zeros(1, 1, 5, 5)
    output = attention(hidden, mask, position_ids, rope_theta=1_000_000.0, rope_dims=(2, 1, 1))
    assert output.shape == hidden.shape
    assert len(spy.calls) == 1
    assert spy.calls[0]["q_shape"] == (1, 4, 5, 8)
    assert spy.calls[0]["k_shape"] == (1, 2, 5, 8)
    assert spy.calls[0]["v_shape"] == (1, 2, 5, 8)
    assert spy.calls[0]["mask"] is mask
    assert spy.calls[0]["enable_gqa"] is True
    assert_kernel_is_not_model_state(attention, spy)


def test_language_mrope_values_match_pinned_upstream_three_axis_sections() -> None:
    head_dim = 8
    rope_dims = (2, 1, 1)
    theta = 1_000_000.0
    position_ids = torch.tensor(
        (
            (0, 1, 2, 3, 4),
            (0, 1, 2, 2, 3),
            (0, 1, 2, 3, 3),
        )
    )
    numerator = torch.arange(0, head_dim, 2).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    frequencies = (inverse[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    sections = rope_dims * 2
    expected_cosine = torch.cat(
        [part[index % 3] for index, part in enumerate(embedding.cos().split(sections, -1))],
        dim=-1,
    ).unsqueeze(0)
    expected_sine = torch.cat(
        [part[index % 3] for index, part in enumerate(embedding.sin().split(sections, -1))],
        dim=-1,
    ).unsqueeze(0)
    half = expected_sine.shape[-1] // 2
    expected = (
        expected_cosine,
        expected_sine[..., :half],
        -expected_sine[..., half:],
    )
    actual = _language_rope(
        position_ids,
        head_dim=head_dim,
        theta=theta,
        rope_dims=rope_dims,
        device=torch.device("cpu"),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)


def test_language_rope_apply_is_bit_identical_to_reference_fused_rotation() -> None:
    """Executed pin of the reference rotation arithmetic
    (comfy/text_encoders/llama.py apply_rope @ b78cec87): the rotated
    halves accumulate through addcmul's single rounding. A revert to
    separate mul+add drifts by one ulp on a fraction of elements and
    breaks Krea 2 conditioning bit-parity with ComfyUI."""
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 4, 5, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    position_ids = torch.arange(5).expand(3, 5)
    frequencies = _language_rope(
        position_ids,
        head_dim=8,
        theta=1_000_000.0,
        rope_dims=(2, 1, 1),
        device=torch.device("cpu"),
    )
    cosine, sine, negative_sine = frequencies

    def reference(value: torch.Tensor) -> torch.Tensor:
        embedded = value * cosine
        half = embedded.shape[-1] // 2
        embedded[..., :half].addcmul_(value[..., half:], negative_sine)
        embedded[..., half:].addcmul_(value[..., :half], sine)
        return embedded

    query_before, key_before = query.clone(), key.clone()
    rotated_query, rotated_key = _apply_language_rope(query, key, frequencies)
    torch.testing.assert_close(rotated_query, reference(query), rtol=0.0, atol=0.0)
    torch.testing.assert_close(rotated_key, reference(key), rtol=0.0, atol=0.0)
    torch.testing.assert_close(query, query_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(key, key_before, rtol=0.0, atol=0.0)


def test_language_model_builds_causal_padding_mask_before_attention() -> None:
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    model = QwenImageLanguageModel.reduced(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        rope_dims=(2, 1, 1),
        attention_kernel=spy,
    )
    ids = torch.tensor(((2, 3, 4, 0),))
    attention_mask = torch.tensor(((1, 1, 1, 0),))
    output = model(ids, attention_mask)
    assert output.shape == (1, 4, 32)
    mask = spy.calls[0]["mask"]
    assert isinstance(mask, torch.Tensor)
    assert tuple(mask.shape) == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1] < -1e20
    assert mask[0, 0, 3, 3] < -1e20
    assert mask[0, 0, 2, 1] == 0


def test_language_model_admits_position_limit_and_refuses_longer_sequences() -> None:
    model = QwenImageLanguageModel.reduced(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        max_position_embeddings=3,
        architecture="test Qwen tower",
    )
    assert model(torch.tensor(((1, 2, 3),))).shape == (1, 3, 8)
    with pytest.raises(ValueError, match="test Qwen tower received 4 tokens; maximum is 3"):
        model(torch.tensor(((1, 2, 3, 4),)))
    with pytest.raises(ValueError, match="test Qwen tower received 4 tokens; maximum is 3"):
        model.tapped_states(torch.tensor(((1, 2, 3, 4),)), tap_layers=(0,))


def test_mrope_position_ids_match_text_and_image_grid_order() -> None:
    ids = qwen_image_mrope_position_ids(
        sequence_length=11,
        image_start=3,
        image_length=6,
        image_grid_thw=torch.tensor(((1, 4, 6),)),
        attention_mask=torch.tensor(((1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1),)),
        device=torch.device("cpu"),
    )
    assert ids.shape == (3, 11)
    assert ids[:, :3].tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert ids[0, 3:9].tolist() == [3] * 6
    assert ids[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
    assert ids[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
    assert ids[:, 9:].tolist() == [[1, 6], [1, 6], [1, 6]]


def test_edit_image_preprocessing_matches_qwen2vl_layout_golden() -> None:
    content = torch.linspace(0, 1, 3 * 37 * 61).reshape(1, 3, 37, 61)
    resized = resize_qwen_image_content(content, target_pixels=384 * 384)
    patches, grid = prepare_qwen_image_vision(content, target_pixels=384 * 384)
    assert resized.shape == (1, 3, 299, 493)
    assert patches.shape == (792, 1176)
    assert grid.tolist() == [[1, 22, 36]]
    selected = patches[[0, 1, 100, -1], [0, 1175, 555, 42]]
    assert selected.tolist() == pytest.approx(
        [
            -1.7922625541687012,
            0.9718279838562012,
            -0.32475200295448303,
            -0.5858546495437622,
        ]
    )
    assert float(patches.sum()) == pytest.approx(164888.515625)
    assert float(patches.square().sum()) == pytest.approx(1292405.875)


def test_vision_attention_segments_windows_through_injected_kernel() -> None:
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    attention = QwenImageVisionAttention(hidden_size=8, num_heads=2, attention_kernel=spy)
    hidden = torch.randn(6, 8)
    position_embeddings = (torch.ones(6, 4), torch.zeros(6, 4))
    output = attention(hidden, position_embeddings, torch.tensor((0, 2, 6)))
    assert output.shape == hidden.shape
    assert [call["q_shape"] for call in spy.calls] == [(1, 2, 2, 4), (1, 2, 4, 4)]
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(attention, spy)


def test_vision_attention_uses_kitchen_split_half_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    attention = QwenImageVisionAttention(hidden_size=8, num_heads=2)
    hidden = torch.randn(6, 8)
    angles = torch.randn(6, 2)
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []
    tables: list[torch.Tensor] = []

    def apply(
        query: torch.Tensor, key: torch.Tensor, matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((query.shape, key.shape, matrix.shape))
        tables.append(matrix.clone())

        def rotate(value: torch.Tensor) -> torch.Tensor:
            pairs = value.reshape(*value.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
            output = matrix[..., 0] * pairs[..., 0] + matrix[..., 1] * pairs[..., 1]
            return output.movedim(-1, -2).reshape_as(value)

        return rotate(query), rotate(key)

    monkeypatch.setattr(dinkster_kitchen, "apply_rope_split_half", apply)

    output = attention(hidden, (cosine, sine), torch.tensor((0, 6)))

    assert output.shape == hidden.shape
    assert calls == [(torch.Size((1, 6, 2, 4)),) * 2 + (torch.Size((1, 6, 1, 2, 2, 2)),)]
    expected = torch.stack(
        (cosine[:, :2], -sine[:, 2:], sine[:, :2], cosine[:, 2:]), dim=-1
    ).reshape(1, 6, 1, 2, 2, 2)
    assert torch.equal(tables[0], expected)


def test_reduced_vision_patches_windows_merges_and_restores_order() -> None:
    model = QwenImageVisionTransformer.reduced(
        hidden_size=8,
        output_size=12,
        intermediate_size=16,
        num_heads=2,
        num_layers=2,
        patch=(2, 2, 2),
        spatial_merge_size=2,
        window_size=8,
        full_attention_blocks=(1,),
    )
    patches = torch.randn(16, 3 * 2 * 2 * 2)
    grid = torch.tensor(((1, 4, 4),))
    output = model(patches, grid)
    assert output.shape == (4, 12)
    assert model.last_attention_segments == ((0, 16), (0, 16))


def test_patch_projection_uses_operations_and_matches_source_convolution() -> None:
    model = QwenImageVisionTransformer.reduced(
        hidden_size=8,
        output_size=12,
        intermediate_size=16,
        num_heads=2,
        num_layers=0,
        patch=(2, 2, 2),
        spatial_merge_size=2,
        window_size=8,
        full_attention_blocks=(),
    )
    patches = torch.randn(5, 24)
    source_weight = torch.randn(8, 3, 2, 2, 2)
    model.patch_embed.load_state_dict({"proj.weight": source_weight}, strict=True)
    expected = torch.nn.functional.conv3d(
        patches.reshape(5, 3, 2, 2, 2), source_weight, stride=(2, 2, 2)
    ).reshape(5, 8)
    torch.testing.assert_close(model.patch_embed(patches), expected, rtol=1e-5, atol=1e-6)
    assert model.patch_embed.state_dict()["proj.weight"].shape == source_weight.shape


class _RecordingLanguage(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(152064, 4)
        self.ids: torch.Tensor | None = None
        self.embeds: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        self.ids = ids
        return self.embed_tokens(ids)

    def validate_sequence_length(self, length: int) -> None:
        assert length > 0

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.embeds = embeds
        self.mask = attention_mask
        self.positions = position_ids
        return torch.arange(embeds.shape[1], dtype=torch.float32).reshape(1, -1, 1)


class _RecordingVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(self, patches: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        self.called = True
        assert patches.shape == (8, 1176)
        assert grid.tolist() == [[1, 2, 4]]
        return torch.arange(8, dtype=torch.float32).reshape(2, 4)


def test_text_model_substitutes_image_then_applies_s4a_slice_and_mask() -> None:
    language = _RecordingLanguage()
    model = QwenImageTextModel(language=language, visual=_RecordingVision())  # type: ignore[arg-type]
    ids = torch.tensor(((151644, 20, 151644, 872, 198, 151655, 30, 31),))
    mask = torch.tensor(((1, 1, 1, 1, 1, 0, 0, 0),))
    patches = torch.randn(8, 1176)
    grid = torch.tensor(((1, 2, 4),))
    output, selected_mask = model(ids, mask, image_patches=patches, image_grid_thw=grid)
    assert language.embeds is not None
    assert language.embeds.shape == (1, 9, 4)
    assert language.embeds[0, 5:7].tolist() == torch.arange(8).reshape(2, 4).tolist()
    assert language.mask is not None
    assert language.mask.tolist() == [[1, 1, 1, 1, 1, 1, 1, 0, 0]]
    assert output.flatten().tolist() == [5.0, 6.0, 7.0, 8.0]
    assert selected_mask is not None
    assert selected_mask.tolist() == [[1, 1, 0, 0]]


def test_text_model_counts_expanded_image_rows_against_position_limit() -> None:
    class Vision(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, patches: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
            self.called = True
            assert patches.shape == (8, 1176)
            assert grid.tolist() == [[1, 2, 4]]
            return torch.zeros(2, 8)

    language = QwenImageLanguageModel.reduced(
        vocab_size=152064,
        hidden_size=8,
        intermediate_size=16,
        num_layers=0,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        final_norm=False,
        max_position_embeddings=8,
        architecture="Qwen Image test tower",
    )
    visual = Vision()
    model = QwenImageTextModel(language=language, visual=visual)  # type: ignore[arg-type]
    embedding_calls: list[bool] = []
    handle = language.embed_tokens.register_forward_pre_hook(
        lambda _module, _args: embedding_calls.append(True)
    )
    ids = torch.tensor(((151644, 20, 151644, 872, 198, 151655, 30, 31),))
    try:
        with pytest.raises(
            ValueError, match="Qwen Image test tower received 9 tokens; maximum is 8"
        ):
            model(
                ids,
                image_patches=torch.randn(8, 1176),
                image_grid_thw=torch.tensor(((1, 2, 4),)),
            )
    finally:
        handle.remove()
    assert model.active_image_features is None
    assert embedding_calls == []
    assert visual.called is False


def test_text_model_refuses_text_only_overflow_before_output_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_selection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("output selection ran before length validation")

    language = QwenImageLanguageModel.reduced(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=0,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        final_norm=False,
        max_position_embeddings=3,
        architecture="Qwen Image test tower",
    )
    model = QwenImageTextModel(language=language)
    monkeypatch.setattr(
        qwen_image_text_module,
        "select_qwen_image_output",
        fail_selection,
    )
    with pytest.raises(ValueError, match="Qwen Image test tower received 4 tokens; maximum is 3"):
        model(torch.tensor(((1, 2, 3, 4),)))


def test_text_model_substitutes_three_images_in_placeholder_order() -> None:
    language = _RecordingLanguage()

    class Vision(torch.nn.Module):
        def forward(self, patches: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
            return patches[:, :4]

    model = QwenImageTextModel(language=language, visual=Vision())  # type: ignore[arg-type]
    ids = torch.tensor(((151644, 20, 151644, 872, 198, 151655, 7, 151655, 8, 151655, 9),))
    patches = tuple(torch.full((4, 1176), float(index)) for index in (1, 2, 3))
    grids = tuple(torch.tensor(((1, 2, 2),)) for _ in patches)
    model(ids, image_patches=patches, image_grid_thw=grids)
    assert language.embeds is not None
    assert language.embeds.shape == (1, 20, 4)
    assert language.embeds[0, 5:9].tolist() == [[1.0] * 4] * 4
    assert language.embeds[0, 10:14].tolist() == [[2.0] * 4] * 4
    assert language.embeds[0, 15:19].tolist() == [[3.0] * 4] * 4
    assert language.positions is not None
    assert language.positions.shape == (3, 20)


@pytest.mark.parametrize(
    ("ids", "mask", "patches", "grid", "match"),
    (
        (torch.tensor((1, 2)), None, None, None, "rank 2"),
        (torch.tensor(((1, 2),)), torch.ones(1, 3), None, None, "mask must match"),
        (
            torch.tensor(((151655, 151655),)),
            None,
            torch.empty(1, 1176),
            torch.tensor(((1, 2, 2),)),
            "matched image placeholders",
        ),
        (
            torch.tensor(((1, 2),)),
            None,
            torch.empty(1, 1176),
            torch.tensor(((1, 2, 2),)),
            "matched image placeholders",
        ),
        (
            torch.tensor(((151655,),)),
            None,
            torch.empty(4, 1176),
            torch.tensor(((1.0, 2.0, 2.0),)),
            "integer dtype",
        ),
        (
            torch.tensor(((151655,),)),
            None,
            torch.empty(8, 1176),
            torch.tensor(((2, 2, 2),)),
            "one positive image",
        ),
        (
            torch.tensor(((151655,),)),
            None,
            torch.empty(6, 1176),
            torch.tensor(((1, 2, 2),)),
            "account for every patch",
        ),
        (
            torch.tensor(((151655,),)),
            None,
            torch.empty(6, 1176),
            torch.tensor(((1, 2, 3),)),
            "divide by the spatial merge",
        ),
        (torch.tensor(((151655,),)), None, None, None, "requires image patches"),
        (torch.tensor(((151655,),)), None, torch.empty(1, 1176), None, "provided together"),
    ),
)
def test_text_model_refuses_malformed_geometry_before_model_work(
    ids: torch.Tensor,
    mask: torch.Tensor | None,
    patches: torch.Tensor | None,
    grid: torch.Tensor | None,
    match: str,
) -> None:
    language = _RecordingLanguage()
    visual = _RecordingVision()
    model = QwenImageTextModel(language=language, visual=visual)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        model(ids, mask, image_patches=patches, image_grid_thw=grid)
    assert language.ids is None
    assert visual.called is False


def test_temporary_image_features_release_after_language_failure() -> None:
    language = _RecordingLanguage()

    def fail(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("language failed")

    language.forward_embeds = fail  # type: ignore[method-assign]
    model = QwenImageTextModel(language=language, visual=_RecordingVision())  # type: ignore[arg-type]
    ids = torch.tensor(((151644, 1, 151644, 872, 198, 151655),))
    with pytest.raises(RuntimeError, match="language failed"):
        model(
            ids,
            torch.ones_like(ids),
            image_patches=torch.randn(8, 1176),
            image_grid_thw=torch.tensor(((1, 2, 4),)),
        )
    assert model.active_image_features is None


def test_training_models_are_public_but_vision_stays_direct_import_only() -> None:
    import dinkster_inference_torch

    assert dinkster_inference_torch.QwenImageLanguageModel is QwenImageLanguageModel
    assert dinkster_inference_torch.QwenImageTextModel is QwenImageTextModel
    assert not hasattr(dinkster_inference_torch, "QwenImageVisionTransformer")
    assert QWEN_IMAGE_TEXT_CONFIG.hidden_size == 3584
