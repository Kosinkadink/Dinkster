"""Normalized post-block capture through the T5 model and encoder."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import EmbeddingSlot, PackedToken, T5Config, TokenizerProfile, WeightedSpan
from dinkster_inference.prompt_tokens import empty_chunk, pack_spans
from dinkster_inference_torch import T5EncodeError, T5TextEncoder, T5TextModel
from dinkster_inference_torch._conditioning_layout import declared_token_count


@pytest.fixture(params=("t5", "umt5"))
def model(request: pytest.FixtureRequest) -> T5TextModel:
    result = T5TextModel(
        T5Config(
            d_model=16,
            d_ff=32,
            d_kv=4,
            num_heads=2,
            num_layers=3,
            vocab_size=32,
            dense_act_fn="gelu_pytorch_tanh",
            is_gated_act=True,
            model_type=request.param,
        )
    )
    generator = torch.Generator().manual_seed(572)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    return result


def default_block_states(
    model: T5TextModel, embeds: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    states: list[torch.Tensor] = []

    def record(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        states.append(cast(torch.Tensor, output[0]).clone())

    hooks = [block.register_forward_hook(record) for block in model.encoder.block]
    try:
        final = model(embeds, mask)
    finally:
        for hook in hooks:
            hook.remove()
    return final, states


@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_capture_is_normalized_post_block_and_preserves_default(
    model: T5TextModel, masked: bool, dtype: torch.dtype
) -> None:
    model.to(dtype)
    config = model.config
    ids = torch.tensor([[3, 4, 5, 1, 0, 0]])
    embeds = model.embed_tokens(ids)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]]) if masked else None
    with torch.no_grad():
        final, states = default_block_states(model, embeds, mask)
        torch.testing.assert_close(
            final, model.encoder.final_layer_norm(states[-1]), rtol=0, atol=0
        )
        torch.testing.assert_close(model(embeds, mask, hidden_layer=None), final, rtol=0, atol=0)
        for positive in range(3):
            expected = model.encoder.final_layer_norm(states[positive])
            for index in (positive, positive - 3):
                actual = model(embeds, mask, hidden_layer=index)
                direct = model.encoder(embeds, mask, hidden_layer=index)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(direct, expected, rtol=0, atol=0)
            assert not torch.equal(expected, states[positive])
        assert not torch.equal(model(embeds, mask, hidden_layer=-2), final)
        torch.testing.assert_close(model(embeds, mask), final, rtol=0, atol=0)
    assert model.config is config


@pytest.mark.parametrize("layer", (-4, 3, 4))
def test_model_and_stack_reject_indices_outside_blocks(model: T5TextModel, layer: int) -> None:
    embeds = model.embed_tokens(torch.tensor([[3, 1]]))
    with pytest.raises(ValueError, match="out of range"):
        model(embeds, hidden_layer=layer)
    with pytest.raises(ValueError, match="out of range"):
        model.encoder(embeds, hidden_layer=layer)


@pytest.mark.parametrize("layer", (True, False, 1.0, 1.5, -1.0, 4.0))
@pytest.mark.parametrize("entrypoint", ("model", "stack", "encode", "encode_chunks"))
def test_hidden_layer_requires_exact_int(
    model: T5TextModel, layer: object, entrypoint: str
) -> None:
    encoder = T5TextEncoder(model)
    spans = (WeightedSpan((3, 4), 1.0),)
    embeds = model.embed_tokens(torch.tensor([[3, 4, 1]]))
    with pytest.raises(ValueError, match="hidden_layer must be an integer or None"):
        if entrypoint == "model":
            model(embeds, hidden_layer=cast(int, layer))
        elif entrypoint == "stack":
            model.encoder(embeds, hidden_layer=cast(int, layer))
        elif entrypoint == "encode":
            encoder.encode(spans, hidden_layer=cast(int, layer))
        else:
            encoder.encode_chunks(pack_spans(spans, encoder.profile), hidden_layer=cast(int, layer))


@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("layer", (None, 0, 1, 2, -3, -2, -1, -4, 4))
def test_encoder_capture_preserves_weights_masks_chunks_and_embedding_rows(
    model: T5TextModel, masked: bool, weighted: bool, layer: int | None
) -> None:
    profile = TokenizerProfile(
        max_length=6,
        start_token=None,
        end_token=1,
        pad_token=0,
        pad_to_max_length=True,
        max_word_length=5,
    )
    vectors = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32

    def lookup(name: str) -> torch.Tensor | None:
        return vectors if name == "glyph" else None

    encoder = T5TextEncoder(
        model, profile=profile, embeddings=lookup, attention_masked=masked, zero_out_masked=masked
    )
    weight = 1.7 if weighted else 1.0
    chunks = (
        tuple(
            PackedToken(unit, weight if index < 4 else 1.0, 0)
            for index, unit in enumerate(
                (3, EmbeddingSlot("glyph", 0), EmbeddingSlot("glyph", 1), 4, 1, 0)
            )
        ),
        tuple(PackedToken(unit, 1.0, 0) for unit in (5, 6, 1, 0, 0, 0)),
    )
    batch = chunks + ((empty_chunk(profile, 6),) if weighted else ())
    ids = torch.tensor(
        [[unit.unit if isinstance(unit.unit, int) else 0 for unit in chunk] for chunk in batch]
    )
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]] + ([[1, 0, 0, 0, 0, 0]] if weighted else [])
    )
    with torch.no_grad():
        embeds = model.embed_tokens(ids)
        embeds[0, 1:3] = vectors
        _, states = default_block_states(model, embeds, mask if masked else None)
        selected = -1 if layer is None or abs(layer) > 3 else layer
        encoded = model.encoder.final_layer_norm(states[selected]).float()
        if masked:
            encoded = encoded * mask.unsqueeze(-1)
        expected = encoded[:2].clone()
        if weighted:
            for row, chunk in enumerate(chunks):
                for index, token in enumerate(chunk):
                    if token.weight != 1.0:
                        expected[row, index] = (
                            expected[row, index] - encoded[-1, index]
                        ) * token.weight + encoded[-1, index]
        result = encoder.encode_chunks(chunks, hidden_layer=layer)
        torch.testing.assert_close(result.embeddings, expected.reshape(1, 12, 16), rtol=0, atol=0)
        assert result.embeddings.dtype == torch.float32
        assert result.pooled is None
        assert declared_token_count(result) == 12
        torch.testing.assert_close(
            encoder.encode_chunks(chunks).embeddings,
            encoder.encode_chunks(chunks, hidden_layer=-1).embeddings,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("zero_out", (False, True))
def test_public_encode_forwards_capture_and_preserves_padding_and_token_count(
    model: T5TextModel, zero_out: bool
) -> None:
    profile = TokenizerProfile(
        max_length=99, start_token=None, end_token=1, pad_token=0, pad_to_max_length=False
    )
    encoder = T5TextEncoder(model, profile=profile, attention_masked=True, zero_out_masked=zero_out)
    spans = (WeightedSpan((3, 4), 1.5),)
    chunks = pack_spans(spans, replace(profile, min_length=9, min_padding=3))
    with torch.no_grad():
        default = encoder.encode_chunks(chunks)
        for layer in (None, 0, -2, -3, 4, -4):
            expected = encoder.encode_chunks(chunks, hidden_layer=layer)
            actual = encoder.encode(spans, hidden_layer=layer, min_length=9, min_padding=3)
            torch.testing.assert_close(actual.embeddings, expected.embeddings, rtol=0, atol=0)
            assert actual.embeddings.shape == (1, 9, 16)
            assert declared_token_count(actual) == (9 if zero_out else 3)
            if zero_out:
                assert torch.count_nonzero(actual.embeddings[:, 3:]) == 0
            if layer in (0, -2, -3):
                assert not torch.equal(actual.embeddings, default.embeddings)
        torch.testing.assert_close(
            encoder.encode_chunks(chunks).embeddings, default.embeddings, rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="out of range"):
        encoder.encode(spans, hidden_layer=3)
    if not zero_out:
        with pytest.raises(T5EncodeError, match="contiguous"):
            encoder.encode_chunks(chunks * 2, hidden_layer=-2)


def test_selected_state_preserves_gradients_without_using_later_blocks(model: T5TextModel) -> None:
    embeds = model.embed_tokens(torch.tensor([[3, 4, 5, 1]])).detach().requires_grad_()
    model(embeds, hidden_layer=0).square().mean().backward()
    assert embeds.grad is not None
    assert torch.isfinite(embeds.grad).all() and torch.count_nonzero(embeds.grad) > 0
    for parameter in model.encoder.block[0].parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for block in model.encoder.block[1:]:
        assert all(parameter.grad is None for parameter in block.parameters())
    assert model.encoder.final_layer_norm.weight.grad is not None
