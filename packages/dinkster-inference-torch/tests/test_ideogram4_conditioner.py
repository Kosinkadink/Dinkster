"""Ideogram 4 Qwen3-VL language-tower and encode-policy tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    IDEOGRAM4_TAP_LAYERS,
    Conditioning,
    ideogram4_language_layout,
    tokenize_ideogram4_prompt,
)
from dinkster_inference_torch import Ideogram4TextEncoder, ideogram4_language_model
from dinkster_inference_torch.qwen_image_text import QwenImageLanguageModel
from unet_fill import fill_state_dict


class RecordingModel:
    def __init__(self, shape: Any) -> None:
        self.shape = shape
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.norm = _TerminalNorm()
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.tap_layers: tuple[int, ...] | None = None

    def validate_sequence_length(self, length: int) -> None:
        if length > self.shape.max_position_embeddings:
            raise ValueError("sequence is too long")

    def tapped_states(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        tap_layers: tuple[int, ...],
    ) -> torch.Tensor:
        self.ids = ids
        self.mask = attention_mask
        self.tap_layers = tap_layers
        batch, length = ids.shape
        taps = len(tap_layers)
        return torch.arange(batch * taps * length * 3, dtype=torch.float32).reshape(
            batch, taps, length, 3
        )


class _TerminalNorm:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1000


def full_shape() -> Any:
    with torch.device("meta"):
        return ideogram4_language_model().shape


def test_full_meta_language_tower_matches_planned_layout() -> None:
    with torch.device("meta"):
        model = ideogram4_language_model()
    assert model.shape.architecture == "Ideogram 4"
    assert model.shape.final_norm is True
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == sorted(
        ideogram4_language_layout().items()
    )


def test_encoder_preserves_full_prompt_and_flattens_taps_last() -> None:
    model = RecordingModel(full_shape())
    encoded: Conditioning[torch.Tensor] = Ideogram4TextEncoder(cast("Any", model)).encode("cat")
    expected_tokens = tokenize_ideogram4_prompt("cat")
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(expected_tokens.ids)]
    assert model.mask.tolist() == [list(expected_tokens.attention_mask)]
    assert model.tap_layers == IDEOGRAM4_TAP_LAYERS
    length = len(expected_tokens.ids)
    source = torch.arange(13 * length * 3, dtype=torch.float32).reshape(1, 13, length, 3)
    source[:, -1:] += 1000
    expected = source.permute(0, 2, 3, 1).reshape(1, length, 39)
    assert torch.equal(encoded.embeddings, expected)
    assert encoded.embeddings.shape == (1, length, 39)
    assert encoded.pooled is None
    assert encoded.attention_mask is None


def test_encoder_retains_only_effective_attention_masks() -> None:
    model = RecordingModel(full_shape())
    encoded = Ideogram4TextEncoder(cast("Any", model)).encode("<|endoftext|> masked tail")
    assert encoded.attention_mask is not None
    first_pad = tokenize_ideogram4_prompt("<|endoftext|> masked tail").ids.index(151643)
    assert encoded.attention_mask[0, first_pad - 1 : first_pad + 2].tolist() == [1, 0, 0]


def test_encoder_refuses_other_qwen_profiles() -> None:
    other = QwenImageLanguageModel.reduced(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        final_norm=False,
    )
    with pytest.raises(ValueError, match="Ideogram 4 language profile"):
        Ideogram4TextEncoder(other)


def test_terminal_tap_is_the_post_decoder_residual() -> None:
    model = QwenImageLanguageModel.reduced(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        final_norm=False,
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    ids = torch.tensor(((1, 2, 3),), dtype=torch.long)
    last: dict[str, torch.Tensor] = {}
    hook = model.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: last.update(output=cast(torch.Tensor, output))
    )
    try:
        tapped = model.tapped_states(ids, tap_layers=(2,))
    finally:
        hook.remove()
    assert torch.equal(tapped[:, 0], last["output"])
