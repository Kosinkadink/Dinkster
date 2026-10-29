"""Regression coverage for the optimized Wan FFN and paired RoPE routes."""

from __future__ import annotations

import pytest
import torch
from dinkster_inference_torch import wan21_model as wan21_model_module
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.wan21_model import WanFeedForward, WanSelfAttention
from unet_fill import fill_state_dict, hashed_input


def test_feed_forward_is_bit_exact_to_unfused_gelu_path() -> None:
    feed_forward = WanFeedForward(
        torch.nn.Linear(7, 13),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(13, 5),
    )
    feed_forward.load_state_dict(
        fill_state_dict([(key, value.shape) for key, value in feed_forward.state_dict().items()])
    )
    input = hashed_input("wan-feed-forward", (2, 3, 7))

    optimized = feed_forward(input)
    reference = feed_forward[2](feed_forward[1](feed_forward[0](input)))

    assert torch.equal(optimized, reference)


def test_self_attention_routes_query_and_key_through_paired_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = WanSelfAttention(
        8,
        2,
        qk_norm=True,
        eps=1e-6,
        operations=wan21_model_module.INITLESS,
        attention_kernel=select_attention("flux").kernel,
    )
    attention.load_state_dict(
        fill_state_dict([(key, value.shape) for key, value in attention.state_dict().items()])
    )
    input = hashed_input("wan-self-attention", (1, 3, 8))
    freqs = hashed_input("wan-rope", (1, 1, 3, 2, 2, 2))
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def paired_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((query, key, frequencies))
        return query, key

    monkeypatch.setattr(wan21_model_module, "apply_rope", paired_rope)

    output = attention(input, freqs)

    assert output.shape == input.shape
    assert len(calls) == 1
    assert calls[0][0].shape == calls[0][1].shape == (1, 3, 2, 4)
    assert calls[0][2] is freqs
