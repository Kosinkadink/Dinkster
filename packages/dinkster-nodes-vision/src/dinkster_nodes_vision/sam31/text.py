"""SAM 3.1 CLIP text encoder with checkpoint-native dimensions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import CLIP_EOS, CLIP_VOCAB_SIZE, load_tokenizer

WIDTH = 1024
HEADS = 16
LAYERS = 24
TOKEN_COUNT = 32


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch, tokens, width = query.shape
    head_width = width // HEADS

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, tokens, HEADS, head_width).transpose(1, 2)

    result = F.scaled_dot_product_attention(
        split(query),
        split(key),
        split(value),
        attn_mask=mask,
    )
    return result.transpose(1, 2).reshape(batch, tokens, width)


class TextAttention(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, WIDTH, device=device)
        self.k_proj = nn.Linear(WIDTH, WIDTH, device=device)
        self.v_proj = nn.Linear(WIDTH, WIDTH, device=device)
        self.out_proj = nn.Linear(WIDTH, WIDTH, device=device)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.out_proj(
            _attention(
                self.q_proj(value),
                self.k_proj(value),
                self.v_proj(value),
                mask,
            )
        )


class TextLayer(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(WIDTH, device=device)
        self.self_attention = TextAttention(device=device)
        self.layer_norm2 = nn.LayerNorm(WIDTH, device=device)
        self.fc1 = nn.Linear(WIDTH, 4096, device=device)
        self.fc2 = nn.Linear(4096, WIDTH, device=device)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + self.self_attention(self.layer_norm1(value), mask)
        hidden = self.fc1(self.layer_norm2(value))
        return value + self.fc2(hidden * torch.sigmoid(1.702 * hidden))


class SAM31TextEncoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(CLIP_VOCAB_SIZE, WIDTH, device=device)
        self.positional_embedding = nn.Parameter(torch.empty(TOKEN_COUNT, WIDTH, device=device))
        self.layers = nn.ModuleList(TextLayer(device=device) for _ in range(LAYERS))
        self.final_layer_norm = nn.LayerNorm(WIDTH, device=device)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        value = self.token_embedding(tokens) + self.positional_embedding
        padding = 1.0 - attention_mask.to(value.dtype)
        mask = padding[:, None, None, :].expand(-1, 1, TOKEN_COUNT, -1)
        mask = mask.masked_fill(mask.bool(), -torch.finfo(value.dtype).max)
        causal = torch.full(
            (TOKEN_COUNT, TOKEN_COUNT),
            -torch.finfo(value.dtype).max,
            dtype=value.dtype,
            device=value.device,
        ).triu_(1)
        mask = mask + causal
        for layer in self.layers:
            value = layer(value, mask)
        return self.final_layer_norm(value)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        batches = load_tokenizer().batches(text)
        tokens = torch.tensor(batches, dtype=torch.long, device=self.positional_embedding.device)
        eos = (tokens == CLIP_EOS).to(dtype=torch.int64).argmax(dim=1)
        positions = torch.arange(TOKEN_COUNT, device=tokens.device).unsqueeze(0)
        mask = (positions <= eos.unsqueeze(1)).to(dtype=torch.long)
        hidden = self(tokens, mask)
        return hidden.reshape(1, -1, WIDTH), mask.reshape(1, -1)


__all__ = ["SAM31TextEncoder"]
