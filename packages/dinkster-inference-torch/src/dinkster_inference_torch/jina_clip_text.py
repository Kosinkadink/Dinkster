"""Native Jina CLIP v2 XLM-RoBERTa text encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference.jina_clip_text import JinaClipTextConfig

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations

_DEFAULT_ATTENTION = select_attention("clip").kernel


def _rope(hidden: torch.Tensor, base: float) -> torch.Tensor:
    length, width = hidden.shape[2], hidden.shape[3]
    inverse = 1.0 / (
        base ** (torch.arange(0, width, 2, device=hidden.device, dtype=torch.float32) / width)
    )
    angles = torch.outer(torch.arange(length, device=hidden.device, dtype=torch.float32), inverse)
    angles = torch.cat((angles, angles), dim=-1).to(hidden.dtype).view(1, 1, length, width)
    half = width // 2
    rotated = torch.cat((-hidden[..., half:], hidden[..., :half]), dim=-1)
    return hidden * angles.cos() + rotated * angles.sin()


class JinaAttention(torch.nn.Module):
    def __init__(
        self,
        config: JinaClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.Wqkv = operations.linear(config.hidden_size, config.hidden_size * 3)
        self.out_proj = operations.linear(config.hidden_size, config.hidden_size)
        self._attention = attention_kernel

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, width = hidden.shape
        heads = self.config.num_attention_heads
        qkv = self.Wqkv(hidden).view(batch, length, 3, heads, width // heads)
        query, key, value = qkv.unbind(2)
        query = _rope(query.transpose(1, 2), self.config.rotary_base)
        key = _rope(key.transpose(1, 2), self.config.rotary_base)
        value = value.transpose(1, 2)
        output = self._attention(query, key, value, mask=mask, causal=False)
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, width))


class JinaBlock(torch.nn.Module):
    def __init__(
        self,
        config: JinaClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.mixer = JinaAttention(config, operations=operations, attention_kernel=attention_kernel)
        self.norm1 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = torch.nn.ModuleDict(
            {
                "fc1": operations.linear(config.hidden_size, config.intermediate_size),
                "fc2": operations.linear(config.intermediate_size, config.hidden_size),
            }
        )
        self.norm2 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self.norm1(self.mixer(hidden, mask) + hidden)
        return self.norm2(self.mlp["fc2"](F.gelu(self.mlp["fc1"](hidden))) + hidden)


class JinaEmbeddings(torch.nn.Module):
    def __init__(self, config: JinaClipTextConfig, operations: Operations) -> None:
        super().__init__()
        self.word_embeddings = operations.embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = operations.embedding(1, config.hidden_size)


class JinaEncoder(torch.nn.Module):
    def __init__(
        self,
        config: JinaClipTextConfig,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            JinaBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )


class JinaCore(torch.nn.Module):
    def __init__(
        self,
        config: JinaClipTextConfig,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.embeddings = JinaEmbeddings(config, operations)
        self.emb_ln = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = JinaEncoder(config, operations, attention_kernel)


class JinaClipTextModel(torch.nn.Module):
    def __init__(
        self,
        config: JinaClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = JinaCore(config, operations, attention_kernel)

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or (attention_mask is not None and attention_mask.shape != ids.shape):
            raise ValueError(
                "Jina ids and attention mask must be matching [batch x tokens] tensors"
            )
        hidden = self.model.embeddings.word_embeddings(ids) if embeds is None else embeds
        if hidden.shape != (*ids.shape, self.config.hidden_size):
            raise ValueError("Jina preembedded input must match ids and hidden size")
        token_types = torch.zeros(ids.shape[1], dtype=torch.long, device=ids.device)
        hidden = self.model.emb_ln(
            hidden + self.model.embeddings.token_type_embeddings(token_types)
        )
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(hidden.dtype).reshape(ids.shape[0], 1, 1, ids.shape[1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(hidden.dtype).max)
        prefetch = make_prefetch_queue(self.model.encoder.layers)
        try:
            for layer in self.model.encoder.layers:
                prefetch_queue_pop(prefetch, layer)
                hidden = layer(hidden, mask)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.to(hidden.dtype)
            pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=-1, keepdim=True)
        return hidden, pooled
