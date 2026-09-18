"""Native LTX-2 text-embedding connectors.

Faithful standalone port of ``comfy/ldm/lightricks/embeddings_connector.py``
(``Embeddings1DConnector``) at the current ComfyUI pin b78cec87, as the LTXAV text
encoder instantiates it: ``split_rope=True`` and ``double_precision_rope=True``.
State-dict names match the reference. Combined checkpoints carry one video
and one audio tower; :class:`LtxTextConnectors` pairs them under the
checkpoint's own submodule names and concatenates video then audio, exactly
as the reference applies them after the single text projection.

Each tower pads its token rows with tiled learnable registers to at least
1024 positions (a constant in the reference forward), applies split RoPE
whose frequency grid is a float64 geometric progression cast to float32,
and runs two pre-norm self-attention blocks under weightless RMS norms,
finishing with one more RMS norm. The reference threads an attention mask
through and zeroes it when registers are appended; on the text-encoder
path the mask is always None, so the port takes tokens only.
"""

from __future__ import annotations

import math

import torch
from dinkster_inference import LTX_TEXT_CONNECTOR_CONFIG, LtxConnectorConfig

from .attention import AttentionKernel, select_attention
from .ltx_model import (  # pyright: ignore[reportPrivateUsage]
    LTXAttention,
    _FeedForward,  # pyright: ignore[reportPrivateUsage]
    _rms,  # pyright: ignore[reportPrivateUsage]
    _rope_matrix,  # pyright: ignore[reportPrivateUsage]
)
from .operations import INITLESS, Operations, ResidencyRouted

_DEFAULT_ATTENTION = select_attention("flux").kernel

#: The reference forward always fills with registers up to this floor,
#: independent of configuration.
_REGISTER_FILL_FLOOR = 1024


def _split_rope_frequencies(
    config: LtxConnectorConfig, length: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, bool]:
    """Per-head rotation matrix for split RoPE.

    The reference's double-precision grid (``generate_freq_grid_np``) is a
    float64 geometric progression from 1 to theta scaled by pi/2, computed
    on the CPU and cast to float32 before use.
    """
    exponents = torch.linspace(0.0, 1.0, config.inner_dim // 2, dtype=torch.float64)
    grid = (torch.pow(config.positional_embedding_theta, exponents) * math.pi / 2).to(
        dtype=torch.float32, device=device
    )
    positions = torch.arange(length, dtype=torch.float32, device=device)
    scale = positions / config.positional_embedding_max_pos * 2 - 1
    frequencies = (grid * scale[:, None]).unsqueeze(0)
    return _rope_matrix(frequencies, 0, True, config.num_attention_heads, dtype), True


class LtxConnectorBlock(torch.nn.Module):
    """The reference ``BasicTransformerBlock1D``: pre-norm self-attention
    and a GELU-tanh MLP, both under weightless RMS norms."""

    def __init__(
        self,
        config: LtxConnectorConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        inner = config.inner_dim
        self.attn1 = LTXAttention(
            inner,
            inner,
            config.num_attention_heads,
            config.attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.ff = _FeedForward(inner, config.ffn_dim, operations=operations)

    def forward(self, hidden: torch.Tensor, rope: tuple[torch.Tensor, bool]) -> torch.Tensor:
        hidden = self.attn1(_rms(hidden), rope=rope) + hidden
        return self.ff(_rms(hidden)) + hidden


class LtxEmbeddingsConnector(ResidencyRouted, torch.nn.Module):
    """One connector tower over projected text embeddings ``[B, T, 3840]``."""

    def __init__(
        self,
        config: LtxConnectorConfig = LTX_TEXT_CONNECTOR_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer_1d_blocks = torch.nn.ModuleList(
            LtxConnectorBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_layers)
        )
        self.learnable_registers = torch.nn.Parameter(
            torch.empty(config.num_learnable_registers, config.inner_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.inner_dim:
            raise ValueError(
                f"connector tokens must be [batch x tokens x {self.config.inner_dim}],"
                f" got {tuple(tokens.shape)}"
            )
        registers = self.learnable_registers.to(device=tokens.device, dtype=tokens.dtype)
        duplications = math.ceil(
            max(_REGISTER_FILL_FLOOR, tokens.shape[1]) / self.config.num_learnable_registers
        )
        tiled = torch.tile(registers, (duplications, 1))
        hidden = torch.cat(
            (tokens, tiled[tokens.shape[1] :].unsqueeze(0).repeat(tokens.shape[0], 1, 1)), dim=1
        )
        rope = _split_rope_frequencies(self.config, hidden.shape[1], hidden.dtype, hidden.device)
        for block in self.transformer_1d_blocks:
            hidden = block(hidden, rope)
        return _rms(hidden)


class LtxTextConnectors(torch.nn.Module):
    """The combined checkpoint's video and audio connector pair; outputs
    concatenate video then audio, doubling the embedding width."""

    def __init__(
        self,
        config: LtxConnectorConfig = LTX_TEXT_CONNECTOR_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.video_embeddings_connector = LtxEmbeddingsConnector(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.audio_embeddings_connector = LtxEmbeddingsConnector(
            config, operations=operations, attention_kernel=attention_kernel
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.video_embeddings_connector(embeddings),
                self.audio_embeddings_connector(embeddings),
            ),
            dim=-1,
        )


__all__ = [
    "LtxConnectorBlock",
    "LtxEmbeddingsConnector",
    "LtxTextConnectors",
]
