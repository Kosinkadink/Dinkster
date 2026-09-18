"""Native Ideogram 4 Qwen3-VL-8B text conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from dinkster_inference import IDEOGRAM4_TEXT_CONFIG, Conditioning
from dinkster_inference.qwen_bpe import tokenize_ideogram4_prompt

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, bound_compute_device
from .qwen_image_text import QwenImageLanguageModel

_DEFAULT_QWEN_ATTENTION = select_attention("qwen").kernel


@dataclass(frozen=True)
class Ideogram4Conditioning(Conditioning[torch.Tensor]):
    attention_mask: torch.Tensor | None = None
    image_only: bool = False


def ideogram4_language_model(
    *,
    operations: Operations = INITLESS,
    attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
) -> QwenImageLanguageModel:
    """Build the exact Ideogram 4 Qwen3-VL-8B language tower."""

    config = IDEOGRAM4_TEXT_CONFIG
    return QwenImageLanguageModel.reduced(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rope_dims=config.rope_dims,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        qkv_bias=config.qkv_bias,
        qk_norm=config.qk_norm,
        final_norm=config.final_norm,
        interleaved_mrope=config.interleaved_mrope,
        max_position_embeddings=config.max_position_embeddings,
        architecture="Ideogram 4",
        operations=operations,
        attention_kernel=attention_kernel,
    )


class Ideogram4TextEncoder:
    def __init__(self, model: QwenImageLanguageModel) -> None:
        config = IDEOGRAM4_TEXT_CONFIG
        expected = {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "rope_dims": config.rope_dims,
            "qkv_bias": config.qkv_bias,
            "qk_norm": config.qk_norm,
            "final_norm": config.final_norm,
            "interleaved_mrope": config.interleaved_mrope,
            "max_position_embeddings": config.max_position_embeddings,
            "architecture": "Ideogram 4",
        }
        if asdict(model.shape) != expected:
            raise ValueError("Ideogram4TextEncoder requires the Ideogram 4 language profile")
        self.model = model

    def encode(self, text: str) -> Ideogram4Conditioning:
        tokens = tokenize_ideogram4_prompt(text)
        self.model.validate_sequence_length(len(tokens.ids))
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(tokens.ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.tensor(tokens.attention_mask, dtype=torch.long, device=device).unsqueeze(
            0
        )
        stacked = self.model.tapped_states(
            ids, attention, tap_layers=IDEOGRAM4_TEXT_CONFIG.tap_layers
        )
        if self.model.norm is None:
            raise RuntimeError("Ideogram 4 language tower has no final norm")
        stacked = torch.cat((stacked[:, :-1], self.model.norm(stacked[:, -1:])), dim=1)
        batch, taps, length, hidden = stacked.shape
        flattened = stacked.permute(0, 2, 3, 1).reshape(batch, length, hidden * taps)
        retained_mask = None if bool(torch.all(attention)) else attention
        return Ideogram4Conditioning(flattened.float(), None, retained_mask)


__all__ = [
    "Ideogram4Conditioning",
    "Ideogram4TextEncoder",
    "ideogram4_language_model",
]
